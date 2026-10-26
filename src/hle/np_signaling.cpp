// sceNpSignaling over the peer table.
//
// The game's SocketState FSM (0x10d8bb0) is the only caller: state 4
// activates a connection to a peer's NpId and expects ESTABLISHED, state 5
// reads the status and expects 2 with the peer's address and port. The
// events are the ones the game's handler acts on (the co-op contract):
// 0x01 established (SocketState+0xac), 0x0c active (+0xad), 0x00 dead.
#include "hle/np.h"

#include "core/thunk.h"
#include "guest_abi.h"
#include "hle/common.h"
#include "log.h"
#include "net/session.h"

#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace {

constexpr int kErrInvalidArg = static_cast<int>(0x80550003);
constexpr std::uint16_t kEvDead = 0x00;
constexpr std::uint16_t kEvEstablished = 0x01;
constexpr std::uint16_t kEvActive = 0x0c;

struct Ctx {
    void* cb = nullptr;
    void* arg = nullptr;
};
struct Conn {
    unsigned ctx = 0;
    std::string online_id;
    std::uint16_t member_id = 0;  // 0 until the peer is known
    bool announced = false;       // ESTABLISHED/ACTIVE fired
    bool dead = false;
};

std::mutex g_mu;
std::map<unsigned, Ctx> g_ctx;
std::map<unsigned, Conn> g_conn;  // by connection id
unsigned g_next_ctx = 1;
unsigned g_next_conn = 1;

bool g_trace = [] {
    const char* e = std::getenv("BBHOST_NP_TRACE");
    return e && e[0] == '1';
}();

// (ctxId, connId, event, errorCode, arg). ESTABLISHED and ACTIVE are queued
// 200 and 400 ms ahead (announce_locked); a connection that died in between
// (deactivated, its peer gone, a reset) gets neither: the game would take it
// as live again. DEAD is always delivered, since it is queued for a
// connection already marked dead.
void fire(unsigned ctx, unsigned conn, std::uint16_t event, int error) {
    Ctx c;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_ctx.find(ctx);
        if (it == g_ctx.end() || !it->second.cb) return;
        if (event != kEvDead) {
            auto ci = g_conn.find(conn);
            if (ci == g_conn.end() || ci->second.dead) {
                host_log("np: conn %u is dead: event 0x%x not delivered", conn, event);
                return;
            }
        }
        c = it->second;
    }
    if (g_trace) host_log("np: NpSignaling event 0x%x conn %u", event, conn);
    hle_call_guest<std::int64_t>(c.cb, ctx, conn, event, error, c.arg);
}

// Once a connection has a known peer: ESTABLISHED at +200 ms, ACTIVE at +400.
void announce_locked(unsigned conn_id, Conn& c) {
    if (c.announced || !c.member_id) return;
    c.announced = true;
    net::peers_set_conn(c.member_id, conn_id, 2);
    const unsigned ctx = c.ctx;
    net::dispatch_after(200, net::Prio::Signaling, [ctx, conn_id] { fire(ctx, conn_id, kEvEstablished, 0); });
    net::dispatch_after(400, net::Prio::Signaling, [ctx, conn_id] { fire(ctx, conn_id, kEvActive, 0); });
}

GUEST_ABI int sig_init(std::uint64_t, int, int, std::uint64_t) {
    host_log("sceNpSignalingInitialize");
    return 0;
}
GUEST_ABI int sig_term() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_ctx.clear();
    g_conn.clear();
    return 0;
}
// (const SceNpId*, callback, arg, ctxId*)
GUEST_ABI int sig_create_ctx(const void*, void* cb, void* arg, unsigned* ctx) {
    if (!ctx) return kErrInvalidArg;
    std::lock_guard<std::mutex> lk(g_mu);
    const unsigned id = g_next_ctx++;
    g_ctx[id] = Ctx{cb, arg};
    *ctx = id;
    host_log("sceNpSignalingCreateContext -> %u", id);
    return 0;
}
GUEST_ABI int sig_delete_ctx(unsigned ctx) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_ctx.erase(ctx);
    for (auto it = g_conn.begin(); it != g_conn.end();) {
        it = it->second.ctx == ctx ? g_conn.erase(it) : std::next(it);
    }
    return 0;
}
// The server's address for a connection whose peer was not known at its
// activation (sig_activate), on the dispatcher: a provisional peer there and
// the connection announced, unless it went away or its peer joined the room
// in the meantime (hle_np_signaling_peer_known announced it then).
void resolve_conn(unsigned conn_id, const std::string& peer) {
    std::uint32_t addr = 0;
    std::uint16_t port = 0;
    std::string error;
    const bool ok = net::server_signaling_resolve(peer, &addr, &port, error);
    std::lock_guard<std::mutex> lk(g_mu);
    if (!ok) {
        host_log("np: conn %u to %s: not known and not resolved: %s", conn_id, peer.c_str(), error.c_str());
        return;
    }
    auto it = g_conn.find(conn_id);
    if (it == g_conn.end() || it->second.dead) {
        host_log("np: conn %u to %s resolved, but the connection went away meanwhile: left alone", conn_id, peer.c_str());
        return;
    }
    if (it->second.member_id) {
        host_log("np: conn %u to %s resolved, but its peer joined the room meanwhile (member %u): left alone", conn_id,
                 peer.c_str(), it->second.member_id);
        return;
    }
    it->second.member_id = net::peers_provisional(peer, addr, port).member_id;
    host_log("np: conn %u to %s resolved %u.%u.%u.%u:%u", conn_id, peer.c_str(), addr & 0xff, (addr >> 8) & 0xff,
             (addr >> 16) & 0xff, addr >> 24, port);
    announce_locked(conn_id, it->second);
}

// (ctxId, const SceNpId* peer, connId*): idempotent per peer, as the fork
// found the game re-activates.
GUEST_ABI int sig_activate(unsigned ctx, const void* npid, unsigned* conn_out) {
    if (!conn_out) return kErrInvalidArg;
    const std::string peer = hle_np_npid_text(npid);
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [id, c] : g_conn) {
        if (c.ctx == ctx && c.online_id == peer && !c.dead) {
            *conn_out = id;
            if (!c.member_id) {
                net::Peer p;
                if (net::peers_find_online(peer, &p)) c.member_id = p.member_id;
            }
            announce_locked(id, c);
            return 0;
        }
    }
    const unsigned id = g_next_conn++;
    Conn c;
    c.ctx = ctx;
    c.online_id = peer;
    net::Peer p;
    if (net::peers_find_online(peer, &p)) c.member_id = p.member_id;
    g_conn[id] = c;
    *conn_out = id;
    if (!c.member_id) {
        // The host activates its connection to the guest it summons before
        // the guest is a room member; on PSN the signaling server resolved
        // the address, here ours does, from the peer's context_start - asked
        // from the dispatcher (resolve_conn), and the connection is announced
        // with the answer. Until 2026-10-25 the game's thread waited here for
        // it, with g_mu held.
        host_log("sceNpSignalingActivateConnection %s -> conn %u (peer not known yet: asking the server)", peer.c_str(),
                 id);
        net::dispatch_after(0, net::Prio::Signaling, [id, peer] { resolve_conn(id, peer); });
        return 0;
    }
    host_log("sceNpSignalingActivateConnection %s -> conn %u", peer.c_str(), id);
    announce_locked(id, g_conn[id]);
    return 0;
}
GUEST_ABI int sig_deactivate(unsigned ctx, unsigned conn) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_conn.find(conn);
    if (it != g_conn.end() && it->second.ctx == ctx) {
        it->second.dead = true;
        if (it->second.member_id) net::peers_set_conn(it->second.member_id, conn, 0);
    }
    return 0;
}
// (ctxId, connId, status*, SceNetInAddr*, SceInPort_t*)
GUEST_ABI int sig_status(unsigned ctx, unsigned conn, int* st, std::uint32_t* addr, std::uint16_t* port) {
    int status = 0;
    std::uint32_t a = 0;
    std::uint16_t p = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_conn.find(conn);
        if (it != g_conn.end() && it->second.ctx == ctx && !it->second.dead) {
            net::Peer peer;
            if (it->second.member_id && net::peers_get(it->second.member_id, &peer) && peer.addr) {
                status = it->second.announced ? 2 : 1;
                a = peer.addr;
                p = static_cast<std::uint16_t>((peer.port << 8) | (peer.port >> 8));
            } else {
                status = 1;
            }
        }
    }
    if (st) *st = status;
    if (addr) *addr = a;
    if (port) *port = p;
    if (g_trace) host_log("sceNpSignalingGetConnectionStatus conn %u -> %d", conn, status);
    return 0;
}

}  // namespace

void hle_np_signaling_peer_known(std::uint16_t member_id) {
    net::Peer p;
    if (!net::peers_get(member_id, &p)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [id, c] : g_conn) {
        if (!c.dead && c.online_id == p.online_id) {
            c.member_id = member_id;
            announce_locked(id, c);
        }
    }
}

void hle_np_signaling_peer_dead(const std::string& online_id) {
    std::vector<std::pair<unsigned, unsigned>> dead;  // ctx, conn
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [id, c] : g_conn) {
            if (!c.dead && c.online_id == online_id) {
                c.dead = true;
                dead.emplace_back(c.ctx, id);
            }
        }
    }
    for (const auto& [ctx, conn] : dead) {
        net::dispatch_after(100, net::Prio::Signaling, [ctx, conn] { fire(ctx, conn, kEvDead, 0); });
    }
}

void hle_np_signaling_reset() {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [id, c] : g_conn) c.dead = true;
}

void hle_register_np_signaling() {
    REG("sceNpSignalingInitialize", sig_init);
    REG("sceNpSignalingTerminate", sig_term);
    REG("sceNpSignalingCreateContext", sig_create_ctx);
    REG("sceNpSignalingDeleteContext", sig_delete_ctx);
    REG("sceNpSignalingActivateConnection", sig_activate);
    REG("sceNpSignalingDeactivateConnection", sig_deactivate);
    REG("sceNpSignalingGetConnectionStatus", sig_status);
}
