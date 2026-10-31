// sceNpSignaling over the peer table, as shadPS4's (src/core/libraries/np,
// signaling_handler.cpp) and the SDK do it.
//
// The game's SocketState FSM (0x10d8bb0) is the only caller: state 4
// activates a connection to a peer's NpId and expects ESTABLISHED, state 5
// reads the status and expects 2 with the peer's address and port. The
// events are the ones the game's handler acts on (the co-op contract):
// 0x01 established (SocketState+0xac), 0x0c mutual activated (+0xad), 0x00
// dead; 0x0b peer deactivated.
//
// A connection is a conversation with the other bbhost (hle/net.cpp,
// kSignalMagic datagrams on the P2P port):
//   activate     -> ACTIVATE every 500 ms until the peer answers ACK (status
//                   1, pending; 30 s without an answer: DEAD, timeout)
//   ACK          -> status 2, ESTABLISHED; MUTUAL_ACTIVATED once the peer has
//                   activated its own connection to us (its ACTIVATE)
//   ACTIVATE     <- the peer activated: answered ACK, and MUTUAL_ACTIVATED on
//                   ours if it is established
//   deactivate   -> CLOSE(deactivate) to the peer when established, and DEAD
//                   (terminated by myself) at once
//   CLOSE        <- PEER_DEACTIVATED when established, else DEAD (terminated
//                   by peer)
#include "hle/np.h"

#include "core/thunk.h"
#include "guest_abi.h"
#include "hle/common.h"
#include "hle/net_p2p.h"
#include "log.h"
#include "net/session.h"

#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <vector>

namespace {

constexpr int kErrInvalidArg = static_cast<int>(0x80552715);
constexpr int kErrConnNotFound = static_cast<int>(0x8055270E);
constexpr int kErrTerminatedByPeer = static_cast<int>(0x80552710);
constexpr int kErrTimeout = static_cast<int>(0x80552711);
constexpr int kErrTerminatedByMyself = static_cast<int>(0x80552718);
constexpr std::uint16_t kEvDead = 0x00;
constexpr std::uint16_t kEvEstablished = 0x01;
constexpr std::uint16_t kEvPeerDeactivated = 0x0b;
constexpr std::uint16_t kEvMutualActivated = 0x0c;

// The control datagram: magic, kind, reason, the sender's online id.
enum Kind : std::uint8_t { kActivate = 1, kAck = 2, kClose = 3 };
enum Reason : std::uint8_t { kDeactivate = 1, kTerminate = 2 };
constexpr std::size_t kIdLen = 16;
constexpr std::size_t kPacketLen = sizeof(kSignalMagic) + 2 + kIdLen;
constexpr int kResendMs = 500;
constexpr int kGiveUpMs = 30000;

struct Ctx {
    void* cb = nullptr;
    void* arg = nullptr;
};
struct Conn {
    unsigned ctx = 0;
    std::string online_id;
    std::uint16_t member_id = 0;  // 0 until the peer is known
    std::uint32_t addr = 0;       // the peer's P2P port (network order), kept past the room's peer table
    std::uint16_t port = 0;       // host order
    bool started = false;         // ACTIVATE being sent
    bool established = false;     // the peer answered: status 2, ESTABLISHED fired
    bool peer_activated = false;  // the peer activated its connection to us
    bool mutual = false;          // MUTUAL_ACTIVATED fired
    bool dead = false;
    int waited_ms = 0;
};

std::mutex g_mu;
std::map<unsigned, Ctx> g_ctx;
std::map<unsigned, Conn> g_conn;    // by connection id
std::set<std::string> g_activated;  // players whose ACTIVATE came before ours
unsigned g_next_ctx = 1;
unsigned g_next_conn = 1;

bool g_trace = [] {
    const char* e = std::getenv("BBHOST_NP_TRACE");
    return e && e[0] == '1';
}();

// (ctxId, connId, event, errorCode, arg), on the dispatcher. A connection that
// died before its event went out gets only DEAD.
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
    if (g_trace) host_log("np: NpSignaling event 0x%x conn %u error 0x%x", event, conn, static_cast<unsigned>(error));
    hle_call_guest<std::int64_t>(c.cb, ctx, conn, event, error, c.arg);
}

void queue_event(unsigned ctx, unsigned conn, std::uint16_t event, int error) {
    net::dispatch_after(0, net::Prio::Signaling, [ctx, conn, event, error] { fire(ctx, conn, event, error); });
}

// A control datagram to a player in the peer table.
void send_locked(Conn& c, Kind kind, Reason reason = kDeactivate) {
    net::Peer p;
    if (c.member_id && net::peers_get(c.member_id, &p) && p.addr && p.port) {
        c.addr = p.addr;
        c.port = p.port;
    }
    if (!c.addr || !c.port) return;
    std::uint8_t pkt[kPacketLen] = {};
    std::memcpy(pkt, kSignalMagic, sizeof(kSignalMagic));
    pkt[4] = kind;
    pkt[5] = reason;
    const std::string me = net::online_id();
    std::memcpy(pkt + 6, me.data(), std::min(me.size(), kIdLen));
    hle_net_p2p_signal(c.addr, c.port, pkt, sizeof(pkt));
}

// ACTIVATE until the peer answers or it is time to give up.
void activate_step(unsigned conn_id) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_conn.find(conn_id);
    if (it == g_conn.end()) return;
    Conn& c = it->second;
    if (c.dead || c.established) return;
    if (c.waited_ms >= kGiveUpMs) {
        host_log("np: conn %u to %s: no answer in %d s: DEAD (timeout)", conn_id, c.online_id.c_str(), kGiveUpMs / 1000);
        c.dead = true;
        if (c.member_id) net::peers_set_conn(c.member_id, conn_id, 0);
        queue_event(c.ctx, conn_id, kEvDead, kErrTimeout);
        return;
    }
    send_locked(c, kActivate);
    c.waited_ms += kResendMs;
    net::dispatch_after(kResendMs, net::Prio::Signaling, [conn_id] { activate_step(conn_id); });
}

// Once a connection's peer is known: status 1 (pending) and ACTIVATE.
void start_locked(unsigned conn_id, Conn& c) {
    if (c.started || !c.member_id || c.dead) return;
    c.started = true;
    if (g_activated.count(c.online_id)) c.peer_activated = true;
    net::peers_set_conn(c.member_id, conn_id, 1);
    net::dispatch_after(0, net::Prio::Signaling, [conn_id] { activate_step(conn_id); });
}

GUEST_ABI int sig_init(std::uint64_t, int, int, std::uint64_t) {
    host_log("sceNpSignalingInitialize");
    return 0;
}
GUEST_ABI int sig_term() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_ctx.clear();
    g_conn.clear();
    g_activated.clear();
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
// activation (sig_activate), on the dispatcher: a provisional peer, and the
// connection started, unless it went away or its peer joined the room in the
// meantime (hle_np_signaling_peer_known started it then).
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
    start_locked(conn_id, it->second);
}

// (ctxId, const SceNpId* peer, connId*): one live connection per peer, as
// the fork found the game re-activates.
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
            start_locked(id, c);
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
        // the guest is a room member: the server resolves the address, from
        // the peer's context_start, on the dispatcher (resolve_conn).
        host_log("sceNpSignalingActivateConnection %s -> conn %u (peer not known yet: asking the server)", peer.c_str(),
                 id);
        net::dispatch_after(0, net::Prio::Signaling, [id, peer] { resolve_conn(id, peer); });
        return 0;
    }
    host_log("sceNpSignalingActivateConnection %s -> conn %u", peer.c_str(), id);
    start_locked(id, g_conn[id]);
    return 0;
}

// Closes the game's connection: CLOSE to the peer when it was established,
// NpMatching2's DEAD kept at its LeaveRoom for that player, then DEAD.
GUEST_ABI int sig_deactivate(unsigned ctx, unsigned conn) {
    std::string peer;
    unsigned dead_ctx = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_conn.find(conn);
        if (it == g_conn.end() || it->second.ctx != ctx) return kErrConnNotFound;
        Conn& c = it->second;
        host_log("sceNpSignalingDeactivateConnection conn %u (%s)%s", conn, c.online_id.c_str(), c.dead ? " (already dead)" : "");
        if (c.dead) return 0;
        if (c.established) send_locked(c, kClose, kDeactivate);
        c.dead = true;
        if (c.member_id) net::peers_set_conn(c.member_id, conn, 0);
        peer = c.online_id;
        dead_ctx = c.ctx;
    }
    hle_np_matching2_peer_closed(peer);
    queue_event(dead_ctx, conn, kEvDead, kErrTerminatedByMyself);
    return 0;
}

// (ctxId, connId, status*, SceNetInAddr*, SceInPort_t*): 0 inactive, 1
// pending (activated, no answer yet), 2 active.
GUEST_ABI int sig_status(unsigned ctx, unsigned conn, int* st, std::uint32_t* addr, std::uint16_t* port) {
    int status = 0;
    std::uint32_t a = 0;
    std::uint16_t p = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_conn.find(conn);
        if (it != g_conn.end() && it->second.ctx == ctx && !it->second.dead) {
            net::Peer peer;
            status = it->second.established ? 2 : 1;
            if (it->second.established) {
                // The room's peer table, else the address the connection kept
                // (the table is cleared when the room ends; the connection is
                // the game's until it closes it).
                std::uint32_t pa = it->second.addr;
                std::uint16_t pp = it->second.port;
                if (it->second.member_id && net::peers_get(it->second.member_id, &peer) && peer.addr) {
                    pa = peer.addr;
                    pp = peer.port;
                }
                a = pa;
                p = static_cast<std::uint16_t>((pp << 8) | (pp >> 8));
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

void hle_np_signaling_control(const std::uint8_t* data, std::size_t len, std::uint32_t addr, std::uint16_t port) {
    if (len < kPacketLen) return;
    const Kind kind = static_cast<Kind>(data[4]);
    const Reason reason = static_cast<Reason>(data[5]);
    std::string from(reinterpret_cast<const char*>(data + 6), kIdLen);
    from.resize(std::strlen(from.c_str()));
    if (from.empty()) return;
    if (g_trace) host_log("np: signaling %s from %s", kind == kActivate ? "ACTIVATE" : kind == kAck ? "ACK" : "CLOSE", from.c_str());
    std::vector<std::tuple<unsigned, unsigned, std::uint16_t, int>> events;  // ctx, conn, event, error
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (kind == kActivate) {
            // The peer activated its connection to us: answered, from the
            // address it came from (the peer table may not have it yet).
            std::uint8_t ack[kPacketLen] = {};
            std::memcpy(ack, kSignalMagic, sizeof(kSignalMagic));
            ack[4] = kAck;
            const std::string me = net::online_id();
            std::memcpy(ack + 6, me.data(), std::min(me.size(), kIdLen));
            hle_net_p2p_signal(addr, static_cast<std::uint16_t>((port << 8) | (port >> 8)), ack, sizeof(ack));
            bool ours = false;
            for (auto& [id, c] : g_conn) {
                if (c.dead || c.online_id != from) continue;
                ours = true;
                c.peer_activated = true;
                if (c.established && !c.mutual) {
                    c.mutual = true;
                    events.emplace_back(c.ctx, id, kEvMutualActivated, 0);
                }
            }
            if (!ours) g_activated.insert(from);
        } else if (kind == kAck) {
            for (auto& [id, c] : g_conn) {
                if (c.dead || c.online_id != from || c.established || !c.started) continue;
                c.established = true;
                if (c.member_id) net::peers_set_conn(c.member_id, id, 2);
                events.emplace_back(c.ctx, id, kEvEstablished, 0);
                if (c.peer_activated && !c.mutual) {
                    c.mutual = true;
                    events.emplace_back(c.ctx, id, kEvMutualActivated, 0);
                }
            }
        } else if (kind == kClose) {
            g_activated.erase(from);
            for (auto& [id, c] : g_conn) {
                if (c.dead || c.online_id != from) continue;
                if (reason == kDeactivate && c.established) {
                    c.peer_activated = false;
                    c.mutual = false;
                    events.emplace_back(c.ctx, id, kEvPeerDeactivated, 0);
                } else {
                    c.dead = true;
                    if (c.member_id) net::peers_set_conn(c.member_id, id, 0);
                    events.emplace_back(c.ctx, id, kEvDead, kErrTerminatedByPeer);
                }
            }
        }
    }
    for (const auto& [ctx, conn, event, error] : events) fire(ctx, conn, event, error);
}

bool hle_np_signaling_live(const std::string& online_id) {
    std::lock_guard<std::mutex> lk(g_mu);
    for (const auto& [id, c] : g_conn) {
        if (!c.dead && c.online_id == online_id) return true;
    }
    return false;
}

void hle_np_signaling_peer_known(std::uint16_t member_id) {
    net::Peer p;
    if (!net::peers_get(member_id, &p)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [id, c] : g_conn) {
        if (!c.dead && c.online_id == p.online_id) {
            c.member_id = member_id;
            start_locked(id, c);
        }
    }
}

void hle_np_signaling_peer_dead(const std::string& online_id) {
    std::vector<std::pair<unsigned, unsigned>> dead;  // ctx, conn
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_activated.erase(online_id);
        for (auto& [id, c] : g_conn) {
            if (!c.dead && c.online_id == online_id) {
                c.dead = true;
                dead.emplace_back(c.ctx, id);
            }
        }
    }
    for (const auto& [ctx, conn] : dead) queue_event(ctx, conn, kEvDead, kErrTerminatedByPeer);
}

void hle_np_signaling_reset() {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [id, c] : g_conn) c.dead = true;
    g_activated.clear();
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
