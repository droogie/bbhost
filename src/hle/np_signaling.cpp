// sceNpSignaling, with the connection handshake shadPS4 uses
// (src/core/libraries/np/signaling_handler.cpp).
//
// The game's SocketState FSM (0x10d8bb0) is the only caller: state 4
// activates a connection to a peer's NpId and expects ESTABLISHED, state 5
// reads the status and expects 2 with the peer's address and port. The
// events are the ones the game's handler acts on (the co-op contract):
// 0x01 established (SocketState+0xac), 0x0c mutual activated (+0xad), 0x00
// dead; 0x0b peer deactivated; 0x0a peer activated.
//
// Two layers, as shadPS4:
//   transport   one per other player: the link to that player and its
//               handshake state (Inactive, SendingOffer, WaitOffer,
//               SendingAccept, WaitAccept, ConnCheck, Established)
//   connection  what the game holds (conn id, per context and player); its
//               status is its transport's: Established 2, Inactive 0, else 1
//
// The messages are kSignalMagic datagrams on the P2P port (hle/net.cpp):
//   control    ActivationRequest, ActivationAck, Established, Close
//   handshake  Offer, Accept, Check, CheckAck
//
// activate          ActivationRequest, and Offer (resent every 500 ms; 30 s
//                   without getting to Established: DEAD, timeout)
// ActivationRequest the peer's game activated: its transport made (or kept),
//                   a connection for it if the game has none (PEER_ACTIVATED
//                   once established), answered ActivationAck. Establishes
//                   nothing unless the transport already is.
// Offer             answered Accept, only if we have a transport for the
//                   sender; Accept -> Check; Check -> CheckAck
// CheckAck          the transport is Established: control Established to the
//                   peer, then ESTABLISHED (connections the game activated),
//                   MUTUAL_ACTIVATED (where the peer activated too),
//                   PEER_ACTIVATED (connections only the peer activated)
// deactivate        Close(deactivate) to the peer when established, then DEAD
//                   (terminated by myself); the connection is gone after
// Close             PEER_DEACTIVATED when it is a deactivate and the transport
//                   is established, else DEAD (terminated by peer)
#include "hle/np.h"

#include "core/thunk.h"
#include "guest_abi.h"
#include "hle/common.h"
#include "hle/net_p2p.h"
#include "log.h"
#include "net/session.h"

#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
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
constexpr std::uint16_t kEvPeerActivated = 0x0a;
constexpr std::uint16_t kEvPeerDeactivated = 0x0b;
constexpr std::uint16_t kEvMutualActivated = 0x0c;

// The datagram: magic, kind, reason, the sender's and the receiver's online
// ids, a nonce (Check / CheckAck).
enum Kind : std::uint8_t {
    kActivationRequest = 0x01,
    kActivationAck = 0x02,
    kClose = 0x03,
    kEstablishedMsg = 0x04,
    kOffer = 0x11,
    kAccept = 0x12,
    kCheck = 0x13,
    kCheckAck = 0x14,
};
enum Reason : std::uint8_t { kDeactivate = 1, kTerminate = 2 };
constexpr std::size_t kIdLen = 16;
constexpr std::size_t kFromAt = sizeof(kSignalMagic) + 2;
constexpr std::size_t kToAt = kFromAt + kIdLen;
constexpr std::size_t kNonceAt = kToAt + kIdLen;
constexpr std::size_t kPacketLen = kNonceAt + 8;
constexpr int kStepMs = 500;
constexpr int kGiveUpMs = 30000;
// Matching2's ESTABLISHED waits this long after the transport is up
// (shadPS4's kMatching2EstablishDelay), and its DEAD on a timeout.
constexpr int kM2EstablishDelayMs = 1000;
constexpr int kM2ErrTimeout = static_cast<int>(0x80550E11);
constexpr int kM2ErrTerminatedByPeer = static_cast<int>(0x80550E10);
constexpr int kM2ErrTerminatedByMyself = static_cast<int>(0x80550E18);

// Matching2's DEAD for that player when its connection closes, after
// NpSignaling's own DEAD on the same queue (shadPS4's
// CloseConnectionAndDispatchDead -> QueueMatching2DeadForTransport).
void queue_m2_dead(const std::string& who, int error) {
    net::dispatch_after(0, net::Prio::Signaling, [who, error] { hle_np_matching2_transport_dead(who, error); });
}

enum class State { Inactive, SendingOffer, WaitOffer, SendingAccept, WaitAccept, ConnCheck, Established };

struct Ctx {
    void* cb = nullptr;
    void* arg = nullptr;
};
struct Transport {
    std::string online_id;
    std::uint16_t member_id = 0;  // the peer table's entry, 0 until known
    std::uint32_t addr = 0;       // network order
    std::uint16_t port = 0;       // host order
    State state = State::SendingOffer;
    bool peer_activated = false;
    bool peer_established = false;
    bool established_fired = false;
    bool stepping = false;  // the resend timer is running
    int waited_ms = 0;      // handshake time so far
};
struct Conn {
    unsigned ctx = 0;
    std::string online_id;
    unsigned transport = 0;
    std::uint16_t member_id = 0;
    bool locally_activated = false;  // the game activated it
    bool peer_activated = false;     // the peer's game activated its own (its ActivationRequest)
    bool established_fired = false;
    bool mutual_fired = false;
    bool peer_activated_fired = false;
    bool dead = false;
    bool start_handshake = false;  // waiting for the peer's address, then handshake
    int resolve_waited_ms = 0;
};
using Event = std::tuple<unsigned, unsigned, std::uint16_t, int>;  // ctx, conn, event, error

std::mutex g_mu;
std::map<unsigned, Ctx> g_ctx;
std::map<unsigned, Transport> g_tr;  // by transport id
std::map<unsigned, Conn> g_conn;     // by connection id
unsigned g_next_ctx = 1;
unsigned g_next_tr = 1;
unsigned g_conn_seed = 0;  // 0 until the first connection: then a random start

// The next free connection id, as shadPS4's AllocateConnectionIdLocked: a
// random start in 1..0x7fff, then counting up and wrapping, so two games
// do not hand out the same ids (bbhost used to start every game at 1).
unsigned alloc_conn_id_locked() {
    if (g_conn_seed == 0) {
        std::random_device rd;
        g_conn_seed = (static_cast<unsigned>(rd()) / 0x4001u) & 0x7fffu;
        if (g_conn_seed == 0) g_conn_seed = 1;
    }
    for (int tried = 0; tried <= 0xffff; ++tried) {
        const unsigned candidate = g_conn_seed++;
        if (g_conn_seed == 0 || g_conn_seed > 0x7fff) g_conn_seed = 1;
        if (candidate != 0 && g_conn.find(candidate) == g_conn.end()) return candidate;
    }
    return 0;
}

bool g_trace = [] {
    const char* e = std::getenv("BBHOST_NP_TRACE");
    return e && e[0] == '1';
}();

std::uint64_t sig_now_us() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

const char* kind_name(std::uint8_t k) {
    switch (k) {
    case kActivationRequest: return "ActivationRequest";
    case kActivationAck: return "ActivationAck";
    case kClose: return "Close";
    case kEstablishedMsg: return "Established";
    case kOffer: return "Offer";
    case kAccept: return "Accept";
    case kCheck: return "Check";
    case kCheckAck: return "CheckAck";
    default: return "?";
    }
}

int status_of(const Transport& t) {
    return t.state == State::Established ? 2 : t.state == State::Inactive ? 0 : 1;
}

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

void fire_all(const std::vector<Event>& events) {
    for (const auto& [ctx, conn, event, error] : events) fire(ctx, conn, event, error);
}

void queue_all(std::vector<Event> events) {
    if (events.empty()) return;
    net::dispatch_after(0, net::Prio::Signaling, [events] { fire_all(events); });
}

// A live (not Inactive) transport to that player, or 0.
unsigned find_transport_locked(const std::string& online_id) {
    for (const auto& [id, t] : g_tr) {
        if (t.state != State::Inactive && t.online_id == online_id) return id;
    }
    return 0;
}

// The player's live transport, or a new one sending Offer. Inactive ones with
// no connection left are dropped.
unsigned ensure_transport_locked(const std::string& online_id, std::uint32_t addr, std::uint16_t port) {
    if (unsigned id = find_transport_locked(online_id)) {
        Transport& t = g_tr[id];
        if (addr && port) {
            t.addr = addr;
            t.port = port;
        }
        return id;
    }
    for (auto it = g_tr.begin(); it != g_tr.end();) {
        bool used = false;
        for (const auto& [cid, c] : g_conn) used |= c.transport == it->first && !c.dead;
        it = it->second.state == State::Inactive && !used ? g_tr.erase(it) : std::next(it);
    }
    const unsigned id = g_next_tr++;
    Transport t;
    t.online_id = online_id;
    t.addr = addr;
    t.port = port;
    net::Peer p;
    if (net::peers_find_online(online_id, &p)) {
        t.member_id = p.member_id;
        if (!t.addr && p.addr && p.port) {
            t.addr = p.addr;
            t.port = p.port;
        }
    }
    g_tr[id] = t;
    return id;
}

// The game's live connection in this context to that player, or 0.
unsigned find_conn_locked(const std::string& online_id, unsigned ctx = 0) {
    for (const auto& [id, c] : g_conn) {
        if (!c.dead && c.online_id == online_id && (!ctx || c.ctx == ctx)) return id;
    }
    return 0;
}

// The peer's current address from the room's peer table, if it is there.
void refresh_addr_locked(Transport& t) {
    net::Peer p;
    if (t.member_id && net::peers_get(t.member_id, &p) && p.addr && p.port) {
        t.addr = p.addr;
        t.port = p.port;
    }
}

void send_locked(Transport& t, Kind kind, Reason reason = kDeactivate, std::uint64_t nonce = 0) {
    refresh_addr_locked(t);
    if (!t.addr || !t.port) return;
    std::uint8_t pkt[kPacketLen] = {};
    std::memcpy(pkt, kSignalMagic, sizeof(kSignalMagic));
    pkt[4] = kind;
    pkt[5] = reason;
    const std::string me = net::online_id();
    std::memcpy(pkt + kFromAt, me.data(), std::min(me.size(), kIdLen));
    std::memcpy(pkt + kToAt, t.online_id.data(), std::min(t.online_id.size(), kIdLen));
    std::memcpy(pkt + kNonceAt, &nonce, sizeof(nonce));
    hle_net_p2p_signal(t.addr, t.port, pkt, sizeof(pkt));
}

// The peer table's view of the game's connections to that player.
void set_peer_status_locked(unsigned tid, int status) {
    for (const auto& [id, c] : g_conn) {
        if (c.transport == tid && !c.dead && c.locally_activated && c.member_id) net::peers_set_conn(c.member_id, id, status);
    }
}

// The handshake step this transport is at, sent (shadPS4's
// StartTransportHandshakeInitiator).
void send_handshake_locked(Transport& t) {
    Kind kind = kOffer;
    std::uint64_t nonce = 0;
    switch (t.state) {
    case State::SendingAccept: kind = kAccept; break;
    case State::ConnCheck:
        kind = kCheck;
        nonce = sig_now_us();
        break;
    case State::Established: return;
    case State::WaitOffer:
    case State::Inactive: t.state = State::SendingOffer; [[fallthrough]];
    default: kind = kOffer; break;
    }
    send_locked(t, kind, kDeactivate, nonce);
}

void close_transport_dead_locked(unsigned tid, int error, std::vector<Event>& events);
void step(unsigned tid);

// The resend timer for a transport still in its handshake.
void start_stepping_locked(unsigned tid) {
    Transport& t = g_tr[tid];
    if (t.stepping) return;
    t.stepping = true;
    net::dispatch_after(kStepMs, net::Prio::Signaling, [tid] { step(tid); });
}

// Every 500 ms until Established: the current step again; DEAD after 30 s.
void step(unsigned tid) {
    std::vector<Event> events;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_tr.find(tid);
        if (it == g_tr.end()) return;
        Transport& t = it->second;
        if (t.state == State::Established || t.state == State::Inactive) {
            t.stepping = false;
            return;
        }
        if (t.addr && t.port) {
            if (t.waited_ms >= kGiveUpMs) {
                host_log("np: transport %u to %s: handshake not done in %d s: DEAD (timeout)", tid, t.online_id.c_str(),
                         kGiveUpMs / 1000);
                const std::string who = t.online_id;
                close_transport_dead_locked(tid, kErrTimeout, events);  // erases the transport
                net::dispatch_after(0, net::Prio::Signaling, [who] { hle_np_matching2_transport_dead(who, kM2ErrTimeout); });
            } else {
                send_handshake_locked(t);
                t.waited_ms += kStepMs;
                net::dispatch_after(kStepMs, net::Prio::Signaling, [tid] { step(tid); });
            }
        } else {
            net::dispatch_after(kStepMs, net::Prio::Signaling, [tid] { step(tid); });
        }
    }
    fire_all(events);
}

// The transport is up: control Established to the peer, and each connection
// on it its events, once each (shadPS4's EstablishConnection).
void establish_locked(unsigned tid, bool peer_activated_hint, std::vector<Event>& events) {
    Transport& t = g_tr[tid];
    if (peer_activated_hint) t.peer_activated = true;
    if (!t.established_fired) {
        t.state = State::Established;
        t.established_fired = true;
        send_locked(t, kEstablishedMsg);
        set_peer_status_locked(tid, 2);
        const std::string who = t.online_id;
        net::dispatch_after(kM2EstablishDelayMs, net::Prio::Signaling, [who] { hle_np_matching2_transport_up(who); });
    }
    for (auto& [id, c] : g_conn) {
        if (c.transport != tid || c.dead) continue;
        if (c.peer_activated && !c.locally_activated && !c.peer_activated_fired) {
            c.peer_activated_fired = true;
            events.emplace_back(c.ctx, id, kEvPeerActivated, 0);
        }
        if (c.locally_activated && !c.established_fired) {
            c.established_fired = true;
            events.emplace_back(c.ctx, id, kEvEstablished, 0);
        }
        if (c.locally_activated && c.peer_activated && !c.mutual_fired) {
            c.mutual_fired = true;
            events.emplace_back(c.ctx, id, kEvMutualActivated, 0);
        }
    }
}

// One connection closed: its transport Inactive, DEAD, and the connection gone.
void close_conn_dead_locked(unsigned conn_id, int error, std::vector<Event>& events) {
    auto it = g_conn.find(conn_id);
    if (it == g_conn.end() || it->second.dead) return;
    Conn& c = it->second;
    auto tr = g_tr.find(c.transport);
    if (tr != g_tr.end()) tr->second.state = State::Inactive;
    c.dead = true;
    if (c.member_id) net::peers_set_conn(c.member_id, conn_id, 0);
    events.emplace_back(c.ctx, conn_id, kEvDead, error);
    g_conn.erase(it);
}

// The whole transport closed: every connection on it DEAD and gone.
void close_transport_dead_locked(unsigned tid, int error, std::vector<Event>& events) {
    auto tr = g_tr.find(tid);
    if (tr == g_tr.end() || tr->second.state == State::Inactive) return;
    tr->second.state = State::Inactive;
    std::vector<unsigned> conns;
    for (const auto& [id, c] : g_conn) {
        if (c.transport == tid && !c.dead) conns.push_back(id);
    }
    for (unsigned id : conns) {
        Conn& c = g_conn[id];
        c.dead = true;
        if (c.member_id) net::peers_set_conn(c.member_id, id, 0);
        events.emplace_back(c.ctx, id, kEvDead, error);
        g_conn.erase(id);
    }
    g_tr.erase(tr);
}

// ActivationRequest out, then the handshake, once the peer's address is known
// (shadPS4's ProcessPendingActivations): the room's peer table, else the
// server. Retried every 500 ms; DEAD (timeout) after 30 s without one.
void activation(unsigned conn_id) {
    std::string peer;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_conn.find(conn_id);
        if (it == g_conn.end() || it->second.dead) return;
        Transport& t = g_tr[it->second.transport];
        refresh_addr_locked(t);
        if (!t.addr || !t.port) {
            net::Peer p;
            if (net::peers_find_online(t.online_id, &p) && p.addr && p.port) {
                t.member_id = p.member_id;
                it->second.member_id = p.member_id;
                t.addr = p.addr;
                t.port = p.port;
            }
        }
        if (!t.addr || !t.port) peer = t.online_id;
    }
    if (!peer.empty()) {
        // Not a room member yet (the host activates its connection to the
        // guest it summons before the guest joins): the server has its
        // address from its context_start.
        std::uint32_t addr = 0;
        std::uint16_t port = 0;
        std::string error;
        const bool ok = net::server_signaling_resolve(peer, &addr, &port, error);
        std::vector<Event> events;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            auto it = g_conn.find(conn_id);
            if (it == g_conn.end() || it->second.dead) return;
            Conn& c = it->second;
            Transport& t = g_tr[c.transport];
            if (ok && !t.addr) {
                const net::Peer p = net::peers_provisional(peer, addr, port);
                t.member_id = p.member_id;
                c.member_id = p.member_id;
                t.addr = addr;
                t.port = port;
                host_log("np: conn %u to %s resolved %u.%u.%u.%u:%u", conn_id, peer.c_str(), addr & 0xff,
                         (addr >> 8) & 0xff, (addr >> 16) & 0xff, addr >> 24, port);
            } else if (!t.addr) {
                c.resolve_waited_ms += kStepMs;
                if (c.resolve_waited_ms >= kGiveUpMs) {
                    host_log("np: conn %u to %s: no address in %d s (%s): DEAD (timeout)", conn_id, peer.c_str(),
                             kGiveUpMs / 1000, error.c_str());
                    close_conn_dead_locked(conn_id, kErrTimeout, events);
                } else {
                    net::dispatch_after(kStepMs, net::Prio::Signaling, [conn_id] { activation(conn_id); });
                }
            }
        }
        fire_all(events);
        if (!ok) return;
    }
    std::vector<Event> events;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_conn.find(conn_id);
        if (it == g_conn.end() || it->second.dead) return;
        Conn& c = it->second;
        const unsigned tid = c.transport;
        Transport& t = g_tr[tid];
        if (!t.addr || !t.port) return;
        if (c.member_id) net::peers_set_conn(c.member_id, conn_id, status_of(t));
        send_locked(t, kActivationRequest);
        if (c.start_handshake) {
            c.start_handshake = false;
            send_handshake_locked(t);
            start_stepping_locked(tid);
        } else if (t.state == State::Established) {
            establish_locked(tid, false, events);
        }
    }
    fire_all(events);
}

GUEST_ABI int sig_init(std::uint64_t, int, int, std::uint64_t) {
    host_log("sceNpSignalingInitialize");
    return 0;
}
GUEST_ABI int sig_term() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_ctx.clear();
    g_conn.clear();
    g_tr.clear();
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

// (ctxId, const SceNpId* peer, connId*), as shadPS4's ActivateSig1: one live
// connection per context and player; a new or re-activated one sends
// ActivationRequest and starts the handshake, unless its transport is
// already up (then ESTABLISHED at once).
GUEST_ABI int sig_activate(unsigned ctx, const void* npid, unsigned* conn_out) {
    if (!conn_out) return kErrInvalidArg;
    const std::string peer = hle_np_npid_text(npid);
    std::vector<Event> events;
    unsigned id = 0;
    bool queue = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        id = find_conn_locked(peer, ctx);
        const unsigned tid = ensure_transport_locked(peer, 0, 0);
        Transport& t = g_tr[tid];
        if (!id) {
            id = alloc_conn_id_locked();
            Conn c;
            c.ctx = ctx;
            c.online_id = peer;
            c.transport = tid;
            c.member_id = t.member_id;
            c.locally_activated = true;
            c.start_handshake = t.state != State::Established;
            g_conn[id] = c;
            queue = true;
        } else {
            Conn& c = g_conn[id];
            const bool was_local = c.locally_activated;
            c.transport = tid;
            c.locally_activated = true;
            if (!c.member_id) c.member_id = t.member_id;
            if (t.state != State::Established) {
                if (t.state == State::Inactive || t.state == State::WaitOffer) t.state = State::SendingOffer;
                c.start_handshake = true;
            }
            queue = !was_local;
        }
        host_log("sceNpSignalingActivateConnection %s -> conn %u%s", peer.c_str(), id,
                 t.state == State::Established ? " (transport already up)" : "");
        if (t.state == State::Established) establish_locked(tid, false, events);
    }
    *conn_out = id;
    if (queue) net::dispatch_after(0, net::Prio::Signaling, [id] { activation(id); });
    queue_all(std::move(events));
    return 0;
}

// As shadPS4's DeactivateSig1: Close(deactivate) to the peer when the
// transport is up, then DEAD, and the connection is gone (a second
// deactivate is CONN_NOT_FOUND). NpMatching2's DEAD for that player is kept
// for its LeaveRoom.
GUEST_ABI int sig_deactivate(unsigned ctx, unsigned conn) {
    std::string peer;
    std::vector<Event> events;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_conn.find(conn);
        if (it == g_conn.end() || it->second.ctx != ctx) {
            host_log("sceNpSignalingDeactivateConnection conn %u: not found", conn);
            return kErrConnNotFound;
        }
        Conn& c = it->second;
        host_log("sceNpSignalingDeactivateConnection conn %u (%s)", conn, c.online_id.c_str());
        peer = c.online_id;
        auto tr = g_tr.find(c.transport);
        if (tr != g_tr.end() && tr->second.state == State::Established && !c.dead) send_locked(tr->second, kClose, kDeactivate);
        close_conn_dead_locked(conn, kErrTerminatedByMyself, events);
    }
    hle_np_matching2_peer_closed(peer);
    queue_all(std::move(events));
    queue_m2_dead(peer, kM2ErrTerminatedByMyself);
    return 0;
}

// (ctxId, connId, status*, SceNetInAddr*, SceInPort_t*): 0 inactive, 1
// pending, 2 active - the connection's transport's.
GUEST_ABI int sig_status(unsigned ctx, unsigned conn, int* st, std::uint32_t* addr, std::uint16_t* port) {
    int status = 0;
    std::uint32_t a = 0;
    std::uint16_t p = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_conn.find(conn);
        if (it != g_conn.end() && it->second.ctx == ctx && !it->second.dead) {
            auto tr = g_tr.find(it->second.transport);
            if (tr != g_tr.end()) {
                Transport& t = tr->second;
                status = status_of(t);
                if (status == 2) {
                    refresh_addr_locked(t);
                    a = t.addr;
                    p = static_cast<std::uint16_t>((t.port << 8) | (t.port >> 8));
                }
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

// A signaling datagram from another bbhost (hle/net.cpp), on the dispatcher.
void hle_np_signaling_control(const std::uint8_t* data, std::size_t len, std::uint32_t addr, std::uint16_t port) {
    if (len < kPacketLen) return;
    const std::uint8_t kind = data[4];
    const Reason reason = static_cast<Reason>(data[5]);
    std::string from(reinterpret_cast<const char*>(data + kFromAt), kIdLen);
    from.resize(std::strlen(from.c_str()));
    if (from.empty()) return;
    std::uint64_t nonce = 0;
    std::memcpy(&nonce, data + kNonceAt, sizeof(nonce));
    const std::uint16_t from_port = static_cast<std::uint16_t>((port << 8) | (port >> 8));
    if (g_trace) host_log("np: signaling %s from %s", kind_name(kind), from.c_str());
    std::vector<Event> events;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        unsigned tid = find_transport_locked(from);
        switch (kind) {
        case kActivationRequest: {
            // The peer's game activated its connection to us.
            if (!tid) tid = ensure_transport_locked(from, addr, from_port);
            Transport& t = g_tr[tid];
            t.peer_activated = true;
            t.addr = addr;
            t.port = from_port;
            if (t.state == State::Inactive) t.state = State::WaitOffer;
            const unsigned ctx = g_ctx.empty() ? 0 : g_ctx.begin()->first;
            unsigned cid = ctx ? find_conn_locked(from, ctx) : 0;
            if (ctx && !cid) {
                cid = alloc_conn_id_locked();
                Conn c;
                c.ctx = ctx;
                c.online_id = from;
                c.transport = tid;
                c.member_id = t.member_id;
                c.peer_activated = true;
                g_conn[cid] = c;
            }
            if (cid) {
                g_conn[cid].peer_activated = true;
                g_conn[cid].transport = tid;
            }
            if (t.state == State::Established) establish_locked(tid, false, events);
            else start_stepping_locked(tid);
            send_locked(t, kActivationAck);
            break;
        }
        case kOffer: {
            if (!tid) break;  // no transport for the sender: ignored
            Transport& t = g_tr[tid];
            t.addr = addr;
            t.port = from_port;
            t.peer_activated = true;
            if (t.state != State::Established) t.state = State::SendingAccept;
            send_locked(t, kAccept);
            break;
        }
        case kAccept: {
            if (!tid) break;
            Transport& t = g_tr[tid];
            t.peer_activated = true;
            t.addr = addr;
            t.port = from_port;
            if (t.state == State::SendingOffer || t.state == State::WaitAccept || t.state == State::SendingAccept ||
                t.state == State::WaitOffer) {
                t.state = State::ConnCheck;
                send_locked(t, kCheck, kDeactivate, sig_now_us());
            }
            break;
        }
        case kCheck: {
            if (!tid) break;
            Transport& t = g_tr[tid];
            send_locked(t, kCheckAck, kDeactivate, nonce);
            if (t.state == State::SendingAccept || t.state == State::WaitOffer) t.state = State::ConnCheck;
            break;
        }
        case kCheckAck: {
            if (!tid) break;
            if (g_tr[tid].state != State::Established) establish_locked(tid, true, events);
            break;
        }
        case kActivationAck: {
            if (!tid) break;
            g_tr[tid].peer_activated = true;
            break;
        }
        case kEstablishedMsg: {
            if (!tid) break;
            Transport& t = g_tr[tid];
            t.peer_established = true;
            if (t.state == State::Established) establish_locked(tid, false, events);
            break;
        }
        case kClose: {
            if (!tid) break;
            const unsigned cid = find_conn_locked(from);
            if (reason == kDeactivate && cid && g_tr[tid].state == State::Established) {
                events.emplace_back(g_conn[cid].ctx, cid, kEvPeerDeactivated, 0);
            } else {
                if (cid) close_conn_dead_locked(cid, kErrTerminatedByPeer, events);
                else g_tr[tid].state = State::Inactive;
                queue_m2_dead(from, kM2ErrTerminatedByPeer);
            }
            break;
        }
        default:
            break;
        }
    }
    fire_all(events);
}

bool hle_np_signaling_live(const std::string& online_id) {
    std::lock_guard<std::mutex> lk(g_mu);
    return find_conn_locked(online_id) != 0;
}

// shadPS4's StartMatching2, for one member.
void hle_np_signaling_m2_connect(const std::string& online_id) {
    std::lock_guard<std::mutex> lk(g_mu);
    const unsigned tid = ensure_transport_locked(online_id, 0, 0);
    Transport& t = g_tr[tid];
    refresh_addr_locked(t);
    if (t.state == State::WaitOffer) t.state = State::SendingOffer;
    if (g_trace) host_log("np: matching2 signaling to %s: transport %u status %d", online_id.c_str(), tid, status_of(t));
    if (t.state == State::Established) {
        const std::string who = online_id;
        net::dispatch_after(kM2EstablishDelayMs, net::Prio::Signaling, [who] { hle_np_matching2_transport_up(who); });
        return;
    }
    send_handshake_locked(t);
    start_stepping_locked(tid);
}

int hle_np_signaling_transport_status(const std::string& online_id) {
    std::lock_guard<std::mutex> lk(g_mu);
    const unsigned tid = find_transport_locked(online_id);
    return tid ? status_of(g_tr[tid]) : 0;
}

// shadPS4's StopMatching2, for one member: the transport closes unless an
// NpSignaling connection of the game's is on it.
void hle_np_signaling_m2_stop(const std::string& online_id) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (find_conn_locked(online_id)) return;
    if (unsigned tid = find_transport_locked(online_id)) g_tr.erase(tid);
}

// A player is now a room member: its transport and connections know its
// peer-table entry (the address), and a connection waiting for it goes on.
void hle_np_signaling_peer_known(std::uint16_t member_id) {
    net::Peer p;
    if (!net::peers_get(member_id, &p)) return;
    std::vector<unsigned> waiting;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [id, t] : g_tr) {
            if (t.online_id != p.online_id || t.state == State::Inactive) continue;
            t.member_id = member_id;
            if (p.addr && p.port) {
                t.addr = p.addr;
                t.port = p.port;
            }
        }
        for (auto& [id, c] : g_conn) {
            if (c.dead || c.online_id != p.online_id) continue;
            c.member_id = member_id;
            if (c.start_handshake) waiting.push_back(id);
        }
    }
    for (unsigned id : waiting) net::dispatch_after(0, net::Prio::Signaling, [id] { activation(id); });
}

// The player left the room for good: its connections DEAD (terminated by peer).
void hle_np_signaling_peer_dead(const std::string& online_id) {
    std::vector<Event> events;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (unsigned tid = find_transport_locked(online_id)) close_transport_dead_locked(tid, kErrTerminatedByPeer, events);
        while (unsigned cid = find_conn_locked(online_id)) close_conn_dead_locked(cid, kErrTerminatedByPeer, events);
    }
    queue_all(std::move(events));
}

void hle_np_signaling_reset() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_conn.clear();
    g_tr.clear();
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
