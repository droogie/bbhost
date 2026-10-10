#include "net/session.h"

#include "core/config.h"
#include "core/portable.h"
#include "core/thunk.h"
#include "hle/net_p2p.h"
#include "hle/hle.h"
#include "log.h"
#include "net/account.h"
#include "net/http.h"

#if defined(_WIN32)
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#endif
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>

namespace net {

namespace {

using clock = std::chrono::steady_clock;

// ---- the dispatcher -------------------------------------------------------

struct Due {
    clock::time_point at;
    int prio;
    std::uint64_t seq;
    std::function<void()> fn;
    bool operator<(const Due& o) const {
        if (at != o.at) return at < o.at;
        if (prio != o.prio) return prio < o.prio;
        return seq < o.seq;
    }
};

std::mutex g_dmu;
std::condition_variable g_dcv;
std::deque<Due> g_due;  // kept sorted on insert
std::uint64_t g_seq = 0;
std::thread g_dthread;
std::atomic<bool> g_drun{false};
std::atomic<std::thread::id> g_dtid{};

void dispatcher_main() {
    host_thread_set_name("bb-np");
    // One guest TCB for the life of the thread: the callbacks the game
    // registered read fs:-relative state, and this is the only thread of ours
    // that calls them (avplayer.cpp does the same for its event callback).
    void* tcb = hle_thread_enter_guest();
    // Host code (libcurl, the allocator) runs with host FS; hle_call_guest6
    // switches to the guest TCB for each callback and back (avplayer.cpp).
    hle_fs_host();
    g_dtid.store(std::this_thread::get_id());
    for (;;) {
        std::function<void()> fn;
        {
            std::unique_lock<std::mutex> lk(g_dmu);
            for (;;) {
                if (!g_drun.load()) break;
                if (!g_due.empty()) {
                    const auto now = clock::now();
                    if (g_due.front().at <= now) {
                        fn = std::move(g_due.front().fn);
                        g_due.pop_front();
                        break;
                    }
                    g_dcv.wait_until(lk, g_due.front().at);
                } else {
                    g_dcv.wait(lk);
                }
            }
            if (!g_drun.load() && !fn) break;
        }
        if (fn) {
            fn();
        }
    }
    hle_thread_leave_guest(tcb);
}

void dispatcher_ensure_locked() {
    if (g_drun.load()) return;
    g_drun.store(true);
    g_dthread = std::thread(dispatcher_main);
    g_dthread.detach();
}

// ---- the peer table -------------------------------------------------------

std::mutex g_pmu;
std::map<std::uint16_t, Peer> g_peers;

std::uint32_t addr_of(const std::string& text) {
    in_addr a{};
    if (text.empty() || inet_pton(AF_INET, text.c_str(), &a) != 1) {
        return 0;
    }
    return a.s_addr;  // already network order
}

// ---- the poller -----------------------------------------------------------

std::mutex g_pollmu;
std::thread g_pollthread;
std::atomic<bool> g_pollrun{false};
std::condition_variable g_pollcv;
std::function<void(const json::Value&)> g_handler;
long long g_cursor = 0;

void poller_main() {
    host_thread_set_name("bb-np-poll");
    const std::string me = online_id();
    int interval_ms = 250;
    while (g_pollrun.load()) {
        json::Value body = json::Value::make_object();
        body.set("OnlineId", me);
        body.set("Cursor", static_cast<long long>(g_cursor));
        json::Value cats = json::Value::make_array();
        cats.push("matching2");
        body.set("Categories", std::move(cats));
        json::Value reply;
        std::string err;
        if (np_post("/np/events/poll", body, reply, err, 4000)) {
            interval_ms = 250;
            const long long next = int_of(reply, "NextCursor", g_cursor);
            if (int_of(reply, "HasEvent", 0) != 0) {
                std::function<void(const json::Value&)> h;
                {
                    std::lock_guard<std::mutex> lk(g_pollmu);
                    h = g_handler;
                }
                host_log("np: event #%lld %s", int_of(reply, "EventId", 0), str_of(reply, "Name").c_str());
                if (h) {
                    h(reply);
                }
                json::Value ack = json::Value::make_object();
                ack.set("OnlineId", me);
                ack.set("Cursor", next);
                json::Value ackreply;
                std::string ackerr;
                np_post("/np/events/ack", ack, ackreply, ackerr, 4000);
                g_cursor = next;
                continue;  // drain without waiting when there was one
            }
            g_cursor = next > g_cursor ? next : g_cursor;
        } else if (account_refused()) {
            // The server refused the sign-in (net/account.h): nothing will
            // arrive until it is linked again, so ask once a minute, not every
            // 4 s - a refused poller ran on dev for days, ~900 requests an hour.
            interval_ms = 60000;
        } else {
            // Back off while the server is unreachable; the game keeps running.
            interval_ms = interval_ms < 4000 ? interval_ms * 2 : 4000;
        }
        std::unique_lock<std::mutex> lk(g_pollmu);
        g_pollcv.wait_for(lk, std::chrono::milliseconds(interval_ms), [] { return !g_pollrun.load(); });
    }
}

// The reflexive address of our P2P port: a STUN Binding exchange
// with online.stun_server (default: the private server's host on 3478,
// "off" disables), asked at context_start and again when the cached answer
// is over two minutes old. The server records it as MappedAddr/MappedPort
// and hands it to peers instead of the LAN address when the two differ.
struct Mapped {
    std::mutex mu;
    std::string addr;
    int port = 0;
    std::int64_t at_ms = 0;
    bool asked = false;
};
Mapped g_mapped;
std::atomic<int> g_stun_rtt_us{0};

void stun_server(std::string* host, std::uint16_t* port) {
    std::string spec = config().stun_server;
    if (spec.empty()) {
        // The private server's host, from the API base (scheme://host[:port]).
        std::string base = np_server_base();
        const std::size_t s0 = base.find("://");
        if (s0 != std::string::npos) base = base.substr(s0 + 3);
        const std::size_t slash = base.find('/');
        if (slash != std::string::npos) base = base.substr(0, slash);
        const std::size_t colon = base.find(':');
        spec = colon == std::string::npos ? base : base.substr(0, colon);
    }
    *port = 3478;
    const std::size_t colon = spec.find(':');
    if (colon != std::string::npos) {
        *port = static_cast<std::uint16_t>(std::atoi(spec.c_str() + colon + 1));
        spec = spec.substr(0, colon);
    }
    *host = spec;
}

void mapped_refresh(bool force) {
    std::lock_guard<std::mutex> lk(g_mapped.mu);
    if (!force && g_mapped.asked && now_ms() - g_mapped.at_ms < 120000) return;
    g_mapped.asked = true;
    g_mapped.at_ms = now_ms();
    if (config().stun_server == "off") return;
    std::string host;
    std::uint16_t sport = 0;
    stun_server(&host, &sport);
    if (host.empty()) return;
    std::uint32_t addr = 0;
    std::uint16_t port = 0;
    bool ok = false;
    // The relay HELLO unless BBHOST_NET_RELAY=0 (then datagrams go to the
    // server's relay ports as addressed, the per-port relay of PS4 clients).
    static const bool use_relay = [] {
        const char* e = std::getenv("BBHOST_NET_RELAY");
        return !(e && e[0] == '0');
    }();
    stun::Relay relay;
    for (int attempt = 0; attempt < 3 && !ok; ++attempt) {
        const auto t0 = std::chrono::steady_clock::now();
        ok = hle_net_p2p_stun(host.c_str(), sport, 1000, &addr, &port, use_relay ? &relay : nullptr);
        if (ok) {
            g_stun_rtt_us = static_cast<int>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    if (!ok) {
        if (!g_mapped.addr.empty()) return;  // keep the last answer
        host_log("stun: no answer from %s:%u; the server gets the LAN address only (online.p2p_addr)", host.c_str(),
                 sport);
        return;
    }
    char text[32];
    std::snprintf(text, sizeof(text), "%u.%u.%u.%u", addr & 0xff, (addr >> 8) & 0xff, (addr >> 16) & 0xff, addr >> 24);
    if (g_mapped.addr != text || g_mapped.port != port) {
        host_log("stun: %s:%u sees our P2P port %d as %s:%u%s", host.c_str(), sport, p2p_port(), text, port,
                 (text == local_addr_text() && port == p2p_port()) ? " (no NAT in between)" : "");
    }
    g_mapped.addr = text;
    g_mapped.port = port;
}

json::Value our_endpoint() {
    mapped_refresh(false);
    json::Value o = json::Value::make_object();
    o.set("OnlineId", online_id());
    o.set("LocalAddr", local_addr_text());
    o.set("LocalPort", p2p_port());
    o.set("PublicAddr", local_addr_text());
    o.set("PublicPort", p2p_port());
    std::lock_guard<std::mutex> lk(g_mapped.mu);
    o.set("MappedAddr", g_mapped.addr);
    o.set("MappedPort", g_mapped.port);
    return o;
}

// A peer with a new address: probes toward it so our NAT lets its
// datagrams in (hle_net_p2p_punch), never toward ourselves.
void punch_if_new(const Peer& before, const Peer& after) {
    if (after.online_id == online_id() || !after.addr || !after.port) return;
    if (before.addr == after.addr && before.port == after.port && before.local_addr == after.local_addr &&
        before.local_port == after.local_port) {
        return;
    }
    hle_net_p2p_punch(after.online_id.c_str(), after.addr, after.port, after.local_addr, after.local_port);
}

}  // namespace

void dispatch_after(int delay_ms, Prio prio, std::function<void()> fn) {
    std::lock_guard<std::mutex> lk(g_dmu);
    dispatcher_ensure_locked();
    Due d{clock::now() + std::chrono::milliseconds(delay_ms < 0 ? 0 : delay_ms), static_cast<int>(prio), g_seq++,
          std::move(fn)};
    auto it = g_due.begin();
    while (it != g_due.end() && *it < d) ++it;
    g_due.insert(it, std::move(d));
    g_dcv.notify_all();
}

bool on_dispatcher() { return g_dtid.load() == std::this_thread::get_id(); }

void session_shutdown() {
    poller_stop();
    {
        std::lock_guard<std::mutex> lk(g_dmu);
        if (!g_drun.load()) return;
        g_drun.store(false);
        g_dcv.notify_all();
    }
}

std::string online_id() {
    // The account's name when logged in; the server enforces the
    // same through the token header, this just keeps the two in step.
    if (const std::string acct = account_name(); !acct.empty()) return acct;
    const std::string& id = config().online_id;
    return id.empty() ? "Player" : id;
}

std::string local_addr_text() {
    const std::string& a = config().p2p_addr;
    return a.empty() ? "127.0.0.1" : a;
}

namespace {
std::atomic<int> g_p2p_port_bound{0};
}  // namespace

int p2p_port() {
    if (const int b = g_p2p_port_bound.load(std::memory_order_relaxed); b > 0) return b;
    const int p = config().p2p_port;
    return p > 0 ? p : 9307;
}

void set_p2p_port_bound(int port) { g_p2p_port_bound.store(port, std::memory_order_relaxed); }

void peers_clear() {
    std::lock_guard<std::mutex> lk(g_pmu);
    g_peers.clear();
}

void peers_upsert(const Peer& p) {
    Peer before;
    {
        std::lock_guard<std::mutex> lk(g_pmu);
        Peer& slot = g_peers[p.member_id];
        before = slot;
        const unsigned conn = p.conn_id ? p.conn_id : slot.conn_id;
        const int sig = p.sig_status ? p.sig_status : slot.sig_status;
        slot = p;
        slot.conn_id = conn;
        slot.sig_status = sig;
    }
    punch_if_new(before, p);
}

bool peers_get(std::uint16_t member_id, Peer* out) {
    std::lock_guard<std::mutex> lk(g_pmu);
    auto it = g_peers.find(member_id);
    if (it == g_peers.end()) return false;
    if (out) *out = it->second;
    return true;
}

bool peers_find_online(const std::string& id, Peer* out) {
    std::lock_guard<std::mutex> lk(g_pmu);
    for (const auto& [m, p] : g_peers) {
        if (p.online_id == id) {
            if (out) *out = p;
            return true;
        }
    }
    return false;
}

bool peers_find_conn(unsigned conn_id, Peer* out) {
    std::lock_guard<std::mutex> lk(g_pmu);
    for (const auto& [m, p] : g_peers) {
        if (p.conn_id == conn_id && conn_id != 0) {
            if (out) *out = p;
            return true;
        }
    }
    return false;
}

void peers_set_conn(std::uint16_t member_id, unsigned conn_id, int sig_status) {
    std::lock_guard<std::mutex> lk(g_pmu);
    auto it = g_peers.find(member_id);
    if (it == g_peers.end()) return;
    it->second.conn_id = conn_id;
    it->second.sig_status = sig_status;
}

void peers_erase(std::uint16_t member_id) {
    std::lock_guard<std::mutex> lk(g_pmu);
    g_peers.erase(member_id);
}

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::vector<Peer> peers_all() {
    std::lock_guard<std::mutex> lk(g_pmu);
    std::vector<Peer> v;
    for (const auto& [m, p] : g_peers) v.push_back(p);
    return v;
}

int peers_from_members(const json::Value& members) {
    if (members.type != json::Value::Type::Array) return 0;
    int n = 0;
    for (const json::Value& m : members.array) {
        Peer p;
        p.member_id = static_cast<std::uint16_t>(int_of(m, "MemberId", 0));
        if (!p.member_id) continue;
        p.online_id = str_of(m, "OnlineId");
        p.addr = addr_of(str_of(m, "Addr"));
        p.port = static_cast<std::uint16_t>(int_of(m, "Port", 0));
        p.local_addr = addr_of(str_of(m, "LocalAddr"));
        p.local_port = static_cast<std::uint16_t>(int_of(m, "LocalPort", 0));
        // The server's own rule, applied here too in case a record carries
        // the raw public address: a STUN-mapped address that differs from
        // the peer's local one means a NAT in between, and only the mapped
        // one reaches the peer.
        const std::uint32_t mapped = addr_of(str_of(m, "MappedAddr"));
        const int mapped_port = static_cast<int>(int_of(m, "MappedPort", 0));
        if (mapped && mapped_port > 0 && mapped_port <= 65535 && mapped != p.local_addr && mapped != p.addr) {
            p.addr = mapped;
            p.port = static_cast<std::uint16_t>(mapped_port);
        }
        if (!p.addr) {
            p.addr = p.local_addr;
            p.port = p.local_port;
        }
        peers_upsert(p);
        ++n;
    }
    return n;
}

bool server_signaling_resolve(const std::string& id, std::uint32_t* addr, std::uint16_t* port, std::string& error) {
    json::Value body = json::Value::make_object();
    body.set("OnlineId", id);
    json::Value reply;
    if (!np_post("/np/signaling/resolve", body, reply, error)) return false;
    if (int_of(reply, "ResKind", -1) != 0) {
        error = "not registered";
        return false;
    }
    const std::uint32_t a = addr_of(str_of(reply, "Addr"));
    const int p = static_cast<int>(int_of(reply, "Port", 0));
    if (!a || p <= 0 || p > 65535) {
        error = "no address";
        return false;
    }
    if (addr) *addr = a;
    if (port) *port = static_cast<std::uint16_t>(p);
    return true;
}

Peer peers_provisional(const std::string& id, std::uint32_t addr, std::uint16_t port) {
    Peer before, after;
    {
        std::lock_guard<std::mutex> lk(g_pmu);
        bool found = false;
        for (auto& [m, p] : g_peers) {
            if (p.online_id == id && m >= 0xff00) {
                before = p;
                p.addr = addr;
                p.port = port;
                after = p;
                found = true;
                break;
            }
        }
        if (!found) {
            std::uint16_t m = 0xff00;
            while (g_peers.count(m)) ++m;
            after.member_id = m;
            after.online_id = id;
            after.addr = addr;
            after.port = port;
            g_peers[m] = after;
        }
    }
    punch_if_new(before, after);
    return after;
}

bool server_context_start(json::Value& reply, std::string& error) {
    mapped_refresh(true);
    json::Value body = json::Value::make_object();
    body.set("OnlineId", online_id());
    body.set("SignalingAddr", local_addr_text());
    body.set("SignalingPort", p2p_port());
    {
        std::lock_guard<std::mutex> lk(g_mapped.mu);
        body.set("MappedAddr", g_mapped.addr);
        body.set("MappedPort", g_mapped.port);
    }
    return np_post("/mp/matching2/context_start", body, reply, error);
}

bool server_create_room(int max_members, const json::Value& extra, json::Value& reply, std::string& error) {
    json::Value body = our_endpoint();
    body.set("MaxMembers", max_members);
    if (extra.type == json::Value::Type::Object) {
        for (const auto& [k, v] : extra.object) body.set(k, v);
    }
    return np_post("/mp/matching2/create_room", body, reply, error);
}

bool server_join_room(std::uint64_t room_id, json::Value& reply, std::string& error) {
    json::Value body = our_endpoint();
    body.set("RoomId", static_cast<long long>(room_id));
    return np_post("/mp/matching2/join_room", body, reply, error);
}

bool server_leave_room(const std::string& session_id, int member_id, json::Value& reply, std::string& error) {
    json::Value body = json::Value::make_object();
    body.set("SessionId", session_id);
    body.set("MemberId", member_id);
    return np_post("/mp/matching2/leave_room", body, reply, error);
}

int server_heartbeat(const std::string& session_id, int member_id) {
    json::Value body = json::Value::make_object();
    body.set("SessionId", session_id);
    body.set("MemberId", member_id);
    json::Value reply;
    std::string err;
    if (!np_post("/mp/matching2/heartbeat", body, reply, err, 3000)) return -1;
    // A server from before 2026-10-06 answers OK whatever happened.
    return int_of(reply, "InRoom", 1) == 0 ? 0 : 1;
}

bool server_kick_member(const std::string& session_id, int member_id, int kicker_id, const std::uint8_t* opt,
                        std::size_t opt_len, std::string& error) {
    static const char* const b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string enc;
    opt_len = opt_len > 16 ? 16 : opt_len;
    for (std::size_t i = 0; i < opt_len; i += 3) {
        const std::uint32_t v = (static_cast<std::uint32_t>(opt[i]) << 16) |
                                (i + 1 < opt_len ? static_cast<std::uint32_t>(opt[i + 1]) << 8 : 0u) |
                                (i + 2 < opt_len ? opt[i + 2] : 0u);
        enc += b64[(v >> 18) & 63];
        enc += b64[(v >> 12) & 63];
        enc += i + 1 < opt_len ? b64[(v >> 6) & 63] : '=';
        enc += i + 2 < opt_len ? b64[v & 63] : '=';
    }
    json::Value body = json::Value::make_object();
    body.set("SessionId", session_id);
    body.set("MemberId", member_id);
    body.set("KickerMemberId", kicker_id);
    body.set("OptData", enc);
    json::Value reply;
    return np_post("/mp/matching2/kick_member", body, reply, error, 4000);
}

bool server_session_blob(json::Value& reply, std::string& error) {
    return np_get("/mp/matching2/session_blob?OnlineId=" + online_id(), reply, error);
}

namespace {

std::mutex g_roommu;
std::string g_room_sid;
int g_room_mid = 0;

// Tells the server our new reflexive address (a relay port that changed:
// the server restarted, the port was released while we were away).
void server_signaling_update() {
    json::Value body = json::Value::make_object();
    body.set("OnlineId", online_id());
    {
        std::lock_guard<std::mutex> lk(g_mapped.mu);
        body.set("MappedAddr", g_mapped.addr);
        body.set("MappedPort", g_mapped.port);
    }
    {
        std::lock_guard<std::mutex> lk(g_roommu);
        body.set("SessionId", g_room_sid);
        body.set("MemberId", g_room_mid);
    }
    json::Value reply;
    std::string err;
    np_post("/mp/matching2/signaling_update", body, reply, err, 4000);
}

// While the context runs: the STUN binding every 15 s. It keeps our NAT
// mapping to the server open (a router forgets an idle UDP mapping in 30 s
// to a few minutes, and then nothing the relay sends gets in) and the
// relay port alive (released after 10 idle minutes - after which peers
// resolved us to 127.0.0.1, dev 2026-10-05). A changed address goes to the
// server at once.
std::atomic<bool> g_keeprun{false};
std::condition_variable g_keepcv;
std::mutex g_keepmu;

void keepalive_main() {
    host_thread_set_name("bb-np-keep");
    std::string last;
    {
        std::lock_guard<std::mutex> lk(g_mapped.mu);
        last = g_mapped.addr + ":" + std::to_string(g_mapped.port);
    }
    while (g_keeprun.load()) {
        {
            std::unique_lock<std::mutex> lk(g_keepmu);
            g_keepcv.wait_for(lk, std::chrono::seconds(15), [] { return !g_keeprun.load(); });
        }
        if (!g_keeprun.load()) break;
        mapped_refresh(true);
        std::string now;
        {
            std::lock_guard<std::mutex> lk(g_mapped.mu);
            now = g_mapped.addr + ":" + std::to_string(g_mapped.port);
        }
        if (now != last && now != ":0") {
            host_log("stun: our address is now %s (was %s); telling the server", now.c_str(), last.c_str());
            server_signaling_update();
            last = now;
        }
    }
}

}  // namespace

void room_set(const std::string& session_id, int member_id) {
    std::lock_guard<std::mutex> lk(g_roommu);
    g_room_sid = session_id;
    g_room_mid = member_id;
}
void room_clear() { room_set("", 0); }
int stun_rtt_us() { return g_stun_rtt_us.load(); }

void poller_start(std::function<void(const json::Value& event)> handler) {
    std::lock_guard<std::mutex> lk(g_pollmu);
    g_handler = std::move(handler);
    if (g_pollrun.load()) return;
    g_pollrun.store(true);
    g_cursor = 0;
    g_pollthread = std::thread(poller_main);
    g_pollthread.detach();
    if (!g_keeprun.exchange(true)) {
        std::thread t(keepalive_main);
        t.detach();
    }
}

void poller_stop() {
    {
        std::lock_guard<std::mutex> lk(g_keepmu);
        g_keeprun.store(false);
        g_keepcv.notify_all();
    }
    {
        std::lock_guard<std::mutex> lk(g_pollmu);
        if (!g_pollrun.load()) return;
        g_pollrun.store(false);
        g_pollcv.notify_all();
    }
}

}  // namespace net
