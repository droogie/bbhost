// Retained NpSignaling connections must not acquire a later room's slot owner.
// Real signaling, Matching2 and peer-table code; HTTP, UDP and guest callbacks
// are boundary stubs. No live server or game is needed.
#include "hle/np_signaling.cpp"
#include "core/config.h"
#include "core/portable.h"
#include "engine/summon_invite.h"
#include "net/http.h"
#include "np_signaling_fixture.h"
#include <array>
#include <cstdio>
#include <cstdlib>

namespace {
int failures = 0;
int checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); } } while (0)
struct Packet { std::uint32_t addr; std::uint16_t port; std::vector<std::uint8_t> bytes; };
std::vector<Packet> packets;
std::vector<std::array<std::int64_t, 6>> callbacks;
HostConfig test_config;
int callback_marker;

std::uint16_t be16(std::uint16_t n) { return static_cast<std::uint16_t>((n << 8) | (n >> 8)); }
net::Peer peer(const char* name, unsigned member, unsigned addr, unsigned port) {
    net::Peer p;
    p.online_id = name;
    p.member_id = static_cast<std::uint16_t>(member);
    p.addr = addr;
    p.port = static_cast<std::uint16_t>(port);
    return p;
}

void receive(const net::Peer& p, Kind kind, const char* to = "Local", Reason reason = kDeactivate) {
    std::uint8_t data[kPacketLen] = {};
    std::memcpy(data, kSignalMagic, sizeof(kSignalMagic));
    data[4] = kind;
    data[5] = reason;
    std::memcpy(data + kFromAt, p.online_id.data(), std::min(p.online_id.size(), kIdLen));
    std::memcpy(data + kToAt, to, std::min(std::strlen(to), kIdLen));
    hle_np_signaling_control(data, sizeof(data), p.addr, be16(p.port));
}

unsigned activate(unsigned ctx, const net::Peer& p) {
    unsigned conn = 0;
    CHECK(sig_activate(ctx, p.online_id.c_str(), &conn) == 0);
    signaling_test::dispatch_ready();
    receive(p, kActivationAck);
    receive(p, kAccept);
    receive(p, kCheckAck);
    int status = 0;
    CHECK(sig_status(ctx, conn, &status, nullptr, nullptr) == 0 && status == 2);
    return conn;
}

void reset() {
    sig_term();
    signaling_test::reset_dispatch();
    net::peers_clear();
    packets.clear();
    callbacks.clear();
}
}

const HostConfig& config() { return test_config; }
void host_thread_set_name(const char*) {}
void* hle_thread_enter_guest() { return nullptr; }
void hle_thread_leave_guest(void*) {}
extern "C" GUEST_ABI void hle_fs_host() {}
extern "C" GUEST_ABI std::int64_t hle_call_guest6(void*, std::int64_t a, std::int64_t b, std::int64_t c,
                                                   std::int64_t d, std::int64_t e, std::int64_t f) {
    callbacks.push_back({a, b, c, d, e, f});
    return 0;
}
void plugins_enter_world(const std::string&) {}
std::string plugins_ruleset() { return {}; }
void register_hle_fn(const char*, void*) {}
bool summon_invite_deliver(const SummonInvite&) { std::abort(); }
void summon_invite_host_extra(json::Value&) { std::abort(); }
void summon_invite_set_signaling_gate() { std::abort(); }
bool hle_net_p2p_stun(const char*, std::uint16_t, int, std::uint32_t*, std::uint16_t*, net::stun::Relay*) {
    std::abort();
}
void hle_net_p2p_punch(const char*, std::uint32_t, std::uint16_t, std::uint32_t, std::uint16_t) {}
void hle_net_p2p_signal(std::uint32_t addr, std::uint16_t port, const void* data, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    packets.push_back({addr, port, {p, p + n}});
}
namespace net {
std::string account_name() { return {}; }
bool account_refused() { return false; }
bool account_logged_in() { return true; }
std::string np_server_base() { std::abort(); }
bool np_get(const std::string&, json::Value&, std::string&, int) { std::abort(); }
bool np_post(const std::string&, const json::Value&, json::Value&, std::string&, const HttpCall&, long*) {
    std::abort();
}
bool np_post(const std::string& path, const json::Value&, json::Value& out, std::string&, int) {
    CHECK(path == "/np/signaling/resolve");
    out = json::Value::make_object();
    out.set("ResKind", 0);
    out.set("Addr", "127.0.0.2");
    out.set("Port", 9307);
    return true;
}
std::string str_of(const json::Value& v, const char* key, const std::string& dflt) {
    const auto* p = v.find(key);
    return p && p->type == json::Value::Type::String ? p->string : dflt;
}
long long int_of(const json::Value& v, const char* key, long long dflt) {
    const auto* p = v.find(key);
    return p && p->type == json::Value::Type::Number ? static_cast<long long>(p->number) : dflt;
}
}

int main() {
    test_config.online_id = "Local";
    g_host_log_state = kHostLogOff;
    reset();
    unsigned ctx = 0;
    CHECK(sig_create_ctx(nullptr, &callback_marker, nullptr, &ctx) == 0);
    const auto old_peer = peer("Jobe", 2, 0x0200007f, 9307);
    const auto new_peer = peer("weedeey", 2, 0x0300007f, 9308);

    // Activation before membership uses resolution and a provisional peer.
    const unsigned old_conn = activate(ctx, old_peer);
    net::Peer stored;
    CHECK(net::peers_find_online(old_peer.online_id, &stored) && stored.member_id >= 0xff00);
    const unsigned old_transport = hle_np_signaling_live_transport(old_peer.online_id);
    signaling_test::enter_room(3141, old_peer, &callback_marker);
    signaling_test::dispatch_ready(1100);
    CHECK(signaling_test::member_status(3141, 2) == 2);
    signaling_test::leave_room();
    CHECK(hle_np_signaling_transport_status(old_transport) == 2);

    signaling_test::enter_room(3173, new_peer, &callback_marker);
    const unsigned new_conn = activate(ctx, new_peer);
    signaling_test::dispatch_ready(1100);
    CHECK(signaling_test::member_status(3173, 2) == 2);
    CHECK(signaling_test::member_status(3141, 2) == 0);

    // Status and a Close must keep Jobe's endpoint, even after member 2 changes.
    std::uint32_t addr = 0;
    std::uint16_t port = 0;
    CHECK(sig_status(ctx, old_conn, nullptr, &addr, &port) == 0);
    CHECK(addr == old_peer.addr && port == be16(old_peer.port));
    packets.clear();
    callbacks.clear();
    CHECK(sig_deactivate(ctx, old_conn) == 0);
    CHECK(packets.size() == 1 && packets[0].bytes[4] == kClose);
    if (packets.size() == 1) {
        CHECK(packets[0].addr == old_peer.addr && packets[0].port == old_peer.port);
        CHECK(std::memcmp(packets[0].bytes.data() + kToAt, "Jobe", 4) == 0);
    }
    signaling_test::dispatch_ready();
    CHECK(net::peers_get(2, &stored) && stored.conn_id == new_conn && stored.sig_status == 2);
    CHECK(signaling_test::member_status(3173, 2) == 2);
    bool old_room_dead = false;
    for (const auto& c : callbacks) {
        if (c[3] == 0x5101) {
            CHECK(c[1] == 3141 && c[2] == 2);
            old_room_dead = true;
        }
    }
    CHECK(old_room_dead);

    // A late status write from an older connection to the SAME player cannot
    // overwrite a new connection. Metadata refresh keeps only the same player.
    net::peers_bind_conn(new_peer.online_id, new_conn + 1, 2);
    net::peers_set_conn(new_peer.online_id, new_conn, 0);
    CHECK(net::peers_get(2, &stored) && stored.conn_id == new_conn + 1 && stored.sig_status == 2);
    net::peers_upsert(new_peer);
    CHECK(net::peers_get(2, &stored) && stored.conn_id == new_conn + 1 && stored.sig_status == 2);
    net::peers_upsert(old_peer);
    CHECK(net::peers_get(2, &stored) && stored.conn_id == 0 && stored.sig_status == 0);
    net::peers_upsert(new_peer);
    net::peers_bind_conn(new_peer.online_id, new_conn, 2);

    // Wrong-recipient control packets must change neither state nor callbacks.
    callbacks.clear();
    packets.clear();
    receive(new_peer, kClose, "SomeoneElse", kTerminate);
    CHECK(callbacks.empty() && packets.empty());
    CHECK(signaling_test::member_status(3173, 2) == 2);
    receive(new_peer, kClose);
    CHECK(callbacks.size() == 1 && callbacks[0][1] == new_conn && callbacks[0][2] == kEvPeerDeactivated);
    CHECK(signaling_test::member_status(3173, 2) == 2);
    const auto stranger = peer("Stranger", 3, 0x0400007f, 9309);
    receive(stranger, kActivationRequest, "SomeoneElse");
    CHECK(find_transport_locked(stranger.online_id) == 0 && packets.empty());
    receive(stranger, kActivationRequest);
    CHECK(find_transport_locked(stranger.online_id) != 0);
    CHECK(!packets.empty() && packets.back().bytes[4] == kActivationAck);
    receive(stranger, kOffer);
    receive(stranger, kCheck);
    receive(stranger, kCheckAck);
    CHECK(hle_np_signaling_transport_status(find_transport_locked(stranger.online_id)) == 2);

    // An address refresh for the actual member updates its own transport.
    auto refreshed = new_peer;
    refreshed.addr = 0x0500007f;
    refreshed.port = 9310;
    receive(refreshed, kOffer);
    CHECK(net::peers_get(2, &stored) && stored.addr == new_peer.addr);
    CHECK(signaling_test::member_status(3173, 2, &addr, &port) == 2);
    CHECK(addr == refreshed.addr && port == be16(refreshed.port));
    net::peers_upsert(refreshed);
    hle_np_signaling_peer_known(refreshed.member_id);
    CHECK(sig_status(ctx, new_conn, nullptr, &addr, &port) == 0);
    CHECK(addr == refreshed.addr && port == be16(refreshed.port));

    // Queued establishment from an earlier room cannot satisfy a new room,
    // even if it keeps the very same live transport. The new room's timer can.
    const unsigned new_transport = hle_np_signaling_live_transport(new_peer.online_id);
    const auto old_epoch = hle_np_matching2_room_epoch();
    signaling_test::enter_room(4000, new_peer, &callback_marker);
    callbacks.clear();
    hle_np_matching2_transport_up(new_transport, old_epoch);
    hle_np_matching2_transport_dead(new_transport, old_epoch, kM2ErrTimeout);
    CHECK(callbacks.empty());
    CHECK(signaling_test::member_status(4000, 2) == 1);
    signaling_test::dispatch_ready(1100);
    CHECK(signaling_test::member_status(4000, 2) == 2);

    // Same-room slot reuse also rejects a delayed callback from the old link.
    const auto epoch = hle_np_matching2_room_epoch();
    signaling_test::bind_member(old_peer);
    callbacks.clear();
    hle_np_matching2_transport_dead(new_transport, epoch, kM2ErrTimeout);
    hle_np_matching2_transport_up(new_transport, epoch);
    CHECK(callbacks.empty());
    CHECK(signaling_test::member_status(4000, 2) == 1);

    // A correctly addressed close still reports loss for the actual peer.
    signaling_test::bind_member(new_peer);
    signaling_test::dispatch_ready(1100);
    callbacks.clear();
    receive(new_peer, kClose, "Local", kTerminate);
    signaling_test::dispatch_ready();
    CHECK(signaling_test::member_status(4000, 2) == 0);
    bool current_dead = false;
    for (const auto& c : callbacks) current_dead |= c[1] == 4000 && c[2] == 2 && c[3] == 0x5101;
    CHECK(current_dead);

    // Relay endpoints may share an address but not a port. They remain just
    // as independent as direct endpoints across a room-slot replacement.
    reset();
    CHECK(sig_create_ctx(nullptr, &callback_marker, nullptr, &ctx) == 0);
    const auto relay_a = peer("RelayA", 2, 0x0400007f, 20001);
    const auto relay_b = peer("RelayB", 2, relay_a.addr, 20002);
    net::peers_upsert(relay_a);
    const unsigned relay_conn = activate(ctx, relay_a);
    signaling_test::enter_room(5000, relay_b, &callback_marker);
    activate(ctx, relay_b);
    packets.clear();
    CHECK(sig_deactivate(ctx, relay_conn) == 0);
    CHECK(packets.size() == 1 && packets[0].addr == relay_a.addr && packets[0].port == relay_a.port);
    std::printf("np_signaling_identity_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
