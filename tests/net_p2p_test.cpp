// Datagrams between two of the game's P2P sockets over the instance's UDP
// port, through hle/net.cpp as the game calls it: directly, and through the
// private server's relay as this test plays it (a STUN answer carrying
// BBHOST-RELAY turns the relay on; the relay's framing both ways).
//
// Bytes that are not the game's chunks go whole, up to the largest UDP
// carries, and EMSGSIZE with the game's errno above that. The game's own
// datagrams, chunks of [kind 0-3][length u16 BE][payload] (everything a frame
// sends one peer: a busy frame is several KB of them), leave as datagrams of
// whole chunks of at most 1200 bytes that together are the datagram - a
// longer chunk alone - each behind its own P2P header, and the relay's when
// relayed; whole with BBHOST_NET_SPLIT_DATAGRAMS=0. Every piece arrives once.
// The port's report counts the datagrams sent as several and those over one
// 1500-byte frame, in and out.
#include "guest_abi.h"
#include "hle/net_p2p.h"
#include "hle/np.h"
#include "log.h"
#include "net/session.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

void hle_register_net();

namespace {
std::map<std::string, void*> g_fns;
int g_port = 0;
int g_fail = 0;
#define CHECK(c)                                                             \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                        \
        }                                                                    \
    } while (0)

template <typename F>
F fn(const char* name) {
    auto it = g_fns.find(name);
    if (it == g_fns.end()) {
        std::fprintf(stderr, "missing %s\n", name);
        std::exit(1);
    }
    return reinterpret_cast<F>(it->second);
}

struct SceSockaddrIn {
    std::uint8_t len;
    std::uint8_t family;
    std::uint16_t port;  // network order
    std::uint32_t addr;  // network order
    std::uint16_t vport; // network order
    std::uint8_t zero[6];
};
static_assert(sizeof(SceSockaddrIn) == 16, "SceNetSockaddrIn is 16 bytes");

SceSockaddrIn sce_addr(std::uint32_t addr_host, std::uint16_t port, std::uint16_t vport) {
    SceSockaddrIn a{};
    a.len = 16;
    a.family = 2;
    a.port = htons(port);
    a.addr = htonl(addr_host);
    a.vport = htons(vport);
    return a;
}

// A UDP socket bound to addr (host order) on a port nothing on the machine
// holds; its port in *port.
int bound_udp(std::uint32_t addr_host, std::uint16_t* port) {
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(addr_host);
    socklen_t sl = sizeof(sa);
    if (fd < 0 || ::bind(fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0 ||
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&sa), &sl) != 0) {
        std::fprintf(stderr, "no free UDP port\n");
        std::exit(1);
    }
    *port = ntohs(sa.sin_port);
    return fd;
}

// A datagram on fd within ms, or empty.
std::vector<std::uint8_t> recv_within(int fd, int ms, sockaddr_in* from = nullptr) {
    pollfd p{fd, POLLIN, 0};
    if (::poll(&p, 1, ms) <= 0) return {};
    std::vector<std::uint8_t> d(65536);
    sockaddr_in sa{};
    socklen_t sl = sizeof(sa);
    const ssize_t n = ::recvfrom(fd, d.data(), d.size(), 0, reinterpret_cast<sockaddr*>(&sa), &sl);
    if (n < 0) return {};
    d.resize(static_cast<std::size_t>(n));
    if (from) *from = sa;
    return d;
}

void put16(std::vector<std::uint8_t>& v, std::uint16_t x) {
    v.push_back(static_cast<std::uint8_t>(x >> 8));
    v.push_back(static_cast<std::uint8_t>(x));
}

void add_chunk(std::vector<std::uint8_t>& d, std::uint8_t kind, std::size_t body, std::uint8_t fill) {
    d.push_back(kind);
    put16(d, static_cast<std::uint16_t>(body));
    for (std::size_t i = 0; i < body; ++i) d.push_back(static_cast<std::uint8_t>(fill + i));
}

// True when [p, p+n) is whole chunks of kinds 0-3.
bool whole_chunks(const std::uint8_t* p, std::size_t n, int* count) {
    std::size_t at = 0;
    *count = 0;
    while (at < n) {
        if (n - at < 3 || p[at] > 3) return false;
        const std::size_t c = 3 + ((static_cast<std::size_t>(p[at + 1]) << 8) | p[at + 2]);
        if (c > n - at) return false;
        at += c;
        ++*count;
    }
    return true;
}

// The pieces' ends expected for d: whole chunks while they fit in 1200
// bytes, a longer chunk alone; one piece when d is 1200 bytes or less, is not
// whole chunks, or nothing is split.
std::vector<std::size_t> expected_ends(const std::vector<std::uint8_t>& d, bool split) {
    std::vector<std::size_t> ends;
    int count = 0;
    if (split && d.size() > 1200 && whole_chunks(d.data(), d.size(), &count)) {
        std::size_t at = 0, start = 0;
        while (at < d.size()) {
            const std::size_t c = 3 + ((static_cast<std::size_t>(d[at + 1]) << 8) | d[at + 2]);
            if (at > start && at + c - start > 1200) {
                ends.push_back(at);
                start = at;
            }
            at += c;
        }
    }
    ends.push_back(d.size());
    return ends;
}

// What the port's report at a socket's close says of a vport: its datagrams
// sent as several, and those over one frame in and out with the largest.
struct Report {
    bool found = false;
    unsigned long long split = 0, in_over = 0, out_over = 0;
    std::size_t frame = 0, in_largest = 0, out_largest = 0;
};
Report report_of(const std::string& log, unsigned vport) {
    Report r;
    const std::size_t line = log.find("at a socket's close");
    if (line == std::string::npos) return r;
    const std::string key = " vport " + std::to_string(vport) + " (sock ";
    const std::size_t at = log.find(key, line);
    if (at == std::string::npos) return r;
    const std::size_t sp = log.find(", split ", at);
    if (sp == std::string::npos) return r;
    r.found = std::sscanf(log.c_str() + sp, ", split %llu, over %zu B in %llu (largest %zu) out %llu (largest %zu);", &r.split,
                          &r.frame, &r.in_over, &r.in_largest, &r.out_over, &r.out_largest) == 6;
    return r;
}

// The relay's ports: the one the test's relay gives this instance, and the
// one of the peer the game sends to (another player behind the relay).
constexpr std::uint16_t kOwnRelayPort = 4123, kPeerRelayPort = 4124;
constexpr std::uint8_t kToken[8] = {9, 8, 7, 6, 5, 4, 3, 2};

// The relay's answer to a Binding Request with the HELLO: MAPPED-ADDRESS
// our relay port, BBHOST-RELAY the token, that port and where the request
// came from (net/stun.h).
std::vector<std::uint8_t> relay_answer(const std::uint8_t* req, std::size_t n, std::uint32_t relay_addr_host,
                                       const sockaddr_in& from) {
    std::vector<std::uint8_t> out;
    if (n < 20 || req[0] != 0x00 || req[1] != 0x01) return out;
    std::vector<std::uint8_t> body = {0x00, 0x01, 0, 8, 0, 1};
    put16(body, kOwnRelayPort);
    for (int s = 24; s >= 0; s -= 8) body.push_back(static_cast<std::uint8_t>(relay_addr_host >> s));
    const std::uint8_t rel[] = {0x81, 0x01, 0, 20, 'b', 'b', 'r', '1'};
    body.insert(body.end(), rel, rel + sizeof(rel));
    body.insert(body.end(), kToken, kToken + 8);
    put16(body, kOwnRelayPort);
    const std::uint32_t seen = ntohl(from.sin_addr.s_addr);
    for (int s = 24; s >= 0; s -= 8) body.push_back(static_cast<std::uint8_t>(seen >> s));
    put16(body, ntohs(from.sin_port));
    put16(out, 0x0101);
    put16(out, static_cast<std::uint16_t>(body.size()));
    out.insert(out.end(), req + 4, req + 20);
    out.insert(out.end(), body.begin(), body.end());
    return out;
}
}  // namespace

// What hle/net.cpp needs from the rest of the host: the HLE table, the
// session layer's P2P port and dispatcher, and NpSignaling's control
// datagrams (none come here).
void register_hle_fn(const char* name, void* f) { g_fns[name] = f; }
namespace net {
int p2p_port() { return g_port; }
void set_p2p_port_bound(int port) { g_port = port; }
void dispatch_after(int, Prio, std::function<void()>) {}
}  // namespace net
void hle_np_signaling_control(const std::uint8_t*, std::size_t, std::uint32_t, std::uint16_t) {}

int main() {
    const char* sp = std::getenv("BBHOST_NET_SPLIT_DATAGRAMS");
    const bool split = !(sp && sp[0] == '0');
    host_log_decide(true);
    {
        std::uint16_t p = 0;
        const int fd = bound_udp(INADDR_LOOPBACK, &p);
        ::close(fd);
        g_port = p;
    }
    hle_register_net();
    using SocketFn = int(GUEST_ABI*)(const char*, int, int, int);
    using BindFn = int(GUEST_ABI*)(int, const void*, int);
    using SendtoFn = int(GUEST_ABI*)(int, const void*, std::uint64_t, int, const void*, int);
    using RecvfromFn = int(GUEST_ABI*)(int, void*, std::uint64_t, int, void*, unsigned*);
    using ErrnoFn = int*(GUEST_ABI*)();
    using CloseFn = int(GUEST_ABI*)(int);
    using InitFn = int(GUEST_ABI*)();
    auto net_init = fn<InitFn>("sceNetInit");
    auto net_socket = fn<SocketFn>("sceNetSocket");
    auto net_bind = fn<BindFn>("sceNetBind");
    auto net_sendto = fn<SendtoFn>("sceNetSendto");
    auto net_recvfrom = fn<RecvfromFn>("sceNetRecvfrom");
    auto net_errno = fn<ErrnoFn>("sceNetErrnoLoc");
    auto net_close = fn<CloseFn>("sceNetSocketClose");
    constexpr int kDgramP2p = 6, kDontWait = 0x80, kFreebsdEmsgsize = 40;
    constexpr std::size_t kHdr = 4, kRelayHdr = 12, kDeliveryHdr = 4, kFrame = 1472;

    net_init();
    // The game's two vports on the instance's one UDP port: a datagram from
    // one to the other leaves the port and comes back to its reader.
    const int a = net_socket("p2p a", 2, kDgramP2p, 0);
    const int b = net_socket("p2p b", 2, kDgramP2p, 0);
    const SceSockaddrIn bind_a = sce_addr(0, 3658, 30), bind_b = sce_addr(0, 3658, 31);
    CHECK(net_bind(a, &bind_a, 16) == 0);
    CHECK(net_bind(b, &bind_b, 16) == 0);
    const SceSockaddrIn to_b = sce_addr(INADDR_LOOPBACK, static_cast<std::uint16_t>(g_port), 31);

    // What the port's report should count, kept as the datagrams go.
    unsigned long long want_split = 0, want_out_over = 0, want_in_over = 0;
    std::size_t want_out_largest = 0, want_in_largest = 0;
    auto out_wire = [&](std::size_t n) {
        if (n > kFrame) ++want_out_over;
        if (n > want_out_largest) want_out_largest = n;
    };
    auto in_wire = [&](std::size_t n) {
        if (n > kFrame) ++want_in_over;
        if (n > want_in_largest) want_in_largest = n;
    };
    // A socket's next datagram within 3 s (-1 when none came), and from where.
    auto recv_on = [&](int s, std::vector<std::uint8_t>& in, SceSockaddrIn* from) {
        unsigned from_len = 16;
        for (int i = 0; i < 300; ++i) {
            const int got = net_recvfrom(s, in.data(), in.size(), kDontWait, from, &from_len);
            if (got >= 0) return got;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return -1;
    };
    auto recv_b = [&](std::vector<std::uint8_t>& in, SceSockaddrIn* from) { return recv_on(b, in, from); };
    // Nothing more for a socket: no piece came twice.
    auto quiet = [&](int s) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::uint8_t sink[16];
        return net_recvfrom(s, sink, sizeof(sink), kDontWait, nullptr, nullptr) < 0;
    };
    auto b_quiet = [&] { return quiet(b); };

    // Bytes that are not chunks go as they are, up to what UDP carries with
    // the 4-byte P2P header: the reader takes the largest.
    const std::size_t largest = 65507 - kHdr;
    for (const std::size_t n : {std::size_t{1}, std::size_t{64}, std::size_t{7 * 276}, std::size_t{8 * 276},
                                std::size_t{15 * 276}, largest}) {
        std::vector<std::uint8_t> out(n);
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<std::uint8_t>(i * 131 + n);
        out[0] = 0xff;  // not a chunk
        CHECK(net_sendto(a, out.data(), n, 0, &to_b, 16) == static_cast<int>(n));
        out_wire(kHdr + n);
        std::vector<std::uint8_t> in(n + 64, 0);
        SceSockaddrIn from{};
        const int got = recv_b(in, &from);
        CHECK(got == static_cast<int>(n) && std::memcmp(in.data(), out.data(), n) == 0);
        CHECK(ntohs(from.vport) == 30);
        if (got == static_cast<int>(n)) in_wire(kHdr + n);
        if (got != static_cast<int>(n)) std::fprintf(stderr, "  %zu bytes: received %d\n", n, got);
        CHECK(b_quiet());
    }

    // The game's datagrams. Pairs of a 254-byte chunk and a 22-byte one (7
    // pairs fit in 2 KB, 8 do not, 15 come to about 4 KB), with the pieces
    // they leave as: four pairs (1104 bytes) to a piece.
    struct Game {
        std::vector<std::uint8_t> d;
        std::vector<std::size_t> pieces;  // the pieces' sizes when split
    };
    std::vector<Game> games;
    for (const int pairs : {7, 8, 15}) {
        Game g;
        for (int i = 0; i < pairs; ++i) {
            add_chunk(g.d, 1, 251, static_cast<std::uint8_t>(i));
            add_chunk(g.d, 1, 19, static_cast<std::uint8_t>(0x80 + i));
        }
        for (int left = pairs; left > 0; left -= 4) g.pieces.push_back(static_cast<std::size_t>(left < 4 ? left : 4) * 276);
        games.push_back(g);
    }
    {
        // A full segment (1238 bytes and its header) is longer than a piece:
        // it goes alone, between short chunks of every kind.
        Game g;
        add_chunk(g.d, 0, 85, 1);
        add_chunk(g.d, 1, 1238, 2);
        add_chunk(g.d, 2, 40, 3);
        add_chunk(g.d, 3, 1238, 4);
        add_chunk(g.d, 1, 6, 5);
        g.pieces = {88, 1241, 43, 1241, 9};
        games.push_back(g);
    }
    {
        // 1200 bytes of chunks go whole; 1201 bytes as two pieces.
        Game g;
        add_chunk(g.d, 1, 597, 6);
        add_chunk(g.d, 0, 597, 7);
        g.pieces = {1200};
        games.push_back(g);
        Game h;
        add_chunk(h.d, 1, 597, 6);
        add_chunk(h.d, 0, 598, 7);
        h.pieces = {600, 601};
        games.push_back(h);
    }
    {
        // A piece filled to exactly 1200 bytes, then the rest.
        Game g;
        add_chunk(g.d, 1, 397, 8);
        add_chunk(g.d, 1, 397, 9);
        add_chunk(g.d, 1, 397, 10);
        add_chunk(g.d, 2, 10, 11);
        g.pieces = {1200, 13};
        games.push_back(g);
    }
    // Not whole chunks, though they start as such: they go whole.
    std::vector<std::vector<std::uint8_t>> not_chunks;
    {
        std::vector<std::uint8_t> d;
        for (int i = 0; i < 6; ++i) add_chunk(d, 1, 297, static_cast<std::uint8_t>(i));
        d.resize(d.size() - 1);  // the last chunk cut
        not_chunks.push_back(d);
        d.clear();
        for (int i = 0; i < 6; ++i) add_chunk(d, i == 3 ? 4 : 1, 297, static_cast<std::uint8_t>(i));  // a kind past 3
        not_chunks.push_back(d);
    }
    for (const auto& d : not_chunks) {
        Game g;
        g.d = d;
        g.pieces = {d.size()};
        games.push_back(g);
    }

    for (const Game& g : games) {
        const std::vector<std::size_t> ends = expected_ends(g.d, split);
        // The sizes the comment above gives, for the game's datagrams.
        if (split && ends.size() > 1) {
            std::vector<std::size_t> sizes;
            for (std::size_t i = 0, begin = 0; i < ends.size(); begin = ends[i++]) sizes.push_back(ends[i] - begin);
            CHECK(sizes == g.pieces);
        }
        if (ends.size() > 1) ++want_split;
        for (std::size_t i = 0, begin = 0; i < ends.size(); begin = ends[i++]) out_wire(kHdr + ends[i] - begin);
        CHECK(net_sendto(a, g.d.data(), g.d.size(), 0, &to_b, 16) == static_cast<int>(g.d.size()));
        std::vector<std::uint8_t> joined;
        std::size_t got_n = 0;
        while (joined.size() < g.d.size()) {
            std::vector<std::uint8_t> in(65536);
            SceSockaddrIn from{};
            const int got = recv_b(in, &from);
            if (got < 0) break;
            // Each datagram ends where a piece does, of whole chunks when the
            // game's bytes are.
            const std::size_t at = joined.size() + static_cast<std::size_t>(got);
            CHECK(got_n < ends.size() && at == ends[got_n]);
            CHECK(ntohs(from.vport) == 30);
            int count = 0;
            if (ends.size() > 1) CHECK(whole_chunks(in.data(), static_cast<std::size_t>(got), &count) &&
                                       (got <= 1200 || count == 1));
            in_wire(kHdr + static_cast<std::size_t>(got));
            ++got_n;
            joined.insert(joined.end(), in.begin(), in.begin() + got);
        }
        CHECK(got_n == ends.size());
        CHECK(joined == g.d);
        CHECK(b_quiet());
        if (got_n != ends.size() || joined != g.d)
            std::fprintf(stderr, "  chunks of %zu bytes: %zu datagrams (%zu expected), %zu bytes back\n", g.d.size(), got_n,
                         ends.size(), joined.size());
    }

    // One byte more than UDP carries: EMSGSIZE, and the game's errno says so.
    std::vector<std::uint8_t> over(largest + 1, 0x5a);
    *net_errno() = 0;
    CHECK(net_sendto(a, over.data(), over.size(), 0, &to_b, 16) < 0);
    CHECK(*net_errno() == kFreebsdEmsgsize);
    CHECK(b_quiet());  // and nothing went out

    // Vports past 255 take the 6-byte P2P header, [0xff][0x83][source u16]
    // [destination u16]: every piece carries it, with its bytes unchanged.
    {
        const int c = net_socket("p2p c", 2, kDgramP2p, 0);
        const int e = net_socket("p2p e", 2, kDgramP2p, 0);
        const SceSockaddrIn bind_c = sce_addr(0, 3658, 300), bind_e = sce_addr(0, 3658, 301);
        CHECK(net_bind(c, &bind_c, 16) == 0);
        CHECK(net_bind(e, &bind_e, 16) == 0);
        const SceSockaddrIn to_e = sce_addr(INADDR_LOOPBACK, static_cast<std::uint16_t>(g_port), 301);
        const std::vector<std::uint8_t>& d = games[2].d;  // 15 pairs: four pieces when split
        const std::vector<std::size_t> ends = expected_ends(d, split);
        CHECK(net_sendto(c, d.data(), d.size(), 0, &to_e, 16) == static_cast<int>(d.size()));
        std::vector<std::uint8_t> joined;
        std::size_t got_n = 0;
        while (joined.size() < d.size()) {
            std::vector<std::uint8_t> in(65536);
            SceSockaddrIn from{};
            const int got = recv_on(e, in, &from);
            if (got < 0) break;
            CHECK(got_n < ends.size() && joined.size() + static_cast<std::size_t>(got) == ends[got_n]);
            CHECK(ntohs(from.vport) == 300);
            ++got_n;
            joined.insert(joined.end(), in.begin(), in.begin() + got);
        }
        CHECK(got_n == ends.size() && joined == d);
        CHECK(quiet(e));
        CHECK(net_close(c) == 0);
        CHECK(net_close(e) == 0);
    }

    // Through the relay: the test answers the instance's STUN request as the
    // private server does (on 127.0.0.2, as the server's relay address), and
    // relays what comes framed for its peer's relay port back to the port.
    std::uint16_t stun_port = 0;
    const int relay = bound_udp(0x7f000002u, &stun_port);
    std::thread answer([&] {
        sockaddr_in from{};
        const std::vector<std::uint8_t> req = recv_within(relay, 3000, &from);
        const std::vector<std::uint8_t> r = relay_answer(req.data(), req.size(), 0x7f000002u, from);
        if (!r.empty())
            ::sendto(relay, r.data(), r.size(), 0, reinterpret_cast<const sockaddr*>(&from), sizeof(from));
    });
    std::uint32_t mapped = 0;
    std::uint16_t mapped_port = 0;
    net::stun::Relay rl;
    const bool on = hle_net_p2p_stun("127.0.0.2", stun_port, 3000, &mapped, &mapped_port, &rl);
    answer.join();
    CHECK(on && rl.present && rl.vport == kOwnRelayPort);
    std::uint32_t server = 0;
    std::uint16_t own = 0;
    CHECK(hle_net_p2p_relay(&server, &own) && server == htonl(0x7f000002u) && own == kOwnRelayPort);
    const SceSockaddrIn to_peer = sce_addr(0x7f000002u, kPeerRelayPort, 31);
    sockaddr_in port_sa{};
    port_sa.sin_family = AF_INET;
    port_sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    port_sa.sin_port = htons(static_cast<std::uint16_t>(g_port));
    // Each frame: [0xfb]['R'][token][the peer's port][P2P header][a piece].
    // The relay takes its own header off and delivers the rest framed
    // [0xfb]['r'][our relay port].
    auto relay_one = [&](const std::vector<std::uint8_t>& d, const std::vector<std::size_t>& ends) {
        std::size_t frames = 0, begin = 0;
        for (std::vector<std::uint8_t> f; !(f = recv_within(relay, frames < ends.size() ? 3000 : 50)).empty();) {
            const bool framed = f.size() >= kRelayHdr + kHdr && f[0] == 0xfb && f[1] == 'R' &&
                                std::memcmp(f.data() + 2, kToken, 8) == 0 && f[10] == (kPeerRelayPort >> 8) &&
                                f[11] == (kPeerRelayPort & 0xff) && f[12] == 0xff && f[13] == 0xc3 && f[14] == 30 &&
                                f[15] == 31;
            CHECK(framed);
            CHECK(frames < ends.size());
            if (!framed || frames >= ends.size()) break;
            const std::size_t piece = f.size() - kRelayHdr - kHdr;
            CHECK(begin + piece == ends[frames] && std::memcmp(f.data() + kRelayHdr + kHdr, d.data() + begin, piece) == 0);
            out_wire(f.size());
            begin += piece;
            ++frames;
            std::vector<std::uint8_t> del = {0xfb, 'r'};
            put16(del, kOwnRelayPort);
            del.insert(del.end(), f.begin() + kRelayHdr, f.end());
            ::sendto(relay, del.data(), del.size(), 0, reinterpret_cast<const sockaddr*>(&port_sa), sizeof(port_sa));
        }
        CHECK(frames == ends.size());
        std::vector<std::uint8_t> joined;
        std::size_t got_n = 0;
        while (got_n < frames) {
            std::vector<std::uint8_t> in(65536);
            SceSockaddrIn from{};
            const int got = recv_b(in, &from);
            if (got < 0) break;
            // From the server's address and the sender's relay port.
            CHECK(from.addr == htonl(0x7f000002u) && ntohs(from.port) == kOwnRelayPort && ntohs(from.vport) == 30);
            in_wire(kDeliveryHdr + kHdr + static_cast<std::size_t>(got));
            ++got_n;
            joined.insert(joined.end(), in.begin(), in.begin() + got);
        }
        CHECK(got_n == ends.size());
        CHECK(joined == d);
        CHECK(b_quiet());
        if (got_n != ends.size() || joined != d)
            std::fprintf(stderr, "  relayed %zu bytes: %zu datagrams (%zu expected), %zu bytes back\n", d.size(), got_n,
                         ends.size(), joined.size());
    };
    for (const Game& g : games) {
        const std::vector<std::size_t> ends = expected_ends(g.d, split);
        if (ends.size() > 1) ++want_split;
        CHECK(net_sendto(a, g.d.data(), g.d.size(), 0, &to_peer, 16) == static_cast<int>(g.d.size()));
        relay_one(g.d, ends);
    }
    // The largest that goes through the relay, its 12-byte header in, and
    // one byte more.
    std::vector<std::uint8_t> most(65507 - kRelayHdr - kHdr, 0x3c);
    CHECK(net_sendto(a, most.data(), most.size(), 0, &to_peer, 16) == static_cast<int>(most.size()));
    relay_one(most, {most.size()});
    most.push_back(0x3c);
    *net_errno() = 0;
    CHECK(net_sendto(a, most.data(), most.size(), 0, &to_peer, 16) < 0);
    CHECK(*net_errno() == kFreebsdEmsgsize);
    CHECK(recv_within(relay, 100).empty());  // nothing went out

    // The port's report at a's close: datagrams sent as several, and those
    // over one frame out (a) and in (b) with the largest.
    std::fflush(stderr);
    int pipe_fds[2];
    CHECK(::pipe(pipe_fds) == 0);
    const int saved_err = ::dup(2);
    ::dup2(pipe_fds[1], 2);
    CHECK(net_close(a) == 0);
    ::dup2(saved_err, 2);
    ::close(saved_err);
    ::close(pipe_fds[1]);
    std::string log;
    char chunk[4096];
    for (ssize_t n; (n = ::read(pipe_fds[0], chunk, sizeof(chunk))) > 0;) log.append(chunk, static_cast<std::size_t>(n));
    ::close(pipe_fds[0]);
    const Report ra = report_of(log, 30), rb = report_of(log, 31);
    CHECK(ra.found && rb.found);
    CHECK(ra.frame == kFrame);
    CHECK(ra.split == want_split);
    CHECK(ra.out_over == want_out_over && ra.out_largest == want_out_largest);
    CHECK(rb.in_over == want_in_over && rb.in_largest == want_in_largest);
    if (!(ra.found && rb.found && ra.split == want_split && ra.out_over == want_out_over &&
          ra.out_largest == want_out_largest && rb.in_over == want_in_over && rb.in_largest == want_in_largest))
        std::fprintf(stderr,
                     "  report: split %llu (%llu expected), out over %llu (%llu) largest %zu (%zu), in over %llu (%llu) "
                     "largest %zu (%zu)\n  %s",
                     ra.split, want_split, ra.out_over, want_out_over, ra.out_largest, want_out_largest, rb.in_over,
                     want_in_over, rb.in_largest, want_in_largest, log.c_str());

    CHECK(net_close(b) == 0);  // the port's last socket: its reader stops
    ::close(relay);
    if (g_fail) {
        std::fprintf(stderr, "net_p2p_test: %d failure(s) (split %s)\n", g_fail, split ? "on" : "off");
        return 1;
    }
    std::printf("net_p2p_test: ok (split %s; %llu of the game's datagrams sent as several, largest payload %zu)\n",
                split ? "on" : "off", want_split, largest);
    return 0;
}
