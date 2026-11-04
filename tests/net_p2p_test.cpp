// Datagrams between two of the game's P2P sockets over the instance's UDP
// port, through hle/net.cpp as the game calls it. Bytes that are not the
// game's chunks go whole, up to the largest UDP carries, and EMSGSIZE with
// the game's errno above that. The game's own datagrams, chunks of
// [kind 0-3][length u16 BE][payload] (everything a frame sends one peer: a
// busy frame is several KB of them), arrive as datagrams
// of whole chunks of at most 1200 bytes that together are the datagram, or
// whole with BBHOST_NET_SPLIT_DATAGRAMS=0. With BBHOST_NET_LARGE_DATAGRAMS=0,
// the 2 KB limit of before.
#include "guest_abi.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// A port nothing on the machine holds, for the instance's P2P port.
int free_udp_port() {
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t sl = sizeof(sa);
    if (fd < 0 || ::bind(fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0 ||
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&sa), &sl) != 0) {
        std::fprintf(stderr, "no free UDP port\n");
        std::exit(1);
    }
    ::close(fd);
    return ntohs(sa.sin_port);
}
}  // namespace

// What hle/net.cpp needs from the rest of the host: the HLE table and the
// session layer's P2P port.
void register_hle_fn(const char* name, void* f) { g_fns[name] = f; }
namespace net {
int p2p_port() { return g_port; }
void set_p2p_port_bound(int port) { g_port = port; }
}  // namespace net

int main() {
    const char* e = std::getenv("BBHOST_NET_LARGE_DATAGRAMS");
    const bool large = !(e && e[0] == '0');
    const char* sp = std::getenv("BBHOST_NET_SPLIT_DATAGRAMS");
    const bool split = large && !(sp && sp[0] == '0');
    g_port = free_udp_port();
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

    net_init();
    // The game's two vports on the instance's one UDP port: a datagram from
    // one to the other leaves the port and comes back to its reader.
    const int a = net_socket("p2p a", 2, kDgramP2p, 0);
    const int b = net_socket("p2p b", 2, kDgramP2p, 0);
    const SceSockaddrIn bind_a = sce_addr(0, 3658, 30), bind_b = sce_addr(0, 3658, 31);
    CHECK(net_bind(a, &bind_a, 16) == 0);
    CHECK(net_bind(b, &bind_b, 16) == 0);
    const SceSockaddrIn to_b = sce_addr(INADDR_LOOPBACK, static_cast<std::uint16_t>(g_port), 31);

    // A payload and the datagram's 4-byte P2P header (one-byte vports).
    const std::size_t largest = (large ? 65507 : 2048) - 4;
    // A pair of the game's messages is 276 bytes of the datagram: 7 pairs
    // fit in 2 KB, 8 do not, and 15 come to about 4 KB.
    std::vector<std::size_t> sizes = {1, 64, 7 * 276};
    if (large) {
        sizes.push_back(8 * 276);
        sizes.push_back(15 * 276);
    }
    sizes.push_back(largest);
    for (const std::size_t n : sizes) {
        std::vector<std::uint8_t> out(n);
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<std::uint8_t>(i * 131 + n);
        out[0] = 0xff;  // not a chunk: these go as they are
        CHECK(net_sendto(a, out.data(), n, 0, &to_b, 16) == static_cast<int>(n));
        std::vector<std::uint8_t> in(n + 64, 0);
        SceSockaddrIn from{};
        unsigned from_len = 16;
        int got = -1;
        for (int i = 0; i < 300 && got < 0; ++i) {
            got = net_recvfrom(b, in.data(), in.size(), kDontWait, &from, &from_len);
            if (got < 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(got == static_cast<int>(n));
        CHECK(got == static_cast<int>(n) && std::memcmp(in.data(), out.data(), n) == 0);
        CHECK(ntohs(from.vport) == 30);
        if (got != static_cast<int>(n)) std::fprintf(stderr, "  %zu bytes: received %d\n", n, got);
    }

    // The game's datagrams: pairs of a 254-byte chunk and a 22-byte one, and
    // a chunk longer than a piece between short ones: it goes alone.
    auto add_chunk = [](std::vector<std::uint8_t>& d, std::uint8_t kind, std::size_t body, std::uint8_t fill) {
        d.push_back(kind);
        d.push_back(static_cast<std::uint8_t>(body >> 8));
        d.push_back(static_cast<std::uint8_t>(body));
        for (std::size_t i = 0; i < body; ++i) d.push_back(static_cast<std::uint8_t>(fill + i));
    };
    std::vector<std::vector<std::uint8_t>> games;
    for (const int pairs : {7, 8, 15}) {
        std::vector<std::uint8_t> d;
        for (int i = 0; i < pairs; ++i) {
            add_chunk(d, 1, 251, static_cast<std::uint8_t>(i));
            add_chunk(d, 1, 19, static_cast<std::uint8_t>(0x80 + i));
        }
        games.push_back(d);
    }
    {
        std::vector<std::uint8_t> d;
        add_chunk(d, 0, 85, 1);
        add_chunk(d, 1, 1238, 2);  // a full segment: 1232 bytes and its header
        add_chunk(d, 2, 40, 3);
        add_chunk(d, 3, 1238, 4);
        add_chunk(d, 1, 6, 5);
        games.push_back(d);
    }
    for (const auto& d : games) {
        // The pieces expected: whole chunks while they fit in 1200 bytes, a
        // longer chunk alone; one piece when nothing splits.
        std::vector<std::size_t> ends;
        if (split && d.size() > 1200) {
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
        const bool fits = large || d.size() + 4 <= 2048;
        const int r = net_sendto(a, d.data(), d.size(), 0, &to_b, 16);
        CHECK(fits ? r == static_cast<int>(d.size()) : r < 0);
        if (!fits) continue;
        std::vector<std::uint8_t> joined;
        std::size_t got_n = 0;
        for (int i = 0; i < 300 && joined.size() < d.size(); ++i) {
            std::vector<std::uint8_t> in(65536);
            SceSockaddrIn from{};
            unsigned from_len = 16;
            const int got = net_recvfrom(b, in.data(), in.size(), kDontWait, &from, &from_len);
            if (got < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            // Each datagram ends where a piece does.
            const std::size_t at = joined.size() + static_cast<std::size_t>(got);
            CHECK(got_n < ends.size() && at == ends[got_n]);
            ++got_n;
            joined.insert(joined.end(), in.begin(), in.begin() + got);
        }
        CHECK(got_n == ends.size());
        CHECK(joined == d);
        if (got_n != ends.size() || joined != d)
            std::fprintf(stderr, "  chunks of %zu bytes: %zu datagrams (%zu expected), %zu bytes back\n", d.size(), got_n,
                         ends.size(), joined.size());
    }

    // One byte more than UDP carries: EMSGSIZE, and the game's errno says so.
    std::vector<std::uint8_t> over(largest + 1, 0x5a);
    *net_errno() = 0;
    CHECK(net_sendto(a, over.data(), over.size(), 0, &to_b, 16) < 0);
    CHECK(*net_errno() == kFreebsdEmsgsize);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::uint8_t sink[16];
    CHECK(net_recvfrom(b, sink, sizeof(sink), kDontWait, nullptr, nullptr) < 0);  // and nothing went out

    CHECK(net_close(a) == 0);
    CHECK(net_close(b) == 0);  // the port's last socket: its reader stops
    if (g_fail) {
        std::fprintf(stderr, "net_p2p_test: %d failure(s) (large datagrams %s, split %s)\n", g_fail, large ? "on" : "off",
                     split ? "on" : "off");
        return 1;
    }
    std::printf("net_p2p_test: ok (large datagrams %s, split %s, largest payload %zu)\n", large ? "on" : "off",
                split ? "on" : "off", largest);
    return 0;
}
