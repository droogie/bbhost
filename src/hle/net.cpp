#include "hle/common.h"
#include "core/futex.h"
#include "hle/platform.h"
#include "hle/hle.h"
#include "hle/net_p2p.h"
#include "net/session.h"
#include "net/stun.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#if defined(__linux__)
#include <sys/epoll.h>
#include <sys/eventfd.h>
#endif
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <csetjmp>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr int kNetErrBase = static_cast<int>(0x80410100);

// The socket API behind a few names, so the layer reads the same on Linux
// and Windows (winsock: SOCKET handles, closesocket, WSA error numbers, no
// MSG_DONTWAIT - the sockets are non-blocking there and a blocking call
// waits with WSAPoll first).
#if defined(_WIN32)
using sockfd_t = SOCKET;
constexpr sockfd_t kBadSock = INVALID_SOCKET;
using socklen_type = int;
using ssize_type = int;
inline void sock_close(sockfd_t fd) { ::closesocket(fd); }
inline int sock_errno() {
    switch (::WSAGetLastError()) {
        case WSAEWOULDBLOCK: return EAGAIN;
        case WSAECONNREFUSED: return ECONNREFUSED;
        case WSAEADDRINUSE: return EADDRINUSE;
        case WSAEACCES: return EACCES;  // a port another program holds exclusively
        case WSAEMSGSIZE: return EMSGSIZE;
        case WSAENOTSOCK: return EBADF;
        case WSAEINTR: return EINTR;
        default: return EIO;
    }
}
inline void sock_set_nonblock(sockfd_t fd, bool on) {
    u_long v = on ? 1 : 0;
    ::ioctlsocket(fd, FIONBIO, &v);
}
inline int sock_poll(sockfd_t fd, short events, int ms) {
    WSAPOLLFD p{fd, events, 0};
    return ::WSAPoll(&p, 1, ms);
}
inline ssize_type sock_recvfrom(sockfd_t fd, void* buf, std::size_t len, bool dontwait, sockaddr_in* sa,
                                socklen_type* sl) {
    if (!dontwait && sock_poll(fd, POLLIN, -1) <= 0) return -1;
    return ::recvfrom(fd, static_cast<char*>(buf), static_cast<int>(len), 0, reinterpret_cast<sockaddr*>(sa), sl);
}
inline ssize_type sock_sendto(sockfd_t fd, const void* buf, std::size_t len, bool dontwait, const sockaddr_in* sa) {
    for (;;) {
        const int n = ::sendto(fd, static_cast<const char*>(buf), static_cast<int>(len), 0,
                               reinterpret_cast<const sockaddr*>(sa), sizeof(*sa));
        if (n >= 0 || dontwait || ::WSAGetLastError() != WSAEWOULDBLOCK) return n;
        sock_poll(fd, POLLOUT, 100);
    }
}
inline ssize_type sock_send(sockfd_t fd, const void* buf, std::size_t len, bool dontwait) {
    for (;;) {
        const int n = ::send(fd, static_cast<const char*>(buf), static_cast<int>(len), 0);
        if (n >= 0 || dontwait || ::WSAGetLastError() != WSAEWOULDBLOCK) return n;
        sock_poll(fd, POLLOUT, 100);
    }
}
inline sockfd_t sock_udp() {
    const sockfd_t fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd != INVALID_SOCKET) sock_set_nonblock(fd, true);  // blocking calls wait with WSAPoll
    return fd;
}
inline void sock_setopt_int(sockfd_t fd, int level, int name, int v) {
    ::setsockopt(fd, level, name, reinterpret_cast<const char*>(&v), sizeof(v));
}
#else
using sockfd_t = int;
constexpr sockfd_t kBadSock = -1;
using socklen_type = socklen_t;
using ssize_type = ssize_t;
inline void sock_close(sockfd_t fd) { ::close(fd); }
inline int sock_errno() { return errno == EWOULDBLOCK ? EAGAIN : errno; }
inline void sock_set_nonblock(sockfd_t fd, bool on) {
    const int fl = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, on ? (fl | O_NONBLOCK) : (fl & ~O_NONBLOCK));
}
inline int sock_poll(sockfd_t fd, short events, int ms) {
    pollfd p{fd, events, 0};
    return ::poll(&p, 1, ms);
}
inline ssize_type sock_recvfrom(sockfd_t fd, void* buf, std::size_t len, bool dontwait, sockaddr_in* sa,
                                socklen_type* sl) {
    return ::recvfrom(fd, buf, len, dontwait ? MSG_DONTWAIT : 0, reinterpret_cast<sockaddr*>(sa), sl);
}
inline ssize_type sock_sendto(sockfd_t fd, const void* buf, std::size_t len, bool dontwait, const sockaddr_in* sa) {
    return ::sendto(fd, buf, len, dontwait ? MSG_DONTWAIT : 0, reinterpret_cast<const sockaddr*>(sa), sizeof(*sa));
}
inline ssize_type sock_send(sockfd_t fd, const void* buf, std::size_t len, bool dontwait) {
    return ::send(fd, buf, len, dontwait ? MSG_DONTWAIT : 0);
}
inline sockfd_t sock_udp() { return ::socket(AF_INET, SOCK_DGRAM, 0); }
inline void sock_setopt_int(sockfd_t fd, int level, int name, int v) { ::setsockopt(fd, level, name, &v, sizeof(v)); }
#endif
inline bool sock_ok(sockfd_t fd) { return fd != kBadSock; }

// The game's epoll over our own wake word instead of the kernel's epoll and
// eventfds: Windows and macOS have neither, and on Linux BBHOST_NET_POLL=1 runs
// this path so it is proven on the same co-op runs. A P2P socket is readable
// when its inbox has a datagram; a UDP socket is polled with a zero timeout; a
// wait sleeps on one word every reader, control change and abort bumps
// (net_wake), ten milliseconds at a time for the UDP sockets.
//
// The word, not a condition variable: a waiter reads it before it looks at
// the sockets and sleeps only while it is unchanged, so a datagram that lands
// between the look and the sleep wakes it (the condition variable's notifier
// did not hold its mutex, and such a datagram waited out the slice). The
// sleep is core/futex.h's - WaitOnAddress, with the timeout honoured below a
// millisecond (KyoPS4x #215 found the empty epoll spinning a core on a
// timeout it ignored; winpthreads' condition variable returned at once on a
// sub-millisecond one, and the loop spun out the rest).
const bool g_net_poll = [] {
#if defined(_WIN32) || defined(__APPLE__)
    return true;
#else
    const char* e = std::getenv("BBHOST_NET_POLL");
    return e && e[0] == '1';
#endif
}();
std::atomic<std::uint32_t> g_net_wake_word{0};
void net_wake() {
    g_net_wake_word.fetch_add(1, std::memory_order_seq_cst);
    host_futex_wake_all(&g_net_wake_word);
}

int net_err(int e) {
    return kNetErrBase | (to_freebsd(e) & 0xff);
}

std::uint16_t bswap16(std::uint16_t v) {
    return static_cast<std::uint16_t>((v << 8) | (v >> 8));
}
std::uint32_t bswap32(std::uint32_t v) {
    return (v << 24) | ((v << 8) & 0x00ff0000u) | ((v >> 8) & 0x0000ff00u) | (v >> 24);
}

int parse_ipv4(const char* s, std::uint32_t* net) {
    if (!s || !net) {
        return 0;
    }
    unsigned a = 0, b = 0, c = 0, d = 0;
    char extra = 0;
    if (std::sscanf(s, "%u.%u.%u.%u%c", &a, &b, &c, &d, &extra) != 4) {
        return 0;
    }
    if (a > 255 || b > 255 || c > 255 || d > 255) {
        return 0;
    }
    *net = bswap32((a << 24) | (b << 16) | (c << 8) | d);
    return 1;
}

void write_ipv4(char* dst, unsigned n, std::uint32_t net) {
    if (!dst || n == 0) {
        return;
    }
    const std::uint32_t host = bswap32(net);
    std::snprintf(dst, n, "%u.%u.%u.%u", (host >> 24) & 0xffu, (host >> 16) & 0xffu,
                  (host >> 8) & 0xffu, host & 0xffu);
}

thread_local int g_net_errno = 0;
int g_net_inited = 0;
int g_net_next = 1;
int g_net_state = 3;
std::mutex g_net_mu;

struct NetPool {
    std::string name;
    int size = 0;
};
// SceNetSockaddrIn: {len, family, port (nbo), addr (nbo), vport (nbo), zero[6]}.
struct SceSockaddrIn {
    std::uint8_t len;
    std::uint8_t family;
    std::uint16_t port;
    std::uint32_t addr;
    std::uint16_t vport;
    std::uint8_t zero[6];
};
static_assert(sizeof(SceSockaddrIn) == 16, "SceNetSockaddrIn is 16 bytes");

// The PS4 kernel multiplexes virtual ports over one UDP port with a header on
// every datagram: [0xff][flags][src vport][dst vport] when flag 0x40 says the
// vports are one byte (Bloodborne's are, 40 and 30), two-byte vports without
// it, four more bytes when flag 0x20 adds a comid. The header is kept on the
// wire so a shadPS4 peer reads ours and we read its.
constexpr std::uint8_t kP2pMagic = 0xff;
constexpr std::uint8_t kP2pFlagP2p = 0x80, kP2pFlagByteVports = 0x40, kP2pFlagComid = 0x20, kP2pType = 0x03;

struct Datagram {
    std::vector<std::uint8_t> data;
    std::uint32_t addr = 0;  // nbo
    std::uint16_t port = 0;  // nbo
    std::uint16_t vport = 0; // host order, the sender's
};
// A P2P game socket's inbox: what the port's reader demultiplexed to its
// vport. qfd is an eventfd counting the queued datagrams, so the game's
// epoll can watch it.
struct SockQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Datagram> q;
    int qfd = -1;  // Linux epoll mode: an eventfd counting the queued datagrams
    int sock = 0;
    std::uint16_t vport = 0;
    // Counters for the desync work: what came in, what went out, how deep
    // the inbox got between the game's reads.
    std::uint64_t in_dgrams = 0, in_bytes = 0, out_dgrams = 0, out_bytes = 0, forwarded = 0;
    std::size_t hiwater = 0;
};
// One UDP socket per P2P port, shared by every game socket that binds a
// vport on it (the game binds 40 at boot and 30 for a session, on the same
// port 3658; two game sockets on one UDP port would otherwise hand the
// kernel the choice). A reader thread drains it into the queues.
struct P2pPort {
    sockfd_t fd = kBadSock;
    std::uint16_t port = 0;  // host order
    std::mutex mu;
    std::map<std::uint16_t, std::shared_ptr<SockQueue>> by_vport;
    std::thread reader;
    std::atomic<bool> stop{false};
    std::uint64_t rx = 0, unmatched = 0, no_header = 0, probes = 0;
    std::chrono::steady_clock::time_point last_report{};
    // One STUN Binding exchange at a time (hle_net_p2p_stun): the reader
    // answers the transaction it finds here (under mu).
    bool stun_pending = false, stun_done = false;
    std::uint8_t stun_txid[net::stun::kTxid] = {};
    std::uint32_t stun_addr = 0;
    std::uint16_t stun_port = 0;
    net::stun::Relay stun_relay;
    std::condition_variable stun_cv;
    // When a datagram last arrived from each source (addr | port << 32, both
    // network order): the hole-punch stops once the peer is heard.
    std::map<std::uint64_t, std::chrono::steady_clock::time_point> heard;
};
// A hole-punch probe: not a P2P datagram (no 0xff), dropped by the reader.
constexpr std::uint8_t kProbe[8] = {0xfe, 'b', 'b', 'h', 'p', 0, 0, 0};

// The private server's relay (net/stun.h): once a STUN answer carried
// BBHOST-RELAY, a datagram for another relay port (the server's address, not
// its STUN port) leaves framed for the STUN port - [0xfb]['R'][token][port]
// - and the relay's deliveries come back from the STUN port framed
// [0xfb]['r'][source port]. One destination keeps one NAT mapping, the one
// the keepalive (the same binding) holds open, so every NAT lets them in.
// The game sees none of it: its peers stay server:port both ways.
constexpr std::uint8_t kRelayMagic = 0xfb, kRelayToServer = 'R', kRelayToClient = 'r';
constexpr std::size_t kRelayHeader = 12;
struct RelayLink {
    std::mutex mu;
    bool on = false;
    std::uint32_t server = 0;     // network order
    std::uint16_t stun_port = 0;  // network order
    std::uint8_t token[net::stun::kTokenLen] = {};
    std::uint16_t vport = 0;      // host order
};
RelayLink g_relay;

// True when addr:port (network order) is a relay port to frame for.
bool relay_target(std::uint32_t addr, std::uint16_t port_nbo, std::uint8_t token[net::stun::kTokenLen],
                  sockaddr_in* stun_sa) {
    std::lock_guard<std::mutex> lk(g_relay.mu);
    if (!g_relay.on || addr != g_relay.server || port_nbo == g_relay.stun_port) return false;
    std::memcpy(token, g_relay.token, net::stun::kTokenLen);
    *stun_sa = sockaddr_in{};
    stun_sa->sin_family = AF_INET;
    stun_sa->sin_addr.s_addr = g_relay.server;
    stun_sa->sin_port = g_relay.stun_port;
    return true;
}
bool from_relay_server(const sockaddr_in& sa) {
    std::lock_guard<std::mutex> lk(g_relay.mu);
    return g_relay.on && sa.sin_addr.s_addr == g_relay.server && sa.sin_port == g_relay.stun_port;
}
inline std::uint64_t source_key(std::uint32_t addr, std::uint16_t port_nbo) {
    return addr | (static_cast<std::uint64_t>(port_nbo) << 32);
}

// One line per port every 60 s (and at close): the stream's shape.
void p2p_report_locked(P2pPort& port, const char* when) {
    std::string line;
    char buf[160];
    for (auto& [vp, q] : port.by_vport) {
        std::lock_guard<std::mutex> lk(q->mu);
        std::snprintf(buf, sizeof(buf), " vport %u (sock %d): in %llu/%llu B, out %llu/%llu B, hiwater %zu, fwd %llu;", vp,
                      q->sock, static_cast<unsigned long long>(q->in_dgrams), static_cast<unsigned long long>(q->in_bytes),
                      static_cast<unsigned long long>(q->out_dgrams), static_cast<unsigned long long>(q->out_bytes),
                      q->hiwater, static_cast<unsigned long long>(q->forwarded));
        line += buf;
    }
    host_log("net: p2p port %u %s: rx %llu, unmatched %llu, no header %llu, probes %llu;%s", port.port, when,
             static_cast<unsigned long long>(port.rx), static_cast<unsigned long long>(port.unmatched),
             static_cast<unsigned long long>(port.no_header), static_cast<unsigned long long>(port.probes), line.c_str());
}
std::map<std::uint16_t, std::shared_ptr<P2pPort>> g_p2p_ports;

struct NetSock {
    int domain = 2;
    int type = 1;
    int nonblock = 0;
    int listening = 0;
    // UDP (type 2) sockets are real datagram sockets; SOCK_DGRAM_P2P (type 6)
    // sockets share the instance's P2P port through `port` and read from
    // `queue`. Stream sockets stay the old fakes: the game's HTTP is sceHttp.
    sockfd_t fd = kBadSock;
    bool p2p = false;
    std::uint16_t vport = 0;  // host order, what the game bound
    std::shared_ptr<P2pPort> port;
    std::shared_ptr<SockQueue> queue;
};
struct NetEpoll {
    std::string name;
    std::mutex mu;
    std::condition_variable cv;
    bool aborting = false;
    std::unordered_map<int, std::uint32_t> watch;
    int efd = -1;    // Linux epoll mode: a real epoll for the real sockets
    int wake = -1;   // Linux epoll mode: an eventfd in it, for abort
    std::unordered_map<int, std::uint64_t> data;  // socket id -> the game's epoll data
};
struct NetResolver {
    std::string name;
    int pool = 0;
};
std::unordered_map<int, NetPool> g_net_pools;
std::unordered_map<int, NetSock> g_net_socks;
std::unordered_map<int, std::unique_ptr<NetEpoll>> g_net_epolls;
std::unordered_map<int, NetResolver> g_net_resolvers;
std::unordered_map<int, std::pair<void (*)(int, void*), void*>> g_net_cbs;
int g_net_cb_next = 1;

// The first datagrams are always logged (the P2P handshake is a few dozen
// small packets and the runs that fail are the ones without a trace);
// BBHOST_NET_TRACE=1 removes the cap.
bool g_net_trace = [] {
    const char* e = std::getenv("BBHOST_NET_TRACE");
    return e && e[0] == '1';
}();
constexpr int kTraceFirst = 400;
bool trace_now(int& count) {
    if (g_net_trace) return true;
    return ++count <= kTraceFirst;
}

int net_alloc() {
    return g_net_next++;
}

// The failing errno of a real call, as the game's sceNetErrnoLoc and return value.
int net_fail() {
    const int e = sock_errno();
    g_net_errno = to_freebsd(e);
    return net_err(e);
}
int net_again() {
    g_net_errno = 35;
    return net_err(EAGAIN);
}

// Splits a datagram at the P2P header; returns the header length (0 when
// there is none) and the source vport.
std::size_t p2p_header(const std::uint8_t* d, std::size_t n, std::uint16_t* src, std::uint16_t* dst) {
    if (n < 4 || d[0] != kP2pMagic) return 0;
    const std::uint8_t fl = d[1];
    std::size_t at = 2;
    if (fl & kP2pFlagComid) at += 4;
    std::size_t hdr;
    if (fl & kP2pFlagByteVports) {
        if (at + 2 > n) return 0;
        *src = d[at];
        *dst = d[at + 1];
        hdr = at + 2;
    } else {
        if (at + 4 > n) return 0;
        *src = static_cast<std::uint16_t>((d[at] << 8) | d[at + 1]);
        *dst = static_cast<std::uint16_t>((d[at + 2] << 8) | d[at + 3]);
        hdr = at + 4;
    }
    return hdr;
}

void p2p_reader(std::shared_ptr<P2pPort> port) {
    std::uint8_t buf[2048];
    int logs = 0;
    while (!port->stop.load(std::memory_order_relaxed)) {
        if (sock_poll(port->fd, POLLIN, 200) <= 0) continue;
        sockaddr_in sa{};
        socklen_type sl = sizeof(sa);
        ssize_type n = sock_recvfrom(port->fd, buf, sizeof(buf), true, &sa, &sl);
        if (n < 0) continue;
        if (n >= 4 && buf[0] == kRelayMagic && buf[1] == kRelayToClient && from_relay_server(sa)) {
            // A relay delivery: the datagram of the owner of the source
            // port, which the game knows as server:port.
            const std::uint16_t from_vport = static_cast<std::uint16_t>((buf[2] << 8) | buf[3]);
            std::memmove(buf, buf + 4, static_cast<std::size_t>(n - 4));
            n -= 4;
            sa.sin_port = bswap16(from_vport);
        }
        std::uint16_t src = 0, dst = 0;
        const std::size_t hdr = p2p_header(buf, static_cast<std::size_t>(n), &src, &dst);
        std::shared_ptr<SockQueue> q;
        bool forwarded = false;
        if (!hdr) {
            // Not the game's: a peer's hole-punch probe, or the STUN
            // server's answer to our Binding Request. Neither reaches a
            // game socket.
            if (n >= 8 && std::memcmp(buf, kProbe, 5) == 0) {
                std::lock_guard<std::mutex> lk(port->mu);
                ++port->rx;
                ++port->probes;
                port->heard[source_key(sa.sin_addr.s_addr, sa.sin_port)] = std::chrono::steady_clock::now();
                continue;
            }
            if (n >= 20 && buf[0] == 0x01 && buf[1] == 0x01) {
                std::lock_guard<std::mutex> lk(port->mu);
                std::uint32_t a = 0;
                std::uint16_t p = 0;
                if (port->stun_pending && net::stun::parse_binding_response(buf, static_cast<std::size_t>(n),
                                                                            port->stun_txid, &a, &p, &port->stun_relay)) {
                    ++port->rx;
                    port->stun_addr = a;
                    port->stun_port = p;
                    port->stun_done = true;
                    port->stun_pending = false;
                    port->stun_cv.notify_all();
                    continue;
                }
            }
        }
        {
            std::lock_guard<std::mutex> lk(port->mu);
            ++port->rx;
            port->heard[source_key(sa.sin_addr.s_addr, sa.sin_port)] = std::chrono::steady_clock::now();
            if (!hdr) ++port->no_header;
            auto it = hdr ? port->by_vport.find(dst) : port->by_vport.end();
            if (it != port->by_vport.end()) {
                q = it->second;
            } else if (!port->by_vport.empty()) {
                // No socket on that vport (or no header): the first bound
                // vport takes it, as the fork forwarded a PS4 peer's vport
                // 30 to the game's 40.
                q = port->by_vport.begin()->second;
                forwarded = true;
                if (hdr) ++port->unmatched;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now - port->last_report > std::chrono::seconds(60)) {
                port->last_report = now;
                p2p_report_locked(*port, "60 s");
            }
        }
        if (!q) continue;
        Datagram dg;
        dg.data.assign(buf + hdr, buf + n);
        dg.addr = sa.sin_addr.s_addr;
        dg.port = sa.sin_port;
        dg.vport = src;
        if (trace_now(logs)) {
            char ip[32];
            write_ipv4(ip, sizeof(ip), sa.sin_addr.s_addr);
            host_log("net: recv %d <- %s:%u vport %u->%u: %zu bytes%s", q->sock, ip, bswap16(sa.sin_port), src, dst,
                     dg.data.size(), hdr ? (dst == q->vport ? "" : " (forwarded)") : " (no p2p header)");
        }
        {
            std::lock_guard<std::mutex> lk(q->mu);
            ++q->in_dgrams;
            q->in_bytes += dg.data.size();
            if (forwarded) ++q->forwarded;
            q->q.push_back(std::move(dg));
            if (q->q.size() > q->hiwater) q->hiwater = q->q.size();
#if !defined(_WIN32)
            const std::uint64_t one = 1;
            if (q->qfd >= 0) (void)!::write(q->qfd, &one, sizeof(one));
#endif
        }
        q->cv.notify_all();
        if (g_net_poll) net_wake();
    }
}

GUEST_ABI int hle_net_init() {
    std::lock_guard<std::mutex> lock(g_net_mu);
#if defined(_WIN32)
    static bool wsa_started = [] {
        WSADATA d{};
        return ::WSAStartup(MAKEWORD(2, 2), &d) == 0;
    }();
    (void)wsa_started;
#endif
    g_net_inited = 1;
    host_log("sceNetInit");
    return 0;
}
GUEST_ABI int hle_net_term() {
    std::lock_guard<std::mutex> lock(g_net_mu);
    g_net_inited = 0;
    return 0;
}
GUEST_ABI int* hle_net_errno_loc() { return &g_net_errno; }

GUEST_ABI int hle_net_pool_create(const char* name, int size, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    const int id = net_alloc();
    g_net_pools[id] = NetPool{name ? name : "", size};
    host_log("sceNetPoolCreate %s size=%d -> %d", name ? name : "", size, id);
    return id;
}
GUEST_ABI int hle_net_pool_destroy(int id) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    g_net_pools.erase(id);
    return 0;
}

GUEST_ABI int hle_net_socket(const char* name, int domain, int type, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    const int id = net_alloc();
    NetSock sock;
    sock.domain = domain;
    sock.type = type;
    if (domain == 2 && type == 2) {
        sock.fd = sock_udp();
    } else if (domain == 2 && type == 6) {
        sock.p2p = true;
        sock.queue = std::make_shared<SockQueue>();
#if defined(__linux__)
        if (!g_net_poll) sock.queue->qfd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC | EFD_SEMAPHORE);
#endif
        sock.queue->sock = id;
    }
    g_net_socks[id] = sock;
    host_log("sceNetSocket %s domain=%d type=%d -> %d%s", name ? name : "", domain, type, id,
             sock.p2p ? " (udp p2p)" : (sock_ok(sock.fd) ? " (udp)" : ""));
    return id;
}
// Under g_net_mu. The instance's P2P UDP port, opened on first use (the
// game's first P2P bind, or the session layer's STUN query before it).
std::shared_ptr<P2pPort> p2p_port_open(std::uint16_t want) {
    auto pit = g_p2p_ports.find(want);
    if (pit != g_p2p_ports.end()) return pit->second;
    auto port = std::make_shared<P2pPort>();
    // The port is this instance's alone. With SO_REUSEADDR a second bbhost on
    // the machine bound it too and the kernel handed it our peers' datagrams:
    // this game kept sending, heard nothing, and left every session ~40 s in
    // (dev, 2026-10-06, a test harness beside a player). Windows also lets a
    // later SO_REUSEADDR socket take a port unless the first is exclusive.
    // When another program has it, the next free port is ours and becomes
    // what the server is told (net::p2p_port).
    constexpr int kTries = 16;
    for (int i = 0; i < kTries; ++i) {
        const auto at = static_cast<std::uint16_t>(want + i);
        port->fd = sock_udp();
        if (!sock_ok(port->fd)) return nullptr;
#if defined(_WIN32)
        sock_setopt_int(port->fd, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1);
#endif
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = bswap16(at);
        sa.sin_addr.s_addr = INADDR_ANY;
        if (::bind(port->fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) == 0) {
            port->port = at;
            break;
        }
        const int e = sock_errno();
        sock_close(port->fd);
        port->fd = kBadSock;
        if (e != EADDRINUSE && e != EACCES) {
            host_log("net: P2P port %u: bind failed: %s", at, std::strerror(e));
            return nullptr;
        }
        host_log("net: P2P port %u is in use by another program%s", at,
                 i + 1 < kTries ? "; trying the next" : "");
    }
    if (!sock_ok(port->fd)) return nullptr;
    if (port->port != want) {
        host_log("net: P2P port %u (online.p2p_port %u was taken); peers reach this game there", port->port, want);
        net::set_p2p_port_bound(port->port);
    }
    port->reader = std::thread(p2p_reader, port);
    g_p2p_ports[port->port] = port;
    return port;
}

// Under g_net_mu. Detaches a P2P socket from its port; the port's reader
// stops with its last socket.
void p2p_detach(NetSock& sock) {
    if (!sock.port) return;
    std::shared_ptr<P2pPort> port = sock.port;
    bool empty = false;
    {
        std::lock_guard<std::mutex> lk(port->mu);
        p2p_report_locked(*port, "at a socket's close");
        auto it = port->by_vport.find(sock.vport);
        if (it != port->by_vport.end() && it->second == sock.queue) port->by_vport.erase(it);
        empty = port->by_vport.empty();
    }
    sock.port.reset();
    if (empty) {
        port->stop.store(true, std::memory_order_relaxed);
        if (port->reader.joinable()) port->reader.join();
        if (sock_ok(port->fd)) sock_close(port->fd);
        port->fd = kBadSock;
        g_p2p_ports.erase(port->port);
    }
}
GUEST_ABI int hle_net_socket_close(int s) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    auto it = g_net_socks.find(s);
    if (it == g_net_socks.end()) {
        return net_err(EBADF);
    }
    if (sock_ok(it->second.fd)) sock_close(it->second.fd);
    p2p_detach(it->second);
#if !defined(_WIN32)
    if (it->second.queue && it->second.queue->qfd >= 0) ::close(it->second.queue->qfd);
#endif
    for (auto& [eid, ep] : g_net_epolls) {
        ep->watch.erase(s);
        ep->data.erase(s);
    }
    g_net_socks.erase(it);
    return 0;
}
GUEST_ABI int hle_net_socket_abort(int s, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    return g_net_socks.count(s) ? 0 : net_err(EBADF);
}
GUEST_ABI int hle_net_bind(int s, const void* addr, int len) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    auto it = g_net_socks.find(s);
    if (it == g_net_socks.end()) return net_err(EBADF);
    NetSock& sock = it->second;
    SceSockaddrIn in{};
    if (addr && len >= static_cast<int>(sizeof(in))) std::memcpy(&in, addr, sizeof(in));
    if (sock.p2p) {
        // A P2P socket binds the instance's own UDP port (online.p2p_port),
        // not the game's 3658: two instances share this machine. The vport
        // is the game's.
        const auto want = static_cast<std::uint16_t>(net::p2p_port());
        std::shared_ptr<P2pPort> port = p2p_port_open(want);
        if (!port) {
            const int r = net_fail();
            host_log("sceNetBind %d port %u vport %u -> udp %u failed: %s", s, bswap16(in.port), bswap16(in.vport), want,
                     std::strerror(sock_errno()));
            return r;
        }
        p2p_detach(sock);
        sock.vport = bswap16(in.vport);
        sock.queue->vport = sock.vport;
        sock.port = port;
        {
            std::lock_guard<std::mutex> lk(port->mu);
            port->by_vport[sock.vport] = sock.queue;
        }
        host_log("sceNetBind %d port %u vport %u -> udp %u (p2p)", s, bswap16(in.port), sock.vport, want);
        return 0;
    }
    if (!sock_ok(sock.fd)) return 0;
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = in.port;
    sa.sin_addr.s_addr = INADDR_ANY;
    if (::bind(sock.fd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) < 0) return net_fail();
    host_log("sceNetBind %d port %u (udp)", s, bswap16(in.port));
    return 0;
}
GUEST_ABI int hle_net_listen(int s, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    auto it = g_net_socks.find(s);
    if (it == g_net_socks.end()) {
        return net_err(EBADF);
    }
    it->second.listening = 1;
    return 0;
}
GUEST_ABI int hle_net_accept(int s, void* addr, unsigned* addrlen) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    if (!g_net_socks.count(s)) {
        return net_err(EBADF);
    }
    if (addr && addrlen && *addrlen >= 16) {
        unsigned char sa[16]{};
        sa[0] = 16;
        sa[1] = 2;
        std::memcpy(addr, sa, 16);
        *addrlen = 16;
    }
    const int id = net_alloc();
    NetSock sock;
    g_net_socks[id] = sock;
    return id;
}
GUEST_ABI int hle_net_connect(int s, const void*, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    return g_net_socks.count(s) ? 0 : net_err(EBADF);
}
GUEST_ABI int hle_net_shutdown(int s, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    return g_net_socks.count(s) ? 0 : net_err(EBADF);
}
GUEST_ABI int hle_net_setsockopt(int s, int level, int name, const void* val, int len) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    auto it = g_net_socks.find(s);
    if (it == g_net_socks.end()) return net_err(EBADF);
    NetSock& sock = it->second;
    int v = 0;
    if (val && len >= 4) std::memcpy(&v, val, 4);
    if (name == 0x1200) {  // SCE_NET_SO_NBIO
        sock.nonblock = v != 0;
#if !defined(_WIN32)
        if (sock_ok(sock.fd)) sock_set_nonblock(sock.fd, v != 0);  // Windows: always non-blocking, see sock_recvfrom
#endif
    } else if (sock_ok(sock.fd) && level == 0xffff && (name == 0x1001 || name == 0x1002)) {  // SO_SNDBUF / SO_RCVBUF
        sock_setopt_int(sock.fd, SOL_SOCKET, name == 0x1001 ? SO_SNDBUF : SO_RCVBUF, v);
    }
    if (g_net_trace) host_log("sceNetSetsockopt %d level 0x%x name 0x%x = %d", s, level, name, v);
    return 0;
}
GUEST_ABI int hle_net_getsockopt(int s, int, int, void* val, unsigned* len) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    if (!g_net_socks.count(s)) {
        return net_err(EBADF);
    }
    if (val && len && *len >= 4) {
        std::memset(val, 0, 4);
        *len = 4;
    }
    return 0;
}
GUEST_ABI int hle_net_getsockname(int s, void* addr, unsigned* addrlen) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    auto it = g_net_socks.find(s);
    if (it == g_net_socks.end()) {
        return net_err(EBADF);
    }
    if (addr && addrlen && *addrlen >= 16) {
        SceSockaddrIn in{};
        in.len = 16;
        in.family = 2;
        const sockfd_t fd = it->second.port ? it->second.port->fd : it->second.fd;
        if (sock_ok(fd)) {
            sockaddr_in sa{};
            socklen_type sl = sizeof(sa);
            if (::getsockname(fd, reinterpret_cast<sockaddr*>(&sa), &sl) == 0) {
                in.port = sa.sin_port;
                in.addr = sa.sin_addr.s_addr;
            }
            in.vport = bswap16(it->second.vport);
        }
        std::memcpy(addr, &in, 16);
        *addrlen = 16;
    }
    return 0;
}

// A copy of the socket's runtime state, without holding g_net_mu across a
// blocking call. Returns false for an unknown socket.
bool sock_state(int s, NetSock* out) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    auto it = g_net_socks.find(s);
    if (it == g_net_socks.end()) return false;
    *out = it->second;
    return true;
}

// Reads one datagram: from the socket's queue for P2P, the fd for UDP.
// Returns the payload length, or the game's error.
int dgram_recv(int s, void* buf, std::uint64_t len, int flags, SceSockaddrIn* from) {
    NetSock sock;
    if (!sock_state(s, &sock)) return net_err(EBADF);
    const bool dontwait = sock.nonblock || (flags & 0x80);  // 0x80: SCE_NET_MSG_DONTWAIT
    if (sock.p2p) {
        if (!sock.queue) return net_again();
        SockQueue& q = *sock.queue;
        Datagram dg;
        {
            std::unique_lock<std::mutex> lk(q.mu);
            if (q.q.empty()) {
                if (dontwait) return net_again();
                q.cv.wait(lk, [&] { return !q.q.empty(); });
            }
            dg = std::move(q.q.front());
            q.q.pop_front();
#if !defined(_WIN32)
            std::uint64_t v = 0;
            if (q.qfd >= 0) (void)!::read(q.qfd, &v, sizeof(v));
#endif
        }
        const std::size_t take = dg.data.size() < len ? dg.data.size() : static_cast<std::size_t>(len);
        if (buf && take) std::memcpy(buf, dg.data.data(), take);
        if (from) {
            std::memset(from, 0, sizeof(*from));
            from->len = 16;
            from->family = 2;
            from->port = dg.port;
            from->addr = dg.addr;
            from->vport = bswap16(dg.vport);
        }
        return static_cast<int>(take);
    }
    if (!sock_ok(sock.fd)) return dontwait ? net_again() : 0;
    sockaddr_in sa{};
    socklen_type sl = sizeof(sa);
    const ssize_type n = sock_recvfrom(sock.fd, buf, static_cast<std::size_t>(len), dontwait, &sa, &sl);
    if (n < 0) return sock_errno() == EAGAIN ? net_again() : net_fail();
    if (from) {
        std::memset(from, 0, sizeof(*from));
        from->len = 16;
        from->family = 2;
        from->port = sa.sin_port;
        from->addr = sa.sin_addr.s_addr;
    }
    return static_cast<int>(n);
}

// Sends one datagram, with the P2P header in front on a P2P socket.
int dgram_send(int s, const void* buf, std::uint64_t len, int flags, const SceSockaddrIn* to) {
    NetSock sock;
    if (!sock_state(s, &sock)) return net_err(EBADF);
    const sockfd_t fd = sock.port ? sock.port->fd : sock.fd;
    if (!sock_ok(fd)) return static_cast<int>(len);
    if (!to) return net_err(EINVAL);
    std::uint8_t pkt[2048];
    std::size_t hdr = 0;
    const std::uint16_t dst_vport = bswap16(to->vport);
    if (sock.p2p) {
        pkt[0] = kP2pMagic;
        if (sock.vport < 256 && dst_vport < 256) {
            pkt[1] = kP2pFlagP2p | kP2pFlagByteVports | kP2pType;
            pkt[2] = static_cast<std::uint8_t>(sock.vport);
            pkt[3] = static_cast<std::uint8_t>(dst_vport);
            hdr = 4;
        } else {
            pkt[1] = kP2pFlagP2p | kP2pType;
            pkt[2] = static_cast<std::uint8_t>(sock.vport >> 8);
            pkt[3] = static_cast<std::uint8_t>(sock.vport);
            pkt[4] = static_cast<std::uint8_t>(dst_vport >> 8);
            pkt[5] = static_cast<std::uint8_t>(dst_vport);
            hdr = 6;
        }
    }
    if (hdr + len > sizeof(pkt)) return net_err(EMSGSIZE);
    if (buf && len) std::memcpy(pkt + hdr, buf, static_cast<std::size_t>(len));
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = to->port;
    sa.sin_addr.s_addr = to->addr;
    const bool dontwait = sock.nonblock || (flags & 0x80);
    ssize_type n;
    std::uint8_t token[net::stun::kTokenLen];
    sockaddr_in stun_sa{};
    if (sock.p2p && relay_target(to->addr, to->port, token, &stun_sa)) {
        std::uint8_t frame[sizeof(pkt) + kRelayHeader];
        frame[0] = kRelayMagic;
        frame[1] = kRelayToServer;
        std::memcpy(frame + 2, token, net::stun::kTokenLen);
        std::memcpy(frame + 10, &to->port, 2);  // network order is the frame's big-endian
        std::memcpy(frame + kRelayHeader, pkt, hdr + static_cast<std::size_t>(len));
        n = sock_sendto(fd, frame, kRelayHeader + hdr + static_cast<std::size_t>(len), dontwait, &stun_sa);
        if (n >= 0) n -= static_cast<ssize_type>(kRelayHeader);
    } else {
        n = sock_sendto(fd, pkt, hdr + static_cast<std::size_t>(len), dontwait, &sa);
    }
    static int logs = 0;
    if (trace_now(logs)) {
        char ip[32];
        write_ipv4(ip, sizeof(ip), to->addr);
        host_log("net: send %d -> %s:%u vport %u->%u: %llu bytes -> %lld", s, ip, bswap16(to->port), sock.vport,
                 dst_vport, static_cast<unsigned long long>(len), static_cast<long long>(n));
    }
    if (n < 0) return sock_errno() == EAGAIN ? net_again() : net_fail();
    if (sock.queue) {
        std::lock_guard<std::mutex> lk(sock.queue->mu);
        ++sock.queue->out_dgrams;
        sock.queue->out_bytes += len;
    }
    return static_cast<int>(len);
}

GUEST_ABI int hle_net_recv(int s, void* buf, std::uint64_t len, int flags) {
    return dgram_recv(s, buf, len, flags, nullptr);
}
GUEST_ABI int hle_net_send(int s, const void* buf, std::uint64_t len, int) {
    NetSock sock;
    if (!sock_state(s, &sock)) return net_err(EBADF);
    if (!sock_ok(sock.fd)) return static_cast<int>(len);
    const ssize_type n = sock_send(sock.fd, buf, static_cast<std::size_t>(len), sock.nonblock);
    if (n < 0) return net_fail();
    return static_cast<int>(n);
}
GUEST_ABI int hle_net_recvfrom(int s, void* buf, std::uint64_t len, int flags, void* from, unsigned* fromlen) {
    SceSockaddrIn in{};
    const int r = dgram_recv(s, buf, len, flags, &in);
    if (r >= 0 && from && fromlen && *fromlen >= 16) {
        std::memcpy(from, &in, 16);
        *fromlen = 16;
    }
    return r;
}
GUEST_ABI int hle_net_sendto(int s, const void* buf, std::uint64_t len, int flags, const void* to, int tolen) {
    SceSockaddrIn in{};
    if (to && tolen >= 16) std::memcpy(&in, to, 16);
    return dgram_send(s, buf, len, flags, (to && tolen >= 16) ? &in : nullptr);
}

GUEST_ABI std::uint16_t hle_net_htons(std::uint16_t v) { return bswap16(v); }
GUEST_ABI std::uint16_t hle_net_ntohs(std::uint16_t v) { return bswap16(v); }
GUEST_ABI std::uint32_t hle_net_htonl(std::uint32_t v) { return bswap32(v); }
GUEST_ABI std::uint32_t hle_net_ntohl(std::uint32_t v) { return bswap32(v); }

GUEST_ABI int hle_net_inet_pton(int af, const char* src, void* dst) {
    if (af != 2 || !dst) {
        return net_err(47);
    }
    std::uint32_t net = 0;
    if (!parse_ipv4(src, &net)) {
        return 0;
    }
    std::memcpy(dst, &net, 4);
    return 1;
}
GUEST_ABI const char* hle_net_inet_ntop(int af, const void* src, char* dst, unsigned size) {
    if (af != 2 || !src || !dst || size < 8) {
        return nullptr;
    }
    std::uint32_t net = 0;
    std::memcpy(&net, src, 4);
    write_ipv4(dst, size, net);
    return dst;
}

GUEST_ABI int hle_net_epoll_create(const char* name, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    const int id = net_alloc();
    auto ep = std::make_unique<NetEpoll>();
    if (name) {
        ep->name = name;
    }
#if defined(__linux__)
    if (!g_net_poll) {
        ep->efd = ::epoll_create1(EPOLL_CLOEXEC);
        ep->wake = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (ep->efd >= 0 && ep->wake >= 0) {
            epoll_event ev{};
            ev.events = EPOLLIN;
            ev.data.u64 = ~0ull;
            ::epoll_ctl(ep->efd, EPOLL_CTL_ADD, ep->wake, &ev);
        }
    }
#endif
    g_net_epolls[id] = std::move(ep);
    host_log("sceNetEpollCreate %s -> %d", name ? name : "", id);
    return id;
}
GUEST_ABI int hle_net_epoll_destroy(int id) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    auto it = g_net_epolls.find(id);
    if (it == g_net_epolls.end()) {
        return net_err(EBADF);
    }
    {
        std::lock_guard<std::mutex> lk(it->second->mu);
        it->second->aborting = true;
        it->second->cv.notify_all();
    }
#if !defined(_WIN32)
    if (it->second->wake >= 0) ::close(it->second->wake);
    if (it->second->efd >= 0) ::close(it->second->efd);
#endif
    g_net_epolls.erase(it);
    net_wake();
    return 0;
}
// The game's SceNetEpollEvent: {u32 events, u32 pad, u64 ident, u64 data}.
struct SceEpollEvent {
    std::uint32_t events;
    std::uint32_t pad;
    std::uint64_t ident;
    std::uint64_t data;
};
GUEST_ABI int hle_net_epoll_control(int id, int op, int sock, void* event) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    auto it = g_net_epolls.find(id);
    if (it == g_net_epolls.end()) {
        return net_err(EBADF);
    }
    NetEpoll& ep = *it->second;
    SceEpollEvent sev{};
    if (event) std::memcpy(&sev, event, sizeof(sev));
    if (op == 3) {
        ep.watch.erase(sock);
        ep.data.erase(sock);
    } else {
        ep.watch[sock] = sev.events;
        ep.data[sock] = sev.data;
    }
#if defined(__linux__)
    if (!g_net_poll) {
        auto sit = g_net_socks.find(sock);
        int fd = -1;
        if (sit != g_net_socks.end()) fd = sit->second.queue ? sit->second.queue->qfd : sit->second.fd;
        if (op == 3) {
            if (fd >= 0 && ep.efd >= 0) ::epoll_ctl(ep.efd, EPOLL_CTL_DEL, fd, nullptr);
        } else if (fd >= 0 && ep.efd >= 0) {
            epoll_event ev{};
            if (sev.events & 0x1) ev.events |= EPOLLIN;
            if (sev.events & 0x2) ev.events |= EPOLLOUT;
            ev.data.u64 = static_cast<std::uint64_t>(sock);
            if (::epoll_ctl(ep.efd, op == 2 ? EPOLL_CTL_MOD : EPOLL_CTL_ADD, fd, &ev) < 0 && errno == EEXIST) {
                ::epoll_ctl(ep.efd, EPOLL_CTL_MOD, fd, &ev);
            }
        }
    }
#endif
    net_wake();
    if (g_net_trace) host_log("sceNetEpollControl %d op %d sock %d events 0x%x", id, op, sock, sev.events);
    return 0;
}
GUEST_ABI int hle_net_epoll_wait(int id, void* events, int maxevents, int timeout) {
    NetEpoll* ep = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_net_mu);
        auto it = g_net_epolls.find(id);
        if (it == g_net_epolls.end()) {
            return net_err(EBADF);
        }
        ep = it->second.get();
    }
    if (g_net_poll) {
        // timeout is in microseconds, -1 forever.
        const auto deadline = timeout < 0 ? std::chrono::steady_clock::time_point::max()
                                          : std::chrono::steady_clock::now() + std::chrono::microseconds(timeout);
        for (;;) {
            // Before the look: a wake after it changes the word, and the
            // sleep below then returns at once.
            const std::uint32_t gen = g_net_wake_word.load(std::memory_order_seq_cst);
            int out = 0;
            {
                std::lock_guard<std::mutex> lock(g_net_mu);
                if (!g_net_epolls.count(id)) return net_err(EBADF);
                for (const auto& [sock, want] : ep->watch) {
                    auto sit = g_net_socks.find(sock);
                    if (sit == g_net_socks.end()) continue;
                    const NetSock& ns = sit->second;
                    std::uint32_t ready = 0;
                    if (ns.p2p && ns.queue) {
                        std::lock_guard<std::mutex> lk(ns.queue->mu);
                        if ((want & 0x1) && !ns.queue->q.empty()) ready |= 0x1;
                        if (want & 0x2) ready |= 0x2;  // a datagram send never waits
                    } else if (sock_ok(ns.fd)) {
                        short ev = 0;
                        if (want & 0x1) ev |= POLLIN;
                        if (want & 0x2) ev |= POLLOUT;
                        if (ev && sock_poll(ns.fd, ev, 0) > 0) {
                            ready = want & 0x3;
                        }
                    }
                    if (!ready) continue;
                    SceEpollEvent sev{};
                    sev.events = ready;
                    sev.ident = static_cast<std::uint64_t>(sock);
                    auto d = ep->data.find(sock);
                    sev.data = d == ep->data.end() ? 0 : d->second;
                    if (events && out < maxevents) {
                        std::memcpy(static_cast<std::uint8_t*>(events) + static_cast<std::size_t>(out) * sizeof(sev),
                                    &sev, sizeof(sev));
                        ++out;
                    }
                }
            }
            if (out) return out;
            {
                std::unique_lock<std::mutex> lk(ep->mu);
                if (ep->aborting) {
                    ep->aborting = false;
                    return 0;
                }
            }
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) return 0;
            const auto slice = std::chrono::milliseconds(10);
            const HostDeadline until = deadline - now < slice ? deadline : now + slice;
            host_futex_wait(&g_net_wake_word, gen, &until);
        }
    }
#if defined(__linux__)
    if (ep->efd >= 0) {
        // timeout is in microseconds, -1 forever.
        const int ms = timeout < 0 ? -1 : static_cast<int>((timeout + 999) / 1000);
        epoll_event evs[16];
        const int want = maxevents < 1 ? 1 : (maxevents > 16 ? 16 : maxevents);
        const int n = ::epoll_wait(ep->efd, evs, want, ms);
        if (n < 0) {
            if (errno == EINTR) return 0;
            return net_fail();
        }
        int out = 0;
        std::lock_guard<std::mutex> lock(g_net_mu);
        for (int i = 0; i < n; ++i) {
            if (evs[i].data.u64 == ~0ull) {
                std::uint64_t v = 0;
                (void)!::read(ep->wake, &v, sizeof(v));
                continue;
            }
            const int sock = static_cast<int>(evs[i].data.u64);
            SceEpollEvent sev{};
            if (evs[i].events & EPOLLIN) sev.events |= 0x1;
            if (evs[i].events & EPOLLOUT) sev.events |= 0x2;
            if (evs[i].events & EPOLLERR) sev.events |= 0x8;
            if (evs[i].events & EPOLLHUP) sev.events |= 0x10;
            sev.ident = static_cast<std::uint64_t>(sock);
            auto d = ep->data.find(sock);
            sev.data = d == ep->data.end() ? 0 : d->second;
            if (events && out < maxevents) {
                std::memcpy(static_cast<std::uint8_t*>(events) + static_cast<std::size_t>(out) * sizeof(sev), &sev,
                            sizeof(sev));
                ++out;
            }
        }
        return out;
    }
#endif
    if (timeout == 0) {
        return 0;
    }
    std::unique_lock<std::mutex> lk(ep->mu);
    if (timeout < 0) {
        ep->cv.wait_for(lk, std::chrono::milliseconds(16), [&] { return ep->aborting; });
    } else {
        ep->cv.wait_for(lk, std::chrono::microseconds(timeout), [&] { return ep->aborting; });
    }
    ep->aborting = false;
    return 0;
}
GUEST_ABI int hle_net_epoll_abort(int id, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    auto it = g_net_epolls.find(id);
    if (it == g_net_epolls.end()) {
        return net_err(EBADF);
    }
    std::lock_guard<std::mutex> lk(it->second->mu);
    it->second->aborting = true;
    it->second->cv.notify_all();
    net_wake();
#if !defined(_WIN32)
    if (it->second->wake >= 0) {
        const std::uint64_t one = 1;
        (void)!::write(it->second->wake, &one, sizeof(one));
    }
#endif
    return 0;
}

GUEST_ABI int hle_net_resolver_create(const char* name, int memid, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    const int id = net_alloc();
    g_net_resolvers[id] = NetResolver{name ? name : "", memid};
    return id;
}
GUEST_ABI int hle_net_resolver_destroy(int id) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    g_net_resolvers.erase(id);
    return 0;
}
GUEST_ABI int hle_net_resolver_ntoa(int id, const char*, void* addr, int, int, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    if (!g_net_resolvers.count(id)) {
        return net_err(EBADF);
    }
    if (addr) {
        std::uint32_t loopback = bswap32(0x7f000001u);
        std::memcpy(addr, &loopback, 4);
    }
    return 0;
}
GUEST_ABI int hle_net_resolver_aton(int id, const void*, char* hostname, int len, int, int, int) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    if (!g_net_resolvers.count(id)) {
        return net_err(EBADF);
    }
    if (hostname && len > 0) {
        std::snprintf(hostname, static_cast<std::size_t>(len), "localhost");
    }
    return 0;
}

GUEST_ABI int hle_net_ctl_get_state(int* state) {
    if (!state) {
        return sce_err(EINVAL);
    }
    *state = g_net_state;
    return 0;
}
GUEST_ABI int hle_net_ctl_get_info(int code, void* info) {
    if (!info) {
        return sce_err(EINVAL);
    }
    std::memset(info, 0, 16);
    switch (code) {
        case 2: {
            static const unsigned char mac[6] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x01};
            std::memcpy(info, mac, 6);
            break;
        }
        case 4: {
            std::uint32_t mtu = 1500;
            std::memcpy(info, &mtu, 4);
            break;
        }
        case 5: {
            std::uint32_t link = 1;
            std::memcpy(info, &link, 4);
            break;
        }
        case 12: {
            std::uint32_t cfg = 1;
            std::memcpy(info, &cfg, 4);
            break;
        }
        case 14:
            std::snprintf(static_cast<char*>(info), 16, "127.0.0.1");
            break;
        case 15:
            std::snprintf(static_cast<char*>(info), 16, "255.0.0.0");
            break;
        case 16:
            std::snprintf(static_cast<char*>(info), 16, "127.0.0.1");
            break;
        case 17:
        case 18:
            std::snprintf(static_cast<char*>(info), 16, "0.0.0.0");
            break;
        default:
            break;
    }
    return 0;
}
GUEST_ABI int hle_net_ctl_register(void (*fn)(int, void*), void* arg, int* cid) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    const int id = g_net_cb_next++;
    g_net_cbs[id] = {fn, arg};
    if (cid) {
        *cid = id;
    }
    return 0;
}
GUEST_ABI int hle_net_ctl_unregister(int cid) {
    std::lock_guard<std::mutex> lock(g_net_mu);
    g_net_cbs.erase(cid);
    return 0;
}
GUEST_ABI int hle_net_ctl_check() { return 0; }
GUEST_ABI int hle_net_ctl_nat(void* info) {
    if (info) {
        int stun = 1;
        int nat = 1;
        std::uint32_t addr = bswap32(0x7f000001u);
        auto* p = static_cast<unsigned char*>(info);
        std::memcpy(p, &stun, 4);
        std::memcpy(p + 4, &nat, 4);
        std::memcpy(p + 8, &addr, 4);
    }
    return 0;
}

}  // namespace

void hle_register_net() {
#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
    REG("sceNetInit", hle_net_init);
    REG("sceNetTerm", hle_net_term);
    REG("sceNetErrnoLoc", hle_net_errno_loc);
    REG("sceNetPoolCreate", hle_net_pool_create);
    REG("sceNetPoolDestroy", hle_net_pool_destroy);
    REG("sceNetSocket", hle_net_socket);
    REG("sceNetSocketClose", hle_net_socket_close);
    REG("sceNetSocketAbort", hle_net_socket_abort);
    REG("sceNetBind", hle_net_bind);
    REG("sceNetListen", hle_net_listen);
    REG("sceNetAccept", hle_net_accept);
    REG("sceNetConnect", hle_net_connect);
    REG("sceNetShutdown", hle_net_shutdown);
    REG("sceNetSetsockopt", hle_net_setsockopt);
    REG("sceNetGetsockopt", hle_net_getsockopt);
    REG("sceNetGetsockname", hle_net_getsockname);
    REG("sceNetRecv", hle_net_recv);
    REG("sceNetSend", hle_net_send);
    REG("sceNetRecvfrom", hle_net_recvfrom);
    REG("sceNetSendto", hle_net_sendto);
    REG("sceNetHtons", hle_net_htons);
    REG("sceNetNtohs", hle_net_ntohs);
    REG("sceNetHtonl", hle_net_htonl);
    REG("sceNetNtohl", hle_net_ntohl);
    REG("sceNetInetPton", hle_net_inet_pton);
    REG("sceNetInetNtop", hle_net_inet_ntop);
    REG("sceNetEpollCreate", hle_net_epoll_create);
    REG("sceNetEpollDestroy", hle_net_epoll_destroy);
    REG("sceNetEpollControl", hle_net_epoll_control);
    REG("sceNetEpollWait", hle_net_epoll_wait);
    REG("sceNetEpollAbort", hle_net_epoll_abort);
    REG("sceNetResolverCreate", hle_net_resolver_create);
    REG("sceNetResolverDestroy", hle_net_resolver_destroy);
    REG("sceNetResolverStartNtoa", hle_net_resolver_ntoa);
    REG("sceNetResolverStartAton", hle_net_resolver_aton);
    REG("sceNetCtlGetState", hle_net_ctl_get_state);
    REG("sceNetCtlGetInfo", hle_net_ctl_get_info);
    REG("sceNetCtlRegisterCallback", hle_net_ctl_register);
    REG("sceNetCtlUnregisterCallback", hle_net_ctl_unregister);
    REG("sceNetCtlCheckCallback", hle_net_ctl_check);
    REG("sceNetCtlGetNatInfo", hle_net_ctl_nat);
#undef REG
}

// --- The reflexive address and hole punching ---------------------------------

bool hle_net_p2p_stun(const char* host, std::uint16_t sport, int timeout_ms, std::uint32_t* mapped_addr,
                      std::uint16_t* mapped_port, net::stun::Relay* relay) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host, nullptr, &hints, &res) != 0 || !res) {
        host_log("stun: cannot resolve %s", host);
        return false;
    }
    sockaddr_in sa{};
    std::memcpy(&sa, res->ai_addr, sizeof(sa));
    sa.sin_port = bswap16(sport);
    freeaddrinfo(res);
    std::shared_ptr<P2pPort> port;
    {
        std::lock_guard<std::mutex> lock(g_net_mu);
        port = p2p_port_open(static_cast<std::uint16_t>(net::p2p_port()));
    }
    if (!port) {
        host_log("stun: no P2P port %d to ask from", net::p2p_port());
        return false;
    }
    // With `relay`, the HELLO: the token this server gave, else zeros.
    std::uint8_t token[net::stun::kTokenLen] = {};
    bool have_token = false;
    if (relay) {
        std::lock_guard<std::mutex> rl(g_relay.mu);
        if (g_relay.on && g_relay.server == sa.sin_addr.s_addr && g_relay.stun_port == sa.sin_port) {
            std::memcpy(token, g_relay.token, sizeof(token));
            have_token = true;
        }
    }
    std::uint8_t req[net::stun::kMaxRequest];
    std::unique_lock<std::mutex> lk(port->mu);
    if (port->stun_pending) return false;  // one at a time
    const std::size_t req_len =
        net::stun::build_binding_request(req, port->stun_txid, relay != nullptr, have_token ? token : nullptr);
    port->stun_pending = true;
    port->stun_done = false;
    lk.unlock();
    (void)sock_sendto(port->fd, req, req_len, true, &sa);
    lk.lock();
    port->stun_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return port->stun_done; });
    const bool ok = port->stun_done;
    port->stun_pending = false;
    if (ok) {
        if (mapped_addr) *mapped_addr = port->stun_addr;
        if (mapped_port) *mapped_port = port->stun_port;
        if (relay) {
            *relay = port->stun_relay;
            std::lock_guard<std::mutex> rl(g_relay.mu);
            const bool was = g_relay.on;
            const std::uint16_t was_vport = g_relay.vport;
            if (relay->present) {
                g_relay.on = true;
                g_relay.server = sa.sin_addr.s_addr;
                g_relay.stun_port = sa.sin_port;
                std::memcpy(g_relay.token, relay->token, sizeof(g_relay.token));
                g_relay.vport = relay->vport;
            } else {
                g_relay.on = false;  // a server without the relay: datagrams go as addressed
            }
            if (g_relay.on != was || g_relay.vport != was_vport) {
                char ip[32];
                write_ipv4(ip, sizeof(ip), sa.sin_addr.s_addr);
                if (g_relay.on) {
                    host_log("net: relay on: %s:%u is our port %u; datagrams for its other ports go through it", ip,
                             sport, g_relay.vport);
                } else {
                    host_log("net: relay off (%s:%u offers none)", ip, sport);
                }
            }
        }
    }
    return ok;
}

bool hle_net_p2p_relay(std::uint32_t* server, std::uint16_t* vport) {
    std::lock_guard<std::mutex> lk(g_relay.mu);
    if (server) *server = g_relay.server;
    if (vport) *vport = g_relay.vport;
    return g_relay.on;
}

void hle_net_p2p_punch(const char* label, std::uint32_t addr, std::uint16_t port_host, std::uint32_t local_addr,
                       std::uint16_t local_port) {
    std::shared_ptr<P2pPort> port;
    {
        std::lock_guard<std::mutex> lock(g_net_mu);
        auto it = g_p2p_ports.find(static_cast<std::uint16_t>(net::p2p_port()));
        if (it != g_p2p_ports.end()) port = it->second;
    }
    if (!port || !addr || !port_host) return;
    {
        std::uint8_t token[net::stun::kTokenLen];
        sockaddr_in stun_sa{};
        if (relay_target(addr, bswap16(port_host), token, &stun_sa)) return;  // the relay needs no hole
    }
    std::vector<sockaddr_in> targets;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = addr;
    a.sin_port = bswap16(port_host);
    targets.push_back(a);
    if (local_addr && local_port && (local_addr != addr || local_port != port_host)) {
        sockaddr_in b{};
        b.sin_family = AF_INET;
        b.sin_addr.s_addr = local_addr;
        b.sin_port = bswap16(local_port);
        targets.push_back(b);
    }
    const std::string name = label ? label : "peer";
    std::thread([port, targets, name] {
        const auto start = std::chrono::steady_clock::now();
        int sent = 0;
        bool heard = false;
        // Anything from either address since the start: the peer is through.
        auto heard_since_start = [&] {
            std::lock_guard<std::mutex> lk(port->mu);
            for (const sockaddr_in& t : targets) {
                auto it = port->heard.find(source_key(t.sin_addr.s_addr, t.sin_port));
                if (it != port->heard.end() && it->second >= start) return true;
            }
            return false;
        };
        for (int i = 0; i < 40 && !port->stop.load(std::memory_order_relaxed); ++i) {
            for (const sockaddr_in& t : targets) {
                if (sock_sendto(port->fd, kProbe, sizeof(kProbe), true, &t) >= 0) ++sent;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (heard_since_start()) {
                heard = true;
                break;
            }
        }
        char ip[32];
        write_ipv4(ip, sizeof(ip), targets[0].sin_addr.s_addr);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        host_log("net: punch %s %s:%u%s: %d probes, %s after %lld ms", name.c_str(), ip, bswap16(targets[0].sin_port),
                 targets.size() > 1 ? " (+local)" : "", sent, heard ? "peer heard" : "nothing heard", static_cast<long long>(ms));
    }).detach();
}
