// The connection test's exchange (net/nat_check.h): sockets alone - no
// configuration, log or server calls - so tests/nat_check_test.cpp runs it
// against a test server of its own on loopback.
#include "net/nat_check.h"
#include "net/stun.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iterator>

namespace net::nat {

namespace {

#if defined(_WIN32)
using Sock = SOCKET;
using AddrLen = int;
constexpr Sock kNoSock = INVALID_SOCKET;
#ifndef SIO_UDP_CONNRESET
#define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR, 12)
#endif
int last_error() { return WSAGetLastError(); }
bool in_use(int e) { return e == WSAEADDRINUSE || e == WSAEACCES; }
void close_sock(Sock s) { closesocket(s); }
std::string error_text(int e) { return "Windows sockets error " + std::to_string(e); }
// Up to `ms` for a datagram.
bool readable(Sock s, int ms) {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(s, &set);
    timeval tv{ms / 1000, (ms % 1000) * 1000};
    return ::select(0, &set, nullptr, nullptr, &tv) > 0;
}
#else
using Sock = int;
using AddrLen = socklen_t;
constexpr Sock kNoSock = -1;
int last_error() { return errno; }
bool in_use(int e) { return e == EADDRINUSE || e == EACCES; }
void close_sock(Sock s) { ::close(s); }
std::string error_text(int e) { return std::strerror(e); }
bool readable(Sock s, int ms) {
    pollfd p{};
    p.fd = s;
    p.events = POLLIN;
    return ::poll(&p, 1, ms) > 0;
}
#endif

// When each request goes out (ms from the start), while it is unanswered.
constexpr int kPlainAt[] = {0, 300, 800, 1600};
// The port check's: again while its second answer has not come, which gets a
// second after the last.
constexpr int kChangeAt[] = {0, 500, 1200};
constexpr int kSecondAnswerWait = 1000;

struct Request {
    std::uint8_t txid[stun::kTxid] = {};
    std::uint8_t data[stun::kMaxRequest] = {};
    std::size_t len = 0;
    std::uint16_t to = 0;  // the test port, host order
    int sends = 0;
    bool answered = false;
};

}  // namespace

ProbeResult probe(const ProbeParams& p) {
    using clock = std::chrono::steady_clock;
    ProbeResult r;
    r.local_port = p.local_port;
    r.at.resize(p.ports.size());
    if (p.ports.empty()) {
        r.error = ProbeResult::Error::Socket;
        r.error_text = "no test ports";
        return r;
    }
#if defined(_WIN32)
    WSADATA wsa{};
    const bool wsa_ok = ::WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
    struct Cleanup {
        bool on;
        ~Cleanup() {
            if (on) ::WSACleanup();
        }
    } cleanup{wsa_ok};
#endif
    const Sock s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == kNoSock) {
        r.error = ProbeResult::Error::Socket;
        r.error_text = error_text(last_error());
        return r;
    }
#if defined(_WIN32)
    // The port is this test's alone, as the game's is (hle/net.cpp), and an
    // ICMP error from a closed port does not end the reads.
    BOOL one = TRUE, off = FALSE;
    ::setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&one), sizeof(one));
    DWORD bytes = 0;
    ::WSAIoctl(s, SIO_UDP_CONNRESET, &off, sizeof(off), nullptr, 0, &bytes, nullptr, nullptr);
#endif
    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_port = htons(p.local_port);
    local.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(s, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) != 0) {
        const int e = last_error();
        close_sock(s);
        r.error = in_use(e) ? ProbeResult::Error::PortInUse : ProbeResult::Error::Bind;
        r.error_text = error_text(e);
        return r;
    }

    std::vector<Request> plain(p.ports.size());
    for (std::size_t i = 0; i < plain.size(); ++i) {
        plain[i].len = stun::build_binding_request(plain[i].data, plain[i].txid);
        plain[i].to = p.ports[i];
    }
    Request change;
    change.len = stun::build_change_request(change.data, change.txid, stun::kChangePort);
    change.to = p.ports[0];

    auto send = [&](Request& q) {
        sockaddr_in to{};
        to.sin_family = AF_INET;
        to.sin_addr.s_addr = p.server;
        to.sin_port = htons(q.to);
        if (::sendto(s, reinterpret_cast<const char*>(q.data), static_cast<int>(q.len), 0, reinterpret_cast<const sockaddr*>(&to),
                     sizeof(to)) >= 0)
            ++r.sent;
        ++q.sends;
    };
    // An address this socket has sent to: an answer from there is one the
    // router expects.
    auto sent_to = [&](std::uint32_t ip, std::uint16_t port) {
        if (ip != p.server) return false;
        for (std::uint16_t tp : p.ports) {
            if (tp == port) return true;
        }
        return false;
    };
    const auto t0 = clock::now();
    for (;;) {
        if (p.cancelled && p.cancelled()) {
            r.error = ProbeResult::Error::Cancelled;
            break;
        }
        const int now = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count());
        bool plain_done = true;
        for (Request& q : plain) {
            if (!q.answered && q.sends < static_cast<int>(std::size(kPlainAt)) && now >= kPlainAt[q.sends]) send(q);
            plain_done = plain_done && q.answered;
        }
        // The port check is over once its second answer came, once a server
        // without it has answered, or a second after its last request.
        const bool change_done = r.reached || (r.change_answered && !r.change_offered) ||
                                 (change.sends == static_cast<int>(std::size(kChangeAt)) &&
                                  now >= kChangeAt[std::size(kChangeAt) - 1] + kSecondAnswerWait);
        if (!change_done && change.sends < static_cast<int>(std::size(kChangeAt)) && now >= kChangeAt[change.sends]) send(change);
        if ((plain_done && change_done) || now >= p.deadline_ms) break;
        if (!readable(s, 40)) continue;
        std::uint8_t buf[1500];
        sockaddr_in from{};
        AddrLen from_len = sizeof(from);
        const auto n = ::recvfrom(s, reinterpret_cast<char*>(buf), sizeof(buf), 0, reinterpret_cast<sockaddr*>(&from), &from_len);
        if (n <= 0) continue;
        const auto len = static_cast<std::size_t>(n);
        const std::uint32_t src = from.sin_addr.s_addr;
        const std::uint16_t src_port = ntohs(from.sin_port);
        std::uint32_t addr = 0;
        std::uint16_t port = 0;
        for (std::size_t i = 0; i < plain.size(); ++i) {
            if (!plain[i].answered && stun::parse_binding_response(buf, len, plain[i].txid, &addr, &port)) {
                plain[i].answered = true;
                r.at[i] = Answer{true, addr, port};
            }
        }
        if (stun::parse_binding_response(buf, len, change.txid, &addr, &port)) {
            std::uint16_t other_port = 0;
            if (stun::response_other_address(buf, len, nullptr, &other_port)) {
                // The usual answer, naming where the second one comes from.
                r.change_answered = r.change_offered = true;
                r.offered_port = other_port;
            } else if (sent_to(src, src_port)) {
                // The usual answer of a server without the check.
                r.change_answered = true;
            } else {
                // From an address this socket never sent to: it got in unasked.
                r.reached = true;
                r.reached_from = src_port;
            }
            // Sent to the first test port, its usual answer saw that mapping too.
            if (sent_to(src, src_port) && src_port == p.ports[0] && !plain[0].answered) {
                plain[0].answered = true;
                r.at[0] = Answer{true, addr, port};
            }
        }
    }
    close_sock(s);
    return r;
}

}  // namespace net::nat
