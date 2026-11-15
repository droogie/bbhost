// The setup window's connection test (net/nat_check.h). The exchange itself
// (net/nat_probe.cpp) from a port of its own against a test server of the
// test's on loopback, which answers as the private server's
// server/nat_test.py does - or as a router in between would make it look:
//   - the same public port at both test ports (a router that keeps one port): direct;
//   - another port, or another public address, at the second (one that maps
//     every destination anew): relay;
//   - nothing (UDP blocked): no answer, at the deadline; one port alone:
//     inconclusive;
//   - the port check: OTHER-ADDRESS and the second answer from a third port
//     (open), OTHER-ADDRESS alone (closed: the router dropped the second
//     answer), neither (a server without it: not available);
//   - the first 2.5 s held, as a security program asking about bbhost.exe
//     holds them: the later sends get through, the port check's too;
//   - every send refused by this PC: counted with the first error, over once
//     nothing is left to try;
//   - a lost first answer, the port held by another socket, a cancel.
// Then what the server's /auth/nat_test answers mean, the ports setting, and
// the words the card and the log use: the owner's three headlines, each
// reading of a finished test (no answer kept or not, the server's blocked,
// its answers stopped, every send refused...), no address in the log line.
#include "net/nat_check.h"
#include "net/nat_examples.h"
#include "net/stun.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);  \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

using net::nat::Inbound;
using net::nat::ProbeResult;
using net::nat::SaveReply;
using net::nat::Verdict;

std::uint16_t rd16(const std::uint8_t* p) { return static_cast<std::uint16_t>((p[0] << 8) | p[1]); }
std::uint32_t rd32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) | p[3];
}
void put16(std::vector<std::uint8_t>& v, std::uint16_t x) {
    v.push_back(static_cast<std::uint8_t>(x >> 8));
    v.push_back(static_cast<std::uint8_t>(x));
}
void put32(std::vector<std::uint8_t>& v, std::uint32_t x) {
    put16(v, static_cast<std::uint16_t>(x >> 16));
    put16(v, static_cast<std::uint16_t>(x));
}

int udp_bound(std::uint16_t port) {
    const int s = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
        ::close(s);
        return -1;
    }
    return s;
}
std::uint16_t port_of(int s) {
    sockaddr_in a{};
    socklen_t n = sizeof(a);
    ::getsockname(s, reinterpret_cast<sockaddr*>(&a), &n);
    return ntohs(a.sin_port);
}
std::uint16_t free_port() {
    const int s = udp_bound(0);
    const std::uint16_t p = port_of(s);
    ::close(s);
    return p;
}

// The test server: two test ports and the port check's, on loopback.
struct Server {
    enum class Mapping { Cone, NewPort, NewAddress, Silent, FirstOnly };
    enum class Check { Open, Closed, None };
    Mapping mapping = Mapping::Cone;
    Check check = Check::Open;
    int drop_first_plain_a = 0;  // the first test port's plain answers to drop
    bool xor_only = false;       // XOR-MAPPED-ADDRESS alone, as RFC 5389 servers answer
    int hold_ms = 0;             // nothing answered until then (a security program holding the first datagrams)
    int fd[3] = {-1, -1, -1};    // A, B, R
    std::uint16_t port[3] = {};
    std::atomic<bool> stop{false};
    std::atomic<int> requests{0}, change_requests{0}, second_answers{0};
    std::chrono::steady_clock::time_point started;
    std::thread thread;

    void begin() {
        for (int i = 0; i < 3; ++i) {
            fd[i] = udp_bound(0);
            port[i] = port_of(fd[i]);
        }
        started = std::chrono::steady_clock::now();
        thread = std::thread([this] { serve(); });
    }
    ~Server() {
        stop = true;
        if (thread.joinable()) thread.join();
        for (int f : fd) {
            if (f >= 0) ::close(f);
        }
    }

    std::vector<std::uint8_t> answer(const std::uint8_t* req, std::uint32_t ip_host, std::uint16_t mport, bool other) {
        std::vector<std::uint8_t> body;
        if (!xor_only) {
            put16(body, 0x0001);
            put16(body, 8);
            body.push_back(0);
            body.push_back(1);
            put16(body, mport);
            put32(body, ip_host);
        }
        put16(body, 0x0020);
        put16(body, 8);
        body.push_back(0);
        body.push_back(1);
        put16(body, static_cast<std::uint16_t>(mport ^ 0x2112));
        put32(body, ip_host ^ 0x2112A442u);
        if (other) {
            put16(body, 0x802C);
            put16(body, 8);
            body.push_back(0);
            body.push_back(1);
            put16(body, port[2]);
            put32(body, 0x7f000001u);
        }
        std::vector<std::uint8_t> out;
        put16(out, 0x0101);
        put16(out, static_cast<std::uint16_t>(body.size()));
        out.insert(out.end(), req + 4, req + 20);
        out.insert(out.end(), body.begin(), body.end());
        return out;
    }

    void serve() {
        int plain_a = 0;
        while (!stop) {
            pollfd p[2] = {{fd[0], POLLIN, 0}, {fd[1], POLLIN, 0}};
            if (::poll(p, 2, 20) <= 0) continue;
            for (int i = 0; i < 2; ++i) {
                if (!(p[i].revents & POLLIN)) continue;
                std::uint8_t buf[1500];
                sockaddr_in from{};
                socklen_t fl = sizeof(from);
                const ssize_t n = ::recvfrom(fd[i], buf, sizeof(buf), 0, reinterpret_cast<sockaddr*>(&from), &fl);
                if (n < 20 || rd16(buf) != 0x0001) continue;
                ++requests;
                // CHANGE-REQUEST's flags
                std::uint32_t flags = 0;
                for (std::size_t at = 20; at + 8 <= static_cast<std::size_t>(n);) {
                    const std::uint16_t type = rd16(buf + at), alen = rd16(buf + at + 2);
                    if (type == 0x0003 && alen == 4) flags = rd32(buf + at + 4);
                    at += 4 + ((alen + 3u) & ~3u);
                }
                const bool change = flags & net::stun::kChangePort;
                if (change) ++change_requests;
                if (std::chrono::steady_clock::now() - started < std::chrono::milliseconds(hold_ms)) continue;
                if (mapping == Mapping::Silent || (mapping == Mapping::FirstOnly && i == 1)) continue;
                if (i == 0 && !change && plain_a++ < drop_first_plain_a) continue;
                std::uint32_t ip = ntohl(from.sin_addr.s_addr);
                std::uint16_t mport = ntohs(from.sin_port);
                if (i == 1 && mapping == Mapping::NewPort) mport = static_cast<std::uint16_t>(mport + 1000);
                if (i == 1 && mapping == Mapping::NewAddress) ip = 0xCB007107u;  // 203.0.113.7
                const bool other = change && check != Check::None;
                const std::vector<std::uint8_t> a = answer(buf, ip, mport, other);
                ::sendto(fd[i], a.data(), a.size(), 0, reinterpret_cast<sockaddr*>(&from), fl);
                if (other && check == Check::Open) {
                    const std::vector<std::uint8_t> b = answer(buf, ip, mport, false);
                    ::sendto(fd[2], b.data(), b.size(), 0, reinterpret_cast<sockaddr*>(&from), fl);
                    ++second_answers;
                }
            }
        }
    }
};

ProbeResult run(Server& s, std::uint16_t local, int deadline_ms = 6000, std::function<bool()> cancelled = {}) {
    net::nat::ProbeParams p;
    p.server = htonl(INADDR_LOOPBACK);
    p.ports = {s.port[0], s.port[1]};
    p.local_port = local;
    p.deadline_ms = deadline_ms;
    p.cancelled = std::move(cancelled);
    return net::nat::probe(p);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void exchange_cases() {
    {   // a router that keeps one port, and lets the port check's second answer in
        Server s;
        s.begin();
        const std::uint16_t local = free_port();
        const auto t0 = std::chrono::steady_clock::now();
        const ProbeResult r = run(s, local);
        const double took = ms_since(t0);
        CHECK(r.error == ProbeResult::Error::None);
        CHECK(r.local_port == local);
        CHECK(r.at.size() == 2 && r.at[0].got && r.at[1].got);
        CHECK(r.at[0].port == local && r.at[1].port == local);
        CHECK(r.at[0].addr == htonl(INADDR_LOOPBACK));
        CHECK(r.local_addr == htonl(INADDR_LOOPBACK));  // this PC's own address toward the server: loopback's here
        CHECK(!net::nat::behind_router(r));             // ... the address the server saw
        CHECK(net::nat::verdict_of(r) == Verdict::Direct);
        CHECK(r.change_answered && r.change_offered && r.offered_port == s.port[2]);
        CHECK(r.reached && r.reached_from == s.port[2]);
        CHECK(net::nat::inbound_of(r) == Inbound::Open);
        CHECK(took < 1000);  // everything in: no waiting out the deadline
        CHECK(s.change_requests == 1 && s.second_answers == 1);
        CHECK(r.sent == 3 && r.refused == 0 && r.send_error.empty());
        CHECK(net::nat::public_ports(r) == "public port " + std::to_string(local) + " at both test ports, the game's own");
    }
    {   // this PC's own address toward the server is the system's choice for that destination, not the server's
        // (127.0.0.2 is reached from 127.0.0.1); nothing answers there
        net::nat::ProbeParams p;
        p.server = htonl(0x7F000002u);
        p.ports = {3479, 3480};
        p.local_port = free_port();
        p.deadline_ms = 100;
        const ProbeResult r = net::nat::probe(p);
        CHECK(r.local_addr == htonl(INADDR_LOOPBACK));
        CHECK(net::nat::ipv4_text(r.local_addr) == "127.0.0.1");
    }
    {   // a new port per destination; the router drops the second answer
        Server s;
        s.mapping = Server::Mapping::NewPort;
        s.check = Server::Check::Closed;
        s.begin();
        const std::uint16_t local = free_port();
        const auto t0 = std::chrono::steady_clock::now();
        const ProbeResult r = run(s, local);
        const double took = ms_since(t0);
        CHECK(net::nat::verdict_of(r) == Verdict::Relay);
        CHECK(r.at[1].port == static_cast<std::uint16_t>(local + 1000));
        CHECK(r.change_offered && !r.reached);
        CHECK(net::nat::inbound_of(r) == Inbound::Closed);
        CHECK(took > 2000 && took < 3000);  // a second after the port check's last request, before the deadline
        CHECK(s.change_requests == 3);       // asked three times for the second answer
        CHECK(net::nat::public_ports(r) ==
              "public ports " + std::to_string(local) + " and " + std::to_string(static_cast<std::uint16_t>(local + 1000)));
    }
    {   // another public address at the second test port (a carrier-grade NAT's pool), XOR-MAPPED-ADDRESS alone
        Server s;
        s.mapping = Server::Mapping::NewAddress;
        s.xor_only = true;
        s.begin();
        const ProbeResult r = run(s, free_port());
        CHECK(r.at[0].port == r.at[1].port && r.at[0].addr != r.at[1].addr);
        CHECK(net::nat::verdict_of(r) == Verdict::Relay);
        CHECK(net::nat::public_ports(r).find("from different public addresses") != std::string::npos);
    }
    {   // nothing answers: no answer, at the deadline
        Server s;
        s.mapping = Server::Mapping::Silent;
        s.begin();
        const auto t0 = std::chrono::steady_clock::now();
        const ProbeResult r = run(s, free_port());
        const double took = ms_since(t0);
        CHECK(r.error == ProbeResult::Error::None);
        CHECK(net::nat::verdict_of(r) == Verdict::Blocked);
        CHECK(net::nat::inbound_of(r) == Inbound::NotTested);
        CHECK(took >= 5900 && took < 6600);
        CHECK(s.requests == 6 + 6 + 5);  // each request six times, the port check's five
        CHECK(r.sent == 17 && r.refused == 0);
        CHECK(net::nat::public_ports(r).empty());
    }
    {   // the first 2.5 s held (a security program asking about bbhost.exe): the sends at 2.8 s get through, the
        // port check's with them
        Server s;
        s.hold_ms = 2500;
        s.begin();
        const std::uint16_t local = free_port();
        const auto t0 = std::chrono::steady_clock::now();
        const ProbeResult r = run(s, local);
        const double took = ms_since(t0);
        CHECK(net::nat::verdict_of(r) == Verdict::Direct);
        CHECK(r.at[0].port == local && r.at[1].port == local);
        CHECK(net::nat::inbound_of(r) == Inbound::Open);
        CHECK(took > 2700 && took < 3600);  // over once the held-up answers are in
        CHECK(s.change_requests == 4);
    }
    {   // held, and the router drops the second answer: closed, a second after the port check's request got through
        Server s;
        s.hold_ms = 2500;
        s.check = Server::Check::Closed;
        s.begin();
        const auto t0 = std::chrono::steady_clock::now();
        const ProbeResult r = run(s, free_port());
        const double took = ms_since(t0);
        CHECK(net::nat::verdict_of(r) == Verdict::Direct);
        CHECK(net::nat::inbound_of(r) == Inbound::Closed);
        CHECK(took > 3700 && took < 4600);
        CHECK(s.change_requests == 4);  // its fifth only while not even the usual answer has come
    }
    {   // this PC refuses every send (here the limited broadcast address, without SO_BROADCAST): counted, the
        // first error kept, over once nothing is left to try
        net::nat::ProbeParams p;
        p.server = htonl(INADDR_BROADCAST);
        p.ports = {3479, 3480};
        p.local_port = free_port();
        const auto t0 = std::chrono::steady_clock::now();
        const ProbeResult r = net::nat::probe(p);
        const double took = ms_since(t0);
        CHECK(r.error == ProbeResult::Error::None);
        CHECK(r.sent == 0 && r.refused == 6 + 6 + 5);
        CHECK(r.send_error_code != 0 && r.send_error == std::strerror(r.send_error_code));
        CHECK(r.send_denied == (r.send_error_code == EACCES || r.send_error_code == EPERM));
        CHECK(net::nat::nothing_sent(r));
        CHECK(net::nat::verdict_of(r) == Verdict::Blocked);
        CHECK(took > 4100 && took < 5000);  // the last refused at 4.2 s: no wait for the deadline
    }
    {   // one test port answers
        Server s;
        s.mapping = Server::Mapping::FirstOnly;
        s.begin();
        const ProbeResult r = run(s, free_port());
        CHECK(net::nat::verdict_of(r) == Verdict::Inconclusive);
    }
    {   // a server without the port check: it answers the request as any other
        Server s;
        s.check = Server::Check::None;
        s.begin();
        const auto t0 = std::chrono::steady_clock::now();
        const ProbeResult r = run(s, free_port());
        CHECK(net::nat::verdict_of(r) == Verdict::Direct);
        CHECK(r.change_answered && !r.change_offered && !r.reached);
        CHECK(net::nat::inbound_of(r) == Inbound::Unavailable);
        CHECK(ms_since(t0) < 1000);  // no wait for an answer that never comes
        CHECK(s.change_requests == 1);
    }
    {   // every plain answer of the first test port lost: the port check's usual answer, from there, fills it in
        Server s;
        s.drop_first_plain_a = 6;
        s.begin();
        const std::uint16_t local = free_port();
        const auto t0 = std::chrono::steady_clock::now();
        const ProbeResult r = run(s, local);
        CHECK(r.at[0].got && r.at[0].port == local);
        CHECK(net::nat::verdict_of(r) == Verdict::Direct);
        CHECK(ms_since(t0) < 1500);  // at once, not at a retry
    }
    {   // the port held by another socket: nothing sent
        Server s;
        s.begin();
        const int holder = udp_bound(0);
        const std::uint16_t taken = port_of(holder);
        net::nat::ProbeParams p;
        p.server = htonl(INADDR_LOOPBACK);
        p.ports = {s.port[0], s.port[1]};
        p.local_port = taken;
        // INADDR_ANY over a loopback-bound port: the same port, refused
        const ProbeResult r = net::nat::probe(p);
        CHECK(r.error == ProbeResult::Error::PortInUse);
        CHECK(net::nat::verdict_of(r) == Verdict::None);
        CHECK(s.requests == 0);
        // the P2P port setting asks the same before it takes a port
        CHECK(!net::nat::port_taken(taken).empty());
        ::close(holder);
        CHECK(net::nat::port_taken(taken).empty());
        // a holder that would share it (SO_REUSEADDR): never shared from here
        const int sharer = ::socket(AF_INET, SOCK_DGRAM, 0);
        const int one = 1;
        ::setsockopt(sharer, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_ANY);
        CHECK(::bind(sharer, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0);
        p.local_port = port_of(sharer);
        CHECK(net::nat::probe(p).error == ProbeResult::Error::PortInUse);
        CHECK(s.requests == 0);
        CHECK(!net::nat::port_taken(p.local_port).empty());
        ::close(sharer);
    }
    {   // cancelled: over at once
        Server s;
        s.mapping = Server::Mapping::Silent;
        s.begin();
        std::atomic<bool> stop{false};
        std::thread t([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            stop = true;
        });
        const auto t0 = std::chrono::steady_clock::now();
        const ProbeResult r = run(s, free_port(), 3000, [&] { return stop.load(); });
        t.join();
        CHECK(r.error == ProbeResult::Error::Cancelled);
        CHECK(ms_since(t0) < 600);
    }
}

void stun_cases() {
    std::uint8_t req[net::stun::kMaxRequest], txid[net::stun::kTxid];
    const std::size_t n = net::stun::build_change_request(req, txid, net::stun::kChangePort);
    CHECK(n == 28);
    CHECK(rd16(req) == 0x0001 && rd16(req + 2) == 8);
    CHECK(std::memcmp(req + 4, txid, 16) == 0 && rd32(txid) == 0x2112A442u);
    CHECK(rd16(req + 20) == 0x0003 && rd16(req + 22) == 4 && rd32(req + 24) == 2);
    // OTHER-ADDRESS read; a response without it
    std::vector<std::uint8_t> resp;
    put16(resp, 0x0101);
    put16(resp, 12);
    resp.insert(resp.end(), txid, txid + 16);
    put16(resp, 0x802C);
    put16(resp, 8);
    resp.push_back(0);
    resp.push_back(1);
    put16(resp, 3481);
    put32(resp, 0xC6336401u);  // 198.51.100.1
    std::uint32_t addr = 0;
    std::uint16_t port = 0;
    CHECK(net::stun::response_other_address(resp.data(), resp.size(), &addr, &port));
    CHECK(port == 3481 && ntohl(addr) == 0xC6336401u);
    resp[20] = 0x80;
    resp[21] = 0x2B;  // RESPONSE-ORIGIN instead
    CHECK(!net::stun::response_other_address(resp.data(), resp.size(), &addr, &port));
    CHECK(!net::stun::response_other_address(resp.data(), 10, &addr, &port));
}

ProbeResult made(std::vector<net::nat::Answer> at, bool answered = false, bool offered = false, bool reached = false) {
    ProbeResult r;
    r.local_port = 9307;
    r.at = std::move(at);
    r.change_answered = answered;
    r.change_offered = offered;
    r.reached = reached;
    return r;
}

void verdict_cases() {
    using net::nat::Answer;
    const Answer a{true, 0x0100000a, 9307}, b{true, 0x0100000a, 40002}, c{true, 0x0200000a, 9307}, none{};
    CHECK(net::nat::verdict_of(made({a, a})) == Verdict::Direct);
    CHECK(net::nat::verdict_of(made({a, b})) == Verdict::Relay);
    CHECK(net::nat::verdict_of(made({a, c})) == Verdict::Relay);
    CHECK(net::nat::verdict_of(made({none, none})) == Verdict::Blocked);
    CHECK(net::nat::verdict_of(made({a, none})) == Verdict::Inconclusive);
    CHECK(net::nat::verdict_of(made({none, a})) == Verdict::Inconclusive);
    CHECK(net::nat::verdict_of(made({a, a, a})) == Verdict::Direct);
    CHECK(net::nat::verdict_of(made({a, a, b})) == Verdict::Relay);
    ProbeResult e = made({a, a});
    e.error = ProbeResult::Error::PortInUse;
    CHECK(net::nat::verdict_of(e) == Verdict::None && net::nat::inbound_of(e) == Inbound::NotTested);
    CHECK(net::nat::inbound_of(made({a, a}, true, true, true)) == Inbound::Open);
    CHECK(net::nat::inbound_of(made({a, a}, false, false, true)) == Inbound::Open);  // the second answer before the first
    CHECK(net::nat::inbound_of(made({a, a}, true, true, false)) == Inbound::Closed);
    CHECK(net::nat::inbound_of(made({a, a}, true, false, false)) == Inbound::Unavailable);
    CHECK(net::nat::inbound_of(made({none, none})) == Inbound::NotTested);
    for (Verdict v : {Verdict::Direct, Verdict::Relay, Verdict::Blocked})
        CHECK(net::nat::verdict_from_word(net::nat::verdict_word(v)) == v);
    CHECK(net::nat::verdict_from_word("maybe") == Verdict::None);
    CHECK(std::string(net::nat::inbound_word(Inbound::Open)) == "open");
    CHECK(std::string(net::nat::inbound_word(Inbound::Unavailable)).empty());  // not sent to the server
    CHECK(net::nat::inbound_from_word("closed") == Inbound::Closed && net::nat::inbound_from_word("") == Inbound::NotTested);
}

void save_reply_cases() {
    using State = SaveReply::State;
    SaveReply r = net::nat::read_save_reply(200, R"({"ResKind":0,"Verdict":"relay","TestedAt":1794268800.5})", "");
    CHECK(r.state == State::Saved && r.verdict == Verdict::Relay && r.tested_at == 1794268800.5);
    r = net::nat::read_save_reply(200, R"({"ResKind":0,"Verdict":null,"Message":"only one test port answered; run the test again"})", "");
    CHECK(r.state == State::Unsaved && r.verdict == Verdict::None && r.message.find("run the test again") != std::string::npos);
    r = net::nat::read_save_reply(429, R"({"ResKind":1,"Message":"too many tests; try again later"})", "");
    CHECK(r.state == State::Limited && r.status == 429);
    r = net::nat::read_save_reply(200, R"({"ResKind":1,"Message":"no such token"})", "");
    CHECK(r.state == State::Refused);
    r = net::nat::read_save_reply(200, R"({"ResKind":1,"Message":"the connection test is not available right now"})", "");
    CHECK(r.state == State::Unsaved && r.message == "the connection test is not available right now");
    r = net::nat::read_save_reply(404, R"({"detail":"Not Found"})", "");
    CHECK(r.state == State::NoService && r.message == "HTTP 404");
    r = net::nat::read_save_reply(502, "<html>Bad Gateway</html>", "");
    CHECK(r.state == State::NoService);
    r = net::nat::read_save_reply(0, "", "Couldn't connect to server");
    CHECK(r.state == State::NoAnswer && r.message == "Couldn't connect to server" && r.status == 0);
    r = net::nat::read_save_reply(200, R"({"ResKind":0,"Verdict":"sideways"})", "");
    CHECK(r.state == State::Unsaved);
}

void ports_cases() {
    using V = std::vector<std::uint16_t>;
    CHECK(net::nat::parse_ports("3479,3480") == (V{3479, 3480}));
    CHECK(net::nat::parse_ports(" 3679 , 3680 ") == (V{3679, 3680}));
    CHECK(net::nat::parse_ports("3479,3480,3481") == (V{3479, 3480, 3481}));
    CHECK(net::nat::parse_ports("") == (V{3479, 3480}));
    CHECK(net::nat::parse_ports("3479") == (V{3479, 3480}));
    CHECK(net::nat::parse_ports("3479,3479") == (V{3479, 3480}));
    CHECK(net::nat::parse_ports("a,b") == (V{3479, 3480}));
    CHECK(net::nat::parse_ports("70000,3480") == (V{3479, 3480}));
    CHECK(net::nat::parse_ports("0,3480") == (V{3479, 3480}));
}

bool has(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

void words_cases() {
    CHECK(net::nat::ago(1000, 1030) == "just now");
    CHECK(net::nat::ago(1000, 1000 + 300) == "5 minutes ago");
    CHECK(net::nat::ago(1000, 1000 + 3 * 3600) == "3 hours ago");
    CHECK(net::nat::ago(1000, 1000 + 2 * 86400) == "2 days ago");
    CHECK(net::nat::ipv4_text(net::nat::ipv4_net(10, 0, 0, 23)) == "10.0.0.23");
    CHECK(net::nat::ipv4_net(127, 0, 0, 1) == htonl(INADDR_LOOPBACK));

    // The log line: the verdict, the port check, the port, what was sent and refused, whether kept - no address,
    // this PC's own on its network neither.
    const std::regex dotted(R"(\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)");
    net::nat::Outcome o;
    o.probe = made({{true, htonl(0xCB007107u), 9307}, {true, htonl(0xCB007107u), 9307}}, true, true, true);
    o.probe.sent = 3;
    o.probe.local_addr = net::nat::ipv4_net(10, 0, 0, 23);
    o.local = net::nat::verdict_of(o.probe);
    o.inbound = net::nat::inbound_of(o.probe);
    o.signed_in = true;
    o.save = net::nat::read_save_reply(200, R"({"ResKind":0,"Verdict":"direct","TestedAt":5})", "");
    std::string line = net::nat::log_line(o);
    CHECK(line == "connection test: direct (public port 9307 at both test ports, the game's own); incoming open; UDP port 9307; sent 3, "
                  "refused 0; saved to the account");
    CHECK(!std::regex_search(line, dotted));
    o.save = net::nat::read_save_reply(200, R"({"ResKind":0,"Verdict":"relay","TestedAt":5})", "");
    CHECK(net::nat::log_line(o).find("saved to the account - the server says relay") != std::string::npos);
    o.signed_in = false;
    o.save = SaveReply{};
    line = net::nat::log_line(o);
    CHECK(line.find("not saved (not signed in)") != std::string::npos && !std::regex_search(line, dotted));
    o.signed_in = true;
    o.save = net::nat::read_save_reply(429, R"({"ResKind":1,"Message":"too many tests; try again later"})", "");
    CHECK(net::nat::log_line(o).find("10 tests an hour") != std::string::npos);
    o.probe = made({{}, {}});
    o.probe.sent = 17;
    o.local = net::nat::verdict_of(o.probe);
    o.inbound = net::nat::inbound_of(o.probe);
    CHECK(net::nat::log_line(o).rfind("connection test: blocked (no answer from the test ports); incoming not checked; UDP port 9307; "
                                      "sent 17, refused 0; not saved (the server takes 10 tests",
                                      0) == 0);
    // every send refused: the count and the first refusal's words; the server not asked
    o.probe.sent = 0;
    o.probe.refused = 17;
    o.probe.send_error_code = EPERM;
    o.probe.send_error = "Operation not permitted";
    o.save = SaveReply{};
    line = net::nat::log_line(o);
    CHECK(line == "connection test: blocked (no answer from the test ports); incoming not checked; UDP port 9307; sent 0, refused 17 "
                  "(Operation not permitted); not saved (nothing was sent)");
    CHECK(!std::regex_search(line, dotted));
    o.probe.sent = 12;
    o.probe.refused = 5;
    CHECK(net::nat::log_line(o).find("; sent 12, refused 5 (Operation not permitted); not saved (the server did not answer") !=
          std::string::npos);
    o.probe.error = ProbeResult::Error::PortInUse;
    o.probe.error_text = "Address already in use";
    CHECK(net::nat::log_line(o) == "connection test: UDP port 9307 is in use by another program (Address already in use); nothing sent");
    o.resolved = false;
    CHECK(net::nat::log_line(o).find("did not resolve") != std::string::npos);

    // The steps that let other hunters in: the router's forward to this PC's address, the firewall, the test again.
    using V = std::vector<std::string>;
    CHECK(net::nat::forward_steps(9307, "10.0.0.23", true, true) ==
          (V{"In your router's settings, forward UDP port 9307 to this PC (10.0.0.23).",
             "Allow bbhost.exe in Windows Firewall (Private and Public).", "Then press Test again."}));
    CHECK(net::nat::forward_steps(9400, "", true, false) ==
          (V{"In your router's settings, forward UDP port 9400 to this PC.", "If this PC runs a firewall, allow UDP port 9400 in it.",
             "Then press Test again."}));
    // this PC holds the public address itself: no router to forward on
    CHECK(net::nat::forward_steps(9307, "203.0.113.7", false, true) ==
          (V{"Allow bbhost.exe in Windows Firewall (Private and Public).", "Then press Test again."}));
    CHECK(net::nat::incoming_line(Inbound::Open) ==
          "Incoming: other hunters can connect to you directly, even those whose routers change ports.");
    CHECK(net::nat::incoming_line(Inbound::Closed) ==
          "Incoming: hunters whose routers change ports can't connect to you directly, so they reach you through the relay.");
    CHECK(net::nat::incoming_line(Inbound::Unavailable).empty() && net::nat::incoming_line(Inbound::NotTested).empty());
    // behind a router that changes ports, open changes nothing; neither says what the relay does (it may be off)
    CHECK(has(net::nat::incoming_detail(Inbound::Open, 9307, Verdict::Relay), "so other hunters still can't connect directly"));
    CHECK(!has(net::nat::incoming_detail(Inbound::Open, 9307, Verdict::Direct), "can't"));
    CHECK(net::nat::incoming_detail(Inbound::Closed, 9307, Verdict::Relay) == net::nat::incoming_detail(Inbound::Closed, 9307, Verdict::Direct));
    CHECK(!has(net::nat::incoming_detail(Inbound::Closed, 9307, Verdict::Relay), "relay"));
    CHECK(net::nat::incoming_detail(Inbound::Unavailable, 9307, Verdict::Direct) == "Incoming: not checked - this server has no port check.");
    CHECK(net::nat::incoming_detail(Inbound::NotTested, 9307, Verdict::Direct) == "Incoming: not checked - the port check got no answer.");

    // The P2P port setting: 1024 to 65535, none of the server's test ports.
    const std::vector<std::uint16_t> tp{3479, 3480};
    CHECK(net::nat::p2p_port_problem(9307, tp).empty() && net::nat::p2p_port_problem(1024, tp).empty() &&
          net::nat::p2p_port_problem(65535, tp).empty());
    CHECK(net::nat::p2p_port_problem(1023, tp) == "Choose a port from 1024 to 65535.");
    CHECK(net::nat::p2p_port_problem(65536, tp) == "Choose a port from 1024 to 65535.");
    CHECK(net::nat::p2p_port_problem(0, tp) == "Choose a port from 1024 to 65535.");
    CHECK(net::nat::p2p_port_problem(3480, tp) == "UDP 3480 is one of the server's test ports: choose another.");
    CHECK(net::nat::p2p_port_problem(3479, {3679, 3680}).empty());  // a server of your own's ports
}

// A finished test as the job leaves it: what came back read, signed in or
// not, and the server's answer to /auth/nat_test (none when not asked).
net::nat::Outcome finished(ProbeResult p, bool signed_in, const char* reply = nullptr) {
    net::nat::Outcome o;
    o.probe = std::move(p);
    o.local = net::nat::verdict_of(o.probe);
    o.inbound = net::nat::inbound_of(o.probe);
    o.signed_in = signed_in;
    if (reply) o.save = net::nat::read_save_reply(200, reply, "");
    o.at = 1000;
    return o;
}

// The card's context: just after the test, the P2P port 9307, the relay
// switch as given, Windows' words, signed in unless said.
net::nat::CardContext cx(net::nat::RelaySetting relay = net::nat::RelaySetting::Unknown, int port = 9307, bool windows = true,
                         bool signed_in = true) {
    net::nat::CardContext x;
    x.now = 1030;
    x.p2p_port = port;
    x.relay = relay;
    x.windows = windows;
    x.signed_in = signed_in;
    return x;
}

net::nat::Outcome example(const char* name) {
    net::nat::Outcome o;
    CHECK(net::nat::example_outcome(name, 9307, 1000, &o));
    return o;
}

// A relay result from a test that did not ask the server (signed out).
net::nat::Outcome unsigned_relay_out() {
    net::nat::Outcome o = example("relay");
    o.signed_in = false;
    o.save = SaveReply{};
    return o;
}

bool any_has(const std::vector<std::string>& lines, const char* part) {
    for (const std::string& l : lines) {
        if (has(l, part)) return true;
    }
    return false;
}

void reading_cases() {
    using net::nat::Answer;
    using net::nat::Reading;
    const Answer a{true, 0x0100000a, 9307}, b{true, 0x0100000a, 40002}, none{};
    const char* kDirect = R"({"ResKind":0,"Verdict":"direct","TestedAt":5})";
    const char* kRelay = R"({"ResKind":0,"Verdict":"relay","TestedAt":5})";
    const char* kBlocked = R"({"ResKind":0,"Verdict":"blocked","TestedAt":5})";
    ProbeResult back = made({a, a}, true, true, false);  // both answers, the port check closed
    back.sent = 3;
    ProbeResult silent = made({none, none});  // sent, nothing came back
    silent.sent = 17;
    ProbeResult refused = made({none, none});  // nothing went out
    refused.refused = 17;
    refused.send_denied = true;
    ProbeResult one = made({a, none});
    one.sent = 7;
    ProbeResult changes = made({a, b});
    changes.sent = 3;
    // what came back, the server's word, signed in or not: each reading
    CHECK(net::nat::reading_of(finished(silent, false)) == Reading::NoAnswer);
    CHECK(net::nat::reading_of(finished(silent, true)) == Reading::NoAnswer);
    CHECK(net::nat::reading_of(finished(silent, true, kBlocked)) == Reading::ServerBlocked);
    CHECK(net::nat::reading_of(finished(silent, true, kDirect)) == Reading::AnswersStopped);
    CHECK(net::nat::reading_of(finished(silent, true, kRelay)) == Reading::AnswersStopped);
    CHECK(net::nat::reading_of(finished(refused, true)) == Reading::SendRefused);
    CHECK(net::nat::reading_of(finished(refused, false)) == Reading::SendRefused);
    CHECK(net::nat::reading_of(finished(back, true, kBlocked)) == Reading::AddressDiffers);
    CHECK(net::nat::reading_of(finished(back, true, kDirect)) == Reading::Direct);
    CHECK(net::nat::reading_of(finished(back, true, kRelay)) == Reading::Relay);  // the server's word over this PC's
    CHECK(net::nat::reading_of(finished(back, false)) == Reading::Direct);
    CHECK(net::nat::reading_of(finished(one, true, kDirect)) == Reading::Direct);
    CHECK(net::nat::reading_of(finished(one, false)) == Reading::Inconclusive);
    CHECK(net::nat::reading_of(finished(changes, false)) == Reading::Relay);
    net::nat::Outcome o = finished(silent, false);
    o.resolved = false;
    CHECK(net::nat::reading_of(o) == Reading::NotResolved);
    o.resolved = true;
    o.probe.error = ProbeResult::Error::PortInUse;
    CHECK(net::nat::reading_of(o) == Reading::PortInUse);
    o.probe.error = ProbeResult::Error::Bind;
    CHECK(net::nat::reading_of(o) == Reading::NoSocket);
    o.probe.error = ProbeResult::Error::Socket;
    CHECK(net::nat::reading_of(o) == Reading::NoSocket);
    o.probe.error = ProbeResult::Error::Cancelled;
    CHECK(net::nat::reading_of(o) == Reading::Cancelled);
}

// The card's words for each kind of test: the headline and its status, the
// thing to do, the steps - the table the setup window shows.
void card_cases() {
    using net::nat::Card;
    using net::nat::Reading;
    using net::nat::RelaySetting;
    using net::nat::Tone;
    const std::string allow = "Allow bbhost.exe in your firewall or antivirus, then press Test again.";
    struct Row {
        const char* name;
        Reading reading;
        Tone tone;
        const char* head;
        std::string action;
        std::size_t steps;
    };
    const Row table[] = {
        {"direct-open", Reading::Direct, Tone::Good, "Online play works directly", "", 0},
        {"direct-closed", Reading::Direct, Tone::Good, "Online play works directly", "To let them connect directly:", 3},
        {"direct-unchecked", Reading::Direct, Tone::Good, "Online play works directly", "", 0},
        {"relay", Reading::Relay, Tone::Good, "Online play works through the relay", "", 0},
        {"inconclusive", Reading::Inconclusive, Tone::Warn, "The test was incomplete", "Press Test again.", 0},
        {"noanswer", Reading::NoAnswer, Tone::Bad, "Something is blocking bbhost", "Sign in (above), then press Test again to find out which.", 0},
        {"noanswer-signedin", Reading::NoAnswer, Tone::Bad, "Something is blocking bbhost",
         "Press Test again to find out which: the server couldn't tell this time.", 0},
        {"serverblocked", Reading::ServerBlocked, Tone::Bad, "Online play is blocked",
         allow + " If the website's connection test fails too, this network blocks online play (school and work networks can).", 0},
        {"answersstopped", Reading::AnswersStopped, Tone::Bad, "Answers aren't reaching bbhost", allow, 0},
        {"addressdiffers", Reading::AddressDiffers, Tone::Warn, "A VPN or proxy may be in the way",
         "Turn off any VPN or proxy, then press Test again.", 0},
        {"refused", Reading::SendRefused, Tone::Bad, "Something on this PC is blocking bbhost", allow, 0},
        {"refused-other", Reading::SendRefused, Tone::Bad, "This PC couldn't send anything",
         "Check this PC's internet connection. If it works, allow bbhost.exe in your firewall or antivirus, then press Test again.", 0},
        {"portinuse", Reading::PortInUse, Tone::Warn, "The game's port is in use",
         "Close the game, or choose another P2P port below, then press Test again.", 0},
        {"nosocket", Reading::NoSocket, Tone::Warn, "The test couldn't use UDP port 9307", "Choose another P2P port below, then press Test again.", 0},
        {"notresolved", Reading::NotResolved, Tone::Warn, "The test server wasn't found",
         "Check this PC's internet connection and the server chosen above, then press Test again.", 0},
        {"cancelled", Reading::Cancelled, Tone::Warn, "The test was stopped", "Press Test again.", 0},
    };
    CHECK(std::size(table) == std::size(net::nat::kExampleNames));
    for (const Row& row : table) {
        const Card c = net::nat::card_of(example(row.name), cx());
        const bool ok = c.reading == row.reading && c.tone == row.tone && c.head == row.head && c.action == row.action &&
                        c.steps.size() == row.steps && !c.sentence.empty();
        if (!ok) std::printf("FAIL card %s: [%d] %s | %s | %zu steps\n", row.name, static_cast<int>(c.tone), c.head.c_str(), c.action.c_str(), c.steps.size());
        CHECK(ok);
        // every sentence says what it means for co-op and PvP, or why nothing was tested
        CHECK(has(c.sentence, "co-op and PvP") || has(c.sentence, "Co-op and PvP") || has(c.sentence, "online play") ||
              has(c.sentence, "nothing") || has(c.sentence, "couldn't run") || has(c.sentence, "the game."));
    }
    // every reading its own headline (a refusal's two included): F10's line, the headline alone, keeps them apart
    for (const Row& x : table) {
        for (const Row& y : table) {
            if (x.reading != y.reading) CHECK(std::string(x.head) != y.head);
        }
    }
    // a name it does not know
    net::nat::Outcome unknown;
    CHECK(!net::nat::example_outcome("sideways", 9307, 0, &unknown));

    // The sentences, word for word, where they carry the distinctions.
    CHECK(net::nat::card_of(example("direct-open"), cx()).sentence == "Co-op and PvP can connect straight to other hunters, without the server's relay.");
    CHECK(net::nat::card_of(example("relay"), cx()).sentence ==
          "Your router changes ports for each connection, so co-op and PvP go through the server's relay, with a little more delay.");
    CHECK(has(net::nat::card_of(example("noanswer"), cx()).sentence, "A firewall or antivirus on this PC, or this network, stops"));
    CHECK(has(net::nat::card_of(example("serverblocked"), cx()).sentence, "None of bbhost's packets reached the server"));
    CHECK(has(net::nat::card_of(example("answersstopped"), cx()).sentence, "none of its answers came back"));
    CHECK(has(net::nat::card_of(example("refused"), cx()).sentence, "refused every packet"));
    CHECK(has(net::nat::card_of(example("portinuse"), cx()).sentence, "Another program has UDP port 9307"));

    // Incoming, in a player's words, for a direct result; the steps when closed - with this PC's address on its network.
    Card c = net::nat::card_of(example("direct-open"), cx());
    CHECK(c.incoming == net::nat::incoming_line(Inbound::Open) && c.steps.empty());
    c = net::nat::card_of(example("direct-closed"), cx());
    CHECK(c.incoming == net::nat::incoming_line(Inbound::Closed));
    CHECK(c.steps == net::nat::forward_steps(9307, "10.0.0.23", true, true));
    CHECK(net::nat::card_of(example("direct-closed"), cx(RelaySetting::Unknown, 9307, false)).steps[1] ==
          "If this PC runs a firewall, allow UDP port 9307 in it.");
    CHECK(net::nat::card_of(example("direct-unchecked"), cx()).incoming.empty());  // not checked: Details says so
    CHECK(any_has(net::nat::card_of(example("direct-unchecked"), cx()).details, "Incoming: not checked - this server has no port check."));
    // behind a router that changes ports, incoming is no player's business: Details has it
    c = net::nat::card_of(example("relay"), cx());
    CHECK(c.incoming.empty() && c.steps.empty() && any_has(c.details, "Incoming: closed"));
    CHECK(c.details[0] == "Tested from UDP port 9307, the game's P2P port: the server saw public ports 9307 and 40002.");
    // this PC holding the public address itself: no router step
    net::nat::Outcome pub = example("direct-closed");
    pub.probe.local_addr = pub.probe.at[0].addr;
    c = net::nat::card_of(pub, cx());
    CHECK(c.steps.size() == 2 && !any_has(c.steps, "router") && !any_has(c.details, "address on its network"));
    // the address not known: the router step without it
    pub.probe.local_addr = 0;
    CHECK(net::nat::card_of(pub, cx()).steps[0] == "In your router's settings, forward UDP port 9307 to this PC.");

    // The relay switch off: where other hunters cannot get in, the card says so and how to fix it.
    c = net::nat::card_of(example("direct-closed"), cx(RelaySetting::Off));
    CHECK(c.tone == Tone::Warn && c.head == "Other hunters may not reach you with the relay off");
    CHECK(has(c.sentence, "You turned the relay off") && has(c.sentence, "stops the first packets"));
    CHECK(c.action == "Forward the port, or turn the relay back on below:" && c.steps.size() == 3 && c.incoming.empty());
    c = net::nat::card_of(example("relay"), cx(RelaySetting::Off));
    CHECK(c.tone == Tone::Warn && c.head == "Other hunters may not reach you with the relay off");
    CHECK(has(c.sentence, "changes ports for each connection") && c.action == "Turn the relay back on below." && c.steps.empty());
    c = net::nat::card_of(example("direct-open"), cx(RelaySetting::Off));  // off, and others get in: all is well
    CHECK(c.tone == Tone::Good && c.head == "Online play works directly" && c.action.empty() &&
          any_has(c.details, "The relay is off for your account."));
    CHECK(net::nat::card_of(example("direct-unchecked"), cx(RelaySetting::Off)).tone == Tone::Good);  // not known: no warning
    CHECK(!any_has(net::nat::card_of(example("direct-closed"), cx(RelaySetting::On)).details, "The relay is off"));

    // A direct result says what happens. The relay on (the live default): co-op and PvP go through it whatever the
    // network allows - never "works directly ... without the server's relay"; turning it off is the way to direct.
    for (const char* n : {"direct-open", "direct-closed", "direct-unchecked"}) {
        c = net::nat::card_of(example(n), cx(RelaySetting::On));
        CHECK(c.tone == Tone::Good && c.head == "Your network allows direct connections");
        CHECK(c.sentence == "The relay is on, so co-op and PvP go through it.");
        CHECK(!has(c.head, "works directly") && !has(c.sentence, "without the server's relay"));
    }
    c = net::nat::card_of(example("direct-open"), cx(RelaySetting::On));
    CHECK(c.incoming == net::nat::incoming_line(Inbound::Open) && c.steps.empty());
    CHECK(c.action == "To connect directly instead, turn the relay off below.");
    c = net::nat::card_of(example("direct-closed"), cx(RelaySetting::On));
    CHECK(c.incoming == net::nat::incoming_line(Inbound::Closed) && c.action == "To connect directly instead:");
    CHECK(c.steps == (std::vector<std::string>{"In your router's settings, forward UDP port 9307 to this PC (10.0.0.23).",
                                               "Allow bbhost.exe in Windows Firewall (Private and Public).",
                                               "Then press Test again; then you can turn the relay off below."}));
    c = net::nat::card_of(example("direct-unchecked"), cx(RelaySetting::On));  // whether others get in not known: no advice
    CHECK(c.action.empty() && c.steps.empty() && c.incoming.empty());
    // the relay off: online play works directly, as it says
    c = net::nat::card_of(example("direct-open"), cx(RelaySetting::Off));
    CHECK(c.head == "Online play works directly" && c.sentence == "Co-op and PvP can connect straight to other hunters, without the server's relay.");
    // signed out: what the network allows, and the sign-in online play needs first - no steps yet
    for (const char* n : {"direct-open", "direct-closed"}) {
        c = net::nat::card_of(example(n), cx(RelaySetting::Unknown, 9307, true, false));
        CHECK(c.tone == Tone::Good && c.head == "Your network allows direct connections");
        CHECK(c.sentence == "Your router keeps one port for the game." && c.action == "Sign in (above) to play online." && c.steps.empty());
    }
    CHECK(net::nat::card_of(example("direct-closed"), cx(RelaySetting::Unknown, 9307, true, false)).incoming ==
          net::nat::incoming_line(Inbound::Closed));
    // a server without the switch, signed in: as before
    c = net::nat::card_of(example("direct-closed"), cx(RelaySetting::Unknown));
    CHECK(c.head == "Online play works directly" && c.action == "To let them connect directly:" && c.steps.size() == 3);
    // relay results are the same whatever the switch says when on, and signed out
    c = net::nat::card_of(example("relay"), cx(RelaySetting::On));
    CHECK(c.head == "Online play works through the relay" && c.action.empty() && c.tone == Tone::Good);
    CHECK(net::nat::card_of(unsigned_relay_out(), cx(RelaySetting::Unknown, 9307, true, false)).action ==
          "Sign in (above), then press Test again, so the server knows to relay you.");
    // a relay result not signed in: the server needs it kept to relay you
    net::nat::Outcome unsigned_relay = finished(made({{true, 0x0100000a, 9307}, {true, 0x0100000a, 40002}}), false);
    unsigned_relay.probe.sent = 3;
    CHECK(net::nat::card_of(unsigned_relay, cx()).action == "Sign in (above), then press Test again, so the server knows to relay you.");

    // Details: the port and what the server saw, when and whether it was kept, the server's word beside this PC's.
    c = net::nat::card_of(example("direct-closed"), cx());
    CHECK(c.details.size() >= 3 &&
          c.details[0] == "Tested from UDP port 9307, the game's P2P port: the server saw public port 9307 at both test ports, the game's own.");
    CHECK(any_has(c.details, "Incoming: closed - your router or this PC's firewall stopped a packet the server sent unasked to UDP port 9307."));
    CHECK(any_has(c.details, "Just now; saved to your account for this network (test again on another one)."));
    CHECK(any_has(c.details, "This PC's address on its network: 10.0.0.23."));
    CHECK(any_has(net::nat::card_of(example("inconclusive"), cx()).details, "this PC's own reading, not saved to your account (sign in above to keep it)."));
    CHECK(any_has(net::nat::card_of(example("noanswer-signedin"), cx()).details,
                  "not saved to your account: the server did not answer (Couldn't connect to server)."));
    CHECK(net::nat::card_of(example("refused"), cx()).details[0] ==
          "This PC refused all 17 packets bbhost tried to send from UDP port 9307 (Operation not permitted).");
    CHECK(any_has(net::nat::card_of(example("answersstopped"), cx()).details, "The server heard bbhost and reads this network as direct."));
    CHECK(any_has(net::nat::card_of(example("addressdiffers"), cx()).details, "the server kept \"blocked\" for your account"));
    CHECK(any_has(net::nat::card_of(example("portinuse"), cx()).details, "The system says: Address already in use."));
    CHECK(net::nat::card_of(example("notresolved"), cx()).details.empty());
    // what is not so is not said: no port check without answers, no server's word where it agrees, nothing heard
    // where the server heard nothing
    CHECK(!any_has(net::nat::card_of(example("noanswer"), cx()).details, "Incoming"));
    CHECK(!any_has(net::nat::card_of(example("direct-closed"), cx()).details, "The server says"));
    CHECK(!any_has(net::nat::card_of(example("serverblocked"), cx()).details, "heard bbhost"));
    {
        net::nat::Outcome differs = example("direct-closed");
        differs.save = net::nat::read_save_reply(200, R"({"ResKind":0,"Verdict":"relay","TestedAt":5})", "");
        c = net::nat::card_of(differs, cx());
        CHECK(c.reading == Reading::Relay && any_has(c.details, "The server says relay; this PC's own reading was direct."));
        net::nat::Outcome lost = example("direct-closed");
        lost.probe.at[1] = net::nat::Answer{};
        lost.local = net::nat::verdict_of(lost.probe);
        CHECK(any_has(net::nat::card_of(lost, cx()).details, "a lost packet, most likely; the server saw both requests"));
        net::nat::Outcome partly = example("direct-open");
        partly.probe.refused = 2;
        partly.probe.send_error = "Operation not permitted";
        c = net::nat::card_of(partly, cx());
        CHECK(c.tone == Tone::Good && any_has(c.details, "This PC refused 2 of the 5 packets bbhost tried to send (Operation not permitted)"));
        CHECK(!any_has(net::nat::card_of(example("direct-open"), cx()).details, "refused"));
    }
    // the details five minutes on
    CHECK(any_has(net::nat::card_of(example("direct-open"), [] { auto x = cx(); x.now = 1300; return x; }()).details, "5 minutes ago;"));
    // the P2P port changed since the test: the old port's result, tested again
    c = net::nat::card_of(example("direct-closed"), cx(RelaySetting::Unknown, 9400));
    CHECK(c.action == "The P2P port is now 9400: press Test again." && c.steps.empty());
    CHECK(net::nat::card_of(example("direct-open"), cx(RelaySetting::Unknown, 9400)).action == "The P2P port is now 9400: press Test again.");
    // nothing came back: no steps, as forwarding a port is not what is wrong
    CHECK(net::nat::card_of(example("noanswer"), cx()).steps.empty());
    // The other answers to a test that could not tell which, signed in.
    net::nat::Outcome limited = example("noanswer-signedin");
    limited.save = net::nat::read_save_reply(429, R"({"ResKind":1,"Message":"too many tests; try again later"})", "");
    CHECK(net::nat::card_of(limited, cx()).action == "Press Test again later to find out which: the server takes 10 tests an hour.");
    limited.save = net::nat::read_save_reply(200, R"({"ResKind":1,"Message":"no such token"})", "");
    CHECK(net::nat::card_of(limited, cx()).action == "Sign in again (above), then press Test again to find out which.");
    limited.save = net::nat::read_save_reply(404, R"({"detail":"Not Found"})", "");
    CHECK(has(net::nat::card_of(limited, cx()).action, "This server can't tell which, so start with this PC"));
}

// The account's last test (whoami's NatTest), before one runs here; and the card before any test.
void record_cases() {
    using net::nat::Record;
    using net::nat::RelaySetting;
    using net::nat::Tone;
    const double three_hours_ago = 1030 - 3 * 3600;
    // bbhost's, from this PC's P2P port: read as a test here would be, without this PC's address
    net::nat::Card c = net::nat::card_of_record(Record{Verdict::Direct, Inbound::Closed, 9307, three_hours_ago}, cx());
    CHECK(c.tone == Tone::Good && c.head == "Online play works directly" && c.incoming == net::nat::incoming_line(Inbound::Closed));
    CHECK(c.steps.size() == 3 && c.steps[0] == "In your router's settings, forward UDP port 9307 to this PC.");
    CHECK(c.details[0] == "Your account's last test, 3 hours ago, from UDP port 9307, this PC's P2P port.");
    CHECK(any_has(c.details, "Incoming: closed"));
    // the relay on: the network allows it, the relay carries co-op and PvP; the steps end at the switch
    c = net::nat::card_of_record(Record{Verdict::Direct, Inbound::Closed, 9307, three_hours_ago}, cx(RelaySetting::On));
    CHECK(c.head == "Your network allows direct connections" && c.action == "To connect directly instead:" && c.steps.size() == 3);
    CHECK(c.steps[0] == "In your router's settings, forward UDP port 9307 to this PC." &&
          c.steps[2] == "Then press Test again; then you can turn the relay off below.");
    c = net::nat::card_of_record(Record{Verdict::Direct, Inbound::NotTested, 0, three_hours_ago}, cx(RelaySetting::On));
    CHECK(c.head == "Your network allows direct connections" && c.action == "Press Test again to check this PC.");
    // turned off, from here: the warning
    c = net::nat::card_of_record(Record{Verdict::Direct, Inbound::Closed, 9307, three_hours_ago}, cx(RelaySetting::Off));
    CHECK(c.tone == Tone::Warn && c.head == "Other hunters may not reach you with the relay off" && c.steps.size() == 3);
    // the website's (no port check), or another port's: test this PC
    c = net::nat::card_of_record(Record{Verdict::Direct, Inbound::NotTested, 0, three_hours_ago}, cx());
    CHECK(c.head == "Online play works directly" && c.action == "Press Test again to check this PC." && c.steps.empty() && c.incoming.empty());
    CHECK(c.details[0] == "Your account's last test, 3 hours ago, on the website or from another PC.");
    c = net::nat::card_of_record(Record{Verdict::Direct, Inbound::Closed, 9400, three_hours_ago}, cx(RelaySetting::Off));
    CHECK(c.tone == Tone::Good && c.action == "Press Test again to check this PC." && c.steps.empty());
    CHECK(c.details[0] == "Your account's last test, 3 hours ago, from UDP port 9400; this PC's P2P port is UDP 9307.");
    c = net::nat::card_of_record(Record{Verdict::Relay, Inbound::Open, 9307, three_hours_ago}, cx());
    CHECK(c.head == "Online play works through the relay" && c.action.empty() && c.incoming.empty());
    c = net::nat::card_of_record(Record{Verdict::Blocked, Inbound::NotTested, 9307, three_hours_ago}, cx());
    CHECK(c.tone == Tone::Bad && c.head == "Online play is blocked" && has(c.action, "Allow bbhost.exe"));
    c = net::nat::card_of_record(Record{Verdict::Blocked, Inbound::NotTested, 0, three_hours_ago}, cx());
    CHECK(c.action == "Press Test again to check this PC.");

    // Not tested yet: what it does; with the relay off, a nudge to test.
    c = net::nat::card_untested(cx(), true);
    CHECK(c.tone == Tone::None && c.head == "Not tested yet" && c.action.empty());
    CHECK(c.details[0] == "It sends a few packets from UDP port 9307, the game's P2P port, and takes a few seconds; the game must be closed.");
    CHECK(c.details[1] == "Signed in, the result is kept for your account.");
    CHECK(net::nat::card_untested(cx(), false).details[1] == "Sign in (above) to keep the result for your account.");
    c = net::nat::card_untested(cx(RelaySetting::Off), true);
    CHECK(c.tone == Tone::Warn && c.action == "You turned the relay off: press Test connection to check that other hunters can reach you.");
}

// The relay switch's guard: off at once only after a direct result with incoming open; else why, and the steps.
void relay_cases() {
    CHECK(net::nat::relay_off_safe(Verdict::Direct, Inbound::Open));
    CHECK(!net::nat::relay_off_safe(Verdict::Direct, Inbound::Closed));
    CHECK(!net::nat::relay_off_safe(Verdict::Direct, Inbound::Unavailable));
    CHECK(!net::nat::relay_off_safe(Verdict::Relay, Inbound::Open));
    CHECK(!net::nat::relay_off_safe(Verdict::Blocked, Inbound::Open));
    const std::string lead = "Without the relay, other hunters must reach this PC directly. ";
    net::nat::RelayOffWarning w = net::nat::relay_off_warning(false, Verdict::None, Inbound::NotTested, cx(), "", true);
    CHECK(w.why == lead + "No test has checked that they can." && w.steps.size() == 3);
    w = net::nat::relay_off_warning(true, Verdict::Direct, Inbound::Closed, cx(), "10.0.0.23", true);
    CHECK(w.why == lead + "This PC's last test found that your router or firewall stops the first packets they send.");
    CHECK(w.steps == net::nat::forward_steps(9307, "10.0.0.23", true, true));
    w = net::nat::relay_off_warning(true, Verdict::Direct, Inbound::Unavailable, cx(), "", true);
    CHECK(w.why == lead + "This PC's last test couldn't check whether they can get in." && w.steps.size() == 3);
    w = net::nat::relay_off_warning(true, Verdict::Relay, Inbound::Closed, cx(), "10.0.0.23", true);
    CHECK(has(w.why, "changes ports for each connection, so they can't, and forwarding a port doesn't change that") && w.steps.empty());
    w = net::nat::relay_off_warning(true, Verdict::Blocked, Inbound::NotTested, cx(), "", true);
    CHECK(w.why == lead + "This PC's last test didn't get through, so online play may not work at all." && w.steps.empty());
    w = net::nat::relay_off_warning(false, Verdict::None, Inbound::NotTested, cx(net::nat::RelaySetting::Unknown, 9500, false), "", false);
    CHECK(w.steps == (std::vector<std::string>{"If this PC runs a firewall, allow UDP port 9500 in it.", "Then press Test again."}));
}

}  // namespace

int main() {
    stun_cases();
    verdict_cases();
    save_reply_cases();
    ports_cases();
    words_cases();
    reading_cases();
    card_cases();
    record_cases();
    relay_cases();
    exchange_cases();
    if (g_fail) {
        std::printf("nat_check_test: %d failures\n", g_fail);
        return 1;
    }
    std::printf("nat_check_test: ok\n");
    return 0;
}
