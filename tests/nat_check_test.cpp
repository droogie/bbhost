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
        CHECK(net::nat::verdict_of(r) == Verdict::Direct);
        CHECK(r.change_answered && r.change_offered && r.offered_port == s.port[2]);
        CHECK(r.reached && r.reached_from == s.port[2]);
        CHECK(net::nat::inbound_of(r) == Inbound::Open);
        CHECK(took < 1000);  // everything in: no waiting out the deadline
        CHECK(s.change_requests == 1 && s.second_answers == 1);
        CHECK(r.sent == 3 && r.refused == 0 && r.send_error.empty());
        CHECK(net::nat::public_ports(r) == "public port " + std::to_string(local) + " at both test ports, the game's own");
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
        ::close(holder);
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

void words_cases() {
    // The owner's words for the three results.
    CHECK(std::string(net::nat::headline(Verdict::Direct)) == "Direct connections: yes, your router keeps one port");
    CHECK(std::string(net::nat::headline(Verdict::Relay)) == "You'll use the relay: your router changes ports per destination");
    CHECK(std::string(net::nat::headline(Verdict::Blocked)) == "UDP is blocked on this network");
    for (Verdict v : {Verdict::None, Verdict::Direct, Verdict::Relay, Verdict::Blocked, Verdict::Inconclusive})
        CHECK(std::string(net::nat::meaning(v)).size() > 40);  // every result says what it means
    CHECK(net::nat::inbound_line(Inbound::Closed, 9307).find("forwarding UDP port 9307") != std::string::npos);
    // behind a router that changes ports, no forwarding advice: the relay carries it either way
    CHECK(net::nat::inbound_line(Inbound::Closed, 9307, Verdict::Relay).find("forwarding") == std::string::npos);
    CHECK(net::nat::inbound_line(Inbound::Closed, 9307, Verdict::Relay).find("the relay carries your sessions") != std::string::npos);
    CHECK(net::nat::inbound_line(Inbound::Open, 9307, Verdict::Relay).find("go through the relay") != std::string::npos);
    CHECK(net::nat::inbound_line(Inbound::Open, 9307).find("can connect to you directly") != std::string::npos);
    CHECK(net::nat::inbound_line(Inbound::Unavailable, 9307).find("port check is not available on this server") != std::string::npos);
    CHECK(net::nat::ago(1000, 1030) == "just now");
    CHECK(net::nat::ago(1000, 1000 + 300) == "5 minutes ago");
    CHECK(net::nat::ago(1000, 1000 + 3 * 3600) == "3 hours ago");
    CHECK(net::nat::ago(1000, 1000 + 2 * 86400) == "2 days ago");

    // The log line: the verdict, the port check, the port, what was sent and refused, whether kept - no address.
    const std::regex dotted(R"(\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)");
    net::nat::Outcome o;
    o.probe = made({{true, htonl(0xCB007107u), 9307}, {true, htonl(0xCB007107u), 9307}}, true, true, true);
    o.probe.sent = 3;
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

bool has(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

void reading_cases() {
    using net::nat::Answer;
    using net::nat::Card;
    using net::nat::Reading;
    using net::nat::Tone;
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
    refused.send_error_code = EPERM;
    refused.send_error = "Operation not permitted";
    refused.send_denied = true;
    ProbeResult one = made({a, none});
    one.sent = 7;
    ProbeResult changes = made({a, b});
    changes.sent = 3;

    // Not signed in, nothing back: this PC cannot tell out from back - no word of the server, nor of "this network"
    // being blocked; signing in tells which.
    net::nat::Outcome o = finished(silent, false);
    CHECK(net::nat::reading_of(o) == Reading::NoAnswer);
    Card c = net::nat::card_of(o, 1030);
    CHECK(c.tone == Tone::Bad && c.head == "No answer reached bbhost");
    CHECK(has(c.meaning, "either they did not get out, or the answers were stopped on the way back"));
    CHECK(has(c.meaning, "Sign in (above) and test again: the server then tells which."));
    CHECK(!has(c.meaning, "reached the server") && !has(c.head, "blocked") && !has(c.meaning, "blocked"));
    CHECK(c.incoming.empty() && c.note.empty());
    CHECK(c.detail == "Tested from UDP port 9307 (the game's P2P port). Just now; this PC's own reading, not saved to your account.");
    // signed in, the server's answer lost: still no answer, test again
    o = finished(silent, true);
    o.save = net::nat::read_save_reply(0, "", "Couldn't connect to server");
    CHECK(net::nat::reading_of(o) == Reading::NoAnswer);
    c = net::nat::card_of(o, 1030);
    CHECK(has(c.meaning, "The server could not tell which this time: test again.") && !has(c.meaning, "Sign in"));
    CHECK(has(c.detail, "not saved to your account: the server did not answer (Couldn't connect to server)."));

    // Kept, the server saw nothing from bbhost: if the website's test passes, something on this PC blocks bbhost.exe.
    o = finished(silent, true, kBlocked);
    CHECK(net::nat::reading_of(o) == Reading::ServerBlocked);
    c = net::nat::card_of(o, 1030);
    CHECK(c.tone == Tone::Bad && c.head == "Nothing from bbhost reached the server");
    CHECK(has(c.meaning, "If the website's connection test passes, your network is fine and a firewall, antivirus or VPN on this "
                         "PC is blocking bbhost.exe: allow it and test again - online play needs it."));
    CHECK(has(c.meaning, "If the website's test fails too, this network blocks UDP"));
    CHECK(has(c.detail, "saved to your account for this network"));

    // Kept direct or relay, yet nothing came back: a warning that the answers are stopped - never "Direct connections:
    // yes", no VPN or proxy blamed.
    o = finished(silent, true, kDirect);
    CHECK(net::nat::reading_of(o) == Reading::AnswersStopped);
    c = net::nat::card_of(o, 1030);
    CHECK(c.tone == Tone::Warn && c.head == "The server hears bbhost, but its answers don't reach it");
    CHECK(has(c.meaning, "reads it as direct") && has(c.meaning, "stopping what comes back to bbhost.exe"));
    CHECK(!has(c.head, "Direct connections") && !has(c.meaning, "Direct connections"));
    CHECK(c.note.empty() && c.incoming.empty());
    o = finished(silent, true, kRelay);
    CHECK(net::nat::reading_of(o) == Reading::AnswersStopped && has(net::nat::card_of(o, 1030).meaning, "reads it as relay"));

    // Every send refused: this PC did not let bbhost send - its words and what to allow; nothing kept.
    o = finished(refused, true);
    CHECK(net::nat::reading_of(o) == Reading::SendRefused);
    c = net::nat::card_of(o, 1030);
    CHECK(c.tone == Tone::Bad && c.head == "This PC did not let bbhost send");
    CHECK(has(c.meaning, "refused every test packet bbhost tried to send from UDP port 9307 (Operation not permitted)"));
    CHECK(has(c.meaning, "A firewall or antivirus on this PC is blocking bbhost.exe: allow bbhost.exe in it and test again"));
    CHECK(has(c.detail, "not saved to your account: nothing was sent.") && c.note.empty() && c.incoming.empty());
    o.signed_in = false;
    CHECK(net::nat::reading_of(o) == Reading::SendRefused);
    // refused for another reason (no route): the network first, the firewall only maybe
    o.probe.send_denied = false;
    o.probe.send_error_code = ENETUNREACH;
    o.probe.send_error = "Network is unreachable";
    c = net::nat::card_of(o, 1030);
    CHECK(net::nat::reading_of(o) == Reading::SendRefused && has(c.meaning, "(Network is unreachable). Check this PC's network connection"));
    CHECK(!has(c.meaning, "is blocking bbhost.exe"));
    // some refused, the rest answered: the result stands, the refusals told
    ProbeResult partly = back;
    partly.refused = 2;
    partly.send_error = "Operation not permitted";
    o = finished(partly, false);
    CHECK(net::nat::reading_of(o) == Reading::Direct);
    CHECK(has(net::nat::card_of(o, 1030).note, "This PC refused 2 of the 5 packets bbhost tried to send (Operation not permitted)"));

    // Kept blocked, yet the test ports answered bbhost: the request came from another address (a VPN or proxy).
    o = finished(back, true, kBlocked);
    CHECK(net::nat::reading_of(o) == Reading::AddressDiffers);
    c = net::nat::card_of(o, 1030);
    CHECK(c.tone == Tone::Warn && c.head == "The server did not see this test from your address");
    CHECK(has(c.meaning, "so UDP gets through") && has(c.meaning, "A VPN or proxy for web traffic"));
    CHECK(!c.incoming.empty());

    // The answers back: the server's verdict once kept, else this PC's.
    o = finished(back, true, kDirect);
    CHECK(net::nat::reading_of(o) == Reading::Direct);
    c = net::nat::card_of(o, 1030);
    CHECK(c.tone == Tone::Good && c.head == net::nat::headline(Verdict::Direct) && c.note.empty());
    CHECK(has(c.incoming, "Incoming: closed") && has(c.detail, ": the server saw public port 9307 at both test ports, the game's own"));
    CHECK(has(c.detail, "Just now; saved to your account for this network"));
    o = finished(back, true, kRelay);  // the server's word and the reading it differs from
    CHECK(net::nat::reading_of(o) == Reading::Relay);
    c = net::nat::card_of(o, 1030);
    CHECK(c.tone == Tone::Relay && c.head == net::nat::headline(Verdict::Relay));
    CHECK(c.note.rfind("This PC's own reading was \"direct\"", 0) == 0);
    o = finished(one, true, kDirect);  // one answer lost on the way back
    CHECK(net::nat::reading_of(o) == Reading::Direct && has(net::nat::card_of(o, 1030).note, "did not come back to bbhost"));
    o = finished(one, false);
    CHECK(net::nat::reading_of(o) == Reading::Inconclusive);
    CHECK(net::nat::card_of(o, 1030).tone == Tone::Warn);
    o = finished(changes, false);
    CHECK(net::nat::reading_of(o) == Reading::Relay && has(net::nat::card_of(o, 1030).note, "the server then knows to relay you"));
    o = finished(back, false);
    CHECK(net::nat::reading_of(o) == Reading::Direct && has(net::nat::card_of(o, 1030).note, "to keep it for your account"));
    CHECK(net::nat::card_of(o, 1000 + 300).detail.find("5 minutes ago;") != std::string::npos);

    // Nothing sent for other reasons: the name, the port, the socket, a cancel.
    o = finished(silent, false);
    o.resolved = false;
    CHECK(net::nat::reading_of(o) == Reading::NotResolved && net::nat::card_of(o, 1030).detail.empty());
    o.resolved = true;
    o.probe.error = ProbeResult::Error::PortInUse;
    CHECK(net::nat::reading_of(o) == Reading::PortInUse && net::nat::card_of(o, 1030).head == "UDP port 9307 is in use: close the game to test");
    o.probe.error = ProbeResult::Error::Bind;
    o.probe.error_text = "Cannot assign requested address";
    CHECK(net::nat::reading_of(o) == Reading::NoSocket && net::nat::card_of(o, 1030).meaning == "Cannot assign requested address.");
    o.probe.error = ProbeResult::Error::Cancelled;
    o.probe.error_text.clear();
    CHECK(net::nat::reading_of(o) == Reading::Cancelled && net::nat::card_of(o, 1030).meaning == "The test was stopped.");
    for (Verdict v : {Verdict::Direct, Verdict::Relay, Verdict::Blocked, Verdict::Inconclusive}) {
        const Tone t = net::nat::tone_of(v);
        CHECK(t == (v == Verdict::Direct ? Tone::Good : v == Verdict::Relay ? Tone::Relay : v == Verdict::Blocked ? Tone::Bad : Tone::Warn));
    }
    // the account's last test said blocked: the website passing while bbhost fails points at this PC
    CHECK(has(net::nat::meaning(Verdict::Blocked), "If the website's test passes while bbhost's does not"));
}

}  // namespace

int main() {
    stun_cases();
    verdict_cases();
    save_reply_cases();
    ports_cases();
    words_cases();
    reading_cases();
    exchange_cases();
    if (g_fail) {
        std::printf("nat_check_test: %d failures\n", g_fail);
        return 1;
    }
    std::printf("nat_check_test: ok\n");
    return 0;
}
