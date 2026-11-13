// The connection test: whether other players can reach this PC directly, or
// its co-op and PvP need the private server's relay - the website's account
// page test, run from the setup window on the game's own P2P port.
//
// The server (server/nat_test.py) answers STUN on two test ports (3479 and
// 3480, online.nat_test_ports) and remembers which public ports reached each
// from the caller's address. One socket - the P2P port - asks both:
//   direct   both saw the same public address and port: the router keeps one
//            port for the socket whatever the destination (a "cone" NAT), so
//            other players can connect to it directly;
//   relay    a different port (or address) at each: the router maps every
//            destination anew (a symmetric or carrier-grade NAT), so a direct
//            connection fails and the server's relay carries the session;
//   blocked  no answer from either: UDP to the server does not get through.
// Signed in, POST /auth/nat_test {Token, Inbound, Port} then has the server
// classify what it saw from this address, keep the result for the account
// and, in its "marked" relay mode, relay this address when it says relay -
// exactly as the website's test does. The request goes over IPv4, as the
// test's datagrams do: the server matches the two by address.
//
// The port check: a third request, to the first test port, carries RFC
// 5780's CHANGE-REQUEST (change port). The server answers it as usual with
// OTHER-ADDRESS naming its port check's port, and once more from that port,
// to which nothing was ever sent from here: when that answer arrives, the
// router and this PC's firewall let in a datagram nobody asked for (a
// forwarded port, a router that filters by address alone or not at all) and
// players whose routers change ports can still connect directly. A server
// without the check answers without OTHER-ADDRESS ("not available").
//
// This header is the reading of what came back and the words for it, with no
// sockets or server calls (tests/nat_check_test.cpp); probe() is the
// exchange itself (net/nat_probe.cpp), and net/nat_job.h runs the whole test
// for the setup window.
#pragma once

#include "replay/json.h"

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <functional>
#include <string>
#include <vector>

namespace net::nat {

enum class Verdict { None, Direct, Relay, Blocked, Inconclusive };
enum class Inbound { NotTested, Open, Closed, Unavailable };

// What one test port saw of the P2P port.
struct Answer {
    bool got = false;
    std::uint32_t addr = 0;  // the public address, network byte order
    std::uint16_t port = 0;  // the public port, host order
};

struct ProbeResult {
    enum class Error { None, Socket, PortInUse, Bind, Cancelled };
    Error error = Error::None;
    std::string error_text;      // the system's words for Socket and Bind
    int local_port = 0;          // the P2P port the requests left from
    std::vector<Answer> at;      // per test port, in the order given
    bool change_answered = false;  // the port check's request had its usual answer
    bool change_offered = false;   // ... with OTHER-ADDRESS: the server checks
    std::uint16_t offered_port = 0;  // the port its second answer leaves from
    bool reached = false;          // the second answer reached the P2P port
    std::uint16_t reached_from = 0;  // the port it came from
    int sent = 0;                  // datagrams the system took to send, retries included
    int refused = 0;               // datagrams it refused (the send failed)
    int send_error_code = 0;       // the first refusal's error (errno, WSAGetLastError)
    std::string send_error;        // ... in the system's words
    bool send_denied = false;      // ... a permission error: a firewall's or a security program's refusal
};

struct ProbeParams {
    std::uint32_t server = 0;            // the test server's IPv4 address, network byte order
    std::vector<std::uint16_t> ports;    // its test ports, host order: two or more
    std::uint16_t local_port = 0;        // the P2P port to send from
    int deadline_ms = 6000;              // the whole exchange
    std::function<bool()> cancelled;     // asked every few tens of milliseconds
};

// The exchange, from a socket bound to `local_port` alone (never shared: a
// game that has the port makes it PortInUse). Requests to every test port,
// each sent again while unanswered (at 0, 0.3, 0.8, 1.6, 2.8 and 4.2 s), and
// the port check's, again while its second answer has not come (at 0, 0.5
// and 1.2 s, and at 2.8 and 4.2 s while not even its usual answer has), until
// everything is in or the deadline: a security program that holds the first
// datagrams while it asks about bbhost.exe does not make the test read "no
// answer". When the system refuses every send and none is left to try, it
// ends there. net/nat_probe.cpp.
ProbeResult probe(const ProbeParams& p);

// What the answers say, without the server's word.
inline Verdict verdict_of(const ProbeResult& r) {
    if (r.error != ProbeResult::Error::None || r.at.empty()) return Verdict::None;
    std::size_t got = 0;
    for (const Answer& a : r.at) got += a.got ? 1 : 0;
    if (got == 0) return Verdict::Blocked;
    if (got < r.at.size()) return Verdict::Inconclusive;
    for (const Answer& a : r.at) {
        if (a.addr != r.at[0].addr || a.port != r.at[0].port) return Verdict::Relay;
    }
    return Verdict::Direct;
}

inline Inbound inbound_of(const ProbeResult& r) {
    if (r.error != ProbeResult::Error::None) return Inbound::NotTested;
    if (r.reached) return Inbound::Open;
    if (r.change_offered) return Inbound::Closed;
    if (r.change_answered) return Inbound::Unavailable;
    return Inbound::NotTested;
}

// The server's words for a verdict ("direct", "relay", "blocked").
inline const char* verdict_word(Verdict v) {
    switch (v) {
    case Verdict::Direct: return "direct";
    case Verdict::Relay: return "relay";
    case Verdict::Blocked: return "blocked";
    case Verdict::Inconclusive: return "inconclusive";
    case Verdict::None: break;
    }
    return "";
}
inline Verdict verdict_from_word(const std::string& w) {
    if (w == "direct") return Verdict::Direct;
    if (w == "relay") return Verdict::Relay;
    if (w == "blocked") return Verdict::Blocked;
    return Verdict::None;
}
// What /auth/nat_test's Inbound says ("" for neither).
inline const char* inbound_word(Inbound i) {
    return i == Inbound::Open ? "open" : i == Inbound::Closed ? "closed" : "";
}
inline Inbound inbound_from_word(const std::string& w) {
    return w == "open" ? Inbound::Open : w == "closed" ? Inbound::Closed : Inbound::NotTested;
}

// The server's answer to POST /auth/nat_test {Token, Inbound, Port}.
struct SaveReply {
    enum class State {
        Saved,      // classified and kept: `verdict` is the server's
        Unsaved,    // answered without a verdict (one test port, IPv6, matchmaking away): `message`
        Limited,    // 429: the account's tests this hour are used up
        Refused,    // the token is no longer known: a new sign-in is needed
        NoService,  // a server without the test (no ResKind)
        NoAnswer,   // nothing came back: `message` is the transport's
    };
    State state = State::NoAnswer;
    Verdict verdict = Verdict::None;
    double tested_at = 0;   // Saved: the server's time of it (Unix seconds)
    std::string message;
    long status = 0;
};

inline SaveReply read_save_reply(long status, const std::string& body, const std::string& transport_error) {
    SaveReply r;
    r.status = transport_error.empty() ? status : 0;
    if (!transport_error.empty() || status <= 0) {
        r.message = transport_error.empty() ? std::string("no answer") : transport_error;
        return r;
    }
    json::Value v;
    std::string perr;
    const json::Value* kind = json::parse(body, v, perr) && v.type == json::Value::Type::Object ? v.find("ResKind") : nullptr;
    if (!kind || kind->type != json::Value::Type::Number) {
        r.state = SaveReply::State::NoService;
        r.message = "HTTP " + std::to_string(status);
        return r;
    }
    const json::Value* msg = v.find("Message");
    r.message = msg && msg->type == json::Value::Type::String ? msg->string : std::string();
    if (kind->number != 0) {
        r.state = status == 429                 ? SaveReply::State::Limited
                  : r.message == "no such token" ? SaveReply::State::Refused
                                                 : SaveReply::State::Unsaved;
        if (r.message.empty()) r.message = "HTTP " + std::to_string(status);
        return r;
    }
    const json::Value* verdict = v.find("Verdict");
    r.verdict = verdict && verdict->type == json::Value::Type::String ? verdict_from_word(verdict->string) : Verdict::None;
    if (r.verdict == Verdict::None || status < 200 || status >= 300) {
        r.state = SaveReply::State::Unsaved;
        if (r.message.empty()) r.message = "no verdict";
        return r;
    }
    r.state = SaveReply::State::Saved;
    if (const json::Value* at = v.find("TestedAt"); at && at->type == json::Value::Type::Number) r.tested_at = at->number;
    return r;
}

// online.nat_test_ports ("3479,3480"): two or more distinct ports, else the
// defaults.
inline std::vector<std::uint16_t> parse_ports(const std::string& text) {
    std::vector<std::uint16_t> out;
    std::size_t at = 0;
    while (at <= text.size()) {
        const std::size_t comma = text.find(',', at);
        const std::string part = text.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        long n = 0;
        bool digits = false, bad = false;
        for (char ch : part) {
            if (ch == ' ' || ch == '\t') continue;
            if (ch < '0' || ch > '9' || n > 65535) {
                bad = true;
                break;
            }
            n = n * 10 + (ch - '0');
            digits = true;
        }
        if (bad || (digits && (n < 1 || n > 65535))) return {3479, 3480};
        if (digits) {
            const auto p = static_cast<std::uint16_t>(n);
            bool seen = false;
            for (std::uint16_t q : out) seen = seen || q == p;
            if (!seen) out.push_back(p);
        }
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    if (out.size() < 2) return {3479, 3480};
    return out;
}

// A finished test: what the setup window's Connection card shows and the log
// line says (net/nat_job.h runs it).
struct Outcome {
    bool resolved = true;            // the test server's name resolved
    ProbeResult probe;
    Verdict local = Verdict::None;   // from the answers alone
    Inbound inbound = Inbound::NotTested;
    bool signed_in = false;          // a token was sent: /auth/nat_test asked
    SaveReply save;
    double at = 0;                   // when it ended (Unix seconds)
    // The server's verdict once it kept one (what matchmaking goes by), else
    // this PC's own.
    Verdict shown() const { return save.state == SaveReply::State::Saved ? save.verdict : local; }
    bool saved() const { return save.state == SaveReply::State::Saved; }
};

// --- The words (the setup window's Connection card, F10's line) ---------------

// The verdict in plain words, and what it means.
inline const char* headline(Verdict v) {
    switch (v) {
    case Verdict::Direct: return "Direct connections: yes, your router keeps one port";
    case Verdict::Relay: return "You'll use the relay: your router changes ports per destination";
    case Verdict::Blocked: return "UDP is blocked on this network";
    case Verdict::Inconclusive: return "Only one test port answered: test again";
    case Verdict::None: break;
    }
    return "Not tested yet";
}
inline const char* meaning(Verdict v) {
    switch (v) {
    case Verdict::Direct:
        return "Other hunters can connect to this PC directly: co-op and PvP do not need the server's relay.";
    case Verdict::Relay:
        return "Other hunters cannot reach this PC directly, so co-op and PvP need the server's relay: it works, with a "
               "little more delay.";
    case Verdict::Blocked:
        return "Nothing reached the server's test ports, so online play will not work from this network. A firewall, "
               "a VPN or a school or work network can block UDP.";
    case Verdict::Inconclusive:
        return "One test port answered and the other did not - a lost packet, most likely. Run the test again.";
    case Verdict::None: break;
    }
    return "Checks whether other hunters can connect to this PC directly, or whether co-op and PvP need the server's "
           "relay. It sends a few UDP packets from the game's port to the server and takes a few seconds.";
}
// The port check in plain words: its result and what it means for others,
// beside the verdict (`v`): behind a router that changes ports a forwarded
// port does not change the verdict, and the relay carries the sessions.
inline std::string inbound_line(Inbound i, int p2p_port, Verdict v = Verdict::Direct) {
    const std::string port = std::to_string(p2p_port);
    const bool relay = v == Verdict::Relay;
    switch (i) {
    case Inbound::Open:
        return relay ? "Incoming: open. A packet the server sent unasked reached UDP port " + port +
                           ", but your router still changes ports for what the game sends, so sessions go through the relay."
                     : "Incoming: open. A packet the server sent unasked, from a port the game never contacted, reached UDP port " +
                           port + ", so even hunters whose routers change ports can connect to you directly.";
    case Inbound::Closed:
        return relay ? "Incoming: closed. Your router or this PC's firewall stopped a packet the server sent unasked; with "
                       "ports changing per destination, the relay carries your sessions either way."
                     : "Incoming: closed. Your router or this PC's firewall stopped a packet the server sent unasked. Hunters "
                       "whose routers change ports reach you through the relay instead; forwarding UDP port " +
                           port + " to this PC would let them connect directly.";
    case Inbound::Unavailable:
        return "Incoming: not checked - the port check is not available on this server.";
    case Inbound::NotTested: break;
    }
    return "Incoming: not checked - no answer came back to check it with.";
}

// "just now", "5 minutes ago", "3 hours ago", "2 days ago" (`now` and `then`
// in Unix seconds).
inline std::string ago(double then, double now) {
    const double s = now - then;
    if (s < 90) return "just now";
    char buf[48];
    if (s < 5400) std::snprintf(buf, sizeof(buf), "%d minutes ago", static_cast<int>(s / 60 + 0.5));
    else if (s < 129600) std::snprintf(buf, sizeof(buf), "%d hours ago", static_cast<int>(s / 3600 + 0.5));
    else std::snprintf(buf, sizeof(buf), "%d days ago", static_cast<int>(s / 86400 + 0.5));
    return buf;
}

// The public port(s) the test ports saw, for the card's detail line:
// "public port 9307 at both test ports, the game's own", "public ports
// 40001 and 40002".
inline std::string public_ports(const ProbeResult& r) {
    std::vector<std::uint16_t> ports;
    for (const Answer& a : r.at) {
        if (!a.got) continue;
        bool seen = false;
        for (std::uint16_t p : ports) seen = seen || p == a.port;
        if (!seen) ports.push_back(a.port);
    }
    if (ports.empty()) return {};
    bool same_addr = true;
    for (const Answer& a : r.at) same_addr = same_addr && (!a.got || a.addr == r.at[0].addr);
    if (ports.size() == 1) {
        std::string s = "public port " + std::to_string(ports[0]) + (r.at.size() == 2 ? " at both test ports" : " at every test port");
        if (!same_addr) s += ", from different public addresses";
        else if (ports[0] == r.local_port) s += ", the game's own";
        return s;
    }
    std::string s = "public ports ";
    for (std::size_t i = 0; i < ports.size(); ++i) {
        if (i) s += i + 1 == ports.size() ? " and " : ", ";
        s += std::to_string(ports[i]);
    }
    return s + (same_addr ? "" : ", from different public addresses");
}

// Every datagram the test tried to send was refused by this PC: nothing went
// out, so the server is not asked.
inline bool nothing_sent(const ProbeResult& r) { return r.sent == 0 && r.refused > 0; }

// Why a test's result was not kept for the account ("" when it was).
inline std::string unsaved_reason(const Outcome& o) {
    if (o.saved()) return {};
    if (!o.signed_in) return "not signed in";
    if (nothing_sent(o.probe)) return "nothing was sent";
    switch (o.save.state) {
    case SaveReply::State::Limited: return "the server takes 10 tests an hour; try again later";
    case SaveReply::State::Refused: return "the sign-in has expired";
    case SaveReply::State::NoService: return "this server keeps no connection tests (" + o.save.message + ")";
    case SaveReply::State::NoAnswer: return "the server did not answer (" + o.save.message + ")";
    case SaveReply::State::Unsaved: return "the server says: " + o.save.message;
    case SaveReply::State::Saved: break;
    }
    return {};
}

// The test's one log line: the verdict, the port check, the port, what was
// sent and refused (the first refusal's words), whether it was kept - and no
// addresses.
inline std::string log_line(const Outcome& o) {
    const int port = o.probe.local_port;
    if (!o.resolved) return "connection test: the test server's name did not resolve; nothing sent";
    switch (o.probe.error) {
    case ProbeResult::Error::PortInUse:
        return "connection test: UDP port " + std::to_string(port) + " is in use by another program (" + o.probe.error_text +
               "); nothing sent";
    case ProbeResult::Error::Socket:
    case ProbeResult::Error::Bind:
        return "connection test: no socket on UDP port " + std::to_string(port) + " (" + o.probe.error_text + "); nothing sent";
    case ProbeResult::Error::Cancelled: return "connection test: cancelled";
    case ProbeResult::Error::None: break;
    }
    std::string s = "connection test: ";
    s += verdict_word(o.local);
    const std::string ports = public_ports(o.probe);
    s += " (" + (ports.empty() ? std::string("no answer from the test ports") : ports) + ")";
    s += "; incoming ";
    switch (o.inbound) {
    case Inbound::Open: s += "open"; break;
    case Inbound::Closed: s += "closed"; break;
    case Inbound::Unavailable: s += "not checked (no port check on this server)"; break;
    case Inbound::NotTested: s += "not checked"; break;
    }
    s += "; UDP port " + std::to_string(port) + "; sent " + std::to_string(o.probe.sent) + ", refused " +
         std::to_string(o.probe.refused);
    if (o.probe.refused > 0) s += " (" + o.probe.send_error + ")";
    s += "; ";
    if (o.saved()) {
        s += "saved to the account";
        if (o.save.verdict != o.local) s += std::string(" - the server says ") + verdict_word(o.save.verdict);
    } else {
        s += "not saved (" + unsaved_reason(o) + ")";
    }
    return s;
}

}  // namespace net::nat
