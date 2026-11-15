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
//   blocked  no answer from either: nothing came back to bbhost.
// Signed in, POST /auth/nat_test {Token, Inbound, Port} then has the server
// classify what it saw from this address, keep the result for the account
// and, in its "marked" relay mode, relay this address when it says relay -
// exactly as the website's test does. The request goes over IPv4, as the
// test's datagrams do: the server matches the two by address.
//
// No answer is not the same as blocked. This PC alone cannot tell whether its
// requests got out and the answers were stopped on the way back, or nothing
// got out; the server can (reading_of). The website's test is the browser's
// datagrams, and only what reaches the server counts there, so it can pass
// while a firewall or antivirus on this PC stops bbhost.exe's - which is what
// the game sends from, so the card says so. An inbound block at the layer
// Windows' firewall rules use stops only datagrams nobody asked for (the port
// check's second answer) - answers to what bbhost sent still get in - and an
// outbound one drops bbhost's datagrams while its sends report success.
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
// The card says it in a player's words: one headline with its status (all is
// well, a warning, a problem), one sentence on what it means for co-op and
// PvP, the one thing to do when there is something (a port to forward, in
// steps naming this PC's address on its network, the address the test's
// requests left from), whether other hunters can get in, and the technical
// lines under Details. The account's relay switch (whoami's RelayDirect: the
// website's account page, or the card's own switch) changes it: turned off
// while other hunters cannot get in, the card says so and how to fix it.
//
// This header is the reading of what came back and the words for it, with no
// sockets or server calls (tests/nat_check_test.cpp); probe() is the
// exchange itself (net/nat_probe.cpp), and net/nat_job.h runs the whole test
// for the setup window.
#pragma once

#include "replay/json.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
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
    std::uint32_t local_addr = 0;  // this PC's own address toward the server, network byte order (0 unknown):
                                   // the one a router forwards to; shown on the card, never logged
};

struct ProbeParams {
    std::uint32_t server = 0;            // the test server's IPv4 address, network byte order
    std::vector<std::uint16_t> ports;    // its test ports, host order: two or more
    std::uint16_t local_port = 0;        // the P2P port to send from
    int deadline_ms = 6000;              // the whole exchange
    std::function<bool()> cancelled;     // asked every few tens of milliseconds
};

// The exchange, from a socket bound to `local_port` alone (never shared: a
// game that has the port makes it PortInUse). First this PC's own address
// toward the server: another socket, connected there - which sends nothing -
// tells which one the system takes. Requests to every test port,
// each sent again while unanswered (at 0, 0.3, 0.8, 1.6, 2.8 and 4.2 s), and
// the port check's, again while its second answer has not come (at 0, 0.5
// and 1.2 s, and at 2.8 and 4.2 s while not even its usual answer has), until
// everything is in or the deadline: a security program that holds the first
// datagrams while it asks about bbhost.exe does not make the test read "no
// answer". When the system refuses every send and none is left to try, it
// ends there. net/nat_probe.cpp.
ProbeResult probe(const ProbeParams& p);

// Whether a socket could have UDP `port` now, bound alone as the test and the
// game bind it: "" when it could, else the system's words (another program has
// it). The P2P port setting asks before it takes a new port. net/nat_probe.cpp.
std::string port_taken(std::uint16_t port);

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

// --- The card's pieces --------------------------------------------------------

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

// --- What a finished test says (the setup window's Connection card) ---------

// The card's reading of a finished test: the verdict it shows, with what came
// back to bbhost and what the server saw.
enum class Reading {
    NotResolved,     // the test server's name did not resolve: nothing sent
    PortInUse,       // another program has the P2P port: nothing sent
    NoSocket,        // no socket on the P2P port (error_text)
    Cancelled,       // stopped before it ended
    SendRefused,     // this PC refused every datagram bbhost tried to send
    Direct,          // the server's verdict once kept, else this PC's - answers came back
    Relay,
    Inconclusive,    // one test port answered (not kept)
    NoAnswer,        // not kept, and no answer came back: out or back, this PC cannot tell
    ServerBlocked,   // kept: nothing from bbhost reached the server, and nothing came back
    AnswersStopped,  // kept direct or relay - the server heard this address - yet no answer reached bbhost
    AddressDiffers,  // kept blocked, yet the test ports answered bbhost: the request came from another address
};

inline Reading reading_of(const Outcome& o) {
    if (!o.resolved) return Reading::NotResolved;
    switch (o.probe.error) {
    case ProbeResult::Error::PortInUse: return Reading::PortInUse;
    case ProbeResult::Error::Socket:
    case ProbeResult::Error::Bind: return Reading::NoSocket;
    case ProbeResult::Error::Cancelled: return Reading::Cancelled;
    case ProbeResult::Error::None: break;
    }
    if (nothing_sent(o.probe)) return Reading::SendRefused;
    const bool answers = o.local == Verdict::Direct || o.local == Verdict::Relay || o.local == Verdict::Inconclusive;
    if (o.saved()) {
        if (o.save.verdict == Verdict::Blocked) return answers ? Reading::AddressDiffers : Reading::ServerBlocked;
        if (!answers) return Reading::AnswersStopped;
        return o.save.verdict == Verdict::Relay ? Reading::Relay : Reading::Direct;
    }
    if (!answers) return Reading::NoAnswer;
    return o.local == Verdict::Direct ? Reading::Direct : o.local == Verdict::Relay ? Reading::Relay : Reading::Inconclusive;
}

// --- The card (the setup window's Connection card; F10's line repeats its headline) ---

// The status icon beside the headline: all is well, a warning, a problem - or
// nothing to say yet.
enum class Tone { None, Good, Warn, Bad };

// The account's relay switch as the sign-in check read it (whoami's
// RelayDirect). Off, co-op and PvP connect directly, so only hunters who can
// get in reach this PC.
enum class RelaySetting { Unknown, On, Off };

inline constexpr bool kWindows =
#if defined(_WIN32)
    true;
#else
    false;
#endif

// What the card goes by besides the test.
struct CardContext {
    double now = 0;                              // Unix seconds ("just now")
    int p2p_port = 9307;                         // the P2P port setting: the port the steps forward
    RelaySetting relay = RelaySetting::Unknown;  // the account's relay switch
    bool signed_in = true;                       // this PC is signed in (a sign-in the server still takes)
    bool windows = kWindows;                     // the firewall step's words
};

struct Card {
    Reading reading = Reading::NoAnswer;  // a finished test's
    Tone tone = Tone::None;
    std::string head;                  // the headline, beside the status icon
    std::string sentence;              // what it means for co-op and PvP
    std::string incoming;              // whether other hunters can get in ("" not known)
    std::string action;                // the thing to do ("" when all is well)
    std::vector<std::string> steps;    // ... in numbered steps after it (none: the action alone)
    std::vector<std::string> details;  // the technical lines, under Details
};

// An IPv4 address held in network byte order, as text: "10.0.0.23".
inline std::string ipv4_text(std::uint32_t addr) {
    unsigned char b[4];
    std::memcpy(b, &addr, sizeof(b));
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u", b[0], b[1], b[2], b[3]);
    return buf;
}
// ... and one made from its four numbers.
inline std::uint32_t ipv4_net(unsigned a, unsigned b, unsigned c, unsigned d) {
    const unsigned char bytes[4] = {static_cast<unsigned char>(a), static_cast<unsigned char>(b), static_cast<unsigned char>(c),
                                    static_cast<unsigned char>(d)};
    std::uint32_t v = 0;
    std::memcpy(&v, bytes, sizeof(v));
    return v;
}

// Whether a router stands between this PC and the internet: the server saw
// another address than this PC's own (yes when either is not known).
inline bool behind_router(const ProbeResult& r) {
    if (!r.local_addr) return true;
    for (const Answer& a : r.at) {
        if (a.got) return a.addr != r.local_addr;
    }
    return true;
}

// The steps that let other hunters in: the router forwards the port to this
// PC (`lan`, its address on its network; "" not known) - left out when this
// PC holds the public address itself - the firewall lets it in, and the test
// says so.
inline std::vector<std::string> forward_steps(int port, const std::string& lan, bool router, bool windows) {
    const std::string p = std::to_string(port);
    std::vector<std::string> s;
    if (router)
        s.push_back("In your router's settings, forward UDP port " + p + " to this PC" +
                    (lan.empty() ? std::string() : " (" + lan + ")") + ".");
    s.push_back(windows ? std::string("Allow bbhost.exe in Windows Firewall (Private and Public).")
                        : "If this PC runs a firewall, allow UDP port " + p + " in it.");
    s.push_back("Then press Test again.");
    return s;
}

// Whether other hunters can get in (the port check), in a player's words: for
// a direct result. Behind a router that changes ports it changes nothing -
// other hunters cannot connect directly either way - and Details has it.
inline std::string incoming_line(Inbound i) {
    if (i == Inbound::Open) return "Incoming: other hunters can connect to you directly, even those whose routers change ports.";
    if (i == Inbound::Closed)
        return "Incoming: hunters whose routers change ports can't connect to you directly, so they reach you through the relay.";
    return {};
}
// The port check as Details says it, beside the result `v`.
inline std::string incoming_detail(Inbound i, int port, Verdict v) {
    const std::string p = std::to_string(port);
    switch (i) {
    case Inbound::Open:
        return "Incoming: open - a packet the server sent unasked, from a port bbhost never sent to, reached UDP port " + p +
               (v == Verdict::Relay ? ", but your router changes ports for what the game sends, so other hunters still can't "
                                      "connect directly."
                                    : ".");
    case Inbound::Closed:
        return "Incoming: closed - your router or this PC's firewall stopped a packet the server sent unasked to UDP port " + p + ".";
    case Inbound::Unavailable: return "Incoming: not checked - this server has no port check.";
    case Inbound::NotTested: break;
    }
    return "Incoming: not checked - the port check got no answer.";
}

// "Allow bbhost.exe ...": the thing to do when something on this PC stops it.
inline constexpr char kAllowBbhost[] = "Allow bbhost.exe in your firewall or antivirus, then press Test again.";

// The words of a direct or relay result - a test run here, or the account's
// last one - and what the relay switch does to them. `lan` and `router` are
// the steps' (forward_steps); `signed_in`: the test asked the server (not
// signed in, a relay result asks for a sign-in, which the relay goes by).
// A direct result says what happens: with the relay on (the account's switch,
// the live server's default) co-op and PvP go through it whatever the network
// allows; signed out, nothing goes online yet. Only with the relay off, or a
// server without the switch, does online play work directly.
inline void result_words(Card& c, Verdict v, Inbound in, const CardContext& x, const std::string& lan, bool router, bool signed_in) {
    const bool off = x.relay == RelaySetting::Off;
    c.tone = Tone::Good;
    if (v == Verdict::Direct && !x.signed_in) {
        c.head = "Your network allows direct connections";
        c.sentence = "Your router keeps one port for the game.";
        c.incoming = incoming_line(in);
        c.action = "Sign in (above) to play online.";
        return;
    }
    if (v == Verdict::Direct && x.relay == RelaySetting::On) {
        c.head = "Your network allows direct connections";
        c.sentence = "The relay is on, so co-op and PvP go through it.";
        c.incoming = incoming_line(in);
        if (in == Inbound::Open) {
            c.action = "To connect directly instead, turn the relay off below.";
        } else if (in == Inbound::Closed) {
            // First the steps that let other hunters in; then the relay can go.
            c.action = "To connect directly instead:";
            c.steps = forward_steps(x.p2p_port, lan, router, x.windows);
            c.steps.back() = "Then press Test again; then you can turn the relay off below.";
        }
        return;
    }
    if (v == Verdict::Direct) {
        c.head = "Online play works directly";
        c.sentence = "Co-op and PvP can connect straight to other hunters, without the server's relay.";
        c.incoming = incoming_line(in);
        if (in != Inbound::Closed) return;
        c.steps = forward_steps(x.p2p_port, lan, router, x.windows);
        if (!off) {
            c.action = "To let them connect directly:";
            return;
        }
        // Turned off while other hunters cannot get in.
        c.tone = Tone::Warn;
        c.head = "Other hunters may not reach you with the relay off";
        c.sentence = "You turned the relay off, but your router or this PC's firewall stops the first packets other hunters "
                     "send you, so direct connections can fail.";
        c.incoming.clear();
        c.action = "Forward the port, or turn the relay back on below:";
        return;
    }
    c.head = "Online play works through the relay";
    c.sentence = "Your router changes ports for each connection, so co-op and PvP go through the server's relay, with a "
                 "little more delay.";
    if (off) {
        c.tone = Tone::Warn;
        c.head = "Other hunters may not reach you with the relay off";
        c.sentence = "You turned the relay off, but your router changes ports for each connection, so direct connections "
                     "to you fail.";
        c.action = "Turn the relay back on below.";
    } else if (!signed_in) {
        c.action = "Sign in (above), then press Test again, so the server knows to relay you.";
    }
}

// The card for a finished test.
inline Card card_of(const Outcome& o, const CardContext& x) {
    Card c;
    c.reading = reading_of(o);
    const ProbeResult& r = o.probe;
    const std::string port = std::to_string(r.local_port);
    c.tone = Tone::Warn;
    switch (c.reading) {
    case Reading::NotResolved:
        c.head = "The test server wasn't found";
        c.sentence = "Its name didn't resolve, so the test couldn't run.";
        c.action = "Check this PC's internet connection and the server chosen above, then press Test again.";
        return c;
    case Reading::PortInUse:
        c.head = "The game's port is in use";
        c.sentence = "Another program has UDP port " + port + ", which the test sends from - most likely bbhost itself, running the game.";
        c.action = "Close the game, or choose another P2P port below, then press Test again.";
        if (!r.error_text.empty()) c.details.push_back("The system says: " + r.error_text + ".");
        return c;
    case Reading::NoSocket:
        c.head = "The test couldn't use UDP port " + port;
        c.sentence = "This PC didn't let bbhost open the game's port, so nothing was tested.";
        c.action = "Choose another P2P port below, then press Test again.";
        if (!r.error_text.empty()) c.details.push_back("The system says: " + r.error_text + ".");
        return c;
    case Reading::Cancelled:
        c.head = "The test was stopped";
        c.sentence = "It ended before it finished, so it tells nothing yet.";
        c.action = "Press Test again.";
        return c;
    default: break;
    }
    const bool answers = o.local == Verdict::Direct || o.local == Verdict::Relay || o.local == Verdict::Inconclusive;
    const std::string lan = r.local_addr ? ipv4_text(r.local_addr) : std::string();
    const bool router = behind_router(r);
    switch (c.reading) {
    case Reading::SendRefused:
        c.tone = Tone::Bad;
        c.head = r.send_denied ? "Something on this PC is blocking bbhost" : "This PC couldn't send anything";
        c.sentence = "This PC refused every packet bbhost tried to send, so co-op and PvP can't work.";
        c.action = r.send_denied ? std::string(kAllowBbhost)
                                 : "Check this PC's internet connection. If it works, allow bbhost.exe in your firewall or "
                                   "antivirus, then press Test again.";
        break;
    case Reading::Direct:
    case Reading::Relay:
        result_words(c, c.reading == Reading::Direct ? Verdict::Direct : Verdict::Relay, o.inbound, x, lan, router, o.signed_in);
        break;
    case Reading::Inconclusive:
        c.head = "The test was incomplete";
        c.sentence = "One of the server's two answers didn't come back, so the test can't tell yet whether co-op and PvP "
                     "connect directly or through the relay.";
        c.action = "Press Test again.";
        break;
    case Reading::NoAnswer:
        c.tone = Tone::Bad;
        c.head = "Something is blocking bbhost";
        c.sentence = "A firewall or antivirus on this PC, or this network, stops bbhost's packets or the answers to them, "
                     "so co-op and PvP can't work.";
        if (!o.signed_in) c.action = "Sign in (above), then press Test again to find out which.";
        else if (o.save.state == SaveReply::State::Limited)
            c.action = "Press Test again later to find out which: the server takes 10 tests an hour.";
        else if (o.save.state == SaveReply::State::Refused) c.action = "Sign in again (above), then press Test again to find out which.";
        else if (o.save.state == SaveReply::State::NoService)
            c.action = "This server can't tell which, so start with this PC: allow bbhost.exe in your firewall or antivirus, "
                       "then press Test again.";
        else c.action = "Press Test again to find out which: the server couldn't tell this time.";
        break;
    case Reading::ServerBlocked:
        c.tone = Tone::Bad;
        c.head = "Online play is blocked";
        c.sentence = "None of bbhost's packets reached the server, so co-op and PvP can't work from here.";
        c.action = std::string(kAllowBbhost) +
                   " If the website's connection test fails too, this network blocks online play (school and work networks can).";
        break;
    case Reading::AnswersStopped:
        c.tone = Tone::Bad;
        c.head = "Answers aren't reaching bbhost";
        c.sentence = "The server got bbhost's packets, but none of its answers came back, so co-op and PvP can't work: a "
                     "firewall or antivirus on this PC is most likely stopping them.";
        c.action = kAllowBbhost;
        break;
    case Reading::AddressDiffers:
        c.head = "A VPN or proxy may be in the way";
        c.sentence = "Your sign-in reached the server from one address and bbhost's packets from another, so online play may fail.";
        c.action = "Turn off any VPN or proxy, then press Test again.";
        break;
    default: break;
    }
    // Details: what went out and what the server saw, the port check, when
    // and whether it was kept, the server's word beside this PC's.
    const std::string ports = public_ports(r);
    if (nothing_sent(r))
        c.details.push_back("This PC refused all " + std::to_string(r.refused) + " packets bbhost tried to send from UDP port " + port +
                            " (" + r.send_error + ").");
    else
        c.details.push_back("Tested from UDP port " + port + ", the game's P2P port: " +
                            (ports.empty() ? std::string("no answer came back from the test ports.") : "the server saw " + ports + "."));
    if (answers) c.details.push_back(incoming_detail(o.inbound, r.local_port, o.shown()));
    std::string when = ago(o.at, x.now);
    if (!when.empty() && when[0] >= 'a' && when[0] <= 'z') when[0] = static_cast<char>(when[0] - 'a' + 'A');
    if (o.saved()) c.details.push_back(when + "; saved to your account for this network (test again on another one).");
    else if (!o.signed_in) c.details.push_back(when + "; this PC's own reading, not saved to your account (sign in above to keep it).");
    else c.details.push_back(when + "; not saved to your account: " + unsaved_reason(o) + ".");
    if (o.saved() && answers && o.save.verdict != Verdict::Blocked) {
        if (o.local == Verdict::Inconclusive)
            c.details.push_back("One of the server's answers didn't come back to bbhost - a lost packet, most likely; the server "
                                "saw both requests.");
        else if (o.local != o.save.verdict)
            c.details.push_back(std::string("The server says ") + verdict_word(o.save.verdict) + "; this PC's own reading was " +
                                verdict_word(o.local) +
                                ". The server compares what reached it from the address this request came from, and a VPN "
                                "or a proxy for web traffic can make the two differ.");
    } else if (o.saved() && !answers && o.save.verdict != Verdict::Blocked) {
        c.details.push_back(std::string("The server heard bbhost and reads this network as ") + verdict_word(o.save.verdict) + ".");
    } else if (c.reading == Reading::AddressDiffers) {
        c.details.push_back("The test ports answered bbhost, so its packets get through, but the server kept \"blocked\" for "
                            "your account: none of them came from the address your request came from.");
    }
    if (r.refused > 0 && r.sent > 0)
        c.details.push_back("This PC refused " + std::to_string(r.refused) + " of the " + std::to_string(r.refused + r.sent) +
                            " packets bbhost tried to send (" + r.send_error +
                            "): if a firewall or antivirus asked about bbhost.exe, allow it.");
    if (!lan.empty() && router) c.details.push_back("This PC's address on its network: " + lan + ".");
    if (x.relay == RelaySetting::Off) c.details.push_back("The relay is off for your account.");
    // A result from a port the setting no longer names: the old port's.
    if (r.local_port > 0 && x.p2p_port > 0 && r.local_port != x.p2p_port) {
        c.action = "The P2P port is now " + std::to_string(x.p2p_port) + ": press Test again.";
        c.steps.clear();
    }
    return c;
}

// The account's last test as whoami tells it (the sign-in check), which the
// card shows until a test runs here.
struct Record {
    Verdict verdict = Verdict::None;       // direct, relay or blocked
    Inbound inbound = Inbound::NotTested;  // bbhost's port check
    int port = 0;                          // the P2P port bbhost's test ran from; 0 for the website's
    double at = 0;                         // Unix seconds
};

// The card for the account's last test. Run from this PC's P2P port, it reads
// as a test run here would (without this PC's address, which only a test
// here finds); else it asks for one.
inline Card card_of_record(const Record& rec, const CardContext& x) {
    Card c;
    const bool here = rec.port > 0 && rec.port == x.p2p_port;
    if (rec.verdict == Verdict::Direct || rec.verdict == Verdict::Relay) {
        c.reading = rec.verdict == Verdict::Direct ? Reading::Direct : Reading::Relay;
        result_words(c, rec.verdict, here ? rec.inbound : Inbound::NotTested, x, "", true, true);
    } else {
        c.reading = Reading::ServerBlocked;
        c.tone = Tone::Bad;
        c.head = "Online play is blocked";
        c.sentence = "Nothing from your account's last test reached the server, so co-op and PvP couldn't work from there.";
        if (here)
            c.action = std::string(kAllowBbhost) +
                       " If the website's connection test fails too, this network blocks online play (school and work networks can).";
    }
    if (!here) {
        c.action = "Press Test again to check this PC.";
        c.steps.clear();
    }
    const std::string when = "Your account's last test, " + ago(rec.at, x.now);
    const std::string p2p = std::to_string(x.p2p_port);
    c.details.push_back(here           ? when + ", from UDP port " + p2p + ", this PC's P2P port."
                        : rec.port > 0 ? when + ", from UDP port " + std::to_string(rec.port) + "; this PC's P2P port is UDP " + p2p + "."
                                       : when + ", on the website or from another PC.");
    if (here && rec.inbound != Inbound::NotTested) c.details.push_back(incoming_detail(rec.inbound, rec.port, rec.verdict));
    if (x.relay == RelaySetting::Off) c.details.push_back("The relay is off for your account.");
    return c;
}

// The card before any test, here or for the account.
inline Card card_untested(const CardContext& x, bool signed_in) {
    Card c;
    c.head = "Not tested yet";
    c.sentence = "The test checks whether other hunters can connect to this PC directly, or whether co-op and PvP go "
                 "through the server's relay.";
    if (x.relay == RelaySetting::Off) {
        c.tone = Tone::Warn;
        c.action = "You turned the relay off: press Test connection to check that other hunters can reach you.";
    }
    c.details.push_back("It sends a few packets from UDP port " + std::to_string(x.p2p_port) +
                        ", the game's P2P port, and takes a few seconds; the game must be closed.");
    c.details.push_back(signed_in ? "Signed in, the result is kept for your account." : "Sign in (above) to keep the result for your account.");
    return c;
}

// --- The relay switch and the P2P port setting ---------------------------------

// Off, co-op and PvP connect directly alone, which needs other hunters to get
// in: a direct result with incoming open.
inline bool relay_off_safe(Verdict v, Inbound i) { return v == Verdict::Direct && i == Inbound::Open; }

// What the relay switch says before it goes off without that: why, and the
// steps that fix it (none where forwarding a port does not help).
struct RelayOffWarning {
    std::string why;
    std::vector<std::string> steps;
};
// `tested`: a result of this PC's is known - `v` and `i`, from its last test.
inline RelayOffWarning relay_off_warning(bool tested, Verdict v, Inbound i, const CardContext& x, const std::string& lan, bool router) {
    RelayOffWarning w;
    const std::string lead = "Without the relay, other hunters must reach this PC directly. ";
    if (tested && v == Verdict::Relay) {
        w.why = lead + "This PC's last test says your router changes ports for each connection, so they can't, and "
                       "forwarding a port doesn't change that.";
        return w;
    }
    if (tested && v != Verdict::Direct) {
        w.why = lead + "This PC's last test didn't get through, so online play may not work at all.";
        return w;
    }
    w.why = lead + (!tested                 ? "No test has checked that they can."
                    : i == Inbound::Closed ? "This PC's last test found that your router or firewall stops the first packets they send."
                                           : "This PC's last test couldn't check whether they can get in.");
    w.steps = forward_steps(x.p2p_port, lan, router, x.windows);
    return w;
}

// The P2P port setting's own check: "" when `port` will do, else why not -
// 1024 to 65535 (the ones below are the system's), and none of the server's
// test ports. Whether another program has it is port_taken's.
inline std::string p2p_port_problem(long port, const std::vector<std::uint16_t>& test_ports) {
    if (port < 1024 || port > 65535) return "Choose a port from 1024 to 65535.";
    for (std::uint16_t t : test_ports) {
        if (t == port) return "UDP " + std::to_string(port) + " is one of the server's test ports: choose another.";
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
