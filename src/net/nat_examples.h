// Made-up finished connection tests, one of each kind the Connection card
// tells apart, as net/nat_job.h would leave them (net/nat_check.h): the setup
// window's natfake: test hook shows them, and tests/nat_check_test.cpp reads
// them. The addresses are documentation's own (203.0.113.0/24) and a home
// network's.
#pragma once

#include "net/nat_check.h"

#include <cerrno>
#include <string>

namespace net::nat {

inline constexpr const char* kExampleNames[] = {
    "direct-open",   "direct-closed", "direct-unchecked", "relay",        "inconclusive", "noanswer",
    "noanswer-signedin", "serverblocked", "answersstopped",   "addressdiffers", "refused",    "refused-other",
    "portinuse",     "nosocket",      "notresolved",      "cancelled",
};

// The test of kind `name` from UDP `port`, ended at `at` (Unix seconds); false
// for a name it does not know.
inline bool example_outcome(const std::string& name, int port, double at, Outcome* out) {
    Outcome o;
    o.at = at;
    ProbeResult& r = o.probe;
    r.local_port = port;
    r.local_addr = ipv4_net(10, 0, 0, 23);
    const std::uint32_t public_addr = ipv4_net(203, 0, 113, 7);
    const Answer seen{true, public_addr, static_cast<std::uint16_t>(port)};
    const Answer other{true, public_addr, 40002}, none{};
    r.at = {seen, seen};
    r.sent = 3;
    r.change_answered = r.change_offered = true;  // the port check closed unless said
    o.signed_in = true;
    const char* kept = nullptr;  // the server's verdict, when it kept one
    if (name == "direct-open") {
        r.reached = true;
        kept = "direct";
    } else if (name == "direct-closed") {
        kept = "direct";
    } else if (name == "direct-unchecked") {
        r.change_offered = false;
        kept = "direct";
    } else if (name == "relay") {
        r.at = {seen, other};
        kept = "relay";
    } else if (name == "inconclusive") {
        r.at = {seen, none};
        r.sent = 7;
        o.signed_in = false;
    } else if (name == "noanswer" || name == "noanswer-signedin" || name == "serverblocked" || name == "answersstopped") {
        r.at = {none, none};
        r.sent = 17;
        r.change_answered = r.change_offered = false;
        o.signed_in = name != "noanswer";
        if (name == "noanswer-signedin") o.save = read_save_reply(0, "", "Couldn't connect to server");
        if (name == "serverblocked") kept = "blocked";
        if (name == "answersstopped") kept = "direct";
    } else if (name == "addressdiffers") {
        kept = "blocked";
    } else if (name == "refused" || name == "refused-other") {
        r.at = {none, none};
        r.sent = 0;
        r.refused = 17;
        r.change_answered = r.change_offered = false;
        r.send_denied = name == "refused";
        r.send_error_code = r.send_denied ? EPERM : ENETUNREACH;
        r.send_error = r.send_denied ? "Operation not permitted" : "Network is unreachable";
    } else if (name == "portinuse" || name == "nosocket" || name == "cancelled") {
        r = ProbeResult{};
        r.local_port = port;
        r.error = name == "portinuse" ? ProbeResult::Error::PortInUse : name == "nosocket" ? ProbeResult::Error::Bind : ProbeResult::Error::Cancelled;
        r.error_text = name == "portinuse" ? "Address already in use" : name == "nosocket" ? "Cannot assign requested address" : "";
    } else if (name == "notresolved") {
        r = ProbeResult{};
        r.local_port = port;
        o.resolved = false;
    } else {
        return false;
    }
    o.local = verdict_of(r);
    o.inbound = inbound_of(r);
    if (kept)
        o.save = read_save_reply(200, std::string(R"({"ResKind":0,"Verdict":")") + kept + R"(","TestedAt":)" + std::to_string(at) + "}", "");
    *out = o;
    return true;
}

}  // namespace net::nat
