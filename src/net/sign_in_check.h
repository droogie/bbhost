// What the server says of this PC's sign-in: its answer to POST /auth/whoami
// {Token}, read from the HTTP status, the body and the transport's error, for
// the setup window's check (net/account.h, account_check_start) and its test
// (tests/sign_in_check_test.cpp).
//
// The account service answers 200 in both cases: {ResKind 0, OnlineId,
// ExpiresAt, ...} for a token it knows, {ResKind 1, Message "no such token"}
// for one it does not - expired, or revoked when this PC was signed out
// (a sign-out elsewhere, a recovery, "sign out everywhere"). Anything else is
// no verdict: no answer, a 5xx or 429, a server without the account service
// (its framework's 404 without ResKind), a body that is not JSON.
//
// A known token's answer also carries the account's last connection test
// (NatTest: {Verdict, TestedAt, Inbound?, Port?}, or null), which the setup
// window's Connection card shows (net/nat_check.h).
#pragma once

#include "replay/json.h"

#include <string>

namespace net {

enum class SignInVerdict {
    Valid,      // the server knows the token
    Refused,    // the server does not: a new sign-in is needed
    Unchecked,  // no verdict; `why` says what came back instead
};

// The account's last connection test as whoami tells it.
struct NatTestRecord {
    bool present = false;
    std::string verdict;  // "direct", "relay" or "blocked"
    double tested_at = 0; // Unix seconds
    std::string inbound;  // bbhost's port check: "open", "closed", or "" (the website's test has none)
    int port = 0;         // the P2P port it ran from (with inbound)
};

struct SignInReply {
    SignInVerdict verdict = SignInVerdict::Unchecked;
    std::string name;       // Valid: the account's name, as the server spells it
    double expires_at = 0;  // Valid: when the token expires (Unix seconds); 0 when not said
    std::string why;        // Refused: the server's Message; Unchecked: what came instead
    long status = 0;        // the HTTP status, 0 when no answer came
    NatTestRecord nat_test; // Valid: the account's last connection test, when it has one
};

namespace sign_in_detail {
inline const json::Value* field(const json::Value& v, const char* key) {
    return v.type == json::Value::Type::Object ? v.find(key) : nullptr;
}
}  // namespace sign_in_detail

inline SignInReply read_sign_in_reply(long status, const std::string& body, const std::string& transport_error) {
    using sign_in_detail::field;
    SignInReply r;
    r.status = transport_error.empty() ? status : 0;
    if (!transport_error.empty() || status <= 0) {
        r.why = transport_error.empty() ? std::string("no answer") : transport_error;
        return r;
    }
    json::Value v;
    std::string perr;
    const json::Value* kind = json::parse(body, v, perr) ? field(v, "ResKind") : nullptr;
    if (!kind || kind->type != json::Value::Type::Number) {
        // A server without the account service: its framework's own "not
        // found", or a proxy's page.
        r.why = "HTTP " + std::to_string(status) + " without ResKind";
        return r;
    }
    if (kind->number != 0) {
        // The account service's refusal. A busy server (429, 5xx) or one that
        // names no reason this way is not a verdict on the token.
        if (status != 200 && status != 401) {
            r.why = "HTTP " + std::to_string(status);
            return r;
        }
        const json::Value* msg = field(v, "Message");
        r.verdict = SignInVerdict::Refused;
        r.why = msg && msg->type == json::Value::Type::String && !msg->string.empty() ? msg->string : "refused";
        return r;
    }
    const json::Value* id = field(v, "OnlineId");
    if (status < 200 || status >= 300 || !id || id->type != json::Value::Type::String || id->string.empty()) {
        r.why = "HTTP " + std::to_string(status) + " without OnlineId";
        return r;
    }
    r.verdict = SignInVerdict::Valid;
    r.name = id->string;
    if (const json::Value* e = field(v, "ExpiresAt"); e && e->type == json::Value::Type::Number) r.expires_at = e->number;
    if (const json::Value* n = field(v, "NatTest"); n && n->type == json::Value::Type::Object) {
        const json::Value* verdict = field(*n, "Verdict");
        const json::Value* at = field(*n, "TestedAt");
        if (verdict && verdict->type == json::Value::Type::String && at && at->type == json::Value::Type::Number) {
            r.nat_test.present = true;
            r.nat_test.verdict = verdict->string;
            r.nat_test.tested_at = at->number;
            if (const json::Value* i = field(*n, "Inbound"); i && i->type == json::Value::Type::String) r.nat_test.inbound = i->string;
            if (const json::Value* p = field(*n, "Port"); p && p->type == json::Value::Type::Number && p->number > 0 && p->number < 65536)
                r.nat_test.port = static_cast<int>(p->number);
        }
    }
    return r;
}

}  // namespace net
