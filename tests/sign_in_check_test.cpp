// The sign-in check's reading of the server's answer to POST /auth/whoami
// (net/sign_in_check.h), and of POST /auth/profile's to the relay switch: a
// token the account service knows, one it does not
// (its refusal comes as HTTP 200 with ResKind 1, or as the NP surfaces' 401),
// and everything that is no verdict on the token - no answer, a busy server
// (429, 5xx, even in the service's own words), a server without the account
// service (its framework's 404 without ResKind), a body that is not the
// service's.
#include "net/sign_in_check.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::printf("FAIL: %s\n", what);
    }
}

using net::SignInVerdict;

net::SignInReply read(long status, const std::string& body, const std::string& error = "") {
    return net::read_sign_in_reply(status, body, error);
}

void valid() {
    const net::SignInReply r = read(200,
                                    R"({"ResKind":0,"OnlineId":"Hunter_01","Kind":"game","ExpiresAt":1797292800.25,)"
                                    R"("Verified":true,"Providers":["discord"],"ProfilePublic":true,"ShowDiscord":false,)"
                                    R"("ModdedSummons":true,"NatTest":null})");
    check(r.verdict == SignInVerdict::Valid, "a known token is valid");
    check(r.name == "Hunter_01", "the account's name as the server spells it");
    check(std::fabs(r.expires_at - 1797292800.25) < 1e-6, "its expiry");
    check(r.status == 200, "the status is kept");
    const net::SignInReply n = read(200, R"({"ResKind":0,"OnlineId":"Hunter"})");
    check(n.verdict == SignInVerdict::Valid && n.expires_at == 0, "valid without ExpiresAt: the expiry unknown");
}

void refused() {
    const net::SignInReply r = read(200, R"({"ResKind":1,"Message":"no such token"})");
    check(r.verdict == SignInVerdict::Refused, "the service's refusal comes as HTTP 200 with ResKind 1");
    check(r.why == "no such token", "with its Message");
    check(r.name.empty(), "and no name");
    const net::SignInReply m = read(401, R"({"ResKind":1,"Message":"bad or expired token"})");
    check(m.verdict == SignInVerdict::Refused && m.why == "bad or expired token", "the NP surfaces' 401 is a refusal");
    const net::SignInReply b = read(401, R"({"ResKind":1,"Message":"this account is banned from the server"})");
    check(b.verdict == SignInVerdict::Refused, "a ban is a refusal");
    const net::SignInReply q = read(200, R"({"ResKind":2})");
    check(q.verdict == SignInVerdict::Refused && q.why == "refused", "a refusal without a Message");
}

void unchecked() {
    const net::SignInReply none = read(0, "", "Couldn't connect to server");
    check(none.verdict == SignInVerdict::Unchecked, "no connection: no verdict");
    check(none.status == 0 && none.why == "Couldn't connect to server", "the transport's error, status 0");
    const net::SignInReply late = read(0, "", "Timeout was reached");
    check(late.verdict == SignInVerdict::Unchecked && late.status == 0, "a timeout: no verdict");
    const net::SignInReply half = read(200, R"({"ResKind":1,"Message":"no such token"})", "Recv failure");
    check(half.verdict == SignInVerdict::Unchecked && half.status == 0, "a transport error wins over a body");
    const net::SignInReply old = read(404, R"({"detail":"Not Found"})");
    check(old.verdict == SignInVerdict::Unchecked, "a server without /auth/whoami (404): no verdict");
    check(old.status == 404 && old.why.find("ResKind") != std::string::npos, "said as an answer without ResKind");
    const net::SignInReply html = read(502, "<html><body>Bad Gateway</body></html>");
    check(html.verdict == SignInVerdict::Unchecked && html.status == 502, "a proxy's page: no verdict");
    const net::SignInReply busy = read(429, R"({"ResKind":1,"Message":"too many requests"})");
    check(busy.verdict == SignInVerdict::Unchecked, "429 in the service's words: not a refusal of the token");
    const net::SignInReply down = read(503, R"({"ResKind":1,"Message":"try again"})");
    check(down.verdict == SignInVerdict::Unchecked, "503 in the service's words: not a refusal");
    const net::SignInReply gone = read(404, R"({"ResKind":1,"Message":"no such token"})");
    check(gone.verdict == SignInVerdict::Unchecked, "a 404 is never a verdict on the token");
    check(read(200, "not json").verdict == SignInVerdict::Unchecked, "a body that is not JSON");
    check(read(200, "").verdict == SignInVerdict::Unchecked, "an empty body");
    check(read(200, "[0]").verdict == SignInVerdict::Unchecked, "JSON that is not an object");
    check(read(200, R"({"ResKind":"0","OnlineId":"Hunter"})").verdict == SignInVerdict::Unchecked, "ResKind not a number");
    check(read(200, R"({"ResKind":0})").verdict == SignInVerdict::Unchecked, "ResKind 0 without OnlineId");
    check(read(200, R"({"ResKind":0,"OnlineId":""})").verdict == SignInVerdict::Unchecked, "ResKind 0 with an empty OnlineId");
    check(read(200, R"({"ResKind":0,"OnlineId":7})").verdict == SignInVerdict::Unchecked, "OnlineId not a string");
    check(read(500, R"({"ResKind":0,"OnlineId":"Hunter"})").verdict == SignInVerdict::Unchecked, "ResKind 0 on a 500");
}

// whoami's NatTest: the account's last connection test, for the setup
// window's Connection card - bbhost's with its port check and port, the
// website's without, none, and shapes that are not one.
void nat_test() {
    const net::SignInReply b = read(200, R"({"ResKind":0,"OnlineId":"Hunter","NatTest":{"Verdict":"relay","TestedAt":1794268800.5,)"
                                         R"("Inbound":"closed","Port":9307}})");
    check(b.nat_test.present && b.nat_test.verdict == "relay", "bbhost's test: its verdict");
    check(b.nat_test.tested_at == 1794268800.5, "its time");
    check(b.nat_test.inbound == "closed" && b.nat_test.port == 9307, "its port check and port");
    const net::SignInReply w = read(200, R"({"ResKind":0,"OnlineId":"Hunter","NatTest":{"Verdict":"direct","TestedAt":5}})");
    check(w.nat_test.present && w.nat_test.verdict == "direct" && w.nat_test.inbound.empty() && w.nat_test.port == 0,
          "the website's test: no port check");
    check(!read(200, R"({"ResKind":0,"OnlineId":"Hunter","NatTest":null})").nat_test.present, "no test yet");
    check(!read(200, R"({"ResKind":0,"OnlineId":"Hunter"})").nat_test.present, "a server without NatTest");
    check(!read(200, R"({"ResKind":0,"OnlineId":"Hunter","NatTest":{"Verdict":"direct"}})").nat_test.present, "no time: none");
    check(!read(200, R"({"ResKind":0,"OnlineId":"Hunter","NatTest":"direct"})").nat_test.present, "not an object: none");
    check(read(200, R"({"ResKind":0,"OnlineId":"Hunter","NatTest":{"Verdict":"direct","TestedAt":5,"Port":70000}})").nat_test.port == 0,
          "a port out of range is left out");
    check(!read(200, R"({"ResKind":1,"Message":"no such token","NatTest":{"Verdict":"direct","TestedAt":5}})").nat_test.present,
          "a refusal carries none");
}

// whoami's RelayDirect: the account's relay switch (true: the player turned
// the relay off), the setup window's Relay row - and a server without it.
void relay_direct() {
    const net::SignInReply off = read(200, R"({"ResKind":0,"OnlineId":"Hunter","ModdedSummons":true,"RelayDirect":true,"NatTest":null})");
    check(off.relay.known && off.relay.direct, "RelayDirect true: the relay is off");
    const net::SignInReply on = read(200, R"({"ResKind":0,"OnlineId":"Hunter","RelayDirect":false})");
    check(on.relay.known && !on.relay.direct, "RelayDirect false: the relay is on");
    check(!read(200, R"({"ResKind":0,"OnlineId":"Hunter"})").relay.known, "a server without the switch: not known");
    check(!read(200, R"({"ResKind":0,"OnlineId":"Hunter","RelayDirect":1})").relay.known, "not a bool: not known");
    check(!read(200, R"({"ResKind":0,"OnlineId":"Hunter","RelayDirect":"true"})").relay.known, "a string: not known");
    check(!read(200, R"({"ResKind":1,"Message":"no such token","RelayDirect":true})").relay.known, "a refusal carries none");
}

// POST /auth/profile {Token, RelayDirect}'s answer: what the account holds
// now, or why not.
void profile_reply() {
    const net::ProfileReply ok = net::read_profile_reply(
        200, R"({"ResKind":0,"OnlineId":"Hunter","ProfilePublic":true,"ShowDiscord":false,"ModdedSummons":true,"RelayDirect":true})", "");
    check(ok.ok && ok.relay.known && ok.relay.direct && ok.why.empty() && ok.status == 200, "taken: the relay is off now");
    const net::ProfileReply back = net::read_profile_reply(200, R"({"ResKind":0,"OnlineId":"Hunter","RelayDirect":false})", "");
    check(back.ok && back.relay.known && !back.relay.direct, "taken: the relay is on now");
    const net::ProfileReply gone = net::read_profile_reply(200, R"({"ResKind":1,"Message":"no such token"})", "");
    check(!gone.ok && gone.refused && gone.why == "no such token", "a token the server no longer knows: refused, a new sign-in needed");
    const net::ProfileReply other = net::read_profile_reply(200, R"({"ResKind":1,"Message":"slow down"})", "");
    check(!other.ok && !other.refused && other.why == "slow down", "another refusal: its words, the sign-in still good");
    const net::ProfileReply none = net::read_profile_reply(0, "", "Couldn't connect to server");
    check(!none.ok && !none.refused && none.why == "Couldn't connect to server" && none.status == 0, "no answer: the transport's words");
    check(net::read_profile_reply(0, "", "").why == "no answer", "no status and no error: no answer");
    const net::ProfileReply old = net::read_profile_reply(404, R"({"detail":"Not Found"})", "");
    check(!old.ok && old.why == "this server has no account settings (HTTP 404)", "a server without /auth/profile");
    check(!net::read_profile_reply(200, "<html>", "").ok, "not JSON");
    const net::ProfileReply busy = net::read_profile_reply(503, R"({"ResKind":0,"RelayDirect":true})", "");
    check(!busy.ok && busy.why == "HTTP 503", "ResKind 0 on a 503 is no answer");
    const net::ProfileReply nosw = net::read_profile_reply(200, R"({"ResKind":0,"OnlineId":"Hunter","ShowDiscord":false})", "");
    check(!nosw.ok && nosw.why == "this server has no relay switch", "taken, but no relay switch to set");
    check(!net::read_profile_reply(200, R"({"ResKind":0,"RelayDirect":"yes"})", "").ok, "RelayDirect not a bool");
    check(!net::read_profile_reply(200, R"({"ResKind":1,"Message":"no such token"})", "Recv failure").refused,
          "a transport error wins over a body");
}

}  // namespace

int main() {
    valid();
    refused();
    unchecked();
    nat_test();
    relay_direct();
    profile_reply();
    if (g_fail) {
        std::printf("%d failed\n", g_fail);
        return 1;
    }
    std::printf("sign_in_check_test: all passed\n");
    return 0;
}
