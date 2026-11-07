// The sign-in check's reading of the server's answer to POST /auth/whoami
// (net/sign_in_check.h): a token the account service knows, one it does not
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

}  // namespace

int main() {
    valid();
    refused();
    unchecked();
    if (g_fail) {
        std::printf("%d failed\n", g_fail);
        return 1;
    }
    std::printf("sign_in_check_test: all passed\n");
    return 0;
}
