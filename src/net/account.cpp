#include "net/account.h"

#include "log.h"
#include "net/http.h"

#include <cstdlib>
#include <mutex>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace net {

namespace {
std::mutex g_mu;
std::string g_name, g_token;
bool g_refused = false;  // the server refused this sign-in (account_refused)

// POST to the account server and read the reply whatever the status: a limit
// is a 429 with the reason in Message, which is what the player should see.
bool auth_call(const std::string& path, const json::Value& body, json::Value& reply, std::string& error) {
    const std::string base = auth_server_base();
    const HttpResult r = http_post_json(base + path, body, 8000);
    if (!r.error.empty()) {
        error = r.error;
        return false;
    }
    std::string perr;
    if (!json::parse(r.body, reply, perr)) {
        error = r.status >= 200 && r.status < 300 ? "不正な応答" : "HTTP " + std::to_string(r.status);
        return false;
    }
    // Every account reply carries ResKind. A server without the account
    // service answers its framework's own "not found" instead, which read as
    // a bare "refused" - a tester who started bbhost without the playtest
    // config was sent to the live server, which has no accounts yet.
    if (reply.type != json::Value::Type::Object || !reply.find("ResKind")) {
        error = base + " にアカウントサービスがありません (HTTP " + std::to_string(r.status) + ")";
        host_log("account: %s%s answered HTTP %ld without ResKind", base.c_str(), path.c_str(), r.status);
        return false;
    }
    if (int_of(reply, "ResKind", 1) != 0) {
        error = str_of(reply, "Message", "refused");
        return false;
    }
    return true;
}

bool take_token(const json::Value& reply, std::string& error) {
    const std::string token = str_of(reply, "Token"), id = str_of(reply, "OnlineId");
    if (token.empty() || id.empty()) {
        error = "応答にトークンがありません";
        return false;
    }
    account_set(id, token);
    return true;
}
}  // namespace

std::string account_name() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_name;
}

std::string account_token() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_token;
}

bool account_logged_in() {
    std::lock_guard<std::mutex> lk(g_mu);
    return !g_token.empty() && !g_name.empty();
}

void account_set(const std::string& name, const std::string& token) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_name = name;
    g_token = token;
    g_refused = false;
}

void account_clear() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_name.clear();
    g_token.clear();
    g_refused = false;
}

bool account_refused() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_refused;
}

void account_mark_refused(const std::string& reason) {
    std::string name;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_refused) return;
        g_refused = true;
        name = g_name;
    }
    host_log("account: the server refused this PC's sign-in%s%s (%s): until it is linked again (F10, Account) "
             "nobody can summon or invite this player",
             name.empty() ? "" : " as ", name.c_str(), reason.c_str());
}

std::string account_device_name() {
    std::string host;
#if defined(_WIN32)
    if (const char* c = std::getenv("COMPUTERNAME")) host = c;
#else
    char buf[256] = {};
    if (gethostname(buf, sizeof(buf) - 1) == 0) host = buf;
#endif
    return host.empty() ? "bbhost" : "bbhost on " + host.substr(0, 48);
}

bool account_link_start(DeviceLink& out, std::string& error) {
    json::Value body = json::Value::make_object();
    body.set("DeviceName", account_device_name());
    json::Value reply;
    if (!auth_call("/auth/device/start", body, reply, error)) return false;
    out.device_code = str_of(reply, "DeviceCode");
    out.user_code = str_of(reply, "UserCode");
    out.verification_uri = str_of(reply, "VerificationUri");
    out.interval = static_cast<int>(int_of(reply, "Interval", 5));
    out.expires_in = static_cast<int>(int_of(reply, "ExpiresIn", 600));
    if (out.device_code.empty() || out.user_code.empty()) {
        error = "応答にコードがありません";
        return false;
    }
    if (out.interval < 1) out.interval = 5;
    host_log("account: link code issued, approve at %s", out.verification_uri.c_str());
    return true;
}

std::string account_link_poll(const DeviceLink& link, std::string& error) {
    json::Value body = json::Value::make_object();
    body.set("DeviceCode", link.device_code);
    json::Value reply;
    if (!auth_call("/auth/device/poll", body, reply, error)) return "";
    const std::string status = str_of(reply, "Status");
    if (status == "approved") {
        if (!take_token(reply, error)) return "";
        host_log("account: linked, signed in as %s", account_name().c_str());
    }
    return status;
}

bool account_create(const std::string& name, std::string& recovery_code, std::string& error) {
    json::Value body = json::Value::make_object();
    body.set("名前", name);
    body.set("DeviceName", account_device_name());
    json::Value reply;
    if (!auth_call("/auth/account/create", body, reply, error) || !take_token(reply, error)) return false;
    recovery_code = str_of(reply, "RecoveryCode");
    host_log("account: created %s on this PC", account_name().c_str());
    return true;
}

bool account_recover(const std::string& name, const std::string& code, std::string& new_code, std::string& error) {
    json::Value body = json::Value::make_object();
    body.set("名前", name);
    body.set("RecoveryCode", code);
    json::Value reply;
    if (!auth_call("/auth/recover", body, reply, error) || !take_token(reply, error)) return false;
    new_code = str_of(reply, "RecoveryCode");
    host_log("account: recovered %s", account_name().c_str());
    return true;
}

bool account_web_code(std::string& code, std::string& error) {
    const std::string token = account_token();
    if (token.empty()) {
        error = "未ログイン";
        return false;
    }
    json::Value body = json::Value::make_object();
    body.set("Token", token);
    json::Value reply;
    if (!auth_call("/auth/web_code", body, reply, error)) return false;
    code = str_of(reply, "Code");
    return !code.empty();
}

void account_logout() {
    const std::string token = account_token();
    account_clear();
    if (token.empty()) return;
    json::Value body = json::Value::make_object();
    body.set("Token", token);
    json::Value reply;
    std::string err;
    auth_post("/auth/logout", body, reply, err, 3000);
    host_log("account: signed out");
}

}  // namespace net
