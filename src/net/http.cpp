#include "net/http.h"

#include "bbhost_version.h"
#include "core/config.h"
#include "hle/guest_fs.h"
#include "host/plugins.h"
#include "log.h"
#include "net/account.h"
#include "net/session.h"

#include <atomic>
#include <mutex>
#include <set>

#if defined(BBHOST_HAVE_CURL)
#include <curl/curl.h>
#endif
#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace net {

namespace {

#if defined(BBHOST_HAVE_CURL)
std::once_flag g_once;

// A request from a thread that runs the game's code - an HLE entry's caller,
// which the game waits on - instead of one of ours (the dispatcher, the
// poller, the keepalive, the options' threads): said once per path, so one
// that comes back shows in the first log that has it.
// sceNpMatching2LeaveRoom and sceNpSignalingActivateConnection waited so
// for the server until 2026-10-25.
void note_game_thread(const std::string& url) {
    if (!t_guest_fs || on_dispatcher()) return;
    std::string path = url;
    if (const std::size_t s = path.find("://"); s != std::string::npos) {
        const std::size_t slash = path.find('/', s + 3);
        path = slash == std::string::npos ? "/" : path.substr(slash);
    }
    if (const std::size_t q = path.find('?'); q != std::string::npos) path.resize(q);
    static std::mutex mu;
    static std::set<std::string> said;
    {
        std::lock_guard<std::mutex> lk(mu);
        if (!said.insert(path).second) return;
    }
    char name[32] = "?";
#if !defined(_WIN32)
    pthread_getname_np(pthread_self(), name, sizeof(name));
#endif
    host_log("net: %s asked on the game's thread %s, which waits for the server's answer", path.c_str(), name);
}

std::size_t write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* user) {
    static_cast<std::string*>(user)->append(ptr, size * nmemb);
    return size * nmemb;
}

// HttpCall::abandon, asked by libcurl about once a second while it waits.
int abandon_cb(void* user, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    return (*static_cast<const std::function<bool()>*>(user))() ? 1 : 0;
}

// A thread's kept handle (HttpCall::keep_connection): libcurl keeps the
// connection in it from one request to the next; reset after each request,
// so it holds none of that request's pointers, and closed with the thread.
struct KeptHandle {
    CURL* c = nullptr;
    ~KeptHandle() {
        if (c) curl_easy_cleanup(c);
    }
};

HttpResult perform(const std::string& url, const std::string* post, const HttpCall& call) {
    std::call_once(g_once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    note_game_thread(url);
    HttpResult r;
    thread_local KeptHandle kept;
    CURL* c = nullptr;
    if (call.keep_connection) {
        if (!kept.c) kept.c = curl_easy_init();
        c = kept.c;
    } else {
        c = curl_easy_init();
    }
    if (!c) {
        r.error = "curl_easy_init failed";
        return r;
    }
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Accept: application/json");
    if (post) {
        headers = curl_slist_append(headers, "Content-Type: application/json");
    }
    // The account's token names us to the server.
    if (const std::string token = account_token(); !token.empty()) {
        headers = curl_slist_append(headers, ("X-BB-Token: " + token).c_str());
    }
    // ... and the rules this session plays by, as on the game's own requests
    // (hle/http.cpp, plugins_ruleset).
    headers = curl_slist_append(headers, ("X-BBHost-Ruleset: " + plugins_ruleset()).c_str());
    if (const std::string adopts = plugins_adopts(); !adopts.empty())
        headers = curl_slist_append(headers, ("X-BBHost-Adopt: " + adopts).c_str());
    curl_easy_setopt(c, CURLOPT_URL, url.c_str());
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &r.body);
    curl_easy_setopt(c, CURLOPT_TIMEOUT_MS, static_cast<long>(call.timeout_ms));
    const int connect_ms = call.connect_ms > 0 && call.connect_ms < call.timeout_ms ? call.connect_ms : call.timeout_ms;
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(connect_ms));
    if (call.abandon) {
        curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, abandon_cb);
        curl_easy_setopt(c, CURLOPT_XFERINFODATA, &call.abandon);
    }
    curl_easy_setopt(c, CURLOPT_USERAGENT, "bbhost/" BBHOST_VERSION);
    if (!config().online_verify_tls) {
        curl_easy_setopt(c, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(c, CURLOPT_SSL_VERIFYHOST, 0L);
    }
#if defined(_WIN32)
    // schannel also asks the CA whether the certificate was revoked and fails
    // when that lookup cannot be made (firewalls, captive networks); the chain
    // and name are still checked.
    curl_easy_setopt(c, CURLOPT_SSL_OPTIONS, static_cast<long>(CURLSSLOPT_REVOKE_BEST_EFFORT));
#endif
    if (post) {
        curl_easy_setopt(c, CURLOPT_POST, 1L);
        curl_easy_setopt(c, CURLOPT_POSTFIELDS, post->c_str());
        curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(post->size()));
    }
    const CURLcode rc = curl_easy_perform(c);
    if (rc != CURLE_OK) {
        r.error = curl_easy_strerror(rc);
        r.abandoned = rc == CURLE_ABORTED_BY_CALLBACK;
    } else {
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &r.status);
    }
    if (call.keep_connection) {
        curl_easy_reset(c);  // the options go, the connection stays
    } else {
        curl_easy_cleanup(c);
    }
    curl_slist_free_all(headers);
    return r;
}
#else
HttpResult perform(const std::string& url, const std::string*, const HttpCall&) {
    HttpResult r;
    r.error = "no libcurl at build time";
    (void)url;
    return r;
}
#endif

bool parse_reply(const std::string& what, const HttpResult& r, json::Value& out, std::string& error) {
    static std::atomic<int> logs{0};
    if (!r.error.empty()) {
        error = r.error;
    } else if (r.status < 200 || r.status >= 300) {
        error = "HTTP " + std::to_string(r.status);
        // 401 on the NP surfaces is the server refusing the sign-in, not a
        // passing failure (the server's account-token check): say so
        // once and let the poller slow down (net/account.h).
        if (r.status == 401) {
            json::Value body;
            std::string perr;
            const std::string why = json::parse(r.body, body, perr) ? str_of(body, "Message") : std::string();
            account_mark_refused(why.empty() ? "HTTP 401" : why);
        }
    } else if (!json::parse(r.body, out, error)) {
        error = "bad JSON: " + error;
    } else {
        return true;
    }
    if (logs.fetch_add(1) < 16) {
        host_log("np: %s -> %s", what.c_str(), error.c_str());
    }
    return false;
}

}  // namespace

std::string np_server_base() {
    const HostConfig& c = config();
    if (!c.np_server.empty()) {
        return c.np_server;
    }
    const std::string scheme = c.online_scheme.empty() ? "http" : c.online_scheme;
    const std::string host = c.online_host.empty() ? "thehuntersdream.com" : c.online_host;
    return scheme + "://" + host + ":18671";
}

namespace {
HttpCall timeout_only(int timeout_ms) {
    HttpCall call;
    call.timeout_ms = timeout_ms;
    return call;
}
}  // namespace

HttpResult http_get(const std::string& url, int timeout_ms) { return perform(url, nullptr, timeout_only(timeout_ms)); }

HttpResult http_post_json(const std::string& url, const json::Value& body, int timeout_ms) {
    const std::string text = json::dump(body, 0);
    return perform(url, &text, timeout_only(timeout_ms));
}

std::string auth_server_base() {
    const HostConfig& c = config();
    if (!c.auth_server.empty()) {
        std::string b = c.auth_server;
        while (!b.empty() && b.back() == '/') b.pop_back();
        return b;
    }
    return np_server_base();
}

bool auth_post(const std::string& path, const json::Value& body, json::Value& out, std::string& error,
               int timeout_ms) {
    const HttpResult r = http_post_json(auth_server_base() + path, body, timeout_ms);
    return parse_reply(path, r, out, error);
}

bool np_post(const std::string& path, const json::Value& body, json::Value& out, std::string& error,
             int timeout_ms) {
    const std::string url = np_server_base() + path;
    return parse_reply("POST " + path, http_post_json(url, body, timeout_ms), out, error);
}

bool np_post(const std::string& path, const json::Value& body, json::Value& out, std::string& error,
             const HttpCall& call, long* status) {
    const std::string text = json::dump(body, 0);
    const HttpResult r = perform(np_server_base() + path, &text, call);
    if (status) *status = r.status;
    if (r.abandoned) {
        error = "given up";
        return false;
    }
    return parse_reply("POST " + path, r, out, error);
}

bool np_get(const std::string& path, json::Value& out, std::string& error, int timeout_ms) {
    const std::string url = np_server_base() + path;
    return parse_reply("GET " + path, http_get(url, timeout_ms), out, error);
}

std::string str_of(const json::Value& obj, const char* key, const std::string& dflt) {
    const json::Value* v = obj.type == json::Value::Type::Object ? obj.find(key) : nullptr;
    if (!v) {
        return dflt;
    }
    if (v->type == json::Value::Type::String) {
        return v->string;
    }
    if (v->type == json::Value::Type::Number) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v->number));
        return buf;
    }
    return dflt;
}

long long int_of(const json::Value& obj, const char* key, long long dflt) {
    const json::Value* v = obj.type == json::Value::Type::Object ? obj.find(key) : nullptr;
    if (!v) {
        return dflt;
    }
    if (v->type == json::Value::Type::Number) {
        return static_cast<long long>(v->number);
    }
    if (v->type == json::Value::Type::String) {
        return std::strtoll(v->string.c_str(), nullptr, 10);
    }
    if (v->type == json::Value::Type::Bool) {
        return v->boolean ? 1 : 0;
    }
    return dflt;
}

}  // namespace net
