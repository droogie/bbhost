#include "net/http.h"

#include "bbhost_version.h"
#include "core/config.h"
#include "host/plugins.h"
#include "log.h"
#include "net/account.h"

#include <atomic>
#include <mutex>

#if defined(BBHOST_HAVE_CURL)
#include "net/curl_tls.h"
#endif

namespace net {

namespace {

#if defined(BBHOST_HAVE_CURL)
std::once_flag g_once;

std::size_t write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* user) {
    static_cast<std::string*>(user)->append(ptr, size * nmemb);
    return size * nmemb;
}

HttpResult perform(const std::string& url, const std::string* post, int timeout_ms) {
    std::call_once(g_once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    HttpResult r;
    CURL* c = curl_easy_init();
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
    curl_easy_setopt(c, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(c, CURLOPT_USERAGENT, "bbhost/" BBHOST_VERSION);
    configure_curl_tls(c, config().online_verify_tls);
    if (post) {
        curl_easy_setopt(c, CURLOPT_POST, 1L);
        curl_easy_setopt(c, CURLOPT_POSTFIELDS, post->c_str());
        curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(post->size()));
    }
    const CURLcode rc = curl_easy_perform(c);
    if (rc != CURLE_OK) {
        r.error = curl_easy_strerror(rc);
    } else {
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &r.status);
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(c);
    return r;
}
#else
HttpResult perform(const std::string& url, const std::string*, int) {
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

HttpResult http_get(const std::string& url, int timeout_ms) { return perform(url, nullptr, timeout_ms); }

HttpResult http_post_json(const std::string& url, const json::Value& body, int timeout_ms) {
    const std::string text = json::dump(body, 0);
    return perform(url, &text, timeout_ms);
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
