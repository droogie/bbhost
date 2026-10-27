// sceHttp / sceSsl over libcurl. Objects are integer handles like the SDK's:
// context -> template (UA, headers) -> connection (base URL) -> request.
// Nonblocking requests (the game's mode) run on a worker thread and report
// completion through sceHttpWaitRequest events; blocking ones run inline.
// URLs to the official hosts are rewritten to the private server from the
// config (online.host / online.scheme).
#include "hle/common.h"
#include "core/futex.h"
#include "hle/hle.h"
#include "hle/modules.h"
#include "core/config.h"
#include "bbhost_version.h"
#include "host/plugins.h"
#include "net/account.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#if !defined(_WIN32)
#include <pthread.h>
#endif
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(BBHOST_HAVE_CURL)
#include "net/curl_tls.h"
#endif

namespace {

constexpr int kHttpInvalidId = static_cast<int>(0x80431016u);   // SCE_HTTP_ERROR_INVALID_ID
constexpr int kHttpInvalidValue = static_cast<int>(0x80431019u);
constexpr int kHttpBeforeSend = static_cast<int>(0x80431002u);
constexpr int kHttpEagain = static_cast<int>(0x80431082u);      // SCE_HTTP_ERROR_EAGAIN
constexpr int kHttpNetwork = static_cast<int>(0x80431063u);     // SCE_HTTP_ERROR_NETWORK
constexpr int kHttpTimeout = static_cast<int>(0x80431068u);
constexpr int kHttpBusy = static_cast<int>(0x80431021u);
constexpr std::uint32_t kEvIn = 0x1, kEvOut = 0x2, kEvSockErr = 0x8, kEvHup = 0x10;

// ---- sceSsl: libcurl owns TLS; contexts are just ids.
int g_ssl_next = 1;
std::mutex g_ssl_mu;
std::unordered_set<int> g_ssl_ctx;

GUEST_ABI int hle_ssl_init(std::uint64_t pool) {
    std::lock_guard<std::mutex> lock(g_ssl_mu);
    int id = g_ssl_next++;
    g_ssl_ctx.insert(id);
    host_log("sceSslInit pool=%llu -> %d", static_cast<unsigned long long>(pool), id);
    return id;
}
GUEST_ABI int hle_ssl_term() {
    std::lock_guard<std::mutex> lock(g_ssl_mu);
    g_ssl_ctx.clear();
    return 0;
}

// ---- objects
struct HttpEpoll;
struct NbEvent {
    std::uint32_t events;
    std::uint32_t detail;
    int id;
    void* user;
};
// The lock and the sleep are core/futex.h's: sceHttpWaitRequest's timeout is
// in microseconds, and winpthreads' condition variable returned at once on
// one under a millisecond, the game's wait loop then spinning out the rest.
struct HttpEpoll {
    HostLock mu;
    HostCondVar cv;
    std::deque<NbEvent> events;
    bool aborting = false;
};
enum class HttpKind : int { Ctx = 1, Tmpl, Conn, Req };
struct Response {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    int error = 0;  // SCE error or 0
    long status = 0;
    std::vector<std::uint8_t> body;
    std::size_t read_off = 0;
};
struct HttpObj {
    HttpKind kind = HttpKind::Ctx;
    int parent = 0;
    int nonblock = -1;  // -1 inherit
    HttpEpoll* ep = nullptr;
    void* user = nullptr;
    std::string ua;
    std::string url;
    int method = 0;
    std::vector<std::string> headers;
    std::uint64_t content_len = 0;
    unsigned connect_timeout_us = 0;
    bool sent = false;
    std::shared_ptr<Response> resp;
};
std::mutex g_http_mu;
int g_http_next = 1;
std::unordered_map<int, HttpObj> g_http;
std::unordered_set<HttpEpoll*> g_http_eps;

int http_alloc(HttpKind kind, int parent) {
    const int id = g_http_next++;
    HttpObj o{};
    o.kind = kind;
    o.parent = parent;
    g_http[id] = o;
    return id;
}
HttpObj* http_get(int id, HttpKind kind) {
    auto it = g_http.find(id);
    if (it == g_http.end() || it->second.kind != kind) {
        return nullptr;
    }
    return &it->second;
}
HttpObj* http_any(int id) {
    auto it = g_http.find(id);
    return it == g_http.end() ? nullptr : &it->second;
}

// "Name: value" -> whether its name is `name` (header names are case-insensitive).
bool header_named(const std::string& line, const std::string& name) {
    const auto colon = line.find(':');
    if (colon == std::string::npos || colon != name.size()) {
        return false;
    }
    for (std::size_t i = 0; i < colon; i++) {
        if (std::tolower(static_cast<unsigned char>(line[i])) != std::tolower(static_cast<unsigned char>(name[i]))) {
            return false;
        }
    }
    return true;
}
std::string header_name(const std::string& line) {
    return line.substr(0, line.find(':'));
}

// Walk request -> connection -> template collecting inherited settings.
struct Effective {
    std::string ua;
    std::vector<std::string> headers;
    bool nonblock = false;
    HttpEpoll* ep = nullptr;
    void* user = nullptr;
    unsigned connect_timeout_us = 0;
};
Effective effective_locked(int req_id) {
    Effective e;
    int nb = -1;
    for (int id = req_id; id;) {
        HttpObj* o = http_any(id);
        if (!o) {
            break;
        }
        // A nearer object's header wins over a parent's of the same name
        // (one object's own repeats, added with SCE_HTTP_HEADER_ADD, stay).
        const std::vector<std::string> nearer = e.headers;
        for (auto it = o->headers.rbegin(); it != o->headers.rend(); ++it) {
            const std::string name = header_name(*it);
            if (std::any_of(nearer.begin(), nearer.end(), [&](const std::string& h) { return header_named(h, name); })) {
                continue;
            }
            e.headers.insert(e.headers.begin(), *it);
        }
        if (e.ua.empty()) {
            e.ua = o->ua;
        }
        if (nb < 0 && o->nonblock >= 0) {
            nb = o->nonblock;
        }
        if (!e.ep && o->ep) {
            e.ep = o->ep;
            e.user = o->user;
        }
        if (!e.connect_timeout_us && o->connect_timeout_us) {
            e.connect_timeout_us = o->connect_timeout_us;
        }
        id = o->parent;
    }
    e.nonblock = nb > 0;
    return e;
}

// online.host replaces the official hostnames; online.scheme forces http/https.
std::string rewrite_url(const std::string& url) {
    const HostConfig& cfg = config();
    std::size_t p = url.find("://");
    if (p == std::string::npos) {
        return url;
    }
    std::string scheme = url.substr(0, p);
    std::size_t host_start = p + 3;
    std::size_t host_end = url.find_first_of(":/", host_start);
    if (host_end == std::string::npos) {
        host_end = url.size();
    }
    std::string host = url.substr(host_start, host_end - host_start);
    std::string rest = url.substr(host_end);
    // The game uploads its play logs to FromSoftware's S3 buckets
    // (https://bb-playlog-{test,prod}.s3.amazonaws.com/<date>/<file>, PUT).
    // They go to the private server's game port instead, which stores them
    // (its PUT /{date}/{file}); left alone they went to Amazon under bucket
    // names we do not own.
    if (host == "bb-playlog-test.s3.amazonaws.com" || host == "bb-playlog-prod.s3.amazonaws.com") {
        const std::string to = cfg.online_host.empty() ? "thehuntersdream.com" : cfg.online_host;
        const std::string sch = cfg.online_scheme.empty() ? "http" : cfg.online_scheme;
        std::size_t path = rest.find('/');
        return sch + "://" + to + ":18671" + (path == std::string::npos ? "/" : rest.substr(path));
    }
    bool official = host.size() >= 15 && host.compare(host.size() - 15, 15, "scej-network.jp") == 0;
    if (official && !cfg.online_host.empty()) {
        host = cfg.online_host;
    }
    if (official && !cfg.online_scheme.empty()) {
        scheme = cfg.online_scheme;
    }
    return scheme + "://" + host + rest;
}

#if defined(BBHOST_HAVE_CURL)
std::once_flag g_curl_once;
std::size_t write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* user) {
    auto* body = static_cast<std::vector<std::uint8_t>*>(user);
    body->insert(body->end(), reinterpret_cast<std::uint8_t*>(ptr), reinterpret_cast<std::uint8_t*>(ptr) + size * nmemb);
    return size * nmemb;
}
#endif

// The calling thread's name, for the log.
std::string thread_name_now() {
#if !defined(_WIN32)
    char n[32] = {};
    pthread_getname_np(pthread_self(), n, sizeof(n));
    return n;
#else
    return "?";
#endif
}

// BBHOST_HTTP_DELAY_MS (tests): every response held back that long before it
// counts as arrived - a server on this machine answers in a millisecond, the
// real one 23 ms and a TLS handshake away, and a caller that waits for the
// answer only shows against the second.
const int g_http_delay_ms = [] {
    const char* e = std::getenv("BBHOST_HTTP_DELAY_MS");
    return e ? std::atoi(e) : 0;
}();

void perform(const std::string& url, int method, const Effective& eff, std::vector<std::uint8_t> post,
             std::uint64_t content_len, std::shared_ptr<Response> resp, bool rewritten) {
    const auto t0 = std::chrono::steady_clock::now();
#if defined(BBHOST_HAVE_CURL)
    std::call_once(g_curl_once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    CURL* c = curl_easy_init();
    long status = 0;
    int err = 0;
    std::vector<std::uint8_t> body;
    if (!c) {
        err = kHttpNetwork;
    } else {
        curl_easy_setopt(c, CURLOPT_URL, url.c_str());
        curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
        curl_easy_setopt(c, CURLOPT_WRITEDATA, &body);
        curl_easy_setopt(c, CURLOPT_TIMEOUT, 30L);
        // The game's own connect timeout (its config value x1000, as usec) is
        // too short for a host across the internet: a loopback server connects
        // inside it, dev.thehuntersdream.com (23 ms away) never did - four
        // ss.info fetches timed out within seconds and nginx saw no request
        // (sceHttpSetConnectTimeOut logs the value it asks for). A PS4 on a real
        // network reached Sony's servers with the same value, so it is a floor
        // of what the game can wait, not a deadline; hold it to kMinConnectMs.
        constexpr long kMinConnectMs = 5000;
        const long want_ms = eff.connect_timeout_us ? static_cast<long>(eff.connect_timeout_us / 1000) : 10000;
        curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, want_ms < kMinConnectMs ? kMinConnectMs : want_ms);
        net::configure_curl_tls(c, config().online_verify_tls);
        // The game's own user agent with bbhost's name and version after it,
        // so a server can tell bbhost clients and their versions apart. The
        // game's text stays the prefix, which is what anything matching on
        // it looks at.
        const std::string ua = (eff.ua.empty() ? std::string() : eff.ua + " ") + "bbhost/" + BBHOST_VERSION;
        curl_easy_setopt(c, CURLOPT_USERAGENT, ua.c_str());
        struct curl_slist* list = nullptr;
        for (const auto& h : eff.headers) {
            const bool is_ua = h.size() > 11 && std::equal(h.begin(), h.begin() + 11, "user-agent:", [](char a, char b) {
                                   return std::tolower(static_cast<unsigned char>(a)) == b;
                               });
            if (is_ua) {
                list = curl_slist_append(list, (h + " bbhost/" + BBHOST_VERSION).c_str());  // one the game set itself
                continue;
            }
            list = curl_slist_append(list, h.c_str());
        }
        list = curl_slist_append(list, "Expect:");
        // The game's own traffic carries the account's token too, so the
        // server's game sessions map to the account. Every host:
        // the game reaches the private server both through rewritten
        // official names and through the addresses ss.info hands it.
        (void)rewritten;
        if (const std::string token = net::account_token(); !token.empty()) {
            list = curl_slist_append(list, ("X-BB-Token: " + token).c_str());
        }
        // ... and the rules this session plays by (plugins_ruleset): the server
        // keeps a randomizer or boss-rush run off the normal map and stats, and
        // shows bloodstains and messages only within the same ruleset and seed.
        list = curl_slist_append(list, ("X-BBHost-Ruleset: " + plugins_ruleset()).c_str());
        // The plugins that can play another player's world (an adopting
        // guest may join a host of other rules).
        if (const std::string adopts = plugins_adopts(); !adopts.empty())
            list = curl_slist_append(list, ("X-BBHost-Adopt: " + adopts).c_str());
        curl_easy_setopt(c, CURLOPT_HTTPHEADER, list);
        // Methods as the game numbers them: 0 GET, 1 POST, 2 HEAD, 4 PUT (the
        // play-log upload). PUT sent as a GET lost its body.
        if (method == 1 || method == 4) {
            curl_easy_setopt(c, CURLOPT_POST, 1L);
            if (method == 4) curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, "PUT");
            curl_easy_setopt(c, CURLOPT_POSTFIELDS, post.data());
            curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE_LARGE,
                             static_cast<curl_off_t>(post.empty() ? content_len : post.size()));
        }
        CURLcode rc = curl_easy_perform(c);
        if (rc != CURLE_OK) {
            err = rc == CURLE_OPERATION_TIMEDOUT ? kHttpTimeout : kHttpNetwork;
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 8) {
                host_log("http: %s -> %s", url.c_str(), curl_easy_strerror(rc));
            }
        } else {
            curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &status);
        }
        curl_slist_free_all(list);
        curl_easy_cleanup(c);
    }
#else
    (void)method;
    (void)eff;
    (void)post;
    (void)content_len;
    long status = 0;
    int err = kHttpNetwork;
    std::vector<std::uint8_t> body;
    host_log("http: no libcurl at build time; %s fails", url.c_str());
#endif
    if (g_http_delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(g_http_delay_ms));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 24) {
        host_log("http: %s %s -> %ld (%zu bytes, %.0f ms)%s", method == 1 ? "POST" : method == 4 ? "PUT" : "GET", url.c_str(), status,
                 body.size(), ms, err ? " error" : "");
    }
    std::lock_guard<std::mutex> lk(resp->mu);
    resp->status = status;
    resp->error = err;
    resp->body = std::move(body);
    resp->done = true;
    resp->cv.notify_all();
}

void post_event(HttpEpoll* ep, int id, void* user, std::uint32_t events, std::uint32_t detail) {
    if (!ep) {
        return;
    }
    std::lock_guard<HostLock> lk(ep->mu);
    ep->events.push_back(NbEvent{events, detail, id, user});
    ep->cv.notify_all();
}

// ---- entry points
GUEST_ABI int hle_http_init(int net_mem, int ssl_ctx, std::uint64_t pool) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    const int id = http_alloc(HttpKind::Ctx, 0);
    host_log("sceHttpInit net=%d ssl=%d pool=%llu -> %d", net_mem, ssl_ctx, static_cast<unsigned long long>(pool), id);
    return id;
}
GUEST_ABI int hle_http_term() {
    std::lock_guard<std::mutex> lock(g_http_mu);
    g_http.clear();
    return 0;
}
GUEST_ABI int hle_http_create_template(int ctx, const char* ua, int ver, int proxy) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    if (!http_get(ctx, HttpKind::Ctx)) {
        return kHttpInvalidId;
    }
    const int id = http_alloc(HttpKind::Tmpl, ctx);
    g_http[id].ua = ua ? ua : "";
    host_log("sceHttpCreateTemplate ctx=%d ua=%s ver=%d proxy=%d -> %d", ctx, ua ? ua : "", ver, proxy, id);
    return id;
}
GUEST_ABI int hle_http_delete_template(int id) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    g_http.erase(id);
    return 0;
}
GUEST_ABI int hle_http_set_nonblock(int id, int enable) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    HttpObj* o = http_any(id);
    if (!o) {
        return kHttpInvalidId;
    }
    o->nonblock = enable ? 1 : 0;
    return 0;
}
GUEST_ABI int hle_http_create_epoll(int ctx, HttpEpoll** out) {
    if (!out) {
        return kHttpInvalidValue;
    }
    std::lock_guard<std::mutex> lock(g_http_mu);
    if (!http_get(ctx, HttpKind::Ctx)) {
        return kHttpInvalidId;
    }
    auto* ep = new HttpEpoll();
    g_http_eps.insert(ep);
    *out = ep;
    host_log("sceHttpCreateEpoll ctx=%d -> %p", ctx, static_cast<void*>(ep));
    return 0;
}
GUEST_ABI int hle_http_destroy_epoll(int, HttpEpoll* ep) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    if (!g_http_eps.erase(ep)) {
        return kHttpInvalidValue;
    }
    for (auto& kv : g_http) {
        if (kv.second.ep == ep) {
            kv.second.ep = nullptr;
        }
    }
    // Waiters hold g_http_mu only briefly; mark and leak rather than free under them.
    {
        std::lock_guard<HostLock> lk(ep->mu);
        ep->aborting = true;
        ep->cv.notify_all();
    }
    return 0;
}
// int sceHttpAddRequestHeader(int id, const char* name, const char* value, int mode)
// mode 0 (SCE_HTTP_HEADER_OVERWRITE) replaces a header of that name, 1
// (SCE_HTTP_HEADER_ADD) adds another. The play-log uploader sets its
// Authorization again before every PUT on one long-lived object: appending
// sent 2, 3, 4... Authorization lines, and nginx answers that with 400.
GUEST_ABI int hle_http_add_header(int id, const char* name, const char* value, unsigned mode) {
    if (!name) {
        return kHttpInvalidValue;
    }
    std::lock_guard<std::mutex> lock(g_http_mu);
    HttpObj* o = http_any(id);
    if (!o) {
        return kHttpInvalidId;
    }
    if (mode != 1) {
        const std::string n(name);
        std::erase_if(o->headers, [&](const std::string& h) { return header_named(h, n); });
    }
    o->headers.push_back(std::string(name) + ": " + (value ? value : ""));
    return 0;
}
// int sceHttpWaitRequest(SceHttpEpollHandle, SceHttpNBEvent* out, int maxevents, int timeout_us)
GUEST_ABI int hle_http_wait_request(HttpEpoll* ep, NbEvent* out, int max_events, int timeout) {
    if (!ep || !out || max_events <= 0) {
        return kHttpInvalidValue;
    }
    {
        std::lock_guard<std::mutex> lock(g_http_mu);
        if (!g_http_eps.count(ep)) {
            return kHttpInvalidValue;
        }
    }
    std::unique_lock<HostLock> lk(ep->mu);
    auto ready = [&] { return !ep->events.empty() || ep->aborting; };
    if (timeout < 0) {
        ep->cv.wait(lk, ready);
    } else if (timeout > 0) {
        ep->cv.wait_for(lk, std::chrono::microseconds(timeout), ready);
    }
    if (ep->aborting) {
        ep->aborting = false;
        return 0;
    }
    int n = 0;
    while (n < max_events && !ep->events.empty()) {
        out[n++] = ep->events.front();
        ep->events.pop_front();
    }
    return n;
}
GUEST_ABI int hle_http_abort_wait(HttpEpoll* ep) {
    if (!ep) {
        return kHttpInvalidValue;
    }
    std::lock_guard<HostLock> lk(ep->mu);
    ep->aborting = true;
    ep->cv.notify_all();
    return 0;
}
GUEST_ABI int hle_https_enable(int, unsigned) { return 0; }
GUEST_ABI int hle_https_disable(int, unsigned) { return 0; }

GUEST_ABI int hle_http_connect_url(int tmpl, const char* url, int) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    if (!http_get(tmpl, HttpKind::Tmpl)) {
        return kHttpInvalidId;
    }
    const int id = http_alloc(HttpKind::Conn, tmpl);
    g_http[id].url = url ? url : "";
    return id;
}
GUEST_ABI int hle_http_request_url(int conn, int method, const char* url, std::uint64_t len) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    if (!http_get(conn, HttpKind::Conn)) {
        return kHttpInvalidId;
    }
    const int id = http_alloc(HttpKind::Req, conn);
    HttpObj& r = g_http[id];
    r.method = method;
    r.url = url ? url : "";
    r.content_len = len;
    r.resp = std::make_shared<Response>();
    return id;
}
GUEST_ABI int hle_http_set_content_len(int id, std::uint64_t len) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    HttpObj* r = http_get(id, HttpKind::Req);
    if (!r) {
        return kHttpInvalidId;
    }
    r->content_len = len;
    return 0;
}
GUEST_ABI int hle_http_set_connect_timeout(int id, unsigned usec) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    HttpObj* o = http_any(id);
    if (!o) {
        return kHttpInvalidId;
    }
    o->connect_timeout_us = usec;
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 4) {
        host_log("sceHttpSetConnectTimeOut id=%d usec=%u", id, usec);
    }
    return 0;
}
GUEST_ABI int hle_http_set_epoll(int id, HttpEpoll* ep, void* user) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    HttpObj* o = http_any(id);
    if (!o) {
        return kHttpInvalidId;
    }
    o->ep = ep;
    o->user = user;
    return 0;
}
GUEST_ABI int hle_http_unset_epoll(int id) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    HttpObj* o = http_any(id);
    if (!o) {
        return kHttpInvalidId;
    }
    o->ep = nullptr;
    return 0;
}
GUEST_ABI int hle_http_delete_request(int id) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    HttpObj* r = http_get(id, HttpKind::Req);
    if (!r) {
        return kHttpInvalidId;
    }
    g_http.erase(id);
    return 0;
}
GUEST_ABI int hle_http_delete_connection(int id) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    g_http.erase(id);
    return 0;
}

// int sceHttpSendRequest(int reqId, const void* postData, size_t size)
GUEST_ABI int hle_http_send(int id, const void* post, std::uint64_t size) {
    std::string url;
    int method;
    Effective eff;
    std::uint64_t content_len;
    std::shared_ptr<Response> resp;
    bool rewritten = false;
    {
        std::lock_guard<std::mutex> lock(g_http_mu);
        HttpObj* r = http_get(id, HttpKind::Req);
        if (!r) {
            return kHttpInvalidId;
        }
        if (r->sent) {
            return kHttpBusy;
        }
        r->sent = true;
        url = rewrite_url(r->url);
        rewritten = url != r->url;
        method = r->method;
        content_len = r->content_len;
        eff = effective_locked(id);
        resp = r->resp;
    }
    std::vector<std::uint8_t> body;
    if (post && size) {
        body.assign(static_cast<const std::uint8_t*>(post), static_cast<const std::uint8_t*>(post) + size);
    }
    if (eff.nonblock) {
        HttpEpoll* ep = eff.ep;
        void* user = eff.user;
        std::thread([=] {
            perform(url, method, eff, body, content_len, resp, rewritten);
            std::uint32_t ev = kEvOut | kEvIn;
            std::uint32_t detail = 0;
            {
                std::lock_guard<std::mutex> lk(resp->mu);
                if (resp->error) {
                    ev = kEvSockErr | kEvHup;
                    detail = static_cast<std::uint32_t>(resp->error);
                }
            }
            post_event(ep, id, user, ev, detail);
        }).detach();
        return 0;
    }
    // Blocking: the whole request on the caller's thread.
    const auto t0 = std::chrono::steady_clock::now();
    perform(url, method, eff, body, content_len, resp, rewritten);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static std::atomic<int> logs{0};
    if (ms >= 5.0 && logs.fetch_add(1) < 32) {
        const std::string who = thread_name_now();
        host_log("http: a blocking sceHttpSendRequest (request %d) held thread %s for %.0f ms", id, who.c_str(), ms);
    }
    std::lock_guard<std::mutex> lk(resp->mu);
    return resp->error;
}

std::shared_ptr<Response> resp_of(int id, bool* nonblock = nullptr) {
    std::lock_guard<std::mutex> lock(g_http_mu);
    HttpObj* r = http_get(id, HttpKind::Req);
    if (nonblock) *nonblock = r && effective_locked(id).nonblock;
    return r && r->sent ? r->resp : nullptr;
}

// A request sent non-blocking answers SCE_HTTP_ERROR_EAGAIN here until its
// response has come, as the PS4's library does: the game sends its requests
// that way and asks for the status from the main loop, once a frame, until it
// has one. Waiting instead held the main loop for every request's round trip
// - each play-log upload (every 5 s since the private server's map wanted
// them often), message and ghost fetch: 111-145 ms frames against a server
// 23 ms and a TLS handshake away. A blocking request waits (up to 35 s); a
// wait of 5 ms or more is logged with the calling thread (the first 32).
void wait_done(std::unique_lock<std::mutex>& lk, const std::shared_ptr<Response>& resp, const char* fn, int id, bool nonblock) {
    if (resp->done || nonblock) return;
    const auto t0 = std::chrono::steady_clock::now();
    resp->cv.wait_for(lk, std::chrono::seconds(35), [&] { return resp->done; });
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static std::atomic<int> logs{0};
    if (ms >= 5.0 && logs.fetch_add(1) < 32) {
        const std::string who = thread_name_now();
        host_log("http: %s waited %.0f ms for request %d's response on thread %s", fn, ms, id, who.c_str());
    }
}

GUEST_ABI int hle_http_status(int id, int* code) {
    if (!code) {
        return kHttpInvalidValue;
    }
    bool nonblock = false;
    auto resp = resp_of(id, &nonblock);
    if (!resp) {
        return kHttpBeforeSend;
    }
    std::unique_lock<std::mutex> lk(resp->mu);
    wait_done(lk, resp, "sceHttpGetStatusCode", id, nonblock);
    if (!resp->done) {
        return kHttpEagain;
    }
    if (resp->error) {
        return resp->error;
    }
    *code = static_cast<int>(resp->status);
    return 0;
}
GUEST_ABI int hle_http_resp_len(int id, std::uint64_t* len) {
    if (!len) {
        return kHttpInvalidValue;
    }
    bool nonblock = false;
    auto resp = resp_of(id, &nonblock);
    if (!resp) {
        return kHttpBeforeSend;
    }
    std::unique_lock<std::mutex> lk(resp->mu);
    wait_done(lk, resp, "sceHttpGetResponseContentLength", id, nonblock);
    if (!resp->done) {
        return kHttpEagain;
    }
    *len = resp->body.size();
    return 0;  // SCE_HTTP_CONTENTLEN_EXIST
}
GUEST_ABI int hle_http_read(int id, void* data, std::uint64_t size) {
    if (!data) {
        return kHttpInvalidValue;
    }
    bool nonblock = false;
    auto resp = resp_of(id, &nonblock);
    if (!resp) {
        return kHttpBeforeSend;
    }
    std::unique_lock<std::mutex> lk(resp->mu);
    wait_done(lk, resp, "sceHttpReadData", id, nonblock);
    if (!resp->done) {
        return kHttpEagain;
    }
    if (resp->error) {
        return resp->error;
    }
    const std::size_t avail = resp->body.size() - resp->read_off;
    const std::size_t n = static_cast<std::size_t>(size < avail ? size : avail);
    std::memcpy(data, resp->body.data() + resp->read_off, n);
    resp->read_off += n;
    return static_cast<int>(n);
}

}  // namespace

void hle_register_http() {
#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
    REG("sceSslInit", hle_ssl_init);
    REG("sceSslTerm", hle_ssl_term);
    REG("sceHttpInit", hle_http_init);
    REG("sceHttpTerm", hle_http_term);
    REG("sceHttpCreateTemplate", hle_http_create_template);
    REG("sceHttpDeleteTemplate", hle_http_delete_template);
    REG("sceHttpSetNonblock", hle_http_set_nonblock);
    REG("sceHttpCreateEpoll", hle_http_create_epoll);
    REG("sceHttpDestroyEpoll", hle_http_destroy_epoll);
    REG("sceHttpAddRequestHeader", hle_http_add_header);
    REG("sceHttpWaitRequest", hle_http_wait_request);
    REG("sceHttpAbortWaitRequest", hle_http_abort_wait);
    REG("sceHttpsEnableOption", hle_https_enable);
    REG("sceHttpsDisableOption", hle_https_disable);
    REG("sceHttpCreateConnectionWithURL", hle_http_connect_url);
    REG("sceHttpCreateRequestWithURL", hle_http_request_url);
    REG("sceHttpSetRequestContentLength", hle_http_set_content_len);
    REG("sceHttpSetConnectTimeOut", hle_http_set_connect_timeout);
    REG("sceHttpSetEpoll", hle_http_set_epoll);
    REG("sceHttpUnsetEpoll", hle_http_unset_epoll);
    REG("sceHttpDeleteRequest", hle_http_delete_request);
    REG("sceHttpDeleteConnection", hle_http_delete_connection);
    REG("sceHttpSendRequest", hle_http_send);
    REG("sceHttpGetStatusCode", hle_http_status);
    REG("sceHttpGetResponseContentLength", hle_http_resp_len);
    REG("sceHttpReadData", hle_http_read);
#undef REG
}
