// A small synchronous HTTP client for the host's own traffic to the private
// server (the session service). The game's own sceHttp calls go
// through hle/http.cpp; this one never touches guest state and may be used
// from any host thread.
#pragma once

#include "replay/json.h"

#include <functional>
#include <string>

namespace net {

struct HttpResult {
    long status = 0;      // HTTP status, 0 when the transport failed
    std::string body;
    std::string error;    // transport error text, empty on success
    bool abandoned = false;  // HttpCall::abandon ended it
    bool ok() const { return error.empty() && status >= 200 && status < 300; }
};

// A request with its own limits, for the one that waits (the event poller's
// long poll, net/session.cpp); every other request is a timeout alone.
struct HttpCall {
    int timeout_ms = 5000;  // the whole exchange
    // The connection within it - name lookup, TCP, TLS - apart from the wait
    // for the answer; 0: timeout_ms, as for every other request.
    int connect_ms = 0;
    // Asked about once a second while the request is out; true gives it up.
    std::function<bool()> abandon;
    // The calling thread's own connection, kept open for its next request
    // that sets this too (HTTP keep-alive), instead of a new connection and
    // TLS handshake every time.
    bool keep_connection = false;
    // Over IPv4 only: the connection test's request, which the server matches
    // by address with the test's datagrams (IPv4 alone; net/nat_check.h).
    bool ipv4_only = false;
};

// online.np_server from the config, else "<online.scheme>://<online.host>:18671".
std::string np_server_base();
// online.auth_server, else np_server_base(): the origin the account calls go to.
std::string auth_server_base();
// POST to "<auth_server_base><path>", otherwise as np_post.
bool auth_post(const std::string& path, const json::Value& body, json::Value& out, std::string& error,
               int timeout_ms = 8000);

HttpResult http_get(const std::string& url, int timeout_ms = 5000);
HttpResult http_post_json(const std::string& url, const json::Value& body, int timeout_ms = 5000);
// The same with a call's own limits (HttpCall).
HttpResult http_post_json(const std::string& url, const json::Value& body, const HttpCall& call);

// POST/GET to "<np_server_base><path>" and parse the JSON reply. Returns
// false (with `error`) on a transport error, a non-2xx status, or a body
// that is not JSON; `out` holds the parsed object on success.
bool np_post(const std::string& path, const json::Value& body, json::Value& out, std::string& error,
             int timeout_ms = 5000);
// The same with a call's own limits; `status` gets the HTTP status (0: no
// answer came). A request `call.abandon` gave up is not logged as a failure.
bool np_post(const std::string& path, const json::Value& body, json::Value& out, std::string& error,
             const HttpCall& call, long* status = nullptr);
bool np_get(const std::string& path, json::Value& out, std::string& error, int timeout_ms = 5000);

// Reads that never throw: the member or a default.
std::string str_of(const json::Value& obj, const char* key, const std::string& dflt = "");
long long int_of(const json::Value& obj, const char* key, long long dflt = 0);

}  // namespace net
