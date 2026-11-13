#include "net/nat_job.h"

#include "core/config.h"
#include "log.h"
#include "net/account.h"
#include "net/http.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include <chrono>
#include <cstring>

namespace net::nat {

namespace {

std::mutex g_remembered_mu;
Remembered g_remembered;

double unix_now() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// The test server's IPv4 address (network byte order), 0 when the name does
// not resolve.
std::uint32_t resolve_ipv4(const std::string& host) {
#if defined(_WIN32)
    WSADATA wsa{};
    const bool wsa_ok = ::WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
#endif
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* res = nullptr;
    std::uint32_t addr = 0;
    if (!host.empty() && ::getaddrinfo(host.c_str(), nullptr, &hints, &res) == 0 && res) {
        sockaddr_in sa{};
        std::memcpy(&sa, res->ai_addr, sizeof(sa));
        addr = sa.sin_addr.s_addr;
        ::freeaddrinfo(res);
    }
#if defined(_WIN32)
    if (wsa_ok) ::WSACleanup();
#endif
    return addr;
}

}  // namespace

std::string test_host() {
    std::string spec = config().stun_server;
    if (spec == "off") spec.clear();
    if (spec.empty()) {
        spec = np_server_base();
        if (const std::size_t s = spec.find("://"); s != std::string::npos) spec = spec.substr(s + 3);
        if (const std::size_t slash = spec.find('/'); slash != std::string::npos) spec.resize(slash);
    }
    if (const std::size_t colon = spec.find(':'); colon != std::string::npos) spec.resize(colon);
    return spec;
}

void remember(const Remembered& r) {
    std::lock_guard<std::mutex> lk(g_remembered_mu);
    g_remembered = r;
}

Remembered remembered() {
    std::lock_guard<std::mutex> lk(g_remembered_mu);
    return g_remembered;
}

Job::~Job() {
    cancel();
    if (thread_.joinable()) thread_.join();
}

bool Job::start(const JobParams& p) {
    std::lock_guard<std::mutex> lk(mu_);
    if (view_.running) return false;
    if (thread_.joinable()) thread_.join();  // the last one has ended (running is false)
    cancel_.store(false);
    view_.running = true;
    view_.finished = false;
    view_.step = "Sending test packets...";
    thread_ = std::thread([this, p] { run(p); });
    return true;
}

Job::View Job::view() const {
    std::lock_guard<std::mutex> lk(mu_);
    return view_;
}

void Job::cancel() { cancel_.store(true); }

void Job::run(JobParams p) {
    Outcome o;
    o.probe.local_port = p.p2p_port;
    const std::uint32_t server = resolve_ipv4(p.host);
    o.resolved = server != 0;
    if (o.resolved) {
        ProbeParams pp;
        pp.server = server;
        pp.ports = p.ports;
        pp.local_port = static_cast<std::uint16_t>(p.p2p_port);
        pp.cancelled = [this] { return cancel_.load(); };
        o.probe = probe(pp);
        o.local = verdict_of(o.probe);
        o.inbound = inbound_of(o.probe);
    }
    // Signed in, and something went out: the server's own verdict, kept for
    // the account (and the relay's marks) as the website's is.
    o.signed_in = !p.token.empty() && !p.save_url.empty();
    if (o.resolved && o.probe.error == ProbeResult::Error::None && o.probe.sent > 0 && o.signed_in && !cancel_.load()) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            view_.step = "Asking the server what it saw...";
        }
        json::Value body = json::Value::make_object();
        body.set("Token", p.token);
        if (const char* w = inbound_word(o.inbound); *w) body.set("Inbound", w);
        body.set("Port", p.p2p_port);
        HttpCall call;
        call.timeout_ms = 8000;
        call.ipv4_only = true;
        call.abandon = [this] { return cancel_.load(); };
        const HttpResult r = http_post_json(p.save_url, body, call);
        o.save = read_save_reply(r.status, r.body, r.abandoned ? std::string("given up") : r.error);
        if (o.save.state == SaveReply::State::Refused) account_mark_refused(o.save.message);
    }
    o.at = unix_now();
    host_log("%s", log_line(o).c_str());
    if (o.probe.error == ProbeResult::Error::None && o.resolved && o.shown() != Verdict::None) {
        const Card c = card_of(o, o.at);
        remember(Remembered{true, o.shown(), o.inbound, o.at, o.saved(), c.head, c.meaning});
    }
    std::lock_guard<std::mutex> lk(mu_);
    view_.outcome = std::move(o);
    view_.running = false;
    view_.finished = true;
    ++view_.tests;
    view_.step = "";
}

}  // namespace net::nat
