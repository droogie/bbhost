#include "host/updater.h"

#include "bbhost_version.h"
#include "core/config.h"
#include "log.h"
#include "replay/json.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>

#if defined(BBHOST_HAVE_CURL)
#include <curl/curl.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#endif
#if defined(BBHOST_HAVE_SDL3)
#include <SDL3/SDL.h>
#endif

namespace updater {

namespace {

namespace fs = std::filesystem;

// The release signing key's public half. The private half signs each
// release's SHA256SUMS and the official plugins in the release workflow (the
// RELEASE_SIGNING_KEY secret); no official plugin it did not sign is loaded.
constexpr std::uint8_t kReleaseKey[32] = {
    0xce, 0x83, 0x68, 0x4e, 0xe9, 0x4e, 0xc4, 0xb9, 0xc9, 0x1c, 0xf6, 0x24, 0x7a, 0x23, 0xd0, 0x8f,
    0x50, 0x1c, 0xdd, 0x7a, 0x19, 0x73, 0x3c, 0x83, 0x48, 0xca, 0x3e, 0xd6, 0xae, 0xa7, 0xad, 0x62,
};
}  // namespace

const std::uint8_t* release_key() { return kReleaseKey; }

namespace {
constexpr const char* kDefaultSource = "https://api.github.com/repos/droogie/bbhost/releases/latest";
constexpr std::size_t kMaxDownload = 256u << 20;

std::mutex g_mu;
std::string g_status, g_detail;  // under g_mu
std::string g_tag, g_page;       // under g_mu: the newer release and its page, when there is one
std::atomic<bool> g_busy{false}, g_checked{false}, g_available{false};

void set_lines(const std::string& status, const std::string& detail) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_status = status;
    g_detail = detail;
}

fs::path exe_path() {
#if defined(_WIN32)
    std::wstring buf(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    buf.resize(n);
    return fs::path(buf);
#else
    std::error_code ec;
    return fs::read_symlink("/proc/self/exe", ec);
#endif
}

std::string read_token() {
    const std::string& file = config().update_token_file;
    if (file.empty()) return {};
    std::ifstream in(file, std::ios::binary);
    std::string t((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ')) t.pop_back();
    return t;
}

struct Fetch {
    long status = 0;
    std::string body, error;
    bool ok() const { return error.empty() && status >= 200 && status < 300; }
};

#if defined(BBHOST_HAVE_CURL)
std::size_t write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* user) {
    auto* s = static_cast<std::string*>(user);
    if (s->size() + size * nmemb > kMaxDownload) return 0;  // ends the transfer
    s->append(ptr, size * nmemb);
    return size * nmemb;
}
#endif

// A GET to GitHub. Only the update token goes with it (and only to
// api.github.com: curl leaves a custom Authorization header behind when a
// redirect takes it to another host, which is where asset downloads go).
Fetch fetch(const std::string& url, bool binary, int timeout_ms, const std::string& token) {
    Fetch r;
#if defined(BBHOST_HAVE_CURL)
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    CURL* c = curl_easy_init();
    if (!c) {
        r.error = "curl_easy_init failed";
        return r;
    }
    struct curl_slist* h = nullptr;
    h = curl_slist_append(h, binary ? "Accept: application/octet-stream" : "Accept: application/vnd.github+json");
    h = curl_slist_append(h, "X-GitHub-Api-Version: 2022-11-28");
    if (!token.empty() && url.rfind("https://api.github.com/", 0) == 0) {
        h = curl_slist_append(h, ("Authorization: Bearer " + token).c_str());
    }
    const std::string agent = std::string("bbhost/") + BBHOST_VERSION;
    curl_easy_setopt(c, CURLOPT_URL, url.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, h);
    curl_easy_setopt(c, CURLOPT_USERAGENT, agent.c_str());
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, 15000L);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &r.body);
#if defined(_WIN32)
    curl_easy_setopt(c, CURLOPT_SSL_OPTIONS, static_cast<long>(CURLSSLOPT_REVOKE_BEST_EFFORT));
#endif
    const CURLcode rc = curl_easy_perform(c);
    if (rc != CURLE_OK) r.error = curl_easy_strerror(rc);
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &r.status);
    curl_slist_free_all(h);
    curl_easy_cleanup(c);
#else
    (void)url;
    (void)binary;
    (void)timeout_ms;
    (void)token;
    r.error = "このビルドには HTTP クライアントがありません";
#endif
    return r;
}

Fetch fetch(const std::string& url, bool binary, int timeout_ms) { return fetch(url, binary, timeout_ms, read_token()); }

void check_thread(const std::string& source, const std::string& token) {
    const Fetch f = fetch(source, false, 20000, token);
    std::string tag, page;
    if (!f.ok() || !parse_release(f.body, tag, page)) {
        std::string why = !f.error.empty() ? f.error : f.ok() ? "応答にリリース情報がありません" : "HTTP " + std::to_string(f.status);
        if (f.status == 404) {
            why = token.empty() ? "リリースを参照できません（非公開の場合は update.token_file が必要）" : "リリースが見つかりません";
        }
        host_log("update: check failed: %s", why.c_str());
        set_lines(std::string("bbhost ") + BBHOST_VERSION + " - 更新を確認できませんでした", why);
        g_busy.store(false);
        return;
    }
    if (!is_newer(tag, BBHOST_VERSION)) {
        host_log("update: %s is current (latest release %s)", BBHOST_VERSION, tag.c_str());
        set_lines(std::string("bbhost ") + BBHOST_VERSION + " - 最新です", "最新リリース: " + tag);
        g_available.store(false);
        g_busy.store(false);
        return;
    }
    host_log("update: %s is available (this is %s): %s", tag.c_str(), BBHOST_VERSION, page.c_str());
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_tag = tag;
        g_page = page;
        g_status = "bbhost " + tag + " が利用できます（現在: " + BBHOST_VERSION + ")";
        g_detail = page;
    }
    g_available.store(true);
    g_busy.store(false);
}

// A check on a thread of its own, unless one is running. `delay` keeps it out
// of the boot's busiest seconds. What it asks with is read here: main loads
// the configuration again after the setup window's Play, while a check the
// window began may still be running.
void begin_check(bool delay) {
    if (config().update_source.empty()) {
        set_lines("日本語版の更新先は未設定です", "配布元の日本語版リリースを確認してください。");
        return;
    }
    if (g_busy.exchange(true)) return;
    g_checked.store(true);
    set_lines(std::string("bbhost ") + BBHOST_VERSION + " - 更新を確認中...", "");
    std::string source = config().update_source.empty() ? kDefaultSource : config().update_source;
    std::thread([delay, source = std::move(source), token = read_token()] {
        if (delay) std::this_thread::sleep_for(std::chrono::seconds(5));
        check_thread(source, token);
    }).detach();
}

bool write_file(const fs::path& p, const std::string& data) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(out);
}

}  // namespace

// --- official plugins ----------------------------------------------------

std::mutex g_plugins_mu;
std::string g_plugins_status;  // under g_plugins_mu
std::atomic<bool> g_plugins_busy{false};

void set_plugins_status(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_plugins_mu);
    g_plugins_status = s;
    host_log("plugins: %s", s.c_str());
}

#if defined(_WIN32)
constexpr const char* kPluginSuffix = "-windows.dll";
constexpr const char* kPluginExt = ".dll";
#else
constexpr const char* kPluginSuffix = "-linux.so";
constexpr const char* kPluginExt = ".so";
#endif

// Every bbhost-plugin-<name><suffix> of the latest release, with its .sig:
// verified against the release key, then written to <dir>/<name><ext>
// (+ .sig) - or <name><ext>.new when that file is in use (a loaded plugin on
// Windows), which plugins_load puts in place at the next start.
void fetch_plugins_thread(std::string dir) {
    if (config().update_source.empty()) {
        set_plugins_status("日本語版の更新先は未設定です。同梱のプラグインを使用してください。");
        g_plugins_busy.store(false);
        return;
    }
    set_plugins_status("公式プラグインの最新版を検索中...");
    const std::string source = config().update_source.empty() ? kDefaultSource : config().update_source;
    const Fetch f = fetch(source, false, 20000);
    json::Value v;
    std::string err;
    if (!f.ok() || !json::parse(f.body, v, err) || v.type != json::Value::Type::Object) {
        set_plugins_status("最新リリースを読み込めません: " + (!f.error.empty() ? f.error : "HTTP " + std::to_string(f.status)) +
                           (f.status == 404 && read_token().empty() ? " （非公開の場合は update.token_file が必要）" : ""));
        g_plugins_busy.store(false);
        return;
    }
    std::string tag;
    if (const json::Value* t = v.find("tag_name"); t && t->type == json::Value::Type::String) tag = t->string;
    std::map<std::string, std::string> files;  // asset name -> API url
    if (const json::Value* a = v.find("assets"); a && a->type == json::Value::Type::Array) {
        for (const json::Value& asset : a->array) {
            const json::Value* n = asset.find("name");
            const json::Value* u = asset.find("url");
            if (n && u && n->type == json::Value::Type::String && u->type == json::Value::Type::String) files[n->string] = u->string;
        }
    }
    std::error_code ec;
    fs::create_directories(dir, ec);
    int installed = 0, failed = 0;
    std::string names;
    for (const auto& [asset, url] : files) {
        const std::string prefix = "bbhost-plugin-", suffix = kPluginSuffix;
        if (asset.rfind(prefix, 0) != 0 || asset.size() <= prefix.size() + suffix.size() ||
            asset.compare(asset.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        const std::string name = asset.substr(prefix.size(), asset.size() - prefix.size() - suffix.size());
        auto sig_it = files.find(asset + ".sig");
        if (name.find_first_of("/\\.:") != std::string::npos || sig_it == files.end()) {
            ++failed;
            continue;
        }
        set_plugins_status("downloading " + name + "...");
        const Fetch bin = fetch(url, true, 300000);
        const Fetch sig = fetch(sig_it->second, true, 30000);
        if (!bin.ok() || !sig.ok() || !signature_ok(bin.body, sig.body, kReleaseKey)) {
            host_log("plugins: %s: %s", name.c_str(), !bin.ok() || !sig.ok() ? "download failed" : "the signature does not match: not installed");
            ++failed;
            continue;
        }
        // Written beside it and renamed over it: a running plugin keeps the
        // file it was loaded from (Linux), or the rename is refused because
        // it is loaded (Windows) and the next start puts the .new in place.
        const fs::path target = fs::path(dir) / (name + kPluginExt);
        const fs::path tmp = fs::path(target.string() + ".part");
        fs::path out = target;
        std::error_code rec;
        if (!write_file(tmp, bin.body)) {
            ++failed;
            continue;
        }
        fs::rename(tmp, target, rec);
        if (rec) {
            out = fs::path(target.string() + ".new");
            fs::rename(tmp, out, rec);
            if (rec) {
                fs::remove(tmp, rec);
                ++failed;
                continue;
            }
        }
        write_file(fs::path(out.string() + ".sig"), sig.body);
        ++installed;
        names += (names.empty() ? "" : ", ") + name;
    }
    if (!installed && !failed) {
        set_plugins_status("release " + tag + " にこの OS 用の公式プラグインがありません");
    } else {
        set_plugins_status("release " + tag + ": " + std::to_string(installed) + " installed" + (names.empty() ? "" : " (" + names + ")") +
                           (failed ? ", " + std::to_string(failed) + " 失敗（ログを確認してください）" : "") + " - 次回起動時から有効");
    }
    g_plugins_busy.store(false);
}

const char* current_version() { return BBHOST_VERSION; }

void fetch_plugins(const std::string& dir) {
    if (g_plugins_busy.exchange(true)) return;
    std::thread(fetch_plugins_thread, dir).detach();
}

std::string plugins_status() {
    std::lock_guard<std::mutex> lk(g_plugins_mu);
    return g_plugins_status;
}

bool plugins_busy() { return g_plugins_busy.load(); }

void start() {
    {
        // The copy an older build's update set aside (bbhost.exe.old,
        // bbhost.old on Linux): gone once nothing runs it.
        std::error_code ec;
        const fs::path cur = exe_path();
        if (!cur.empty()) fs::remove(fs::path(cur).concat(".old"), ec);
    }
#if defined(BBHOST_RELEASE_BUILD)
    const bool release = true;
#else
    const bool release = false;
#endif
    // BBHOST_UPDATE_TEST=1: check whatever the build and settings say, and
    // log what is found.
    const bool test = std::getenv("BBHOST_UPDATE_TEST") != nullptr;
    if (config().headless && !test) return;  // nobody to offer it to
    if (g_checked.load()) return;            // the setup window has checked
    const int want = test ? 1 : config().update_check;
    set_lines(std::string("bbhost ") + BBHOST_VERSION, "");
    if (want == 0 || (want < 0 && !release)) {
        if (want < 0) set_lines(std::string("bbhost ") + BBHOST_VERSION, "ソースからのビルド: 指定時のみ更新確認 (update.check)");
        return;
    }
    begin_check(true);
}

void check_once() {
    if (!g_checked.load()) begin_check(false);
}

std::string status() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_status;
}

std::string detail() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_detail;
}

bool busy() { return g_busy.load(); }
bool update_available() { return g_available.load(); }

std::string latest_tag() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_tag;
}

void check_now() { begin_check(false); }

bool open_release_page() {
    std::string url;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        url = g_page;
    }
    if (!g_available.load() || !release_page_ok(url)) return false;
#if defined(BBHOST_HAVE_SDL3)
    // The system's handler for links, the default browser: ShellExecute on
    // Windows, xdg-open on Linux (not waited for).
    if (SDL_OpenURL(url.c_str())) {
        host_log("update: opened %s", url.c_str());
        return true;
    }
    host_log("update: could not open %s: %s", url.c_str(), SDL_GetError());
#else
    // A build without SDL has no window, so nothing calls this.
    host_log("update: no browser to open %s in", url.c_str());
#endif
    std::lock_guard<std::mutex> lk(g_mu);
    g_detail = "ブラウザーを開けませんでした。ページ: " + url;
    return false;
}

}  // namespace updater
