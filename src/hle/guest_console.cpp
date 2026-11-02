// The game's console prints, a repeating line written once and then counted
// (guest_console.h).
#include "hle/guest_console.h"
#include "log.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace bb::guest_console {

std::string Filter::feed(std::uint64_t thread, std::string_view text, std::uint64_t now_ms) {
    std::string out;
    if (!o_.collapse) {
        out.assign(text);
        return out;
    }
    std::string& part = partial_[thread];
    std::size_t at = 0;
    while (at < text.size()) {
        const std::size_t nl = text.find('\n', at);
        if (nl == std::string_view::npos) {
            part.append(text.substr(at));
            break;
        }
        part.append(text.substr(at, nl + 1 - at));
        line(part, now_ms, out);
        part.clear();
        at = nl + 1;
    }
    if (part.size() >= o_.max_partial) {  // no newline coming: a line as it is
        line(part, now_ms, out);
        part.clear();
    }
    if (part.empty()) partial_.erase(thread);
    counts(now_ms, false, out);
    return out;
}

std::string Filter::flush_thread(std::uint64_t thread, std::uint64_t now_ms) {
    std::string out;
    const auto it = partial_.find(thread);
    if (it == partial_.end()) return out;
    const std::string part = std::move(it->second);
    partial_.erase(it);
    if (!part.empty()) line(part, now_ms, out);
    return out;
}

std::string Filter::flush_all(std::uint64_t now_ms) {
    std::string out;
    for (auto& [thread, part] : partial_) {
        if (!part.empty()) line(part + "\n", now_ms, out);
    }
    partial_.clear();
    counts(now_ms, true, out);
    return out;
}

void Filter::line(std::string_view l, std::uint64_t now_ms, std::string& out) {
    auto it = seen_.find(std::string(l));
    if (it != seen_.end() && now_ms - it->second.last_ms < o_.window_ms) {
        // Again within the window of its last: counted.
        ++it->second.count;
        it->second.last_ms = now_ms;
        ++suppressed_total_;
        return;
    }
    if (it != seen_.end() && it->second.count) count_line(it->first, it->second, now_ms, out);  // the run before it, first
    out.append(l);
    Seen& s = it != seen_.end() ? it->second : seen_[std::string(l)];
    s.last_ms = now_ms;
    s.since_ms = now_ms;
    s.count = 0;
}

void Filter::counts(std::uint64_t now_ms, bool all, std::string& out) {
    // At most once a second, unless the lines remembered are too many.
    if (!all && now_ms - last_sweep_ms_ < 1000 && seen_.size() <= o_.max_lines) return;
    last_sweep_ms_ = now_ms;
    for (auto it = seen_.begin(); it != seen_.end();) {
        Seen& s = it->second;
        if (s.count && (all || now_ms - s.since_ms >= o_.window_ms)) {
            count_line(it->first, s, now_ms, out);
            s.count = 0;
            s.since_ms = now_ms;
        }
        // Quiet for a window, or too many lines: forgotten (its next time is written).
        if (!s.count && (all || now_ms - s.last_ms >= o_.window_ms || seen_.size() > o_.max_lines)) {
            it = seen_.erase(it);
        } else {
            ++it;
        }
    }
}

void Filter::count_line(std::string_view l, const Seen& s, std::uint64_t, std::string& out) {
    std::string_view text = l;
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.remove_suffix(1);
    char head[128];
    std::snprintf(head, sizeof(head), "[bbhost] the game printed this %llu more time%s (over %.1f s): ",
                  static_cast<unsigned long long>(s.count), s.count == 1 ? "" : "s", (s.last_ms - s.since_ms) / 1000.0);
    out += head;
    out.append(text);
    out += '\n';
}

namespace {

struct Stream {
    std::mutex mu;
    Filter filter;
};
Stream& out_stream() {
    static Stream* s = new Stream;  // never destroyed: the exit flush may run after static destructors
    return *s;
}
Stream& err_stream() {
    static Stream* s = new Stream;
    return *s;
}

bool collapse_on() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_GUEST_PRINTS");
        return !(e && std::strcmp(e, "all") == 0);
    }();
    return on;
}

std::uint64_t now_ms() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::uint64_t thread_key() { return static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())); }

Stream* stream_of(std::FILE* f) {
    if (f == stdout) return &out_stream();
    if (f == stderr) return &err_stream();
    return nullptr;
}

}  // namespace

bool write(std::FILE* f, const char* data, std::size_t n) {
    Stream* s = stream_of(f);
    if (!s) return false;
    if (!host_logging()) return true;  // Tooling and logging off: the game's prints are not written either
    if (!collapse_on()) {
        std::fwrite(data, 1, n, f);
        return true;
    }
    static std::once_flag once;
    std::call_once(once, [] { std::atexit(flush_all); });
    std::string out;
    {
        std::lock_guard<std::mutex> lk(s->mu);
        out = s->filter.feed(thread_key(), std::string_view(data, n), now_ms());
    }
    if (!out.empty()) std::fwrite(out.data(), 1, out.size(), f);
    return true;
}

void flush(std::FILE* f) {
    Stream* s = stream_of(f);
    if (!s || !collapse_on() || !host_logging()) return;
    std::string out;
    {
        std::lock_guard<std::mutex> lk(s->mu);
        out = s->filter.flush_thread(thread_key(), now_ms());
    }
    if (!out.empty()) std::fwrite(out.data(), 1, out.size(), f);
}

void flush_all() {
    if (!collapse_on() || !host_logging()) return;
    for (std::FILE* f : {stdout, stderr}) {
        Stream* s = stream_of(f);
        std::string out;
        {
            std::lock_guard<std::mutex> lk(s->mu);
            out = s->filter.flush_all(now_ms());
        }
        if (!out.empty()) std::fwrite(out.data(), 1, out.size(), f);
        std::fflush(f);
    }
}

}  // namespace bb::guest_console
