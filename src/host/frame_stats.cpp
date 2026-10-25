#include "host/frame_stats.h"

#include "hle/modules.h"
#include "host/gpu.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <dirent.h>
#include <unistd.h>
#else
#include "core/portable.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct ThreadTime {
    std::uint64_t ticks = 0;
    std::string name;
};

// utime + stime of every thread of this process, in clock ticks, by tid.
std::unordered_map<int, ThreadTime> thread_times() {
    std::unordered_map<int, ThreadTime> out;
#if !defined(_WIN32)
    DIR* d = opendir("/proc/self/task");
    if (!d) return out;
    while (dirent* e = readdir(d)) {
        const int tid = std::atoi(e->d_name);
        if (tid <= 0) continue;
        char path[64];
        std::snprintf(path, sizeof(path), "/proc/self/task/%d/stat", tid);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        char buf[1024];
        const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        buf[n] = 0;
        // pid (comm) state ... : comm may hold spaces, so parse after the last ')'.
        const char* open = std::strchr(buf, '(');
        const char* close = std::strrchr(buf, ')');
        if (!open || !close || close < open) continue;
        ThreadTime t;
        t.name.assign(open + 1, close);
        unsigned long long utime = 0, stime = 0;
        // Fields after ')': state(3) ppid pgrp session tty tpgid flags minflt
        // cminflt majflt cmajflt utime(14) stime(15).
        if (std::sscanf(close + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &utime, &stime) != 2) continue;
        t.ticks = utime + stime;
        out.emplace(tid, std::move(t));
    }
    closedir(d);
#else
    // Kernel + user time of the threads host_thread_set_name named, in
    // 100 ns units (core/portable.h): the same list /proc gives on Linux.
    for (const HostThreadTime& h : host_thread_times()) {
        ThreadTime t;
        t.ticks = h.ticks;
        t.name = h.name;
        out.emplace(static_cast<int>(h.tid), std::move(t));
    }
#endif
    return out;
}

struct State {
    std::mutex mu;
    Clock::time_point window_start{}, last_flip{};
    std::uint64_t displayed_at_start = 0;  // completed flips when the window opened
    std::vector<double> intervals_ms;
    std::unordered_map<int, ThreadTime> threads;
    GpuBusy gpu{};  // host_gpu_busy() when the window opened
    GpuStats stats{};  // host_gpu_stats() when it opened (BBHOST_FRAME_DETAIL=1)
    std::uint64_t workers[4] = {};  // host_render_worker_jobs() when it opened (BBHOST_FRAME_DETAIL=1)
    std::uint64_t import_waits = 0, import_wait_us = 0;  // host_gpu_import_waits() when it opened
};

// BBHOST_FRAME_DETAIL=1: the second's draws by how their pixel shader ran,
// its draw failures by reason, and every flip interval in it, in lines short
// enough for the log's line.
void log_detail(const GpuStats& was, const GpuStats& now, const std::vector<double>& intervals_ms) {
    std::uint64_t ps[kDrawPsCount];
    for (int i = 0; i < kDrawPsCount; ++i) ps[i] = now.draws_ps[i] - was.draws_ps[i];
    std::string why;
    for (int k = 0; k < 16; ++k) {
        const std::uint64_t n = now.draw_fail_why[k] - was.draw_fail_why[k];
        const char* name = host_gpu_draw_fail_name(k);
        if (n && name) why += (why.empty() ? " (" : ", ") + std::string(name) + " " + std::to_string(n);
    }
    if (!why.empty()) why += ")";
    host_log("frame detail: draws %llu: pixel shader lifted %llu, translated %llu, fallback %llu, none %llu; draw failures %llu%s",
             static_cast<unsigned long long>(now.draws - was.draws), static_cast<unsigned long long>(ps[kDrawPsLifted]),
             static_cast<unsigned long long>(ps[kDrawPsTranslated]), static_cast<unsigned long long>(ps[kDrawPsFallback]),
             static_cast<unsigned long long>(ps[kDrawPsNone]), static_cast<unsigned long long>(now.draw_failures - was.draw_failures),
             why.c_str());
    std::uint64_t vs[kDrawVsCount];
    for (int i = 0; i < kDrawVsCount; ++i) vs[i] = now.draws_vs[i] - was.draws_vs[i];
    host_log("frame detail: vertex shader lifted %llu, translated %llu, fetch shader or tessellation %llu, fallback %llu; pixel shader "
             "discarding by clip distances %llu",
             static_cast<unsigned long long>(vs[kDrawVsLifted]), static_cast<unsigned long long>(vs[kDrawVsTranslated]),
             static_cast<unsigned long long>(vs[kDrawVsOther]), static_cast<unsigned long long>(vs[kDrawVsFallback]),
             static_cast<unsigned long long>(now.draws_clip_discard - was.draws_clip_discard));
    if (now.oob_on) {
        unsigned long long d[11];
        for (int k = 0; k < 11; ++k) d[k] = static_cast<unsigned long long>(now.oob[k] - was.oob[k]);
        host_log("frame detail: out-of-range buffer accesses: typed loads by index past num_records %llu, past their binding %llu; walks by "
                 "index past num_records %llu",
                 d[0], d[1], d[2]);
        host_log("frame detail: out-of-range image accesses: loads outside the view's size %llu, past its levels %llu, past its layers "
                 "%llu; stores outside its size %llu, past its layers %llu, at a level other than 0 %llu; the view smaller than the T# %llu, "
                 "larger %llu",
                 d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]);
    }
    std::uint64_t ts[kDrawTessCount];
    for (int i = 0; i < kDrawTessCount; ++i) ts[i] = now.draws_tess[i] - was.draws_tess[i];
    if (ts[kDrawTess]) {
        host_log("frame detail: tessellated draws %llu: domain shader lifted %llu, LS lifted %llu, hull lifted %llu, all three %llu "
                 "(native hull pipelines %llu)",
                 static_cast<unsigned long long>(ts[kDrawTess]), static_cast<unsigned long long>(ts[kDrawTessDomainLifted]),
                 static_cast<unsigned long long>(ts[kDrawTessLsLifted]), static_cast<unsigned long long>(ts[kDrawTessHullLifted]),
                 static_cast<unsigned long long>(ts[kDrawTessAllLifted]), static_cast<unsigned long long>(ts[kDrawTessNative]));
    }
    if (const std::uint64_t n = now.shader_dispatches - was.shader_dispatches) {
        const std::uint64_t lifted = now.lifted_dispatches - was.lifted_dispatches;
        host_log("frame detail: dispatches %llu: lifted %llu, translated %llu", static_cast<unsigned long long>(n),
                 static_cast<unsigned long long>(lifted), static_cast<unsigned long long>(n - lifted));
    }
    constexpr std::size_t kPerLine = 100;  // 100 x " 16.67": well inside host_log's 1 KiB
    for (std::size_t at = 0; at < intervals_ms.size(); at += kPerLine) {
        std::string line;
        for (std::size_t i = at; i < intervals_ms.size() && i < at + kPerLine; ++i) {
            char v[24];
            std::snprintf(v, sizeof(v), " %.2f", intervals_ms[i]);
            line += v;
        }
        host_log("frame intervals ms:%s", line.c_str());
    }
}

}  // namespace

void frame_stats_on_flip() {
    const bool on = frame_stats_enabled();
    if (!on) return;
    static State s;
    static const long hz = [] {
#if !defined(_WIN32)
        return sysconf(_SC_CLK_TCK);
#else
        return 10000000L;  // GetThreadTimes counts 100 ns units
#endif
    }();
    std::lock_guard<std::mutex> lk(s.mu);
    const auto now = Clock::now();
    if (s.last_flip.time_since_epoch().count() == 0) {
        s.window_start = s.last_flip = now;
        s.threads = thread_times();
        s.gpu = host_gpu_busy();
        if (frame_stats_detail()) {
            s.stats = host_gpu_stats();
            host_render_worker_jobs(s.workers);
        }
        host_gpu_import_waits(&s.import_waits, &s.import_wait_us);
        return;
    }
    s.intervals_ms.push_back(std::chrono::duration<double, std::milli>(now - s.last_flip).count());
    s.last_flip = now;
    const double secs = std::chrono::duration<double>(now - s.window_start).count();
    if (secs < 1.0) return;

    double sum = 0, worst = 0;
    int over16 = 0, over33 = 0;
    for (const double v : s.intervals_ms) {
        sum += v;
        worst = std::max(worst, v);
        over16 += v > 17.5;
        over33 += v > 34.5;
    }
    const std::size_t n = s.intervals_ms.size();
    std::vector<double> sorted = s.intervals_ms;
    std::sort(sorted.begin(), sorted.end());
    const double p95 = sorted[std::min(n - 1, static_cast<std::size_t>(0.95 * static_cast<double>(n)))];

    auto cur = thread_times();
    struct Busy {
        double pct;
        std::string name;
    };
    std::vector<Busy> busy;
    double total = 0;
    for (const auto& [tid, t] : cur) {
        const auto it = s.threads.find(tid);
        const std::uint64_t was = it != s.threads.end() ? it->second.ticks : 0;
        const double pct = 100.0 * static_cast<double>(t.ticks - was) / static_cast<double>(hz) / secs;
        total += pct;
        if (pct >= 5.0) busy.push_back({pct, t.name + ":" + std::to_string(tid)});
    }
    std::sort(busy.begin(), busy.end(), [](const Busy& a, const Busy& b) { return a.pct > b.pct; });
    std::string top;
    for (std::size_t i = 0; i < busy.size() && i < 8; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " %s %.0f%%", busy[i].name.c_str(), busy[i].pct);
        top += buf;
    }
    const std::uint64_t sync_ns = main_thread_waited_ns(0), sleep_ns = main_thread_waited_ns(1), file_ns = main_thread_waited_ns(2);
    std::string work;
    if (std::vector<std::uint32_t> busy = main_frames_busy_us(); !busy.empty()) {
        std::sort(busy.begin(), busy.end());
        double busy_sum = 0;
        int busy_over = 0;
        for (std::uint32_t us : busy) {
            busy_sum += us;
            busy_over += us > 16667;
        }
        char wb[160];
        std::snprintf(wb, sizeof(wb), "; main loop work avg %.1f p95 %.1f max %.1f ms, %d over 16.7", busy_sum / busy.size() / 1e3,
                      busy[busy.size() * 95 / 100] / 1e3, busy.back() / 1e3, busy_over);
        work = wb;
    }
    // The flips above are the game's submits. What the display actually showed
    // is the completion counter, which the flip clock paces: the
    // two differ when the game submits faster than the display can show.
    const std::uint64_t displayed_now = hle_video_flip_count();
    const double displayed = static_cast<double>(displayed_now - s.displayed_at_start) / secs;
    s.displayed_at_start = displayed_now;
    // How much of the second our work kept the GPU busy (host/gpu_busy.cpp:
    // the union of our command buffers' spans on the GPU's clock), last on
    // the line so the summaries' patterns for what is before it still hold.
    // The spans are counted a frame or two after they ran.
    std::string gpu;
    const GpuBusy gpu_now = host_gpu_busy();
    if (gpu_now.on && s.gpu.on) {
        const double ms = static_cast<double>(gpu_now.busy_ns - s.gpu.busy_ns) / 1e6 / secs;
        // The buffer shadow's uploads part of it, after it so the patterns
        // reading `gpu busy N ms/s` still hold.
        const double up = static_cast<double>(gpu_now.upload_ns - s.gpu.upload_ns) / 1e6 / secs;
        char gb[128];
        std::snprintf(gb, sizeof(gb), "; gpu busy %.0f ms/s (%.0f%%); gpu uploads %.0f ms/s", ms, ms / 10.0, up);
        gpu = gb;
    }
    s.gpu = gpu_now;
    // Guest memory imports the command processor waited for in the window
    // (window imports, BBHOST_IMPORT_WINDOWS=1: made at a binding rather than
    // by the import thread), last again, and only when there were any.
    std::string imports;
    {
        std::uint64_t in = 0, ius = 0;
        host_gpu_import_waits(&in, &ius);
        if (in > s.import_waits) {
            char ib[96];
            std::snprintf(ib, sizeof(ib), "; imports waited for %llu in %.1f ms", static_cast<unsigned long long>(in - s.import_waits),
                          static_cast<double>(ius - s.import_wait_us) / 1e3);
            imports = ib;
        }
        s.import_waits = in;
        s.import_wait_us = ius;
    }
    host_log("frames: %.2f s: %zu flips (%.1f/s), displayed %.1f/s, interval avg %.1f p95 %.1f max %.1f ms, %d over 16.7, %d over 33.3; "
             "cpu %.0f%%:%s; main loop waited %.0f ms, slept %.0f ms, file I/O %.0f ms%s%s%s",
             secs, n, static_cast<double>(n) / secs, displayed, sum / static_cast<double>(n), p95, worst, over16, over33, total,
             top.c_str(), static_cast<double>(sync_ns) / 1e6, static_cast<double>(sleep_ns) / 1e6,
             static_cast<double>(file_ns) / 1e6, work.c_str(), gpu.c_str(), imports.c_str());
    if (frame_stats_detail()) {
        const GpuStats stats = host_gpu_stats();
        log_detail(s.stats, stats, s.intervals_ms);
        s.stats = stats;
        // The command processor's workers: their jobs in the second, and each
        // one's CPU (named bb-tmask0-3 and bb-lean0-1).
        std::uint64_t w[4];
        host_render_worker_jobs(w);
        std::string cpu;
        for (const auto& [tid, t] : cur) {
            if (t.name.rfind("bb-tmask", 0) != 0 && t.name.rfind("bb-lean", 0) != 0) continue;
            const auto it = s.threads.find(tid);
            const std::uint64_t was = it != s.threads.end() ? it->second.ticks : 0;
            char b[48];
            std::snprintf(b, sizeof(b), " %s %.1f ms/s", t.name.c_str(),
                          1000.0 * static_cast<double>(t.ticks - was) / static_cast<double>(hz) / secs);
            cpu += b;
        }
        host_log("frame detail: command-processor workers: table read masks done %llu (queued %llu), no-fallback builds done %llu "
                 "(queued %llu); cpu:%s",
                 static_cast<unsigned long long>(w[0] - s.workers[0]), static_cast<unsigned long long>(w[1] - s.workers[1]),
                 static_cast<unsigned long long>(w[2] - s.workers[2]), static_cast<unsigned long long>(w[3] - s.workers[3]),
                 cpu.empty() ? " none named" : cpu.c_str());
        std::memcpy(s.workers, w, sizeof(w));
    }
    s.intervals_ms.clear();
    s.window_start = now;
    s.threads = std::move(cur);
}
