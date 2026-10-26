// The bottleneck meter's collector (host/perf_meter.h): the counters of each
// second, the verdict (host/bottleneck.h) and the lines.
#include "host/perf_meter.h"

#include "core/host_clock.h"
#include "core/portable.h"
#include "engine/loading.h"
#include "engine/world_chr.h"
#include "hle/modules.h"
#include "host/bottleneck.h"
#include "host/frame_stats.h"
#include "host/gpu.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

using bottleneck::Stage;
using bottleneck::kStages;

std::atomic<bool> g_on{false};

// The GX job barrier's two spin loops, in Binary Ninja's addresses: whether
// every worker is in (or new work flagged), and each worker's state (2:
// done). The loops have no `pause`: the main thread is on a core all the
// while, so a CPU counter cannot tell them from its own work.
constexpr std::uint64_t kSpinBn[2][2] = {{0x25b8890, 0x25b88a2}, {0x25b8960, 0x25b8967}};
std::uint64_t g_spin[2][2] = {};
bool g_spin_known = false;

bool in_spin(std::uint64_t pc) {
    return (pc >= g_spin[0][0] && pc < g_spin[0][1]) || (pc >= g_spin[1][0] && pc < g_spin[1][1]);
}

// The flips since the meter last took them, as monotonic nanoseconds.
std::mutex g_flip_mu;
std::vector<std::uint64_t> g_flip_ns;  // under g_flip_mu
std::atomic<std::uint64_t> g_flips{0};

std::atomic<std::uint64_t> g_limiter_ns{0}, g_limiter_spin_ns{0}, g_limiter_frames{0};
std::atomic<float> g_cap_fps{0.0f};
std::atomic<std::uint64_t> g_backlog_ns{0};

#if defined(_WIN32)
std::atomic<HANDLE> g_main{nullptr};
#else
std::atomic<int> g_main_tid{0};
#endif

// The main thread's place: samples, and those in the barrier's spin.
struct MainSamples {
    bool on = false;
    std::uint64_t samples = 0, spin = 0;
    std::uint64_t cpu_ns = 0;  // Linux: the task clock's count (its CPU time); Windows: unused
    std::uint64_t lost = 0;
    std::uint64_t stopped_ns = 0;  // Windows: the time the main thread stood suspended for samples
};

#if !defined(_WIN32)
// A perf_events task clock on the main thread: a sample of its user-mode
// instruction pointer every 2 ms of its CPU time, into a ring the meter reads
// once a second. Nothing is delivered to the thread (signals to the main loop
// hung it: host/sampler.cpp). kernel.perf_event_paranoid 2 (the default on
// most distributions) allows it for one's own threads; 3 and up do not.
constexpr std::uint64_t kPeriodNs = 2000000;
struct PerfMain {
    int fd = -1;
    void* map = nullptr;
    std::size_t data_size = 0;

    bool open(int tid, int* err) {
        perf_event_attr a{};
        a.size = sizeof(a);
        a.type = PERF_TYPE_SOFTWARE;
        a.config = PERF_COUNT_SW_TASK_CLOCK;
        a.sample_period = kPeriodNs;
        a.sample_type = PERF_SAMPLE_IP;
        a.exclude_kernel = 1;
        a.exclude_hv = 1;
        const long f = syscall(SYS_perf_event_open, &a, tid, -1, -1, PERF_FLAG_FD_CLOEXEC);
        if (f < 0) {
            *err = errno;
            return false;
        }
        const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        data_size = page * 16;  // 64 KiB: 4,096 samples of 16 bytes, eight seconds of a busy thread
        map = mmap(nullptr, page + data_size, PROT_READ | PROT_WRITE, MAP_SHARED, static_cast<int>(f), 0);
        if (map == MAP_FAILED) {
            *err = errno;
            close(static_cast<int>(f));
            map = nullptr;
            return false;
        }
        fd = static_cast<int>(f);
        return true;
    }

    void drain(MainSamples& m) {
        auto* meta = static_cast<perf_event_mmap_page*>(map);
        const auto* data = static_cast<const std::uint8_t*>(map) + static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        const std::uint64_t head = __atomic_load_n(&meta->data_head, __ATOMIC_ACQUIRE);
        std::uint64_t tail = meta->data_tail;
        while (tail + sizeof(perf_event_header) <= head) {
            perf_event_header h;
            for (std::size_t k = 0; k < sizeof(h); ++k) reinterpret_cast<std::uint8_t*>(&h)[k] = data[(tail + k) % data_size];
            if (!h.size) break;
            if (h.type == PERF_RECORD_SAMPLE && h.size >= sizeof(h) + 8) {
                std::uint64_t ip = 0;
                for (std::size_t k = 0; k < 8; ++k) reinterpret_cast<std::uint8_t*>(&ip)[k] = data[(tail + sizeof(h) + k) % data_size];
                ++m.samples;
                if (in_spin(ip)) ++m.spin;
            } else if (h.type == PERF_RECORD_LOST && h.size >= sizeof(h) + 16) {
                std::uint64_t lost = 0;
                for (std::size_t k = 0; k < 8; ++k) reinterpret_cast<std::uint8_t*>(&lost)[k] = data[(tail + sizeof(h) + 8 + k) % data_size];
                m.lost += lost;
            }
            tail += h.size;
        }
        __atomic_store_n(&meta->data_tail, tail, __ATOMIC_RELEASE);
        std::uint64_t count = 0;
        if (read(fd, &count, sizeof(count)) == static_cast<ssize_t>(sizeof(count))) m.cpu_ns = count;
    }
};
#endif

// Threads by what they are to the meter.
enum Role { kRoleMain, kRoleRender, kRoleCp, kRoleRecord, kRolePresent, kRoleGx, kRoleCpWorker, kRoleMeter, kRoles, kRoleOther };

Role role_of(const char* name, bool is_main) {
    if (is_main) return kRoleMain;
    const auto starts = [name](const char* p) { return std::strncmp(name, p, std::strlen(p)) == 0; };
    if (starts("GXRenderThread")) return kRoleRender;
    if (std::strcmp(name, "bb-cp0") == 0) return kRoleCp;
    if (std::strcmp(name, "bb-record") == 0) return kRoleRecord;
    if (std::strcmp(name, "bb-present") == 0) return kRolePresent;
    if (starts("GXWorker")) return kRoleGx;
    if (starts("bb-tmask") || starts("bb-lean")) return kRoleCpWorker;
    if (std::strcmp(name, "bb-meter") == 0) return kRoleMeter;
    return kRoleOther;
}

// The CPU each role had in a window, in ms: the main thread's, the busiest
// command processor's (several queues each have a bb-cp0), the others summed.
struct RoleCpu {
    double ms[kRoles];
    int count[kRoles];
};

class ThreadClock {
public:
    // The CPU time of the roles' threads since the last call; `scan` looks
    // for new ones first.
    RoleCpu take(bool scan) {
        RoleCpu r{};
        for (int k = 0; k < kRoles; ++k) r.ms[k] = -1;
#if !defined(_WIN32)
        if (scan || threads_.empty()) rescan();
        static const double tick_ms = 1000.0 / static_cast<double>(sysconf(_SC_CLK_TCK));
        for (Watched& t : threads_) {
            std::uint64_t ticks = 0;
            if (!read_ticks(t.tid, &ticks)) continue;
            const double ms = t.have ? static_cast<double>(ticks - t.ticks) * tick_ms : 0.0;
            t.ticks = ticks;
            const bool first = !t.have;
            t.have = true;
            if (first) continue;
            add(r, t.role, ms);
        }
#else
        (void)scan;
        const DWORD main_tid = main_tid_.load(std::memory_order_relaxed);
        for (const HostThreadTime& h : host_thread_times()) {
            const Role role = role_of(h.name, h.tid == main_tid || std::strcmp(h.name, "main") == 0);
            if (role == kRoleOther) continue;
            auto [it, fresh] = last_.emplace(h.tid, h.ticks);
            const double ms = fresh ? 0.0 : static_cast<double>(h.ticks - it->second) / 1e4;  // 100 ns units
            it->second = h.ticks;
            if (!fresh) add(r, role, ms);
        }
#endif
        return r;
    }
#if defined(_WIN32)
    void note_main(DWORD tid) { main_tid_.store(tid, std::memory_order_relaxed); }
#endif

private:
    static void add(RoleCpu& r, Role role, double ms) {
        if (r.ms[role] < 0) r.ms[role] = 0;
        r.ms[role] = role == kRoleCp ? std::max(r.ms[role], ms) : r.ms[role] + ms;
        ++r.count[role];
    }
#if !defined(_WIN32)
    struct Watched {
        int tid;
        Role role;
        std::uint64_t ticks = 0;
        bool have = false;
    };
    std::vector<Watched> threads_;

    static bool read_ticks(int tid, std::uint64_t* ticks) {
        char path[64], buf[1024];
        std::snprintf(path, sizeof(path), "/proc/self/task/%d/stat", tid);
        const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) return false;
        const ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
        ::close(fd);
        if (n <= 0) return false;
        buf[n] = 0;
        const char* close_paren = std::strrchr(buf, ')');
        if (!close_paren) return false;
        unsigned long long utime = 0, stime = 0;
        // After ')': state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime.
        if (std::sscanf(close_paren + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &utime, &stime) != 2) return false;
        *ticks = utime + stime;
        return true;
    }

    // The threads worth watching, by name (every ten seconds: guest threads
    // come and go with loads).
    void rescan() {
        std::vector<Watched> next;
        const int main_tid = g_main_tid.load(std::memory_order_relaxed);
        DIR* d = opendir("/proc/self/task");
        if (!d) return;
        while (dirent* e = readdir(d)) {
            const int tid = std::atoi(e->d_name);
            if (tid <= 0) continue;
            char path[64], name[32] = {};
            std::snprintf(path, sizeof(path), "/proc/self/task/%d/comm", tid);
            const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
            if (fd < 0) continue;
            const ssize_t n = ::read(fd, name, sizeof(name) - 1);
            ::close(fd);
            if (n <= 0) continue;
            name[n] = 0;
            if (char* nl = std::strchr(name, '\n')) *nl = 0;
            const Role role = role_of(name, tid == main_tid);
            if (role == kRoleOther) continue;
            Watched w{tid, role};
            for (const Watched& old : threads_) {
                if (old.tid == tid) w = old;  // keeps its last reading
            }
            w.role = role;
            next.push_back(w);
        }
        closedir(d);
        threads_ = std::move(next);
    }
#else
    std::unordered_map<std::uint32_t, std::uint64_t> last_;
    std::atomic<DWORD> main_tid_{0};
#endif
};

ThreadClock g_threads;

// Counters at one moment; a window is two of them.
struct Snap {
    std::uint64_t t_ns = 0;
    std::uint64_t flips = 0, displayed = 0;
    std::uint64_t limiter_ns = 0, limiter_spin_ns = 0, limiter_frames = 0, backlog_ns = 0;
    std::uint64_t wait_ns[3] = {};
    MainSamples main;
    bool gpu_on = false;
    std::uint64_t gpu_busy_ns = 0;
    bool rec_on = false;
    std::uint64_t rec_replay_ns = 0, rec_spin_ns = 0;
    std::uint64_t cp_fence_us = 0, cp_label_ns = 0, cp_pipeline_ns = 0;
    std::uint64_t import_n = 0, import_us = 0;
};

// A window's lines, and the sums its summary keeps.
struct Extra {
    double gx_workers_ms = -1, cp_workers_ms = -1, record_cpu_ms = -1, meter_ms = -1;
    int gx_workers = 0;
    double stopped_ms = 0;  // Windows: the main thread suspended for samples
};

struct Summary {
    double secs = 0;
    bottleneck::Window sum{};  // the fields summed (median_ms: unused)
    double seconds[kStages] = {};
    std::vector<double> intervals;
    Extra extra{};
    std::uint32_t block = 0;
    bool block_known = false;
};

std::mutex g_summary_mu;  // the summaries (the meter thread; perf_meter_exit)
Summary g_summary;
double g_run_seconds[kStages] = {};
double g_run_secs = 0;
double g_summary_period_s = 60;

std::string block_name(bool known, std::uint32_t b) {
    if (!known) return "";
    char buf[24];
    std::snprintf(buf, sizeof(buf), " at m%02u_%02u", (b >> 24) & 0xffu, (b >> 16) & 0xffu);
    return buf;
}

std::string pct(double load) {
    if (load < 0) return "-";
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.0f%%", load * 100.0);
    return buf;
}

std::string loads_text(const bottleneck::Verdict& v) {
    std::string s;
    static const Stage order[] = {Stage::kMain, Stage::kGx, Stage::kRender, Stage::kCp, Stage::kRecord, Stage::kGpu, Stage::kPresent};
    for (Stage st : order) s += std::string(s.empty() ? "" : " ") + bottleneck::stage_name(st) + " " + pct(v.load[static_cast<int>(st)]);
    return s;
}

// The main loop's frame, split: its own work, the barrier's spin, its waits
// on other threads and file I/O, the limiter, in ms a frame.
std::string ms_a_frame(double ms, double frames) {
    if (ms < 0) return "-";
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%.2f", ms / frames);
    return buf;
}

std::string frame_text(const bottleneck::Window& w) {
    if (w.flips <= 0) return "no frames";
    const double n = w.flips;
    const double own = w.main_cpu_ms < 0 ? -1 : std::max(0.0, w.main_cpu_ms - std::max(0.0, w.gx_spin_ms));
    return "own " + ms_a_frame(own, n) + " gx " + ms_a_frame(w.gx_spin_ms, n) + " waits " + ms_a_frame(w.main_wait_ms, n) + " file " +
           ms_a_frame(w.main_file_ms, n) + " limiter " + ms_a_frame(w.limiter_ms, n) + " ms";
}

std::string verdict_head(const bottleneck::Verdict& v) {
    std::string s = std::string("limit ") + bottleneck::stage_name(v.limit);
    if (v.limit_load >= 0) s += " " + pct(v.limit_load);
    if (v.next != Stage::kNone) s += std::string(", next ") + bottleneck::stage_name(v.next) + " " + pct(v.next_load);
    s += std::string(" (") + v.why + ")";
    return s;
}

void add_window(Summary& s, const bottleneck::Window& w, const bottleneck::Verdict& v, const std::vector<double>& iv, const Extra& x) {
    s.secs += w.secs;
    s.seconds[static_cast<int>(v.limit)] += w.secs;
    bottleneck::Window& a = s.sum;
    const auto acc = [](double& to, double add) {
        if (add < 0) return;
        to = to < 0 ? add : to + add;
    };
    a.secs += w.secs;
    a.flips += w.flips;
    acc(a.displayed, w.displayed);
    a.cap_fps = w.cap_fps;  // the latest
    a.long_ms += w.long_ms;
    a.loading_ms += w.loading_ms;
    acc(a.limiter_ms, w.limiter_ms);
    acc(a.main_cpu_ms, w.main_cpu_ms);
    acc(a.gx_spin_ms, w.gx_spin_ms);
    a.main_wait_ms += w.main_wait_ms;
    a.main_file_ms += w.main_file_ms;
    a.backlog_ms += w.backlog_ms;
    acc(a.render_cpu_ms, w.render_cpu_ms);
    acc(a.cp_cpu_ms, w.cp_cpu_ms);
    a.cp_gpu_wait_ms += w.cp_gpu_wait_ms;
    acc(a.rec_busy_ms, w.rec_busy_ms);
    acc(a.gpu_busy_ms, w.gpu_busy_ms);
    acc(a.present_cpu_ms, w.present_cpu_ms);
    a.import_ms += w.import_ms;
    a.pipeline_ms += w.pipeline_ms;
    if (s.intervals.size() < 60000) s.intervals.insert(s.intervals.end(), iv.begin(), iv.end());
    acc(s.extra.gx_workers_ms, x.gx_workers_ms);
    acc(s.extra.cp_workers_ms, x.cp_workers_ms);
    acc(s.extra.record_cpu_ms, x.record_cpu_ms);
    acc(s.extra.meter_ms, x.meter_ms);
    s.extra.gx_workers = std::max(s.extra.gx_workers, x.gx_workers);
    s.extra.stopped_ms += x.stopped_ms;
}

// The summary of the window so far, as one line, and a fresh one after it.
void flush_summary_locked(const char* tag) {
    Summary& s = g_summary;
    if (s.secs < 1.0) {
        s = Summary{};
        return;
    }
    bottleneck::Window w = s.sum;
    std::vector<double> iv = s.intervals;
    std::sort(iv.begin(), iv.end());
    if (!iv.empty()) w.median_ms = iv[iv.size() / 2];
    // The window's long frames were counted against each second's median.
    const bottleneck::Verdict v = bottleneck::classify(w);
    const double low = iv.empty() ? 0 : 1000.0 / iv[std::min(iv.size() - 1, iv.size() * 99 / 100)];
    std::string by;
    std::vector<std::pair<double, int>> order;
    for (int k = 0; k < kStages; ++k) {
        if (s.seconds[k] > 0) order.push_back({s.seconds[k], k});
    }
    std::sort(order.rbegin(), order.rend());
    for (const auto& [secs, k] : order) {
        char b[48];
        std::snprintf(b, sizeof(b), "%s%s %.0f s", by.empty() ? "" : ", ", bottleneck::stage_name(static_cast<Stage>(k)), secs);
        by += b;
    }
    char meter[64] = "";
    if (s.extra.meter_ms >= 0) std::snprintf(meter, sizeof(meter), "; the meter %.2f%% of a core", s.extra.meter_ms / s.secs / 10.0);
    char stopped[80] = "";
    if (s.extra.stopped_ms > 0) std::snprintf(stopped, sizeof(stopped), ", the main thread stopped %.2f ms/s for it", s.extra.stopped_ms / s.secs);
    host_log("%s: %.0f s%s: %.1f fps (1%% low %.1f); %s; seconds by limit: %s; loads %s; main loop a frame %.2f ms: %s; held back by the "
             "command processor %.1f%%; long frames %.0f ms%s%s",
             tag, s.secs, block_name(s.block_known, s.block).c_str(), v.fps, low, verdict_head(v).c_str(), by.c_str(), loads_text(v).c_str(),
             v.frame_ms, frame_text(w).c_str(), v.backlog * 100.0, w.long_ms, meter, stopped);
    s = Summary{};
}

void log_second(const bottleneck::Window& w, const bottleneck::Verdict& v, const std::string& at, const Extra& x) {
    char gx[96] = "";
    if (x.gx_workers_ms >= 0 && x.gx_workers > 0) {
        std::snprintf(gx, sizeof(gx), "; GX workers %d at %.0f%% in all", x.gx_workers, x.gx_workers_ms / w.secs / 10.0);
    }
    host_log("perf: %.2f s%s: %s; %.1f fps %.2f ms; loads %s; main loop a frame: %s; held back by the command processor %.1f%%; "
             "long frames %.0f ms%s",
             w.secs, at.c_str(), verdict_head(v).c_str(), v.fps, v.frame_ms, loads_text(v).c_str(), frame_text(w).c_str(), v.backlog * 100.0,
             w.long_ms, gx);
}

Snap take_snap(MainSamples& main) {
    Snap s;
    s.t_ns = host_clock_monotonic_ns();
    s.flips = g_flips.load(std::memory_order_relaxed);
    s.displayed = hle_video_flip_count();
    s.limiter_ns = g_limiter_ns.load(std::memory_order_relaxed);
    s.limiter_spin_ns = g_limiter_spin_ns.load(std::memory_order_relaxed);
    s.limiter_frames = g_limiter_frames.load(std::memory_order_relaxed);
    s.backlog_ns = g_backlog_ns.load(std::memory_order_relaxed);
    for (int k = 0; k < 3; ++k) s.wait_ns[k] = main_thread_waited_total_ns(k);
    s.main = main;
    const GpuBusy b = host_gpu_busy();
    s.gpu_on = b.on;
    s.gpu_busy_ns = b.busy_ns;
    s.rec_on = host_gpu_recorder_ns(&s.rec_replay_ns, &s.rec_spin_ns);
    const GpuStats st = host_gpu_stats();
    s.cp_fence_us = st.gpu_us;
    s.cp_label_ns = st.phase_ns[kPhaseWait];
    s.cp_pipeline_ns = st.phase_ns[kPhasePipeline];
    host_gpu_import_waits(&s.import_n, &s.import_us);
    return s;
}

void meter_thread() {
    host_thread_set_name("bb-meter");
    MainSamples main;
#if !defined(_WIN32)
    PerfMain perf;
    bool perf_tried = false;
    constexpr int kTickMs = 100;  // the loading screen's sampling; the perf ring is read once a second
#else
    constexpr int kTickMs = 10;
#endif
    const bool detail = frame_stats_enabled();
#if defined(_WIN32)
    // A sample stops the main thread for ~40 us (the Radeon 680M laptop: 4.0
    // ms a second at 100 a second). 25 a second keeps that near 0.1% and
    // still gives the minute's summary 1,500 samples; the per-second lines
    // (BBHOST_FRAME_STATS=1) take 100. BBHOST_PERF_METER_HZ sets it (1-100).
    const int sample_every = [detail] {
        const char* e = std::getenv("BBHOST_PERF_METER_HZ");
        const int hz = e && *e ? std::clamp(std::atoi(e), 1, 100) : detail ? 100 : 25;
        return std::max(1, 100 / hz);
    }();
#endif
    Snap prev = take_snap(main);
    std::uint64_t last_flip_ns = 0;
    std::uint64_t loading_ticks = 0, ticks = 0;
    double since_scan = -1;
    auto next = std::chrono::steady_clock::now();
    for (;;) {
        next += std::chrono::milliseconds(kTickMs);
        std::this_thread::sleep_until(next);
        ++ticks;
        loading_ticks += loading_screen_up() ? 1 : 0;
#if !defined(_WIN32)
        if (!perf_tried && g_spin_known) {
            if (const int tid = g_main_tid.load(std::memory_order_relaxed)) {
                perf_tried = true;
                int err = 0;
                main.on = perf.open(tid, &err);
                if (main.on) {
                    host_log("perf meter: the main thread's place every 2 ms of its CPU time (perf_events); the GX barrier's spin measured");
                } else {
                    host_log("perf meter: perf_event_open on the main thread failed (errno %d: %s); the GX barrier's spin is not measured "
                             "(kernel.perf_event_paranoid 2 or lower allows it)",
                             err, std::strerror(err));
                }
            }
        }
#else
        if (HANDLE h = g_main.load(std::memory_order_relaxed); h && g_spin_known && ticks % sample_every == 0) {
            main.on = true;
            const std::uint64_t t0 = host_clock_monotonic_ns();
            if (SuspendThread(h) != static_cast<DWORD>(-1)) {
                alignas(16) CONTEXT ctx{};
                ctx.ContextFlags = CONTEXT_CONTROL;
                const bool got = GetThreadContext(h, &ctx) != 0;
                ResumeThread(h);
                main.stopped_ns += host_clock_monotonic_ns() - t0;
                if (got) {
                    ++main.samples;
                    if (in_spin(ctx.Rip)) ++main.spin;
                }
            }
        }
#endif
        if (ticks * kTickMs < 1000) continue;
        // A second: the window since the last.
#if !defined(_WIN32)
        if (main.on) perf.drain(main);
#endif
        const Snap now = take_snap(main);
        const double secs = static_cast<double>(now.t_ns - prev.t_ns) / 1e9;
        std::vector<std::uint64_t> flips;
        {
            std::lock_guard<std::mutex> lk(g_flip_mu);
            flips.swap(g_flip_ns);
        }
        std::vector<double> iv;
        iv.reserve(flips.size());
        for (std::uint64_t t : flips) {
            if (last_flip_ns && t > last_flip_ns) iv.push_back(static_cast<double>(t - last_flip_ns) / 1e6);
            last_flip_ns = t;
        }
        bottleneck::Window w;
        w.secs = secs;
        w.flips = static_cast<double>(now.flips - prev.flips);
        w.displayed = static_cast<double>(now.displayed - prev.displayed);
        w.cap_fps = g_cap_fps.load(std::memory_order_relaxed);
        if (!iv.empty()) {
            std::vector<double> sorted = iv;
            std::sort(sorted.begin(), sorted.end());
            w.median_ms = sorted[sorted.size() / 2];
            const double over = std::max(2.0 * w.median_ms, 25.0);
            for (double v : iv) {
                if (v > over) w.long_ms += v - w.median_ms;
            }
        }
        w.loading_ms = secs * 1000.0 * static_cast<double>(loading_ticks) / static_cast<double>(ticks);
        if (now.limiter_frames > 0) w.limiter_ms = static_cast<double>(now.limiter_ns - prev.limiter_ns) / 1e6;
        const bool scan = since_scan < 0 || since_scan >= 10;
        const RoleCpu cpu = g_threads.take(scan);
        since_scan = scan ? 0 : since_scan + secs;
        // The limiter's spin is on a core too, but it is the limiter's wait.
        const double limiter_spin_ms = static_cast<double>(now.limiter_spin_ns - prev.limiter_spin_ns) / 1e6;
#if !defined(_WIN32)
        w.main_cpu_ms = main.on ? static_cast<double>(now.main.cpu_ns - prev.main.cpu_ns) / 1e6 : cpu.ms[kRoleMain];
        if (main.on) {
            w.gx_spin_ms = std::min(w.main_cpu_ms, static_cast<double>(now.main.spin - prev.main.spin) * (kPeriodNs / 1e6));
        }
#else
        w.main_cpu_ms = cpu.ms[kRoleMain];
        if (main.on && now.main.samples > prev.main.samples) {
            const double share = static_cast<double>(now.main.spin - prev.main.spin) / static_cast<double>(now.main.samples - prev.main.samples);
            w.gx_spin_ms = share * secs * 1000.0;
            if (w.main_cpu_ms >= 0) w.gx_spin_ms = std::min(w.gx_spin_ms, w.main_cpu_ms);
        }
#endif
        if (w.main_cpu_ms >= 0) w.main_cpu_ms = std::max(0.0, w.main_cpu_ms - limiter_spin_ms);
        w.main_wait_ms = static_cast<double>((now.wait_ns[0] - prev.wait_ns[0]) + (now.wait_ns[1] - prev.wait_ns[1])) / 1e6;
        w.main_file_ms = static_cast<double>(now.wait_ns[2] - prev.wait_ns[2]) / 1e6;
        w.backlog_ms = static_cast<double>(now.backlog_ns - prev.backlog_ns) / 1e6;
        w.render_cpu_ms = cpu.ms[kRoleRender];
        w.cp_cpu_ms = cpu.ms[kRoleCp];
        w.present_cpu_ms = cpu.ms[kRolePresent];
        w.cp_gpu_wait_ms = static_cast<double>(now.cp_fence_us - prev.cp_fence_us) / 1e3 + static_cast<double>(now.cp_label_ns - prev.cp_label_ns) / 1e6;
        if (now.rec_on && prev.rec_on) w.rec_busy_ms = static_cast<double>(now.rec_replay_ns - prev.rec_replay_ns) / 1e6;
        if (now.gpu_on && prev.gpu_on) w.gpu_busy_ms = static_cast<double>(now.gpu_busy_ns - prev.gpu_busy_ns) / 1e6;
        w.import_ms = static_cast<double>(now.import_us - prev.import_us) / 1e3;
        w.pipeline_ms = static_cast<double>(now.cp_pipeline_ns - prev.cp_pipeline_ns) / 1e6;
        Extra x;
        x.gx_workers_ms = cpu.ms[kRoleGx];
        x.gx_workers = cpu.count[kRoleGx];
        x.cp_workers_ms = cpu.ms[kRoleCpWorker];
        x.record_cpu_ms = cpu.ms[kRoleRecord];
        x.meter_ms = cpu.ms[kRoleMeter];
        x.stopped_ms = static_cast<double>(now.main.stopped_ns - prev.main.stopped_ns) / 1e6;
        const bottleneck::Verdict v = bottleneck::classify(w);
        std::uint32_t block = 0;
        const bool block_known = world_player_block(&block);
        if (detail) log_second(w, v, block_name(block_known, block), x);
        {
            std::lock_guard<std::mutex> lk(g_summary_mu);
            Summary& s = g_summary;
            // A summary is one place: the player's map changed, the last one's goes out.
            if (s.secs > 0 && (block_known != s.block_known || (block_known && block != s.block))) flush_summary_locked("perf summary");
            g_summary.block = block;
            g_summary.block_known = block_known;
            add_window(g_summary, w, v, iv, x);
            g_run_seconds[static_cast<int>(v.limit)] += secs;
            g_run_secs += secs;
            if (g_summary.secs >= g_summary_period_s) flush_summary_locked("perf summary");
        }
        prev = now;
        ticks = loading_ticks = 0;
        next = std::chrono::steady_clock::now();
    }
}

}  // namespace

void perf_meter_start(std::uint64_t guest_lo, std::uint64_t guest_hi, bool eboot_109) {
    (void)guest_hi;
    const char* e = std::getenv("BBHOST_PERF_METER");
    if (e && e[0] == '0') {
        host_log("perf meter: off (BBHOST_PERF_METER=0)");
        return;
    }
    if (!host_tooling() && !(e && e[0] == '1')) return;
    if (const char* p = std::getenv("BBHOST_PERF_SUMMARY_S")) g_summary_period_s = std::max(5.0, std::atof(p));
    if (eboot_109) {
        for (int k = 0; k < 2; ++k) {
            g_spin[k][0] = guest_lo + kSpinBn[k][0] - 0x400000ull;
            g_spin[k][1] = guest_lo + kSpinBn[k][1] - 0x400000ull;
        }
        g_spin_known = true;
    }
    frame_stats_time_main_waits();
    g_on.store(true, std::memory_order_release);
    host_log("perf meter: on (bb-meter): which stage holds the frame rate, a `perf summary:` line about every %.0f s%s%s",
             g_summary_period_s, frame_stats_enabled() ? " and a `perf:` line each second (BBHOST_FRAME_STATS=1)" : "",
             eboot_109 ? "" : "; not the 1.09 eboot: the GX barrier's spin is not measured");
    std::thread(meter_thread).detach();
}

void perf_meter_note_main_thread() {
#if defined(_WIN32)
    HANDLE h = nullptr;
    if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h,
                        THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0)) {
        g_main.store(h, std::memory_order_relaxed);
    }
    g_threads.note_main(GetCurrentThreadId());
#else
    g_main_tid.store(static_cast<int>(syscall(SYS_gettid)), std::memory_order_relaxed);
#endif
}

void perf_meter_flip() {
    if (!g_on.load(std::memory_order_relaxed)) return;
    g_flips.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t t = host_clock_monotonic_ns();
    std::lock_guard<std::mutex> lk(g_flip_mu);
    if (g_flip_ns.size() < 8192) g_flip_ns.push_back(t);
}

void perf_meter_limiter(std::uint64_t wait_ns, std::uint64_t spin_ns, float target_fps) {
    if (!g_on.load(std::memory_order_relaxed)) return;
    g_limiter_ns.fetch_add(wait_ns, std::memory_order_relaxed);
    g_limiter_spin_ns.fetch_add(spin_ns, std::memory_order_relaxed);
    g_limiter_frames.fetch_add(1, std::memory_order_relaxed);
    g_cap_fps.store(target_fps, std::memory_order_relaxed);
}

void perf_meter_backlog_wait(std::uint64_t ns) {
    if (g_on.load(std::memory_order_relaxed)) g_backlog_ns.fetch_add(ns, std::memory_order_relaxed);
}

void perf_meter_exit() {
    if (!g_on.load(std::memory_order_acquire)) return;
    // The exit reports can run from a signal: never wait for the meter.
    std::unique_lock<std::mutex> lk(g_summary_mu, std::try_to_lock);
    if (!lk.owns_lock()) return;
    flush_summary_locked("perf summary");
    if (g_run_secs < 1.0) return;
    std::string by;
    std::vector<std::pair<double, int>> order;
    for (int k = 0; k < kStages; ++k) {
        if (g_run_seconds[k] > 0) order.push_back({g_run_seconds[k], k});
    }
    std::sort(order.rbegin(), order.rend());
    for (const auto& [secs, k] : order) {
        char b[48];
        std::snprintf(b, sizeof(b), "%s%s %.0f s", by.empty() ? "" : ", ", bottleneck::stage_name(static_cast<Stage>(k)), secs);
        by += b;
    }
    host_log("perf run: %.0f s: seconds by limit: %s", g_run_secs, by.c_str());
}
