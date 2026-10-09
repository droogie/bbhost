#include "hle/equeue.h"
#include "hle/modules.h"
#include "hle/common.h"
#include "core/portable.h"
#include "hle/platform.h"
#include "hle/hle.h"
#include "engine/guest.h"
#include "host/gpu.h"
#include "core/config.h"
#include "host/window.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <csetjmp>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace {

struct VideoPort {
    int handle = 0;
    int flip_rate = 0;
    std::uint64_t flip_count = 0;
    std::int64_t flip_arg = -1;
    std::uint64_t submit_tsc = 0;
    int current_buffer = -1;
    int pending = 0;                 // queued flips (for GetFlipStatus.flip_pending_num)
    // `mark`: the draw count when the flip was queued (host_gpu_draw_mark), the
    // frame's last draw for the glitch hunt. `gpu_need`: the submissions that
    // must have finished before it may complete (BBHOST_FLIP_AFTER_GPU).
    // `present`: what the presenter is told of it (when it arrived, the
    // submissions its frame is in); `shown`: already presented when it
    // arrived (BBHOST_PRESENT_ON_ARRIVAL), so its vblank only completes it.
    struct FlipReq {
        int buffer;
        std::int64_t arg;
        std::uint64_t mark;
        std::uint64_t gpu_need;
        PresentFlip present;
        bool shown;
    };
    std::deque<FlipReq> flip_queue;  // requested flips awaiting a vblank
    std::chrono::steady_clock::time_point last_complete{};
    HostEqueue* flip_eq = nullptr;
    void* flip_udata = nullptr;
    bool open = false;
    std::uint64_t buffers[16] = {};  // sceVideoOutRegisterBuffers addresses
    unsigned display_w = 0, display_h = 0;  // from the buffer attribute
};

std::mutex g_vo_mu;
std::unordered_map<int, VideoPort> g_vo;
int g_video_next = 1;
std::once_flag g_vsync_once;
std::atomic<bool> g_vsync_run{true};
std::atomic<std::uint64_t> g_flips_done{0};
std::atomic<int> g_fps_cap{30};  // video.fps_cap: upper bound on presented frames/s

// Present on arrival (KyoPS4x #249's split, made here natively): a frame the
// command processor has finished recording is handed to the presenter at
// once, and only its flip - the status, the event the game paces itself on -
// waits for the vblank. Bloodborne keeps two flips queued; showing a frame at
// the tick that completed it held every frame two vblanks after it was ready
// (Kyo measured input 43-48 ms old when shown at 60 fps, 32.5 ms of it this
// queue). The game sees the same cadence and the same pending count, and the
// presenter's blit is submitted after the frame's own work on the one queue,
// earlier than before, so it cannot meet a later frame's writes to that
// buffer either (the game draws into a buffer only after the flip that
// showed it is complete). BBHOST_PRESENT_ON_ARRIVAL=0: shown at completion.
// (g_present_on_arrival is defined with the presenter's flip, below.)

// The pacing line of the 300-flip report (hle_video_pacing_window): how late
// the vblank clock's ticks woke, and how long a flip took from being queued
// to being shown and to being completed. Written by the vblank thread and
// the command processor, read and reset by the report.
struct PacingStats {
    std::atomic<std::uint64_t> ticks{0}, tick_late_us{0}, tick_late_max_us{0}, ticks_late_1ms{0};
    std::atomic<std::uint64_t> shown{0}, shown_us{0}, shown_max_us{0};
    std::atomic<std::uint64_t> completed{0}, completed_us{0}, completed_max_us{0};
};
PacingStats g_pacing;
void pacing_max(std::atomic<std::uint64_t>& m, std::uint64_t v) {
    std::uint64_t cur = m.load(std::memory_order_relaxed);
    while (v > cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}
std::uint64_t steady_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
void pacing_note_shown(std::uint64_t queued_ns) {
    const std::uint64_t us = (steady_ns() - queued_ns) / 1000;
    g_pacing.shown.fetch_add(1, std::memory_order_relaxed);
    g_pacing.shown_us.fetch_add(us, std::memory_order_relaxed);
    pacing_max(g_pacing.shown_max_us, us);
}
void pacing_note_completed(std::uint64_t queued_ns) {
    const std::uint64_t us = (steady_ns() - queued_ns) / 1000;
    g_pacing.completed.fetch_add(1, std::memory_order_relaxed);
    g_pacing.completed_us.fetch_add(us, std::memory_order_relaxed);
    pacing_max(g_pacing.completed_max_us, us);
}

void post_flip_event(VideoPort& p) {
    if (!p.flip_eq) {
        return;
    }
    HostEvent ev{};
    ev.ident = 0;
    ev.filter = -13;
    ev.fflags = static_cast<std::uint32_t>(p.flip_count);
    ev.data = static_cast<std::intptr_t>(p.flip_arg);
    ev.udata = p.flip_udata;
    equeue_post(p.flip_eq, ev);
}

void finish_flip_locked(VideoPort& p, int buffer, std::int64_t arg) {
    p.current_buffer = buffer;
    p.flip_arg = arg;
    p.submit_tsc = rdtsc_now();
    p.pending = 0;
    p.flip_count += 1;
    post_flip_event(p);
}

// A flip completes on a vblank whose count is a multiple of (flip_rate+1):
// flip_rate 0 -> every vblank (60 fps), 1 -> every second (30 fps), matching
// hardware. This paces the game's frame loop, which waits on the flip event
// before submitting the next frame; without it the loop ran as fast as the
// GPU finished each frame and menus/animations ran at the display rate.
std::atomic<bool> g_flip_uncapped{false};
// While a loading screen is up (engine/loading.cpp): flips complete at once,
// as under BBHOST_UNCAP, so the game's loaders - stepped once a frame - are
// not held to the frame cap.
std::atomic<bool> g_flip_loading{false};
std::atomic<std::uint64_t> g_picture{0};  // hle_video_set_picture: w << 32 | h, 0 for the whole buffer

// Hands a frame to the presenter (host/window.cpp): the display buffer, cut to
// the picture the game draws in it (hle_video_set_picture).
void show_frame(int buffer, std::uint64_t display_va, unsigned dw, unsigned dh, const PresentFlip& flip) {
    if (const std::uint64_t pic = g_picture.load(std::memory_order_relaxed)) {
        dw = std::min(dw, static_cast<unsigned>(pic >> 32));
        dh = std::min(dh, static_cast<unsigned>(pic & 0xffffffffu));
    }
    host_present(buffer, display_va, dw, dh, flip);
}

// BBHOST_PRESENT_ON_ARRIVAL (on unless 0): a flip the command processor queues
// is shown at once, and its vblank only completes it. Bloodborne keeps two
// flips pending, and the vblank both showed and completed one a tick, oldest
// first, so every frame waited behind the two before it: two flip periods
// (33 ms at 60, 67 at 30) between the frame being in the GPU's queue and the
// presenter even seeing it. Kyo measured the same in his fork - input 43-48 ms
// old when shown - and split showing from completing there (KyoPS4x #249);
// this is that design on our flip queue. What the game paces on is untouched:
// flip status, the flip event and the pending count still change on the tick,
// one flip per tick, oldest first. Showing early cannot race the game reusing
// the buffer: the game draws into a buffer only once a later flip has
// completed over it, and the presenter's copy of it is queued on the GPU
// behind the frame's own work, ahead of anything the game queues afterwards.
// Only flips the command processor queued after recording their frame
// (sceGnmSubmitAndFlipCommandBuffers) are shown early; a flip requested from a
// game thread (sceGnmRequestFlipAndSubmitDone) may be ahead of its frame's
// commands, and waits for its vblank as before. BBHOST_PRESENT_LEGACY=1 (the
// presenter's switch back, host/window.cpp) turns it off with the rest.
const bool g_present_on_arrival = [] {
    if (const char* e = std::getenv("BBHOST_PRESENT_ON_ARRIVAL")) return e[0] != '0';
    const char* legacy = std::getenv("BBHOST_PRESENT_LEGACY");
    return !(legacy && legacy[0] == '1');
}();

// BBHOST_TEST_REQUESTS=<dir>: a test harness's requests to the running game
// (tools/area_check.py, through its probe plugin, at moments the game's own
// clock picks). Each <dir>/*.req file holds one line, carried out in name
// order and then deleted:
//   dump <path>           this displayed frame to <path>, as BBHOST_DUMP_FRAME writes one
//   capture <n> [<dir>]   with BBHOST_CAPTURE_ARMED=1, BBHOST_CAPTURE_DRAW may take up to n
//                         more draws, into <dir> (draw_capture.h); 0 stops it
// The folder is read ten times a second, on the flips.
void test_requests(std::uint64_t display_va, std::uint64_t count) {
    static const std::string dir = [] {
        const char* e = std::getenv("BBHOST_TEST_REQUESTS");
        return std::string(e && *e ? e : "");
    }();
    if (dir.empty()) return;
    // The vblank's flips and the uncapped ones come here from two threads.
    static std::mutex mu;
    std::unique_lock<std::mutex> lock(mu, std::try_to_lock);
    if (!lock.owns_lock()) return;
    static std::chrono::steady_clock::time_point last{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::milliseconds(100)) return;
    last = now;
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<fs::path> requests;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().extension() == ".req") requests.push_back(it->path());
    }
    std::sort(requests.begin(), requests.end());
    for (const fs::path& p : requests) {
        std::string verb, arg, arg2;
        {
            std::ifstream f(p);
            f >> verb >> arg >> arg2;
        }
        fs::remove(p, ec);
        if (verb == "dump" && !arg.empty()) {
            const bool ok = display_va && host_gpu_dump_display(display_va, arg.c_str());
            host_log("test: frame dump at flip %llu to %s%s", static_cast<unsigned long long>(count), arg.c_str(), ok ? "" : " failed");
        } else if (verb == "capture" && !arg.empty()) {
            const int n = std::atoi(arg.c_str());
            host_gpu_capture_arm(n, arg2.c_str());
            host_log("test: draw captures armed for %d at flip %llu%s%s", n, static_cast<unsigned long long>(count), arg2.empty() ? "" : " into ",
                     arg2.c_str());
        } else {
            host_log("test: request %s not understood: %s %s", p.filename().string().c_str(), verb.c_str(), arg.c_str());
        }
    }
}

// `shown`: the presenter already has the frame (BBHOST_PRESENT_ON_ARRIVAL);
// everything else a completed flip does still happens here.
void present_flip(int buffer, std::uint64_t display_va, unsigned dw, unsigned dh, std::uint64_t count, std::uint64_t mark, bool shown,
                  const PresentFlip& flip) {
    // BBHOST_DUMP_FRAME=N[,M,...]: write the displayed frame at those flips.
    static std::vector<long> dump_at = [] {
        std::vector<long> out;
        const char* e = std::getenv("BBHOST_DUMP_FRAME");
        while (e && *e) {
            char* end = nullptr;
            const long v = std::strtol(e, &end, 10);
            if (end == e) break;
            out.push_back(v);
            e = *end == ',' ? end + 1 : end;
        }
        return out;
    }();
    static std::size_t next_dump = 0;
    if (next_dump < dump_at.size() && static_cast<long>(count) >= dump_at[next_dump] && display_va) {
        char path[64];
        std::snprintf(path, sizeof(path), "build/frame-%ld.ppm", dump_at[next_dump]);
        ++next_dump;
        host_gpu_dump_display(display_va, path);
    }
    test_requests(display_va, count);
    // BBHOST_STALL_TEST=<flip>:<ms>
    static const std::pair<long, long> stall_test = [] {
        const char* e = std::getenv("BBHOST_STALL_TEST");
        if (!e || !e[0]) return std::pair<long, long>{-1, 0};
        char* end = nullptr;
        const long at = std::strtol(e, &end, 10);
        return std::pair<long, long>{at, *end == ':' ? std::strtol(end + 1, nullptr, 10) : 1000};
    }();
    if (stall_test.first >= 0 && static_cast<long>(count) == stall_test.first) {
        host_gpu_stall_test(static_cast<unsigned>(stall_test.second));
    }
    host_gpu_watch_display(display_va);
    host_gpu_glitch_watch(display_va, mark);
    // BBHOST_F12_AT=N[,M,...]: the F12 dump at those flips, for scripted runs.
    static std::vector<long> f12_at = [] {
        std::vector<long> out;
        const char* e = std::getenv("BBHOST_F12_AT");
        while (e && *e) {
            char* end = nullptr;
            const long v = std::strtol(e, &end, 10);
            if (end == e) break;
            out.push_back(v);
            e = *end == ',' ? end + 1 : end;
        }
        return out;
    }();
    static std::size_t next_f12 = 0;
    if (next_f12 < f12_at.size() && static_cast<long>(count) >= f12_at[next_f12]) {
        ++next_f12;
        host_gpu_request_dump();
    }
    // BBHOST_RT_REFILL_TEST=1: the re-created-target check, once the device is busy.
    static bool refill_test = [] {
        const char* e = std::getenv("BBHOST_RT_REFILL_TEST");
        return e && e[0] == '1';
    }();
    if (refill_test && count >= 120) {
        refill_test = false;
        host_gpu_refill_selftest();
    }
    // F12: the images and the draw list in the folder the request made.
    if (std::string dir; host_gpu_take_dump_request(&dir) && display_va) {
        const std::string path = dir + "/f12-" + std::to_string(count) + host_gpu_capture_ext();
        host_gpu_dump_display(display_va, path.c_str(), true);
    }
    if (!shown) show_frame(buffer, display_va, dw, dh, flip);  // hands it over; the presenter's thread waits for the display
    engine_on_flip(count);
}

// Completes at most one queued flip per open port, gated by the flip rate.
// BBHOST_FLIP_AFTER_GPU=1: a flip completes only once the GPU has run its
// frame. The command processor queues a flip when it has *recorded* the
// frame, and the vblank used to complete it whether or not the GPU had
// executed any of it - while on the console a flip is the GPU reaching it,
// so a game may take the flip for "the GPU is done with that frame's
// buffers" and write the next frame's data over them.
const bool g_flip_after_gpu = [] {
    const char* e = std::getenv("BBHOST_FLIP_AFTER_GPU");
    return e && e[0] == '1';
}();
std::atomic<std::uint64_t> g_flips_held{0};

void vblank_tick(std::uint64_t vcount) {
    const std::uint64_t gpu_done = g_flip_after_gpu ? host_gpu_submissions_completed() : ~0ull;
    struct Done {
        int buffer;
        std::uint64_t va;
        unsigned w, h, count;
        std::uint64_t mark;
        bool shown;
        PresentFlip flip;
    };
    std::vector<Done> present;
    {
        std::lock_guard<std::mutex> lock(g_vo_mu);
        for (auto& kv : g_vo) {
            VideoPort& p = kv.second;
            if (!p.open || p.flip_queue.empty()) {
                continue;
            }
            if (vcount % (static_cast<std::uint64_t>(p.flip_rate) + 1) != 0) {
                continue;
            }
            if (p.flip_queue.front().gpu_need > gpu_done) {
                const std::uint64_t n = g_flips_held.fetch_add(1, std::memory_order_relaxed);
                if (n == 0 || n % 3600 == 0) {
                    host_log("video: a flip waited a vblank for the GPU to run its frame (%llu vblanks so far; BBHOST_FLIP_AFTER_GPU)",
                             static_cast<unsigned long long>(n + 1));
                }
                continue;  // the GPU has not run the frame yet: a later vblank
            }
            // Also honour video.fps_cap: Bloodborne asks for flip_rate 0 (60)
            // but is a 30 fps title paced by GPU time on hardware; the cap
            // stands in for that so a fast host does not run it at 60.
            const int cap = g_fps_cap.load();
            if (cap > 0) {
                const auto now = std::chrono::steady_clock::now();
                const auto min_gap = std::chrono::microseconds(1000000 / cap - 1000000 / 120);  // allow a little jitter
                if (p.last_complete.time_since_epoch().count() != 0 && now - p.last_complete < min_gap) {
                    continue;
                }
                p.last_complete = now;
            }
            const VideoPort::FlipReq req = p.flip_queue.front();
            p.flip_queue.pop_front();
            p.pending = static_cast<int>(p.flip_queue.size());
            finish_flip_locked(p, req.buffer, req.arg);
            g_flips_done.store(p.flip_count);
            pacing_note_completed(req.present.arrived_ns);
            if (!req.shown) pacing_note_shown(req.present.arrived_ns);
            const std::uint64_t va = (req.buffer >= 0 && req.buffer < 16) ? p.buffers[req.buffer] : 0;
            // A flip not shown on arrival behind one that was (a game-thread
            // flip among the command processor's): showing it now would take
            // the display back a frame, so it only completes.
            bool shown = req.shown;
            for (const VideoPort::FlipReq& later : p.flip_queue) shown = shown || later.shown;
            present.push_back({req.buffer, va, p.display_w, p.display_h, static_cast<unsigned>(p.flip_count), req.mark, shown, req.present});
        }
    }
    for (const Done& d : present) {
        present_flip(d.buffer, d.va, d.w, d.h, d.count, d.mark, d.shown, d.flip);
    }
}

void start_vsync() {
    std::call_once(g_vsync_once, [] {
        // Set, never cleared: the game at 90 or uncapped has set it already
        // (hle_video_set_game_pace).
        if (const char* e = std::getenv("BBHOST_UNCAP"); e && e[0] == '1') g_flip_uncapped.store(true);
        host_log("video: %s", g_present_on_arrival
                                  ? "a frame is shown when the command processor has it, its flip completes on the vblank "
                                    "(BBHOST_PRESENT_ON_ARRIVAL=0: shown when its flip completes)"
                                  : "a frame is shown when its flip completes on the vblank (BBHOST_PRESENT_ON_ARRIVAL=0)");
        std::thread([] {
            // Named on both hosts (Windows: for the sampler and frame
            // statistics' thread list too), and above every other thread on
            // Windows: it does little, and a tick a guest worker delays is a
            // flip completed late (core/host_clock.h, BBHOST_THREAD_PRIO).
            host_thread_set_name("bb-vblank");
            host_thread_set_class(HostThreadClass::Pacing, "bb-vblank");
            // BBHOST_VBLANK_SPIN_US=<n>: sleep to n us before each tick on the
            // high-resolution timer and spin the rest with `pause`, for a tick
            // on the microsecond. 0 (the default, as Kyo's vblank thread in
            // KyoPS4x #249): the timer alone - its lateness is on the pacing
            // line, and a spinning core takes the APU's GPU power.
            static const std::int64_t spin_us = [] {
                const char* e = std::getenv("BBHOST_VBLANK_SPIN_US");
                return e ? std::clamp<std::int64_t>(std::strtoll(e, nullptr, 10), 0, 4000) : 0;
            }();
            if (spin_us) host_log("video: the flip clock spins the last %lld us before each tick (BBHOST_VBLANK_SPIN_US)", static_cast<long long>(spin_us));
            // The flip clock runs at the display's own refresh rate,
            // not at an assumed 59.94 Hz. A flip still completes at the rate
            // the game asked for (flip_rate) and the frame cap, so this only
            // decides which display refreshes a completed flip may land on -
            // which is what a PC game's V-Sync does. video.vblank_hz overrides
            // it for a display that reports the wrong rate; headless runs and
            // a display that says nothing keep 60.
            // 60.00 Hz when the display does not say, which is what a PC
            // monitor runs at and what this clock used before it asked.
            // (59.94 is the NTSC video rate the PS4 sends over HDMI, not a
            // rate a PC display has.)
            constexpr auto kDefaultPeriod = std::chrono::nanoseconds(16666667);
            auto period = kDefaultPeriod;
            float announced_hz = -1.0f;  // -1: nothing said yet, so the first look always says
            const auto period_for = [](float hz) {
                return std::chrono::nanoseconds(static_cast<std::int64_t>(1e9f / hz));
            };
            bool forced_rate = false;
            if (const int forced = config().vblank_hz; forced >= 20 && forced <= 1000) {
                period = period_for(static_cast<float>(forced));
                announced_hz = static_cast<float>(forced);
                forced_rate = true;
                host_log("video: flip clock at %d Hz (video.vblank_hz)", forced);
            }
            auto next = std::chrono::steady_clock::now();
            std::uint64_t vcount = 0;
            std::uint64_t last_flips = 0;
            std::uint64_t stuck_since = 0;
            bool reported = false;
            while (g_vsync_run.load()) {
                // Follow the display's rate, asked once a second so a window
                // dragged to another monitor follows it.
                //
                // Not its *phase*: that needs a real vsync signal to lock to
                // (VK_KHR_present_wait, or the presentation-timing
                // extension). Steering by when the presenting thread's last
                // present returned was tried and removed - that thread runs
                // behind this one, and where there is no real vsync (a
                // compositor, a virtual display) the present returns at once,
                // so the timestamp carries no phase at all and the loop
                // chases its own tail: asked for 30 Hz it ran at 40, and
                // correcting modulo the period made it wander instead
                // (26.7 and 57.1 for 30 and 50). A rate with no phase is
                // honest; a phase locked to nothing is not.
                if (!forced_rate && (vcount % 60) == 0) {
                    const float hz = host_window_refresh_hz();
                    const auto want = hz >= 20.0f && hz <= 1000.0f ? period_for(hz) : kDefaultPeriod;
                    period = want;
                    if (hz != announced_hz) {
                        announced_hz = hz;
                        if (hz >= 20.0f && hz <= 1000.0f) {
                            host_log("video: flip clock at %.2f Hz, the display's own", static_cast<double>(hz));
                        } else {
                            host_log("video: flip clock at 60 Hz (the display does not say its refresh rate)");
                        }
                    }
                }
                next += period;
                // On the high-resolution waitable timer (host_sleep_until ->
                // host_sleep_us), to an absolute deadline, so a late tick does
                // not move the ones after it.
                if (spin_us) {
                    host_sleep_until(next - std::chrono::microseconds(spin_us));
                    while (std::chrono::steady_clock::now() < next) {
#if defined(__x86_64__) || defined(_M_X64)
                        __builtin_ia32_pause();
#endif
                    }
                } else {
                    host_sleep_until(next);
                }
                {
                    const auto late = std::chrono::steady_clock::now() - next;
                    const auto late_us = static_cast<std::uint64_t>(std::max<std::int64_t>(
                        0, std::chrono::duration_cast<std::chrono::microseconds>(late).count()));
                    g_pacing.ticks.fetch_add(1, std::memory_order_relaxed);
                    g_pacing.tick_late_us.fetch_add(late_us, std::memory_order_relaxed);
                    pacing_max(g_pacing.tick_late_max_us, late_us);
                    if (late_us >= 1000) g_pacing.ticks_late_1ms.fetch_add(1, std::memory_order_relaxed);
                }
                vblank_tick(++vcount);
                // BBHOST_VBLANK_LOG=1: what the clock and the completions
                // actually came out at, once a second.
                static const bool tick_log = [] {
                    const char* e = std::getenv("BBHOST_VBLANK_LOG");
                    return e && e[0] == '1';
                }();
                if (tick_log) {
                    static auto since = std::chrono::steady_clock::now();
                    static std::uint64_t ticks_then = 0, flips_then = 0;
                    const auto now = std::chrono::steady_clock::now();
                    const double secs = std::chrono::duration<double>(now - since).count();
                    if (secs >= 1.0) {
                        const std::uint64_t flips_now = g_flips_done.load();
                        host_log("vblank: %.1f ticks/s, %.1f flips completed/s", (vcount - ticks_then) / secs,
                                 (flips_now - flips_then) / secs);
                        since = now;
                        ticks_then = vcount;
                        flips_then = flips_now;
                    }
                }
                // Watchdog: the game freezing shows up as flips stopping while
                // the vblank keeps ticking. Say so once per stall and print the
                // last draws, which is usually enough to name the shader or
                // resource involved.
                // BBHOST_POOL_LOG=1: the same pool every 10 s, to see it fill
                // (and what it looks like on a run that does not freeze).
                static const bool pool_log = [] {
                    const char* e = std::getenv("BBHOST_POOL_LOG");
                    return e && e[0] == '1';
                }();
                // BBHOST_POOL_LOG=2: a thread that watches the pool as fast as
                // it can, so a hold that lasts microseconds is still seen -
                // how many entries are out at once says whether two threads
                // are ever in that path together.
                static const bool pool_watch = [] {
                    const char* e = std::getenv("BBHOST_POOL_LOG");
                    return e && e[0] == '2';
                }();
                if (pool_watch) {
                    static std::once_flag once;
                    std::call_once(once, [] {
                        std::thread([] {
                            pthread_setname_np(pthread_self(), "bb-poolwatch");
                            int lowest = 99, last = -2;
                            for (;;) {
                                const int free_now = hle_guest_pool_free();
                                if (free_now >= 0 && free_now < lowest) {
                                    lowest = free_now;
                                    host_log("pool: as few as %d entries free at flip %llu", lowest,
                                             static_cast<unsigned long long>(hle_video_flip_count()));
                                }
                                if (free_now != last && free_now >= 0) last = free_now;
                                std::this_thread::yield();
                            }
                        }).detach();
                    });
                }
                if (pool_log) {
                    // Every change in how many entries are free, with the flip
                    // it happened on.
                    static int last_free = -2;
                    const int free_now = hle_guest_pool_free();
                    if (free_now != last_free) {
                        last_free = free_now;
                        host_log("pool: %d entries free at flip %llu", free_now,
                                 static_cast<unsigned long long>(g_flips_done.load()));
                    }
                }
                // In seconds, not vblanks: the clock runs at the display's
                // rate, and at 180 Hz twenty seconds of vblanks at 60 were
                // under seven.
                const std::uint64_t flips = g_flips_done.load();
                static auto stuck_at = std::chrono::steady_clock::now();
                if (flips != last_flips) {
                    last_flips = flips;
                    stuck_since = vcount;
                    stuck_at = std::chrono::steady_clock::now();
                    reported = false;
                } else if (flips > 60 && !reported && std::chrono::steady_clock::now() - stuck_at > std::chrono::seconds(20)) {
                    reported = true;
                    host_log("watchdog: no flip for %lld s at flip %llu; the game or the renderer is stuck",
                             static_cast<long long>(std::chrono::duration_cast<std::chrono::seconds>(
                                                        std::chrono::steady_clock::now() - stuck_at)
                                                        .count()),
                             static_cast<unsigned long long>(flips));
                    host_gpu_hang_report();
                    hle_gnm_label_report();
                    hle_gnm_wait_report();
                    hle_guest_pool_report();
                }
            }
        }).detach();
    });
}

VideoPort* vo_get(int handle) {
    auto it = g_vo.find(handle);
    if (it == g_vo.end() || !it->second.open) {
        return nullptr;
    }
    return &it->second;
}

GUEST_ABI int hle_video_open(int, int, int, const void*) {
    start_vsync();
    std::lock_guard<std::mutex> lock(g_vo_mu);
    int h = g_video_next++;
    VideoPort p{};
    p.handle = h;
    p.open = true;
    g_vo[h] = p;
    host_log("sceVideoOutOpen -> %d", h);
    return h;
}

GUEST_ABI int hle_video_close(int handle) {
    std::lock_guard<std::mutex> lock(g_vo_mu);
    auto it = g_vo.find(handle);
    if (it == g_vo.end()) {
        return static_cast<int>(0x80290001);
    }
    it->second.open = false;
    it->second.flip_eq = nullptr;
    return 0;
}

GUEST_ABI int hle_video_flip_rate(int handle, int rate) {
    std::lock_guard<std::mutex> lock(g_vo_mu);
    VideoPort* p = vo_get(handle);
    if (!p) {
        return static_cast<int>(0x80290001);
    }
    p->flip_rate = rate;
    host_log("sceVideoOutSetFlipRate handle=%d rate=%d (%d fps)", handle, rate, rate >= 0 ? 60 / (rate + 1) : 60);
    return 0;
}

GUEST_ABI int hle_video_add_flip(HostEqueue* eq, int handle, void* udata) {
    if (!eq || !equeue_live(eq)) {
        return static_cast<int>(0x80290004);
    }
    std::lock_guard<std::mutex> lock(g_vo_mu);
    VideoPort* p = vo_get(handle);
    if (!p) {
        return static_cast<int>(0x80290001);
    }
    p->flip_eq = eq;
    p->flip_udata = udata;
    host_log("AddFlipEvent handle=%d eq=%p", handle, static_cast<void*>(eq));
    return 0;
}

GUEST_ABI int hle_video_set_attr(void* attr, unsigned fmt, unsigned tile, unsigned aspect,
                                unsigned w, unsigned h, unsigned pitch) {
    if (attr) {
        auto* a = static_cast<unsigned*>(attr);
        a[0] = fmt;
        a[1] = tile;
        a[2] = aspect;
        a[3] = w;
        a[4] = h;
        a[5] = pitch;
    }
    return 0;
}

GUEST_ABI int hle_video_register(int handle, int start, void** addrs, int n, const void* attr) {
    std::lock_guard<std::mutex> lock(g_vo_mu);
    if (!vo_get(handle)) {
        return static_cast<int>(0x80290001);
    }
    const auto* at = static_cast<const unsigned*>(attr);
    host_log("RegisterBuffers handle=%d start=%d n=%d addr0=%p attr=%ux%u pitch=%u fmt=%u tile=%u", handle, start, n,
             addrs && n > 0 ? addrs[0] : nullptr, at ? at[3] : 0, at ? at[4] : 0, at ? at[5] : 0, at ? at[0] : 0,
             at ? at[1] : 0);
    VideoPort* p = vo_get(handle);
    for (int i = 0; addrs && i < n; ++i) {
        const int idx = start + i;
        if (idx >= 0 && idx < 16) p->buffers[idx] = reinterpret_cast<std::uint64_t>(addrs[i]);
    }
    if (attr) {
        const auto* a = static_cast<const unsigned*>(attr);
        p->display_w = a[3];
        p->display_h = a[4];
    }
    return 0;
}

GUEST_ABI int hle_video_flip(int handle, int buffer, int, std::int64_t arg) {
    std::lock_guard<std::mutex> lock(g_vo_mu);
    VideoPort* p = vo_get(handle);
    if (!p) {
        return static_cast<int>(0x80290001);
    }
    // Completed on a later vblank (paced by flip_rate). A short queue holds
    // in-flight flips; report FLIP_QUEUE_FULL when it backs up.
    if (p->flip_queue.size() >= 4) {
        return static_cast<int>(0x80290012);  // SCE_VIDEO_OUT_ERROR_FLIP_QUEUE_FULL
    }
    PresentFlip flip;
    flip.arrived_ns = steady_ns();
    p->flip_queue.push_back({buffer, arg, host_gpu_draw_mark(), g_flip_after_gpu ? host_gpu_work_needs() : 0, flip, false});
    p->pending = static_cast<int>(p->flip_queue.size());
    static int flip_logs;
    if (flip_logs < 8) {
        host_log("SubmitFlip handle=%d buf=%d arg=%lld count=%llu", handle, buffer,
                 static_cast<long long>(arg), static_cast<unsigned long long>(p->flip_count));
        ++flip_logs;
    }
    return 0;
}

GUEST_ABI int hle_video_status(int handle, void* st) {
    if (!st) {
        return sce_err(EINVAL);
    }
    std::lock_guard<std::mutex> lock(g_vo_mu);
    VideoPort* p = vo_get(handle);
    if (!p) {
        return static_cast<int>(0x80290001);
    }
    struct FlipStatus {
        std::uint64_t count;
        std::uint64_t process_time;
        std::uint64_t tsc;
        std::int64_t flip_arg;
        std::uint64_t submit_tsc;
        std::uint64_t reserved0;
        std::int32_t gc_queue_num;
        std::int32_t flip_pending_num;
        std::int32_t current_buffer;
        std::uint32_t reserved1;
    };
    FlipStatus out{};
    out.count = p->flip_count;
    out.process_time = now_us();
    out.tsc = rdtsc_now();
    out.flip_arg = p->flip_arg;
    out.submit_tsc = p->submit_tsc;
    out.gc_queue_num = 0;
    out.flip_pending_num = p->pending;
    out.current_buffer = p->current_buffer;
    std::memcpy(st, &out, sizeof(out));
    return 0;
}

}  // namespace

void hle_video_set_picture(unsigned w, unsigned h) {
    g_picture.store(w && h ? static_cast<std::uint64_t>(w) << 32 | h : 0, std::memory_order_relaxed);
}

std::uint64_t hle_video_flip_count() { return g_flips_done.load(); }

std::string hle_video_pacing_window() {
    const auto take = [](std::atomic<std::uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
    const std::uint64_t ticks = take(g_pacing.ticks), late = take(g_pacing.tick_late_us), late_max = take(g_pacing.tick_late_max_us),
                        late_1ms = take(g_pacing.ticks_late_1ms);
    const std::uint64_t shown = take(g_pacing.shown), shown_us = take(g_pacing.shown_us), shown_max = take(g_pacing.shown_max_us);
    const std::uint64_t done = take(g_pacing.completed), done_us = take(g_pacing.completed_us), done_max = take(g_pacing.completed_max_us);
    const auto avg = [](std::uint64_t sum, std::uint64_t n) { return n ? static_cast<double>(sum) / static_cast<double>(n) : 0.0; };
    char buf[400];
    std::snprintf(buf, sizeof(buf),
                  "vblank ticks %llu woke late avg %.0f us max %llu us (%llu over 1 ms); flips queued->shown avg %.2f ms max %.2f "
                  "(%llu), queued->completed avg %.2f ms max %.2f (%llu)%s",
                  static_cast<unsigned long long>(ticks), avg(late, ticks), static_cast<unsigned long long>(late_max),
                  static_cast<unsigned long long>(late_1ms), avg(shown_us, shown) / 1000.0, static_cast<double>(shown_max) / 1000.0,
                  static_cast<unsigned long long>(shown), avg(done_us, done) / 1000.0, static_cast<double>(done_max) / 1000.0,
                  static_cast<unsigned long long>(done), g_present_on_arrival ? " (shown on arrival)" : " (shown on completion)");
    return buf;
}

void hle_video_set_loading_uncapped(bool on) { g_flip_loading.store(on, std::memory_order_relaxed); }

// The game's pace once engine/frame_rate.cpp chose it; -1 before.
std::atomic<int> g_game_pace{-1};

void hle_video_set_game_pace(int fps) {
    g_game_pace.store(fps);
    if (fps == 90 || fps == 0) {
        // The game paces itself - our SprjFlipper::Update waits out 1/90 s, or
        // the uncapped limit - and steps by the frame time it measured, so a
        // flip completes as soon as the frame is recorded, as under
        // BBHOST_UNCAP: held to the display's refresh, 90 on a 60 Hz display
        // would run at 60 with 90's steps. The presenter still shows one frame
        // a refresh with V-Sync on (host/window.cpp drops the others).
        g_flip_uncapped.store(true);
        g_fps_cap.store(0);
        host_log("video: flips complete as soon as they are recorded (the game paces itself at %s)", fps ? "90" : "uncapped");
    } else {
        g_fps_cap.store(fps);
    }
}

void hle_video_set_fps_cap(int fps) {
    // The game's pace is chosen when it starts (engine/frame_rate.h): its
    // steps are converted for that rate, so capping it lower now would slow
    // the game itself down. A frame rate picked while it runs applies on the
    // next run.
    if (const int pace = g_game_pace.load(); pace >= 0) {
        if (fps != pace) {
            host_log("video: frame rate %d applies on the next run (the game is running at %d; 0 is uncapped)", fps, pace);
        }
        return;
    }
    g_fps_cap.store(fps);
}

// A loading screen's flips complete at once, so the loaders, which step once a
// frame, run as fast as the machine allows (engine/loading.h; KyoPS4x #225,
// #235, #238). The window needs none of the extra frames, and each one shown
// was a blit and a present: ~300 a second through a load on the Radeon 8060S,
// bb-present at 53-58% of a core and the load's seconds at 450-540% CPU. So
// a loading frame is shown at most once a frame period of the game's pace
// (the display's refresh when it runs uncapped); the rest complete unshown.
// Their work is on the GPU already (the command processor submits after
// every job), only the copy to the window is left out.
static std::atomic<std::uint64_t> g_loading_shown_ns{0};
static std::atomic<std::uint64_t> g_loading_unshown{0};
static bool loading_show_due(std::uint64_t now_ns) {
    const int pace = g_game_pace.load(std::memory_order_relaxed);
    float hz = pace > 0 ? static_cast<float>(pace) : host_window_refresh_hz();
    if (hz < 30.0f) hz = 60.0f;
    const auto period = static_cast<std::uint64_t>(1e9f / hz);
    // An eighth of slack: frames ~3 ms apart still land a 60 pace on 60.
    if (now_ns - g_loading_shown_ns.load(std::memory_order_relaxed) < period - period / 8) {
        g_loading_unshown.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    g_loading_shown_ns.store(now_ns, std::memory_order_relaxed);
    return true;
}
std::uint64_t hle_video_loading_unshown() { return g_loading_unshown.load(std::memory_order_relaxed); }

void hle_video_finish_flip(int handle, int buffer, std::int64_t arg, bool recorded) {
    // Called from the CP thread once it has recorded the frame's GPU work
    // (not once the GPU has run it: BBHOST_FLIP_AFTER_GPU). The flip is
    // queued and completed by the vblank tick at the game's flip rate, so the
    // game's frame loop paces on the flip event. BBHOST_UNCAP=1 completes it
    // at once (fast headless dumps). `recorded`: from the command processor,
    // after the frame's commands (else a game thread's request, which may be
    // ahead of them).
    const std::uint64_t mark = host_gpu_draw_mark();
    // The submissions the frame is in: the presenter submits the renderer's
    // recording only when the frame is still in it (host_gpu_submit_for_flip).
    const std::uint64_t work = g_flip_after_gpu || recorded ? host_gpu_work_needs() : 0;
    const std::uint64_t gpu_need = g_flip_after_gpu ? work : 0;
    PresentFlip flip;
    flip.submit_need = recorded ? work : ~0ull;
    flip.arrived_ns = steady_ns();
    if (!g_flip_uncapped.load() && !g_flip_loading.load(std::memory_order_relaxed)) {
        std::uint64_t display_va = 0;
        unsigned dw = 0, dh = 0;
        const bool show = g_present_on_arrival && recorded;
        {
            std::lock_guard<std::mutex> lock(g_vo_mu);
            VideoPort* p = vo_get(handle);
            if (!p) {
                return;
            }
            if (p->flip_queue.size() >= 4) {
                p->flip_queue.pop_front();  // drop the oldest rather than stall the CP
            }
            flip.on_arrival = show;
            p->flip_queue.push_back({buffer, arg, mark, gpu_need, flip, show});
            p->pending = static_cast<int>(p->flip_queue.size());
            if (!show) return;
            if (buffer >= 0 && buffer < 16) display_va = p->buffers[buffer];
            dw = p->display_w;
            dh = p->display_h;
        }
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) {
            host_log("video: frames are shown when their flip is queued and completed on the vblank (BBHOST_PRESENT_ON_ARRIVAL=0: "
                     "both on the vblank)");
        }
        show_frame(buffer, display_va, dw, dh, flip);
        pacing_note_shown(flip.arrived_ns);
        return;
    }
    std::uint64_t display_va = 0;
    std::uint64_t count = 0;
    unsigned dw = 0, dh = 0;
    {
        std::lock_guard<std::mutex> lock(g_vo_mu);
        VideoPort* p = vo_get(handle);
        if (!p) {
            return;
        }
        // Flips queued before a loading screen lifted the cap complete first,
        // in order (not presented: this one is newer). Left to the vblank,
        // they completed after this one and the flip status's arg went
        // backwards - and the game's flip (Gnm_WaitAndSubmitFlip, 0x2ad6e70)
        // waits on the flip event until that arg reaches the oldest frame in
        // its swap chain: no flip was coming, so it waited for ever (a Steam
        // Deck at 60 fps, the title's Continue: 4 runs in 41).
        if (!p->flip_queue.empty()) {
            static std::atomic<int> said{0};
            if (said.fetch_add(1) < 8) {
                host_log("video: %zu flip(s) still queued when the cap lifted completed before flip %lld (args %lld..%lld)",
                         p->flip_queue.size(), static_cast<long long>(arg), static_cast<long long>(p->flip_queue.front().arg),
                         static_cast<long long>(p->flip_queue.back().arg));
            }
        }
        while (!p->flip_queue.empty()) {
            const VideoPort::FlipReq req = p->flip_queue.front();
            p->flip_queue.pop_front();
            finish_flip_locked(*p, req.buffer, req.arg);
        }
        finish_flip_locked(*p, buffer, arg);
        if (buffer >= 0 && buffer < 16) display_va = p->buffers[buffer];
        count = p->flip_count;
        g_flips_done.store(count);
        dw = p->display_w;
        dh = p->display_h;
    }
    // BBHOST_UNCAP's flips (benchmarks) are all shown; a loading screen's when
    // due (loading_show_due). `shown` true: present_flip leaves the window be.
    const bool unshown = !g_flip_uncapped.load() && !loading_show_due(flip.arrived_ns);
    present_flip(buffer, display_va, dw, dh, count, mark, unshown, flip);
}

void hle_video_detach_equeue(HostEqueue* eq) {
    std::lock_guard<std::mutex> lock(g_vo_mu);
    for (auto& kv : g_vo) {
        if (kv.second.flip_eq == eq) {
            kv.second.flip_eq = nullptr;
        }
    }
}

void hle_register_video() {
#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
    REG("sceVideoOutOpen", hle_video_open);
    REG("sceVideoOutClose", hle_video_close);
    REG("sceVideoOutSetFlipRate", hle_video_flip_rate);
    REG("sceVideoOutAddFlipEvent", hle_video_add_flip);
    REG("sceVideoOutSetBufferAttribute", hle_video_set_attr);
    REG("sceVideoOutRegisterBuffers", hle_video_register);
    REG("sceVideoOutSubmitFlip", hle_video_flip);
    REG("sceVideoOutGetFlipStatus", hle_video_status);
#undef REG
}
