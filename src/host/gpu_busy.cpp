// The GPU busy meter: how much of the wall clock our work kept the GPU busy,
// read off the GPU's own clock, for two timestamps a command buffer.
//
// Why it is ours to measure. Windows' Task Manager takes a process's GPU %
// from the per-process engine counters (`\GPU Engine(pid_*)\Running Time`),
// which the kernel fills from what the display driver reports. On the Radeon
// 8060S with AMD's driver 32.0.31021.1015 and hardware-accelerated GPU
// scheduling on (dxdiag: "Hardware Scheduling: ... Enabled:True") the 3D
// engine's time is not reported for anyone: read 2026-10-08 after a day's
// uptime, dwm.exe - which composes the desktop with the 3D engine - had 0 of
// it, and no process more than 0.4 ms, while the copy and video engines were
// counted (dwm's copy 0.7 s, Firefox's video decode 137 s). bbhost's perf
// passes read 0 on every engine with its GPU memory counted (2.5 GiB
// dedicated). So Task Manager cannot show this machine's 3D work, ours or
// anyone's. The lines below are the number to read instead.
//
// What it counts. A TOP_OF_PIPE timestamp as each command buffer starts and a
// BOTTOM_OF_PIPE one after its last command - the game's (begin_recording_locked
// / submit_locked) and the presenter's blit (window.cpp) - read once its fence
// has signalled. The spans overlap: the command processor starts the next
// command buffer while the last one's draws still run, and the presenter's
// blit is queued behind the game's work on the same queue. So busy time is
// the union of the spans on the GPU's timeline, not their sum: a span counts
// only past the latest end counted before it. A span includes the bubbles
// inside its command buffer (barriers draining, a pass waiting on the last);
// the GPU is ours there too, as Windows would count it. Not counted: the
// buffer shadow's upload command buffer that goes in the same submission
// ahead of the game's (off on integrated GPUs) - counted since an RTX 5080's
// log read 64% here while Task Manager's 3D engine sat at 100%: its per-draw
// vertex copies, 9-37 MiB a frame from host memory, ran there - and the
// start-up one-shots. The uploads go in the same submission just ahead of the
// game's command buffer, so its span starts at theirs (a third timestamp,
// written by the uploads).
// The presenter's blit waits for the swapchain image's semaphore; AMD's,
// NVIDIA's and Mesa's drivers wait for a submission's semaphores before its
// command buffers start, so its span does not hold the display's wait - and
// were one to, the presenter's own figure in the 300-flip line would show it
// (near 1000 ms/s at V-Sync).
//
// Order. Spans arrive late and from two places: a game slot is read when the
// command processor reuses it, up to 16 submissions after it went in; the
// presenter's when it waits for its fence before recording the next. A span
// can be counted only once nothing earlier can still arrive, or a late one
// would find its time already passed over. Each source's spans come in
// submission order, which on one queue is start order, so a source with work
// out can only add spans starting at or after the last start it delivered;
// pending spans up to the least of those (over the sources with work out)
// are final. A source with nothing out holds nothing back (headless, no
// presenter). Past kMaxPending held spans - a presenter stuck for a second -
// the oldest are counted anyway (`merged early` in the report).
//
// Timestamps are timestampValidBits wide and wrap at that width (64 on AMD
// and NVIDIA; 36 on some Intel parts): values are unwrapped against the
// latest one seen, so a span may straddle a wrap. The pair is reset in the
// command buffer itself, a two-query vkCmdResetQueryPool before its first
// timestamp (the profiler resets 8,192 there): no device feature to enable,
// and nothing on the host between the read and the next use.
//
// Read in: the `frames:` line (BBHOST_FRAME_STATS: `; gpu busy N ms/s (P%)`),
// the FPS counter (host setting fps_counter: `GPU P%`), and a `gpu busy:`
// line in the 300-flip report. BBHOST_GPU_BUSY=0 turns all of it off.
#include "host/gpu.h"
#include "host/gpu_internal.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace gpu {

namespace {

constexpr int kGame = 0, kPresenter = 1, kSources = 2;
// Spans held for order before the oldest are counted regardless: about a
// second of command buffers at 60 frames a second.
constexpr std::size_t kMaxPending = 1024;

struct Span {
    std::uint64_t t0 = 0, t1 = 0;  // unwrapped ticks
    int source = kGame;
};

struct Meter {
    std::mutex mu;
    bool on = false;
    double tick_ns = 1.0;
    std::uint64_t mask = ~0ull;  // timestampValidBits
    // Unwrapping: the latest raw value seen and what it unwrapped to.
    bool have_ref = false;
    std::uint64_t ref_raw = 0, ref_ext = 0;
    // Ordering: command buffers submitted and not yet read, by source, and
    // the latest start each source delivered.
    int out[kSources] = {};
    bool delivered[kSources] = {};
    std::uint64_t last_start[kSources] = {};
    std::vector<Span> pending;  // by start
    // The union so far: where it ends, and its length; by source, the same.
    std::uint64_t union_end = 0, busy = 0;
    std::uint64_t src_end[kSources] = {}, src_busy[kSources] = {}, spans[kSources] = {};
    std::uint64_t unreadable = 0, early = 0;
    std::chrono::steady_clock::time_point started{};  // when it was turned on
};
Meter g_meter;

// The presenter's pair: it has one command buffer, and waits for its fence
// before recording it again.
VkQueryPool g_present_pool = VK_NULL_HANDLE;
std::atomic<bool> g_present_written{false};

std::uint64_t unwrap_locked(Meter& m, std::uint64_t raw) {
    raw &= m.mask;
    if (m.mask == ~0ull) return raw;
    if (!m.have_ref) {
        // One wrap of headroom below the first value, for a span read later
        // that started before it.
        m.have_ref = true;
        m.ref_raw = raw;
        m.ref_ext = raw + m.mask + 1;
        return m.ref_ext;
    }
    const std::uint64_t ahead = (raw - m.ref_raw) & m.mask;
    if (ahead <= (m.mask >> 1)) {
        m.ref_raw = raw;
        m.ref_ext += ahead;
        return m.ref_ext;
    }
    return m.ref_ext - ((m.ref_raw - raw) & m.mask);
}

void count_locked(Meter& m, const Span& s) {
    if (s.t1 > m.union_end) {
        m.busy += s.t1 - std::max(s.t0, m.union_end);
        m.union_end = s.t1;
    }
    std::uint64_t& end = m.src_end[s.source];
    if (s.t1 > end) {
        m.src_busy[s.source] += s.t1 - std::max(s.t0, end);
        end = s.t1;
    }
    ++m.spans[s.source];
}

// Counts the pending spans no later delivery can precede.
void drain_locked(Meter& m) {
    std::uint64_t final_up_to = ~0ull;
    for (int s = 0; s < kSources; ++s) {
        if (m.out[s] > 0) final_up_to = std::min(final_up_to, m.delivered[s] ? m.last_start[s] : 0);
    }
    std::size_t i = 0;
    while (i < m.pending.size() && (m.pending[i].t0 <= final_up_to || m.pending.size() - i > kMaxPending)) {
        if (m.pending[i].t0 > final_up_to) ++m.early;
        count_locked(m, m.pending[i++]);
    }
    if (i) m.pending.erase(m.pending.begin(), m.pending.begin() + static_cast<std::ptrdiff_t>(i));
}

// A command buffer of `source` is done: its two raw timestamps, or `ok`
// false when they could not be read (a lost device).
void deliver(int source, std::uint64_t raw0, std::uint64_t raw1, bool ok) {
    Meter& m = g_meter;
    std::lock_guard<std::mutex> lk(m.mu);
    if (m.out[source] > 0) --m.out[source];
    if (ok) {
        Span s;
        s.source = source;
        s.t0 = unwrap_locked(m, raw0);
        s.t1 = s.t0 + ((raw1 - raw0) & m.mask);
        // Ten seconds in one command buffer would have been a device loss:
        // a pair that was never written.
        if (static_cast<double>(s.t1 - s.t0) * m.tick_ns > 10e9) {
            ++m.unreadable;
        } else {
            auto it = m.pending.end();
            while (it != m.pending.begin() && (it - 1)->t0 > s.t0) --it;
            m.pending.insert(it, s);
            m.delivered[source] = true;
            m.last_start[source] = std::max(m.last_start[source], s.t0);
        }
    } else {
        ++m.unreadable;
    }
    drain_locked(m);
}

VkQueryPool make_pair_pool() {
    VkQueryPoolCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = 3;  // the pair, and the game's slots' upload start
    VkQueryPool pool = VK_NULL_HANDLE;
    if (vkCreateQueryPool(g.device, &qci, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
    return pool;  // reset by each command buffer that writes it, before its first timestamp
}

// Two timestamps of a pool, if the GPU wrote both: the fence that covers
// them has signalled, so this does not wait (VK_NOT_READY otherwise).
bool read_pair(VkQueryPool pool, std::uint64_t (&ts)[2]) {
    return vkGetQueryPoolResults(g.device, pool, 0, 2, sizeof(ts), ts, sizeof(ts[0]), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
}

}  // namespace

void busy_init_locked() {
    g.busy = false;
    const char* e = std::getenv("BBHOST_GPU_BUSY");
    if (e && e[0] == '0') {
        host_log("gpu: busy meter off (BBHOST_GPU_BUSY=0)");
        return;
    }
    if (!host_tooling() && !(e && e[0] == '1')) {
        host_log("gpu: busy meter off (tooling off; BBHOST_GPU_BUSY=1 turns it on)");
        return;
    }
    std::uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qs(nq);
    if (nq) vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, qs.data());
    const std::uint32_t bits = g.family < nq ? qs[g.family].timestampValidBits : 0;
    if (bits == 0 || !(g.timestamp_period_ns > 0.0f)) {
        host_log("gpu: busy meter off: the queue family writes no timestamps");
        return;
    }
    bool made = true;
    for (int k = 0; k < g.slot_count && made; ++k) made = (g.slots[k].busy_pool = make_pair_pool()) != VK_NULL_HANDLE;
    if (made) made = (g_present_pool = make_pair_pool()) != VK_NULL_HANDLE;
    if (!made) {
        for (int k = 0; k < g.slot_count; ++k) {
            if (g.slots[k].busy_pool) vkDestroyQueryPool(g.device, g.slots[k].busy_pool, nullptr);
            g.slots[k].busy_pool = VK_NULL_HANDLE;
        }
        host_log("gpu: busy meter off: no timestamp query pool");
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_meter.mu);
        g_meter.mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
        g_meter.tick_ns = g.timestamp_period_ns;
        g_meter.started = std::chrono::steady_clock::now();
        g_meter.on = true;
    }
    g.busy = true;
    host_log("gpu: busy meter on: a timestamp at each command buffer's start and end, the game's and the presenter's "
             "(%u-bit, %.2f ns a tick); `gpu busy` in the frame statistics, the FPS counter and the 300-flip report; "
             "BBHOST_GPU_BUSY=0 turns it off",
             bits, static_cast<double>(g.timestamp_period_ns));
}

// With the command stream (recorder.cpp) these are ops the recorder replays
// right after its vkBeginCommandBuffer and before its vkEndCommandBuffer.
// Without it the callers have just taken g_cmd() (vkBeginCommandBuffer, the
// barrier before vkEndCommandBuffer), so the recorder holds nothing of this
// command buffer: g.cmd_ is it, and these two do not show up as recording
// sites in BBHOST_GPU_PROFILE=2's gaps.
void busy_begin_locked(Gpu::Slot& sl) {
    // A slot is retired before it is recorded again, which reads its pair; one
    // that was not would hold the game's spans back for good (`out`), so it
    // goes as unreadable.
    if (sl.busy_written) deliver(kGame, 0, 0, false);
    sl.busy_written = false;
    sl.busy_upload = false;
    if (!g.busy || !sl.busy_pool) return;
    // Outside any render pass: nothing is recorded yet.
    if (stream_on()) {
        rec().reset_query_pool(sl.busy_pool, 0, 2);
        rec().write_timestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sl.busy_pool, 0);
        return;
    }
    vkCmdResetQueryPool(g.cmd_, sl.busy_pool, 0, 2);
    vkCmdWriteTimestamp(g.cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sl.busy_pool, 0);
}

void busy_end_locked(Gpu::Slot& sl) {
    if (!g.busy || !sl.busy_pool) return;
    if (stream_on()) {
        rec().write_timestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sl.busy_pool, 1);
    } else {
        vkCmdWriteTimestamp(g.cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sl.busy_pool, 1);
    }
    sl.busy_written = true;
}

// The buffer shadow's upload command buffer as it begins (buffer_shadow.cpp,
// upload_cmd_locked): its start is the span's start when the slot retires.
void busy_upload_begin_locked(Gpu::Slot& sl, VkCommandBuffer cmd) {
    if (!g.busy || !sl.busy_pool || !cmd) return;
    vkCmdResetQueryPool(cmd, sl.busy_pool, 2, 1);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sl.busy_pool, 2);
    sl.busy_upload = true;
}

void busy_submitted_locked(Gpu::Slot& sl) {
    if (!sl.busy_written) return;
    std::lock_guard<std::mutex> lk(g_meter.mu);
    ++g_meter.out[kGame];
}

std::string busy_exit_report() {
    const GpuBusy b = host_gpu_busy();
    if (!b.on) return {};
    double secs = 0.0;
    {
        std::lock_guard<std::mutex> lk(g_meter.mu);
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_meter.started).count();
    }
    if (secs <= 0.0) return {};
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "gpu: busy over the run: %.1f%% of %.0f s (%.0f ms/s; the game's command buffers %.0f s, the presenter's %.1f s; "
                  "%llu command buffers, %llu presents)",
                  static_cast<double>(b.busy_ns) / 1e7 / secs, secs, static_cast<double>(b.busy_ns) / 1e6 / secs,
                  static_cast<double>(b.game_ns) / 1e9, static_cast<double>(b.present_ns) / 1e9,
                  static_cast<unsigned long long>(b.cmdbufs), static_cast<unsigned long long>(b.presents));
    return buf;
}

void busy_retired_locked(Gpu::Slot& sl, bool ok) {
    if (!sl.busy_written) return;
    sl.busy_written = false;
    std::uint64_t ts[2] = {0, 0};
    const bool have = ok && read_pair(sl.busy_pool, ts);
    if (have && sl.busy_upload) {
        std::uint64_t up = 0;
        if (vkGetQueryPoolResults(g.device, sl.busy_pool, 2, 1, sizeof(up), &up, sizeof(up), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS &&
            ((up - ts[0]) & g_meter.mask) > (g_meter.mask >> 1)) {  // earlier than the game's start, wrap-aware
            ts[0] = up;
        }
    }
    sl.busy_upload = false;
    deliver(kGame, ts[0], ts[1], have);
}

}  // namespace gpu

using namespace gpu;

void host_gpu_busy_present_begin(void* cmd) {
    if (!g.busy || !g_present_pool) return;
    const auto c = static_cast<VkCommandBuffer>(cmd);
    vkCmdResetQueryPool(c, g_present_pool, 0, 2);
    vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_present_pool, 0);
}

void host_gpu_busy_present_end(void* cmd) {
    if (!g.busy || !g_present_pool) return;
    vkCmdWriteTimestamp(static_cast<VkCommandBuffer>(cmd), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_present_pool, 1);
    g_present_written.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_meter.mu);
    ++g_meter.out[kPresenter];
}

void host_gpu_busy_present_done() {
    if (!g_present_written.exchange(false, std::memory_order_relaxed)) return;
    std::uint64_t ts[2] = {0, 0};
    const bool have = read_pair(g_present_pool, ts);
    deliver(kPresenter, ts[0], ts[1], have);
}

GpuBusy host_gpu_busy() {
    GpuBusy b;
    std::lock_guard<std::mutex> lk(g_meter.mu);
    if (!g_meter.on) return b;
    const double k = g_meter.tick_ns;
    b.on = true;
    b.busy_ns = static_cast<std::uint64_t>(static_cast<double>(g_meter.busy) * k);
    b.game_ns = static_cast<std::uint64_t>(static_cast<double>(g_meter.src_busy[kGame]) * k);
    b.present_ns = static_cast<std::uint64_t>(static_cast<double>(g_meter.src_busy[kPresenter]) * k);
    b.cmdbufs = g_meter.spans[kGame];
    b.presents = g_meter.spans[kPresenter];
    return b;
}

std::string host_gpu_busy_report() {
    // The first call (the first flip's report) only sets the window's start.
    static bool started = false;
    static GpuBusy last;
    static std::chrono::steady_clock::time_point last_t;
    const GpuBusy now = host_gpu_busy();
    const auto t = std::chrono::steady_clock::now();
    const double secs = started ? std::chrono::duration<double>(t - last_t).count() : 0.0;
    started = true;
    std::uint64_t unreadable = 0, early = 0;
    {
        std::lock_guard<std::mutex> lk(g_meter.mu);
        unreadable = g_meter.unreadable;
        early = g_meter.early;
    }
    std::string r;
    if (now.on && secs >= 0.5) {  // not the first flips' reports, a few ms apart
        const double ms = static_cast<double>(now.busy_ns - last.busy_ns) / 1e6 / secs;
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "gpu busy: %.1f%% of %.1f s (%.0f ms/s; the game's command buffers %.0f ms/s, the presenter's %.1f ms/s; "
                      "%llu command buffers, %llu presents; unreadable %llu, merged early %llu)",
                      ms / 10.0, secs, ms, static_cast<double>(now.game_ns - last.game_ns) / 1e6 / secs,
                      static_cast<double>(now.present_ns - last.present_ns) / 1e6 / secs,
                      static_cast<unsigned long long>(now.cmdbufs - last.cmdbufs),
                      static_cast<unsigned long long>(now.presents - last.presents), static_cast<unsigned long long>(unreadable),
                      static_cast<unsigned long long>(early));
        r = buf;
    }
    last = now;
    last_t = t;
    return r;
}
