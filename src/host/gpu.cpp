#include "core/config.h"
#include "core/host_clock.h"
#include "core/paced_file.h"
#include "core/portable.h"
#include "engine/gx_resources.h"
#include "engine/live_resolution.h"
#include "host/apu_memory.h"
#include "host/heap_compact.h"
#include "host/heap_place.h"
#include "host/list_resolve.h"
#include "host/foreign_hooks.h"
#include "host/draw_capture.h"
#include "host/gpu_internal.h"
#include "host/gxvk.h"
#include "host/import_mode.h"
#include "host/import_prefetch.h"
#include "host/host_lists.h"
#include "host/label_landing.h"
#include "host/ring_budget.h"
#include "host/shader_patch.h"
#include "host/shaders/gpu_stall.spv.h"
#include "host/shader_bundles.h"
#include "host/translation_cache.h"
#include "host/twin_filter.h"
#include "host/options.h"

#include "gcn/container.h"
#include "gcn/half.h"
#include "gcn/isa.h"
#include "hle/fs.h"
#include "hle/modules.h"
#include "log.h"

#include <algorithm>
#include <sys/stat.h>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <memory>
#include <optional>
#include <iterator>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <condition_variable>
#include <deque>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace gpu {

Gpu g;
std::atomic<std::uint64_t> g_validation_messages{0};  // warnings and errors the validation layer reported
thread_local constinit const char* t_import_site = nullptr;     // gpu_internal.h: the window imports' census

void device_lost_locked(const char* where, bool declared = false);  // below, with the reports it makes
bool take_marked_start(std::string& why);    // the same: a start after a lost device runs with markers

namespace {

const bool g_enabled = [] {
    const char* e = std::getenv("BBHOST_GPU");
    return !(e && e[0] == '0');
}();
const bool g_validate = [] {
    const char* e = std::getenv("BBHOST_VK_VALIDATE");
    return e && e[0] == '1';
}();
// BBHOST_RENDER_MIN=1 (recorder.cpp has the draws): no dispatches either.
const bool g_render_min = [] {
    const char* e = std::getenv("BBHOST_RENDER_MIN");
    return e && e[0] == '1';
}();

std::vector<std::string> g_instance_exts;
bool g_want_present = false;

// The validation census (BBHOST_VK_VALIDATE=1): every message id counted, with
// its first message, when it came first and last, and what it was about - the
// names of the objects it named (a draw's pipeline is named by its programs,
// vk_debug_name) and what the thread was building (ValidationScope) - for the
// exit report's lines (host_gpu_report). The layer's own limit of ten messages
// an id hides the counts: run with VK_KHRONOS_VALIDATION_ENABLE_MESSAGE_LIMIT=0.
struct ValidationId {
    std::uint64_t count = 0;
    double first_s = 0, last_s = 0;
    std::uint32_t severities = 0, types = 0;
    std::string first;                            // the first message
    std::map<std::string, std::uint64_t> about;   // the first 24 subjects, each counted
    std::uint64_t about_other = 0;                // messages about a subject past those
    std::unordered_set<std::string> subjects;     // every subject, up to 4096, for the count of them
};
std::mutex g_validation_mu;
std::map<std::string, ValidationId> g_validation_ids;  // under g_validation_mu
const auto g_validation_t0 = std::chrono::steady_clock::now();
PFN_vkSetDebugUtilsObjectNameEXT g_set_object_name = nullptr;  // with the layer

const char* vk_object_type_name(VkObjectType t) {
    switch (t) {
    case VK_OBJECT_TYPE_PIPELINE: return "pipeline";
    case VK_OBJECT_TYPE_SHADER_MODULE: return "module";
    case VK_OBJECT_TYPE_COMMAND_BUFFER: return "cmd";
    case VK_OBJECT_TYPE_IMAGE: return "image";
    case VK_OBJECT_TYPE_IMAGE_VIEW: return "view";
    case VK_OBJECT_TYPE_BUFFER: return "buffer";
    case VK_OBJECT_TYPE_DESCRIPTOR_SET: return "set";
    case VK_OBJECT_TYPE_PIPELINE_LAYOUT: return "layout";
    case VK_OBJECT_TYPE_DEVICE_MEMORY: return "memory";
    case VK_OBJECT_TYPE_SAMPLER: return "sampler";
    case VK_OBJECT_TYPE_QUEUE: return "queue";
    case VK_OBJECT_TYPE_DEVICE: return "device";
    default: return "object";
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT types, const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void*) {
    if (severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        return VK_FALSE;
    }
    g_validation_messages.fetch_add(1, std::memory_order_relaxed);
    std::string id = data && data->pMessageIdName ? data->pMessageIdName : "";
    if (id.empty()) id = "id " + std::to_string(data ? data->messageIdNumber : 0);
    // Its subject: the named objects (pipelines, by their programs), else what
    // the thread was building.
    std::string subject, objects;
    for (std::uint32_t i = 0; data && i < data->objectCount; ++i) {
        const VkDebugUtilsObjectNameInfoEXT& o = data->pObjects[i];
        char b[64];
        std::snprintf(b, sizeof(b), "%s%s 0x%llx", objects.empty() ? "" : ", ", vk_object_type_name(o.objectType),
                      static_cast<unsigned long long>(o.objectHandle));
        objects += b;
        if (o.pObjectName && *o.pObjectName) {
            objects += std::string(" \"") + o.pObjectName + "\"";
            if (o.objectType == VK_OBJECT_TYPE_PIPELINE || o.objectType == VK_OBJECT_TYPE_SHADER_MODULE || subject.empty()) {
                subject += (subject.empty() ? "" : " ") + std::string(o.pObjectName);
            }
        }
    }
    if (subject.empty() && t_validation_scope) subject = std::string("building ") + t_validation_scope;
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_validation_t0).count();
    // Each message id twice, up to 200 ids: a run's first 40 messages were
    // often one id's repeats, which hid every later kind.
    bool log = false;
    {
        std::lock_guard<std::mutex> lk(g_validation_mu);
        auto it = g_validation_ids.find(id);
        if (it == g_validation_ids.end()) {
            if (g_validation_ids.size() >= 2000) return VK_FALSE;  // counted in g_validation_messages alone
            it = g_validation_ids.emplace(id, ValidationId{}).first;
            it->second.first_s = now;
            it->second.first = data && data->pMessage ? data->pMessage : "?";
        }
        ValidationId& v = it->second;
        log = v.count < 2 && g_validation_ids.size() <= 200;
        ++v.count;
        v.last_s = now;
        v.severities |= static_cast<std::uint32_t>(severity);
        v.types |= types;
        if (!subject.empty()) {
            if (auto a = v.about.find(subject); a != v.about.end()) ++a->second;
            else if (v.about.size() < 24) v.about.emplace(subject, 1);
            else ++v.about_other;
            if (v.subjects.size() < 4096) v.subjects.insert(subject);
        }
    }
    if (log) {
        // The objects and the scope first: host_log keeps 1 KiB of a line.
        host_log("vk validation: [%s%s%s] %s", objects.c_str(), t_validation_scope ? "; building " : "", t_validation_scope ? t_validation_scope : "",
                 data && data->pMessage ? data->pMessage : "?");
    }
    return VK_FALSE;
}

// Window imports: direct memory imported a window at a time, on demand -
// BBHOST_IMPORT_WINDOWS=1, or gpu.import_windows = true in the config (the
// Steam Deck kit's default: the host heap under its budget, the start ~3 s
// sooner, the same frames and frame rate), and by default on an integrated
// GPU under Windows, where the whole spans kept ~1.5 GB more of the game's
// memory resident for the GPU, all of it counted as the process's GPU memory
// (a Radeon 680M at a lamp, with the told meshes read in place from their
// windows: Task Manager's shared GPU memory 8.0 -> 6.2 GB, the working set
// 12.7 -> 11.1 GB, the frame rate and the frames the same;
// imports_whole_decide). Not yet under Linux, where an integrated GPU imports
// through udmabuf dma-bufs with tight memory, unmeasured with windows there.
// Off by default on a discrete GPU: a tenth
// of the draws keep page-table walks through V#s and pointers the host does
// not resolve (the exit report's "walk" lines), and what they reach must
// already be imported or they read the sink. Off, each chunk's GPU-visible
// span is imported whole when the mappings are made. Decided once, at the
// GPU's start (after the config is read) or an earlier first ask; the
// environment wins over the config. imports_whole() (gpu_internal.h) reads it.
std::atomic<std::uint64_t> g_imports_made{0}, g_imports_merged{0}, g_import_bytes{0}, g_import_us{0};

// ---- The submission thread (BBHOST_SUBMIT_THREAD, on unless 0) ----
// vkQueueSubmit off the command processor. The kernel's half of a submission
// grows with the buffers it has to make resident - all of them, since shaders
// reach memory by address - and on a Steam Deck (RADV, ~60 imported dma-bufs
// of direct memory, 11-12 submissions a frame) it was a third of the command
// processor's time: the game's own thread waiting in an ioctl. The command
// processor now hands the finished command buffer over and records on; this
// thread submits in order. Anyone else who uses the queue goes through
// QueueGuard. A failed submission is passed back (g_sub_failed) and handled on
// the command processor as before: no more GPU execution.
struct SubmitJob {
    VkCommandBuffer cmds[2] = {};
    std::uint32_t ncmds = 0;
    VkSemaphore wait = VK_NULL_HANDLE;
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSemaphore signal = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::uint32_t items = 0;
    std::uint64_t ticket = 0;
    std::uint64_t serial = ~0ull;  // a game submission's Gpu::flushes, published once it went (stream_submitted_serial)
    std::uint64_t foreign = 0;     // a kForeignSubmit's id, published the same way (host_gpu_wait_submitted)
    std::uint64_t enqueued_ns = 0;  // BBHOST_TEST_GPU_DELAY_MS: when it was handed over
};
std::mutex g_sub_mu;
std::condition_variable g_sub_cv, g_sub_idle;
std::deque<SubmitJob> g_sub_jobs;
bool g_sub_busy = false;  // the thread holds a job it took (under g_sub_mu)
bool g_sub_running = false;
std::uint64_t g_sub_enqueued = 0;               // tickets handed out (under g_sub_mu)
std::atomic<std::uint64_t> g_sub_submitted{0};  // the last ticket submitted
std::atomic<int> g_sub_failed{0};  // the VkResult of a failed submission
std::atomic<std::uint32_t> g_sub_failed_items{0};
const bool g_submit_thread_on = [] {
    const char* e = std::getenv("BBHOST_SUBMIT_THREAD");
    return !(e && e[0] == '0');
}();

// The game's submissions gone to the queue (the last one's serial + 1), and
// the last kForeignSubmit id: published after vkQueueSubmit returned.
std::atomic<std::uint64_t> g_stream_submitted{0};
std::atomic<std::uint64_t> g_foreign_submitted{0};
std::atomic<bool> g_timeline_on{false};  // Gpu::timeline made: set once, before any submission
// BBHOST_LABEL_LANDING (on by default but on Windows), with a timeline
// semaphore to say when a submission has run: the command processor's labels
// are written by the host then (label_landing.h); without one, or with
// BBHOST_LABEL_LANDING=0, they stay GPU fills.
bool landing_on() { return label_landing::on() && g_timeline_on.load(std::memory_order_relaxed); }
void start_label_landing();  // once the timeline is made (init_locked)
constexpr std::uint64_t kForeignTicket = 1ull << 63;  // host_gpu_submit_presenter's tickets for a kForeignSubmit

// A game submission (`serial` its Gpu::flushes) also signals the timeline
// with serial + 1: added to `si`'s signals, the binary ones it has keeping
// their places (their values are ignored), held in this until the submit.
struct TimelineSignal {
    VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    VkSemaphore sems[2] = {};
    std::uint64_t values[2] = {}, wait_values[2] = {};
    void add(VkSubmitInfo& si, std::uint64_t serial) {
        if (!g.timeline || serial == ~0ull || si.signalSemaphoreCount > 1 || si.waitSemaphoreCount > 2) return;
        const std::uint32_t n = si.signalSemaphoreCount;
        for (std::uint32_t i = 0; i < n; ++i) sems[i] = si.pSignalSemaphores[i];
        sems[n] = g.timeline;
        values[n] = serial + 1;
        si.signalSemaphoreCount = n + 1;
        si.pSignalSemaphores = sems;
        ti.signalSemaphoreValueCount = n + 1;
        ti.pSignalSemaphoreValues = values;
        ti.waitSemaphoreValueCount = si.waitSemaphoreCount;
        ti.pWaitSemaphoreValues = si.waitSemaphoreCount ? wait_values : nullptr;
        ti.pNext = si.pNext;
        si.pNext = &ti;
    }
};

// One job to the queue, under its lock; a failure is kept for the command
// processor (take_submit_failure_locked).
void submit_job_now(const SubmitJob& job) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = job.ncmds;
    si.pCommandBuffers = job.cmds;
    if (job.wait) {
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &job.wait;
        si.pWaitDstStageMask = &job.wait_stage;
    }
    if (job.signal) {
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &job.signal;
    }
    TimelineSignal tl;
    tl.add(si, job.serial);
    VkResult r;
    {
        std::lock_guard<std::mutex> q(g.queue_mu);
        r = vkQueueSubmit(g.queue, 1, &si, job.fence);
    }
    if (r != VK_SUCCESS) {
        int none = 0;
        g_sub_failed.compare_exchange_strong(none, static_cast<int>(r));
        g_sub_failed_items.fetch_add(job.items);
    }
    if (job.serial != ~0ull) g_stream_submitted.store(job.serial + 1, std::memory_order_release);
    if (job.foreign) g_foreign_submitted.store(job.foreign, std::memory_order_release);
}

// Game submissions queued behind each other go to the queue as one
// (BBHOST_SUBMIT_MERGE=<n>, at most n of them, 8; 1 or 0: one by one). The
// kernel's part of a submission is mostly the buffers it makes resident -
// every one, since shaders reach memory by address - so on RADV each
// submission pays for all ~70 dma-buf imports of direct memory again: a Steam
// Deck's bb-submit was a whole core at the menus, 5 submissions a flip, and
// the menus waited for it. Jobs queue up here only while the thread is behind,
// so where submitting is cheap (NVIDIA) nothing changes. A merged submission
// holds the jobs' command buffers in order (each job's shadow uploads before
// its own commands, as before) and signals the timeline with the last one's
// value; the earlier jobs' fences are signalled by fence-only submissions
// after it (no buffers, so no residency), when everything before them has
// run - their own work included. A job that waits on a semaphore (a sparse
// bind) starts a submission; one that signals one (the presenter's) goes on
// its own.
const std::uint32_t g_submit_merge = [] {
    const char* e = std::getenv("BBHOST_SUBMIT_MERGE");
    const long n = e && *e ? std::strtol(e, nullptr, 10) : 8;
    return static_cast<std::uint32_t>(std::clamp<long>(n, 1, 16));
}();
// BBHOST_TEST_SUBMIT_HOLD_US=<us> (tests): the thread waits this long before it
// takes the jobs queued, so they queue up and the merged path runs where the
// queue is otherwise kept empty.
const std::uint32_t g_submit_hold_us = [] {
    const char* e = std::getenv("BBHOST_TEST_SUBMIT_HOLD_US");
    return e && *e ? static_cast<std::uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
}();
// BBHOST_TEST_GPU_DELAY_MS=<ms> (tests): each submission goes to the queue no
// sooner than this long after it was handed over - a GPU that runs everything
// as fast as before but that much later. Its labels land that much later with
// it, and so does all that waits on them (a flip, a wait for the GPU, a
// chunk's retirement), whether the host lands them or the GPU fills them: a
// slow device's timing - the command processor behind, labels landing tens of
// milliseconds after their submission - at will (the counts of writes into
// memory written again since, label_landing.h).
const std::uint64_t g_test_gpu_delay_ns = [] {
    const char* e = std::getenv("BBHOST_TEST_GPU_DELAY_MS");
    return e && *e ? static_cast<std::uint64_t>(std::strtoul(e, nullptr, 10)) * 1000000ull : 0ull;
}();
// Written by the submission thread only (relaxed stores), read by the report.
std::atomic<std::uint64_t> g_sub_jobs_done{0}, g_sub_queue_submits{0}, g_sub_jobs_merged{0}, g_sub_fence_submits{0};

bool merge_candidate(const SubmitJob& j) { return j.serial != ~0ull && !j.foreign && !j.signal && j.ncmds <= 2; }

void add_relaxed_counter(std::atomic<std::uint64_t>& c, std::uint64_t n = 1) { c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed); }

// Jobs [0, n) of `jobs` (n >= 2, every one a merge candidate, only the first
// waiting on a semaphore) as one queue submission.
void submit_jobs_merged(const SubmitJob* jobs, std::uint32_t n) {
    VkCommandBuffer cmds[2 * 16];
    std::uint32_t ncmds = 0, items = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t k = 0; k < jobs[i].ncmds; ++k) cmds[ncmds++] = jobs[i].cmds[k];
        items += jobs[i].items;
    }
    const SubmitJob& first = jobs[0];
    const SubmitJob& last = jobs[n - 1];
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = ncmds;
    si.pCommandBuffers = cmds;
    if (first.wait) {
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &first.wait;
        si.pWaitDstStageMask = &first.wait_stage;
    }
    TimelineSignal tl;
    tl.add(si, last.serial);
    VkResult r;
    std::uint32_t fences = 0;
    {
        std::lock_guard<std::mutex> q(g.queue_mu);
        r = vkQueueSubmit(g.queue, 1, &si, last.fence);
        for (std::uint32_t i = 0; i + 1 < n && r == VK_SUCCESS; ++i) {
            if (!jobs[i].fence) continue;
            r = vkQueueSubmit(g.queue, 0, nullptr, jobs[i].fence);
            ++fences;
        }
    }
    if (r != VK_SUCCESS) {
        int none = 0;
        g_sub_failed.compare_exchange_strong(none, static_cast<int>(r));
        g_sub_failed_items.fetch_add(items);
    }
    add_relaxed_counter(g_sub_jobs_merged, n - 1);
    add_relaxed_counter(g_sub_fence_submits, fences);
    // Published once every fence of the batch has gone to the queue: a waiter
    // that sees its serial submitted may wait on its fence (which must not be
    // used while vkQueueSubmit holds it).
    g_stream_submitted.store(last.serial + 1, std::memory_order_release);
}

void submit_thread() {
    host_thread_set_name("bb-submit");
    host_thread_set_class(HostThreadClass::GpuFeed, "bb-submit");  // core/host_clock.h
    SubmitJob jobs[16];
    for (;;) {
        std::uint32_t n = 0;
        {
            std::unique_lock<std::mutex> lk(g_sub_mu);
            g_sub_cv.wait(lk, [] { return !g_sub_jobs.empty(); });
            if (g_submit_hold_us) {
                lk.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(g_submit_hold_us));
                lk.lock();
            }
            jobs[n++] = g_sub_jobs.front();
            g_sub_jobs.pop_front();
            if (merge_candidate(jobs[0])) {
                while (n < g_submit_merge && !g_sub_jobs.empty() && merge_candidate(g_sub_jobs.front()) && !g_sub_jobs.front().wait) {
                    jobs[n++] = g_sub_jobs.front();
                    g_sub_jobs.pop_front();
                }
            }
            g_sub_busy = true;
        }
        if (g_test_gpu_delay_ns && jobs[0].enqueued_ns) {
            const std::uint64_t due = jobs[0].enqueued_ns + g_test_gpu_delay_ns, now = host_clock_monotonic_ns();
            if (due > now) host_sleep_us((due - now) / 1000);
        }
        if (n == 1) {
            submit_job_now(jobs[0]);
        } else {
            submit_jobs_merged(jobs, n);
        }
        add_relaxed_counter(g_sub_jobs_done, n);
        add_relaxed_counter(g_sub_queue_submits);
        std::lock_guard<std::mutex> lk(g_sub_mu);
        g_sub_busy = false;
        g_sub_submitted.store(jobs[n - 1].ticket, std::memory_order_release);
        g_sub_idle.notify_all();  // a drain, or a wait for these tickets
    }
}

// Queues a job; its ticket, for host_gpu_wait_submitted.
std::uint64_t enqueue_submit(SubmitJob job) {
    std::uint64_t ticket;
    {
        std::lock_guard<std::mutex> lk(g_sub_mu);
        job.ticket = ticket = ++g_sub_enqueued;
        g_sub_jobs.push_back(job);
    }
    g_sub_cv.notify_one();
    return ticket;
}

// A failed submission seen by the thread, made the command processor's own:
// GPU execution stops, as when the submit failed in place. Under g.mu.
void take_submit_failure_locked() {
    const int r = g_sub_failed.load(std::memory_order_relaxed);
    if (!r || !g.ok) return;
    host_log("gpu: submit of %u items failed (%d)%s", g_sub_failed_items.load(), r,
             r == VK_ERROR_DEVICE_LOST ? "; device lost, GPU execution disabled" : "");
    g.failures.fetch_add(g_sub_failed_items.load());
    if (r == VK_ERROR_DEVICE_LOST) device_lost_locked("the submission thread's submit");
}

std::string submit_thread_report_impl() {
    if (!g_sub_running) return "submission thread off";
    std::size_t jobs = 0;
    bool busy = false;
    {
        std::lock_guard<std::mutex> lk(g_sub_mu);
        jobs = g_sub_jobs.size();
        busy = g_sub_busy;
    }
    const bool queue_free = g.queue_mu.try_lock();
    if (queue_free) g.queue_mu.unlock();
    char buf[160];
    std::snprintf(buf, sizeof(buf), "submission thread: %zu job(s) pending, %s, the queue's lock %s, failure %d", jobs,
                  busy ? "submitting one" : "idle", queue_free ? "free" : "held", g_sub_failed.load());
    return buf;
}

void start_submit_thread() {
    if (!g_submit_thread_on || g_sub_running) return;
    g_sub_running = true;
    std::thread(submit_thread).detach();
    host_log("gpu: submissions go to the queue from their own thread (BBHOST_SUBMIT_THREAD=0 submits in place)");
}

// What a direct-memory import's buffer is used for: bindings, copies, and
// device addresses for the page tables.
constexpr VkBufferUsageFlags kImportUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

void free_import(Chunk& c) {
    if (c.buffer) vkDestroyBuffer(g.device, c.buffer, nullptr);
    if (c.memory) vkFreeMemory(g.device, c.memory, nullptr);
    for (VkDeviceMemory m : c.pieces) vkFreeMemory(g.device, m, nullptr);
    c = Chunk{};
}

// [ptr, ptr + size) of the mirror into c as one host-pointer import.
VkResult host_span(Chunk& c, void* ptr, std::uint64_t size) {
    VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (const VkResult r = g.get_host_pointer_props(g.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &hp);
        r != VK_SUCCESS) {
        host_log("gpu: host pointer properties failed for %p: %d", ptr, r);
        return r;
    }
    const std::uint32_t type = find_memory_type(hp.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) {
        host_log("gpu: no host-coherent memory type for a host pointer (bits 0x%x)", hp.memoryTypeBits);
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &ext;
    bci.size = size;
    bci.usage = kImportUsage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (const VkResult r = vkCreateBuffer(g.device, &bci, nullptr, &c.buffer); r != VK_SUCCESS) return r;
    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = ptr;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.pNext = &imp;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &flags;
    mai.allocationSize = size;
    mai.memoryTypeIndex = type;
    MemOwner own_mem("guest memory imports");
    if (const VkResult r = vkAllocateMemory(g.device, &mai, nullptr, &c.memory); r != VK_SUCCESS) return r;
    return vkBindBufferMemory(g.device, c.buffer, c.memory, 0);
}

#if !defined(_WIN32)
// Direct memory as dma-bufs. AMD's Linux kernel driver takes only anonymous
// memory as a host pointer - amdgpu makes every userptr ANONONLY, so a mapping
// with a file behind it is refused (-EPERM, which RADV reports as
// VK_ERROR_UNKNOWN) - and direct memory is a memfd: on a Steam Deck every
// import failed and only the clears drew. /dev/udmabuf makes a dma-buf of the
// memfd's pages instead, 64 MiB at most (the module's size_limit_mb), and a
// longer span is a sparse buffer with its pieces bound in order, so it keeps
// one range of device addresses.
// The pieces are as large as the module allows: every dma-buf is a buffer
// object the kernel makes resident at every submission (RADV attaches all of
// them, since shaders reach memory by address), ~60 of them for a Steam
// Deck's direct memory at the default 64 MiB. A system whose udmabuf takes
// more (size_limit_mb, a module parameter root can raise) gets fewer, up to
// 1 GiB each; BBHOST_DMABUF_PIECE_MB asks for less (tests).
std::uint64_t dmabuf_piece_bytes() {
    static const std::uint64_t bytes = [] {
        std::uint64_t mb = 64;
        if (FILE* f = std::fopen("/sys/module/udmabuf/parameters/size_limit_mb", "r")) {
            unsigned long v = 0;
            if (std::fscanf(f, "%lu", &v) == 1 && v >= 64) mb = std::min<unsigned long>(v, 1024);
            std::fclose(f);
        }
        if (const char* e = std::getenv("BBHOST_DMABUF_PIECE_MB"); e && *e) {
            const unsigned long v = std::strtoul(e, nullptr, 10);
            if (v >= 64) mb = std::min<std::uint64_t>(mb, v);
        }
        mb &= ~std::uint64_t{63};  // a whole number of 64 MiB: whole pages, and the sparse binding's alignment
        return mb << 20;
    }();
    return bytes;
}

// [offset, offset + size) of direct memory as one dma-buf import, of a memory
// type in `bits`. VK_NULL_HANDLE when it cannot be made.
VkDeviceMemory dmabuf_memory(std::uint64_t offset, std::uint64_t size, std::uint32_t bits) {
    udmabuf_create uc{};
    uc.memfd = static_cast<__u32>(hle_kernel_dmem_fd());
    uc.flags = UDMABUF_FLAGS_CLOEXEC;
    uc.offset = offset;
    uc.size = size;
    const int fd = ioctl(g.udmabuf, UDMABUF_CREATE, &uc);
    if (fd < 0) {
        host_log("gpu: a udmabuf of %llu MiB at +%llu MiB of direct memory failed: errno %d", static_cast<unsigned long long>(size >> 20),
                 static_cast<unsigned long long>(offset >> 20), errno);
        return VK_NULL_HANDLE;
    }
    VkMemoryFdPropertiesKHR fp{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    std::uint32_t type = UINT32_MAX;
    if (g.get_memory_fd_props(g.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp) == VK_SUCCESS) {
        type = find_memory_type(fp.memoryTypeBits & bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type == UINT32_MAX) type = find_memory_type(fp.memoryTypeBits & bits, 0);
    }
    if (type == UINT32_MAX) {
        host_log("gpu: no memory type for a dma-buf of direct memory (its 0x%x, the buffer's 0x%x)", fp.memoryTypeBits, bits);
        close(fd);
        return VK_NULL_HANDLE;
    }
    VkImportMemoryFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    imp.fd = fd;  // the driver's once the import is made
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.pNext = &imp;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &flags;
    mai.allocationSize = size;
    mai.memoryTypeIndex = type;
    VkDeviceMemory m = VK_NULL_HANDLE;
    MemOwner own_mem("guest memory imports");
    if (const VkResult r = vkAllocateMemory(g.device, &mai, nullptr, &m); r != VK_SUCCESS) {
        host_log("gpu: importing a dma-buf of %llu MiB at +%llu MiB of direct memory failed: %d", static_cast<unsigned long long>(size >> 20),
                 static_cast<unsigned long long>(offset >> 20), r);
        close(fd);
        return VK_NULL_HANDLE;
    }
    return m;
}

// [offset, offset + size) of direct memory into c as dma-buf imports: a
// buffer bound to one, or a sparse buffer over several. Under the GPU lock,
// as every use of the renderer's queue is: the sparse binds go on it.
VkResult dmabuf_span(Chunk& c, std::uint64_t offset, std::uint64_t size) {
    const std::uint64_t piece = dmabuf_piece_bytes();
    const bool sparse = size > piece;
    if (sparse && !g.dmabuf_sparse) {
        static std::atomic<int> said{0};
        if (said.fetch_add(1) < 4)
            host_log("gpu: %llu MiB of direct memory is more than a dma-buf holds, and the device has no sparse binding to join several",
                     static_cast<unsigned long long>(size >> 20));
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &ext;
    bci.flags = sparse ? VK_BUFFER_CREATE_SPARSE_BINDING_BIT : 0;
    bci.size = size;
    bci.usage = kImportUsage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (const VkResult r = vkCreateBuffer(g.device, &bci, nullptr, &c.buffer); r != VK_SUCCESS) return r;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, c.buffer, &req);
    // Spans are whole 64 KiB pages, so the pieces meet the binding alignment.
    if (req.size != size || (sparse && piece % req.alignment)) {
        host_log("gpu: a buffer over %llu bytes of direct memory wants %llu (alignment %llu)", static_cast<unsigned long long>(size),
                 static_cast<unsigned long long>(req.size), static_cast<unsigned long long>(req.alignment));
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::vector<VkSparseMemoryBind> binds;
    for (std::uint64_t at = 0; at < size; at += piece) {
        const std::uint64_t n = std::min(piece, size - at);
        const VkDeviceMemory m = dmabuf_memory(offset + at, n, req.memoryTypeBits);
        if (!m) return VK_ERROR_INVALID_EXTERNAL_HANDLE;
        c.pieces.push_back(m);
        binds.push_back({at, n, m, 0, 0});
    }
    if (!sparse) {
        c.memory = c.pieces[0];
        c.pieces.clear();
        return vkBindBufferMemory(g.device, c.buffer, c.memory, 0);
    }
    const VkSparseBufferMemoryBindInfo bbi{c.buffer, static_cast<std::uint32_t>(binds.size()), binds.data()};
    VkBindSparseInfo bsi{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
    bsi.bufferBindCount = 1;
    bsi.pBufferBinds = &bbi;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    if (const VkResult r = vkCreateFence(g.device, &fci, nullptr, &fence); r != VK_SUCCESS) return r;
    VkResult r;
    {
        QueueGuard queue;
        r = vkQueueBindSparse(g.queue, 1, &bsi, fence);
    }
    if (r == VK_SUCCESS) r = vkWaitForFences(g.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(g.device, fence, nullptr);
    return r;
}
#endif

// Which path asked for each import (the exit report's import line): a
// binding or copy resolved through locate(), a device address (the GPU
// untile, occlusion dumps), what a shader walks (import_windows: dispatches'
// reads and stores, the LS pass, unbound constant buffers), the prefetch of a
// GX resource's memory (bb-import), the whole spans.
enum ImportWho { kIwBinding, kIwAddress, kIwWalk, kIwPrefetch, kIwSpan, kImportWhos };
thread_local int t_import_who = kIwBinding;
// The guest address whose locate(), device address, walk or prefetch asked
// for the import under way (the census of what still needs window imports:
// by the path that asked - t_import_site, gpu_internal.h - and the GX
// resource there: no GX object, a buffer by its usage and bind flags, a
// texture - counted when a window import is made).
thread_local std::uint64_t t_import_va = 0;
// The census of guest memory imported, read and written in place (the exit
// report's window imports by the path that asked and what lies there, the
// in-place users by path, the GX buffers GPU copies write):
// BBHOST_IMPORT_CENSUS=1. Off - the default - nothing is counted, so the
// default path does no work for it at any access.
const bool g_census = [] {
    const char* e = std::getenv("BBHOST_IMPORT_CENSUS");
    return e && e[0] == '1';
}();
struct ImportAsker {
    std::uint64_t n = 0, bytes = 0, first_va = 0;
    std::uint64_t map_va = 0, map_len = 0;  // the guest mapping the first lies in
    int map_prot = 0;
};
std::map<std::string, ImportAsker> g_import_askers;  // under g.mu
void note_import_asker_locked(std::uint64_t bytes) {
    if (!g_census || !t_import_va || t_import_who == kIwSpan) return;
    static const char* const whos[] = {"binding", "device address", "walk", "prefetch"};
    const char* const who = whos[t_import_who];
    const char* const site = t_import_site ? t_import_site : "other";
    GxResourceInfo info;
    char kind[160];
    if (!gx_resource_overlapping(t_import_va, 1, &info)) {
        std::snprintf(kind, sizeof(kind), "%s (%s), no GX object", who, site);
    } else if (info.type == 1) {
        static const char* const usages[] = {"default", "immutable", "dynamic", "staging"};
        std::snprintf(kind, sizeof(kind), "%s (%s), buffer %s bind 0x%x", who, site, info.usage < 4 ? usages[info.usage] : "?", info.bind);
    } else {
        std::snprintf(kind, sizeof(kind), "%s (%s), texture", who, site);
    }
    ImportAsker& a = g_import_askers[kind];
    if (!a.n) {
        a.first_va = t_import_va;
        for (const GuestMapInfo& mi : g.maps) {
            if (t_import_va < mi.va || t_import_va >= mi.va + mi.len) continue;
            a.map_va = mi.va, a.map_len = mi.len, a.map_prot = mi.prot;
            break;
        }
    }
    ++a.n;
    a.bytes += bytes;
}
// Guest memory served in place, by the path (t_import_site): each locate()
// or device address an import answered - what still reads or writes guest
// memory on the GPU, whoever made the import (the makers' census above counts
// only the first asker of a window).
struct SiteUse {
    const char* site = nullptr;
    std::uint64_t n = 0, bytes = 0;
};
SiteUse g_site_uses[64];  // under g.mu
void note_site_use_locked(std::uint64_t bytes) {
    if (!g_census) return;
    const char* const site = t_import_site ? t_import_site : "other";
    for (SiteUse& u : g_site_uses) {
        if (!u.site) u.site = site;
        if (u.site != site) continue;
        ++u.n;
        u.bytes += bytes;
        return;
    }
}
// The GX buffers GPU copies wrote (copy tokens' destinations, copy versions'):
// by generation, with their usage, bind flags and size - candidates for a
// device-local twin of their own when the CPU never writes them.
struct CopiedBuffer {
    std::uint8_t usage = 0, bind = 0;
    std::uint32_t bytes = 0;
    std::uint64_t copies = 0;
};
std::unordered_map<std::uint64_t, CopiedBuffer> g_copied_buffers;  // under g.mu
std::uint64_t g_copied_other = 0;                                // under g.mu: into no GX buffer
void note_copied_locked(std::uint64_t dst) {
    if (!g_census) return;
    GxResourceInfo info;
    if (!gx_resource_at(dst, 1, &info) || info.type != 1) {
        ++g_copied_other;
        return;
    }
    CopiedBuffer& c = g_copied_buffers[info.gen];
    c.usage = info.usage, c.bind = info.bind, c.bytes = info.bytes;
    ++c.copies;
}
std::atomic<std::uint64_t> g_import_who_n[kImportWhos], g_import_who_bytes[kImportWhos];
// The driver's time of each import by the path that made it: the import
// thread's (bb-import) keeps nobody waiting; any other import's time is the
// time of the thread that asked for it - the command processor's, at a
// binding, a device address or a walk (under the renderer's lock, so whatever
// else wants it waits too), the mapping thread's for a whole span.
std::atomic<std::uint64_t> g_import_who_us[kImportWhos], g_import_wait_max_us{0};
std::atomic<std::uint64_t> g_imports_retired{0}, g_import_retired_bytes{0};
std::atomic<std::uint64_t> g_dispatch_read_imports{0}, g_dispatch_untraced{0};  // window imports of what dispatches read

// An import can fail on a player's machine as well as from a full card: the
// driver's own limit on imported memory, the address space, a udmabuf size
// limit on Linux. Every path that asks for one takes its failure: a binding
// is bound from the sink or walks the page table, a draw that needs one is
// skipped, a mirror's copy is left for a later draw. These are counted by the
// path that asked (the exit report), and a window whose import failed is not
// asked of the driver again for a second (import_backoff_ms) - a draw a frame
// asked again for the same window, 73,300 times in one lavapipe run.
//
// BBHOST_TEST_IMPORT_FAIL (a test switch) fails imports on purpose, so these
// paths can be run on any device: "all" every one; a number N every N-th; or
// "after:M" every one once M MiB have been imported in all.
struct ImportFailTest {
    int mode = 0;  // 0 off, 1 all, 2 every N-th, 3 after M MiB
    std::uint64_t every = 0, after_bytes = 0;
};
const ImportFailTest g_import_fail_test = [] {
    ImportFailTest t;
    const char* e = std::getenv("BBHOST_TEST_IMPORT_FAIL");
    if (!e || !*e || std::strcmp(e, "0") == 0) return t;
    if (std::strcmp(e, "all") == 0) {
        t.mode = 1;
    } else if (std::strncmp(e, "after:", 6) == 0) {
        t.mode = 3;
        t.after_bytes = std::strtoull(e + 6, nullptr, 10) << 20;
    } else if (const std::uint64_t n = std::strtoull(e, nullptr, 10); n) {
        t.mode = n == 1 ? 1 : 2;
        t.every = n;
    }
    return t;
}();
std::atomic<std::uint64_t> g_import_tries{0}, g_import_failed{0}, g_import_failed_bytes{0}, g_import_forced{0}, g_import_backed_off{0};
std::atomic<std::uint64_t> g_import_failed_who[kImportWhos];
constexpr std::uint64_t kImportBackoffMs = 1000;
bool import_fail_forced(std::uint64_t size) {
    const std::uint64_t n = g_import_tries.fetch_add(1, std::memory_order_relaxed) + 1;
    switch (g_import_fail_test.mode) {
    case 1: return true;
    case 2: return n % g_import_fail_test.every == 0;
    case 3: return g_import_bytes.load(std::memory_order_relaxed) + size > g_import_fail_test.after_bytes;
    default: return false;
    }
}

// [lo, hi) of direct-memory chunk `index` (offsets in the chunk; hi may run
// past its end, into the next chunk's memory) as a new import, not yet in the
// chunk's list: the driver's slow part, which needs no lock of ours.
std::unique_ptr<Chunk> make_import(std::uint32_t index, std::uint64_t lo, std::uint64_t hi) {
    const std::uint64_t offset = index * kChunkBytes + lo;
    if (offset >= g.mirror_size || hi <= lo) {
        return nullptr;
    }
    const std::uint64_t size = std::min(hi - lo, g.mirror_size - offset);
    void* ptr = static_cast<std::uint8_t*>(g.mirror) + offset;
    // The driver locks the pages in RAM; a whole-span import of ~3 GiB has
    // stopped a Windows laptop run silently here, so a big one is said first.
    if (size > (256ull << 20)) {
        host_log("gpu: importing dmem chunk %u: %llu MiB from +%llu MiB", index, static_cast<unsigned long long>(size >> 20),
                 static_cast<unsigned long long>(lo >> 20));
    }
    auto c = std::make_unique<Chunk>();
    const auto t0 = std::chrono::steady_clock::now();
    const bool forced = import_fail_forced(size);
#if defined(_WIN32)
    const VkResult r = forced ? VK_ERROR_OUT_OF_DEVICE_MEMORY : host_span(*c, ptr, size);
#else
    const VkResult r = forced ? VK_ERROR_OUT_OF_DEVICE_MEMORY : g.dmem_dmabuf ? dmabuf_span(*c, offset, size) : host_span(*c, ptr, size);
#endif
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    if (r != VK_SUCCESS) {
        const std::uint64_t failed = g_import_failed.fetch_add(1, std::memory_order_relaxed) + 1;
        g_import_failed_bytes.fetch_add(size, std::memory_order_relaxed);
        if (forced) g_import_forced.fetch_add(1, std::memory_order_relaxed);
        const int who = t_import_who >= 0 && t_import_who < kImportWhos ? t_import_who : kIwBinding;
        g_import_failed_who[who].fetch_add(1, std::memory_order_relaxed);
        if (failed <= 8 || failed % 10000 == 0)
            host_log("gpu: importing dmem chunk %u (%llu MiB from +%llu MiB) failed after %lld us: %d%s (%llu so far; its draws read the sink or "
                     "walk the page table)",
                     index, static_cast<unsigned long long>(size >> 20), static_cast<unsigned long long>(lo >> 20), static_cast<long long>(us), r,
                     forced ? ", BBHOST_TEST_IMPORT_FAIL" : "", static_cast<unsigned long long>(failed));
        free_import(*c);
        return nullptr;
    }
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = c->buffer;
    c->address = vkGetBufferDeviceAddress(g.device, &bai);
    c->size = size;
    c->lo = lo;
    g_import_us.fetch_add(static_cast<std::uint64_t>(us), std::memory_order_relaxed);
    g_import_bytes.fetch_add(size, std::memory_order_relaxed);
    const int who = t_import_who >= 0 && t_import_who < kImportWhos ? t_import_who : kIwBinding;
    g_import_who_n[who].fetch_add(1, std::memory_order_relaxed);
    g_import_who_bytes[who].fetch_add(size, std::memory_order_relaxed);
    g_import_who_us[who].fetch_add(static_cast<std::uint64_t>(us), std::memory_order_relaxed);
    if (who != kIwPrefetch && who != kIwSpan) {
        std::uint64_t was = g_import_wait_max_us.load(std::memory_order_relaxed);
        while (static_cast<std::uint64_t>(us) > was &&
               !g_import_wait_max_us.compare_exchange_weak(was, static_cast<std::uint64_t>(us), std::memory_order_relaxed)) {
        }
    }
    if (g_imports_made.fetch_add(1, std::memory_order_relaxed) < 8 || size > (256ull << 20)) {
        host_log("gpu: dmem chunk %u: +%llu MiB, %llu MiB imported at device address 0x%llx in %lld us", index,
                 static_cast<unsigned long long>(lo >> 20), static_cast<unsigned long long>(size >> 20),
                 static_cast<unsigned long long>(c->address), static_cast<long long>(us));
    }
    return c;
}

// The same, into the chunk's list of live imports.
Chunk* import_span(std::uint32_t index, std::uint64_t lo, std::uint64_t hi) {
    std::unique_ptr<Chunk> c = make_import(index, lo, hi);
    if (!c) return nullptr;
    g.dmem[index].imports.push_back(std::move(c));
    return g.dmem[index].imports.back().get();
}

// How direct memory reaches the GPU, tried on its first page the way every
// chunk will be: as host pointers, or as dma-bufs where the driver refuses the
// memfd's pages (AMD's, on Linux).
void choose_dmem_import() {
    Chunk c;
    const VkResult r = host_span(c, g.mirror, 1ull << gcn::kPageShift);
    free_import(c);
    if (r == VK_SUCCESS) return;
#if !defined(_WIN32)
    if (g.can_dmabuf) {
        g.udmabuf = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
        if (g.udmabuf < 0) {
            host_log("gpu: the driver refuses direct memory as a host pointer (%d), and /dev/udmabuf, the other way in, does not open "
                     "(errno %d) - nothing will draw. systemd 256 and later give it to the logged-in user (docs/running.md)",
                     r, errno);
            return;
        }
        g.dmem_dmabuf = true;
        const VkResult d = dmabuf_span(c, 0, 1ull << gcn::kPageShift);
        free_import(c);
        if (d == VK_SUCCESS) {
            host_log("gpu: the driver refuses direct memory as a host pointer (%d); importing it as dma-bufs (/dev/udmabuf) of %llu MiB "
                     "(udmabuf's size_limit_mb)%s",
                     r, static_cast<unsigned long long>(dmabuf_piece_bytes() >> 20),
                     g.dmabuf_sparse ? "" : ", one at most a span: no sparse binding");
            return;
        }
        g.dmem_dmabuf = false;
        host_log("gpu: the driver refuses direct memory as a host pointer (%d) and as a dma-buf (%d) - nothing will draw", r, d);
        return;
    }
#endif
    host_log("gpu: the driver refuses direct memory as a host pointer (%d) - nothing will draw", r);
}

void fill_sink(DevBuffer& t) {
    auto* e = static_cast<std::uint64_t*>(t.map);
    for (std::uint32_t k = 0; k < gcn::kL2Entries; ++k) e[k] = g.sink.address;
}

// The page table's entry for the 64 KiB guest page at va.
bool set_page_locked(std::uint64_t va, std::uint64_t dev) {
    const std::uint32_t i1 = static_cast<std::uint32_t>(va >> 32);
    if (i1 >= gcn::kL1Entries) return false;
    auto it = g.l2.find(i1);
    if (it == g.l2.end()) {
        DevBuffer t;
        MemOwner own_mem("page tables");
        if (!create_dev_buffer(t, gcn::kL2Entries * 8, true)) return false;
        fill_sink(t);
        it = g.l2.emplace(i1, t).first;
        static_cast<std::uint64_t*>(g.l1.map)[i1] = t.address;
    }
    static_cast<std::uint64_t*>(it->second.map)[(va >> gcn::kPageShift) & (gcn::kL2Entries - 1)] = dev;
    return true;
}

// The guest pages over [lo, hi) of chunk `index` - every mapping of them -
// point into import `c`.
void map_import_pages_locked(std::uint32_t index, const Chunk& c, std::uint64_t lo, std::uint64_t hi) {
    const std::uint64_t page = 1ull << gcn::kPageShift;
    const std::uint64_t base = static_cast<std::uint64_t>(index) * kChunkBytes;
    for (const GuestMapInfo& mi : g.maps) {
        if (!mi.dmem || !(mi.prot & 1) || mi.len == 0 || (mi.va & (page - 1)) || (mi.len & (page - 1))) continue;
        const std::uint64_t a = std::max(static_cast<std::uint64_t>(mi.phys), base + lo);
        const std::uint64_t b = std::min(static_cast<std::uint64_t>(mi.phys) + mi.len, base + hi);
        for (std::uint64_t phys = a & ~(page - 1); phys < b; phys += page) {
            set_page_locked(mi.va + (phys - static_cast<std::uint64_t>(mi.phys)), c.address + (phys - base - c.lo));
        }
    }
}
}  // namespace

thread_local constinit const char* t_validation_scope = nullptr;
bool validation_on() { return g_validate; }
void vk_debug_name(VkObjectType type, std::uint64_t handle, const char* name) {
    if (!g_validate || !g_set_object_name || !handle || !name || !g.device) return;
    VkDebugUtilsObjectNameInfoEXT ni{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    ni.objectType = type;
    ni.objectHandle = handle;
    ni.pObjectName = name;
    g_set_object_name(g.device, &ni);
}

// The census's lines in the exit report, the most frequent id first: its
// count, when it came, the first message without the spec's quotation, and
// what it was about (host_log keeps a line to 1 KiB: one line each).
void log_validation_census() {
    if (!g_validate) return;
    std::vector<std::pair<std::string, ValidationId>> ids;
    {
        std::lock_guard<std::mutex> lk(g_validation_mu);
        ids.assign(g_validation_ids.begin(), g_validation_ids.end());
    }
    std::sort(ids.begin(), ids.end(), [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
    host_log("gpu: validation census: %llu messages (warnings and errors), %zu ids", static_cast<unsigned long long>(g_validation_messages.load()),
             ids.size());
    for (const auto& [id, v] : ids) {
        std::string first = v.first;
        if (const std::size_t at = first.find("The Vulkan spec states"); at != std::string::npos) first.resize(at);
        while (!first.empty() && (first.back() == '\n' || first.back() == ' ')) first.pop_back();
        for (char& c : first) {
            if (c == '\n') c = ' ';
        }
        host_log("gpu:   %llu x %s (%s%s; %.0f-%.0f s; %zu%s subjects)", static_cast<unsigned long long>(v.count), id.c_str(),
                 (v.severities & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "error" : "warning",
                 (v.types & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) ? ", performance" : "", v.first_s, v.last_s, v.subjects.size(),
                 v.subjects.size() >= 4096 ? "+" : "");
        for (std::size_t at = 0; at < first.size() && at < 2400; at += 900) host_log("gpu:     %s", first.substr(at, 900).c_str());
        if (v.about.empty()) continue;
        std::vector<std::pair<std::uint64_t, std::string>> about;
        for (const auto& [s, n] : v.about) about.emplace_back(n, s);
        std::sort(about.rbegin(), about.rend());
        std::string line;
        for (const auto& [n, s] : about) {
            char b[64];
            std::snprintf(b, sizeof(b), " (%llu);", static_cast<unsigned long long>(n));
            if (line.size() + s.size() + 24 > 900) {
                host_log("gpu:     about:%s", line.c_str());
                line.clear();
            }
            line += " " + s + b;
        }
        if (v.about_other) line += " others " + std::to_string(v.about_other);
        if (!line.empty()) host_log("gpu:     about:%s", line.c_str());
    }
}

// Waits until the submission thread has submitted everything it was given -
// and, first, until the recorder has handed over every submission in the
// blocks published so far (the command stream's kEndSubmit and
// kForeignSubmit): work queued after this goes behind them.
void queue_drain() {
    stream_drain_published();
    if (!g_sub_running) return;
    std::unique_lock<std::mutex> lk(g_sub_mu);
    g_sub_idle.wait(lk, [] { return g_sub_jobs.empty() && !g_sub_busy; });
}

void stream_submit_job(const VkCommandBuffer* cmds, std::uint32_t ncmds, VkSemaphore wait, VkPipelineStageFlags wait_stage, VkSemaphore signal,
                       VkFence fence, std::uint32_t items, std::uint64_t serial, std::uint64_t foreign_id) {
    SubmitJob job;
    job.ncmds = ncmds;
    for (std::uint32_t k = 0; k < ncmds && k < 2; ++k) job.cmds[k] = cmds[k];
    job.wait = wait;
    job.wait_stage = wait_stage;
    job.signal = signal;
    job.fence = fence;
    job.items = items;
    job.serial = serial;
    job.foreign = foreign_id;
    if (g_test_gpu_delay_ns) job.enqueued_ns = host_clock_monotonic_ns();
    if (g_sub_running) {
        enqueue_submit(job);
    } else {
        submit_job_now(job);  // BBHOST_SUBMIT_THREAD=0: on the recorder, under the queue's lock
    }
}

std::uint64_t stream_submitted_serial() { return g_stream_submitted.load(std::memory_order_acquire); }

QueueGuard::QueueGuard() {
    queue_drain();
    g.queue_mu.lock();
}
QueueGuard::~QueueGuard() { g.queue_mu.unlock(); }
std::string submit_thread_report() { return submit_thread_report_impl(); }
std::string submit_merge_report() {
    if (!g_sub_running) return {};
    const std::uint64_t jobs = g_sub_jobs_done.load(std::memory_order_relaxed), subs = g_sub_queue_submits.load(std::memory_order_relaxed);
    char buf[200];
    std::snprintf(buf, sizeof(buf), "submissions: %llu jobs in %llu queue submissions, %llu merged into an earlier one (BBHOST_SUBMIT_MERGE=%u), "
                  "%llu fence-only", static_cast<unsigned long long>(jobs), static_cast<unsigned long long>(subs),
                  static_cast<unsigned long long>(g_sub_jobs_merged.load(std::memory_order_relaxed)), g_submit_merge,
                  static_cast<unsigned long long>(g_sub_fence_submits.load(std::memory_order_relaxed)));
    return buf;
}

namespace {
// How far an import of chunk `index` may run: to its span's end, or on past
// the chunk's end into the next chunk's span by up to kChunkOverlap when that
// span starts at the next chunk's beginning - a buffer the game placed across
// the seam then lies in one import, as the whole spans' overlap gives (f26afe4).
std::uint64_t span_reach(std::uint32_t index) {
    const DmemChunk& d = g.dmem[index];
    if (d.span_hi != kChunkBytes || index + 1 >= gcn::kDmemChunks) return d.span_hi;
    const DmemChunk& n = g.dmem[index + 1];
    if (n.span_hi <= n.span_lo || n.span_lo != 0) return d.span_hi;
    return kChunkBytes + std::min<std::uint64_t>(n.span_hi, kChunkOverlap);
}

// A window import no window names any more: its pages are in the import that
// replaced it (the window and page-table entries moved there), so it is let go
// once the work recorded with it has run (the current slot's retirement). What
// may name its buffer later than that goes first: a copy version's copy into
// place, the cached descriptor sets (constant buffers in them, without push
// descriptors - with them they are pushed per draw).
void retire_import_locked(std::uint32_t index, Chunk* old) {
    DmemChunk& d = g.dmem[index];
    const auto it = std::find_if(d.imports.begin(), d.imports.end(), [&](const std::unique_ptr<Chunk>& p) { return p.get() == old; });
    if (it == d.imports.end()) return;
    if (copy_versions_name_buffer_locked(old->buffer)) copy_versions_put_in_place_locked(kCvRetired);
    if (!cb_push_on()) set_cache_drop_all_locked();
    g_imports_retired.fetch_add(1, std::memory_order_relaxed);
    g_import_retired_bytes.fetch_add(old->size, std::memory_order_relaxed);
    g.slots[g.slot].dead_imports.push_back(std::move(*old));
    d.imports.erase(it);
}

// The windows of chunk `index` that import `c` covers name it from now on and
// their guest pages point into it; the imports it replaced whole retire.
void install_import_locked(std::uint32_t index, Chunk* c) {
    DmemChunk& d = g.dmem[index];
    const std::uint64_t end = std::min(c->lo + c->size, kChunkBytes);
    if (end <= c->lo) return;
    const std::uint32_t a = static_cast<std::uint32_t>(c->lo >> kImportWindowShift);
    const std::uint32_t b = static_cast<std::uint32_t>((end - 1) >> kImportWindowShift);
    Chunk* replaced[kImportWindows];
    std::uint32_t nreplaced = 0;
    for (std::uint32_t w = a; w <= b; ++w) {
        Chunk* o = d.window[w];
        if (o && o != c && std::find(replaced, replaced + nreplaced, o) == replaced + nreplaced) replaced[nreplaced++] = o;
        d.window[w] = c;
    }
    map_import_pages_locked(index, *c, c->lo, end);
    if (nreplaced) g_imports_merged.fetch_add(1, std::memory_order_relaxed);
    for (std::uint32_t k = 0; k < nreplaced; ++k) {
        if (std::find(std::begin(d.window), std::end(d.window), replaced[k]) == std::end(d.window)) retire_import_locked(index, replaced[k]);
    }
}
}  // namespace

// ---- The window imports' idle census (BBHOST_IMPORT_IDLE_CENSUS=1; off, nothing is counted) ----
// How long each import goes without a use - a binding, a device address, a
// walk, a copy: every dmem_import that finds it - and what letting one go
// after 60 s without a use would give back, against the imports it would
// have had to make again: a use after such a gap is one, of its size, waited
// for by whoever needed it (on a Radeon 680M 5-46 ms a 16 MiB window). Also
// by game frames (1,800 and 3,600 flips: 30 and 60 s at 60 fps), for a run
// slower than the game's own pace (lavapipe).
namespace {
std::int64_t steady_ms();  // below, beside the image heap
const bool g_idle_census = [] {
    const char* e = std::getenv("BBHOST_IMPORT_IDLE_CENSUS");
    return e && e[0] == '1';
}();
constexpr std::uint64_t kIdleLetGoMs = 60000;
// Uses after a gap of 10-60 s, 60-120 s, 120 s or more: how many, and the imports' MiB.
std::uint64_t g_idle_gap_n[3] = {}, g_idle_gap_bytes[3] = {};
std::uint64_t g_idle_uses = 0;
// Uses that letting go after 60 s (and after 1,800 / 3,600 flips) would have made imports again.
std::uint64_t g_idle_again_n = 0, g_idle_again_bytes = 0, g_idle_again_f30_n = 0, g_idle_again_f30_bytes = 0, g_idle_again_f60_n = 0,
              g_idle_again_f60_bytes = 0;
// At the census reports: the live imports idle 60 s or more (and 3,600 flips or more), their sum (for the mean) and most.
std::uint64_t g_idle_reports = 0, g_idle_over_sum = 0, g_idle_over_max = 0, g_idle_over_f_sum = 0, g_idle_over_f_max = 0;

void idle_note_use_locked(Chunk& c) {
    if (!g_idle_census) return;
    const std::uint64_t now = static_cast<std::uint64_t>(steady_ms()), flip = hle_video_flip_count();
    if (c.used_ms) {
        const std::uint64_t gap = now - c.used_ms, gapf = flip - c.used_flip;
        if (gap >= 10000) {
            const int b = gap < 60000 ? 0 : gap < 120000 ? 1 : 2;
            ++g_idle_gap_n[b];
            g_idle_gap_bytes[b] += c.size;
        }
        if (gap >= kIdleLetGoMs) {
            ++g_idle_again_n;
            g_idle_again_bytes += c.size;
        }
        if (gapf >= 1800) {
            ++g_idle_again_f30_n;
            g_idle_again_f30_bytes += c.size;
        }
        if (gapf >= 3600) {
            ++g_idle_again_f60_n;
            g_idle_again_f60_bytes += c.size;
        }
    }
    ++g_idle_uses;
    c.used_ms = now;
    c.used_flip = flip;
}
}  // namespace

// The census line (the video memory report and the exit report).
std::string import_idle_report_locked() {
    if (!g_idle_census) return {};
    const std::uint64_t now = static_cast<std::uint64_t>(steady_ms()), flip = hle_video_flip_count();
    std::uint64_t live = 0, n = 0, under10 = 0, to60 = 0, over = 0, over_f = 0;
    for (const DmemChunk& d : g.dmem) {
        for (const auto& c : d.imports) {
            live += c->size;
            ++n;
            const std::uint64_t idle = c->used_ms ? now - c->used_ms : 0;
            (idle < 10000 ? under10 : idle < kIdleLetGoMs ? to60 : over) += c->size;
            if (c->used_ms && flip - c->used_flip >= 3600) over_f += c->size;
        }
    }
    ++g_idle_reports;
    g_idle_over_sum += over;
    g_idle_over_max = std::max(g_idle_over_max, over);
    g_idle_over_f_sum += over_f;
    g_idle_over_f_max = std::max(g_idle_over_f_max, over_f);
    const auto mib = [](std::uint64_t b) { return static_cast<unsigned long long>(b >> 20); };
    const double ms_per_mib = g_import_bytes.load() ? static_cast<double>(g_import_us.load()) / 1e3 / (static_cast<double>(g_import_bytes.load()) / (1 << 20)) : 0;
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "import idle census: live %llu MiB in %llu imports; idle now under 10 s %llu, 10-60 s %llu, 60 s or more %llu MiB (3,600 "
                  "flips or more %llu MiB); at the %llu reports so far idle 60 s or more mean %llu, most %llu MiB (3,600 flips: mean %llu, "
                  "most %llu); uses %llu, after a gap of 10-60 s %llu (%llu MiB), 60-120 s %llu (%llu MiB), 120 s or more %llu (%llu MiB); "
                  "letting an import go after 60 s idle would have made %llu again (%llu MiB), after 1,800 flips %llu (%llu MiB), after 3,600 "
                  "flips %llu (%llu MiB) - at this run's %.3f ms a MiB of the driver's time, %.0f ms",
                  mib(live), static_cast<unsigned long long>(n), mib(under10), mib(to60), mib(over), mib(over_f),
                  static_cast<unsigned long long>(g_idle_reports), mib(g_idle_over_sum / g_idle_reports), mib(g_idle_over_max),
                  mib(g_idle_over_f_sum / g_idle_reports), mib(g_idle_over_f_max), static_cast<unsigned long long>(g_idle_uses),
                  static_cast<unsigned long long>(g_idle_gap_n[0]), mib(g_idle_gap_bytes[0]), static_cast<unsigned long long>(g_idle_gap_n[1]),
                  mib(g_idle_gap_bytes[1]), static_cast<unsigned long long>(g_idle_gap_n[2]), mib(g_idle_gap_bytes[2]),
                  static_cast<unsigned long long>(g_idle_again_n), mib(g_idle_again_bytes), static_cast<unsigned long long>(g_idle_again_f30_n),
                  mib(g_idle_again_f30_bytes), static_cast<unsigned long long>(g_idle_again_f60_n), mib(g_idle_again_f60_bytes), ms_per_mib,
                  ms_per_mib * static_cast<double>(g_idle_again_bytes >> 20));
    return buf;
}

// A window whose import failed less than kImportBackoffMs ago is not asked of
// the driver again (counted); and a failure's time, over its windows.
bool import_backing_off(const DmemChunk& d, std::uint32_t w0, std::uint32_t w1) {
    const std::uint64_t now = static_cast<std::uint64_t>(steady_ms());
    for (std::uint32_t w = w0; w <= w1 && w < kImportWindows; ++w) {
        if (d.failed_ms[w] && now - d.failed_ms[w] < kImportBackoffMs) {
            g_import_backed_off.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}
void import_failed_at(DmemChunk& d, std::uint32_t w0, std::uint32_t w1) {
    const std::uint64_t now = std::max<std::uint64_t>(static_cast<std::uint64_t>(steady_ms()), 1);
    for (std::uint32_t w = w0; w <= w1 && w < kImportWindows; ++w) d.failed_ms[w] = now;
}

const Chunk* dmem_import(std::uint32_t index, std::uint64_t in, std::uint64_t bytes, bool past_end) {
    if (index >= gcn::kDmemChunks) return nullptr;
    DmemChunk& d = g.dmem[index];
    if (!bytes) bytes = 1;
    if (imports_whole()) {
        // The whole spans (imported when the mappings are made): a range their
        // windows do not cover lies outside every GPU-visible mapping.
        if (in >= kChunkBytes || bytes > kChunkBytes - in) return nullptr;
        const std::uint32_t w0 = static_cast<std::uint32_t>(in >> kImportWindowShift);
        if (const Chunk* c = d.window[w0]; c && chunk_covers(*c, in, bytes)) return c;
        if (in < d.span_lo || in + bytes > d.span_hi) return nullptr;  // not GPU-visible: the sink
        if (run_part_stop(kRunStopIndex)) return nullptr;  // imported in order (BBHOST_RUN_COMMIT)
        const std::uint32_t w1 = static_cast<std::uint32_t>((in + bytes - 1) >> kImportWindowShift);
        const std::uint64_t lo = std::max(d.span_lo, static_cast<std::uint64_t>(w0) << kImportWindowShift);
        const std::uint64_t hi = std::min(d.span_hi, static_cast<std::uint64_t>(w1 + 1) << kImportWindowShift);
        bool merged = false;
        for (std::uint32_t w = w0; w <= w1; ++w) merged |= d.window[w] != nullptr;
        if (import_backing_off(d, w0, w1)) return nullptr;
        Chunk* c = import_span(index, lo, hi);
        if (!c) {
            import_failed_at(d, w0, w1);
            return nullptr;
        }
        if (merged) g_imports_merged.fetch_add(1, std::memory_order_relaxed);
        for (std::uint32_t w = w0; w <= w1; ++w) {
            // A window another import already held keeps its pages' entries.
            if (!d.window[w]) {
                const std::uint64_t wlo = std::max(lo, static_cast<std::uint64_t>(w) << kImportWindowShift);
                const std::uint64_t whi = std::min(hi, static_cast<std::uint64_t>(w + 1) << kImportWindowShift);
                map_import_pages_locked(index, *c, wlo, whi);
            }
            d.window[w] = c;
        }
        return c;
    }
    // Windows (BBHOST_IMPORT_WINDOWS=1): the whole windows the range touches,
    // widened to the whole of every import over them, which it replaces.
    // Past the chunk's end only for a caller that takes the seam's overlap
    // (past_end: a binding, a device address); any other range - a mirror's
    // pages above all, which end with the chunk (buffer_shadow.cpp phys_of) -
    // stays inside the chunk, as with whole imports, also where an import over
    // the seam would cover it: the mirror's page index ran past its pages,
    // and its sparse page binds past the end of its buffer, which the driver
    // refused with a lost device.
    if (in >= kChunkBytes || (!past_end && bytes > kChunkBytes - in)) return nullptr;
    const std::uint32_t w0 = static_cast<std::uint32_t>(in >> kImportWindowShift);
    if (Chunk* c = d.window[w0]; c && chunk_covers(*c, in, bytes)) {
        idle_note_use_locked(*c);
        return c;
    }
    const std::uint64_t reach = past_end ? span_reach(index) : std::min<std::uint64_t>(d.span_hi, kChunkBytes);
    if (in < d.span_lo || in >= reach || bytes > reach - in) return nullptr;  // not GPU-visible: the sink
    if (run_part_stop(kRunStopIndex)) return nullptr;  // imported in order (BBHOST_RUN_COMMIT)
    constexpr std::uint64_t kWindow = 1ull << kImportWindowShift;
    std::uint64_t lo = std::max(d.span_lo, in & ~(kWindow - 1));
    std::uint64_t hi = std::min(reach, ((in + bytes - 1) | (kWindow - 1)) + 1);
    for (bool grew = true; grew;) {
        grew = false;
        const std::uint32_t a = static_cast<std::uint32_t>(lo >> kImportWindowShift);
        const std::uint32_t b = static_cast<std::uint32_t>((std::min(hi, kChunkBytes) - 1) >> kImportWindowShift);
        for (std::uint32_t w = a; w <= b; ++w) {
            const Chunk* o = d.window[w];
            if (!o) continue;
            if (o->lo < lo) lo = o->lo, grew = true;
            if (o->lo + o->size > hi) hi = o->lo + o->size, grew = true;
        }
    }
    const std::uint32_t wa = static_cast<std::uint32_t>(lo >> kImportWindowShift);
    const std::uint32_t wb = static_cast<std::uint32_t>((std::min(hi, kChunkBytes) - 1) >> kImportWindowShift);
    if (import_backing_off(d, wa, wb)) return nullptr;
    Chunk* c = import_span(index, lo, hi);
    if (!c) {
        import_failed_at(d, wa, wb);
        return nullptr;
    }
    note_import_asker_locked(c->size);
    install_import_locked(index, c);
    if (g_idle_census) c->made_ms = static_cast<std::uint64_t>(steady_ms());
    idle_note_use_locked(*c);
    return c;
}

void import_windows(std::uint64_t va, std::uint64_t bytes) {
    const int who = t_import_who;
    const std::uint64_t asked = t_import_va;
    t_import_who = kIwWalk;
    const std::uint64_t end = va + bytes;
    for (const GuestMapInfo& mi : g.maps) {
        if (!mi.dmem) continue;
        const std::uint64_t lo = std::max(va, mi.va), hi = std::min(end, mi.va + mi.len);
        for (std::uint64_t at = lo; at < hi;) {
            const std::uint64_t phys = static_cast<std::uint64_t>(mi.phys) + (at - mi.va);
            t_import_va = at;
            dmem_import(static_cast<std::uint32_t>(phys >> gcn::kDmemChunkShift), phys & (kChunkBytes - 1), 1);
            at += (1ull << kImportWindowShift) - (phys & ((1ull << kImportWindowShift) - 1));  // the next window
        }
    }
    t_import_va = asked;
    t_import_who = who;
}

std::atomic<signed char> g_imports_whole{-1};
std::atomic<signed char> g_integrated_known{-1};  // the chosen device is integrated (1) or not (0); -1 before
// BBHOST_IMPORT_WINDOWS (1 windows, 0 whole spans), else gpu.import_windows
// (true, false), else by the device and the platform: windows on an
// integrated GPU under Windows, whole spans elsewhere (host/import_mode.h).
// Asked before the device is chosen with neither set, it answers whole spans
// without deciding (nothing is imported before).
bool imports_whole_decide() {
#if defined(_WIN32)
    constexpr bool kWindowsHost = true;
#else
    constexpr bool kWindowsHost = false;
#endif
    const std::string cfg = config_value("gpu.import_windows");
    const import_mode::Mode m = import_mode::decide(std::getenv("BBHOST_IMPORT_WINDOWS"), cfg.c_str(),
                                                    g_integrated_known.load(std::memory_order_relaxed), kWindowsHost);
    if (m == import_mode::Mode::kUndecided) return true;
    const bool whole = m == import_mode::Mode::kWhole;
    g_imports_whole.store(whole ? 1 : 0, std::memory_order_relaxed);
    return whole;
}

// ---- import prefetch (window imports) ----
// A GX buffer made over GPU-visible direct memory that the GPU reads in place
// (host_gpu_import_prefetch says which) is read soon after. Its windows are
// imported on a thread of our own (bb-import) - the driver's slow part, pinning the pages
// (~0.25 ms a MiB on the RTX 4070), without the renderer's lock, put in place
// under it - instead of by the binding that first needs them, on the command
// processor. A run of windows some binding imported meanwhile is dropped (an
// import is never put over part of another: no page imported twice).
// BBHOST_IMPORT_PREFETCH=0: imports at the first use only.
// The meshes too (host/import_prefetch.h): a vertex or index buffer the
// buffer shadow reads in place when GX tells its writes (BBHOST_SHADOW_IN_PLACE,
// on an integrated GPU), once the device says the shadow does.
// BBHOST_IMPORT_PREFETCH_MESHES=0: their windows at their first draw's
// binding, on the command processor.
namespace {
const bool g_prefetch_on = [] {
    const char* e = std::getenv("BBHOST_IMPORT_PREFETCH");
    return !(e && e[0] == '0');
}();
const bool g_prefetch_meshes = [] {
    const char* e = std::getenv("BBHOST_IMPORT_PREFETCH_MESHES");
    return !(e && e[0] == '0');
}();
struct PrefetchJob {
    std::uint64_t va = 0, bytes = 0;
    bool meshes = false;  // import_prefetch::Ask::kMeshes: only if the shadow reads told meshes in place
};
std::mutex g_pf_mu;
std::condition_variable g_pf_cv;
std::deque<PrefetchJob> g_pf_jobs;  // under g_pf_mu
bool g_pf_started = false;          // under g_pf_mu
std::atomic<std::uint64_t> g_pf_asked{0}, g_pf_known{0}, g_pf_checked{0}, g_pf_imports{0}, g_pf_bytes{0}, g_pf_raced{0},
    g_pf_dropped{0};
std::atomic<std::uint64_t> g_pf_mesh_asked{0}, g_pf_mesh_skipped{0}, g_pf_mesh_imports{0}, g_pf_mesh_bytes{0};

void prefetch_thread() {
    host_thread_set_name("bb-import");
    t_import_who = kIwPrefetch;
    t_import_site = "a GX buffer made";
    constexpr std::uint64_t kWindow = 1ull << kImportWindowShift;
    // This thread's own view: the mappings (refreshed when they change) and
    // the windows it has seen imported - an import is only ever replaced by a
    // larger one, so a window once imported stays so.
    std::vector<GuestMapInfo> maps;
    std::uint64_t maps_gen = ~0ull;
    std::uint64_t known[gcn::kDmemChunks] = {};      // imported (stays so)
    std::uint64_t invisible[gcn::kDmemChunks] = {};  // outside every GPU-visible span (until the mappings change)
    struct Run {
        std::uint32_t chunk;
        std::uint64_t lo, hi;
    };
    std::vector<Run> runs;
    for (;;) {
        PrefetchJob job;
        {
            std::unique_lock<std::mutex> lk(g_pf_mu);
            g_pf_cv.wait(lk, [] { return !g_pf_jobs.empty(); });
            job = g_pf_jobs.front();
            g_pf_jobs.pop_front();
        }
        if (const std::uint64_t gen = hle_kernel_maps_generation(); gen != maps_gen) {
            maps_gen = gen;
            maps.clear();
            hle_kernel_snapshot_maps(maps);
            std::fill(std::begin(invisible), std::end(invisible), 0);
        }
        // The job's windows this thread has not seen imported.
        bool unknown = false;
        const std::uint64_t end = job.va + job.bytes;
        for (const GuestMapInfo& mi : maps) {
            if (!mi.dmem || mi.phys < 0) continue;
            const std::uint64_t lo = std::max(job.va, mi.va), hi = std::min(end, mi.va + mi.len);
            for (std::uint64_t at = lo; at < hi && !unknown;) {
                const std::uint64_t phys = static_cast<std::uint64_t>(mi.phys) + (at - mi.va);
                const std::uint64_t k = phys >> gcn::kDmemChunkShift;
                const std::uint32_t w = static_cast<std::uint32_t>((phys & (kChunkBytes - 1)) >> kImportWindowShift);
                unknown = k < gcn::kDmemChunks && !(((known[k] | invisible[k]) >> w) & 1);
                at += kWindow - (phys & (kWindow - 1));
            }
        }
        if (!unknown) {
            g_pf_known.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        g_pf_checked.fetch_add(1, std::memory_order_relaxed);
        // Under the lock: which of those windows lie in a GPU-visible span and
        // no import covers yet, as runs of adjacent windows.
        runs.clear();
        {
            std::lock_guard<GpuMutex> lock(g.mu);
            if (!g.ok || !rebuild_page_tables()) continue;
            if (job.meshes && !shadow_in_place_on()) {  // (asked again: the mirrors copy them through staging, no import)
                g_pf_mesh_skipped.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            for (const GuestMapInfo& mi : g.maps) {
                if (!mi.dmem || mi.phys < 0) continue;
                const std::uint64_t lo = std::max(job.va, mi.va), hi = std::min(end, mi.va + mi.len);
                for (std::uint64_t at = lo; at < hi;) {
                    const std::uint64_t phys = static_cast<std::uint64_t>(mi.phys) + (at - mi.va);
                    at += kWindow - (phys & (kWindow - 1));
                    const std::uint64_t k = phys >> gcn::kDmemChunkShift;
                    if (k >= gcn::kDmemChunks) continue;
                    const DmemChunk& d = g.dmem[k];
                    const std::uint64_t in = phys & (kChunkBytes - 1);
                    const std::uint32_t w = static_cast<std::uint32_t>(in >> kImportWindowShift);
                    const std::uint64_t wlo = std::max(d.span_lo, static_cast<std::uint64_t>(w) << kImportWindowShift);
                    const std::uint64_t whi = std::min(d.span_hi, static_cast<std::uint64_t>(w + 1) << kImportWindowShift);
                    if (wlo >= whi) {  // not GPU-visible
                        invisible[k] |= 1ull << w;
                        continue;
                    }
                    if (const Chunk* c = d.window[w]; c && chunk_covers(*c, wlo, whi - wlo)) {
                        known[k] |= 1ull << w;
                        continue;
                    }
                    if (d.window[w]) continue;  // partly imported: the binding that needs the rest widens over it
                    if (!runs.empty() && runs.back().chunk == k && runs.back().hi == wlo) {
                        runs.back().hi = whi;
                    } else {
                        runs.push_back({static_cast<std::uint32_t>(k), wlo, whi});
                    }
                }
            }
        }
        for (const Run& r : runs) {
            std::unique_ptr<Chunk> c = make_import(r.chunk, r.lo, r.hi);  // the slow part, unlocked
            if (!c) continue;
            std::lock_guard<GpuMutex> lock(g.mu);
            DmemChunk& d = g.dmem[r.chunk];
            bool taken = false;
            for (std::uint64_t w = r.lo >> kImportWindowShift; w <= (r.hi - 1) >> kImportWindowShift; ++w) taken |= d.window[w] != nullptr;
            if (taken) {
                free_import(*c);  // nothing names it yet
                g_pf_raced.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            g_pf_imports.fetch_add(1, std::memory_order_relaxed);
            g_pf_bytes.fetch_add(c->size, std::memory_order_relaxed);
            if (job.meshes) {
                g_pf_mesh_imports.fetch_add(1, std::memory_order_relaxed);
                g_pf_mesh_bytes.fetch_add(c->size, std::memory_order_relaxed);
            }
            d.imports.push_back(std::move(c));
            install_import_locked(r.chunk, d.imports.back().get());
            t_import_va = job.va;  // the census: by the buffer the job is for
            note_import_asker_locked(d.imports.back()->size);
            t_import_va = 0;
            for (std::uint64_t w = r.lo >> kImportWindowShift; w <= (r.hi - 1) >> kImportWindowShift; ++w) known[r.chunk] |= 1ull << w;
        }
    }
}
}  // namespace

namespace {

// The span of chunk `index` that GPU-visible direct-memory mappings cover
// (SCE_KERNEL_PROT_GPU_READ/WRITE), in offsets of the chunk, 64 KiB-aligned;
// false when none does. On the console the GPU cannot reach a mapping without
// those bits, and an audit (BBHOST_IMPORT_AUDIT) found no GPU-side read of
// one: the main heap, Havok's, the runtime heap and the CPU-side GX heap -
// 1.2 GiB - and the pool's never-allocated tail need no import.
bool gpu_span(const std::vector<GuestMapInfo>& maps, std::uint32_t index, std::uint64_t& lo, std::uint64_t& hi) {
    constexpr std::uint64_t kAlign = 1ull << gcn::kPageShift;
    const std::uint64_t base = static_cast<std::uint64_t>(index) * kChunkBytes;
    lo = kChunkBytes;
    hi = 0;
    for (const GuestMapInfo& mi : maps) {
        if (!mi.dmem || !(mi.prot & kGuestProtGpu) || mi.len == 0) continue;
        const std::uint64_t a = static_cast<std::uint64_t>(mi.phys), b = a + mi.len;
        if (b <= base || a >= base + kChunkBytes) continue;
        lo = std::min(lo, (std::max(a, base) - base) & ~(kAlign - 1));
        hi = std::max(hi, (std::min(b, base + kChunkBytes) - base + kAlign - 1) & ~(kAlign - 1));
    }
    return hi > lo;
}

// Import one anonymous (flexible memory) guest mapping.
const Chunk* import_anon(std::uint64_t va, std::uint64_t len) {
    auto it = g.anon_imports.find(va);
    if (it != g.anon_imports.end()) {
        return &it->second;
    }
    Chunk c;
    void* ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(va));
    VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (g.get_host_pointer_props(g.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &hp) != VK_SUCCESS) {
        return nullptr;
    }
    const std::uint32_t type = find_memory_type(hp.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) return nullptr;
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &ext;
    bci.size = len;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (vkCreateBuffer(g.device, &bci, nullptr, &c.buffer) != VK_SUCCESS) return nullptr;
    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = ptr;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.pNext = &imp;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &flags;
    mai.allocationSize = len;
    mai.memoryTypeIndex = type;
    static std::atomic<int> said{0};
    const bool say = len >= (256ull << 20) || said.fetch_add(1) < 4;  // as the dmem chunks: before, in case it never returns
    if (say) host_log("gpu: importing flexible mapping 0x%llx (%llu KiB)", static_cast<unsigned long long>(va), static_cast<unsigned long long>(len >> 10));
    const auto t0 = std::chrono::steady_clock::now();
    MemOwner own_mem("guest memory imports");
    const VkResult r = vkAllocateMemory(g.device, &mai, nullptr, &c.memory);
    const double import_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (say || import_ms >= 50) host_log("gpu: flexible mapping 0x%llx: %s in %.0f ms", static_cast<unsigned long long>(va), r == VK_SUCCESS ? "imported" : "failed", import_ms);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(g.device, c.buffer, nullptr);
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 4) {
            host_log("gpu: importing flexible mapping 0x%llx (%llu KiB) failed", static_cast<unsigned long long>(va),
                     static_cast<unsigned long long>(len >> 10));
        }
        return nullptr;
    }
    vkBindBufferMemory(g.device, c.buffer, c.memory, 0);
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = c.buffer;
    c.address = vkGetBufferDeviceAddress(g.device, &bai);
    c.size = len;
    return &(g.anon_imports[va] = c);
}

// BBHOST_DUMMY_CLEAR (on unless 0): the shaped dummies below hold opaque black,
// as the 2D one does, and a zero variant each for an invalid T#, instead of
// whatever their memory held: a missing cube, array or 3D texture sampled
// whatever a driver's allocation had (zeros on NVIDIA, garbage elsewhere).
const bool g_dummy_clear = [] {
    const char* e = std::getenv("BBHOST_DUMMY_CLEAR");
    return !(e && e[0] == '0');
}();
// BBHOST_TEST_DUMMIES=1 (tests): every dummy image is read back once made and
// its texels checked against what it should hold (check_dummies; the line
// "gpu: dummy images checked: ..."). They are made copyable for it.
const bool g_test_dummies = [] {
    const char* e = std::getenv("BBHOST_TEST_DUMMIES");
    return e && e[0] == '1';
}();
// The 2D dummy above covers most bindings; shaders also declare 1D, 3D, cube
// and arrayed images, and Vulkan requires the view type to match exactly. Make
// a 1x1 image for each so no declared binding is ever left unwritten, cleared
// to opaque black (key (dim << 1) | arrayed) and to zero (the same | 0x100).
void create_dim_dummies() {
    struct Kind {
        std::uint32_t dim;
        bool arrayed;
        VkImageType type;
        VkImageViewType view;
        std::uint32_t layers;
        VkImageCreateFlags flags;
    };
    static const Kind kinds[] = {
        {0, false, VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D, 1, 0},
        {0, true, VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1, 0},
        {1, true, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, 0},
        {2, false, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 1, 0},
        {3, false, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
        {3, true, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    };
    for (const Kind& k : kinds) {
        for (int variant = 0; variant < (g_dummy_clear ? 2 : 1); ++variant) {
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.flags = k.flags;
            ici.imageType = k.type;
            ici.format = VK_FORMAT_R8G8B8A8_UNORM;
            ici.extent = {1, 1, 1};
            ici.mipLevels = 1;
            ici.arrayLayers = k.layers;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | (g_dummy_clear ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u) |
                        (g_test_dummies ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0u);
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImage image = VK_NULL_HANDLE;
            if (vkCreateImage(g.device, &ici, nullptr, &image) != VK_SUCCESS) continue;
            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(g.device, image, &req);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, 0);
            VkDeviceMemory mem = VK_NULL_HANDLE;
            MemOwner own_mem("dummy images and buffers");
            if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &mem) != VK_SUCCESS) {
                vkDestroyImage(g.device, image, nullptr);
                continue;
            }
            vkBindImageMemory(g.device, image, mem, 0);
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = image;
            vci.viewType = k.view;
            vci.format = ici.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, k.layers};
            VkImageView view = VK_NULL_HANDLE;
            if (vkCreateImageView(g.device, &vci, nullptr, &view) != VK_SUCCESS) {
                vkDestroyImage(g.device, image, nullptr);
                vkFreeMemory(g.device, mem, nullptr);
                continue;
            }
            // GENERAL, the layout the descriptors claim, with the texel the
            // 2D dummies hold: opaque black, or zero for an invalid T#.
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(g_cmd(), &bi);
            ++g.record_serial;  // dynamic state recorded so far is gone
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = image;
            b.subresourceRange = vci.subresourceRange;
            b.dstAccessMask = g_dummy_clear ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_dummy_clear ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &b);
            if (g_dummy_clear) {
                const VkClearColorValue texel{{0.0f, 0.0f, 0.0f, variant ? 0.0f : 1.0f}};
                vkCmdClearColorImage(g_cmd(), image, VK_IMAGE_LAYOUT_GENERAL, &texel, 1, &vci.subresourceRange);
                b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
            }
            vkEndCommandBuffer(g_cmd());
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &g.cmd_;
            vkResetFences(g.device, 1, &g.fence);
            {
                QueueGuard queue;
                vkQueueSubmit(g.queue, 1, &si, g.fence);
            }
            vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
            vkResetFences(g.device, 1, &g.fence);  // the slot's own: its first real submission wants it unsignaled
            const std::uint32_t key = (k.dim << 1) | (k.arrayed ? 1u : 0u) | (variant ? 0x100u : 0u);
            g.dummy_dim_images[key] = image;
            g.dummy_dim_mem[key] = mem;
            g.dummy_dim_views[key] = view;
        }
    }
}

// The far-plane depth dummies (Gpu::dummy_depth_views): D32_SFLOAT where the
// device samples it with a comparison, cleared to 1.0, GENERAL.
void create_depth_dummies() {
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(g.phys, VK_FORMAT_D32_SFLOAT, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) return;
    struct Kind {
        std::uint32_t dim;
        bool arrayed;
        VkImageViewType view;
        std::uint32_t layers;
        VkImageCreateFlags flags;
    };
    static const Kind kinds[] = {
        {1, false, VK_IMAGE_VIEW_TYPE_2D, 1, 0},
        {1, true, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, 0},
        {3, false, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
        {3, true, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    };
    for (const Kind& k : kinds) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = k.flags;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_D32_SFLOAT;
        ici.extent = {1, 1, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = k.layers;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | (g_test_dummies ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0u);
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage image = VK_NULL_HANDLE;
        if (vkCreateImage(g.device, &ici, nullptr, &image) != VK_SUCCESS) continue;
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(g.device, image, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, 0);
        VkDeviceMemory mem = VK_NULL_HANDLE;
        MemOwner own_mem("dummy images and buffers");
        if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &mem) != VK_SUCCESS) {
            vkDestroyImage(g.device, image, nullptr);
            continue;
        }
        vkBindImageMemory(g.device, image, mem, 0);
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = image;
        vci.viewType = k.view;
        vci.format = ici.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, k.layers};
        VkImageView view = VK_NULL_HANDLE;
        if (vkCreateImageView(g.device, &vci, nullptr, &view) != VK_SUCCESS) {
            vkDestroyImage(g.device, image, nullptr);
            vkFreeMemory(g.device, mem, nullptr);
            continue;
        }
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(g_cmd(), &bi);
        ++g.record_serial;  // dynamic state recorded so far is gone
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = vci.subresourceRange;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        const VkClearDepthStencilValue far{1.0f, 0};
        vkCmdClearDepthStencilImage(g_cmd(), image, VK_IMAGE_LAYOUT_GENERAL, &far, 1, &vci.subresourceRange);
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        vkEndCommandBuffer(g_cmd());
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g.cmd_;
        vkResetFences(g.device, 1, &g.fence);
        {
            QueueGuard queue;
            vkQueueSubmit(g.queue, 1, &si, g.fence);
        }
        vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
        vkResetFences(g.device, 1, &g.fence);  // the slot's own: its first real submission wants it unsignaled
        const std::uint32_t key = (k.dim << 1) | (k.arrayed ? 1u : 0u);
        g.dummy_depth_images[key] = image;
        g.dummy_depth_mem[key] = mem;
        g.dummy_depth_views[key] = view;
    }
}

// BBHOST_TEST_DUMMIES=1: each dummy image's every layer read back and checked -
// the 2D and shaped colour ones opaque black (0, 0, 0, 255) or, the zero
// variants, all zero; the depth ones 1.0.
void check_dummies() {
    struct One {
        VkImage image;
        std::uint32_t layers;
        bool depth;
        std::uint32_t want;  // the texel's bits
        std::string what;
    };
    std::vector<One> all = {{g.dummy_image, 1, false, 0xff000000u, "2D"}, {g.dummy_zero_image, 1, false, 0u, "2D zero"}};
    for (const auto& [key, image] : g.dummy_dim_images) {
        const std::uint32_t dim = (key & 0xff) >> 1;
        all.push_back({image, dim == 3 ? 6u : 1u, false, (key & 0x100) ? 0u : 0xff000000u,
                       "dim " + std::to_string(dim) + ((key & 1) ? " array" : "") + ((key & 0x100) ? " zero" : "")});
    }
    for (const auto& [key, image] : g.dummy_depth_images) {
        all.push_back({image, (key >> 1) == 3 ? 6u : 1u, true, 0x3f800000u, "depth dim " + std::to_string(key >> 1) + ((key & 1) ? " array" : "")});
    }
    std::uint32_t texels = 0;
    for (const One& o : all) texels += o.layers;
    DevBuffer rb;
    if (!create_dev_buffer(rb, texels * 4, true, true)) {
        host_log("gpu: dummy images checked: no readback buffer");
        return;
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g_cmd(), &bi);
    ++g.record_serial;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    std::uint32_t at = 0;
    for (const One& o : all) {
        for (std::uint32_t l = 0; l < o.layers; ++l, ++at) {
            VkBufferImageCopy c{};
            c.bufferOffset = at * 4;
            c.imageSubresource = {static_cast<VkImageAspectFlags>(o.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, l, 1};
            c.imageExtent = {1, 1, 1};
            vkCmdCopyImageToBuffer(g_cmd(), o.image, VK_IMAGE_LAYOUT_GENERAL, rb.buffer, 1, &c);
        }
    }
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(g_cmd());
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g.cmd_;
    vkResetFences(g.device, 1, &g.fence);
    {
        QueueGuard queue;
        vkQueueSubmit(g.queue, 1, &si, g.fence);
    }
    vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g.device, 1, &g.fence);
    std::uint32_t wrong = 0;
    std::string first;
    at = 0;
    for (const One& o : all) {
        for (std::uint32_t l = 0; l < o.layers; ++l, ++at) {
            std::uint32_t got;
            std::memcpy(&got, static_cast<const std::uint8_t*>(rb.map) + at * 4, 4);
            if (got == o.want) continue;
            ++wrong;
            if (first.empty()) {
                char b[160];
                std::snprintf(b, sizeof(b), "; first: %s layer %u holds %08x, not %08x", o.what.c_str(), l, got, o.want);
                first = b;
            }
        }
    }
    host_log("gpu: dummy images checked: %zu images, %u texels, %u wrong%s", all.size(), texels, wrong, first.c_str());
    vkDestroyBuffer(g.device, rb.buffer, nullptr);
    vkFreeMemory(g.device, rb.memory, nullptr);
}

bool create_dummy_resources() {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {1, 1, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                (g_test_dummies ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0u);
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g.device, &ici, nullptr, &g.dummy_image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, g.dummy_image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, 0);
    MemOwner own_mem("dummy images and buffers");
    if (vkAllocateMemory(g.device, &mai, nullptr, &g.dummy_mem) != VK_SUCCESS) return false;
    vkBindImageMemory(g.device, g.dummy_image, g.dummy_mem, 0);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = g.dummy_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ici.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g.device, &vci, nullptr, &g.dummy_view) != VK_SUCCESS) return false;
    if (vkCreateImage(g.device, &ici, nullptr, &g.dummy_zero_image) != VK_SUCCESS) return false;
    vkGetImageMemoryRequirements(g.device, g.dummy_zero_image, &req);
    mai.allocationSize = req.size;
    if (vkAllocateMemory(g.device, &mai, nullptr, &g.dummy_zero_mem) != VK_SUCCESS) return false;
    vkBindImageMemory(g.device, g.dummy_zero_image, g.dummy_zero_mem, 0);
    vci.image = g.dummy_zero_image;
    if (vkCreateImageView(g.device, &vci, nullptr, &g.dummy_zero_view) != VK_SUCCESS) return false;
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    if (vkCreateSampler(g.device, &sci, nullptr, &g.dummy_sampler) != VK_SUCCESS) return false;
    // Bound where a constant buffer could not be (cb_valid clear): the shader
    // does not read it then, but every declared binding needs a buffer.
    VkBufferCreateInfo sbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    sbi.size = 64;
    sbi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(g.device, &sbi, nullptr, &g.dummy_ssbo) != VK_SUCCESS) return false;
    vkGetBufferMemoryRequirements(g.device, g.dummy_ssbo, &req);
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, 0);
    if (vkAllocateMemory(g.device, &mai, nullptr, &g.dummy_ssbo_mem) != VK_SUCCESS) return false;
    vkBindBufferMemory(g.device, g.dummy_ssbo, g.dummy_ssbo_mem, 0);
    // Opaque black for missing T#s; zeros for an invalid T# (GCN sample of
    // type&8==0 returns 0). A=1 on that path scaled YEBIS blur to full strength.
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g_cmd(), &bi);
    ++g.record_serial;  // dynamic state recorded so far is gone
    VkImageMemoryBarrier b[2]{};
    for (int i = 0; i < 2; ++i) {
        b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].image = i ? g.dummy_zero_image : g.dummy_image;
        b[i].subresourceRange = vci.subresourceRange;
        b[i].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                         2, b);
    VkClearColorValue black{{0.0f, 0.0f, 0.0f, 1.0f}};
    VkClearColorValue zero{{0.0f, 0.0f, 0.0f, 0.0f}};
    vkCmdClearColorImage(g_cmd(), g.dummy_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &vci.subresourceRange);
    vkCmdClearColorImage(g_cmd(), g.dummy_zero_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &vci.subresourceRange);
    for (int i = 0; i < 2; ++i) {
        b[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }
    vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 2, b);
    vkEndCommandBuffer(g_cmd());
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g.cmd_;
    vkResetFences(g.device, 1, &g.fence);
    {
        QueueGuard queue;
        vkQueueSubmit(g.queue, 1, &si, g.fence);
    }
    vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g.device, 1, &g.fence);  // the slot's own: its first real submission wants it unsignaled
    create_dim_dummies();
    create_depth_dummies();
    if (g_test_dummies) check_dummies();
    return true;
}

}  // namespace

// A dword at a time: pipeline and view keys are hashed on every draw. The
// xorshift carries high bits down, since hash tables bucket on the low ones.
std::uint64_t fnv1a(const void* data, std::size_t n, std::uint64_t h) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        std::uint64_t v;
        std::memcpy(&v, p + i, 8);
        h = (h ^ v) * 1099511628211ull;
        h ^= h >> 32;
    }
    for (; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

std::uint32_t find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags want) {
    for (std::uint32_t i = 0; i < g.mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (g.mem_props.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

// Whether what finds video memory full goes to host memory (images, device
// buffers) and has the buffer shadow give its mirrors back.
// BBHOST_VRAM_FALLBACK=0: it fails, as before, for comparison.
bool vram_fallback_on() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_VRAM_FALLBACK");
        return !(e && e[0] == '0');
    }();
    return on;
}

// A memory type for `type_bits` outside video memory (a heap without
// DEVICE_LOCAL): where an image goes when video memory has no room for it.
std::uint32_t host_memory_type(std::uint32_t type_bits) {
    for (std::uint32_t i = 0; i < g.mem_props.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        const VkMemoryType& t = g.mem_props.memoryTypes[i];
        if (!(g.mem_props.memoryHeaps[t.heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) return i;
    }
    return UINT32_MAX;
}

bool create_dev_buffer(DevBuffer& b, std::uint64_t size, bool host_visible, bool cached) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    // Vertex and index usage too: a draw binds copies made in these buffers as
    // its vertex or index data - a copy version (GX's copy-back, in staging),
    // a YEBIS window's copy - and a binding without the usage is invalid
    // (VUID-vkCmdBindVertexBuffers-pBuffers-00627, seen on lavapipe with
    // window imports); indirect usage for a GX buffer's device-local twin
    // that holds a dispatch's or a draw's arguments.
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (vkCreateBuffer(g.device, &bci, nullptr, &b.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, b.buffer, &req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &flags;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, host_visible
                                                                   ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                                                   : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Readbacks the CPU reads whole: uncached (write-combined) memory reads at
    // a fraction of the speed - an F12 dump spent 1.4 s just copying out.
    if (host_visible && cached) {
        const std::uint32_t c = find_memory_type(
            req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (c != UINT32_MAX) mai.memoryTypeIndex = c;
    }
    if (mai.memoryTypeIndex == UINT32_MAX) return false;
    if (vkAllocateMemory(g.device, &mai, nullptr, &b.memory) != VK_SUCCESS) {
        // Video memory is full: the buffer in host memory, read across the
        // bus but there (and the buffer shadow gives its mirrors back).
        const std::uint32_t host = host_visible || !vram_fallback_on() ? UINT32_MAX : host_memory_type(req.memoryTypeBits);
        mai.memoryTypeIndex = host;
        if (host == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &b.memory) != VK_SUCCESS) {
            vkDestroyBuffer(g.device, b.buffer, nullptr);
            b.buffer = VK_NULL_HANDLE;
            return false;
        }
        b.in_host = true;
        g.vram_short.store(true, std::memory_order_relaxed);
    }
    vkBindBufferMemory(g.device, b.buffer, b.memory, 0);
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = b.buffer;
    b.address = vkGetBufferDeviceAddress(g.device, &bai);
    b.size = size;
    if (host_visible) {
        vkMapMemory(g.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map);
    }
    return true;
}

// Whether the rings the CPU writes and every draw's shaders read - the
// constant-buffer copies (render.cpp's scratch ring: 87% of the constant
// buffers bound in an RTX 5090's log) and the params ring - go in video
// memory the CPU can write, on a discrete GPU. In host memory (as on the
// integrated GPU they were measured on, where it is the same RAM) every
// shader's constant reads crossed the bus. BBHOST_RINGS_IN_VRAM=0: host memory.
// Each one goes there only while the heap's budget leaves room for it
// (ring_fits_budget); =1 puts them there whatever the budget says (a
// measurement).
bool rings_in_vram() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_RINGS_IN_VRAM");
        if (e && *e) return e[0] == '1';
        return !g.integrated;
    }();
    return on;
}

namespace {
const bool g_rings_forced = [] {
    const char* e = std::getenv("BBHOST_RINGS_IN_VRAM");
    return e && e[0] == '1';
}();
// The rings rings_in_vram offered to video memory the CPU writes: placed there,
// or left in host memory because the heap's budget had no room (the memory
// report's "rings:" line).
std::atomic<int> g_rings_vram{0}, g_rings_host{0};
std::atomic<std::uint64_t> g_rings_vram_bytes{0}, g_rings_host_bytes{0};
}  // namespace

void note_ring_placed(bool vram, VkDeviceSize bytes) {
    (vram ? g_rings_vram : g_rings_host).fetch_add(1, std::memory_order_relaxed);
    (vram ? g_rings_vram_bytes : g_rings_host_bytes).fetch_add(bytes, std::memory_order_relaxed);
}

bool ring_fits_budget(std::uint32_t type, VkDeviceSize bytes, const char* what, std::string* note) {
    if (type >= g.mem_props.memoryTypeCount) return false;
    const std::uint32_t h = g.mem_props.memoryTypes[type].heapIndex;
    const VkDeviceSize heap = g.mem_props.memoryHeaps[h].size;
    const bool have = g.has_memory_budget && g.phys;
    VkDeviceSize budget = 0, used = 0;
    if (have) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT mb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        mp.pNext = &mb;
        vkGetPhysicalDeviceMemoryProperties2(g.phys, &mp);
        budget = mb.heapBudget[h];
        used = mb.heapUsage[h];
    }
    const RingBudget r = ring_budget_fits(heap, have, budget, used, bytes);
    char buf[200];
    if (have) {
        std::snprintf(buf, sizeof(buf), "heap %u of %llu MiB: %llu of %llu MiB of its budget in use, %llu MiB kept free", h,
                      static_cast<unsigned long long>(heap >> 20), static_cast<unsigned long long>(used >> 20),
                      static_cast<unsigned long long>(budget >> 20), static_cast<unsigned long long>(r.keep >> 20));
    } else {
        std::snprintf(buf, sizeof(buf), "heap %u of %llu MiB, no budget from the driver", h, static_cast<unsigned long long>(heap >> 20));
    }
    if (note) *note = buf;
    if (r.fits || g_rings_forced) return true;
    host_log("gpu: %s (%llu MiB) in host memory, read across the bus: video memory the CPU writes has no room for it in its budget "
             "(%s)%s",
             what, static_cast<unsigned long long>(bytes >> 20), buf,
             have ? "" : " - a window under 1 GiB is shared with every other program");
    note_ring_placed(false, bytes);
    return false;
}

std::string rings_placement_report() {
    const int v = g_rings_vram.load(std::memory_order_relaxed), h = g_rings_host.load(std::memory_order_relaxed);
    if (!v && !h) return {};
    char buf[200];
    std::snprintf(buf, sizeof(buf), "gpu: rings: %d in video memory the CPU writes (%llu MiB), %d in host memory by the heap's budget (%llu MiB)%s",
                  v, static_cast<unsigned long long>(g_rings_vram_bytes.load(std::memory_order_relaxed) >> 20), h,
                  static_cast<unsigned long long>(g_rings_host_bytes.load(std::memory_order_relaxed) >> 20),
                  g_rings_forced ? " (BBHOST_RINGS_IN_VRAM=1: the budget not asked)" : "");
    return buf;
}

bool create_upload_buffer(DevBuffer& b, std::uint64_t size, const char* what) {
    if (rings_in_vram()) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(g.device, &bci, nullptr, &b.buffer) == VK_SUCCESS) {
            VkMemoryRequirements req{};
            vkGetBufferMemoryRequirements(g.device, b.buffer, &req);
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.pNext = &flags;
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            std::string note;
            const bool room = mai.memoryTypeIndex != UINT32_MAX && ring_fits_budget(mai.memoryTypeIndex, req.size, what, &note);
            if (room && vkAllocateMemory(g.device, &mai, nullptr, &b.memory) == VK_SUCCESS) {
                vkBindBufferMemory(g.device, b.buffer, b.memory, 0);
                VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
                bai.buffer = b.buffer;
                b.address = vkGetBufferDeviceAddress(g.device, &bai);
                b.size = size;
                vkMapMemory(g.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map);
                note_ring_placed(true, req.size);
                host_log("gpu: %s in video memory the CPU writes (memory type %u, %s before it; BBHOST_RINGS_IN_VRAM=0: host memory)", what,
                         mai.memoryTypeIndex, note.c_str());
                return true;
            }
            vkDestroyBuffer(g.device, b.buffer, nullptr);
            b.buffer = VK_NULL_HANDLE;
            if (mai.memoryTypeIndex == UINT32_MAX || room) {
                host_log("gpu: %s: no room in video memory the CPU writes (%s); in host memory, read across the bus", what,
                         mai.memoryTypeIndex == UINT32_MAX ? "the GPU has none" : "it is full");
                if (room) note_ring_placed(false, req.size);
            }
        }
    }
    return create_dev_buffer(b, size, true);
}

std::atomic<std::size_t> g_pipeline_cache_bytes{0};  // the cache's size when last loaded or saved
std::atomic<bool> g_cache_divert{false};             // PipelineCacheUse (gpu_internal.h)
std::atomic<int> g_cache_users{0};
std::atomic<std::uint64_t> g_cache_diverted{0};

// <data root>/bbhost/vulkan-pipeline-cache.bin, or empty when there is no data
// root or BBHOST_PIPELINE_CACHE=0.
std::string pipeline_cache_path() {
    static const bool enabled = [] {
        const char* e = std::getenv("BBHOST_PIPELINE_CACHE");
        return !(e && e[0] == '0');
    }();
    const char* root = hle_fs_data_root();
    if (!enabled || !root || !*root) return {};
    return std::string(root) + "/bbhost/vulkan-pipeline-cache.bin";
}

// <data root>/bbhost/stage-manifest.bin, beside the pipeline cache and off with it.
std::string stage_manifest_path() {
    std::string path = pipeline_cache_path();
    if (path.empty()) return path;
    return path.substr(0, path.rfind('/') + 1) + "stage-manifest.bin";
}

// <data root>/bbhost/pipeline-manifest.bin, the same.
std::string pipeline_manifest_path() {
    std::string path = pipeline_cache_path();
    if (path.empty()) return path;
    return path.substr(0, path.rfind('/') + 1) + "pipeline-manifest.bin";
}

// <data root>/bbhost/translation-cache.bin (host/translation_cache.h), the same.
std::string translation_cache_path() {
    std::string path = pipeline_cache_path();
    if (path.empty()) return path;
    return path.substr(0, path.rfind('/') + 1) + "translation-cache.bin";
}

// The shader caches belong to the translator that made them. A translation is
// keyed by the program, every option and switch it was translated with and the
// translator's fingerprint - its sources and the cache's own key and entry
// code - and a pipeline by the driver, by its shaders and state; so an update
// that leaves the translator alone keeps both and starts with every shader
// built. Another translator - <data root>/bbhost/shader-cache-build.txt names
// the one that made them - starts both empty: its translations would not be
// taken anyway, and the pipelines its shaders replace would stay in the
// pipeline cache, which the driver keeps whole (one translator's are 25 MiB on
// the Steam Deck to 100 MiB on NVIDIA; the file grew past 400 MiB over a few
// weeks of builds). The stage manifest stays: it holds the guest's inputs,
// which the running build translates again on the precompile workers.
// BBHOST_KEEP_SHADER_CACHE=1 keeps them across translators.
static void shader_caches_check_build() {
    const std::string pipelines = pipeline_cache_path();
    if (pipelines.empty()) return;
    const std::string dir = pipelines.substr(0, pipelines.rfind('/') + 1);
    const std::string stamp = dir + "shader-cache-build.txt";
    const std::string ours = translator_fingerprint();
    std::string theirs;
    {
        std::ifstream in(stamp);
        std::getline(in, theirs);
    }
    if (theirs == ours) return;
    std::error_code ec;
    if (const char* e = std::getenv("BBHOST_KEEP_SHADER_CACHE"); e && e[0] == '1') {
        host_log("gpu: shader caches made by translator %s kept for translator %s (BBHOST_KEEP_SHADER_CACHE=1)",
                 theirs.empty() ? "of an earlier build" : theirs.c_str(), ours.c_str());
    } else {
        std::uintmax_t freed = 0;
        int removed = 0;
        for (const std::string& path : {pipelines, translation_cache_path()}) {
            const std::uintmax_t bytes = std::filesystem::file_size(path, ec);
            if (ec) continue;
            if (std::filesystem::remove(path, ec)) freed += bytes, ++removed;
        }
        if (removed)
            host_log("gpu: shader caches made by translator %s, this is translator %s: cleared (%.1f MiB); the precompile builds them again",
                     theirs.empty() ? "of an earlier build" : theirs.c_str(), ours.c_str(), freed / 1048576.0);
    }
    std::filesystem::create_directories(dir, ec);
    std::ofstream(stamp, std::ios::trunc) << ours << '\n';
}

// Rebuild the guest page tables when the mapping list changed.
bool rebuild_page_tables() {
    static std::uint64_t seen_gen = 0;
    const std::uint64_t gen = hle_kernel_maps_generation();
    if (gen == seen_gen) {
        return true;
    }
    // A run part's draw (BBHOST_RUN_COMMIT): rebuilt in order, by the command
    // processor's commit of it.
    if (run_part_stop(kRunStopTables)) return false;
    seen_gen = gen;
    std::vector<GuestMapInfo> maps;
    hle_kernel_snapshot_maps(maps);
    std::uint64_t h = 1469598103934665603ull;
    for (const GuestMapInfo& mi : maps) {
        h = fnv1a(&mi.va, 8, h);
        h = fnv1a(&mi.len, 8, h);
        h = fnv1a(&mi.phys, 8, h);
        h = fnv1a(&mi.prot, 4, h);
        h = fnv1a(&mi.dmem, 1, h);
    }
    if (h == g.maps_hash) {
        return true;
    }
    g.maps_hash = h;
    g.maps = maps;
    for (auto& kv : g.l2) fill_sink(kv.second);
    const std::uint64_t high = hle_kernel_dmem_high();
    std::size_t pages = 0, skipped = 0;
    // Each chunk's GPU-visible span: what may be imported (the whole of it
    // now with BBHOST_IMPORT_WINDOWS=0, so the pages below find it).
    for (std::uint32_t k = 0; k < gcn::kDmemChunks; ++k) {
        std::uint64_t lo = 0, hi = 0;
        DmemChunk& d = g.dmem[k];
        if (static_cast<std::uint64_t>(k) * kChunkBytes >= high + kChunkBytes || !gpu_span(maps, k, lo, hi)) {
            d.span_lo = d.span_hi = 0;
            continue;
        }
        d.span_lo = lo;
        d.span_hi = hi;
        if (imports_whole() && !(d.window[lo >> kImportWindowShift] && chunk_covers(*d.window[lo >> kImportWindowShift], lo, hi - lo))) {
            // A span that runs to the chunk's end, where the next chunk's
            // starts, goes on into it (kChunkOverlap): a buffer the game placed
            // across the seam - a vertex buffer, a constant array - then lies
            // in one import. Without it such a draw kept its fetch shader and
            // walked the page table for every vertex: on the Deck, where the
            // smaller targets moved the heap, 16 G-buffer draws a frame cost
            // 2-6 times as much GPU time as their neighbours.
            // BBHOST_CHUNK_OVERLAP=0: each import stops at its chunk's end.
            static const bool overlap = [] {
                const char* e = std::getenv("BBHOST_CHUNK_OVERLAP");
                return !(e && e[0] == '0');
            }();
            std::uint64_t ext = hi;
            std::uint64_t nlo = 0, nhi = 0;
            if (overlap && hi == kChunkBytes && k + 1 < gcn::kDmemChunks && gpu_span(maps, k + 1, nlo, nhi) && nlo == 0) {
                ext = kChunkBytes + std::min<std::uint64_t>(nhi, kChunkOverlap);
            }
            const int who = t_import_who;
            t_import_who = kIwSpan;
            if (Chunk* c = import_span(k, lo, ext)) {
                for (std::uint64_t w = lo >> kImportWindowShift; w <= (hi - 1) >> kImportWindowShift; ++w) d.window[w] = c;
            }
            t_import_who = who;
        }
    }
    for (const GuestMapInfo& mi : maps) {
        // A page the GPU cannot reach on the console stays out (it reads the
        // sink): direct memory outside every chunk's import (the CPU-only
        // heaps - a GPU alias of imported memory still maps), and flexible
        // mappings without GPU bits (ours from hle_kernel_map_host have them).
        if (!(mi.prot & 1) || mi.len == 0) continue;
        if (!mi.dmem && !(mi.prot & kGuestProtGpu)) continue;
        const std::uint64_t page = 1ull << gcn::kPageShift;
        if ((mi.va & (page - 1)) || (mi.len & (page - 1))) {
            ++skipped;
            continue;
        }
        std::uint64_t anon_base = 0;
        if (!mi.dmem) {
            const Chunk* c = import_anon(mi.va, mi.len);
            if (!c) {
                ++skipped;
                continue;
            }
            anon_base = c->address;
        }
        for (std::uint64_t off = 0; off < mi.len; off += page) {
            const std::uint64_t va = mi.va + off;
            const std::uint32_t i1 = static_cast<std::uint32_t>(va >> 32);
            if (i1 >= gcn::kL1Entries) break;
            std::uint64_t dev = 0;
            if (mi.dmem) {
                const std::uint64_t phys = static_cast<std::uint64_t>(mi.phys) + off;
                const std::uint32_t chunk = static_cast<std::uint32_t>(phys >> gcn::kDmemChunkShift);
                if (chunk >= gcn::kDmemChunks || phys >= high + kChunkBytes) break;
                const std::uint64_t in = phys & (kChunkBytes - 1);
                const Chunk* c = g.dmem[chunk].window[in >> kImportWindowShift];
                if (!c || !chunk_covers(*c, in, 1)) continue;  // not imported (yet): the sink
                dev = c->address + (in - c->lo);
            } else {
                dev = anon_base + off;
            }
            if (!set_page_locked(va, dev)) return false;
            ++pages;
        }
    }
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 12) {
        host_log("gpu: page tables rebuilt: %zu mappings, %zu pages of 64 KiB, %zu skipped, %zu L2 tables", maps.size(),
                 pages, skipped, g.l2.size());
    }
    return true;
}

// BBHOST_IMPORT_AUDIT=1: every guest range the GPU side resolves to imported
// memory (bindings and copies through locate, device addresses for the GPU
// untile, the vertex/index mirror's sources), counted per guest mapping and
// printed with the 300-flip memory lines - which mappings the GPU really
// reads, to decide what has to be imported at all.
const bool g_import_audit = [] {
    const char* e = std::getenv("BBHOST_IMPORT_AUDIT");
    return e && e[0] == '1';
}();
struct AuditEntry {
    std::atomic<std::uint64_t> va{0};
    std::uint64_t len = 0, phys = 0;
    int prot = 0;
    std::atomic<std::uint64_t> hits[3] = {};  // locate, device address, mirror source
};
AuditEntry g_audit[64];

void import_audit(const GuestMapInfo& mi, int kind) {
    for (AuditEntry& e : g_audit) {
        std::uint64_t cur = e.va.load(std::memory_order_relaxed);
        if (cur == 0) {
            if (e.va.compare_exchange_strong(cur, mi.va)) {
                e.len = mi.len;
                e.phys = mi.phys < 0 ? ~0ull : static_cast<std::uint64_t>(mi.phys);
                e.prot = mi.prot;
            } else if (cur != mi.va) {
                continue;
            }
        } else if (cur != mi.va) {
            continue;
        }
        e.hits[kind].fetch_add(1, std::memory_order_relaxed);
        return;
    }
}

// The direct-memory pages (64 KiB) the GPU side resolved ranges in, by kind
// (ImportUse): how much of the import the GPU reads at all, per 64 MiB window.
constexpr std::uint64_t kAuditPages = static_cast<std::uint64_t>(gcn::kDmemChunks) * (kChunkBytes >> gcn::kPageShift);
std::atomic<std::uint64_t> g_audit_pages[kImportUses][kAuditPages / 64];
std::atomic<std::uint64_t> g_audit_flex_bytes[kImportUses];

// BBHOST_LABEL_READ_CHECK=1 (with BBHOST_IMPORT_AUDIT=1 and
// BBHOST_LABEL_LANDING=1): every guest range the GPU side resolves
// (audit_range: bindings, device addresses, the mirror's sources, transfers,
// index and indirect reads) against the labels the host writes. The landing
// rests on no GPU work reading one: such a read would see the label only once
// the host has written it, whatever the queue's order says.
const bool g_label_check = [] {
    const char* e = std::getenv("BBHOST_LABEL_READ_CHECK");
    return e && e[0] == '1';
}();
std::mutex g_label_check_mu;
struct LabelDword {
    std::uint32_t off;     // in its 4 KiB page
    std::uint32_t value;   // the last written
    std::uint64_t serial;  // the submission that carries the last write
};
std::unordered_map<std::uint64_t, std::vector<LabelDword>> g_label_pages;  // 4 KiB page -> its label dwords
std::uint64_t g_label_hits[kImportUses] = {}, g_label_hits_waiting[kImportUses] = {};
std::uint64_t g_label_dwords = 0;

void label_check_note_locked(std::uint64_t va, const void* data, std::uint64_t bytes, std::uint64_t serial) {
    if (!g_label_check) return;
    std::lock_guard<std::mutex> lk(g_label_check_mu);
    for (std::uint64_t a = va; a < va + bytes; a += 4) {
        std::vector<LabelDword>& dws = g_label_pages[a >> 12];
        const std::uint32_t off = static_cast<std::uint32_t>(a & 0xfff);
        std::uint32_t v = 0;
        std::memcpy(&v, static_cast<const std::uint8_t*>(data) + (a - va), 4);
        auto it = std::find_if(dws.begin(), dws.end(), [off](const LabelDword& d) { return d.off == off; });
        if (it == dws.end()) {
            dws.push_back({off, v, serial});
            ++g_label_dwords;
        } else {
            it->value = v;
            it->serial = serial;
        }
    }
}

void label_check_range(std::uint64_t va, std::uint64_t bytes, ImportUse use) {
    // (A one-byte mirror range is the buffer shadow told of a write -
    // shadow_written_locked's lookup of the page, a label's own among them -
    // not a read.)
    if (!g_label_check || !bytes || (use == kUseMirror && bytes == 1)) return;
    std::lock_guard<std::mutex> lk(g_label_check_mu);
    if (g_label_pages.empty()) return;
    for (std::uint64_t p = va >> 12; p <= (va + bytes - 1) >> 12; ++p) {
        const auto it = g_label_pages.find(p);
        if (it == g_label_pages.end()) continue;
        for (const LabelDword& d : it->second) {
            const std::uint64_t a = (p << 12) | d.off;
            if (a + 4 <= va || a >= va + bytes) continue;
            // Its landing still to come, from an earlier submission: a read
            // recorded now may run before it lands, where after a GPU fill it
            // would have read it. (One in this same submission it reads as
            // it read a fill: the fills went at the command buffer's end.)
            const bool waiting = d.serial >= label_landing::landed() && d.serial < g.flushes;
            static const char* const names[kImportUses] = {"bound", "pagetable", "untile", "mirror", "transfer", "index"};
            if (std::accumulate(std::begin(g_label_hits), std::end(g_label_hits), std::uint64_t{0}) < 48) {
                host_log("gpu: label check: a %s range 0x%llx..0x%llx (%llu bytes) holds the label at 0x%llx (+0x%llx) the host writes: "
                         "last %u, in submission %llu, %s (now recording %llu, flip %llu)",
                         names[use], static_cast<unsigned long long>(va), static_cast<unsigned long long>(va + bytes),
                         static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(a), static_cast<unsigned long long>(a - va),
                         d.value, static_cast<unsigned long long>(d.serial),
                         waiting ? "still to land" : d.serial >= g.flushes ? "in this submission" : "landed",
                         static_cast<unsigned long long>(g.flushes), static_cast<unsigned long long>(hle_video_flip_count()));
            }
            ++g_label_hits[use];
            if (waiting) ++g_label_hits_waiting[use];
            return;
        }
    }
}

std::string label_check_report() {
    if (!g_label_check) return {};
    std::lock_guard<std::mutex> lk(g_label_check_mu);
    char b[480];
    using ull = unsigned long long;
    std::snprintf(b, sizeof(b),
                  "label check (BBHOST_LABEL_READ_CHECK=1%s): %llu label dwords the host writes in %zu pages; GPU-side ranges over one "
                  "(of them with its landing from an earlier submission still to come): bound %llu (%llu), page table %llu (%llu), untile %llu (%llu), mirror %llu "
                  "(%llu), transfer %llu (%llu), index/indirect %llu (%llu)",
                  g_import_audit ? "" : ", without BBHOST_IMPORT_AUDIT=1 nothing is checked", static_cast<ull>(g_label_dwords),
                  g_label_pages.size(), static_cast<ull>(g_label_hits[kUseBound]), static_cast<ull>(g_label_hits_waiting[kUseBound]),
                  static_cast<ull>(g_label_hits[kUsePageTable]), static_cast<ull>(g_label_hits_waiting[kUsePageTable]),
                  static_cast<ull>(g_label_hits[kUseUntile]), static_cast<ull>(g_label_hits_waiting[kUseUntile]),
                  static_cast<ull>(g_label_hits[kUseMirror]), static_cast<ull>(g_label_hits_waiting[kUseMirror]),
                  static_cast<ull>(g_label_hits[kUseTransfer]), static_cast<ull>(g_label_hits_waiting[kUseTransfer]),
                  static_cast<ull>(g_label_hits[kUseIndex]), static_cast<ull>(g_label_hits_waiting[kUseIndex]));
    return b;
}

void audit_range(std::uint64_t va, std::uint64_t bytes, ImportUse use) {
    if (!bytes) return;
    label_check_range(va, bytes, use);
    const std::uint64_t end = va + bytes;
    for (const GuestMapInfo& mi : g.maps) {
        const std::uint64_t lo = std::max(va, mi.va), hi = std::min(end, mi.va + mi.len);
        if (lo >= hi) continue;
        if (!mi.dmem) {
            g_audit_flex_bytes[use].fetch_add(hi - lo, std::memory_order_relaxed);
            continue;
        }
        const std::uint64_t p0 = (static_cast<std::uint64_t>(mi.phys) + (lo - mi.va)) >> gcn::kPageShift;
        const std::uint64_t p1 = (static_cast<std::uint64_t>(mi.phys) + (hi - mi.va) - 1) >> gcn::kPageShift;
        for (std::uint64_t p = p0; p <= p1 && p < kAuditPages; ++p) {
            const std::uint64_t bit = 1ull << (p & 63);
            std::atomic<std::uint64_t>& w = g_audit_pages[use][p >> 6];
            if (!(w.load(std::memory_order_relaxed) & bit)) w.fetch_or(bit, std::memory_order_relaxed);
        }
    }
}

std::string audit_pages_report() {
    static const char* const names[kImportUses] = {"bound", "pagetable", "untile", "mirror", "transfer", "index"};
    std::string out = "import pages, MiB:";
    char b[160];
    std::vector<std::uint64_t> all(kAuditPages / 64, 0);
    for (int u = 0; u < kImportUses; ++u) {
        std::uint64_t pages = 0;
        for (std::uint64_t k = 0; k < kAuditPages / 64; ++k) {
            const std::uint64_t w = g_audit_pages[u][k].load(std::memory_order_relaxed);
            pages += static_cast<std::uint64_t>(__builtin_popcountll(w));
            all[k] |= w;
        }
        std::snprintf(b, sizeof(b), " %s %llu+%llu", names[u], static_cast<unsigned long long>(pages << gcn::kPageShift >> 20),
                      static_cast<unsigned long long>(g_audit_flex_bytes[u].load() >> 20));
        out += b;
    }
    // The union, and what importing by 64 MiB / 2 MiB windows would need.
    std::uint64_t pages = 0, win64 = 0, win2 = 0;
    constexpr std::uint64_t kPer64 = (64ull << 20) >> gcn::kPageShift, kPer2 = (2ull << 20) >> gcn::kPageShift;
    for (std::uint64_t p = 0; p < kAuditPages; p += kPer2) {
        bool any = false;
        for (std::uint64_t q = p; q < p + kPer2; ++q) {
            if ((all[q >> 6] >> (q & 63)) & 1) {
                ++pages;
                any = true;
            }
        }
        if (any) ++win2;
    }
    for (std::uint64_t p = 0; p < kAuditPages; p += kPer64) {
        for (std::uint64_t q = p; q < p + kPer64; ++q) {
            if ((all[q >> 6] >> (q & 63)) & 1) {
                ++win64;
                break;
            }
        }
    }
    // Each touched 64 MiB window of direct memory: its index (physical / 64 MiB)
    // and the uses that touched it, one letter each (b p u m t i), with how many
    // MiB of its pages they touched in all.
    std::string wins;
    for (std::uint64_t p = 0; p < kAuditPages; p += kPer64) {
        std::uint64_t n = 0;
        std::string uses;
        for (int u = 0; u < kImportUses; ++u) {
            bool hit = false;
            for (std::uint64_t q = p; q < p + kPer64; ++q) {
                if ((g_audit_pages[u][q >> 6].load(std::memory_order_relaxed) >> (q & 63)) & 1) {
                    hit = true;
                    break;
                }
            }
            if (hit) uses += "bpumti"[u];
        }
        for (std::uint64_t q = p; q < p + kPer64; ++q) n += (all[q >> 6] >> (q & 63)) & 1;
        if (!uses.empty()) wins += " " + std::to_string(p / kPer64) + ":" + uses + ":" + std::to_string((n << gcn::kPageShift) >> 20);
    }
    if (std::getenv("BBHOST_IMPORT_AUDIT_WINDOWS")) {
        for (std::size_t at = 0; at < wins.size(); at += 900) {
            host_log("import windows (64 MiB index:uses:MiB touched):%s", wins.substr(at, 900).c_str());
        }
    }
    std::uint64_t imported = 0;
    for (const DmemChunk& d : g.dmem) {
        for (std::uint32_t w = 0; w < kImportWindows; ++w) {
            if (const Chunk* c = d.window[w]) {
                const std::uint64_t lo = std::max(c->lo, static_cast<std::uint64_t>(w) << kImportWindowShift);
                const std::uint64_t hi = std::min(c->lo + c->size, static_cast<std::uint64_t>(w + 1) << kImportWindowShift);
                imported += hi > lo ? hi - lo : 0;
            }
        }
    }
    std::snprintf(b, sizeof(b), "; all %llu, by 2 MiB %llu, by 64 MiB %llu; imported %llu (+N: flexible MiB resolved, not distinct)",
                  static_cast<unsigned long long>(pages << gcn::kPageShift >> 20), static_cast<unsigned long long>(win2 * 2),
                  static_cast<unsigned long long>(win64 * 64), static_cast<unsigned long long>(imported >> 20));
    out += b;
    return out;
}

std::string import_audit_report() {
    std::string out = "import audit (mapping va/len/prot/phys: locate, device address, mirror source):";
    char b[160];
    for (const AuditEntry& e : g_audit) {
        const std::uint64_t va = e.va.load();
        if (!va) continue;
        std::snprintf(b, sizeof(b), " 0x%llx/%lluMiB/0x%x/%s: %llu, %llu, %llu;", static_cast<unsigned long long>(va),
                      static_cast<unsigned long long>(e.len >> 20), e.prot,
                      e.phys == ~0ull ? "flex" : std::to_string(e.phys >> 20).append("MiB").c_str(),
                      static_cast<unsigned long long>(e.hits[0].load()), static_cast<unsigned long long>(e.hits[1].load()),
                      static_cast<unsigned long long>(e.hits[2].load()));
        out += b;
    }
    return out;
}

namespace {
Located located_in(const GuestMapInfo& mi, std::uint64_t va, std::uint64_t bytes) {
    if (g_import_audit) import_audit(mi, 0);
    Located out;
    const std::uint64_t off = va - mi.va;
    if (mi.dmem) {
        const std::uint64_t phys = static_cast<std::uint64_t>(mi.phys) + off;
        const std::uint32_t chunk = static_cast<std::uint32_t>(phys >> gcn::kDmemChunkShift);
        const std::uint64_t in = phys & (kChunkBytes - 1);
        // Window imports may run on across the seam into the next chunk (span_reach).
        const std::uint64_t reach = imports_whole() ? kChunkBytes : span_reach(chunk);
        t_import_va = va;
        const Chunk* c = dmem_import(chunk, in, std::min<std::uint64_t>({bytes, mi.len - off, reach > in ? reach - in : 1}), true);
        t_import_va = 0;
        if (!c) return out;
        note_site_use_locked(bytes);
        out.buffer = c->buffer;
        out.offset = in - c->lo;
        out.avail = std::min(mi.len - off, c->size - out.offset);
    } else {
        auto it = g.anon_imports.find(mi.va);
        if (it == g.anon_imports.end()) return out;
        out.buffer = it->second.buffer;
        out.offset = off;
        out.avail = mi.len - off;
    }
    return out;
}
}  // namespace

Located locate(std::uint64_t va, std::uint64_t bytes) {
    // A GX buffer the GPU writes, kept as a device-local twin (BBHOST_GX_TWINS=1).
    if (twins_on()) {
        Located t;
        if (twin_locate_locked(va, bytes, t)) return t;
    }
    return locate_guest(va, bytes);
}

Located locate_guest(std::uint64_t va, std::uint64_t bytes) {
    // The mapping this thread's last lookup found: most land in the same one
    // (the big direct-memory mapping), and one range check over a line
    // already in cache beats the walk. Mappings do not overlap, so whichever
    // holds `va` is the one the walk would find.
    thread_local std::size_t t_last = 0;
    if (t_last < g.maps.size()) {
        const GuestMapInfo& mi = g.maps[t_last];
        if (va >= mi.va && va < mi.va + mi.len) return located_in(mi, va, bytes);
    }
    for (std::size_t k = 0; k < g.maps.size(); ++k) {
        const GuestMapInfo& mi = g.maps[k];
        if (va >= mi.va && va < mi.va + mi.len) {
            t_last = k;
            return located_in(mi, va, bytes);
        }
    }
    return {};
}

// BBHOST_HANG_WATCHDOG=<seconds>: when the game stops flipping for that long,
// say once what the renderer is in the middle of. A frozen frame is otherwise
// silent: the log stops with it.
void start_hang_watchdog() {
    static const double after = [] {
        const char* e = std::getenv("BBHOST_HANG_WATCHDOG");
        return e ? std::atof(e) : 0.0;
    }();
    if (after <= 0) return;
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread([] {
            host_thread_set_name("bb-hangdog");
            std::uint64_t seen = 0;
            auto since = std::chrono::steady_clock::now();
            bool said = false;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                const std::uint64_t flips = hle_video_flip_count();
                const auto now = std::chrono::steady_clock::now();
                if (flips != seen) {
                    seen = flips;
                    since = now;
                    said = false;
                    continue;
                }
                const double still = std::chrono::duration<double>(now - since).count();
                if (said || still < after) continue;
                said = true;
                const int slot = g.waiting_slot.load(std::memory_order_relaxed);
                double waited = 0;
                if (slot >= 0) {
                    const std::chrono::steady_clock::time_point t0{
                        std::chrono::steady_clock::duration(g.wait_started_ns.load(std::memory_order_relaxed))};
                    waited = std::chrono::duration<double>(now - t0).count();
                }
                double present_for = 0;
                const char* const step = host_present_step(present_for);
                host_log("hang: the presenting thread is %s (%.1f s)", step, present_for);
                void* const site = g.mu.holder();
                host_log("hang: the renderer's lock is held by %s (taken %llu times)", GpuMutex::name_of(site).c_str(),
                         static_cast<unsigned long long>(g.mu.acquires()));
                host_log("hang: no flip for %.1f s at flip %llu; renderer %s, %u params queued, slot %d%s%.1f s; "
                         "draws %llu (failed %llu), submissions %llu",
                         still, static_cast<unsigned long long>(flips), g.recording ? "recording" : "idle", g.queued, g.slot,
                         slot >= 0 ? ", waiting on the GPU for " : ", not waiting on the GPU, ", waited,
                         static_cast<unsigned long long>(g.draws.load()), static_cast<unsigned long long>(g.draw_failures.load()),
                         static_cast<unsigned long long>(g.flushes));
                host_gpu_hang_report();
                hle_guest_pool_report();
            }
        }).detach();
    });
}

void start_memory_reserve();  // below, beside the image heap
void vram_reserve_init();     // the same

bool g_use_buffer_marker = false;  // VK_AMD_buffer_marker checkpoints (gpu_checkpoint)

// Every device that can run the host, scored: a discrete GPU before an
// integrated one (a laptop lists its on-board GPU first, and the first
// usable device used to be taken), then the most device-local memory.
// BBHOST_GPU_DEVICE=<index or part of the name> picks one instead.
struct Candidate {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    std::uint32_t family = 0, queues = 0;
    bool sparse = false, maint8 = false, swapchain = false, push = false, fault = false, checkpoints = false, buffer_marker = false;
    bool memory_fd = false, dma_buf = false;
    int type_rank = 0;
    VkDeviceSize local = 0;
    VkDeviceSize heaps = 0;  // every heap, device-local or not (memory_tight)
    // What the heap of its first device-local type has free for this process
    // now: the driver's budget less its usage (VK_EXT_memory_budget), else the
    // heap's size (small_vram).
    VkDeviceSize local_free = 0;
};

// The usable devices of `instance` and the index of the one to run on in
// `usable` (-1: none); each device is logged when `say`.
int choose_device(VkInstance instance, bool say, std::vector<Candidate>& usable) {
    usable.clear();
    std::uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(instance, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(instance, &ndev, devs.data());
    const char* want = std::getenv("BBHOST_GPU_DEVICE");
    int chosen = -1;
    for (std::uint32_t index = 0; index < devs.size(); ++index) {
        VkPhysicalDevice d = devs[index];
        std::uint32_t next = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &next, nullptr);
        std::vector<VkExtensionProperties> exts(next);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &next, exts.data());
        Candidate c;
        c.device = d;
        bool host_import = false, memory_budget = false;
        for (const VkExtensionProperties& e : exts) {
            if (std::strcmp(e.extensionName, "VK_EXT_external_memory_host") == 0) host_import = true;
            if (std::strcmp(e.extensionName, "VK_EXT_memory_budget") == 0) memory_budget = true;
            if (std::strcmp(e.extensionName, "VK_KHR_maintenance8") == 0) c.maint8 = true;
            if (std::strcmp(e.extensionName, "VK_KHR_push_descriptor") == 0) c.push = true;
            if (std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) c.swapchain = true;
            if (std::strcmp(e.extensionName, "VK_EXT_device_fault") == 0) c.fault = true;
            if (std::strcmp(e.extensionName, "VK_NV_device_diagnostic_checkpoints") == 0) c.checkpoints = true;
            if (std::strcmp(e.extensionName, "VK_AMD_buffer_marker") == 0) c.buffer_marker = true;
            if (std::strcmp(e.extensionName, "VK_KHR_external_memory_fd") == 0) c.memory_fd = true;
            if (std::strcmp(e.extensionName, "VK_EXT_external_memory_dma_buf") == 0) c.dma_buf = true;
        }
        VkPhysicalDeviceProperties dp{};
        vkGetPhysicalDeviceProperties(d, &dp);
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(d, &mp);
        for (std::uint32_t h = 0; h < mp.memoryHeapCount; ++h) {
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) c.local += mp.memoryHeaps[h].size;
            c.heaps += mp.memoryHeaps[h].size;
        }
        for (std::uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
            if (!(mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
            const std::uint32_t h = mp.memoryTypes[t].heapIndex;
            c.local_free = mp.memoryHeaps[h].size;
            if (memory_budget) {
                VkPhysicalDeviceMemoryBudgetPropertiesEXT mb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
                VkPhysicalDeviceMemoryProperties2 mp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
                mp2.pNext = &mb;
                vkGetPhysicalDeviceMemoryProperties2(d, &mp2);
                if (mb.heapBudget[h]) c.local_free = std::min(c.local_free, mb.heapBudget[h] - std::min(mb.heapBudget[h], mb.heapUsage[h]));
            }
            break;
        }
        c.type_rank = dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 4
                      : dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3
                      : dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 2
                      : dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU            ? 1
                                                                                  : 0;
        static const char* const kType[5] = {"other", "CPU", "virtual", "integrated", "discrete"};
        const char* missing = nullptr;
        if (!host_import) missing = "no VK_EXT_external_memory_host";
        else if (dp.apiVersion < VK_API_VERSION_1_3) missing = "Vulkan below 1.3";
        if (!missing) {
            std::uint32_t nq = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
            std::vector<VkQueueFamilyProperties> qs(nq);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
            missing = "no graphics+compute queue";
            for (std::uint32_t i = 0; i < nq; ++i) {
                if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                    c.family = i;
                    c.queues = qs[i].queueCount;
                    c.sparse = (qs[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0;
                    missing = nullptr;
                    break;
                }
            }
        }
        if (say)
            host_log("gpu: device %u: %s (%s, %llu MiB device-local)%s%s", index, dp.deviceName, kType[c.type_rank],
                     static_cast<unsigned long long>(c.local >> 20), missing ? " - unusable: " : "", missing ? missing : "");
        if (missing) continue;
        if (want && *want) {
            char* end = nullptr;
            const long n = std::strtol(want, &end, 10);
            std::string lname(dp.deviceName), lwant(want);
            for (char& ch : lname) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            for (char& ch : lwant) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            // A number names the device's index alone: as a part of the name
            // "0" matched the "3070" of an RTX 3070 Ti and picked it over
            // device 0.
            const bool number = end && end != want && *end == 0;
            if (number ? n == static_cast<long>(index) : lname.find(lwant) != std::string::npos) chosen = static_cast<int>(usable.size());
        }
        usable.push_back(c);
    }
    if (usable.empty()) return -1;
    if (chosen < 0) {
        if (want && *want && say) host_log("gpu: BBHOST_GPU_DEVICE=%s matches no usable device; choosing by type and memory", want);
        chosen = 0;
        for (std::size_t k = 1; k < usable.size(); ++k) {
            const Candidate& a = usable[k];
            const Candidate& best = usable[static_cast<std::size_t>(chosen)];
            if (a.type_rank > best.type_rank || (a.type_rank == best.type_rank && a.local > best.local)) chosen = static_cast<int>(k);
        }
    }
    return chosen;
}

// Whether the device has little memory for everything it holds: under 12 GiB
// in all its heaps together. Every buffer the host makes has to fit in them
// at once (RADV keeps every allocation resident, for device addresses), and
// direct memory's imports alone are ~4.7 GiB: a Steam Deck (1 GiB of VRAM,
// 8 GiB of GTT) failed its submissions in the world. BBHOST_TIGHT_MEMORY=0/1
// says instead.
bool memory_tight(const Candidate& c) {
    const char* e = std::getenv("BBHOST_TIGHT_MEMORY");
    if (e && *e) return e[0] != '0';
    return c.heaps < (12ull << 30);
}
std::atomic<int> g_tight{-1};  // memory_tight of the device run on, -1 until known

// Whether the device is a discrete GPU with little video memory free for us:
// under 7.5 GiB at the start (BBHOST_TEST_VRAM_MB's limit when a smaller card
// is acted out). Then the settings that hold memory ahead are cut - spare
// image heap blocks, kept untile scratch, display buffers for sizes the run
// may never use - on top of the reserve and the live mirror share (the
// tessellation ring grows as it is needed on every card). Not on an
// integrated GPU or with tight memory, which have settings of their own (a
// Steam Deck). BBHOST_SMALL_VRAM=0/1 says instead.
std::uint64_t start_vram_free(const Candidate& c) {
    std::uint64_t free = c.local_free;
    if (const char* t = std::getenv("BBHOST_TEST_VRAM_MB"); t && std::strtoull(t, nullptr, 10))
        free = std::min<std::uint64_t>(free, std::strtoull(t, nullptr, 10) << 20);
    return free;
}
bool small_vram(const Candidate& c) {
    const char* e = std::getenv("BBHOST_SMALL_VRAM");
    if (e && *e) return e[0] != '0';
    if (c.type_rank == 3 || memory_tight(c)) return false;
    return start_vram_free(c) < (7680ull << 20);
}
std::atomic<int> g_small{-1};  // small_vram of the device run on, -1 until known (the first answer holds)
std::atomic<int> g_integrated{-1};  // whether the device run on is an integrated GPU, -1 until known
// And whether, integrated, it has 8 GiB of video memory or more (a Strix
// Halo's carve-out; buffer_shadow.cpp integrated_with_room asks the same of
// the device made), -1 until known.
std::atomic<int> g_integrated_room{-1};
bool integrated_room(const Candidate& c) { return c.type_rank == 3 && c.local >= (8ull << 30); }

// OBS's game capture for Vulkan is an implicit layer (in graphics-hook64.dll)
// that the loader puts into every Vulkan program while OBS is installed,
// running or not, and leaves out while DISABLE_VULKAN_OBS_CAPTURE is set. On
// an AMD RX 580 (AMD's Windows driver) both starts with it in hung the GPU at
// the online notice - Windows reset it - and the start without it played on.
// video.obs_capture: "auto" (the default) leaves it out
// on an AMD card and in on others, "on" lets it in, "off" keeps it out; OBS's
// Window Capture needs no hook. Decided before any instance is made: the
// loader reads the variable then. Linux's obs-vkcapture is opt-in already.
void obs_capture_policy_once();
void obs_capture_policy() {
    static std::once_flag once;  // the probe instance and the device's may be made on two threads
    std::call_once(once, obs_capture_policy_once);
}
void obs_capture_policy_once() {
#if defined(_WIN32)
    if (std::getenv("DISABLE_VULKAN_OBS_CAPTURE")) return;  // the player's own say
    std::string mode = config_value("video.obs_capture");
    for (char& ch : mode) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (mode == "on" || mode == "true" || mode == "1") return;
    // Listing the layers reads their manifests and loads none of them.
    std::uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> layers(n);
    if (n) vkEnumerateInstanceLayerProperties(&n, layers.data());
    bool obs = false;
    for (std::uint32_t i = 0; i < n; ++i) obs |= std::strstr(layers[i].layerName, "OBS") != nullptr;
    if (!obs) return;
    _putenv_s("DISABLE_VULKAN_OBS_CAPTURE", "1");
    const bool off = mode == "off" || mode == "false" || mode == "0";
    std::uint32_t vendor = 0;
    if (!off) {
        // "auto": the card bbhost will run on, from an instance the layer is
        // kept out of.
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "bbhost";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&ici, nullptr, &instance) == VK_SUCCESS) {
            std::vector<Candidate> usable;
            const int chosen = choose_device(instance, false, usable);
            if (chosen >= 0) {
                VkPhysicalDeviceProperties dp{};
                vkGetPhysicalDeviceProperties(usable[static_cast<std::size_t>(chosen)].device, &dp);
                vendor = dp.vendorID;
            }
            vkDestroyInstance(instance, nullptr);
        }
        if (vendor != 0x1002) {
            _putenv_s("DISABLE_VULKAN_OBS_CAPTURE", "");
            host_log("gpu: OBS's game capture layer is installed and let in (not an AMD card; video.obs_capture = \"off\" keeps it out)");
            return;
        }
    }
    host_log("gpu: OBS's game capture layer is kept out (%s): on an AMD card it hung the GPU at the online notice. "
             "OBS's Window Capture records bbhost without it; video.obs_capture = \"on\" lets it in",
             off ? "video.obs_capture = \"off\"" : "an AMD card");
#endif
}

bool init_locked() {
    if (g.tried) {
        return g.ok;
    }
    g.tried = true;
    if (!g_enabled) {
        host_log("gpu: disabled by BBHOST_GPU=0");
        return false;
    }
    obs_capture_policy();
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "bbhost";
    app.apiVersion = VK_API_VERSION_1_3;
    // BBHOST_VK_APP_NAME / BBHOST_VK_ENGINE_NAME: the names the driver is
    // told (bbhost, and no engine, otherwise). A driver may pick a profile of
    // its own by them as well as by the executable's name; an A/B for what it
    // does differently with a name it knows.
    if (const char* e = std::getenv("BBHOST_VK_APP_NAME"); e && *e) app.pApplicationName = e;
    if (const char* e = std::getenv("BBHOST_VK_ENGINE_NAME"); e && *e) app.pEngineName = e;
    if (std::getenv("BBHOST_VK_APP_NAME") || std::getenv("BBHOST_VK_ENGINE_NAME"))
        host_log("gpu: the Vulkan instance says application \"%s\", engine \"%s\" (BBHOST_VK_APP_NAME, BBHOST_VK_ENGINE_NAME)",
                 app.pApplicationName, app.pEngineName ? app.pEngineName : "");
    std::vector<const char*> iext;
    for (const std::string& e : g_instance_exts) iext.push_back(e.c_str());
    std::vector<const char*> layers;
    if (g_validate) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        iext.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<std::uint32_t>(iext.size());
    ici.ppEnabledExtensionNames = iext.data();
    ici.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    if (vkCreateInstance(&ici, nullptr, &g.instance) != VK_SUCCESS) {
        host_log("gpu: no Vulkan 1.3 instance%s", g_validate ? " (validation layer requested)" : "");
        return false;
    }
    if (g_validate) {
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(g.instance, "vkCreateDebugUtilsMessengerEXT"));
        if (create) {
            VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            // Performance too: the best-practices checks' warnings
            // (VK_KHRONOS_VALIDATION_VALIDATE_BEST_PRACTICES=1) come as that type.
            mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            mci.pfnUserCallback = debug_callback;
            create(g.instance, &mci, nullptr, &g.messenger);
        }
        g_set_object_name = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(vkGetInstanceProcAddr(g.instance, "vkSetDebugUtilsObjectNameEXT"));
        host_log("gpu: validation layer enabled");
    }
    bool has_swapchain = false;
    std::vector<Candidate> usable;
    const int chosen = choose_device(g.instance, true, usable);
    if (chosen < 0) {
        host_log("gpu: no Vulkan 1.3 device with VK_EXT_external_memory_host and a graphics+compute queue");
        return false;
    }
    {
        const Candidate& c = usable[static_cast<std::size_t>(chosen)];
        g.phys = c.device;
        g.family = c.family;
        g.family_queues = c.queues;
        g.has_sparse = c.sparse;
        g.has_maint8 = c.maint8;
        g.has_push_descriptor = c.push;
        g.has_device_fault = c.fault;
        g.integrated = c.type_rank == 3;
        g_integrated_known.store(g.integrated ? 1 : 0, std::memory_order_relaxed);
        if (g_imports_whole.load(std::memory_order_relaxed) < 0) imports_whole_decide();  // by the device, unless set
        twins_decide(c.type_rank == 4);
        g.tight = memory_tight(c);
        {
            int unknown = -1;
            g_small.compare_exchange_strong(unknown, small_vram(c) ? 1 : 0);
            g.small_vram = g_small.load() == 1;
            if (g.small_vram)
                host_log("gpu: %llu MiB of video memory free at the start: small video memory (BBHOST_SMALL_VRAM=0 says otherwise) - no "
                         "spare image heap blocks in play, less untile scratch kept",
                         static_cast<unsigned long long>(start_vram_free(c) >> 20));
        }
        if (g_tight.exchange(g.tight ? 1 : 0) == (g.tight ? 0 : 1))
            host_log("gpu: the device chosen has %s memory than the one the guest patches were sized for", g.tight ? "less" : "more");
        if (g.tight) host_log("gpu: %llu MiB in all heaps: tight memory (BBHOST_TIGHT_MEMORY=0 says otherwise) - no buffer shadow, fewer spares during loads, live resolution up to 1080p's pixels",
                              static_cast<unsigned long long>(c.heaps >> 20));
        if (g_integrated.exchange(g.integrated ? 1 : 0) == (g.integrated ? 0 : 1))
            host_log("gpu: the device chosen is %s, unlike the one GX's buffers were decided for (host/gxvk.h)",
                     g.integrated ? "an integrated GPU" : "not an integrated GPU");
        if (const bool room = integrated_room(c); g_integrated_room.exchange(room ? 1 : 0) == (room ? 0 : 1))
            host_log("gpu: the device chosen is %s, unlike the one GX's buffers were decided for (host/gxvk.h)",
                     room ? "an integrated GPU with room" : "not an integrated GPU with room");
#if !defined(_WIN32)
        g.can_dmabuf = c.memory_fd && c.dma_buf;
#endif
        has_swapchain = c.swapchain;
        host_log("gpu: diagnostics available: VK_EXT_device_fault=%d VK_NV_device_diagnostic_checkpoints=%d VK_AMD_buffer_marker=%d",
                 static_cast<int>(c.fault), static_cast<int>(c.checkpoints), static_cast<int>(c.buffer_marker));
        {
            // Which driver, by its own name and version: AMD's Windows driver
            // for older cards (Polaris, Vega) is a branch of its own.
            VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
            VkPhysicalDeviceProperties2 dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            dp.pNext = &drv;
            vkGetPhysicalDeviceProperties2(c.device, &dp);
            host_log("gpu: driver %s, %s (id %d, conformance %u.%u.%u.%u; driverVersion 0x%x)", drv.driverName, drv.driverInfo,
                     static_cast<int>(drv.driverID), drv.conformanceVersion.major, drv.conformanceVersion.minor,
                     drv.conformanceVersion.subminor, drv.conformanceVersion.patch, dp.properties.driverVersion);
        }
        // Other programs' hooks, loaded with the instance (most come in as an
        // implicit Vulkan layer): said now, and again if the device is lost.
        if (const std::string hooks = host_foreign_hooks(); !hooks.empty()) host_log("gpu: hooked into this process: %s", hooks.c_str());
        // BBHOST_GPU_CHECKPOINTS=1: a checkpoint before every draw and
        // dispatch, so a device-lost report says how far the GPU got. Off by
        // default: vkCmdSetCheckpointNV was ~160 ns of each draw's ~11 us on
        // the command processor.
        const char* e = std::getenv("BBHOST_GPU_CHECKPOINTS");
        bool want_nv = c.checkpoints && e && e[0] == '1';
        // AMD's buffer markers name the draw or dispatch the GPU was in when
        // the device is lost (the first AMD run lost it at the title, and the
        // driver's VK_EXT_device_fault said nothing). On while AMD runs were
        // new; off by default since they cost the Steam Deck ~2% - two marker
        // writes a draw, and a draw recorded with checkpoints cannot go to
        // the recorder thread, so the command processor recorded every one.
        // BBHOST_GPU_CHECKPOINTS=amd turns them on.
        bool want_amd = c.buffer_marker && e && std::strcmp(e, "amd") == 0;
        // Unset, and the last run lost the device (its note beside the
        // pipeline cache): these starts run with whichever this driver has -
        // buffer markers first where there are any (NVIDIA's drivers have
        // them too): they land in host memory, readable after the loss, and
        // NVIDIA's Linux driver returned no checkpoints at all at three
        // losses (an RTX 2060, 615.71). BBHOST_GPU_CHECKPOINTS=0 keeps them off.
        std::string why;
        if (!e && (c.checkpoints || c.buffer_marker) && take_marked_start(why)) {
            (c.buffer_marker ? want_amd : want_nv) = true;
            host_log("gpu: the last run lost the device (%s): this start runs with the GPU's progress markers (%s), so another loss "
                     "names the draw the GPU was in; BBHOST_GPU_CHECKPOINTS=0 turns them off",
                     why.c_str(), c.buffer_marker ? "buffer markers" : "NVIDIA's checkpoints");
        }
        g.has_checkpoints = want_nv || want_amd;
        g_use_buffer_marker = want_amd;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g.phys, &props);
    vkGetPhysicalDeviceMemoryProperties(g.phys, &g.mem_props);
    {
        // Video memory: the heap of the first device-local type, where images,
        // targets and the mirrors go. BBHOST_TEST_VRAM_MB=<n>: its types hold
        // at most n MiB, as on a card with that much free (an 8 GB card's
        // budget is about 7171) - set before anything is allocated in them.
        const std::uint32_t t = find_memory_type(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        g.local_heap = t == UINT32_MAX ? 0 : g.mem_props.memoryTypes[t].heapIndex;
        std::uint32_t types = 0, other_device = 0;
        for (std::uint32_t i = 0; i < g.mem_props.memoryTypeCount; ++i) {
            const std::uint32_t h = g.mem_props.memoryTypes[i].heapIndex;
            if (h == g.local_heap) types |= 1u << i;
            else if (g.mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) other_device |= 1u << i;
        }
        if (t != UINT32_MAX) bb_memory_set_local_types(types);
        bb_memory_set_heap_classes(other_device);
        if (const char* e = std::getenv("BBHOST_TEST_VRAM_MB"); e && std::strtoull(e, nullptr, 10)) {
            bb_memory_test_limit(types, std::strtoull(e, nullptr, 10) << 20);
        }
    }
    g.timestamp_period_ns = props.limits.timestampPeriod;
    g.max_lds_bytes = props.limits.maxComputeSharedMemorySize;
    host_log("gpu: stage interface limits: vertex outputs %u, geometry outputs %u, fragment inputs %u components",
             props.limits.maxVertexOutputComponents, props.limits.maxGeometryOutputComponents, props.limits.maxFragmentInputComponents);
    g.max_anisotropy = props.limits.maxSamplerAnisotropy;
    g.max_lod_bias = props.limits.maxSamplerLodBias;
    g.ssbo_align = props.limits.minStorageBufferOffsetAlignment ? props.limits.minStorageBufferOffsetAlignment : 1;
    g.ssbo_max_range = props.limits.maxStorageBufferRange;
    // BBHOST_CB_SSBO=0: constant buffers stay on the page-table path. =1: they
    // are bound, but every shader keeps the page-table fallback (no lean
    // variant for draws whose buffers all bind).
    const char* cb_env = std::getenv("BBHOST_CB_SSBO");
    g.cb_ssbo = !(cb_env && cb_env[0] == '0');
    g.cb_lean = g.cb_ssbo && !(cb_env && cb_env[0] == '1');
    const char* exec_env = std::getenv("BBHOST_EXEC_KNOWN");
    g.exec_known = !(exec_env && exec_env[0] == '0');
    g.profile = [] {
        const char* e = std::getenv("BBHOST_GPU_PROFILE");
        return e && (e[0] == '1' || e[0] == '2' || e[0] == '3');
    }() && props.limits.timestampComputeAndGraphics;
    g.profile_gaps = g.profile && std::getenv("BBHOST_GPU_PROFILE")[0] == '2';
    // =3: a timestamp pair around each render pass, named by its first draw's
    // pipeline. A barrier ends every pass (BBHOST_PASS_BARRIERS=full, the
    // default), so a pass's pair is its own GPU time; draws' pairs overlap.
    g.profile_passes = g.profile && std::getenv("BBHOST_GPU_PROFILE")[0] == '3';
    // =3, or =1 with BBHOST_GPU_PROFILE_STATS=1: pipeline statistics beside
    // the pairs (gpu_internal.h profile_stats).
    if (g.profile_passes || (g.profile && !g.profile_gaps && std::getenv("BBHOST_GPU_PROFILE_STATS") &&
                             std::getenv("BBHOST_GPU_PROFILE_STATS")[0] == '1')) {
        VkPhysicalDeviceFeatures have{};
        vkGetPhysicalDeviceFeatures(g.phys, &have);
        g.profile_stats = have.pipelineStatisticsQuery == VK_TRUE;
    }
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceSubgroupProperties subp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    subp.pNext = &hostp;
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &subp;
    vkGetPhysicalDeviceProperties2(g.phys, &p2);
    g.host_pointer_align = static_cast<std::uint32_t>(hostp.minImportedHostPointerAlignment);
    // Which stages may run the subgroup operations the translator emits for
    // the EXEC mask. Tessellation is the one commonly left out, and the host's
    // tessellator needs to know rather than find out from a driver crash.
    g.subgroup_stages = subp.supportedStages;
    // The translator and the lifter emit subgroup operations only in those
    // stages, and only with the basic, ballot and shuffle operations there:
    // elsewhere an invocation runs as a wave of its own (gcn::subgroup_ops_in).
    // Lavapipe has them in fragment and compute shaders alone, and its vertex
    // stages - the host tessellator's LS, the particles' export shaders - took
    // ballots the validation layer refuses (VUID-RuntimeSpirv-None-06343).
    // BBHOST_SUBGROUP_GATE=0: every stage, as before.
    {
        const VkSubgroupFeatureFlags need = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        const char* e = std::getenv("BBHOST_SUBGROUP_GATE");
        const bool gate = !(e && e[0] == '0');
        const std::uint32_t stages = !gate ? 0x7fffffffu : (subp.supportedOperations & need) == need ? subp.supportedStages : 0u;
        gcn::set_subgroup_stages(stages);
        if (gate && (stages & 0x3f) != 0x3f) {
            host_log("gpu: subgroup operations in the stages 0x%x only (operations 0x%x): the others' shaders run each invocation as a wave of "
                     "its own (BBHOST_SUBGROUP_GATE=0: every stage)",
                     stages, subp.supportedOperations);
        }
    }
    {
        // Shader-level compile paths: D3D11 titles compile
        // each stage once when it is created and keep state separate; these are
        // the Vulkan extensions that allow that on this driver.
        std::uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> exts(n);
        vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &n, exts.data());
        const auto has = [&](const char* name) {
            for (const VkExtensionProperties& e : exts) {
                if (std::strcmp(e.extensionName, name) == 0) return true;
            }
            return false;
        };
        const bool gpl = has("VK_EXT_graphics_pipeline_library") && has("VK_KHR_pipeline_library");
        VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT gplp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
        if (gpl) {
            VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            pp.pNext = &gplp;
            vkGetPhysicalDeviceProperties2(g.phys, &pp);
        }
        host_log("gpu: compile paths: VK_EXT_shader_object=%d VK_EXT_graphics_pipeline_library=%d (fast linking %d, independent "
                 "interpolation decoration %d) VK_EXT_vertex_input_dynamic_state=%d VK_EXT_dynamic_state3=%d",
                 static_cast<int>(has("VK_EXT_shader_object")), static_cast<int>(gpl), static_cast<int>(gplp.graphicsPipelineLibraryFastLinking),
                 static_cast<int>(gplp.graphicsPipelineLibraryIndependentInterpolationDecoration),
                 static_cast<int>(has("VK_EXT_vertex_input_dynamic_state")), static_cast<int>(has("VK_EXT_extended_dynamic_state3")));
        g.has_gpl = gpl && gplp.graphicsPipelineLibraryFastLinking;  // BBHOST_PIPELINE_LIBRARY (render.cpp)
        // Depth clamp as the draw's state: the shadow casters draw with it, and
        // baked into their pre-rasterization library it made a second library
        // of a vertex shader already compiled without it - on the command
        // processor, 170-200 ms on a cold driver cache.
        if (g.has_gpl && has("VK_EXT_extended_dynamic_state3")) {
            VkPhysicalDeviceExtendedDynamicState3FeaturesEXT q3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
            VkPhysicalDeviceFeatures2 qf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            qf.pNext = &q3;
            vkGetPhysicalDeviceFeatures2(g.phys, &qf);
            const char* e = std::getenv("BBHOST_DYNAMIC_DEPTH_CLAMP");
            g.dynamic_depth_clamp = q3.extendedDynamicState3DepthClampEnable && !(e && e[0] == '0');
        }
        // VK_EXT_memory_budget: what the driver says this process holds per
        // heap - the number to compare with nvidia-smi when instances starve.
        g.has_memory_budget = has("VK_EXT_memory_budget");
        // BBHOST_PIPELINE_STATS: the driver's statistics and disassembly of named pipelines (render.cpp).
        g.pipeline_stats = std::getenv("BBHOST_PIPELINE_STATS") && has("VK_KHR_pipeline_executable_properties");
        // Depth targets in layouts of their own (render.cpp, rt_set_layout_locked) where GENERAL costs
        // them their compression: AMD's own drivers - the proprietary one and AMDVLK, which share its
        // layout handling - without VK_KHR_unified_image_layouts. There a depth target in GENERAL has no
        // fast clear and is tested and written uncompressed: a Radeon 680M drew 14-15% more frames at
        // 540p and at 1080p with the layouts. Other drivers keep GENERAL until one is measured
        // (BBHOST_DEPTH_LAYOUTS=1 uses the layouts on any device, =0 nowhere).
        {
            const char* e = std::getenv("BBHOST_DEPTH_LAYOUTS");
            const bool unified = has("VK_KHR_unified_image_layouts");
            VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
            VkPhysicalDeviceProperties2 dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            dp.pNext = &drv;
            vkGetPhysicalDeviceProperties2(g.phys, &dp);
            const bool amd = drv.driverID == VK_DRIVER_ID_AMD_PROPRIETARY || drv.driverID == VK_DRIVER_ID_AMD_OPEN_SOURCE;
            const bool forced = e && *e;
            g.depth_layouts = forced ? e[0] == '1' : amd && !unified;
            host_log("gpu: depth targets %s: %s (BBHOST_DEPTH_LAYOUTS=%s)",
                     g.depth_layouts ? "in layouts of their own (attachment-optimal drawn, read-only sampled, transfer-dst cleared)"
                                     : "in GENERAL like every image",
                     forced ? "BBHOST_DEPTH_LAYOUTS set"
                     : amd  ? (unified ? "AMD's driver with VK_KHR_unified_image_layouts" : "AMD's driver without VK_KHR_unified_image_layouts")
                            : "not AMD's driver, where they are not measured yet",
                     g.depth_layouts ? "0 keeps GENERAL" : "1 uses the layouts");
        }
#if defined(VK_KHR_present_mode_fifo_latest_ready)
        // The present mode FIFO_LATEST_READY (window.cpp, vk_create_swapchain):
        // V-Sync with no queue of frames behind the display - at each refresh
        // the newest finished frame goes up and older ones are dropped, which
        // is MAILBOX's latency on a driver that has no MAILBOX (AMD's Windows
        // driver lists IMMEDIATE, FIFO, FIFO_RELAXED and this). The KHR
        // extension or its EXT original, and the feature switched on.
        if (g_want_present) {
            const char* name = has(VK_KHR_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME) ? VK_KHR_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME
                               : has("VK_EXT_present_mode_fifo_latest_ready")          ? "VK_EXT_present_mode_fifo_latest_ready"
                                                                                       : nullptr;
            if (name) {
                VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR ql{};
                ql.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_KHR;
                VkPhysicalDeviceFeatures2 qf{};
                qf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                qf.pNext = &ql;
                vkGetPhysicalDeviceFeatures2(g.phys, &qf);
                if (ql.presentModeFifoLatestReady) g.fifo_latest_ready_ext = name;
            }
        }
#endif
    }

    // A second queue, when the family has one, is the presenter's: a present
    // can block (in the window system's copy, or on a full FIFO), and on the
    // renderer's queue it would have to hold the renderer's lock while it did.
    // BBHOST_VK_QUEUES=1: the one queue only, the presents on the renderer's
    // queue under its lock - as shadPS4 and vkcube make their devices. An A/B
    // for what Windows' per-process GPU counters see of a program that
    // presents from a queue of its own; it costs the command processor the present's blocking.
    const float prio[2] = {1.0f, 1.0f};
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g.family;
    const char* one_queue = std::getenv("BBHOST_VK_QUEUES");
    if (one_queue && one_queue[0] == '1') g.family_queues = 1;
    qci.queueCount = g.family_queues >= 2 ? 2 : 1;
    qci.pQueuePriorities = prio;
    host_log("gpu: queue family %u (%u queues): %u queue%s, priority 1.0%s", g.family, usable[static_cast<std::size_t>(chosen)].queues,
             qci.queueCount, qci.queueCount == 2 ? "s - the second the presenter's" : ", the presents on it too",
             one_queue && one_queue[0] == '1' ? " (BBHOST_VK_QUEUES=1)" : "");
    std::vector<const char*> dext = {"VK_EXT_external_memory_host"};
    if (g.has_maint8) dext.push_back("VK_KHR_maintenance8");
    if (g.has_push_descriptor) dext.push_back("VK_KHR_push_descriptor");
    // VK_EXT_device_fault: on a device loss the driver can tell us the faulting
    // address and what it was doing, which is the difference between "the GPU
    // died" and a specific resource to look at.
    if (g.has_device_fault) dext.push_back("VK_EXT_device_fault");
    if (g.can_dmabuf) {
        dext.push_back("VK_KHR_external_memory_fd");
        dext.push_back("VK_EXT_external_memory_dma_buf");
    }
    if (g.has_memory_budget) dext.push_back("VK_EXT_memory_budget");
    if (g.pipeline_stats) dext.push_back("VK_KHR_pipeline_executable_properties");
    if (g.has_checkpoints && !g_use_buffer_marker) dext.push_back("VK_NV_device_diagnostic_checkpoints");
    if (g_use_buffer_marker) dext.push_back("VK_AMD_buffer_marker");
    if (g.has_gpl) {
        dext.push_back("VK_KHR_pipeline_library");
        dext.push_back("VK_EXT_graphics_pipeline_library");
    }
    if (g.dynamic_depth_clamp) dext.push_back("VK_EXT_extended_dynamic_state3");
    if (g_want_present && has_swapchain) {
        dext.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        g.present_capable = true;
        if (g.fifo_latest_ready_ext) dext.push_back(g.fifo_latest_ready_ext);
    } else {
        g.fifo_latest_ready_ext = nullptr;
    }
    VkPhysicalDeviceFaultFeaturesEXT ffault{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    ffault.deviceFault = VK_TRUE;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    if (g.has_device_fault) f13.pNext = &ffault;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT fgpl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    fgpl.graphicsPipelineLibrary = VK_TRUE;
    if (g.has_gpl) {
        fgpl.pNext = f13.pNext;
        f13.pNext = &fgpl;
    }
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fpe{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    fpe.pipelineExecutableInfo = VK_TRUE;
    if (g.pipeline_stats) {
        fpe.pNext = f13.pNext;
        f13.pNext = &fpe;
    }
    // The translator reads a sample's offset from a register, as GCN does: a
    // run-time Offset image operand, which Vulkan allows on sample instructions
    // only with maintenance8's feature (enabling the extension alone left it
    // off: the validation layer's VUID-RuntimeSpirv-Offset-10213), and on
    // gathers with shaderImageGatherExtended (VUID-...-pCode-08740). NVIDIA's
    // driver took the shaders anyway; others need not. Without the feature a
    // sample's offset moves its coordinates instead
    // (gcn::set_runtime_sample_offsets, below).
    VkPhysicalDeviceMaintenance8FeaturesKHR fm8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
    if (g.has_maint8) {
        VkPhysicalDeviceMaintenance8FeaturesKHR q8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 qf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        qf.pNext = &q8;
        vkGetPhysicalDeviceFeatures2(g.phys, &qf);
        fm8.maintenance8 = q8.maintenance8;
        if (q8.maintenance8) {
            fm8.pNext = f13.pNext;
            f13.pNext = &fm8;
        }
    }
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT feds3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    feds3.extendedDynamicState3DepthClampEnable = VK_TRUE;
    if (g.dynamic_depth_clamp) {
        feds3.pNext = f13.pNext;
        f13.pNext = &feds3;
    }
    // Pixel shaders at wave32 (gcn/wave.h).
    // The translated stages use subgroup operations in SPIR-V 1.5 without
    // asking for a size, and Vulkan then holds every stage to the device's
    // default: on AMD's drivers (subgroupSize 64, sizes 32-64) every pixel
    // shader is compiled wave64, where RDNA's own choice for them is wave32 -
    // KyoPS4x measured the forced wave64 at ~10% of a 4K frame on a Radeon
    // 8060S. Subgroup size control (core in 1.3) lets the fragment stage of a
    // program that computes the same in a 32-lane subgroup (gcn/wave.h) ask
    // for 32. BBHOST_PS_WAVE32=0 keeps the device's size for all; =driver lets
    // the driver choose between 32 and its default, as SPIR-V 1.6 would.
    {
        VkPhysicalDeviceVulkan13Features q13{};
        q13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        VkPhysicalDeviceFeatures2 qf{};
        qf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        qf.pNext = &q13;
        vkGetPhysicalDeviceFeatures2(g.phys, &qf);
        VkPhysicalDeviceSubgroupSizeControlProperties sc{};
        sc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
        VkPhysicalDeviceProperties2 pp{};
        pp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        pp.pNext = &sc;
        vkGetPhysicalDeviceProperties2(g.phys, &pp);
        g.subgroup_size = subp.subgroupSize;
        g.subgroup_min = sc.minSubgroupSize;
        g.subgroup_max = sc.maxSubgroupSize;
        g.subgroup_required_stages = sc.requiredSubgroupSizeStages;
        // The driver's choice by default: on a Radeon 8060S (AMD's Windows
        // driver, RDNA 3.5) a required 32 made the world's pixel shaders
        // slower - 105 against 118 fps uncapped at a save's spawn, 9.3
        // against 8.3 ms of GPU a flip, two alternating runs each - and the
        // driver's choice measured as 64 does (117.7/115.7 against 117.6):
        // RDNA 3's wave64 issues both halves at once. A device whose own
        // sizes are below 32 (driver choice not offered) gets 32 required, as
        // BBHOST_PS_WAVE32=32 asks anywhere; =0 the device's size for all.
        const char* e = std::getenv("BBHOST_PS_WAVE32");
        const bool driver = !(e && (std::strcmp(e, "32") == 0 || std::strcmp(e, "1") == 0));
        const bool fragment = (sc.requiredSubgroupSizeStages & VK_SHADER_STAGE_FRAGMENT_BIT) != 0;
        const char* why = nullptr;
        if (e && e[0] == '0') why = "BBHOST_PS_WAVE32=0";
        else if (subp.subgroupSize <= 32) why = "the device's subgroups are 32 wide or narrower already";
        else if (!q13.subgroupSizeControl) why = "no subgroup size control";
        else if (sc.minSubgroupSize > 32 || sc.maxSubgroupSize < 32) why = "32 is not one of the device's subgroup sizes";
        // The driver's choice only where every size it may take is 32 or
        // more: the programs are judged for 32 lanes, not 8 or 16.
        else if (driver && sc.minSubgroupSize == 32) g.ps_wave = 1;
        else if (!fragment) why = "the fragment stage cannot be given a subgroup size";
        else g.ps_wave = 32;
        if (g.ps_wave) f13.subgroupSizeControl = VK_TRUE;
        if (g.ps_wave == 32) {
            host_log("gpu: pixel shaders at wave32: a required subgroup size of 32 for each program that computes the same in 32 lanes, "
                     "the device's %u for the rest (sizes %u-%u, required-size stages 0x%x%s; BBHOST_PS_WAVE32=0 keeps %u for all)",
                     subp.subgroupSize, sc.minSubgroupSize, sc.maxSubgroupSize, sc.requiredSubgroupSizeStages,
                     driver ? "; the driver's choice wanted, but the device has sizes below 32" : "", subp.subgroupSize);
        } else if (g.ps_wave == 1) {
            host_log("gpu: pixel shaders at the driver's choice of subgroup size (sizes %u-%u; BBHOST_PS_WAVE32=32 requires 32, =0 the device's) for each program "
                     "that computes the same in 32 lanes, the device's %u for the rest",
                     sc.minSubgroupSize, sc.maxSubgroupSize, subp.subgroupSize);
        } else {
            host_log("gpu: pixel shaders at the device's subgroup size %u: %s", subp.subgroupSize, why);
        }
    }
#if defined(VK_KHR_present_mode_fifo_latest_ready)
    VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR flatest{};
    flatest.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_KHR;
    flatest.presentModeFifoLatestReady = VK_TRUE;
    if (g.fifo_latest_ready_ext) {
        flatest.pNext = f13.pNext;
        f13.pNext = &flatest;
    }
#endif
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    {
        // Required by Vulkan 1.3, asked anyway: compute pipelines are taken
        // from the pipeline cache without compiling when it has them
        // (first_compute_pipeline).
        VkPhysicalDeviceVulkan13Features q13{};
        q13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        VkPhysicalDeviceFeatures2 qf{};
        qf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        qf.pNext = &q13;
        vkGetPhysicalDeviceFeatures2(g.phys, &qf);
        g.cache_control = q13.pipelineCreationCacheControl == VK_TRUE;
        f13.pipelineCreationCacheControl = q13.pipelineCreationCacheControl;
        // Also required by Vulkan 1.3: a pixel the clip distances cut out
        // (TranslateOptions::kVsOutCntlClipVarying) becomes a helper of its
        // quad, as the PS4's clipper leaves it, rather than ending - the quad's
        // derivatives need it (gcn::set_clip_demote). BBHOST_CLIP_DEMOTE=0:
        // OpKill, as before.
        const char* e = std::getenv("BBHOST_CLIP_DEMOTE");
        const bool demote = q13.shaderDemoteToHelperInvocation == VK_TRUE && !(e && e[0] == '0');
        if (demote) f13.shaderDemoteToHelperInvocation = VK_TRUE;
        gcn::set_clip_demote(demote);
        // BBHOST_ROBUST_IMAGES=1: robust image access (also required by 1.3).
        // GCN gives 0 for an image load outside the image and drops a store
        // there; the translation and the lift load and store at the
        // program's coordinates (image_load as OpImageFetch, image_store as
        // OpImageWrite), which Vulkan leaves undefined outside the image -
        // a 1x1 dummy bound for a missing texture among them. Off by default:
        // the driver compiles every pipeline again once with it.
        const char* ri = std::getenv("BBHOST_ROBUST_IMAGES");
        if (ri && ri[0] == '1' && q13.robustImageAccess == VK_TRUE) {
            f13.robustImageAccess = VK_TRUE;
            host_log("gpu: robust image access on (BBHOST_ROBUST_IMAGES=1)");
        }
    }
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.pNext = &f13;
    f12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    f2.features.shaderInt64 = VK_TRUE;
    f2.features.shaderStorageImageReadWithoutFormat = VK_TRUE;
    f2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    f2.features.geometryShader = VK_TRUE;
    f2.features.independentBlend = VK_TRUE;
    f2.features.fillModeNonSolid = VK_TRUE;
    f2.features.depthClamp = VK_TRUE;
    f2.features.dualSrcBlend = VK_TRUE;
    f2.features.shaderClipDistance = VK_TRUE;
    f2.features.shaderCullDistance = VK_TRUE;
    f2.features.samplerAnisotropy = VK_TRUE;
    // The buffer shadow's mirrors are sparse buffers with a 64 KiB device page
    // bound per guest page actually copied (buffer_shadow.cpp): a full 1 GiB
    // block per guest chunk cost 4-5 GiB of VRAM an instance.
    {
        VkPhysicalDeviceFeatures sp{};
        vkGetPhysicalDeviceFeatures(g.phys, &sp);
        g.dmabuf_sparse = g.can_dmabuf && g.has_sparse && sp.sparseBinding;  // a queue with SPARSE_BINDING so far
        g.has_sparse = g.has_sparse && sp.sparseBinding && sp.sparseResidencyBuffer;
        // Only where they have been run: an AMD Radeon 8060S on AMD's Windows
        // driver lost the device at the first binds, ~9 frames in, and the
        // title played on behind a white window (2026-09-28). Elsewhere the
        // mirrors are whole blocks, which costs VRAM, not correctness.
        // BBHOST_SHADOW_SPARSE=1 turns them on anyway, =0 off everywhere.
        VkPhysicalDeviceProperties dp{};
        vkGetPhysicalDeviceProperties(g.phys, &dp);
        g.vendor_id = dp.vendorID;
        const bool tested = dp.vendorID == 0x10de;  // NVIDIA
        const char* e = std::getenv("BBHOST_SHADOW_SPARSE");
        if (e && e[0] == '0') g.has_sparse = false;
        else if (!(e && e[0] == '1') && !tested) g.has_sparse = false;
        f2.features.sparseBinding = g.has_sparse || g.dmabuf_sparse ? VK_TRUE : VK_FALSE;
        f2.features.sparseResidencyBuffer = g.has_sparse ? VK_TRUE : VK_FALSE;
        if (!buffer_shadow_on())
            host_log("gpu: buffer shadow off (%s): draws read vertex and index data from the imports",
                     std::getenv("BBHOST_BUFFER_SHADOW") ? "BBHOST_BUFFER_SHADOW=0"
                     : g.integrated ? "an integrated GPU: its device-local memory is the same RAM; BBHOST_BUFFER_SHADOW=1 turns it on"
                                    : "tight memory; BBHOST_BUFFER_SHADOW=1 turns it on");
        host_log("gpu: sparse mirrors %s", g.has_sparse ? "on"
                                           : !tested && !(e && e[0] == '0') ? "off (not yet run on this vendor's driver; BBHOST_SHADOW_SPARSE=1 tries them)"
                                                                           : "off (no sparse residency, or BBHOST_SHADOW_SPARSE=0)");
    }
    // BBHOST_ROBUST=1: bounds-check buffer reads. Off, an out-of-range read of
    // a storage buffer is undefined and in practice returns whatever lies past
    // the binding - and the guest's dynamic buffers rotate between addresses
    // every frame, so the same contents can be followed by different bytes.
    // That is the shape of the menu flash: identical data in, different
    // geometry out, one frame in a hundred.
    if (const char* e = std::getenv("BBHOST_ROBUST"); e && e[0] == '1') {
        f2.features.robustBufferAccess = VK_TRUE;  // core since 1.0, always supported
        host_log("gpu: robust buffer access on (BBHOST_ROBUST=1)");
    }
    // BBHOST_IMAGE_STORE_BOUNDS (on by default): an image store's lanes outside
    // the view bound for its T# skip it, as GCN writes nothing outside the
    // image (gcn::set_image_store_bounds) - Vulkan leaves such a write
    // undefined unless robust image access is on, and the game's compute
    // passes make them where a target's height is not a multiple of their
    // tiles. =0: every lane writes, as before.
    {
        const char* e = std::getenv("BBHOST_IMAGE_STORE_BOUNDS");
        gcn::set_image_store_bounds(!(e && e[0] == '0'));
        host_log("gpu: image stores outside the bound view %s (BBHOST_IMAGE_STORE_BOUNDS=%c)",
                 gcn::image_store_bounds() ? "skip those lanes, as GCN drops them" : "write as before", gcn::image_store_bounds() ? '1' : '0');
    }
    // BBHOST_INDEX_BOUNDS=1: the shaders' own check instead, where GCN makes
    // one - a typed load by index past the V#'s num_records reads 0
    // (gcn::set_index_bounds), on every driver, at a compare and a select a
    // load. Off by default: the modules of about 1,100 of the game's 2,680
    // programs change with it (a pipeline compile each, once).
    {
        const char* e = std::getenv("BBHOST_INDEX_BOUNDS");
        gcn::set_index_bounds(e && e[0] == '1');
        if (gcn::index_bounds()) host_log("gpu: typed loads by index read 0 past the V#'s records (BBHOST_INDEX_BOUNDS=1)");
    }
    f2.features.fragmentStoresAndAtomics = VK_TRUE;
    f2.features.vertexPipelineStoresAndAtomics = VK_TRUE;
    f2.features.pipelineStatisticsQuery = g.profile_stats ? VK_TRUE : VK_FALSE;
    glitch_device_features(f2.features);  // BBHOST_GLITCH=1 only: precise occlusion queries
    occlusion_device_features(f2.features, dext, &f13.pNext);  // precise queries, conditional rendering
    // DB_DEPTH_CONTROL bit 3: the deferred lights cull by depth range.
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(g.phys, &supported);
    f2.features.shaderImageGatherExtended = supported.shaderImageGatherExtended;
    // BBHOST_RUNTIME_OFFSETS=0: the coordinates move even with the feature, to
    // compare the two.
    const char* ro = std::getenv("BBHOST_RUNTIME_OFFSETS");
    const bool runtime_offsets = fm8.maintenance8 == VK_TRUE && !(ro && ro[0] == '0');
    gcn::set_runtime_sample_offsets(runtime_offsets);
    host_log("gpu: run-time image offsets: maintenance8 %s, gather-extended %s",
             runtime_offsets   ? "on"
             : fm8.maintenance8 ? "on, but a sample's offset moves its coordinates (BBHOST_RUNTIME_OFFSETS=0)"
                                : "unsupported (a sample's offset moves its coordinates instead)",
             supported.shaderImageGatherExtended ? "on" : "unsupported");
    g.has_depth_bounds = supported.depthBounds == VK_TRUE;
    f2.features.depthBounds = supported.depthBounds;
    // Cube-array views: the game's cube maps with more than one cube (and the
    // renderer's cube-array dummy) are viewed as VK_IMAGE_VIEW_TYPE_CUBE_ARRAY,
    // which needs the feature (VUID-VkImageViewCreateInfo-viewType-01004);
    // a validated world run found it off. Every desktop driver has it.
    f2.features.imageCubeArray = supported.imageCubeArray;
    if (!supported.imageCubeArray) host_log("gpu: no imageCubeArray: cube-array views are invalid on this device");
    // A rasterizer state's depth bias clamp; without the
    // feature the bias still applies, unclamped.
    g.has_depth_bias_clamp = supported.depthBiasClamp == VK_TRUE;
    f2.features.depthBiasClamp = supported.depthBiasClamp;
    // The game's particles are patches through LS/HS/domain-VS. We emulate the
    // tessellator today; the host's own stages take them, so the
    // hull shader becomes a tessellation control shader and the domain shader
    // an evaluation shader. Optional: without it the emulation stays.
    g.has_tessellation = supported.tessellationShader == VK_TRUE;
    f2.features.tessellationShader = supported.tessellationShader;
    // f16 rounded toward zero by the device itself: GCN's v_cvt_pkrtz_f16_f32
    // and the packed colour exports that read it become two native conversions
    // a pair (gcn/half.h native_half_rtz), where the exact integer form was ~6
    // operations a component - a quarter of the hot G-buffer pixel shaders' ALU
    // on a Steam Deck. BBHOST_HALF_NATIVE=0: the integer form.
    {
        VkPhysicalDeviceVulkan12Features q12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        q2.pNext = &q12;
        vkGetPhysicalDeviceFeatures2(g.phys, &q2);
        VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
        VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        pp.pNext = &p12;
        vkGetPhysicalDeviceProperties2(g.phys, &pp);
        const char* e = std::getenv("BBHOST_HALF_NATIVE");
        const bool native = q12.shaderFloat16 && p12.shaderRoundingModeRTZFloat16 && p12.shaderDenormPreserveFloat16 && !(e && e[0] == '0');
        if (q12.shaderFloat16) f12.shaderFloat16 = VK_TRUE;
        if (q12.timelineSemaphore && !(std::getenv("BBHOST_TIMELINE") && std::getenv("BBHOST_TIMELINE")[0] == '0'))
            f12.timelineSemaphore = VK_TRUE;
        // GCN's MIRROR_ONCE_LAST_TEXEL is Vulkan's MIRROR_CLAMP_TO_EDGE, which
        // a sampler may name only with this feature on (textures.cpp
        // sampler_for falls back to MIRRORED_REPEAT without it).
        if (q12.samplerMirrorClampToEdge && sampler_fixes_on()) {
            f12.samplerMirrorClampToEdge = VK_TRUE;
            g.mirror_clamp = true;
        }
        if (supported.shaderInt16) f2.features.shaderInt16 = VK_TRUE;
        g.has_f16_math = q12.shaderFloat16 && supported.shaderInt16;
        gcn::set_native_half_rtz(native);
        // The legacy multiply's zero rule as a select on AMD's own drivers,
        // whose compiler matches neither form to v_mul_legacy_f32 (gcn/half.h).
        VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        dp.pNext = &drv;
        vkGetPhysicalDeviceProperties2(g.phys, &dp);
        g.driver_id = drv.driverID;
        gcn::set_legacy_mul_select(drv.driverID == VK_DRIVER_ID_AMD_PROPRIETARY || drv.driverID == VK_DRIVER_ID_AMD_OPEN_SOURCE);
        host_log("gpu: legacy multiplies' zero rule as %s", gcn::legacy_mul_plain() ? "nothing (BBHOST_LEGACY_MUL=plain)"
                                                             : gcn::legacy_mul_min_form() ? "min(|a|, |b|) == 0" : "a == 0 || b == 0");
        host_log("gpu: f16 rounding toward zero by the device: %s (shaderFloat16 %d, RTZ %d, denorm preserve %d)", native ? "yes" : "no",
                 static_cast<int>(q12.shaderFloat16), static_cast<int>(p12.shaderRoundingModeRTZFloat16),
                 static_cast<int>(p12.shaderDenormPreserveFloat16));
        // The GPU particles (the draws that bind a geometry shader): as one
        // vertex stage on every device; BBHOST_GX_PARTICLES=gs, a debug switch,
        // draws them with their geometry shader (on GCN under AMD's driver
        // those came out as streaks across the screen).
        if (particle_path() == ParticlePath::kGs) {
            host_log("gpu: GPU particles drawn with their geometry shader (BBHOST_GX_PARTICLES=gs, a debug switch%s)",
                     (drv.driverID == VK_DRIVER_ID_AMD_PROPRIETARY || drv.driverID == VK_DRIVER_ID_AMD_OPEN_SOURCE) && g.subgroup_min == 64 &&
                             g.subgroup_max == 64
                         ? "; GCN under AMD's driver, where they come out as streaks"
                         : "");
        } else {
            host_log("gpu: GPU particles drawn as one vertex stage, without a geometry shader (BBHOST_GX_PARTICLES=gs draws them with it)");
        }
    }
    // BBHOST_BINDLESS=1: descriptor indexing for the global set
    // of views and samplers (bindless.cpp). It needs the constant buffers
    // pushed (set 2), so the global set can be set 3 of every layout.
    if (bindless_wanted()) {
        VkPhysicalDeviceVulkan12Features q12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        q2.pNext = &q12;
        vkGetPhysicalDeviceFeatures2(g.phys, &q2);
        g.bindless = g.has_push_descriptor && q12.runtimeDescriptorArray && q12.descriptorBindingPartiallyBound &&
                     q12.descriptorBindingSampledImageUpdateAfterBind && q12.descriptorBindingStorageImageUpdateAfterBind &&
                     q12.descriptorBindingUpdateUnusedWhilePending && q2.features.shaderSampledImageArrayDynamicIndexing &&
                     q2.features.shaderStorageImageArrayDynamicIndexing;
        if (g.bindless) {
            f12.descriptorIndexing = q12.descriptorIndexing;
            f12.runtimeDescriptorArray = VK_TRUE;
            f12.descriptorBindingPartiallyBound = VK_TRUE;
            f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
            f12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
            f12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
            f2.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
            f2.features.shaderStorageImageArrayDynamicIndexing = VK_TRUE;
        } else {
            host_log("gpu: bindless asked for, but the device lacks descriptor indexing or push descriptors; off");
        }
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<std::uint32_t>(dext.size());
    dci.ppEnabledExtensionNames = dext.data();
    if (vkCreateDevice(g.phys, &dci, nullptr, &g.device) == VK_SUCCESS && g.has_push_descriptor) {
        g.cmd_push_descriptor_set =
            reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(g.device, "vkCmdPushDescriptorSetKHR"));
        if (!g.cmd_push_descriptor_set) g.has_push_descriptor = false;
    }
    if (g.device) {
        // How much video memory is ours: the driver's budget for the heap now,
        // before anything of ours is in it (the card's memory less what the
        // desktop and other programs hold), else the heap's size.
        const std::uint64_t heap = g.mem_props.memoryHeaps[g.local_heap].size;
        std::uint64_t budget = heap;
        if (g.has_memory_budget) {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT mb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
            VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
            mp.pNext = &mb;
            vkGetPhysicalDeviceMemoryProperties2(g.phys, &mp);
            if (mb.heapBudget[g.local_heap]) budget = std::min(heap, mb.heapBudget[g.local_heap] - std::min(mb.heapBudget[g.local_heap], mb.heapUsage[g.local_heap]));
        }
        const char* test = std::getenv("BBHOST_TEST_VRAM_MB");
        const std::uint64_t test_bytes = test ? std::strtoull(test, nullptr, 10) << 20 : 0;
        if (test_bytes) budget = std::min(budget, test_bytes);
        g.local_budget = budget;
        host_log("gpu: video memory: heap %u, %llu MiB, %llu MiB of it free for this process%s%s", g.local_heap,
                 static_cast<unsigned long long>(heap >> 20), static_cast<unsigned long long>(budget >> 20),
                 g.has_memory_budget ? " (the driver's budget)" : " (no budget from the driver: the heap's size)",
                 test_bytes ? " - BBHOST_TEST_VRAM_MB, acted out" : "");
        vram_reserve_init();
        shadow_budget_locked();
    }
    if (g.device && g_use_buffer_marker) {
        g.cmd_buffer_marker = reinterpret_cast<PFN_vkCmdWriteBufferMarkerAMD>(vkGetDeviceProcAddr(g.device, "vkCmdWriteBufferMarkerAMD"));
        if (!g.cmd_buffer_marker || !create_dev_buffer(g.marker_buf, 256, true)) {
            g.cmd_buffer_marker = nullptr;
            g.has_checkpoints = false;
        } else {
            std::memset(g.marker_buf.map, 0, 256);
            host_log("gpu: AMD buffer markers on each draw and dispatch (BBHOST_GPU_CHECKPOINTS=amd; a device loss names the work)");
        }
    }
    if (g.device && g.has_checkpoints && !g_use_buffer_marker) {
        g.cmd_checkpoint = reinterpret_cast<PFN_vkCmdSetCheckpointNV>(vkGetDeviceProcAddr(g.device, "vkCmdSetCheckpointNV"));
    }
    if (g.device && g.dynamic_depth_clamp) {
        g.cmd_set_depth_clamp_enable =
            reinterpret_cast<PFN_vkCmdSetDepthClampEnableEXT>(vkGetDeviceProcAddr(g.device, "vkCmdSetDepthClampEnableEXT"));
        if (!g.cmd_set_depth_clamp_enable) g.dynamic_depth_clamp = false;
        host_log("gpu: depth clamp %s", g.dynamic_depth_clamp ? "dynamic (VK_EXT_extended_dynamic_state3)" : "in the pipeline");
    }
    if (g.device == VK_NULL_HANDLE) {
        host_log("gpu: device creation failed (needs Vulkan 1.3 dynamic rendering, bufferDeviceAddress, shaderInt64, geometry shaders, host memory import)");
        return false;
    }
    g.get_host_pointer_props = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
        vkGetDeviceProcAddr(g.device, "vkGetMemoryHostPointerPropertiesEXT"));
    if (!g.get_host_pointer_props) {
        host_log("gpu: vkGetMemoryHostPointerPropertiesEXT missing");
        return false;
    }
    if (g.can_dmabuf) {
        g.get_memory_fd_props = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(vkGetDeviceProcAddr(g.device, "vkGetMemoryFdPropertiesKHR"));
        g.can_dmabuf = g.get_memory_fd_props != nullptr;
    }
    vkGetDeviceQueue(g.device, g.family, 0, &g.queue);
    if (g.family_queues >= 2) vkGetDeviceQueue(g.device, g.family, 1, &g.present_queue);
    if (f12.timelineSemaphore) {
        VkSemaphoreTypeCreateInfo st{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        st.initialValue = 0;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = &st;
        if (vkCreateSemaphore(g.device, &sci, nullptr, &g.timeline) != VK_SUCCESS) g.timeline = VK_NULL_HANDLE;
        if (g.timeline && vkCreateSemaphore(g.device, &sci, nullptr, &g.present_timeline) != VK_SUCCESS)
            g.present_timeline = VK_NULL_HANDLE;
    }
    g_timeline_on.store(g.timeline != VK_NULL_HANDLE, std::memory_order_release);
    host_log("gpu: submissions %s", g.timeline ? "signal a timeline semaphore: completion is read and waited for without the renderer's lock"
                                              : "are followed by their fences alone (no timeline semaphore, or BBHOST_TIMELINE=0)");
    start_label_landing();
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = g.family;
    vkCreateCommandPool(g.device, &pci, nullptr, &g.pool);
    if (const char* e = std::getenv("BBHOST_GPU_SLOTS"); e && *e) {
        g.slot_count = std::clamp(std::atoi(e), 2, kSlots);
        host_log("gpu: %d command buffers in flight at most (BBHOST_GPU_SLOTS)", g.slot_count);
    }
    for (int k = 0; k < g.slot_count; ++k) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = g.pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        vkAllocateCommandBuffers(g.device, &cai, &g.slots[k].cmd);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(g.device, &fci, nullptr, &g.slots[k].fence);
        glitch_slot_init_locked(g.slots[k]);
        if (g.profile) {
            VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qci.queryCount = kMaxQueued * kStageSlots * 2;
            if (vkCreateQueryPool(g.device, &qci, nullptr, &g.slots[k].qpool) != VK_SUCCESS) g.profile = false;
            if (g.profile_stats) {
                VkQueryPoolCreateInfo sci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                sci.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
                sci.queryCount = kMaxQueued * kStageSlots;
                sci.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
                                         VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
                                         VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
                                         VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
                                         VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
                                         VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
                if (vkCreateQueryPool(g.device, &sci, nullptr, &g.slots[k].stat_pool) != VK_SUCCESS) g.profile_stats = false;
            }
        }
    }
    busy_init_locked();
    g.cmd_ = g.slots[0].cmd;
    g.fence = g.slots[0].fence;
    occlusion_device_ready_locked();
    // The pipeline cache persists under the data root, so a later run creates
    // the pipelines this one compiled without the driver compiling them again.
    // The driver ignores data from another device or driver version; another
    // translator throws it away itself (and its translations with it).
    shader_caches_check_build();
    std::vector<char> cache_data;
    if (const std::string path = pipeline_cache_path(); !path.empty()) {
        std::ifstream in(path, std::ios::binary);
        cache_data.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pcci.initialDataSize = cache_data.size();
    pcci.pInitialData = cache_data.empty() ? nullptr : cache_data.data();
    if (vkCreatePipelineCache(g.device, &pcci, nullptr, &g.cache) != VK_SUCCESS && !cache_data.empty()) {
        host_log("gpu: the saved pipeline cache was rejected; starting with an empty one");
        pcci.initialDataSize = 0;
        pcci.pInitialData = nullptr;
        vkCreatePipelineCache(g.device, &pcci, nullptr, &g.cache);
    } else if (!cache_data.empty()) {
        g_pipeline_cache_bytes.store(cache_data.size());
        host_log("gpu: pipeline cache loaded, %zu KiB", cache_data.size() >> 10);
    }
    VkPipelineCacheCreateInfo side{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (vkCreatePipelineCache(g.device, &side, nullptr, &g.side_cache) != VK_SUCCESS) g.side_cache = VK_NULL_HANDLE;
    if (vkCreatePipelineCache(g.device, &side, nullptr, &g.opt_cache) != VK_SUCCESS) g.opt_cache = VK_NULL_HANDLE;

    // Descriptor layout: params UBO, then sampled images, samplers, storage images.
    std::vector<VkDescriptorSetLayoutBinding> binds;
    binds.push_back({gcn::kBindingParams, params_descriptor_type(), 1, VK_SHADER_STAGE_ALL, nullptr});
    for (std::uint32_t k = 0; k < kMaxImages; ++k) {
        binds.push_back({gcn::kBindingImage0 + k, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr});
        binds.push_back({gcn::kBindingSampler0 + k, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_ALL, nullptr});
        binds.push_back({gcn::kBindingStorageImage0 + k, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr});
    }
    for (std::uint32_t k = 0; k < gcn::kMaxBuffers && !cb_push_on(); ++k) {
        binds.push_back({gcn::kBindingStorageBuffer0 + k, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr});
    }
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = static_cast<std::uint32_t>(binds.size());
    lci.pBindings = binds.data();
    if (vkCreateDescriptorSetLayout(g.device, &lci, nullptr, &g.set_layout) != VK_SUCCESS) {
        host_log("gpu: descriptor set layout failed");
        return false;
    }
    if (cb_push_on()) {
        std::vector<VkDescriptorSetLayoutBinding> cbs;
        for (std::uint32_t k = 0; k < 2 * gcn::kMaxBuffers; ++k) {
            cbs.push_back({gcn::kBindingStorageBuffer0 + k, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr});
        }
        VkDescriptorSetLayoutCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        pci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        pci.bindingCount = static_cast<std::uint32_t>(cbs.size());
        pci.pBindings = cbs.data();
        if (vkCreateDescriptorSetLayout(g.device, &pci, nullptr, &g.cb_push_layout) != VK_SUCCESS) {
            host_log("gpu: constant-buffer push descriptor layout failed");
            return false;
        }
        host_log("gpu: constant buffers are pushed per draw (set 2); descriptor sets cached by content");
    }
    // Stage 2's GX views (textures.cpp texture_view_by_id): their entries, or today's lookup and refresh.
    switch (textures_images_mode()) {
    case 0: host_log("gpu: GX views found by their id with today's lookup and refresh (BBHOST_GXVK_IMAGES=0; unset, they bind from their entries)"); break;
    case 1: host_log("gpu: GX views bind from their entries, an image nothing has written without its surface's refresh (BBHOST_GXVK_IMAGES=1, the default; =0 today's lookup and refresh)"); break;
    default: host_log("gpu: GX views found by their id with today's lookup and refresh, their entries compared beside it (BBHOST_GXVK_IMAGES=2)"); break;
    }
    // Graphics stages under bindless keep only their params
    // block in their own sets; images and samplers are set 3's.
    g.gfx_set_layout = g.set_layout;
    if (g.bindless) {
        VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
        VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        pp.pNext = &p12;
        vkGetPhysicalDeviceProperties2(g.phys, &pp);
        const VkDescriptorSetLayoutBinding params{gcn::kBindingParams, params_descriptor_type(), 1, VK_SHADER_STAGE_ALL, nullptr};
        VkDescriptorSetLayoutCreateInfo sci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        sci.bindingCount = 1;
        sci.pBindings = &params;
        g.bindless = vkCreateDescriptorSetLayout(g.device, &sci, nullptr, &g.gfx_set_layout) == VK_SUCCESS &&
                     bindless_create_locked(std::min(p12.maxDescriptorSetUpdateAfterBindSampledImages, p12.maxDescriptorSetUpdateAfterBindStorageImages),
                                            p12.maxDescriptorSetUpdateAfterBindSamplers);
        if (!g.bindless) g.gfx_set_layout = g.set_layout;
    }
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g.set_layout;
    vkCreatePipelineLayout(g.device, &plci, nullptr, &g.pipe_layout);
    if (cb_push_on()) {  // ComputeLean: set 0 as above, its buffers pushed in set 1
        const VkDescriptorSetLayout cs_sets[2] = {g.set_layout, g.cb_push_layout};
        plci.setLayoutCount = 2;
        plci.pSetLayouts = cs_sets;
        if (vkCreatePipelineLayout(g.device, &plci, nullptr, &g.cs_buffers_layout) != VK_SUCCESS) g.cs_buffers_layout = VK_NULL_HANDLE;
    }
    VkDescriptorSetLayout four[4] = {g.gfx_set_layout, g.gfx_set_layout, g.cb_push_layout, g.bindless_layout};
    plci.setLayoutCount = g.bindless ? 4 : cb_push_on() ? 3 : 2;
    plci.pSetLayouts = four;
    vkCreatePipelineLayout(g.device, &plci, nullptr, &g.gfx_pipe_layout);
    // Allocating a set consumes every binding the layout declares, bound or not.
    // kSpareSets: room for the sets alloc_set_locked allocates ahead.
    constexpr std::uint32_t kSpareSets = 1024;
    const std::uint32_t max_sets = kMaxQueued * kStageSlots + kSpareSets;
    VkDescriptorPoolSize sizes[] = {
        {params_descriptor_type(), max_sets},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, max_sets * kMaxImages},
        {VK_DESCRIPTOR_TYPE_SAMPLER, max_sets * kMaxImages},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, max_sets * kMaxImages},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_sets * gcn::kMaxBuffers},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = max_sets;
    dpci.poolSizeCount = 5;
    dpci.pPoolSizes = sizes;
    for (int k = 0; k < g.slot_count; ++k) {
        if (vkCreateDescriptorPool(g.device, &dpci, nullptr, &g.slots[k].pool) != VK_SUCCESS) {
            host_log("gpu: descriptor pool failed");
            return false;
        }
    }
    g.desc_pool = g.slots[0].pool;
    // Params ring (host visible), one region per slot.
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = static_cast<VkDeviceSize>(kParamsStride) * kMaxQueued * kStageSlots * g.slot_count;
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    vkCreateBuffer(g.device, &bci, nullptr, &g.ubo);
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.device, g.ubo, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    // Host-cached where there is such a type: the draws write two params
    // blocks each, and in write-combined memory the next locked instruction
    // waited for them to drain (BBHOST_PARAMS_WC=1 keeps write-combined).
    static const bool params_wc = [] {
        const char* e = std::getenv("BBHOST_PARAMS_WC");
        return e && e[0] == '1';
    }();
    mai.memoryTypeIndex = params_wc ? UINT32_MAX
                                    : find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                                               VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX) {
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    // A discrete GPU's shaders read every draw's params block: in its video
    // memory rather than across the bus, where it has some the CPU can write
    // (rings_in_vram); host memory if that is full.
    bool params_allocated = false;
    if (rings_in_vram()) {
        const std::uint32_t vram = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::string note;
        if (vram != UINT32_MAX && ring_fits_budget(vram, req.size, "the params ring", &note)) {
            const std::uint32_t host_type = mai.memoryTypeIndex;
            mai.memoryTypeIndex = vram;
            MemOwner own_mem("params ring");
            params_allocated = vkAllocateMemory(g.device, &mai, nullptr, &g.ubo_mem) == VK_SUCCESS;
            if (!params_allocated) mai.memoryTypeIndex = host_type;
            note_ring_placed(params_allocated, req.size);
            if (params_allocated) host_log("gpu: the params ring in video memory the CPU writes (%s before it)", note.c_str());
        }
    }
    host_log("gpu: params ring in memory type %u (flags 0x%x)", mai.memoryTypeIndex,
             mai.memoryTypeIndex < g.mem_props.memoryTypeCount ? g.mem_props.memoryTypes[mai.memoryTypeIndex].propertyFlags : 0u);
    if (!params_allocated) {
        MemOwner own_mem("params ring");
        params_allocated = vkAllocateMemory(g.device, &mai, nullptr, &g.ubo_mem) == VK_SUCCESS;
    }
    if (!params_allocated) {
        host_log("gpu: params buffer allocation failed");
        return false;
    }
    vkBindBufferMemory(g.device, g.ubo, g.ubo_mem, 0);
    vkMapMemory(g.device, g.ubo_mem, 0, VK_WHOLE_SIZE, 0, &g.ubo_map);
    g_params_desc = {g.ubo, 0, sizeof(gcn::StageParams)};
    if (!create_dummy_resources()) {
        host_log("gpu: dummy resources failed");
        return false;
    }
    bindless_dummies_locked(g.dummy_view, g.dummy_view, g.dummy_sampler);  // slot 0 of the global arrays
    bool tables_made = false;
    {
        MemOwner own_mem("page tables");
        tables_made = create_dev_buffer(g.l1, gcn::kL1Entries * 8, true) && create_dev_buffer(g.sink, 1ull << gcn::kPageShift, false) &&
                      create_dev_buffer(g.sink_l2, gcn::kL2Entries * 8, true);
    }
    if (!tables_made) {
        host_log("gpu: page table buffers failed");
        return false;
    }
    for (std::uint32_t k = 0; k < gcn::kL2Entries; ++k) static_cast<std::uint64_t*>(g.sink_l2.map)[k] = g.sink.address;
    for (std::uint32_t k = 0; k < gcn::kL1Entries; ++k) static_cast<std::uint64_t*>(g.l1.map)[k] = g.sink_l2.address;
    // BBHOST_OOB_COUNT=1 (a diagnostic): vertex, pixel and compute stages
    // with no tessellation or GS role, translated or lifted, count into this
    // buffer the buffer accesses that leave what GCN would read and the image
    // accesses at integer coordinates outside the view bound for them
    // (gcn::set_oob_count; StageParams::lds_address points at it); the frame
    // detail lines and the exit report read it (oob_totals). The counters
    // live in the lifter's SPIR-V, so the HLSL path is off while they count
    // (hlsl_mode). =2: the control (gcn::set_oob_control) - every limit
    // halved and one access in 256 past it counted, to show each kind reaches
    // the host at a small part of the draws' time.
    if (const char* e = std::getenv("BBHOST_OOB_COUNT"); e && (e[0] == '1' || e[0] == '2')) {
        if (create_dev_buffer(g.oob, std::uint64_t{gcn::kOobSlots} * gcn::kOobKinds * 4, true, true)) {
            std::memset(g.oob.map, 0, g.oob.size);
            gcn::set_oob_count(true);
            gcn::set_oob_control(e[0] == '2');
            host_log("gpu: out-of-range accesses counted (BBHOST_OOB_COUNT=%c): buffers past their records or binding, images at integer "
                     "coordinates outside the bound view's size, levels or layers; vertex, pixel and compute stages without a "
                     "tessellation or GS role, translated or lifted; the HLSL path off%s",
                     e[0], e[0] == '2' ? "; the control: every limit halved, one access in 256 past it counted" : "");
        }
    }

    // Mirror-map the whole direct memory so it can be imported in chunks.
    g.mirror_size = hle_kernel_dmem_size();
    g.mirror = hle_kernel_dmem_mirror();
    if (!g.mirror) {
        host_log("gpu: mirror mapping of dmem failed");
        return false;
    }
    choose_dmem_import();
    host_log("gpu: %s (Vulkan %u.%u), subgroup size %u (stages 0x%x: %s%s%s), host pointer alignment %u%s%s%s", props.deviceName,
             VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), subp.subgroupSize,
             g.subgroup_stages, (g.subgroup_stages & VK_SHADER_STAGE_VERTEX_BIT) ? "vertex " : "",
             (g.subgroup_stages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) ? "tess-eval " : "",
             (g.subgroup_stages & VK_SHADER_STAGE_COMPUTE_BIT) ? "compute" : "", g.host_pointer_align,
             g.has_maint8 ? ", maintenance8" : "", g.present_capable ? ", presentable" : "",
             g.has_tessellation ? ", tessellation" : "");
    // Before g.ok: the precompile workers wait for it, and their first
    // translations look in the cache. After the device's features went to the
    // translator above (they are part of every key).
    // The Shaders setting (host/options.cpp), before the first lift decides
    // the HLSL path's mode; BBHOST_HLSL wins over it.
    hlsl_set_setting(host_opt_index("shaders") != 1);
    if (const char* root = hle_fs_data_root(); root && *root) hlsl_set_folders(std::string(root) + "/bbhost");
    hlsl_set_name_lookup(&shader_bundle_file_name);
    translation_cache_load(translation_cache_path());
    g.ok = true;
    start_submit_thread();
    start_hang_watchdog();
    start_memory_reserve();
    stage_manifest_load(stage_manifest_path());
    {
        // A shipped pipeline list beside the executable (tools/pipeline_list.py),
        // or BBHOST_PIPELINE_LIST's (empty: none).
        const char* e = std::getenv("BBHOST_PIPELINE_LIST");
        std::string list = e ? std::string(e) : config_exe_dir() + "/pipeline-list.bin";
        if (!e && !std::filesystem::exists(list)) list.clear();
        shader_programs_set_eboot(config().eboot);
        pipeline_manifest_load(pipeline_manifest_path(), list);
    }
    if (const std::string cache = pipeline_cache_path(); !cache.empty()) {
        shader_sweep_start(cache.substr(0, cache.rfind('/') + 1) + "shader-sweep.txt", config().eboot);
    }
    return true;
}

// Extract the program at `code_va`: the ShaderBinaryInfo footer that follows
// the code in memory gives the length; otherwise decode to the last s_endpgm.
bool extract_program(std::uint64_t code_va, std::vector<std::uint32_t>& words, std::string& name) {
    if (!hle_kernel_va_mapped(code_va, 4)) {
        return false;
    }
    const auto* p = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(code_va));
    std::size_t limit = 0;
    while (limit < (1u << 20) && hle_kernel_va_mapped(code_va + limit, 4096)) {
        limit += 4096;
    }
    if (limit == 0) {
        return false;
    }
    for (std::size_t i = 0; i + 32 <= limit; i += 4) {
        if (std::memcmp(p + i, "OrbShdr", 7) == 0) {
            std::uint32_t lenfield;
            std::memcpy(&lenfield, p + i + 8, 4);
            const std::uint32_t length = lenfield >> 8;
            if (length && length <= i && (length & 3) == 0) {
                words.resize(length / 4);
                std::memcpy(words.data(), p, length);
                std::uint32_t h0;
                std::memcpy(&h0, p + i + 16, 4);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%08x", h0);
                name = buf;
                shader_patch_apply(name, words);  // host/shader_patch.h
                return true;
            }
            break;
        }
    }
    const std::size_t n = limit / 4;
    const gcn::Program prog = gcn::decode(reinterpret_cast<const std::uint32_t*>(p), n);
    std::uint32_t end = 0;
    for (const gcn::Inst& in : prog.insts) {
        if (in.enc == gcn::Enc::SOPP && in.op == 1) {
            end = in.offset + 4;
        }
        if (!prog.errors.empty() && in.offset >= prog.errors[0].offset) {
            break;
        }
    }
    if (!end) {
        return false;
    }
    words.resize(end / 4);
    std::memcpy(words.data(), p, end);
    name = "nofooter";
    return true;
}

void begin_recording_locked() {
    if (!g.recording) {
        if (stream_on()) {
            rec().begin(g.cmd_);  // the recorder begins it (kBegin)
        } else {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            const VkCommandBuffer cmd = g_cmd();
            const std::uint64_t t0 = host_clock_monotonic_ns();
            vkBeginCommandBuffer(cmd, &bi);
            bump(g_stream_stats.endbegin_ns, host_clock_monotonic_ns() - t0);
        }
        ++g.record_serial;  // dynamic state recorded so far is gone
        g.recording = true;
        twins_arm_wanted_locked();  // BBHOST_PARTICLE_TWINS: from this recording's first use of each
        gxvk_begin_recording_locked();  // GX buffers by handle: retired entries taken back
        busy_begin_locked(g.slots[g.slot]);
        gpu_checkpoint(kGpuMarkSubmission, g.flushes * 4 + 1);
        glitch_begin_recording_locked();
        occlusion_begin_recording_locked();
        if (g.profile) {
            Gpu::Slot& sl = g.slots[g.slot];
            rec().reset_query_pool(sl.qpool, 0, kQueriesPerSlot);
            sl.qreset = true;
            rec().write_timestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sl.qpool, kQueriesPerSlot - 2);
            sl.span = true;
        }
    }
}

// A params block into its slot in the ring, which is write-combined memory,
// as whole 64-byte lines: a line written in part stays in a write-combining
// buffer until the next store fence - the draw's publish - which then flushes
// it in pieces. The slot (kParamsStride) holds the rounded size; the
// binding's range stays sizeof(StageParams). (A block that is whole lines
// already, as it is now at 512 bytes, is copied as it is.)
// BBHOST_WC_LINES=0: the params blocks and the constant-buffer snapshots
// (render.cpp) written at their exact sizes.
const bool g_wc_whole_lines = [] {
    const char* e = std::getenv("BBHOST_WC_LINES");
    return !(e && e[0] == '0');
}();
constexpr std::size_t kParamsLines = (sizeof(gcn::StageParams) + 63) & ~std::size_t{63};
static_assert(kParamsLines <= kParamsStride, "a params slot holds its block in whole lines");
void put_params_lines(std::uint8_t* dst, const gcn::StageParams& params) {
    if constexpr (kParamsLines == sizeof(gcn::StageParams)) {
        std::memcpy(dst, &params, sizeof(params));
    } else {
        if (!g_wc_whole_lines) {
            std::memcpy(dst, &params, sizeof(params));
            return;
        }
        alignas(64) std::uint8_t lines[kParamsLines];
        std::memcpy(lines, &params, sizeof(params));
        std::memset(lines + sizeof(params), 0, kParamsLines - sizeof(params));
        std::memcpy(dst, lines, kParamsLines);
    }
}

std::uint32_t alloc_params_slot_locked(const gcn::StageParams& params, VkDescriptorBufferInfo& out) {
    if (draw_ctx()) list_resolve::count(list_resolve::kCtxOutParams);  // BBHOST_DRAW_CONTEXT's census
    std::uint32_t slot;
    if (RunDraw* const r = t_run_draw; r && r->part) {
        // A run part's draw: from its reservation. One past it stops the draw
        // (render.cpp kRunStopParams), which is then committed again in order:
        // the block goes over its own last slot meanwhile.
        if (r->params_next == r->params_end) {
            if (r->stop < 0) r->stop = kRunStopParams;
            slot = r->params_end - 1;
        } else {
            slot = r->params_next++;
        }
    } else {
        slot = g.queued++;
    }
    const std::uint32_t index = static_cast<std::uint32_t>(g.slot) * kMaxQueued * kStageSlots + slot;
    put_params_lines(static_cast<std::uint8_t*>(g.ubo_map) + static_cast<std::size_t>(index) * kParamsStride, params);
    out = {g.ubo, static_cast<VkDeviceSize>(index) * kParamsStride, sizeof(gcn::StageParams)};
    return slot;
}

void rewrite_params_slot_locked(const gcn::StageParams& params, const VkDescriptorBufferInfo& slot) {
    put_params_lines(static_cast<std::uint8_t*>(g.ubo_map) + static_cast<std::size_t>(slot.offset), params);
}

// Sets come from the current slot's pool eight at a time: a draw takes two,
// and one vkAllocateDescriptorSets per set was ~3% of the command processor.
// The spares go with the pool's reset (retire_slot_locked); the pool has room
// for them (kSpareSets).
VkDescriptorSet alloc_set_locked(VkDescriptorSetLayout layout) {
    if (draw_ctx()) list_resolve::count(list_resolve::kCtxOutSets);  // BBHOST_DRAW_CONTEXT's census
    if (!layout) layout = g.set_layout;
    auto& spares = g.slots[g.slot].spare_sets;
    std::vector<VkDescriptorSet>* list = nullptr;
    for (auto& e : spares) {
        if (e.first == layout) list = &e.second;
    }
    if (list && !list->empty()) {
        const VkDescriptorSet set = list->back();
        list->pop_back();
        return set;
    }
    constexpr std::uint32_t kBatch = 8;
    VkDescriptorSetLayout layouts[kBatch];
    for (VkDescriptorSetLayout& l : layouts) l = layout;
    VkDescriptorSet sets[kBatch] = {};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = g.desc_pool;
    dsai.descriptorSetCount = kBatch;
    dsai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(g.device, &dsai, sets) != VK_SUCCESS) {
        dsai.descriptorSetCount = 1;  // the pool is nearly full: one, as before
        if (vkAllocateDescriptorSets(g.device, &dsai, sets) != VK_SUCCESS) return VK_NULL_HANDLE;
        return sets[0];
    }
    if (!list) list = &spares.emplace_back(layout, std::vector<VkDescriptorSet>{}).second;
    list->insert(list->end(), sets + 1, sets + kBatch);
    return sets[0];
}

std::atomic<std::uint64_t> g_compare_guarded{0};
VkImageView compare_guard_view(VkImageView view, const std::uint32_t* t, std::uint32_t dim, bool arrayed) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_COMPARE_GUARD");
        return !(e && e[0] == '0');
    }();
    if (!on) return view;
    // R32F (DATA_FORMAT 4, NUM_FORMAT 7: the shadow maps' T#s) where the
    // device compares in it keeps its view; anything else is no depth format.
    static const bool r32f_compares = [] {
        VkFormatProperties3 f3{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 f2{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
        f2.pNext = &f3;
        vkGetPhysicalDeviceFormatProperties2(g.phys, VK_FORMAT_R32_SFLOAT, &f2);
        return (f3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT) != 0;
    }();
    const bool r32f = ((t[1] >> 20) & 0x3f) == 4 && ((t[1] >> 26) & 0xf) == 7;
    if (r32f && r32f_compares) return view;
    const auto it = g.dummy_depth_views.find((dim << 1) | (arrayed ? 1u : 0u));
    if (it == g.dummy_depth_views.end()) return view;
    g_compare_guarded.fetch_add(1, std::memory_order_relaxed);
    return it->second;
}

VkImageView dummy_view_for(std::uint32_t dim, bool arrayed, bool zero) {
    if (zero && dim == 1 && !arrayed && g.dummy_zero_view) return g.dummy_zero_view;
    if (dim == 1 && !arrayed) return g.dummy_view;
    const std::uint32_t key = (dim << 1) | (arrayed ? 1u : 0u);
    if (zero) {  // an invalid T#'s zeros, where the shape has them (create_dim_dummies)
        if (const auto z = g.dummy_dim_views.find(key | 0x100u); z != g.dummy_dim_views.end()) return z->second;
    }
    auto it = g.dummy_dim_views.find(key);
    return it == g.dummy_dim_views.end() ? VK_NULL_HANDLE : it->second;
}

void bind_dummy_images(VkDescriptorSet set, const gcn::TranslateResult& meta, std::vector<VkWriteDescriptorSet>& writes,
                       std::vector<VkDescriptorImageInfo>& infos, VkDescriptorImageInfo& smp_info) {
    smp_info = {g.dummy_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    for (const gcn::ImageBinding& b : meta.images) {
        const VkImageView view = dummy_view_for(b.dim, b.arrayed);
        if (!view) continue;
        infos.push_back({VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL});
        VkWriteDescriptorSet wi{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wi.dstSet = set;
        wi.dstBinding = b.binding;
        wi.descriptorCount = 1;
        wi.descriptorType = b.storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        wi.pImageInfo = &infos.back();
        writes.push_back(wi);
        g.dummy_images.fetch_add(1);
    }
    for (const gcn::SamplerBinding& b : meta.samplers) {
        VkWriteDescriptorSet ws{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        ws.dstSet = set;
        ws.dstBinding = b.binding;
        ws.descriptorCount = 1;
        ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        ws.pImageInfo = &smp_info;
        writes.push_back(ws);
    }
}

namespace {

// Recognises the engine's clear shader: s_buffer_load_dwordx4 of the value,
// v_movs, buffer_store_format_xyzw / buffer_store_dwordx4 to one V#, and
// nothing else touching memory.
void detect_fill(const gcn::Program& prog, ComputePipeline& pl) {
    int load_dst = -1, load_base = -1;
    std::uint32_t load_off = 0;
    int store_srsrc = -1;
    bool other_memory = false, format_store = false;
    int stores = 0;
    for (const gcn::Inst& in : prog.insts) {
        switch (in.enc) {
            case gcn::Enc::SMRD:
                if (in.op == 10 && in.imm_flag && load_dst < 0) {  // s_buffer_load_dwordx4
                    load_dst = in.sdst;
                    load_base = in.src0;
                    load_off = static_cast<std::uint32_t>(in.imm);
                } else {
                    other_memory = true;
                }
                break;
            case gcn::Enc::MUBUF:
                if ((in.op == 7 || in.op == 30) && (store_srsrc < 0 || store_srsrc == in.srsrc)) {
                    store_srsrc = in.srsrc;
                    format_store = format_store || in.op == 7;
                    ++stores;
                } else {
                    other_memory = true;
                }
                break;
            case gcn::Enc::MTBUF:
            case gcn::Enc::MIMG:
            case gcn::Enc::DS:
                other_memory = true;
                break;
            default:
                break;
        }
    }
    if (other_memory || load_dst < 0 || store_srsrc < 0 || stores == 0) return;
    if (store_srsrc + 3 >= 16 || load_base + 3 >= 16) return;  // both V#s must be user data
    pl.fill = true;
    pl.fill_dst_sgpr = store_srsrc;
    pl.fill_cb_sgpr = load_base;
    pl.fill_cb_dw = load_off;
    pl.fill_format = format_store;
}

void detect_copy(const gcn::Program& prog, ComputePipeline& pl) {
    int src = -1, dst = -1;
    bool other_memory = false;
    for (const gcn::Inst& in : prog.insts) {
        switch (in.enc) {
            case gcn::Enc::MUBUF:
                if (in.op == 0 && src < 0) src = in.srsrc;            // buffer_load_format_x
                else if (in.op == 4 && dst < 0) dst = in.srsrc;       // buffer_store_format_x
                else other_memory = true;
                break;
            case gcn::Enc::SMRD: case gcn::Enc::MTBUF: case gcn::Enc::MIMG: case gcn::Enc::DS:
                other_memory = true;
                break;
            default:
                break;
        }
    }
    if (other_memory || src < 0 || dst < 0 || src + 3 >= 16 || dst + 3 >= 16) return;
    pl.copy = true;
    pl.copy_src_sgpr = src;
    pl.copy_dst_sgpr = dst;
}

// The two shapes that move images. The buffer copy above insists on
// MUBUF in and MUBUF out and treats every other memory encoding as
// disqualifying; these two read their rectangle from a constant buffer, so
// SMRD has to be allowed. Nothing else may touch memory - the point of
// recognising a dispatch is that its whole effect is the copy.
void detect_image_copy(const gcn::Program& prog, ComputePipeline& pl) {
    int mimg_loads = 0, mimg_stores = 0, mubuf_loads = 0, src_sgpr = -1;
    bool other_memory = false;
    for (const gcn::Inst& in : prog.insts) {
        switch (in.enc) {
            case gcn::Enc::MIMG:
                if (in.op == 0 || in.op == 1) ++mimg_loads;        // image_load(_mip)
                else if (in.op == 8 || in.op == 9) ++mimg_stores;  // image_store(_mip)
                else other_memory = true;
                break;
            case gcn::Enc::MUBUF:
                // buffer_load_format_{x,xy,xyz,xyzw} are ops 0..3.
                if (in.op <= 3 && mubuf_loads == 0) {
                    ++mubuf_loads;
                    src_sgpr = in.srsrc;
                } else {
                    other_memory = true;
                }
                break;
            case gcn::Enc::SMRD:  // the rectangle comes from a constant buffer
                break;
            case gcn::Enc::MTBUF: case gcn::Enc::DS:
                other_memory = true;
                break;
            default:
                break;
        }
    }
    if (other_memory || mimg_stores != 1) return;
    if (mimg_loads == 1 && mubuf_loads == 0) {
        pl.image_blit = true;
    } else if (mimg_loads == 0 && mubuf_loads == 1 && src_sgpr >= 0 && src_sgpr + 3 < 16) {
        pl.buffer_to_image = true;
        pl.img_src_sgpr = src_sgpr;
    }
}

// ---- compute shaders compiled when GX creates them ----
// A dispatch's pipeline is keyed by its program, COMPUTE_PGM_RSRC1/2, the
// thread counts, and the dimensions and sampler modes of what it binds. The
// shader's container holds the same registers (CsStageRegs, from the third word
// of the register block at Shdr + 0x10), and image dimensions are predicted from
// the program, so a worker (render.cpp's precompile pool) compiles the pipeline
// when GX creates the shader. The first dispatch takes it, or waits for the build.
std::uint64_t compute_key(std::uint64_t code_hash, std::uint32_t rsrc1, std::uint32_t rsrc2, const std::uint32_t* threads,
                          const std::vector<std::pair<std::uint32_t, bool>>& dims, const std::vector<bool>& modes) {
    std::uint64_t key = code_hash;
    key = fnv1a(&rsrc1, 4, key);
    key = fnv1a(&rsrc2, 4, key);
    key = fnv1a(threads, 12, key);
    for (const auto& dm : dims) {
        key = fnv1a(&dm.first, 4, key);
        key = fnv1a(&dm.second, 1, key);
    }
    for (bool mode : modes) key = fnv1a(&mode, sizeof(mode), key);
    return key;
}

// ---- compute shaders' buffers through storage buffers (BBHOST_CS_BUFFERS) ----
// A dispatch bound no buffers: its shader walked the page table (two
// dependent 64-bit loads, then the data) for every dword it read - every
// constant, every table entry, every typed load of a particle record. The
// particle update 2da7fe60, ~4,900 instructions on the console, came out of
// AMD's compiler at 36,677 with 1,106 page-table loads and 251 VGPRs. Each
// compute shader now gets a second translation beside the first, with its
// constant buffers, tables and typed loads read from storage buffers
// (TranslateOptions::cb_ssbo) and no page-table fallback for them, its
// buffers pushed into set 1 (Gpu::cs_buffers_layout) the way a draw's are
// (render.cpp, resolve_stage_buffers): bound where they are, or copied as the
// dispatch records for memory the game writes on the CPU. A dispatch every one
// of whose buffers binds runs it; any other, the translation that walks. In
// Kyo's hall on the Radeon 8060S every dispatch bound (30,345 of a run, 38
// shaders): 2da7fe60 22,594 instructions, 180 page-table loads (its stores'),
// 207 VGPRs, 0.55 -> 0.43 ms a frame; all compute 0.96 -> 0.81.
// BBHOST_CS_BUFFERS=0: every dispatch walks, as before.
bool cs_buffers_on() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_CS_BUFFERS");
        return !(e && e[0] == '0');
    }();
    return on && g.cb_ssbo && g.cs_buffers_layout != VK_NULL_HANDLE;
}
// BBHOST_LIFT_CS_BUFFERS: a lifted compute program's variant with its buffers
// pushed is the lift of its lean translation (on unless 0; 0: a lifted
// program has none, and its dispatches walk the page table as before).
bool cs_lift_buffers_on() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_LIFT_CS_BUFFERS");
        return !(e && e[0] == '0');
    }();
    return on;
}
// The variant's key for the optimizer (ComputeLean), beside its pipeline's.
constexpr std::uint64_t kCsBuffersTag = 0xcb55b0f0e5000000ull;
std::atomic<std::uint64_t> g_cs_lean_built{0}, g_cs_lean_none{0}, g_cs_lean_failed{0}, g_cs_lean_dispatches{0}, g_cs_lean_fallbacks{0},
    g_cs_lean_lifted{0}, g_cs_lean_lift_refused{0};

// `buffers`: the ComputeLean translation.
gcn::TranslateOptions compute_options(std::uint32_t rsrc1, std::uint32_t rsrc2, const std::uint32_t* threads,
                                      const std::vector<std::pair<std::uint32_t, bool>>& dims, const std::vector<bool>& modes,
                                      bool buffers = false) {
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Compute;
    o.gds_counters = host_gpu_gds_counters();
    o.gds_subgroup = host_gpu_gds_subgroup();
    o.rsrc1 = rsrc1;
    o.rsrc2 = rsrc2;
    o.image_dims = dims;
    o.sampler_force_unnormalized = modes;
    o.max_lds_bytes = g.max_lds_bytes;
    o.cs_threads[0] = threads[0];
    o.cs_threads[1] = threads[1];
    o.cs_threads[2] = threads[2];
    o.exec_known = g.exec_known;
    if (buffers) {
        o.cb_ssbo = true;
        o.cb_no_fallback = true;
        o.cb_descriptor_set = 1;               // pushed (Gpu::cs_buffers_layout), from binding 112
        o.cb_bias_zero = cb_exact_bindings();  // each at its V# base, as a draw's (resolve_stage_buffers)
    }
    return o;
}

// The ComputeLean beside a pipeline of `key`, translated from the same
// program, inputs and registers: null when it would bind nothing, when its
// translation or pipeline failed, when the program stores through a V# that is
// not traced (a write the dispatch could not keep its copies off), or when it
// binds images, samplers or counters otherwise than `regular` - the dispatch
// binds those from one list for either.
std::shared_ptr<ComputeLean> build_compute_lean(const gcn::Program& prog, std::uint32_t rsrc1, std::uint32_t rsrc2, const std::uint32_t* threads,
                                                const std::vector<std::pair<std::uint32_t, bool>>& dims, const std::vector<bool>& modes,
                                                const gcn::TranslateResult& regular, std::uint64_t key, const std::string& name, bool lift) {
    if (!cs_buffers_on() || (lift && !cs_lift_buffers_on())) return nullptr;
    auto lean = std::make_shared<ComputeLean>();
    const gcn::TranslateOptions lean_options = compute_options(rsrc1, rsrc2, threads, dims, modes, true);
    lean->meta = translate_cached(prog, lean_options);
    const auto same = [&] {
        const gcn::TranslateResult& m = lean->meta;
        if (m.images.size() != regular.images.size() || m.samplers.size() != regular.samplers.size() || m.gds_counters != regular.gds_counters ||
            m.lds_bytes != regular.lds_bytes) {
            return false;
        }
        for (std::size_t k = 0; k < m.images.size(); ++k) {
            const gcn::ImageBinding& a = m.images[k];
            const gcn::ImageBinding& b = regular.images[k];
            if (a.path.str() != b.path.str() || a.binding != b.binding || a.storage != b.storage || a.r128 != b.r128 || a.dim != b.dim ||
                a.arrayed != b.arrayed || a.depth != b.depth || a.cube != b.cube || a.kind != b.kind) {
                return false;
            }
        }
        for (std::size_t k = 0; k < m.samplers.size(); ++k) {
            const gcn::SamplerBinding& a = m.samplers[k];
            const gcn::SamplerBinding& b = regular.samplers[k];
            if (a.path.str() != b.path.str() || a.binding != b.binding || a.compare != b.compare) return false;
        }
        return true;
    };
    std::string why;
    if (!lean->meta.ok()) {
        why = "translation: " + lean->meta.errors[0];
    } else if (lean->meta.buffers.empty()) {
        g_cs_lean_none.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    } else if (lean->meta.untraced_stores) {
        why = "stores through an untraced V#";
    } else if (!same()) {
        why = "images, samplers, counters or LDS differ";
    }
    // A lifted program's variant is the lift of this translation, which reads
    // what the translation binds from the pushed buffers and walks the rest
    // as it does (gcn::lift_compute_shader); refused, the program has none and
    // keeps its walking lift.
    if (why.empty() && lift) {
        hlsl_name_next(name);
        gcn::LiftResult l = lift_compute_cached(prog, lean_options, lean->meta);
        if (l.ok()) {
            lean->meta.spirv = std::move(l.spirv);
            lean->lifted = true;
        } else {
            if (g_cs_lean_lift_refused.fetch_add(1, std::memory_order_relaxed) < 16) {
                host_log("gpu: compute shader %s: its lean translation's lift was refused (%s); its dispatches walk the page table", name.c_str(),
                         l.rejections.empty() ? "no module" : l.rejections[0].c_str());
            }
            return nullptr;
        }
    }
    if (why.empty()) {
        if (const char* e = std::getenv("BBHOST_DUMP_SPIRV"); e && e[0] == '1') {
            host_mkdir("build/spv");
            if (FILE* f = std::fopen(("build/spv/" + name + "-cs-buffers.spv").c_str(), "wb")) {
                std::fwrite(lean->meta.spirv.data(), 4, lean->meta.spirv.size(), f);
                std::fclose(f);
            }
        }
        // DXC's module of a lift (BBHOST_HLSL) the driver refuses: the lift's
        // own, as pipeline_for does for the walking lift.
        for (bool own = false;;) {
            VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smci.codeSize = lean->meta.spirv.size() * 4;
            smci.pCode = lean->meta.spirv.data();
            if (vkCreateShaderModule(g.device, &smci, nullptr, &lean->module) != VK_SUCCESS) {
                lean->module = VK_NULL_HANDLE;
                why = "shader module";
            } else {
                lean->pipeline = first_compute_pipeline(lean->module, key ^ kCsBuffersTag, name, lean->meta.spirv, lean->standin,
                                                        name + " (buffers pushed)", g.cs_buffers_layout);
                if (!lean->pipeline) why = "pipeline";
            }
            std::vector<std::uint32_t> lift;
            if (why.empty() || !lean->lifted || own || !hlsl_lift_of(lean->meta.spirv, lift)) break;
            if (lean->module) vkDestroyShaderModule(g.device, lean->module, nullptr);
            lean->module = VK_NULL_HANDLE;
            lean->meta.spirv = std::move(lift);
            own = true;
            why.clear();
        }
    }
    if (!why.empty()) {
        if (g_cs_lean_failed.fetch_add(1, std::memory_order_relaxed) < 16) {
            host_log("gpu: compute shader %s: no variant with its buffers pushed (%s); its dispatches walk the page table", name.c_str(), why.c_str());
        }
        if (lean->module) vkDestroyShaderModule(g.device, lean->module, nullptr);
        return nullptr;
    }
    g_cs_lean_built.fetch_add(1, std::memory_order_relaxed);
    if (lean->lifted) g_cs_lean_lifted.fetch_add(1, std::memory_order_relaxed);
    return lean;
}

// BBHOST_LIFT_COMPUTE: compute shaders run as their typed lifts
// (gcn::lift_compute_shader) - on by default, 0 keeps every compute shader
// translated (an A/B switch). A program also needs BBHOST_DECOMP to select it
// ("all", the default; 0 none; a list of program hashes), as the pixel and
// vertex lifts do (render.cpp decomp_selects). The lift binds what the
// translation binds - the params block, its images and samplers, the GDS
// counters - so the dispatch's descriptor set is the same either way; a
// program the lifter refuses keeps its translation, and the exit report
// counts those by reason.
bool compute_lift_selects(const std::string& name) {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_LIFT_COMPUTE");
        return !(e && e[0] == '0');
    }();
    static const std::pair<bool, std::set<std::string>> selection = [] {
        std::pair<bool, std::set<std::string>> s{true, {}};
        const char* e = std::getenv("BBHOST_DECOMP");
        if (!e || !*e || std::strcmp(e, "all") == 0) return s;
        s.first = false;
        if (std::strcmp(e, "0") == 0 || std::strcmp(e, "1") == 0) return s;  // 1: the pilot pixel shader alone
        std::string list(e);
        for (std::size_t at = 0; at <= list.size();) {
            const std::size_t comma = std::min(list.find(',', at), list.size());
            if (comma > at) s.second.insert(list.substr(at, comma - at));
            at = comma + 1;
        }
        return s;
    }();
    return on && (selection.first || selection.second.count(name) != 0);
}
// BBHOST_TEST_HLSL_REFUSE=<n> (checks): every n-th compute pipeline to be made
// from DXC's module is taken as refused before it is asked for (1: every
// one), so the rebuild from the lift's own module runs on a driver that
// refuses none (render.cpp has the same for graphics pipelines).
bool hlsl_test_refused_compute(const std::vector<std::uint32_t>& spirv) {
    static const int every = [] {
        const char* e = std::getenv("BBHOST_TEST_HLSL_REFUSE");
        return e ? std::max(0, std::atoi(e)) : 0;
    }();
    if (!every || !hlsl_on()) return false;
    std::vector<std::uint32_t> lift;
    if (!hlsl_lift_of(spirv, lift)) return false;
    static std::atomic<std::uint64_t> seen{0};
    return seen.fetch_add(1, std::memory_order_relaxed) % static_cast<std::uint64_t>(every) == 0;
}

// The program's lift in place of its translation's SPIR-V, when selected and
// accepted; `lifted` and `why` (the first rejection, or empty) say which.
// Every compute program a pipeline was made for, at GX's creation or at a
// dispatch, by how it runs: empty for its lift, or why it runs translated
// (compute_lift_report). A program counts as lifted while every pipeline of
// it is.
std::mutex g_lift_decided_mu;
std::map<std::string, std::string> g_lift_decided;
void note_compute_lift(const std::string& name, bool lifted, const std::string& why) {
    std::lock_guard<std::mutex> lk(g_lift_decided_mu);
    std::string& at = g_lift_decided[name];
    if (!lifted && at.empty()) at = why.empty() ? std::string("not asked for (BBHOST_LIFT_COMPUTE=0 or BBHOST_DECOMP)") : why;
}
void apply_compute_lift(const gcn::Program& prog, const gcn::TranslateOptions& o, gcn::TranslateResult& meta, const std::string& name, bool& lifted,
                        std::string& why) {
    lifted = false;
    why.clear();
    if (!meta.ok()) return;
    if (!compute_lift_selects(name)) {
        note_compute_lift(name, false, why);
        return;
    }
    hlsl_name_next(name);
    gcn::LiftResult l = lift_compute_cached(prog, o, meta);
    if (l.ok()) {
        meta.spirv = std::move(l.spirv);
        lifted = true;
        note_compute_lift(name, true, why);
        return;
    }
    why = l.rejections.empty() ? std::string("no module") : l.rejections[0];
    note_compute_lift(name, false, why);
}

struct PrecompiledCompute {
    struct Entry {
        bool done = false, failed = false;  // failed: nothing to take (a failed build, taken, or claimed by a dispatch)
        bool gave_up = false;               // a dispatch stopped waiting for it: its dispatches skip until it is done
        gcn::TranslateResult meta;
        VkShaderModule module = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        bool standin = false;  // unoptimized, the optimized one queued (first_compute_pipeline)
        std::shared_ptr<ComputeLean> lean;  // build_compute_lean
        bool lifted = false;   // ComputePipeline::lifted, lift_why
        std::string lift_why;
        std::uint64_t us = 0;  // translation and compile
    };
    enum State { kQueued, kCompiling, kDone, kFailed };
    struct Record {  // by shader name: what it was compiled with
        State state = kQueued;
        std::string why;  // kFailed
        std::uint64_t created_flip = 0;
        std::uint32_t rsrc1 = 0, rsrc2 = 0, threads[3] = {};
        std::vector<std::pair<std::uint32_t, bool>> dims;
        std::vector<bool> modes;
    };
    std::mutex mu;
    std::condition_variable cv;
    std::unordered_map<std::uint64_t, Entry> by_key;   // under mu
    std::unordered_map<std::string, Record> records;  // under mu
    std::uint64_t compiled = 0, failed = 0, skipped = 0, compile_us = 0, taken = 0, waited = 0, wait_us = 0;  // under mu
    std::uint64_t miss_never = 0, miss_queued = 0, miss_failed = 0, miss_rsrc = 0, miss_threads = 0, miss_dims = 0, miss_modes = 0,
                  miss_other = 0;  // under mu
    std::vector<std::string> notes;  // under mu: the first dispatch builds it did not serve, and why
    // Pipelines in the driver's compiler now, for the watchdog: when they went
    // in, and their SPIR-V (owned by the compiling thread until it leaves).
    struct Compiling {
        const std::vector<std::uint32_t>* spirv = nullptr;
        std::chrono::steady_clock::time_point since{};
        bool written = false;
    };
    std::unordered_map<std::string, Compiling> compiling;  // under mu
    std::uint64_t gave_up = 0;  // under mu
    std::atomic<std::uint64_t> skipped_dispatches{0};
};
PrecompiledCompute g_cs_pre;

}  // namespace

namespace {

// One pipeline into the driver's compiler, recorded for the watchdog while it
// is there.
struct CompilingScope {
    std::string name;
    CompilingScope(const std::string& n, const std::vector<std::uint32_t>* spirv) : name(n) {
        std::lock_guard<std::mutex> lk(g_cs_pre.mu);
        g_cs_pre.compiling[name] = {spirv, std::chrono::steady_clock::now(), false};
    }
    ~CompilingScope() {
        std::lock_guard<std::mutex> lk(g_cs_pre.mu);
        g_cs_pre.compiling.erase(name);
    }
};

// ---- compute pipelines and the driver's optimizer (BBHOST_CS_OPTIMIZE) ----
// 2026-09-28: AMD's Windows driver (amdvlk64, Radeon 8060S) was
// still compiling four of the game's compute shaders minutes after GX created
// them, its memory climbing, and the command processor waiting on one of them:
// the loading screen after a new character never ended. Every AMD device then
// built every compute pipeline without the optimizer - and the cause, the
// program-counter dispatcher those four had, was removed the same day (loop
// blocks: 24 s -> 6.5 s cold for 42f2a521 on NVIDIA). What stayed was -O0 code
// for the two 4-5k-instruction loop shaders that run every third frame and the
// small ones that run every frame (gpu-time item 6).
//
// By default now (kBackground, every vendor) nothing waits for the optimizer:
//   - a pipeline the pipeline cache has optimized is taken at once
//     (VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT answers without
//     compiling) - every start after the first;
//   - otherwise the unoptimized pipeline (VK_PIPELINE_CREATE_DISABLE_
//     OPTIMIZATION_BIT, the build every AMD start made until now) stands in for
//     its first dispatches, and a bb-cs-optimize thread builds the optimized
//     one, which the shader's next dispatch swaps in (adopt_optimized_compute) -
//     the graphics pipelines' optimized relink (render.cpp,
//     BBHOST_LIBRARY_RELINK), for compute. Each optimized build goes to a
//     pipeline cache of its own, merged when it is done into g.opt_cache and
//     that into the main one when it is saved (merge_side_caches): the next
//     start takes them at once, and a build that takes minutes holds no cache
//     a save or the exit waits for.
// Such a build costs one of those threads, not the game; one still running at
// exit after a minute is written to <data>/bbhost/compute-unoptimized.txt, and
// later starts keep that shader unoptimized (delete the file to try again).
// BBHOST_CS_OPTIMIZE=0: never optimized; =1: optimized and waited for (every
// other vendor's default until now). BBHOST_CS_OPTIMIZE_THREADS=<n>: the
// threads (default 2); BBHOST_CS_OPTIMIZE_DELAY_MS=<ms> (checks) holds each
// optimized build that long, to watch the stand-ins on a fast driver.
enum class CsOptimize { kNever, kAlways, kBackground };
enum class CsBuild { kOptimized, kUnoptimized, kOptimizedCached };

struct CsOptimizer {
    struct Job {
        std::uint64_t key = 0;
        std::string name;
        std::vector<std::uint32_t> spirv;
        std::uint64_t flip = 0;        // when its stand-in was built
        std::uint64_t standin_us = 0;  // and how long that took
        VkPipelineLayout layout = VK_NULL_HANDLE;  // null: Gpu::pipe_layout
    };
    struct Running {
        std::string name;  // empty: the thread is idle
        std::chrono::steady_clock::time_point since{};
    };
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> jobs;                                  // under mu
    std::unordered_set<std::uint64_t> queued;              // under mu: one optimized build a key
    std::unordered_map<std::uint64_t, VkPipeline> ready;   // under mu: the optimized pipelines built
    std::atomic<std::uint64_t> gen{0};                     // bumped with each one put in `ready`
    std::vector<Running> running;                          // under mu: by thread, the build in the compiler now
    std::unordered_set<std::string> keep_unoptimized;      // compute-unoptimized.txt, read once
    std::uint64_t slowest_us = 0;                          // under mu
    std::string slowest;                                   // under mu
    int threads = 2;
    std::atomic<std::uint64_t> from_cache{0}, standins{0}, standins_cp{0}, built{0}, failed{0}, build_us{0}, adopted{0}, kept{0};
    std::atomic<std::uint64_t> window_built{0}, window_us{0}, window_adopted{0};
};
CsOptimizer g_cs_opt;
// g.opt_cache is a merge's destination (each build's cache into it) and a
// source (it into the main cache): Vulkan has a merge's destination externally
// synchronized, so both go under this.
std::mutex g_opt_cache_mu;
// The precompile workers' builds at creation, told apart from the command
// processor's for the report (set in precompile_compute).
thread_local bool t_compute_worker = false;
// A build still in the compiler at exit after this long is taken for one that
// may never end (the dispatcher shapes of 2026-09-28 did not); a cold world
// load's slowest compute compiles were 13 s each.
constexpr std::uint64_t kCsGiveUpS = 60;

// <data root>/bbhost/compute-unoptimized.txt, beside the pipeline cache.
std::string compute_unoptimized_path() {
    std::string path = pipeline_cache_path();
    if (path.empty()) return path;
    return path.substr(0, path.rfind('/') + 1) + "compute-unoptimized.txt";
}

CsOptimize cs_optimize_mode() {
    static const CsOptimize mode = [] {
        const char* e = std::getenv("BBHOST_CS_OPTIMIZE");
        const int v = e && e[0] ? std::atoi(e) : -1;
        const CsOptimize m = v == 0 ? CsOptimize::kNever : v == 1 ? CsOptimize::kAlways : CsOptimize::kBackground;
        if (m == CsOptimize::kBackground) {
            if (const std::string path = compute_unoptimized_path(); !path.empty()) {
                std::ifstream in(path);
                for (std::string line; std::getline(in, line);) {
                    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
                    if (!line.empty()) g_cs_opt.keep_unoptimized.insert(line);
                }
            }
            if (const char* t = std::getenv("BBHOST_CS_OPTIMIZE_THREADS")) g_cs_opt.threads = std::clamp(std::atoi(t), 1, 8);
            host_log("gpu: compute pipelines: the driver's optimizer in the background - one the pipeline cache does not have optimized "
                     "is built unoptimized for its first dispatches and swapped for the optimized one, built on %d bb-cs-optimize "
                     "thread%s (BBHOST_CS_OPTIMIZE=0 never optimizes, =1 waits for the optimizer)%s; %zu kept unoptimized "
                     "(compute-unoptimized.txt)",
                     g_cs_opt.threads, g_cs_opt.threads == 1 ? "" : "s",
                     g.cache_control ? "" : "; no pipeline creation cache control: each start builds the optimized ones again",
                     g_cs_opt.keep_unoptimized.size());
        } else {
            host_log("gpu: compute pipelines built %s (BBHOST_CS_OPTIMIZE=%d; unset builds them in the background)",
                     m == CsOptimize::kNever ? "without the driver's optimizer" : "with the driver's optimizer, waited for", v);
        }
        return m;
    }();
    return mode;
}

// One compute pipeline from `module`. kOptimizedCached: only if the pipeline
// cache has it optimized, VK_NULL_HANDLE at once otherwise. `own_cache`: a
// bb-cs-optimize build's, instead of the main cache.
VkPipeline create_compute(VkShaderModule module, CsBuild how, const std::string& watch, const std::vector<std::uint32_t>* spirv,
                          VkPipelineCache own_cache = VK_NULL_HANDLE, VkPipelineLayout layout = VK_NULL_HANDLE) {
    if (how == CsBuild::kOptimizedCached && !g.cache_control) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.layout = layout ? layout : g.pipe_layout;
    cpci.flags = how == CsBuild::kUnoptimized        ? VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT
                 : how == CsBuild::kOptimizedCached ? VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT
                                                     : 0;
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r;
    const ValidationScope validation(watch.c_str());
    if (how == CsBuild::kOptimizedCached) {
        r = vkCreateComputePipelines(g.device, PipelineCacheUse().cache, 1, &cpci, nullptr, &p);  // VK_PIPELINE_COMPILE_REQUIRED: not there
    } else if (own_cache != VK_NULL_HANDLE) {
        CompilingScope compiling(watch, spirv);
        r = vkCreateComputePipelines(g.device, own_cache, 1, &cpci, nullptr, &p);
    } else {
        CompilingScope compiling(watch, spirv);
        r = vkCreateComputePipelines(g.device, PipelineCacheUse().cache, 1, &cpci, nullptr, &p);
    }
    if (r == VK_SUCCESS) vk_debug_name(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<std::uint64_t>(p), watch.c_str());
    return r == VK_SUCCESS ? p : VK_NULL_HANDLE;
}

std::uint64_t cs_us_since(std::chrono::steady_clock::time_point t0) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}

// bb-cs-optimize: the optimized builds, behind everything else.
void cs_optimize_thread(int index) {
    host_thread_set_name("bb-cs-optimize");
    static const int delay_ms = [] {
        const char* d = std::getenv("BBHOST_CS_OPTIMIZE_DELAY_MS");
        return d ? std::atoi(d) : 0;
    }();
    for (;;) {
        CsOptimizer::Job job;
        {
            std::unique_lock<std::mutex> lk(g_cs_opt.mu);
            g_cs_opt.running[index].name.clear();
            g_cs_opt.cv.wait(lk, [] { return !g_cs_opt.jobs.empty(); });
            job = std::move(g_cs_opt.jobs.front());
            g_cs_opt.jobs.pop_front();
            g_cs_opt.running[index] = {job.name, std::chrono::steady_clock::now()};
        }
        if (delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        const auto t0 = std::chrono::steady_clock::now();
        VkShaderModule module = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = job.spirv.size() * 4;
        smci.pCode = job.spirv.data();
        VkPipeline p = VK_NULL_HANDLE;
        if (vkCreateShaderModule(g.device, &smci, nullptr, &module) == VK_SUCCESS) {
            char watch[32];
            std::snprintf(watch, sizeof(watch), " (optimized, %04llx)", static_cast<unsigned long long>(job.key & 0xffff));
            // Its own cache while it compiles, merged into g.opt_cache after.
            VkPipelineCache own = VK_NULL_HANDLE;
            VkPipelineCacheCreateInfo pcci{};
            pcci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
            if (g.opt_cache == VK_NULL_HANDLE || vkCreatePipelineCache(g.device, &pcci, nullptr, &own) != VK_SUCCESS) own = VK_NULL_HANDLE;
            p = create_compute(module, CsBuild::kOptimized, job.name + watch, &job.spirv, own, job.layout);
            if (own != VK_NULL_HANDLE) {
                if (p) {
                    std::lock_guard<std::mutex> lk(g_opt_cache_mu);
                    vkMergePipelineCaches(g.device, g.opt_cache, 1, &own);
                }
                vkDestroyPipelineCache(g.device, own, nullptr);
            }
            vkDestroyShaderModule(g.device, module, nullptr);  // the pipeline keeps what it needs
        }
        const std::uint64_t us = cs_us_since(t0);
        {
            std::lock_guard<std::mutex> lk(g_cs_opt.mu);
            if (p) {
                g_cs_opt.ready[job.key] = p;
                if (us > g_cs_opt.slowest_us) {
                    g_cs_opt.slowest_us = us;
                    g_cs_opt.slowest = job.name;
                }
            }
        }
        if (p) {
            g_cs_opt.built.fetch_add(1, std::memory_order_relaxed);
            g_cs_opt.build_us.fetch_add(us, std::memory_order_relaxed);
            g_cs_opt.window_built.fetch_add(1, std::memory_order_relaxed);
            g_cs_opt.window_us.fetch_add(us, std::memory_order_relaxed);
            g_cs_opt.gen.fetch_add(1, std::memory_order_release);
        } else {
            g_cs_opt.failed.fetch_add(1, std::memory_order_relaxed);
        }
        host_log("gpu: compute shader %s %s in %llu ms on bb-cs-optimize (unoptimized it took %llu ms, and stood in from flip %llu; "
                 "now flip %llu)%s",
                 job.name.c_str(), p ? "optimized" : "not optimized: the build failed", static_cast<unsigned long long>(us / 1000),
                 static_cast<unsigned long long>(job.standin_us / 1000), static_cast<unsigned long long>(job.flip),
                 static_cast<unsigned long long>(hle_video_flip_count()), p ? "" : "; the unoptimized one stays");
    }
}

void queue_compute_optimize(std::uint64_t key, const std::string& name, const std::vector<std::uint32_t>& spirv, std::uint64_t standin_us,
                            VkPipelineLayout layout) {
    static std::once_flag started;
    std::call_once(started, [] {
        {
            std::lock_guard<std::mutex> lk(g_cs_opt.mu);
            g_cs_opt.running.resize(static_cast<std::size_t>(g_cs_opt.threads));
        }
        for (int i = 0; i < g_cs_opt.threads; ++i) std::thread(cs_optimize_thread, i).detach();
    });
    {
        std::lock_guard<std::mutex> lk(g_cs_opt.mu);
        if (!g_cs_opt.queued.insert(key).second) return;  // queued, building or built
        CsOptimizer::Job job;
        job.key = key;
        job.name = name;
        job.spirv = spirv;
        job.flip = hle_video_flip_count();
        job.standin_us = standin_us;
        job.layout = layout;
        g_cs_opt.jobs.push_back(std::move(job));
    }
    g_cs_opt.cv.notify_one();
}

}  // namespace

VkPipeline first_compute_pipeline(VkShaderModule module, std::uint64_t opt_key, const std::string& name,
                                  const std::vector<std::uint32_t>& spirv, bool& standin, const std::string& watch, VkPipelineLayout layout) {
    standin = false;
    // BBHOST_PIPELINE_STATS=<name>: its optimized build once more with the
    // driver's statistics and disassembly captured, written and let go.
    if (g.pipeline_stats && compute_pipeline_stats_wanted(name)) {
        static std::mutex mu;
        static std::set<std::string> done;
        const std::string file = name + (layout ? "-cs-buffers" : "-cs");  // the variant with its buffers pushed (ComputeLean)
        bool first;
        {
            std::lock_guard<std::mutex> lk(mu);
            first = done.insert(file).second;
        }
        if (first) {
            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = module;
            cpci.stage.pName = "main";
            cpci.layout = layout ? layout : g.pipe_layout;
            cpci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
            VkPipeline p = VK_NULL_HANDLE;
            if (vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p) == VK_SUCCESS) {
                write_compute_pipeline_stats(p, file);
                vkDestroyPipeline(g.device, p, nullptr);
            }
        }
    }
    switch (cs_optimize_mode()) {
    case CsOptimize::kNever:
        return create_compute(module, CsBuild::kUnoptimized, watch, &spirv, VK_NULL_HANDLE, layout);
    case CsOptimize::kAlways:
        return create_compute(module, CsBuild::kOptimized, watch, &spirv, VK_NULL_HANDLE, layout);
    case CsOptimize::kBackground:
        break;
    }
    if (g_cs_opt.keep_unoptimized.count(name)) {
        g_cs_opt.kept.fetch_add(1, std::memory_order_relaxed);
        return create_compute(module, CsBuild::kUnoptimized, watch, &spirv, VK_NULL_HANDLE, layout);
    }
    {
        // Built this run already, for an earlier pipeline of the same key.
        std::lock_guard<std::mutex> lk(g_cs_opt.mu);
        if (const auto it = g_cs_opt.ready.find(opt_key); it != g_cs_opt.ready.end()) return it->second;
    }
    if (const VkPipeline p = create_compute(module, CsBuild::kOptimizedCached, watch, &spirv, VK_NULL_HANDLE, layout)) {
        g_cs_opt.from_cache.fetch_add(1, std::memory_order_relaxed);
        return p;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const VkPipeline p = create_compute(module, CsBuild::kUnoptimized, watch, &spirv, VK_NULL_HANDLE, layout);
    if (!p) return p;
    standin = true;
    g_cs_opt.standins.fetch_add(1, std::memory_order_relaxed);
    if (!t_compute_worker) g_cs_opt.standins_cp.fetch_add(1, std::memory_order_relaxed);
    queue_compute_optimize(opt_key, name, spirv, cs_us_since(t0), layout);
    return p;
}

bool adopt_optimized_compute(std::uint64_t opt_key, VkPipeline& pipeline, bool& standin, std::uint64_t& seen) {
    const std::uint64_t gen = g_cs_opt.gen.load(std::memory_order_acquire);
    if (gen == seen) return false;
    seen = gen;
    std::lock_guard<std::mutex> lk(g_cs_opt.mu);
    const auto it = g_cs_opt.ready.find(opt_key);
    if (it == g_cs_opt.ready.end()) return false;
    pipeline = it->second;  // dispatches already recorded keep the stand-in: pipelines are never destroyed
    standin = false;
    g_cs_opt.adopted.fetch_add(1, std::memory_order_relaxed);
    g_cs_opt.window_adopted.fetch_add(1, std::memory_order_relaxed);
    return true;
}

std::uint64_t compute_optimized_built() { return g_cs_opt.built.load(std::memory_order_relaxed); }

void compute_optimize_report() {
    if (cs_optimize_mode() != CsOptimize::kBackground) return;
    std::string slowest;
    std::size_t waiting = 0;
    std::uint64_t slowest_us = 0;
    std::vector<std::pair<std::string, std::uint64_t>> running;  // name, seconds in the compiler
    {
        std::lock_guard<std::mutex> lk(g_cs_opt.mu);
        waiting = g_cs_opt.jobs.size();
        for (const CsOptimizer::Running& r : g_cs_opt.running) {
            if (!r.name.empty()) running.emplace_back(r.name, cs_us_since(r.since) / 1000000);
        }
        slowest = g_cs_opt.slowest;
        slowest_us = g_cs_opt.slowest_us;
    }
    std::string building = running.empty() ? "none building" : "building";
    for (const auto& [name, s] : running) building += " " + name + " (" + std::to_string(s) + " s)";
    host_log("gpu: compute pipelines, the optimizer in the background: %llu taken optimized from the pipeline cache, %llu built "
             "unoptimized to stand in (%llu of them on the command processor), %llu optimized on bb-cs-optimize (%.1f s; the slowest %s, "
             "%llu ms), %llu failed, %llu swapped in; at exit %zu waiting, %s; %llu kept unoptimized (compute-unoptimized.txt)",
             static_cast<unsigned long long>(g_cs_opt.from_cache.load()), static_cast<unsigned long long>(g_cs_opt.standins.load()),
             static_cast<unsigned long long>(g_cs_opt.standins_cp.load()), static_cast<unsigned long long>(g_cs_opt.built.load()),
             g_cs_opt.build_us.load() / 1e6, slowest.empty() ? "none" : slowest.c_str(), static_cast<unsigned long long>(slowest_us / 1000),
             static_cast<unsigned long long>(g_cs_opt.failed.load()), static_cast<unsigned long long>(g_cs_opt.adopted.load()), waiting,
             building.c_str(), static_cast<unsigned long long>(g_cs_opt.kept.load()));
    // A build that has had its thread this long may never end: later starts
    // keep that shader unoptimized.
    for (const auto& [name, s] : running) {
        if (s < kCsGiveUpS) continue;
        const std::string path = compute_unoptimized_path();
        if (path.empty()) break;
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
        if (std::ofstream out(path, std::ios::app); out) {
            out << name << "\n";
            host_log("gpu: compute shader %s: its optimized build was still running after %llu s; later starts keep it unoptimized (%s)",
                     name.c_str(), static_cast<unsigned long long>(s), path.c_str());
        }
    }
}

// Since the last call, for the 300-flip report; empty when nothing happened.
std::string compute_optimize_window() {
    const std::uint64_t built = g_cs_opt.window_built.exchange(0), us = g_cs_opt.window_us.exchange(0);
    const std::uint64_t adopted = g_cs_opt.window_adopted.exchange(0);
    if (!built && !adopted) return {};
    char buf[128];
    std::snprintf(buf, sizeof(buf), "compute pipelines optimized %llu (%llu ms), swapped in %llu", static_cast<unsigned long long>(built),
                  static_cast<unsigned long long>(us / 1000), static_cast<unsigned long long>(adopted));
    return buf;
}

namespace {

// Compute pipelines dispatches built on the command processor (pipeline_for).
std::atomic<std::uint64_t> g_cp_compute_us{0}, g_cp_compute_n{0};

// A dispatch's first build of a pipeline (under g.mu): take what a worker
// compiled for the same key, waiting while it builds - but not for ever: a
// driver can take minutes over one, and the command processor waiting on it
// held the whole game (the AMD run behind BBHOST_CS_OPTIMIZE). After a couple
// of seconds the dispatch goes without, and so do the ones after it, without
// waiting, until the build is done. kBuildHere: build it here; the key is
// claimed, so a worker that has not reached it skips it.
enum class Precompiled { kTaken, kBuildHere, kPending };
constexpr auto kPrecompileWait = std::chrono::seconds(2);
Precompiled take_precompiled_compute(std::uint64_t key, const GpuDispatch& d, const std::string& name,
                                     const std::vector<std::pair<std::uint32_t, bool>>& dims, const std::vector<bool>& modes,
                                     ComputePipeline& pl) {
    if (!compute_precompile_enabled()) return Precompiled::kBuildHere;
    std::unique_lock<std::mutex> lk(g_cs_pre.mu);
    if (!g_cs_pre.by_key.count(key)) {
        PrecompiledCompute::Entry claimed;
        claimed.done = claimed.failed = true;
        g_cs_pre.by_key.emplace(key, std::move(claimed));
        const auto dims_str = [](const std::vector<std::pair<std::uint32_t, bool>>& v) {
            std::string s;
            for (const auto& p : v) s += " " + std::to_string(p.first) + (p.second ? "a" : "");
            return s;
        };
        std::string why = "never created through the hook";
        std::uint64_t* count = &g_cs_pre.miss_never;
        if (const auto r = g_cs_pre.records.find(name); r != g_cs_pre.records.end()) {
            const PrecompiledCompute::Record& c = r->second;
            if (c.state == PrecompiledCompute::kQueued) {
                why = "still queued";
                count = &g_cs_pre.miss_queued;
            } else if (c.state == PrecompiledCompute::kFailed) {
                why = "failed at creation: " + c.why;
                count = &g_cs_pre.miss_failed;
            } else if (c.rsrc1 != d.rsrc1 || c.rsrc2 != d.rsrc2) {
                why = "resource registers";
                count = &g_cs_pre.miss_rsrc;
            } else if (std::memcmp(c.threads, d.threads, sizeof(c.threads)) != 0) {
                why = "thread counts";
                count = &g_cs_pre.miss_threads;
            } else if (c.dims != dims) {
                why = "image dims: predicted" + dims_str(c.dims) + ", bound" + dims_str(dims);
                count = &g_cs_pre.miss_dims;
            } else if (c.modes != modes) {
                why = "sampler modes";
                count = &g_cs_pre.miss_modes;
            } else {
                why = "other";
                count = &g_cs_pre.miss_other;
            }
        }
        *count += 1;
        if (g_cs_pre.notes.size() < 16) g_cs_pre.notes.push_back(name + " (" + why + ")");
        return Precompiled::kBuildHere;
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool waited = false;
    if (!g_cs_pre.by_key[key].done) {
        if (g_cs_pre.by_key[key].gave_up) return Precompiled::kPending;
        waited = true;
        g_cs_pre.cv.wait_for(lk, kPrecompileWait, [&] { return g_cs_pre.by_key[key].done; });
        if (!g_cs_pre.by_key[key].done) {
            g_cs_pre.by_key[key].gave_up = true;
            g_cs_pre.gave_up += 1;
            host_log("gpu: compute shader %s: its pipeline is still in the driver's compiler after %lld s; its dispatches go "
                     "without it until it is built",
                     name.c_str(), static_cast<long long>(kPrecompileWait.count()));
            return Precompiled::kPending;
        }
    }
    const std::uint64_t wait_us =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    if (waited) {
        g_cs_pre.waited += 1;
        g_cs_pre.wait_us += wait_us;
    }
    PrecompiledCompute::Entry& e = g_cs_pre.by_key[key];
    if (e.failed) return Precompiled::kBuildHere;  // the dispatch's own build reports the failure
    pl.meta = std::move(e.meta);
    pl.module = e.module;
    pl.pipeline = e.pipeline;
    pl.standin = e.standin;
    pl.lean = std::move(e.lean);
    pl.lifted = e.lifted;
    pl.lift_why = std::move(e.lift_why);
    e.failed = true;  // taken
    g_cs_pre.taken += 1;
    if (e.us >= 100000 || waited) {
        const auto r = g_cs_pre.records.find(name);
        host_log("gpu: compute shader %s: pipeline compiled at creation (flip %llu; now flip %llu) in %llu ms, the dispatch waited %llu ms",
                 name.c_str(), static_cast<unsigned long long>(r != g_cs_pre.records.end() ? r->second.created_flip : 0),
                 static_cast<unsigned long long>(hle_video_flip_count()), static_cast<unsigned long long>(e.us / 1000),
                 static_cast<unsigned long long>(waited ? wait_us / 1000 : 0));
    }
    return Precompiled::kTaken;
}

// A pipeline not waited for (kPending): has its worker finished it?
bool precompiled_compute_done(std::uint64_t key) {
    std::lock_guard<std::mutex> lk(g_cs_pre.mu);
    const auto it = g_cs_pre.by_key.find(key);
    return it == g_cs_pre.by_key.end() || it->second.done;
}

ComputePipeline& pipeline_for(const GpuDispatch& d, std::uint64_t key, const std::vector<std::uint32_t>& words,
                              const std::string& name, const std::vector<std::pair<std::uint32_t, bool>>& dims,
                              const std::vector<bool>& sampler_modes) {
    auto it = g.pipelines.find(key);
    if (it != g.pipelines.end()) {
        if (!it->second.pending) {
            ComputePipeline& have = it->second;
            if (have.standin) adopt_optimized_compute(key, have.pipeline, have.standin, have.opt_seen);
            if (have.lean && have.lean->standin) {
                adopt_optimized_compute(key ^ kCsBuffersTag, have.lean->pipeline, have.lean->standin, have.lean->opt_seen);
            }
            return have;
        }
        // A worker's build not waited for: skipped until it is done, then
        // made again from the start, which takes it.
        if (!precompiled_compute_done(key)) {
            g_cs_pre.skipped_dispatches.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }
        g.pipelines.erase(it);
    }
    ComputePipeline& pl = g.pipelines[key];
    pl.name = name;
    const gcn::Program prog = gcn::decode(words.data(), words.size());
    if (!prog.errors.empty()) {
        host_log("gpu: shader %s: decode error at %06x: %s", name.c_str(), prog.errors[0].offset, prog.errors[0].what.c_str());
        pl.failed = true;
        return pl;
    }
    const gcn::TranslateOptions o = compute_options(d.rsrc1, d.rsrc2, d.threads, dims, sampler_modes);
    detect_fill(prog, pl);
    if (pl.fill) host_log("gpu: shader %s fills the V# in user[%d:%d] (render targets are cleared instead)", name.c_str(), pl.fill_dst_sgpr, pl.fill_dst_sgpr + 3);
    detect_copy(prog, pl);
    detect_image_copy(prog, pl);
    if (pl.image_blit) host_log("gpu: shader %s copies a rectangle from one image to another", name.c_str());
    if (pl.buffer_to_image) {
        host_log("gpu: shader %s copies the V# in user[%d:%d] into an image", name.c_str(), pl.img_src_sgpr, pl.img_src_sgpr + 3);
    }
    if (pl.copy) host_log("gpu: shader %s copies the V# in user[%d:%d] to user[%d:%d] (render targets become snapshots)", name.c_str(), pl.copy_src_sgpr, pl.copy_src_sgpr + 3, pl.copy_dst_sgpr, pl.copy_dst_sgpr + 3);
    switch (take_precompiled_compute(key, d, name, dims, sampler_modes, pl)) {
    case Precompiled::kTaken:
        g.translated.fetch_add(1);
        recipe_note_compute(name, words, key, d.rsrc1, d.rsrc2, d.threads, dims, sampler_modes, false);  // a compute record's: its use counted
        return pl;
    case Precompiled::kPending:
        pl.pending = pl.failed = true;
        g_cs_pre.skipped_dispatches.fetch_add(1, std::memory_order_relaxed);
        return pl;
    case Precompiled::kBuildHere:
        break;
    }
    // The dispatch builds it itself: what that costs the command processor
    // (host_gpu_pipeline_work), however it ends.
    struct BuildTime {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~BuildTime() {
            if (t_compute_worker) return;
            g_cp_compute_us.fetch_add(cs_us_since(t0), std::memory_order_relaxed);
            g_cp_compute_n.fetch_add(1, std::memory_order_relaxed);
        }
    } build_time;
    pl.meta = translate_cached(prog, o);
    if (!pl.meta.ok()) {
        host_log("gpu: shader %s: %s", name.c_str(), pl.meta.errors[0].c_str());
        pl.failed = true;
        return pl;
    }
    apply_compute_lift(prog, o, pl.meta, name, pl.lifted, pl.lift_why);
    if (const char* e = std::getenv("BBHOST_DUMP_SPIRV"); e && e[0] == '1') {
        host_mkdir("build/spv");
        const std::string path = "build/spv/" + name + (pl.lifted ? "-cs-lifted.spv" : "-cs.spv");
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(pl.meta.spirv.data(), 4, pl.meta.spirv.size(), f);
            std::fclose(f);
        }
        host_log("gpu: creating compute pipeline %s", name.c_str());
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool from_lift = false;  // BBHOST_HLSL: DXC's module was refused, this is the lift's own
    for (;;) {
        const char* failed = nullptr;
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = pl.meta.spirv.size() * 4;
        smci.pCode = pl.meta.spirv.data();
        if (vkCreateShaderModule(g.device, &smci, nullptr, &pl.module) != VK_SUCCESS) {
            pl.module = VK_NULL_HANDLE;
            failed = "module creation";
        } else if (hlsl_test_refused_compute(pl.meta.spirv)) {
            failed = "pipeline creation";
        } else {
            pl.pipeline = first_compute_pipeline(pl.module, key, name, pl.meta.spirv, pl.standin, name + " (at its dispatch)");
            if (!pl.pipeline) failed = "pipeline creation";
        }
        if (from_lift) {
            hlsl_note_refused("a compute pipeline", 1, !failed);
            from_lift = false;
        }
        if (!failed) break;
        if (!pl.lifted) {
            host_log("gpu: shader %s: %s failed", name.c_str(), failed);
            pl.failed = true;
            return pl;
        }
        // BBHOST_HLSL: the driver refused DXC's module - the lift's own next.
        if (std::vector<std::uint32_t> lift; hlsl_lift_of(pl.meta.spirv, lift)) {
            if (pl.module) vkDestroyShaderModule(g.device, pl.module, nullptr);
            pl.module = VK_NULL_HANDLE;
            pl.meta.spirv = std::move(lift);
            from_lift = true;
            continue;
        }
        // The driver refused what the lift built: the translation instead.
        host_log("gpu: shader %s: the lift's %s failed; it runs translated", name.c_str(), failed);
        if (pl.module) vkDestroyShaderModule(g.device, pl.module, nullptr);
        pl.module = VK_NULL_HANDLE;
        pl.meta = translate_cached(prog, o);
        pl.lifted = false;
        pl.lift_why = std::string("the driver refused the lift's ") + failed;
        note_compute_lift(name, false, pl.lift_why);
    }
    // A lifted program runs its lift (render/next); ComputeLean, the translation
    // with its buffers pushed, is the form a translated one runs in, and the
    // lift of that translation a lifted one's (BBHOST_LIFT_CS_BUFFERS) - its
    // walking lift when a buffer does not bind.
    pl.lean = build_compute_lean(prog, d.rsrc1, d.rsrc2, d.threads, dims, sampler_modes, pl.meta, key, name, pl.lifted);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    g.translated.fetch_add(1);
    static const char* trace_name = std::getenv("BBHOST_TRACE_DISPATCH");
    if (trace_name && name == trace_name) {
        for (const gcn::Inst& in : prog.insts) host_log("  cs: %s", gcn::format(in).c_str());
        for (const gcn::ImageBinding& b : pl.meta.images) host_log("  cs image: %s%s", b.path.str().c_str(), b.storage ? " (storage)" : "");
        host_log("  cs user: %08x %08x %08x %08x %08x %08x %08x %08x", d.user_data[0], d.user_data[1], d.user_data[2], d.user_data[3],
                 d.user_data[4], d.user_data[5], d.user_data[6], d.user_data[7]);
        host_log("  cs dims %ux%ux%u threads %ux%ux%u", d.dim[0], d.dim[1], d.dim[2], d.threads[0], d.threads[1], d.threads[2]);
        auto dump = [&](const char* what, std::uint64_t va, int n) {
            if (!hle_kernel_va_mapped(va, static_cast<std::size_t>(n) * 4)) { host_log("  %s at 0x%llx: unmapped", what, static_cast<unsigned long long>(va)); return; }
            const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va));
            const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(va));
            host_log("  %s at 0x%llx: %08x %08x %08x %08x %08x %08x %08x %08x (floats %g %g %g %g)", what, static_cast<unsigned long long>(va),
                     u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], f[0], f[1], f[2], f[3]);
        };
        const std::uint64_t cb_ptr = static_cast<std::uint64_t>(d.user_data[2]) | (static_cast<std::uint64_t>(d.user_data[3]) << 32);
        dump("V# via user[2:3]", cb_ptr, 8);
        if (hle_kernel_va_mapped(cb_ptr, 16)) {
            const auto* v = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(cb_ptr));
            dump("constant buffer", static_cast<std::uint64_t>(v[0]) | (static_cast<std::uint64_t>(v[1] & 0xff) << 32), 8);
        }
        dump("source V# base (user[4:7])", static_cast<std::uint64_t>(d.user_data[4]) | (static_cast<std::uint64_t>(d.user_data[5] & 0xff) << 32), 8);
    }
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 40) {
        host_log("gpu: compute shader %s: %zu instructions, %zu images, %zu samplers, threads %ux%ux%u, pipeline in %lld ms%s",
                 name.c_str(), prog.insts.size(), pl.meta.images.size(), pl.meta.samplers.size(), d.threads[0],
                 d.threads[1], d.threads[2], static_cast<long long>(ms), pl.standin ? " (unoptimized, standing in)" : "");
    }
    // Built here, on the command processor: the next start (or a list's
    // first start) builds it on a precompile worker instead.
    if (pl.pipeline) recipe_note_compute(name, words, key, d.rsrc1, d.rsrc2, d.threads, dims, sampler_modes, true);
    return pl;
}

}  // namespace

void note_cp_compute_build(std::uint64_t us) {
    g_cp_compute_us.fetch_add(us, std::memory_order_relaxed);
    g_cp_compute_n.fetch_add(1, std::memory_order_relaxed);
}

void compute_pipeline_work(std::uint64_t out[2]) {
    out[0] = g_cp_compute_us.load(std::memory_order_relaxed);
    out[1] = g_cp_compute_n.load(std::memory_order_relaxed);
}

bool compute_precompile_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_PRECOMPILE");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool note_compute_created(const std::string& name, std::uint64_t flip) {
    std::lock_guard<std::mutex> lk(g_cs_pre.mu);
    const auto [it, fresh] = g_cs_pre.records.try_emplace(name);
    if (fresh) it->second.created_flip = flip;
    return fresh;
}

namespace {
// The pipeline a dispatch of this program would build under its key, on a
// precompile worker: from a container (precompile_compute) or a compute
// record (precompile_compute_recorded). False when it failed or another build
// had the key.
bool precompile_compute_key(const std::string& name, const std::vector<std::uint32_t>& words, const gcn::Program& prog, std::uint32_t rsrc1,
                            std::uint32_t rsrc2, const std::uint32_t threads[3], const std::vector<std::pair<std::uint32_t, bool>>& dims,
                            const std::vector<bool>& modes, std::uint64_t created_flip, std::chrono::steady_clock::time_point t0) {
    const std::uint64_t key = compute_key(fnv1a(words.data(), words.size() * 4), rsrc1, rsrc2, threads, dims, modes);
    {
        std::lock_guard<std::mutex> lk(g_cs_pre.mu);
        PrecompiledCompute::Record& r = g_cs_pre.records[name];
        r.created_flip = created_flip;
        r.rsrc1 = rsrc1;
        r.rsrc2 = rsrc2;
        std::memcpy(r.threads, threads, sizeof(r.threads));
        r.dims = dims;
        r.modes = modes;
        if (!g_cs_pre.by_key.emplace(key, PrecompiledCompute::Entry{}).second) {  // a dispatch built it first
            r.state = PrecompiledCompute::kDone;
            g_cs_pre.skipped += 1;
            return false;
        }
        r.state = PrecompiledCompute::kCompiling;
    }
    // From here every way out marks the entry done: a dispatch may be waiting on it.
    PrecompiledCompute::Entry e;
    std::string why;
    const gcn::TranslateOptions co = compute_options(rsrc1, rsrc2, threads, dims, modes);
    e.meta = translate_cached(prog, co);
    if (!e.meta.ok()) why = "translation: " + e.meta.errors[0];
    if (why.empty()) apply_compute_lift(prog, co, e.meta, name, e.lifted, e.lift_why);
    if (why.empty()) {
        if (const char* env = std::getenv("BBHOST_DUMP_SPIRV"); env && env[0] == '1') {
            host_mkdir("build/spv");
            const std::string path = "build/spv/" + name + (e.lifted ? "-cs-lifted.spv" : "-cs.spv");
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(e.meta.spirv.data(), 4, e.meta.spirv.size(), f);
                std::fclose(f);
            }
            // The program too, for `gcndis --raw` and `gcn2spv --raw ... cs <rsrc1> <rsrc2>`.
            if (FILE* f = std::fopen(("build/spv/" + name + "-cs.bin").c_str(), "wb")) {
                std::fwrite(words.data(), 4, words.size(), f);
                std::fclose(f);
            }
            if (FILE* f = std::fopen(("build/spv/" + name + "-cs.txt").c_str(), "w")) {
                std::fprintf(f, "rsrc1 0x%08x rsrc2 0x%08x threads %u %u %u\n", rsrc1, rsrc2, threads[0], threads[1], threads[2]);
                std::fclose(f);
            }
        }
        // BBHOST_CS_COMPILE_DELAY_MS=<ms> (checks): a compiler that takes that
        // long over every compute pipeline built at creation - the dispatch's
        // bounded wait and the watchdog's line, tried on a fast driver.
        static const int delay_ms = [] {
            const char* d = std::getenv("BBHOST_CS_COMPILE_DELAY_MS");
            return d ? std::atoi(d) : 0;
        }();
        if (delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        t_compute_worker = true;
        bool from_lift = false;  // BBHOST_HLSL: DXC's module was refused, this is the lift's own
        for (;;) {
            VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smci.codeSize = e.meta.spirv.size() * 4;
            smci.pCode = e.meta.spirv.data();
            if (vkCreateShaderModule(g.device, &smci, nullptr, &e.module) != VK_SUCCESS) {
                e.module = VK_NULL_HANDLE;
                why = "shader module";
            } else if (hlsl_test_refused_compute(e.meta.spirv)) {
                why = "pipeline";
            } else {
                e.pipeline = first_compute_pipeline(e.module, key, name, e.meta.spirv, e.standin, name);
                if (!e.pipeline) why = "pipeline";
            }
            if (from_lift) {
                hlsl_note_refused("a compute pipeline", 1, why.empty());
                from_lift = false;
            }
            if (why.empty() || !e.lifted) break;
            // BBHOST_HLSL: the driver refused DXC's module - the lift's own next.
            if (std::vector<std::uint32_t> lift; hlsl_lift_of(e.meta.spirv, lift)) {
                if (e.module) vkDestroyShaderModule(g.device, e.module, nullptr);
                e.module = VK_NULL_HANDLE;
                e.meta.spirv = std::move(lift);
                from_lift = true;
                why.clear();
                continue;
            }
            // The driver refused what the lift built: the translation instead.
            host_log("gpu: shader %s: the lift's %s failed; it runs translated", name.c_str(), why.c_str());
            if (e.module) vkDestroyShaderModule(g.device, e.module, nullptr);
            e.module = VK_NULL_HANDLE;
            e.meta = translate_cached(prog, co);
            e.lifted = false;
            e.lift_why = "the driver refused the lift's " + why;
            note_compute_lift(name, false, e.lift_why);
            why.clear();
        }
    }
    if (why.empty()) e.lean = build_compute_lean(prog, rsrc1, rsrc2, threads, dims, modes, e.meta, key, name, e.lifted);  // as in pipeline_for
    e.us = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    e.done = true;
    e.failed = !why.empty();
    const std::uint64_t us = e.us;
    const bool standin = e.standin;
    {
        std::lock_guard<std::mutex> lk(g_cs_pre.mu);
        g_cs_pre.by_key[key] = std::move(e);
        PrecompiledCompute::Record& r = g_cs_pre.records[name];
        r.state = why.empty() ? PrecompiledCompute::kDone : PrecompiledCompute::kFailed;
        r.why = why;
        (why.empty() ? g_cs_pre.compiled : g_cs_pre.failed) += 1;
        g_cs_pre.compile_us += us;
    }
    g_cs_pre.cv.notify_all();
    if (us >= 1000000) {
        host_log("gpu: compute shader %s compiled at creation (flip %llu) in %llu ms%s%s%s", name.c_str(), static_cast<unsigned long long>(created_flip),
                 static_cast<unsigned long long>(us / 1000), standin ? " (unoptimized, standing in)" : "", why.empty() ? "" : ", failed: ",
                 why.c_str());
    }
    return why.empty();
}
}  // namespace

// On a precompile worker: the pipeline a dispatch of this shader would build,
// from what its container gives.
void precompile_compute(const std::string& name, const std::vector<std::uint8_t>& c, std::uint64_t created_flip) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto fail = [&](const std::string& why) {  // before the key is known: no dispatch can be waiting
        std::lock_guard<std::mutex> lk(g_cs_pre.mu);
        PrecompiledCompute::Record& r = g_cs_pre.records[name];
        r.state = PrecompiledCompute::kFailed;
        r.why = why;
        g_cs_pre.failed += 1;
    };
    std::size_t shdr = 0;
    while (shdr + 0x60 <= c.size() && shdr < 0x100 && std::memcmp(c.data() + shdr, "Shdr", 4) != 0) ++shdr;
    gcn::ShaderCode code;
    if (shdr + 0x60 > c.size() || shdr >= 0x100 || !gcn::shader_code(c, code) || code.type != 4) return fail("not a compute container");
    std::uint32_t regs[16];
    std::memcpy(regs, c.data() + shdr + 16, sizeof(regs));
    const std::uint32_t rsrc1 = regs[4], rsrc2 = regs[5];  // CsStageRegs: pgm_lo, pgm_hi, rsrc1, rsrc2, num_thread_x/y/z
    const std::uint32_t threads[3] = {regs[6], regs[7], regs[8]};
    const gcn::Program prog = gcn::decode(code.words.data(), code.words.size());
    if (!prog.errors.empty()) return fail("decode");
    gcn::TranslateOptions paths_options;  // as paths_for
    paths_options.stage = gcn::Stage::Compute;
    paths_options.rsrc1 = rsrc1;
    paths_options.rsrc2 = rsrc2;
    const gcn::TranslateResult paths = translate_cached(prog, paths_options, TranslationUse::kPaths);
    offer_paths(code.words, paths);  // the first dispatch's paths_for takes them instead of translating on the command processor
    std::vector<std::pair<std::uint32_t, bool>> dims;
    for (const gcn::PredictedImage& p : gcn::predict_image_dims(prog, paths)) dims.emplace_back(p.dim, p.arrayed);
    const std::vector<bool> modes(paths.samplers.size(), false);
    precompile_compute_key(name, code.words, prog, rsrc1, rsrc2, threads, dims, modes, created_flip, t0);
}

// On a precompile worker: a compute record's pipeline (the pipeline manifest's,
// a list's), under the key its dispatch made - the record's images and
// samplers, not a guess from the program.
bool precompile_compute_recorded(const std::string& name, const std::vector<std::uint32_t>& words, std::uint32_t rsrc1, std::uint32_t rsrc2,
                                 const std::uint32_t threads[3], const std::vector<std::pair<std::uint32_t, bool>>& dims,
                                 const std::vector<bool>& modes) {
    const auto t0 = std::chrono::steady_clock::now();
    const gcn::Program prog = gcn::decode(words.data(), words.size());
    if (!prog.errors.empty()) return false;
    gcn::TranslateOptions paths_options;  // as paths_for
    paths_options.stage = gcn::Stage::Compute;
    paths_options.rsrc1 = rsrc1;
    paths_options.rsrc2 = rsrc2;
    offer_paths(words, translate_cached(prog, paths_options, TranslationUse::kPaths));
    return precompile_compute_key(name, words, prog, rsrc1, rsrc2, threads, dims, modes, 0, t0);
}

void compute_compiling_report() {
    std::lock_guard<std::mutex> lk(g_cs_pre.mu);
    const auto now = std::chrono::steady_clock::now();
    std::string line;
    for (auto& [name, c] : g_cs_pre.compiling) {
        const auto s = std::chrono::duration_cast<std::chrono::seconds>(now - c.since).count();
        line += " " + name + " (" + std::to_string(s) + " s, " + std::to_string(c.spirv ? c.spirv->size() : 0) + " words)";
        // Its SPIR-V, once, for whoever reads the log: a module the driver
        // chokes on is the thing to look at.
        if (s >= 5 && !c.written && c.spirv && !c.spirv->empty()) {
            c.written = true;
            std::string file = name;
            for (char& ch : file) {
                if (ch == ' ' || ch == '(' || ch == ')') ch = '_';
            }
            const std::string path = config().tmp + "/slow-compute-" + file + ".spv";
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(c.spirv->data(), 4, c.spirv->size(), f);
                std::fclose(f);
                line += " -> " + path;
            }
        }
    }
    if (!line.empty()) host_log("gpu: compute pipelines in the driver's compiler now:%s", line.c_str());
    if (g_cs_pre.gave_up) {
        host_log("gpu: %llu compute pipelines not waited for any longer, %llu dispatches gone without them",
                 static_cast<unsigned long long>(g_cs_pre.gave_up),
                 static_cast<unsigned long long>(g_cs_pre.skipped_dispatches.load()));
    }
}

void compute_precompile_report() {
    if (cs_buffers_on()) {
        host_log("gpu: compute shaders with their buffers pushed (BBHOST_CS_BUFFERS): %llu built (%llu of them lifts, BBHOST_LIFT_CS_BUFFERS; "
                 "%llu lifts refused), %llu with none to push, %llu not built; dispatches %llu with them, %llu walking the page table because "
                 "one did not bind",
                 static_cast<unsigned long long>(g_cs_lean_built.load()), static_cast<unsigned long long>(g_cs_lean_lifted.load()),
                 static_cast<unsigned long long>(g_cs_lean_lift_refused.load()), static_cast<unsigned long long>(g_cs_lean_none.load()),
                 static_cast<unsigned long long>(g_cs_lean_failed.load()), static_cast<unsigned long long>(g_cs_lean_dispatches.load()),
                 static_cast<unsigned long long>(g_cs_lean_fallbacks.load()));
    }
    if (!compute_precompile_enabled()) return;
    compute_compiling_report();  // anything still in the driver's compiler at exit
    std::lock_guard<std::mutex> lk(g_cs_pre.mu);
    std::map<std::string, std::uint64_t> reasons;
    for (const auto& [name, r] : g_cs_pre.records) {
        if (r.state == PrecompiledCompute::kFailed) reasons[r.why] += 1;
    }
    std::string failed, notes;
    for (const auto& [why, n] : reasons) failed += " " + why + " x" + std::to_string(n) + ";";
    for (const std::string& s : g_cs_pre.notes) notes += " " + s + ";";
    host_log("gpu: compute shaders compiled at creation: %zu created, %llu compiled (%.1f s), %llu failed, %llu skipped (a dispatch came first); "
             "dispatch pipelines that took one %llu, %llu of them after waiting for the build (%.1f s)",
             g_cs_pre.records.size(), static_cast<unsigned long long>(g_cs_pre.compiled), g_cs_pre.compile_us / 1e6,
             static_cast<unsigned long long>(g_cs_pre.failed), static_cast<unsigned long long>(g_cs_pre.skipped),
             static_cast<unsigned long long>(g_cs_pre.taken), static_cast<unsigned long long>(g_cs_pre.waited), g_cs_pre.wait_us / 1e6);
    host_log("gpu: compute pipelines built at the dispatch instead: never created through the hook %llu, still queued %llu, failed at creation %llu, "
             "resource registers %llu, thread counts %llu, image dims %llu, sampler modes %llu, other %llu;%s",
             static_cast<unsigned long long>(g_cs_pre.miss_never), static_cast<unsigned long long>(g_cs_pre.miss_queued),
             static_cast<unsigned long long>(g_cs_pre.miss_failed), static_cast<unsigned long long>(g_cs_pre.miss_rsrc),
             static_cast<unsigned long long>(g_cs_pre.miss_threads), static_cast<unsigned long long>(g_cs_pre.miss_dims),
             static_cast<unsigned long long>(g_cs_pre.miss_modes), static_cast<unsigned long long>(g_cs_pre.miss_other), notes.c_str());
    if (!failed.empty()) host_log("gpu: compute shaders not compiled at creation, by reason:%s", failed.c_str());
}

void collect_profile_locked(Gpu::Slot& sl);

// 32 bits a marker: the kind in the top two, the index below.
std::uint32_t gpu_marker_value(unsigned kind, std::uint64_t index) {
    return (static_cast<std::uint32_t>(kind & 3) << 30) | static_cast<std::uint32_t>(index & 0x3fffffff);
}

void gpu_write_markers(VkCommandBuffer cmd, std::uint32_t value) {
    g.cmd_buffer_marker(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g.marker_buf.buffer, 0, value);
    g.cmd_buffer_marker(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.marker_buf.buffer, 4, value);
}

void gpu_checkpoint(unsigned kind, std::uint64_t index) {
    if (g.cmd_buffer_marker) {
        // An op of the stream: recording it in place would drain the
        // recorder at every dispatch of a start after a lost device.
        rec().marker(gpu_marker_value(kind, index));
        return;
    }
    if (!g.cmd_checkpoint) return;
    g.cmd_checkpoint(g_cmd(), reinterpret_cast<void*>(static_cast<std::uintptr_t>((static_cast<std::uint64_t>(kind) << 56) | index)));
}

// The same into a command buffer of its own (the buffer shadow's uploads,
// recorded on the command processor beside the stream).
void gpu_checkpoint_cmd(VkCommandBuffer cmd, unsigned kind, std::uint64_t index) {
    if (g.cmd_buffer_marker) {
        gpu_write_markers(cmd, gpu_marker_value(kind, index));
        return;
    }
    if (!g.cmd_checkpoint) return;
    g.cmd_checkpoint(cmd, reinterpret_cast<void*>(static_cast<std::uintptr_t>((static_cast<std::uint64_t>(kind) << 56) | index)));
}

// A kGpuMarkSubmission mark's words: the submission and where in it.
std::string submission_mark(std::uint64_t which) {
    static const char* const kWhere[4] = {"its vertex and index uploads (after its wait for their sparse binds)",
                                          "its recording's start", "its recording's end", "?"};
    return "submission " + std::to_string(which >> 2) + ", " + kWhere[which & 3];
}

// The last dispatches by index, for a device-loss report to name.
const char* g_dispatch_names[256] = {};
void note_dispatch_name(std::uint64_t index, const char* name) { g_dispatch_names[index % 256] = name; }

// Where the GPU actually got to, when VK_NV_device_diagnostic_checkpoints is
// there: the recorded draw list is recording order, this is execution order.
void report_checkpoints() {
    if (!g.has_checkpoints) return;
    if (g.cmd_buffer_marker && g.marker_buf.map) {
        std::uint32_t started = 0, finished = 0;
        std::memcpy(&started, g.marker_buf.map, 4);
        std::memcpy(&finished, static_cast<const std::uint8_t*>(g.marker_buf.map) + 4, 4);
        const auto say = [](const char* what, std::uint32_t v) {
            const unsigned kind = v >> 30;
            const std::uint64_t which = v & 0x3fffffff;
            if (kind == kGpuMarkSubmission) {
                host_log("gpu: AMD marker: the GPU had %s %s", what, submission_mark(which).c_str());
                return;
            }
            host_log("gpu: AMD marker: the GPU had %s %s %llu", what, kind == 1 ? "draw" : kind == 2 ? "dispatch" : "(none yet)",
                     static_cast<unsigned long long>(which));
            if (kind == 1) describe_draw_record(which);
            if (kind == 2) {
                const char* n = g_dispatch_names[which % 256];
                host_log("    dispatch[%llu] %s", static_cast<unsigned long long>(which), n ? n : "?");
            }
        };
        say("finished everything before", finished);
        say("started", started);
        return;
    }
    auto get_data = reinterpret_cast<PFN_vkGetQueueCheckpointDataNV>(vkGetDeviceProcAddr(g.device, "vkGetQueueCheckpointDataNV"));
    if (!get_data) return;
    std::uint32_t n = 0;
    get_data(g.queue, &n, nullptr);
    if (!n) {
        host_log("gpu: the driver kept no checkpoints for the device loss (NVIDIA's Linux driver has given none at three); "
                 "BBHOST_GPU_CHECKPOINTS=amd names the draw through buffer markers instead");
        return;
    }
    std::vector<VkCheckpointDataNV> data(n, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
    get_data(g.queue, &n, data.data());
    host_log("gpu: %u checkpoint(s) at the device loss (the last work each stage reached):", n);
    for (std::uint32_t i = 0; i < n && i < 16; ++i) {
        const std::uint64_t m = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data[i].pCheckpointMarker));
        const unsigned kind = static_cast<unsigned>(m >> 56);
        const std::uint64_t which = m & 0xffffffffffffffull;
        if (kind == kGpuMarkSubmission) {
            host_log("  stage 0x%x: %s", data[i].stage, submission_mark(which).c_str());
            continue;
        }
        host_log("  stage 0x%x: %s %llu", data[i].stage, kind == 1 ? "draw" : kind == 2 ? "dispatch" : "?",
                 static_cast<unsigned long long>(which));
        if (kind == 1) describe_draw_record(which);
    }
}

// What the driver knows about the fault, when VK_EXT_device_fault is there.
// Not asked of NVIDIA's Windows driver, which wrote to 0 inside it after a
// loss (617.42; its Linux driver answers, with no addresses).
// BBHOST_DEVICE_FAULT_QUERY=1 asks it anyway.
void report_device_fault() {
    if (!g.has_device_fault) return;
#if defined(_WIN32)
    static const bool ask_nvidia = [] {
        const char* e = std::getenv("BBHOST_DEVICE_FAULT_QUERY");
        return e && e[0] == '1';
    }();
    if (g.vendor_id == 0x10de && !ask_nvidia) {
        host_log("gpu: the driver's fault report not asked for (NVIDIA's Windows driver crashed in it; BBHOST_DEVICE_FAULT_QUERY=1 asks)");
        return;
    }
#endif
    auto get_info = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(vkGetDeviceProcAddr(g.device, "vkGetDeviceFaultInfoEXT"));
    if (!get_info) return;
    VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    if (get_info(g.device, &counts, nullptr) != VK_SUCCESS) return;
    std::vector<VkDeviceFaultAddressInfoEXT> addrs(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
    VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
    info.pAddressInfos = addrs.empty() ? nullptr : addrs.data();
    info.pVendorInfos = vendors.empty() ? nullptr : vendors.data();
    if (get_info(g.device, &counts, &info) != VK_SUCCESS) return;
    host_log("gpu: device fault: %s (%u address info(s), %u vendor info(s))", info.description, counts.addressInfoCount,
             counts.vendorInfoCount);
    static const char* kind[] = {"none", "read invalid", "write invalid", "execute invalid",
                                 "instruction pointer unknown", "instruction pointer invalid", "instruction pointer fault"};
    for (std::uint32_t i = 0; i < counts.addressInfoCount && i < 8; ++i) {
        const VkDeviceFaultAddressInfoEXT& a = addrs[i];
        const unsigned t = static_cast<unsigned>(a.addressType);
        host_log("  fault %u: %s at 0x%llx (precision 0x%llx)", i, t < 7 ? kind[t] : "?",
                 static_cast<unsigned long long>(a.reportedAddress), static_cast<unsigned long long>(a.addressPrecision));
    }
    // Which of our device allocations the faulting address is in (or next to):
    // a read just past the end of an imported region looks exactly like this.
    auto describe = [&](std::uint64_t addr) {
        for (std::uint32_t c = 0; c < gcn::kDmemChunks; ++c) for (const auto& imp : g.dmem[c].imports) {
            const Chunk& ch = *imp;
            if (!ch.buffer || !ch.size) continue;
            if (addr >= ch.address && addr < ch.address + ch.size) {
                host_log("    inside dmem chunk %u (0x%llx +0x%llx), offset 0x%llx", c,
                         static_cast<unsigned long long>(ch.address), static_cast<unsigned long long>(ch.size),
                         static_cast<unsigned long long>(addr - ch.address));
                return;
            }
            if (addr >= ch.address + ch.size && addr < ch.address + ch.size + 0x100000) {
                host_log("    0x%llx past the end of dmem chunk %u (0x%llx +0x%llx)",
                         static_cast<unsigned long long>(addr - (ch.address + ch.size)), c,
                         static_cast<unsigned long long>(ch.address), static_cast<unsigned long long>(ch.size));
                return;
            }
        }
        for (const auto& kv : g.anon_imports) {
            const Chunk& ch = kv.second;
            if (!ch.size) continue;
            if (addr >= ch.address && addr < ch.address + ch.size) {
                host_log("    inside the import of guest 0x%llx (0x%llx +0x%llx), offset 0x%llx",
                         static_cast<unsigned long long>(kv.first), static_cast<unsigned long long>(ch.address),
                         static_cast<unsigned long long>(ch.size), static_cast<unsigned long long>(addr - ch.address));
                return;
            }
            if (addr >= ch.address + ch.size && addr < ch.address + ch.size + 0x100000) {
                host_log("    0x%llx past the end of the import of guest 0x%llx (0x%llx +0x%llx)",
                         static_cast<unsigned long long>(addr - (ch.address + ch.size)),
                         static_cast<unsigned long long>(kv.first), static_cast<unsigned long long>(ch.address),
                         static_cast<unsigned long long>(ch.size));
                return;
            }
        }
        host_log("    not in any imported region (sink 0x%llx, l1 0x%llx)", static_cast<unsigned long long>(g.sink.address),
                 static_cast<unsigned long long>(g.l1.address));
    };
    for (std::uint32_t i = 0; i < counts.addressInfoCount && i < 8; ++i) {
        describe(addrs[i].reportedAddress);
        // And where it is against the tessellation LDS ring, the buffer the
        // game's own hull draws keep their patches in (an RX 9070 XT lost the
        // device in those draws on a read nowhere near any import, 2026-10-08).
        const std::string tess = tess_lds_describe(addrs[i].reportedAddress, addrs[i].addressPrecision);
        if (!tess.empty()) host_log("    %s", tess.c_str());
    }
    for (std::uint32_t i = 0; i < counts.vendorInfoCount && i < 8; ++i) {
        host_log("  vendor %u: %s (code 0x%llx, data 0x%llx)", i, vendors[i].description,
                 static_cast<unsigned long long>(vendors[i].vendorFaultCode),
                 static_cast<unsigned long long>(vendors[i].vendorFaultData));
    }
}

// A run that loses the device leaves a note beside the pipeline cache, and the
// next kMarkedRuns starts run with the GPU's progress markers on (NVIDIA's
// checkpoints, AMD's buffer markers): off by default for what they cost, they
// are what names the draw the GPU was in when it is lost again - AMD's driver
// has said nothing through VK_EXT_device_fault.
constexpr int kMarkedRuns = 3;
std::string device_lost_note_path() {
    const char* root = hle_fs_data_root();
    if (!root || !*root) return {};
    return std::string(root) + "/bbhost/gpu-device-lost.txt";
}

// The note's starts left: true (and one fewer left) when this start is to run
// with markers; `why` says which run lost the device.
bool take_marked_start(std::string& why) {
    const std::string path = device_lost_note_path();
    if (path.empty()) return false;
    std::ifstream in(path);
    int left = 0;
    std::string word;
    if (!(in >> word >> left) || word != "starts-left" || left <= 0) return false;
    std::getline(in, why);
    std::getline(in, why);
    in.close();
    std::error_code ec;
    if (left == 1) {
        std::filesystem::remove(path, ec);
    } else {
        std::ofstream out(path, std::ios::trunc);
        out << "starts-left " << (left - 1) << '\n' << why << '\n';
    }
    return true;
}

std::atomic<bool> g_device_lost{false};

// The device is lost, however that showed (a wait, a submit in place or on the
// submission thread): GPU execution stops, and once - what the driver knows of
// the fault, how far the GPU's markers got, the last draws recorded, the other
// programs hooked in; the note for the next starts; and the flag the window's
// loop reads to tell the player and close (host_gpu_device_lost).
// `declared`: no driver said so - bbhost takes the device for lost because
// a submission never finished (fence_wait_locked).
void device_lost_locked(const char* where, bool declared) {
    g.ok = false;
    // Nothing recorded will run now: land the CP writes still queued.
    for (const auto& [va, span] : g.pending_writes) {
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(va)), span.data.data(), span.data.size());
    }
    g.pending_writes.clear();
    if (g_device_lost.exchange(true)) return;
    const unsigned long long flip = static_cast<unsigned long long>(hle_video_flip_count());
    if (declared) {
        host_log("gpu: the device is taken for lost (%s, flip %llu): the driver reports no loss, but the GPU never finished "
                 "that work, so nothing more is sent to it and bbhost ends",
                 where, flip);
    } else {
        host_log("gpu: the device is lost (%s, flip %llu): the GPU stopped responding and the system reset it", where, flip);
    }
    // The note for the next starts first, then what is ours to say, and the
    // driver's own diagnostics last: NVIDIA's Windows driver (617.42) crashed
    // inside vkGetDeviceFaultInfoEXT after a loss - a GTX 1660 SUPER's and an
    // RTX 2060's logs ended at this line with a write to 0 in nvoglv64.dll -
    // so the note was never written and their next starts ran without the
    // markers that would have said where the GPU stopped.
    const std::string hooks = host_foreign_hooks();
    const std::string path = device_lost_note_path();
    if (!path.empty()) {
        std::ofstream out(path, std::ios::trunc);
        out << "starts-left " << kMarkedRuns << '\n' << "flip " << flip << ", " << where << (hooks.empty() ? "" : ", with ") << hooks << '\n';
    }
    if (!g.has_checkpoints) host_log("gpu: no progress markers this run; the next %d starts run with them", kMarkedRuns);
    // Everything stops after this, so say what was in flight: the last draws
    // name the pipeline that faulted or hung.
    host_gpu_hang_report();  // with the command stream's last op (recorder.cpp)
    if (!hooks.empty()) host_log("gpu: hooked into this process: %s - such hooks are a common cause of lost devices", hooks.c_str());
    report_checkpoints();
    report_device_fault();
}

// Records a CP write that lands on the GPU later (see Gpu::pending_writes),
// replacing whatever pending bytes it overlaps.
void note_pending_write_locked(std::uint64_t va, const std::uint8_t* data, std::size_t bytes) {
    auto& spans = g.pending_writes;
    const std::uint64_t end = va + bytes;
    auto tail_of = [end](const std::pair<const std::uint64_t, Gpu::PendingSpan>& s) {
        Gpu::PendingSpan tail;
        tail.serial = s.second.serial;
        tail.data.assign(s.second.data.begin() + static_cast<std::ptrdiff_t>(end - s.first), s.second.data.end());
        return tail;
    };
    auto it = spans.upper_bound(va);
    if (it != spans.begin()) {
        auto prev = std::prev(it);
        const std::uint64_t prev_end = prev->first + prev->second.data.size();
        if (prev_end > va) {
            if (prev_end > end) spans.emplace(end, tail_of(*prev));
            prev->second.data.resize(va - prev->first);
            if (prev->second.data.empty()) spans.erase(prev);
        }
    }
    for (it = spans.lower_bound(va); it != spans.end() && it->first < end;) {
        if (it->first + it->second.data.size() > end) {
            Gpu::PendingSpan tail = tail_of(*it);
            spans.erase(it);
            spans.emplace(end, std::move(tail));
            break;
        }
        it = spans.erase(it);
    }
    Gpu::PendingSpan span;
    span.serial = g.flushes;
    span.data.assign(data, data + bytes);
    spans.emplace(va, std::move(span));
}

void read_guest_locked(std::uint64_t va, std::size_t bytes, void* out) {
    std::memcpy(out, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), bytes);
    if (g.pending_writes.empty()) return;
    const std::uint64_t end = va + bytes;
    auto it = g.pending_writes.upper_bound(va);
    if (it != g.pending_writes.begin()) --it;
    for (; it != g.pending_writes.end() && it->first < end; ++it) {
        const std::uint64_t lo = std::max(va, it->first);
        const std::uint64_t hi = std::min(end, it->first + it->second.data.size());
        if (lo < hi) {
            std::memcpy(static_cast<std::uint8_t*>(out) + (lo - va), it->second.data.data() + (lo - it->first), hi - lo);
        }
    }
}

namespace {
void heap_release(const ImageMemory& m);  // the image heap, below
void scratch_let_go_locked(const DevBuffer& b);  // the untile scratch, below

// An upload staging chunk no command buffer reads any more: kept for reuse up
// to kStagingKeepBytes, else freed.
void staging_let_go_locked(const DevBuffer& b) {
    if (g.staging_free_bytes + b.size <= kStagingKeepBytes) {
        g.staging_free.push_back(b);
        g.staging_free_bytes += b.size;
    } else {
        vkDestroyBuffer(g.device, b.buffer, nullptr);
        vkFreeMemory(g.device, b.memory, nullptr);
    }
}
}  // namespace

namespace {

// How long one wait for a submission's fence goes before the GPU is said to be
// still busy with it, and how long in all before the GPU is taken for stuck.
// BBHOST_GPU_WAIT_STEP_MS and BBHOST_GPU_WAIT_LIMIT_S shorten them for tests.
std::uint64_t fence_step_ns() {
    static const std::uint64_t ns = [] {
        const char* e = std::getenv("BBHOST_GPU_WAIT_STEP_MS");
        const std::uint64_t ms = e && *e ? std::strtoull(e, nullptr, 10) : 10000;
        return std::max<std::uint64_t>(ms, 100) * 1000000ull;
    }();
    return ns;
}
double fence_limit_s() {
    static const double s = [] {
        const char* e = std::getenv("BBHOST_GPU_WAIT_LIMIT_S");
        return e && *e ? std::max(1.0, std::atof(e)) : 60.0;
    }();
    return s;
}

// The first time a wait for slot k's submission runs past a step: what it
// holds, what stays held until it finishes, and the hang report with the
// GPU's progress markers where they can be read without a lost device.
void say_stuck_locked(int k, double secs) {
    const Gpu::Slot& sl = g.slots[k];
    std::size_t writes = 0;
    for (const auto& [va, span] : g.pending_writes) writes += span.serial >= sl.serial ? 1 : 0;
    host_log("gpu: submission %llu (slot %d, recorded at flip %llu) is still running after %.1f s: %llu draws, %llu of them "
             "tessellated, %llu dispatches. Until it finishes nothing it may still be reading is reset, freed or handed on - "
             "its fence and descriptor pool, %zu buffers, %zu staging and %zu scratch chunks, %zu image-heap ranges, %zu images "
             "and %zu views - and %zu CP writes it and the submissions after it carry wait; after %.0f s in all the GPU is taken "
             "for stuck",
             static_cast<unsigned long long>(sl.serial), k, static_cast<unsigned long long>(sl.held_flip), secs,
             static_cast<unsigned long long>(sl.held_draws), static_cast<unsigned long long>(sl.held_tess),
             static_cast<unsigned long long>(sl.held_dispatches), sl.garbage.size(), sl.staging_chunks.size(),
             sl.scratch_chunks.size(), sl.dead_ranges.size(), sl.dead_images.size(), sl.dead_views.size(), writes,
             fence_limit_s());
    host_gpu_hang_report();
    if (g.cmd_buffer_marker) report_checkpoints();  // AMD buffer markers: a buffer the GPU writes, readable at any time
}

// Waits for slot k's fence: true once it has signalled. A submission the GPU
// has not finished after a step is said once (say_stuck_locked) and waited
// for on - it may still be reading its descriptors, its memory and the
// staging, scratch and image-heap ranges it holds, and the game takes the CP
// writes it carries for the GPU being done, so none of that may be reset or
// handed to other work before the fence signals. After ten seconds the wait
// used to give up, reset the fence and the descriptor pool, free the rest
// and count the submission done while the GPU still ran it, so the
// recordings after it could write over what it read - a GPU that is slow
// once can then stay slow (an integration run in the Forbidden Woods went on
// at a frame every 90-110 s). One not finished within the limit is the GPU
// stuck: device_lost_locked, so bbhost ends with the report instead of going
// on; after that, or a real loss, nothing is waited for again. `waited_us`:
// the time spent here.
bool fence_wait_locked(int k, std::uint64_t& waited_us) {
    Gpu::Slot& sl = g.slots[k];
    waited_us = 0;
    if (g_device_lost.load(std::memory_order_acquire)) return false;
    const auto t0 = std::chrono::steady_clock::now();
    g.wait_started_ns.store(static_cast<std::uint64_t>(t0.time_since_epoch().count()), std::memory_order_relaxed);
    g.waiting_slot.store(k, std::memory_order_relaxed);
    VkResult r = VK_TIMEOUT;
    double secs = 0.0;
    bool slow = false;
    for (;;) {
        r = vkWaitForFences(g.device, 1, &sl.fence, VK_TRUE, fence_step_ns());
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (r != VK_TIMEOUT) break;
        slow = true;
        if (!sl.said_stuck) {
            sl.said_stuck = true;
            say_stuck_locked(k, secs);
        } else if (secs < fence_limit_s()) {
            host_log("gpu: submission %llu is still running after %.0f s", static_cast<unsigned long long>(sl.serial), secs);
        }
        if (secs >= fence_limit_s()) break;
    }
    g.waiting_slot.store(-1, std::memory_order_relaxed);
    waited_us = static_cast<std::uint64_t>(secs * 1e6);
    if (r == VK_SUCCESS) {
        if (slow) {
            host_log("gpu: submission %llu finished after %.1f s; what it held is let go now",
                     static_cast<unsigned long long>(sl.serial), secs);
        }
        return true;
    }
    if (r == VK_TIMEOUT) {
        host_log("gpu: submission %llu has not finished after %.0f s: the GPU is taken for stuck, and bbhost stops",
                 static_cast<unsigned long long>(sl.serial), secs);
        device_lost_locked("a submission the GPU had not finished within the limit", true);
        return false;
    }
    host_log("gpu: waiting for submission %d failed (%d)%s", k, r, r == VK_ERROR_DEVICE_LOST ? "; device lost, GPU execution disabled" : "");
    device_lost_locked(r == VK_ERROR_DEVICE_LOST ? "a wait for a submission" : "a failed wait for a submission");
    return false;
}

// BBHOST_TEST_GPU_STALL: the submission the job went into, waited for once it is in (stall_test_wait_locked).
std::uint64_t g_stall_serial = 0;
bool g_stall_pending = false;

// BBHOST_TEST_GPU_STALL=<flip>:<ms>: the first submission from that flip on
// carries a GPU job (shaders/gpu_stall.comp) that runs until the host lets it
// go <ms> later - 0: never, the job then ends by itself once its pieces are
// done - so a wait for its fence meets a GPU that is still busy:
// fence_wait_locked's steps, what stays held, the resume and the stop, tested
// without a real hang. The job is BBHOST_TEST_GPU_STALL_PIECES pieces (2000)
// of at most BBHOST_TEST_GPU_STALL_TURNS turns (4096) each, about 8 ms a piece
// on an RTX 4070, so the driver can switch the GPU to other work in between,
// and the command buffer stays small: one of 200,000 commands held
// vkQueueSubmit until the GPU had worked through most of it.
void stall_test_record_locked() {
    struct Cfg {
        std::uint64_t flip = 0, ms = 0, pieces = 2000, turns = 4096;
        bool on = false;
    };
    static const Cfg cfg = [] {
        Cfg c;
        if (const char* e = std::getenv("BBHOST_TEST_GPU_STALL"); e && *e) {
            char* end = nullptr;
            c.flip = std::strtoull(e, &end, 10);
            c.ms = end && *end == ':' ? std::strtoull(end + 1, nullptr, 10) : 0;
            c.on = true;
        }
        if (const char* e = std::getenv("BBHOST_TEST_GPU_STALL_PIECES"); e && *e) c.pieces = std::max<std::uint64_t>(1, std::strtoull(e, nullptr, 10));
        if (const char* e = std::getenv("BBHOST_TEST_GPU_STALL_TURNS"); e && *e) c.turns = std::strtoull(e, nullptr, 10);
        return c;
    }();
    static bool done = false;
    if (!cfg.on || done || hle_video_flip_count() < cfg.flip) return;
    done = true;
    static VkPipelineLayout layout = VK_NULL_HANDLE;
    static VkPipeline pipeline = VK_NULL_HANDLE;
    static DevBuffer flag;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(k_gpu_stall_spv);
    smi.pCode = k_gpu_stall_spv;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    bool ok = vkCreateShaderModule(g.device, &smi, nullptr, &module) == VK_SUCCESS &&
              vkCreatePipelineLayout(g.device, &pli, nullptr, &layout) == VK_SUCCESS;
    if (ok) {
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = "main";
        ci.layout = layout;
        ok = vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline) == VK_SUCCESS;
    }
    if (module) vkDestroyShaderModule(g.device, module, nullptr);
    if (!ok || !create_dev_buffer(flag, 64, true) || !flag.map) {
        host_log("gpu: test: the GPU job of BBHOST_TEST_GPU_STALL could not be made");
        return;
    }
    volatile std::uint32_t* w = static_cast<volatile std::uint32_t*>(flag.map);
    for (int i = 0; i < 4; ++i) w[i] = 0;
    struct Push {
        std::uint64_t flag;
        std::uint32_t max_turns, pad;
    } push{flag.address, static_cast<std::uint32_t>(std::min<std::uint64_t>(cfg.turns, 0xffffffffu)), 0};
    g_stall_serial = g.flushes;  // this recording's submission
    g_stall_pending = true;
    rec().bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    rec().push_constants(layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    for (std::uint64_t p = 0; p < cfg.pieces; ++p) {
        rec().dispatch(1, 1, 1);
        if (p + 1 < cfg.pieces) {
            rec().pipeline_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
    }
    char when[48];
    if (cfg.ms) std::snprintf(when, sizeof(when), "%llu ms later", static_cast<unsigned long long>(cfg.ms));
    else std::snprintf(when, sizeof(when), "never");
    host_log("gpu: test: submission %llu (flip %llu) carries a GPU job of %llu pieces of up to %llu turns that runs until the host "
             "lets it go: %s (BBHOST_TEST_GPU_STALL)",
             static_cast<unsigned long long>(g.flushes), static_cast<unsigned long long>(hle_video_flip_count()),
             static_cast<unsigned long long>(cfg.pieces), static_cast<unsigned long long>(cfg.turns), when);
    if (cfg.ms) {
        std::thread([ms = cfg.ms, w] {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            const std::uint32_t pieces = w[2], turns = w[1];
            w[0] = 1;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            host_log("gpu: test: the GPU job is let go %llu ms after it was recorded, %u pieces and %u turns in",
                     static_cast<unsigned long long>(ms), pieces, turns);
        }).detach();
    }
}

// BBHOST_TEST_GPU_STALL: once the job's submission is in, the command processor
// waits for it, as it does when it needs that slot again or flushes. The game
// itself would first wait for its labels, and the job would never meet a wait.
void stall_test_wait_locked() {
    if (!g_stall_pending || g.flushes <= g_stall_serial) return;
    g_stall_pending = false;
    host_log("gpu: test: the command processor waits for submission %llu", static_cast<unsigned long long>(g_stall_serial));
    retire_through_locked(g_stall_serial);
}

}  // namespace

namespace {
const bool g_indirect_stats = [] {
    const char* e = std::getenv("BBHOST_INDIRECT_STATS");
    return e && e[0] == '1';
}();
struct IndirectRec {
    std::uint64_t serial, va;
    std::uint8_t kind;
    bool counting;
};
std::vector<IndirectRec> g_indirect_recs;  // under g.mu
struct IndirectAgg {
    std::uint64_t dispatches = 0, counting = 0, nonzero = 0, groups = 0, max_groups = 0, over_limit = 0;
    std::uint64_t draws = 0, drawn = 0, elements = 0, max_elements = 0;
    std::uint64_t first_flip = 0;
};
IndirectAgg g_indirect_agg;

// The arguments of the indirect work submission `serial` held, read now that
// it is done; once a second (60 flips) what they came to.
void indirect_stats_retired_locked(std::uint64_t serial) {
    if (!g_indirect_stats || g_indirect_recs.empty()) return;
    IndirectAgg& a = g_indirect_agg;
    std::size_t keep = 0;
    for (const IndirectRec& r : g_indirect_recs) {
        if (r.serial > serial) {
            g_indirect_recs[keep++] = r;
            continue;
        }
        std::uint32_t v[5] = {};
        if (!hle_kernel_va_mapped(r.va, r.kind == 2 ? 20 : r.kind == 1 ? 16 : 12)) continue;
        std::memcpy(v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(r.va)), r.kind == 2 ? 20 : r.kind == 1 ? 16 : 12);
        if (r.kind == 0) {
            ++a.dispatches;
            a.counting += r.counting ? 1 : 0;
            const std::uint64_t groups = static_cast<std::uint64_t>(v[0]) * v[1] * v[2];
            a.nonzero += groups ? 1 : 0;
            a.groups += groups;
            a.max_groups = std::max(a.max_groups, groups);
            a.over_limit += v[0] > 65535 || v[1] > 65535 || v[2] > 65535 ? 1 : 0;
        } else {
            ++a.draws;
            const std::uint64_t elements = static_cast<std::uint64_t>(v[0]) * v[1];
            a.drawn += elements ? 1 : 0;
            a.elements += elements;
            a.max_elements = std::max(a.max_elements, elements);
        }
    }
    g_indirect_recs.resize(keep);
    const std::uint64_t flip = hle_video_flip_count();
    if (!a.first_flip) a.first_flip = flip ? flip : 1;
    if (flip < a.first_flip + 60) return;
    host_log("gpu: indirect at flips %llu..%llu: %llu dispatches (%llu counting in GDS), %llu with work, %llu workgroups (most %llu, %llu over "
             "65535 in a dimension); %llu draws, %llu drawing, %llu elements (most %llu)",
             static_cast<unsigned long long>(a.first_flip), static_cast<unsigned long long>(flip), static_cast<unsigned long long>(a.dispatches),
             static_cast<unsigned long long>(a.counting), static_cast<unsigned long long>(a.nonzero), static_cast<unsigned long long>(a.groups),
             static_cast<unsigned long long>(a.max_groups), static_cast<unsigned long long>(a.over_limit), static_cast<unsigned long long>(a.draws),
             static_cast<unsigned long long>(a.drawn), static_cast<unsigned long long>(a.elements), static_cast<unsigned long long>(a.max_elements));
    a = IndirectAgg{};
    a.first_flip = flip + 1;
}
}  // namespace

void note_indirect_locked(std::uint64_t args_va, int kind, bool counting) {
    if (!g_indirect_stats || !args_va || g_indirect_recs.size() >= (1u << 20)) return;
    g_indirect_recs.push_back({g.flushes, args_va, static_cast<std::uint8_t>(kind), counting});
}

// The presenter's own-queue copy recorded while a slot was the one being
// recorded reads what retiring that slot frees: the game's display image, the
// slot's descriptor sets. It waits on the GPU only for the game's submissions
// up to its frame, which the slot's fence says are done, so it has run or is
// about to: a short wait, bounded (5 s) for a submit that never went in.
// The counter is read and the thread sleeps between reads rather than waiting
// inside vkWaitSemaphores: on NVIDIA's Windows driver a thread in that wait
// held the driver's own lock, and the presenter's submit - what signals the
// value - queued behind it with every other thread's driver call (a 5080 hung
// at the title in each of a friend's runs, 2026-10-14). The wait is nearly
// always over at the first read.
void present_reads_wait(std::uint64_t target) {
    if (!g.present_timeline || g_device_lost.load(std::memory_order_relaxed)) return;
    const auto start = std::chrono::steady_clock::now();
    for (int k = 0;; ++k) {
        std::uint64_t v = 0;
        if (vkGetSemaphoreCounterValue(g.device, g.present_timeline, &v) != VK_SUCCESS || v >= target ||
            g_device_lost.load(std::memory_order_relaxed))
            return;
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) break;
        if (k < 64) std::this_thread::yield();
        else host_sleep_us(250);
    }
    static std::atomic<int> said{0};
    if (said.fetch_add(1) < 4)
        host_log("gpu: the presenter's copy %llu not done 5 s after its frame; its slot retires anyway", static_cast<unsigned long long>(target));
}

namespace {
// The label landing's view of the GPU (label_landing::Hooks): the timeline's
// own value and wait - host_gpu_submissions_completed and
// host_gpu_wait_completed land, and the thread is the one landing.
std::uint64_t timeline_value() {
    std::uint64_t v = 0;
    if (g_device_lost.load(std::memory_order_relaxed) || vkGetSemaphoreCounterValue(g.device, g.timeline, &v) != VK_SUCCESS) return ~0ull;
    return v;
}
bool timeline_wait(std::uint64_t need, std::uint64_t timeout_ns) {
    if (g_device_lost.load(std::memory_order_relaxed)) return true;
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &g.timeline;
    wi.pValues = &need;
    return vkWaitSemaphores(g.device, &wi, timeout_ns) == VK_SUCCESS;
}
void landing_thread_start() {
    host_thread_set_name("bb-labels");
    host_thread_set_class(HostThreadClass::GpuFeed, "bb-labels");
}

void start_label_landing() {
    if (!label_landing::on()) return;
    if (!g_timeline_on.load(std::memory_order_relaxed)) {
        host_log("gpu: BBHOST_LABEL_LANDING=1 needs a timeline semaphore: the command processor's labels stay GPU fills");
        return;
    }
    label_landing::Hooks h;
    h.completed = timeline_value;
    h.submitted = stream_submitted_serial;
    h.wait = timeline_wait;
    h.mapped = hle_kernel_va_mapped;
    h.thread_start = landing_thread_start;
    h.reused = [](std::uint64_t va, std::uint32_t bytes, std::uint64_t epoch, label_landing::Reused* out) {
        host_lists::Reuse at;
        if (host_lists::rounds_since(va, bytes, epoch, &at) <= 0) return false;
        out->chunk_lo = at.chunk_lo;
        out->chunk_end = at.chunk_end;
        out->rounds_later = at.rounds_later;
        out->written_to = at.written_to;
        out->arena = at.arena;
        out->unsubmitted = at.unsubmitted;
        return true;
    };
    label_landing::start(h);
    host_log("gpu: the command processor's labels are written by the host once their submission has run, not filled by the GPU "
             "(BBHOST_LABEL_LANDING=1)");
}

// At submit: the labels this submission carries go to the landing, written
// by the host once it has run, in this order (label_landing.h) - in place of
// flush_deferred_writes_locked's fills and the barriers between them.
void hand_labels_to_landing_locked() {
    if (g.deferred_writes.empty()) return;
    static std::vector<label_landing::Write> ws;  // under g.mu
    ws.clear();
    for (const Gpu::SmallWrite& w : g.deferred_writes) {
        label_landing::Write l{w.va, w.bytes, {}, w.epoch, w.op};
        std::memcpy(l.data, w.data, sizeof(l.data));
        ws.push_back(l);
    }
    label_landing::add(g.flushes, ws.data(), ws.size());
    g.deferred_writes.clear();
}
}  // namespace

void fills_retired_locked(Gpu::Slot& sl);  // below, with the fills
void retire_slot_locked(int k) {
    Gpu::Slot& sl = g.slots[k];
    if (!sl.in_flight) return;
    // A fence that never went to the queue would hold the wait below for its
    // ten seconds: the recorder hands the submission over first (it has, 16
    // submissions on, unless it fell that far behind).
    stream_wait_submitted(sl.serial);
    const std::uint64_t retire_t0 = host_clock_monotonic_ns();
    struct RetireTime {
        std::uint64_t t0;
        ~RetireTime() {
            bump(g_stream_stats.retire_ns, host_clock_monotonic_ns() - t0);
            bump(g_stream_stats.retires);
        }
    } retire_time{retire_t0};
    take_submit_failure_locked();
    if (!g.ok && g_sub_failed.load(std::memory_order_relaxed)) {
        // Its submission may never have gone in: the fence would never signal.
        QueueGuard queue;
        vkDeviceWaitIdle(g.device);
        sl.in_flight = false;
        busy_retired_locked(sl, false);
        return;
    }
    std::uint64_t fence_us = 0;
    const bool done = fence_wait_locked(k, fence_us);
    g.gpu_us.fetch_add(fence_us);
    bump(g_stream_stats.fence_ns, fence_us * 1000);
    if (!done) {
        // The GPU is lost, or taken for stuck: what the submission holds stays
        // as it is (the GPU may still read it), nothing more goes to the GPU
        // (submit_locked), and bbhost ends.
        sl.in_flight = false;
        return;
    }
    if (sl.present_target) {
        present_reads_wait(sl.present_target);
        sl.present_target = 0;
    }
    vkResetFences(g.device, 1, &sl.fence);
    indirect_stats_retired_locked(sl.serial);
    indirect_clamp_retired_locked(sl.serial);
    particle_args_retired_locked(sl.serial);
    vb_race_retired_locked(sl.serial);
    vkResetDescriptorPool(g.device, sl.pool, 0);
    sl.spare_sets.clear();
    for (Chunk& c : sl.dead_imports) free_import(c);  // window imports a merged one replaced
    sl.dead_imports.clear();
    for (DevBuffer& b : sl.garbage) {
        vkDestroyBuffer(g.device, b.buffer, nullptr);
        vkFreeMemory(g.device, b.memory, nullptr);
    }
    sl.garbage.clear();
    untile_checks_run_locked(sl.untile_checks);  // before their staging is reused
    gxvk_slot_retired_locked(k);  // BBHOST_GXVK_BUFFERS=2: the copies this slot's uploads made, read back
    for (const DevBuffer& b : sl.staging_chunks) {
        const auto it = g.staging_users.find(b.buffer);
        if (it == g.staging_users.end() || --it->second) continue;
        g.staging_users.erase(it);
        if (b.buffer != g.staging_cur.buffer) staging_let_go_locked(b);
    }
    sl.staging_chunks.clear();
    for (const DevBuffer& b : sl.scratch_chunks) {
        const auto it = g.scratch_users.find(b.buffer);
        if (it == g.scratch_users.end() || --it->second) continue;
        g.scratch_users.erase(it);
        if (b.buffer != g.scratch_cur.buffer) scratch_let_go_locked(b);
    }
    sl.scratch_chunks.clear();
    bindless_slot_done_locked(k);  // after this slot's fence: no work in flight reads its retired views' slots
    for (VkImageView v : sl.dead_views) vkDestroyImageView(g.device, v, nullptr);
    sl.dead_views.clear();
    for (VkImage i : sl.dead_images) vkDestroyImage(g.device, i, nullptr);
    sl.dead_images.clear();
    for (VkDeviceMemory m : sl.dead_memory) vkFreeMemory(g.device, m, nullptr);
    sl.dead_memory.clear();
    for (const ImageMemory& m : sl.dead_ranges) heap_release(m);  // after the images bound to them are gone
    sl.dead_ranges.clear();
    for (const auto& [pool, set] : sl.dead_sets) vkFreeDescriptorSets(g.device, pool, 1, &set);
    sl.dead_sets.clear();
    sl.in_flight = false;
    // Fences on one queue complete in submission order, so every CP write
    // recorded into this submission or an earlier one is in memory now - the
    // labels the host writes too, landed here if the landing thread has not
    // yet (label_landing.h).
    if (landing_on()) label_landing::land_through(sl.serial + 1);
    if (!sl.fill_checks.empty()) fills_retired_locked(sl);
    g.completed_submits = std::max(g.completed_submits, sl.serial + 1);
    g_completed_submits_seen.store(g.completed_submits, std::memory_order_relaxed);
    for (auto it = g.pending_writes.begin(); it != g.pending_writes.end();) {
        it = it->second.serial < g.completed_submits ? g.pending_writes.erase(it) : std::next(it);
    }
    collect_profile_locked(sl);
    busy_retired_locked(sl, true);
    glitch_slot_done_locked(sl);
    textures_retired(k);
}

void defer_destroy(const DevBuffer& b) { g.slots[g.slot].garbage.push_back(b); }

// Texture uploads allocated and mapped a staging buffer each, and freed it once
// the copy retired: 258 ms of a 639-upload world-load stall, and size-classed
// reuse still left 211 ms because most sizes were new. Uploads are now placed
// one after another in 64 MiB mapped chunks (a larger upload gets a chunk of its
// own) that belong to the command buffer recording them; once it retired, its
// chunks are free for reuse, and up to kStagingKeepBytes of them are kept.
std::atomic<std::uint64_t> g_staging_created{0}, g_staging_placed{0};
std::uint64_t staging_chunk_bytes() {
    static const std::uint64_t bytes = [] {
        const char* e = std::getenv("BBHOST_STAGING_CHUNK_MB");
        const std::uint64_t mib = e && *e ? std::strtoull(e, nullptr, 10) : 64;
        return std::clamp<std::uint64_t>(mib, 4, 64) << 20;
    }();
    return bytes;
}
// Chunks the memory keeper gave back in play (BBHOST_STAGING_RETURN), and their vkFreeMemory in microseconds.
std::atomic<std::uint64_t> g_staging_returned{0}, g_staging_return_us{0}, g_staging_return_us_max{0};

//
// The command buffers that follow one another share a chunk: one per command
// buffer held a whole 64 MiB chunk for a few uploads until it retired, and with
// 16 in flight that was 128 MiB more staging in play (2026-10-04).
bool acquire_staging_locked(DevBuffer& b, std::uint64_t size, VkDeviceSize& offset) {
    Gpu::Slot& sl = g.slots[g.slot];
    const std::uint64_t need = (size + 63) & ~63ull;  // a multiple of every texel block size
    // No command buffer reads the current chunk any more: start it over.
    if (g.staging_cur.buffer && !g.staging_users.count(g.staging_cur.buffer)) g.staging_cur_used = 0;
    if (!g.staging_cur.buffer || g.staging_cur_used + need > g.staging_cur.size) {
        const std::uint64_t want = std::max(need, staging_chunk_bytes());
        DevBuffer chunk;
        auto it = std::find_if(g.staging_free.begin(), g.staging_free.end(), [&](const DevBuffer& c) { return c.size >= want; });
        if (it != g.staging_free.end()) {
            chunk = *it;
            g.staging_free_bytes -= it->size;
            g.staging_free.erase(it);
        } else {
            MemOwner own_mem("upload staging");
            if (!create_dev_buffer(chunk, want, true)) return false;
            g_staging_created.fetch_add(1, std::memory_order_relaxed);
        }
        if (g.staging_cur.buffer && !g.staging_users.count(g.staging_cur.buffer)) staging_let_go_locked(g.staging_cur);
        g.staging_cur = chunk;
        g.staging_cur_used = 0;
    }
    if (sl.staging_chunks.empty() || sl.staging_chunks.back().buffer != g.staging_cur.buffer) {
        sl.staging_chunks.push_back(g.staging_cur);
        ++g.staging_users[g.staging_cur.buffer];
    }
    b = g.staging_cur;
    b.map = static_cast<std::uint8_t*>(g.staging_cur.map) + g.staging_cur_used;
    b.size = need;
    offset = g.staging_cur_used;
    g.staging_cur_used += need;
    g_staging_placed.fetch_add(1, std::memory_order_relaxed);
    return true;
}

namespace {
constexpr std::uint64_t kScratchChunkBytes = 64ull << 20;
constexpr std::uint64_t kScratchKeepBytes = 256ull << 20;
std::atomic<std::uint64_t> g_scratch_created{0}, g_scratch_placed{0};
std::atomic<std::int64_t> g_scratch_used_ms{0};      // when a chunk was last taken (steady ms)
std::atomic<std::uint64_t> g_scratch_returned{0};  // chunks the memory keeper gave back in play
std::atomic<std::uint64_t> g_scratch_refusals{0};  // chunks not made for want of room above the reserve
void scratch_let_go_locked(const DevBuffer& b) {
    // Not kept while video memory is down to the reserve, nor one in host
    // memory (made while video memory was full): the next chunk is made in
    // video memory if there is room by then (vram_return_on). With small
    // video memory (small_vram) one chunk is kept, not four.
    const std::uint64_t keep = g.small_vram ? kScratchChunkBytes : kScratchKeepBytes;
    if (g.scratch_free_bytes + b.size <= keep && !vram_within_reserve() && !(b.in_host && vram_return_on())) {
        g.scratch_free.push_back(b);
        g.scratch_free_bytes += b.size;
    } else {
        vkDestroyBuffer(g.device, b.buffer, nullptr);
        vkFreeMemory(g.device, b.memory, nullptr);
    }
}
}  // namespace

bool acquire_scratch_locked(DevBuffer& b, std::uint64_t size, VkDeviceSize& offset) {
    Gpu::Slot& sl = g.slots[g.slot];
    g_scratch_used_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(),
                            std::memory_order_relaxed);
    const std::uint64_t need = (size + 255) & ~255ull;
    if (g.scratch_cur.buffer && !g.scratch_users.count(g.scratch_cur.buffer)) g.scratch_cur_used = 0;
    if (!g.scratch_cur.buffer || g.scratch_cur_used + need > g.scratch_cur.size) {
        const std::uint64_t want = std::max(need, kScratchChunkBytes);
        DevBuffer chunk;
        auto it = std::find_if(g.scratch_free.begin(), g.scratch_free.end(), [&](const DevBuffer& c) { return c.size >= want; });
        if (it != g.scratch_free.end()) {
            chunk = *it;
            g.scratch_free_bytes -= it->size;
            g.scratch_free.erase(it);
        } else {
            // No new chunk out of the reserve: the upload takes the staging
            // route (host memory) instead. A chunk made there was what
            // overflowed a 6 GB card at a load and took every mirror with it.
            if (vram_within_reserve(want)) {
                g_scratch_refusals.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            MemOwner own_mem("untile scratch");
            if (!create_dev_buffer(chunk, want, false)) return false;
            g_scratch_created.fetch_add(1, std::memory_order_relaxed);
        }
        if (g.scratch_cur.buffer && !g.scratch_users.count(g.scratch_cur.buffer)) scratch_let_go_locked(g.scratch_cur);
        g.scratch_cur = chunk;
        g.scratch_cur_used = 0;
    }
    if (sl.scratch_chunks.empty() || sl.scratch_chunks.back().buffer != g.scratch_cur.buffer) {
        sl.scratch_chunks.push_back(g.scratch_cur);
        ++g.scratch_users[g.scratch_cur.buffer];
    }
    b = g.scratch_cur;
    offset = g.scratch_cur_used;
    g.scratch_cur_used += need;
    g_scratch_placed.fetch_add(1, std::memory_order_relaxed);
    return true;
}

std::string staging_pool_stats() {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "chunks created %llu, uploads placed %llu, given back in play %llu (vkFreeMemory %.1f ms in all, the longest %.1f ms); "
                  "device-local untile scratch: chunks created %llu, spans placed %llu",
                  static_cast<unsigned long long>(g_staging_created.load()), static_cast<unsigned long long>(g_staging_placed.load()),
                  static_cast<unsigned long long>(g_staging_returned.load()), g_staging_return_us.load() / 1000.0,
                  g_staging_return_us_max.load() / 1000.0, static_cast<unsigned long long>(g_scratch_created.load()),
                  static_cast<unsigned long long>(g_scratch_placed.load()));
    return buf;
}

namespace {
constexpr VkDeviceSize kHeapBlock = 256ull << 20;
constexpr VkDeviceSize kHeapLargest = 64ull << 20;  // larger images get their own memory
struct HeapBlock {
    VkDeviceMemory memory = VK_NULL_HANDLE;  // null: released, the slot free for the next block
    std::uint32_t type = 0;
    std::map<VkDeviceSize, VkDeviceSize> free;  // offset -> size, coalesced
    VkDeviceSize used = 0;
    std::chrono::steady_clock::time_point empty_since{};  // when used last fell to 0
    // What it holds (HeapClass, gpu_internal.h): set by its first image,
    // cleared when it empties (-1: nothing yet, any class may take it).
    std::int8_t cls = -1;
};
std::vector<HeapBlock> g_heap;  // under g.mu; an image's ImageMemory::block indexes it
std::uint64_t g_heap_allocs = 0, g_heap_own = 0, g_heap_released = 0, g_heap_cp_blocks = 0;
// Blocks made on the command processor while the game played (in the world,
// no loading screen up): what a compaction must not cause - each a driver
// allocation in the middle of a frame (~190 ms on a Radeon 680M).
std::uint64_t g_heap_cp_play = 0, g_heap_cp_play_us = 0, g_heap_cp_play_max_us = 0;

// A block being emptied (drain_target_blocks_locked or, in its place,
// compact_image_heap_locked, render.cpp): no image goes in it while it is, so
// what moves out of it and whatever comes next fill room elsewhere, and it can
// empty for the keeper to give back.
int g_heap_avoid = -1;

// A new block's slot: a released one if any, so the indices live images
// hold never move.
void heap_add_block_locked(VkDeviceMemory memory, std::uint32_t type) {
    HeapBlock nb;
    nb.memory = memory;
    nb.type = type;
    nb.free[0] = kHeapBlock;
    nb.empty_since = std::chrono::steady_clock::now();
    for (std::size_t k = 0; k < g_heap.size(); ++k) {
        HeapBlock& b = g_heap[k];
        if (b.memory == VK_NULL_HANDLE) {
            b = std::move(nb);
            // A new block in the slot of one being emptied (given back since)
            // is not that block: images may go in it.
            if (g_heap_avoid == static_cast<int>(k)) g_heap_avoid = -1;
            return;
        }
    }
    g_heap.push_back(std::move(nb));
}

std::size_t heap_live_blocks_locked() {
    std::size_t n = 0;
    for (const HeapBlock& b : g_heap) n += b.memory != VK_NULL_HANDLE;
    return n;
}

// Render targets and textures in blocks of their own (BBHOST_HEAP_CLASSES,
// on unless 0). An area's targets are made as it loads and live as long as
// it does, its textures come and go as the game streams them: placed in the
// same blocks, every block kept some of an area's targets, so none ever
// emptied for the keeper to give back - after ten places 9 blocks (2.3 GiB)
// held 1.3-1.9 GiB of images (render/next22 at 1080p). Apart, the textures'
// blocks empty as the game lets an area's textures go. 0: one class.
// A texture still takes free room in the targets' blocks before an empty
// block or a new one: kept apart, the targets' holes held 330-360 MiB at a
// lamp that no texture could use, and a Radeon 680M's run ended with 9 blocks
// where one class had needed 6. A target never goes in the textures' blocks,
// so those still empty. strict: each class in its own blocks only.
const bool g_heap_classes = [] {
    const char* e = std::getenv("BBHOST_HEAP_CLASSES");
    return !(e && e[0] == '0');
}();
const bool g_heap_overflow = [] {
    const char* e = std::getenv("BBHOST_HEAP_CLASSES");
    return !(e && std::strcmp(e, "strict") == 0);
}();

// Whether an image of class `cls` may go in a block of class `block_cls`
// (-1: empty).
bool heap_class_fits(int block_cls, int cls) { return heap_place::class_fits(g_heap_classes, g_heap_overflow, block_cls, cls); }

VkDeviceSize heap_free_locked(std::uint32_t type, std::size_t* live) {
    VkDeviceSize free = 0;
    std::size_t n = 0;
    for (const HeapBlock& b : g_heap) {
        if (b.memory == VK_NULL_HANDLE || b.type != type) continue;
        free += kHeapBlock - b.used;
        ++n;
    }
    if (live) *live = n;
    return free;
}

// The room the memory keeper counts as its spare in memory type `type`: with
// the image heap's classes, the empty blocks and the least free room the next
// target or the next texture can take (a target the targets' blocks only, a
// texture its own blocks and, unless strict, the targets' holes) - room the
// next image cannot use makes a block on the command processor (5-10 ms on
// NVIDIA, ~190 ms on a Radeon 680M). Without them every block's free room, as
// before.
VkDeviceSize heap_spare_room_locked(std::uint32_t type) {
    if (!g_heap_classes) return heap_free_locked(type, nullptr);
    VkDeviceSize empty = 0, holes[2] = {};
    for (const HeapBlock& b : g_heap) {
        if (b.memory == VK_NULL_HANDLE || b.type != type) continue;
        if (b.cls < 0) empty += kHeapBlock - b.used;
        else holes[b.cls == kHeapTextures ? 1 : 0] += kHeapBlock - b.used;
    }
    const VkDeviceSize for_textures = holes[1] + (g_heap_overflow ? holes[0] : 0);
    return empty + std::min(holes[0], for_textures);
}

void heap_release(const ImageMemory& m) {
    HeapBlock& b = g_heap[static_cast<std::size_t>(m.block)];
    b.used -= m.size;
    if (b.used == 0) {
        b.empty_since = std::chrono::steady_clock::now();
        b.cls = -1;
    }
    auto next = b.free.lower_bound(m.offset);
    VkDeviceSize off = m.offset, size = m.size;
    if (next != b.free.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == off) {
            off = prev->first;
            size += prev->second;
            b.free.erase(prev);
        }
    }
    if (next != b.free.end() && off + size == next->first) {
        size += next->second;
        b.free.erase(next);
    }
    b.free[off] = size;
}
}  // namespace

namespace {
// When video memory last refused a block or an image of its own (steady
// milliseconds). For a second after, only room already in its blocks is tried
// there before host memory, and for ten the keeper makes no spare blocks -
// each refusal is a driver call, and an area's load makes hundreds of images.
std::atomic<std::int64_t> g_vram_refused_ms{-(1ll << 40)};
std::int64_t steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::uint64_t g_heap_host = 0;  // images placed in host memory for want of video memory

// The image heap's placement in memory type `type`: a free range in one of
// its blocks; else, when `grow`, a new block, or memory of its own for an
// image too large for one.
// Blocks are tried fullest first: new images go where the heap is already
// dense, so the emptier blocks drain as the game frees what they hold and the
// keeper can give them back. First fit in block order kept every block a
// little used - after ten places 9 blocks (2.3 GiB) for 1.2-1.9 GiB of
// images. BBHOST_HEAP_FULLEST=0: block order, as before.
const bool g_heap_fullest = [] {
    const char* e = std::getenv("BBHOST_HEAP_FULLEST");
    return !(e && e[0] == '0');
}();

bool heap_place_locked(const VkMemoryRequirements& req, std::uint32_t type, bool grow, ImageMemory& out, int cls) {
    const VkDeviceSize align = std::max<VkDeviceSize>(req.alignment, 256);
    if (!g_heap_classes) cls = kHeapTargets;
    if (req.size <= kHeapLargest) {
        for (int pass = 0; pass < 2; ++pass) {
            thread_local std::vector<std::size_t> order;
            order.clear();
            for (std::size_t k = 0; k < g_heap.size(); ++k) {
                const HeapBlock& b = g_heap[k];
                if (b.memory != VK_NULL_HANDLE && b.type == type && kHeapBlock - b.used >= req.size && heap_class_fits(b.cls, cls) &&
                    static_cast<int>(k) != g_heap_avoid)
                    order.push_back(k);
            }
            // Its own class's blocks, then (a texture) the targets' holes, then
            // an empty block; fullest first in each (host/heap_place.h, which
            // the compaction's trial placement shares).
            heap_place::order(g_heap, order, cls, g_heap_fullest);
            for (const std::size_t k : order) {
                HeapBlock& b = g_heap[k];
                std::uint64_t at = 0;
                if (!heap_place::first_fit(b.free, req.size, align, at)) continue;
                heap_place::take(b.free, at, req.size);
                b.used += req.size;
                if (b.cls < 0) b.cls = static_cast<std::int8_t>(cls);
                out = {b.memory, at, req.size, static_cast<std::int32_t>(k), type};
                ++g_heap_allocs;
                return true;
            }
            if (pass || !grow) break;
            // No room: another block, made here on the command processor
            // (5-10 ms) because the keeper's spare ran out (memory_keeper).
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = kHeapBlock;
            mai.memoryTypeIndex = type;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            MemOwner own_mem("image heap blocks");
            const auto t0 = std::chrono::steady_clock::now();
            if (vkAllocateMemory(g.device, &mai, nullptr, &mem) != VK_SUCCESS) break;
            const std::uint64_t us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
            heap_add_block_locked(mem, type);
            ++g_heap_cp_blocks;
            const bool loading = loading_screen_now(), in_play = !loading && world_reached();
            if (in_play) {
                ++g_heap_cp_play;
                g_heap_cp_play_us += us;
                g_heap_cp_play_max_us = std::max(g_heap_cp_play_max_us, us);
            }
            host_log("gpu: image heap block made on the command processor (%zu live, %llu MiB each, %.1f ms, %s)%s", heap_live_blocks_locked(),
                     static_cast<unsigned long long>(kHeapBlock >> 20), us / 1000.0,
                     in_play ? "in play" : loading ? "during a load" : "before the world",
                     g.mem_props.memoryTypes[type].heapIndex == g.local_heap ? "" : " in host memory");
        }
    }
    if (!grow) return false;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    MemOwner own_mem("images of their own");
    if (vkAllocateMemory(g.device, &mai, nullptr, &out.memory) != VK_SUCCESS) return false;
    out.size = req.size;
    out.type = type;
    ++g_heap_own;
    return true;
}

// The video memory budget, followed as it changes (vram_headroom): the
// driver's figures read at most every 100 ms, and what our own allocations
// took or gave back since - which the driver's next reading includes - from
// our own count, so a load's burst of blocks is seen at once.
std::mutex g_budget_mu;
std::int64_t g_budget_read_ms = -(1ll << 40);
std::int64_t g_budget_room = 0;            // heapBudget - heapUsage when read
std::uint64_t g_budget_local_at_read = 0;  // bb_memory_local_bytes() then
std::uint64_t g_vram_reserve = 0;          // vram_reserve_init
std::atomic<std::uint64_t> g_reserve_refusals{0};
}  // namespace

std::uint64_t image_heap_vram_free_locked(int cls) {
    std::uint64_t free = 0;
    for (const HeapBlock& b : g_heap) {
        if (b.memory == VK_NULL_HANDLE || g.mem_props.memoryTypes[b.type].heapIndex != g.local_heap) continue;
        if (cls >= 0 && !heap_class_fits(b.cls, cls)) continue;
        free += kHeapBlock - b.used;
    }
    return free;
}

bool image_heap_classes_on() { return g_heap_classes; }

int image_heap_drain_candidate_locked(std::uint64_t* class_free, std::uint64_t* used, const std::vector<int>& skip) {
    std::uint64_t free = 0;
    int best = -1;
    std::size_t blocks = 0;
    for (std::size_t k = 0; k < g_heap.size(); ++k) {
        const HeapBlock& b = g_heap[k];
        if (b.memory == VK_NULL_HANDLE || b.cls != kHeapTextures || g.mem_props.memoryTypes[b.type].heapIndex != g.local_heap) continue;
        free += kHeapBlock - b.used;
        ++blocks;
        if (std::find(skip.begin(), skip.end(), static_cast<int>(k)) != skip.end()) continue;
        if (best < 0 || b.used < g_heap[static_cast<std::size_t>(best)].used) best = static_cast<int>(k);
    }
    if (class_free) *class_free = free;
    if (used) *used = best >= 0 ? g_heap[static_cast<std::size_t>(best)].used : 0;
    return blocks >= 2 ? best : -1;  // the last texture block is never drained into nothing
}

int image_heap_target_drain_candidate_locked(std::uint64_t* used, std::uint64_t* room_elsewhere) {
    int best = -1;
    std::size_t blocks = 0;
    std::uint64_t room = 0;
    for (std::size_t k = 0; k < g_heap.size(); ++k) {
        const HeapBlock& b = g_heap[k];
        if (b.memory == VK_NULL_HANDLE || b.cls != kHeapTargets || g.mem_props.memoryTypes[b.type].heapIndex != g.local_heap) continue;
        ++blocks;
        room += kHeapBlock - b.used;
        if (best < 0 || b.used < g_heap[static_cast<std::size_t>(best)].used) best = static_cast<int>(k);
    }
    if (best < 0 || blocks < 2) return -1;
    const HeapBlock& b = g_heap[static_cast<std::size_t>(best)];
    if (used) *used = b.used;
    if (room_elsewhere) *room_elsewhere = room - (kHeapBlock - b.used);
    return best;
}

void image_heap_avoid_block_locked(int block) { g_heap_avoid = block; }
int image_heap_avoided_block_locked() { return g_heap_avoid; }

std::uint64_t image_heap_block_used_locked(int block) {
    if (block < 0 || static_cast<std::size_t>(block) >= g_heap.size() || g_heap[static_cast<std::size_t>(block)].memory == VK_NULL_HANDLE) return 0;
    return g_heap[static_cast<std::size_t>(block)].used;
}

void image_heap_blocks_locked(std::vector<heap_compact::Block>& out, std::uint32_t* type, bool* classes, bool* overflow, bool* fullest) {
    out.assign(g_heap.size(), heap_compact::Block{});
    std::uint32_t t = UINT32_MAX;
    for (std::size_t k = 0; k < g_heap.size(); ++k) {
        const HeapBlock& b = g_heap[k];
        heap_compact::Block& o = out[k];
        o.live = b.memory != VK_NULL_HANDLE && g.mem_props.memoryTypes[b.type].heapIndex == g.local_heap;
        if (!o.live) continue;
        o.cls = b.cls;
        o.used = b.used;
        o.type = b.type;
        o.free = b.free;
        if (t == UINT32_MAX) t = b.type;
    }
    if (type) *type = t == UINT32_MAX ? 0 : t;
    if (classes) *classes = g_heap_classes;
    if (overflow) *overflow = g_heap_overflow;
    if (fullest) *fullest = g_heap_fullest;
}

bool image_memory_place_existing_locked(const VkMemoryRequirements& req, ImageMemory& out, int cls) {
    out = ImageMemory{};
    const std::uint32_t local = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (local == UINT32_MAX || req.size > kHeapLargest) return false;
    return heap_place_locked(req, local, false, out, cls);
}

int image_memory_class_locked(const ImageMemory& m) {
    if (m.block < 0 || static_cast<std::size_t>(m.block) >= g_heap.size()) return -2;
    return g_heap[static_cast<std::size_t>(m.block)].cls;
}

bool image_memory_in_vram(const ImageMemory& m) {
    return m.type >= g.mem_props.memoryTypeCount || g.mem_props.memoryTypes[m.type].heapIndex == g.local_heap;
}

std::int64_t vram_headroom() {
    std::uint64_t limit = 0, used = 0;
    if (bb_memory_test_limited(&limit, &used)) return static_cast<std::int64_t>(limit) - static_cast<std::int64_t>(used);
    const std::uint64_t local = bb_memory_local_bytes();
    if (!g.has_memory_budget || !g.phys) return static_cast<std::int64_t>(g.local_budget) - static_cast<std::int64_t>(local);
    std::lock_guard<std::mutex> lk(g_budget_mu);
    const std::int64_t now = steady_ms();
    if (now - g_budget_read_ms >= 100) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT mb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        mp.pNext = &mb;
        vkGetPhysicalDeviceMemoryProperties2(g.phys, &mp);
        g_budget_room = static_cast<std::int64_t>(mb.heapBudget[g.local_heap]) - static_cast<std::int64_t>(mb.heapUsage[g.local_heap]);
        g_budget_local_at_read = bb_memory_local_bytes();
        g_budget_read_ms = now;
        return g_budget_room;
    }
    return g_budget_room - (static_cast<std::int64_t>(local) - static_cast<std::int64_t>(g_budget_local_at_read));
}

// The reserve: on a discrete GPU a twelfth of the budget at the start, from
// 256 to 512 MiB (512 from about 6 GB free, so an 8 GB card's as before);
// none on an integrated one (its video memory is the same RAM, and a Steam
// Deck's 1 GiB carve-out would be half reserve) or with tight memory, whose
// own settings apply. With 512 a 4 GB card's budget kept 800 MiB free in play
// - the reserve, the mirrors' cushion above it and a block - while half the
// vertex data was read across the bus: 42-44 fps at the first Central Yharnam
// lamp on the RTX 3070 Ti Laptop held to 2,950 MiB (BBHOST_TEST_VRAM_MB),
// 58 with 256 and a 128 MiB cushion. BBHOST_VRAM_RESERVE_MB sets it; 0
// places everything until an allocation fails, as before.
void vram_reserve_init() {
    const char* e = std::getenv("BBHOST_VRAM_RESERVE_MB");
    const bool given = e && *e;
    const std::uint64_t scaled = std::clamp<std::uint64_t>((g.local_budget >> 20) / 12, 256, 512);
    const std::uint64_t mib = given ? std::strtoull(e, nullptr, 10) : (g.integrated || g.tight ? 0 : scaled);
    g_vram_reserve = mib << 20;
    if (g_vram_reserve)
        host_log("gpu: video memory followed against the driver's budget, the last %llu MiB kept for render targets and device buffers%s",
                 static_cast<unsigned long long>(mib), given ? " (BBHOST_VRAM_RESERVE_MB)" : "");
    else
        host_log("gpu: video memory placed until an allocation fails (%s)",
                 given ? "BBHOST_VRAM_RESERVE_MB=0" : g.integrated ? "an integrated GPU" : "tight memory");
    cache_budget_init();
}

// The caches' budget and the lean image heap (host/apu_memory.h). With no
// reserve - an integrated GPU, whose video memory is the same RAM - nothing
// trims the caches in play: the textures made ahead of their first bind (on
// a Radeon 8060S 736 MiB of them never bound), those no bind has looked up
// for 20 s, the image heap's free room and its spare block. Over the budget
// they yield as they do down at a card's reserve: textures are made at their
// first bind rather than ahead, the cold ones go (those made ahead and never
// bound first), the textures' blocks drain at the lean thresholds, empty
// blocks go back at once and no spare block is made in play. What the game
// draws with - targets, textures in use, buffers - is placed as before, over
// the budget or not. BBHOST_CACHE_BUDGET_MB / gpu.cache_budget_mb (MiB, any
// device; 0 none); BBHOST_HEAP_LEAN / gpu.heap_lean the lean heap alone.
namespace {
std::atomic<std::uint64_t> g_cache_budget{0};
std::atomic<bool> g_heap_lean{false};
}  // namespace

void cache_budget_init() {
#if defined(_WIN32)
    constexpr bool kWindowsHost = true;
#else
    constexpr bool kWindowsHost = false;
#endif
    const int integrated = g.integrated ? 1 : 0;
    const std::string budget_cfg = config_value("gpu.cache_budget_mb"), lean_cfg = config_value("gpu.heap_lean");
    const std::uint64_t mib = apu_memory::cache_budget_mib(std::getenv("BBHOST_CACHE_BUDGET_MB"), budget_cfg.c_str(), integrated, kWindowsHost);
    g_cache_budget.store(mib << 20, std::memory_order_relaxed);
    g_heap_lean.store(apu_memory::heap_lean(std::getenv("BBHOST_HEAP_LEAN"), lean_cfg.c_str(), integrated, kWindowsHost),
                      std::memory_order_relaxed);
    if (mib)
        host_log("gpu: the caches yield over %llu MiB of video memory held (BBHOST_CACHE_BUDGET_MB, gpu.cache_budget_mb): textures made at "
                 "their first bind rather than ahead, those not bound for 20 s evicted, the textures' blocks drained at the lean thresholds, "
                 "empty blocks given back at once%s",
                 static_cast<unsigned long long>(mib), g_heap_lean.load() ? "; the image heap lean (BBHOST_HEAP_LEAN)" : "");
    else
        host_log("gpu: the caches have no budget (BBHOST_CACHE_BUDGET_MB, gpu.cache_budget_mb)%s",
                 g_heap_lean.load() ? "; the image heap lean - no spare block in play, the textures' blocks drained at half a block of "
                                      "room (BBHOST_HEAP_LEAN)"
                                    : "");
}

std::uint64_t cache_budget_bytes() { return g_cache_budget.load(std::memory_order_relaxed); }
bool cache_over_budget(std::uint64_t more) {
    return apu_memory::over_budget(g_cache_budget.load(std::memory_order_relaxed), bb_memory_local_bytes(), more);
}

// The world settled: no loading screen, and BBHOST_CACHE_SETTLE_S (30) since
// the last one ended - an area's arrival over, the textures it made ahead
// bound or not. Before the first world load ends, never. What the budget and
// the lean heap take before it is what the area being entered needs: on an
// RTX 4070 they made 3,583 textures at their binds against 610 and turned a
// 100 ms arrival stall into 272 ms, a 4.8 s load into 7.1.
std::atomic<std::uint64_t> g_loading_ended_ms{0};
bool world_settled() {
    static const std::uint64_t settle_ms = [] {
        const char* e = std::getenv("BBHOST_CACHE_SETTLE_S");
        return (e && *e ? std::strtoull(e, nullptr, 10) : 30ull) * 1000;
    }();
    return apu_memory::settled(g_loading_ended_ms.load(std::memory_order_relaxed), static_cast<std::uint64_t>(steady_ms()), loading_screen_now(),
                               settle_ms);
}
bool cache_budget_pressing(std::uint64_t more) { return world_settled() && cache_over_budget(more); }
std::uint64_t cache_budget_excess() { return apu_memory::excess(g_cache_budget.load(std::memory_order_relaxed), bb_memory_local_bytes()); }
bool heap_lean_on() { return g_heap_lean.load(std::memory_order_relaxed); }

// The image heap's compaction (render.cpp compact_image_heap_locked,
// host/heap_compact.h): BBHOST_HEAP_COMPACT=1 on any device, =0 off; unset,
// on where the image heap is lean (an integrated GPU under Windows,
// host/apu_memory.h) and off elsewhere until measured there. Read once the
// device is chosen (the submissions ask, never before).
bool heap_compact_on() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_HEAP_COMPACT");
        if (e && *e) return e[0] != '0';
        return heap_lean_on();
    }();
    return on;
}

std::uint64_t vram_reserve_bytes() { return g_vram_reserve; }

bool vram_within_reserve(std::uint64_t more) {
    return g_vram_reserve && vram_headroom() < static_cast<std::int64_t>(g_vram_reserve + more);
}

std::uint64_t vram_reserve_refusals() { return g_reserve_refusals.load(std::memory_order_relaxed); }

// BBHOST_VRAM_RETURN=0: what a shortage put in host memory stays there for
// the session, as before. On with the reserve.
bool vram_return_on() {
    static const bool env_off = [] {
        const char* e = std::getenv("BBHOST_VRAM_RETURN");
        return e && e[0] == '0';
    }();
    return g_vram_reserve && !env_off;
}

bool vram_room_for(std::uint64_t bytes, std::uint64_t margin) {
    return vram_headroom() >= static_cast<std::int64_t>(g_vram_reserve + margin + bytes);
}

std::string textures_return_report();  // textures.cpp
std::string render_return_report();    // render.cpp
std::string vram_return_report() {
    if (!vram_return_on()) return {};
    return textures_return_report() + render_return_report();
}

bool image_memory_alloc(const VkMemoryRequirements& req, ImageMemory& out, bool texture, int cls) {
    out = ImageMemory{};
    if (cls < 0) cls = texture ? kHeapTextures : kHeapTargets;
    const std::uint32_t local = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const std::int64_t now = steady_ms();
    bool may_grow = now - g_vram_refused_ms.load(std::memory_order_relaxed) >= 1000;
    // The reserve: a texture takes no new video memory out of it - room
    // already in the heap's blocks, else host memory, which it is only read
    // from - so the render targets made after it still find video memory. A
    // target in host memory is written and read across the bus every frame:
    // two at 4K held a 12 GB card's session at 3 fps.
    bool held_back = false;
    if (texture && may_grow && local != UINT32_MAX && vram_within_reserve(req.size <= kHeapLargest ? kHeapBlock : req.size)) {
        may_grow = false;
        held_back = true;
    }
    if (local != UINT32_MAX && heap_place_locked(req, local, may_grow, out, cls)) return true;
    if (held_back) {
        const std::uint32_t host = host_memory_type(req.memoryTypeBits);
        if (host != UINT32_MAX && heap_place_locked(req, host, true, out, cls)) {
            if (g_reserve_refusals.fetch_add(1, std::memory_order_relaxed) == 0)
                host_log("gpu: video memory is down to the reserve (%lld MiB left): textures go to host memory until there is room",
                         static_cast<long long>(vram_headroom() >> 20));
            ++g_heap_host;
            return true;
        }
    }
    // Video memory has no room for it: host memory, which the GPU reads across
    // the bus - slower, but drawn. A target that found no memory was never
    // made, and every pass reading it read nothing: on an 8 GB card whose
    // video memory the buffer shadow's mirrors had taken, the world drew
    // black. The mirrors are given back at the next
    // submission (g.vram_short), so the images after these find video memory.
    const std::uint32_t host = vram_fallback_on() ? host_memory_type(req.memoryTypeBits) : UINT32_MAX;
    if (host == UINT32_MAX) return false;
    if (local != UINT32_MAX && may_grow) {
        g_vram_refused_ms.store(now, std::memory_order_relaxed);
        g.vram_short.store(true, std::memory_order_relaxed);
        static bool said = false;
        if (!said) {
            said = true;
            host_log("gpu: video memory is full (%llu MiB free for this process at the start): images go to host memory until there "
                     "is room - slower, but drawn; the buffer shadow gives its mirrors back",
                     static_cast<unsigned long long>(g.local_budget >> 20));
            const std::string budget = host_gpu_memory_budget_report();
            if (!budget.empty()) host_log("  %s", budget.c_str());
        }
    }
    if (!heap_place_locked(req, host, true, out, cls)) return false;
    ++g_heap_host;
    return true;
}

void defer_destroy_image(VkImage image, const ImageMemory& m) {
    if (m.block < 0) {
        defer_destroy_image(image, m.memory);
        return;
    }
    bump_view_epoch();
    if (image) g.slots[g.slot].dead_images.push_back(image);
    g.slots[g.slot].dead_ranges.push_back(m);
}

// A whole GX texture copied as memory (src_gen / dst_gen, a CopyResource)
// goes by its resources' images (BBHOST_COPY_BY_RESOURCE; it covers
// textures.cpp's copy-image sides and binds by resource too):
// [0] the source's target, its generation's, copied as an image (the
// render-target path); [1] the source's surface a shader or copy wrote,
// copied image to image (textures_copy_resource_locked); [2] a target at the
// source's memory another resource's (a dead one's: the source was made with
// texels of its own), so the memory is copied, not that target; [3] the
// source's memory its content: copied as memory; [4] of those, ones into a
// destination the host draws into as a target, which keeps its pixels (a
// census: not seen so far); [5] targets copied into, relabelled the
// destination's generation; [6] sources a shader or copy wrote, copied as
// their stale memory (level 1).
// 2 (the default): targets go by their resources (relabelled, another
// resource's not used), and a source whose surface a shader or copy wrote is
// copied image to image - the copy carries what the GPU wrote, as the
// console's memory would. 1: such a source is copied as its memory, which
// the GPU's writes never reach (stale), as before. 0: by addresses.
// Level 2 changes no frame where it was measured (2026-10-18): in steady play
// the one such copy is the Hunter's Nightmare's 960x540 half-float texture,
// which the game clears every frame (GX's fill) and a later pass reads as
// cleared, into a copy no shader binds; the G-buffer copies take that path
// only before the G-buffer pass first draws after a load (the title and the
// menu after it: identical frames at either level, outside the text the
// screens animate themselves). Command processor and GPU time within noise.
const int g_copy_by_resource = [] {
    const char* e = std::getenv("BBHOST_COPY_BY_RESOURCE");
    return e && *e ? std::atoi(e) : 2;
}();
std::atomic<std::uint64_t> g_copy_res[7] = {};
bool copy_by_resource_on() { return g_copy_by_resource > 0 && res_identity_on(); }

std::string image_heap_report() {
    VkDeviceSize used = 0;
    for (const HeapBlock& b : g_heap) used += b.used;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "image heap: %zu blocks, %llu MiB in use; %llu textures placed in it, %llu with memory of their own, %llu in host memory "
                  "for want of video memory; blocks made on the command processor %llu, released %llu (made in play %llu, %.1f ms in all, the "
                  "longest %.1f ms); upload staging free %llu MiB; view epoch %llu",
                  heap_live_blocks_locked(), static_cast<unsigned long long>(used >> 20), static_cast<unsigned long long>(g_heap_allocs),
                  static_cast<unsigned long long>(g_heap_own), static_cast<unsigned long long>(g_heap_host),
                  static_cast<unsigned long long>(g_heap_cp_blocks), static_cast<unsigned long long>(g_heap_released),
                  static_cast<unsigned long long>(g_heap_cp_play), g_heap_cp_play_us / 1000.0, g_heap_cp_play_max_us / 1000.0,
                  static_cast<unsigned long long>(g.staging_free_bytes >> 20), static_cast<unsigned long long>(view_epoch()));
    if (const std::string sites = view_epoch_sites_report(); !sites.empty()) host_log("  %s", sites.c_str());
    host_log("  %s", image_reuse_report_locked().c_str());
    host_log("  %s", draw_gen_report_locked().c_str());
    host_log("  whole textures copied as memory, by their resources%s: the source's target as an image %llu, its surface image to image %llu "
             "(GPU-written ones copied as their stale memory, BBHOST_COPY_BY_RESOURCE=1, %llu), another resource's target at the "
             "source - its memory copied %llu, its memory its content %llu (into a destination drawn as a target, which keeps its pixels, %llu), "
             "targets copied into made the destination's %llu; image to image: %s",
             g_copy_by_resource ? "" : " (BBHOST_COPY_BY_RESOURCE=0: by address)", static_cast<unsigned long long>(g_copy_res[0].load()),
             static_cast<unsigned long long>(g_copy_res[1].load()), static_cast<unsigned long long>(g_copy_res[6].load()),
             static_cast<unsigned long long>(g_copy_res[2].load()), static_cast<unsigned long long>(g_copy_res[3].load()),
             static_cast<unsigned long long>(g_copy_res[4].load()), static_cast<unsigned long long>(g_copy_res[5].load()),
             textures_resource_copy_report().c_str());
    {
        VkDeviceSize free_vram = 0, free_host = 0;
        for (const HeapBlock& b : g_heap) {
            if (b.memory == VK_NULL_HANDLE) continue;
            (g.mem_props.memoryTypes[b.type].heapIndex == g.local_heap ? free_vram : free_host) += kHeapBlock - b.used;
        }
        host_log("  video memory: headroom %lld MiB, reserve %llu MiB; image heap free %llu MiB in video memory, %llu MiB in host "
                 "blocks; textures kept out of the reserve %llu, untile scratch chunks %llu%s",
                 static_cast<long long>(vram_headroom() >> 20), static_cast<unsigned long long>(g_vram_reserve >> 20),
                 static_cast<unsigned long long>(free_vram >> 20), static_cast<unsigned long long>(free_host >> 20),
                 static_cast<unsigned long long>(g_reserve_refusals.load()), static_cast<unsigned long long>(g_scratch_refusals.load()),
                 vram_return_report().c_str());
    }
    return buf;
}

// The video memory report: where the device-local memory this process holds
// goes, by owner and kind, and how much of it is what the run must hold
// against what caches took because the card had room - the mirrors, textures
// no bind has looked up for a while, free room in the image heap's blocks -
// which give it back when video memory runs short. Every 1,800 flips
// (BBHOST_VRAM_REPORT_FLIPS) and at exit, after the "memory by heap" line.
// BBHOST_VRAM_REPORT=0: not made.
namespace {
std::atomic<std::uint64_t> g_spare_play_blocks{1};  // the memory keeper's spare in play (start_memory_reserve)
std::uint64_t g_vram_peak_usage = 0;                // the most the driver counted in the video memory heap at a report
}  // namespace

std::uint64_t image_heap_spare_in_play() { return g_spare_play_blocks.load(std::memory_order_relaxed); }

bool vram_report_on() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_VRAM_REPORT");
        return !(e && e[0] == '0');
    }();
    return on;
}

std::atomic<std::uint64_t> g_glance_cache{0};  // video memory the last census found to be cache
std::atomic<bool> g_glance_census{false};
std::string video_memory_report_locked(bool at_exit) {
    using ull = unsigned long long;
    const auto mib = [](std::uint64_t b) { return static_cast<ull>((b + (1ull << 19)) >> 20); };
    const auto smib = [](std::int64_t b) { return static_cast<long long>(b / (1ll << 20)); };
    const auto t0 = std::chrono::steady_clock::now();
    std::string out;
    char buf[640];
    // The driver's count (VK_EXT_memory_budget) for every device-local heap.
    VkPhysicalDeviceMemoryBudgetPropertiesEXT mb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    const bool have = g.has_memory_budget && g.phys;
    if (have) {
        mp.pNext = &mb;
        vkGetPhysicalDeviceMemoryProperties2(g.phys, &mp);
    }
    const std::vector<MemOwnerTotal> owners = bb_memory_owner_totals();
    std::uint64_t ours[3] = {};
    for (const MemOwnerTotal& o : owners) {
        for (int c = 0; c < 3; ++c) ours[c] += o.bytes[c];
    }
    std::uint64_t local_peak = 0, device_peak = 0;
    bb_memory_peaks(&local_peak, &device_peak);
    const std::uint64_t usage = have ? mb.heapUsage[g.local_heap] : ours[0];
    const std::int64_t driver_own = static_cast<std::int64_t>(usage) - static_cast<std::int64_t>(ours[0]);
    g_vram_peak_usage = std::max(g_vram_peak_usage, usage);

    // The images by kind (render.cpp, textures.cpp) and the image heap's
    // blocks in video memory.
    VramCensus c;
    unsigned dw = 0, dh = 0;
    if (!live_resolution_display_size(&dw, &dh)) dw = dh = 0;
    {
        // At the display buffers' own size the scene's targets have it too:
        // no line between them then.
        unsigned rw = 0, rh = 0;
        live_resolution_current(&rw, &rh);
        if (rw == dw && rh == dh) dw = dh = 0;
    }
    render_vram_census_locked(c, dw, dh);
    textures_vram_census_locked(c);
    VkDeviceSize used_vram = 0, free_vram = 0;
    std::size_t blocks_vram = 0;
    VkDeviceSize cls_used[3] = {}, cls_free[3] = {};  // targets, textures, none yet
    std::size_t cls_blocks[3] = {};
    for (const HeapBlock& b : g_heap) {
        if (b.memory == VK_NULL_HANDLE || g.mem_props.memoryTypes[b.type].heapIndex != g.local_heap) continue;
        used_vram += b.used;
        free_vram += kHeapBlock - b.used;
        ++blocks_vram;
        const int k = b.cls == kHeapTargets ? 0 : b.cls == kHeapTextures ? 1 : 2;
        cls_used[k] += b.used;
        cls_free[k] += kHeapBlock - b.used;
        ++cls_blocks[k];
    }
    ShadowCensus sc;
    shadow_vram_census_locked(sc);
    // Device buffers and anything else with memory of its own, by owner; the
    // image heap's blocks and images of their own are broken down by kind
    // above, the mirrors are a cache.
    std::uint64_t own_images = 0, devbuf_bytes = 0, mirror_bytes = 0;
    std::string devbufs;
    for (const MemOwnerTotal& o : owners) {
        if (!o.bytes[0]) continue;
        if (o.name == "images of their own") {
            own_images += o.bytes[0];
            continue;
        }
        if (o.name == "image heap blocks") continue;
        if (o.name == "mirrors" || o.name == "mirror blocks" || o.name == "mirror windows") {
            mirror_bytes += o.bytes[0];
            continue;
        }
        devbuf_bytes += o.bytes[0];
        if (o.bytes[0] < (1ull << 20)) continue;
        std::snprintf(buf, sizeof(buf), "%s%s %llu", devbufs.empty() ? "" : ", ", o.name.c_str(), mib(o.bytes[0]));
        devbufs += buf;
    }
    const std::uint64_t scratch_kept = std::min<std::uint64_t>(g.scratch_free_bytes, devbuf_bytes);
    const std::uint64_t targets = c.rt_colour.bytes + c.rt_depth.bytes + c.rt_display.bytes;
    const std::uint64_t gpu_written = c.gpu_written.bytes + c.gpu_written_idle.bytes;
    const std::uint64_t images = targets + c.snapshots.bytes + gpu_written + c.regions.bytes + c.tex_bound.bytes + c.tex_cold.bytes +
                                 c.tex_ahead.bytes;
    // Image heap bytes in use that no container holds: ranges of images
    // destroyed in submissions still in flight, which go when they retire.
    const std::int64_t unattributed = static_cast<std::int64_t>(used_vram + own_images) - static_cast<std::int64_t>(images);
    const std::uint64_t held = targets + c.snapshots.bytes + gpu_written + c.regions.bytes + c.tex_bound.bytes + (devbuf_bytes - scratch_kept);
    const std::uint64_t cached = mirror_bytes + c.tex_cold.bytes + c.tex_ahead.bytes + free_vram + scratch_kept;
    g_glance_cache.store(cached, std::memory_order_relaxed);  // the memory counter's (host_gpu_memory_glance)
    g_glance_census.store(true, std::memory_order_relaxed);

    std::snprintf(buf, sizeof(buf),
                  "gpu: video memory (MiB): heap %u %s %llu of %llu = ours %llu (held %llu, caches %llu, in flight %lld) + the driver's own %lld; "
                  "ours at most %llu, the driver's count at most %llu at a report",
                  g.local_heap, have ? "the driver counts" : "ours", mib(usage), mib(have ? mb.heapBudget[g.local_heap] : g.local_budget),
                  mib(ours[0]), mib(held), mib(cached), smib(unattributed), smib(driver_own), mib(local_peak), mib(g_vram_peak_usage));
    out += buf;
    if (have) {
        for (std::uint32_t h = 0; h < g.mem_props.memoryHeapCount; ++h) {
            if (h == g.local_heap || !(g.mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
            std::snprintf(buf, sizeof(buf), "; device-local heap %u (%llu) %llu of %llu", h, mib(g.mem_props.memoryHeaps[h].size),
                          mib(mb.heapUsage[h]), mib(mb.heapBudget[h]));
            out += buf;
        }
    }
    if (ours[1]) {
        std::string bar;
        for (const MemOwnerTotal& o : owners) {
            if (o.bytes[1] < (1ull << 20)) continue;
            std::snprintf(buf, sizeof(buf), "%s%s %llu", bar.empty() ? "" : ", ", o.name.c_str(), mib(o.bytes[1]));
            bar += buf;
        }
        std::snprintf(buf, sizeof(buf), ", ours there %llu (%s)", mib(ours[1]), bar.c_str());
        out += buf;
    }
    {
        std::string host;
        for (const MemOwnerTotal& o : owners) {
            if (o.bytes[2] < (16ull << 20)) continue;
            std::snprintf(buf, sizeof(buf), "%s%s %llu", host.empty() ? "" : ", ", o.name.c_str(), mib(o.bytes[2]));
            host += buf;
        }
        std::snprintf(buf, sizeof(buf), "; host memory the GPU reads, which Windows counts as shared GPU memory: ours %llu (%s)", mib(ours[2]),
                      host.c_str());
        out += buf;
    }
    std::snprintf(buf, sizeof(buf),
                  "\n[bbhost]   video memory held %llu: targets %llu (colour %llu in %llu, depth %llu in %llu, the display buffers' %ux%u %llu in "
                  "%llu; %llu MiB of them not written for 10 s), snapshots %llu (%llu), surfaces a shader or a copy wrote %llu (%llu; %llu "
                  "MiB of them not written for 10 s), copies of targets' regions and levels %llu (%llu), textures bound in the last 20 s %llu "
                  "(%llu)",
                  mib(held), mib(targets), mib(c.rt_colour.bytes), static_cast<ull>(c.rt_colour.n), mib(c.rt_depth.bytes),
                  static_cast<ull>(c.rt_depth.n), dw, dh, mib(c.rt_display.bytes), static_cast<ull>(c.rt_display.n), mib(c.rt_idle.bytes),
                  mib(c.snapshots.bytes), static_cast<ull>(c.snapshots.n), mib(gpu_written), static_cast<ull>(c.gpu_written.n + c.gpu_written_idle.n),
                  mib(c.gpu_written_idle.bytes), mib(c.regions.bytes), static_cast<ull>(c.regions.n), mib(c.tex_bound.bytes),
                  static_cast<ull>(c.tex_bound.n));
    out += buf;
    std::snprintf(buf, sizeof(buf), "; device buffers %llu without the untile scratch kept (%s)", mib(devbuf_bytes - scratch_kept),
                  devbufs.c_str());
    out += buf;
    std::snprintf(buf, sizeof(buf),
                  "\n[bbhost]   video memory caches %llu (given back when it runs short): mirrors %llu (%u whole of %llu, %u blocks or windows; %llu "
                  "hold page copies, in %u windows of 64; %s), textures not bound for 20 s %llu (%llu) and made ahead, never bound %llu (%llu) - the cold eviction "
                  "takes them only %s; image heap room %llu in %zu blocks (in use %llu; spare %llu block(s) in play), untile "
                  "scratch kept %llu",
                  mib(cached), mib(mirror_bytes), sc.whole, mib(kChunkBytes), sc.blocks, mib(sc.data_bytes), sc.windows,
                  !buffer_shadow_on() ? "the buffer shadow is off"
                  : sc.given_back     ? "given back"
                  : sc.live           ? "the share follows the budget"
                                      : "a fixed share",
                  mib(c.tex_cold.bytes), static_cast<ull>(c.tex_cold.n), mib(c.tex_ahead.bytes), static_cast<ull>(c.tex_ahead.n),
                  cache_budget_bytes() ? "down at the reserve or over the caches' budget" : "down at the reserve", mib(free_vram),
                  blocks_vram, mib(used_vram), static_cast<ull>(g_spare_play_blocks.load(std::memory_order_relaxed)), mib(scratch_kept));
    out += buf;
    if (g_heap_classes) {
        std::uint64_t drained[4] = {};
        textures_drain_counts(drained);
        std::snprintf(buf, sizeof(buf),
                      "\n[bbhost]   video memory, the image heap's blocks by what they hold (BBHOST_HEAP_CLASSES): targets %zu (%llu in use, %llu free), textures %zu "
                      "(%llu in use, %llu free), empty %zu; GPU-written surfaces in the textures' blocks %llu (%llu), uploaded textures in the "
                      "targets' %llu (%llu); drained so far %llu block(s), %llu textures (%llu MiB), %llu left for a GPU-written surface",
                      cls_blocks[0], mib(cls_used[0]), mib(cls_free[0]), cls_blocks[1], mib(cls_used[1]), mib(cls_free[1]), cls_blocks[2],
                      mib(c.gpu_written_in_tex.bytes), static_cast<ull>(c.gpu_written_in_tex.n), mib(c.tex_in_targets.bytes),
                      static_cast<ull>(c.tex_in_targets.n), static_cast<ull>(drained[0]), static_cast<ull>(drained[1]), mib(drained[2]),
                      static_cast<ull>(drained[3]));
        out += buf;
        out += target_drain_report();
        out += heap_compact_report();
        out += textures_relocate_report();
        // The targets' blocks one by one: what each holds (MiB of colour,
        // depth, snapshots, GPU-written surfaces, textures, region copies)
        // and its free room - all of it, the largest hole, the holes.
        std::string tb;
        for (std::size_t k = 0; k < g_heap.size(); ++k) {
            const HeapBlock& b = g_heap[k];
            if (b.memory == VK_NULL_HANDLE || b.cls != kHeapTargets || g.mem_props.memoryTypes[b.type].heapIndex != g.local_heap) continue;
            VkDeviceSize largest = 0;
            for (const auto& [o, sz] : b.free) largest = std::max(largest, sz);
            std::array<std::uint64_t, VramCensus::kBlockKinds> by{};
            if (k < c.by_block.size()) by = c.by_block[k];
            char one[160];
            std::snprintf(one, sizeof(one), "%s#%zu %llu/%llu/%llu/%llu/%llu/%llu free %llu (largest %llu, %zu holes)", tb.empty() ? "" : "; ", k,
                          mib(by[0]), mib(by[1]), mib(by[2]), mib(by[3]), mib(by[4]), mib(by[5]), mib(kHeapBlock - b.used), mib(largest),
                          b.free.size());
            tb += one;
        }
        if (!tb.empty()) {
            out += "\n[bbhost]   video memory, the targets' blocks (MiB of colour/depth/snapshots/GPU-written/textures/region copies): ";
            out += tb;
        }
    }
    if (c.rt_host.n || c.tex_host.n) {
        std::snprintf(buf, sizeof(buf), "; in host memory for want of video memory: targets and snapshots %llu (%llu), textures %llu (%llu)",
                      mib(c.rt_host.bytes), static_cast<ull>(c.rt_host.n), mib(c.tex_host.bytes), static_cast<ull>(c.tex_host.n));
        out += buf;
    }
    if (at_exit) {
        // The owners that held the most at once over the run (video memory).
        std::vector<const MemOwnerTotal*> by_peak;
        for (const MemOwnerTotal& o : owners) {
            if (o.peak >= (1ull << 20)) by_peak.push_back(&o);
        }
        std::sort(by_peak.begin(), by_peak.end(), [](const MemOwnerTotal* x, const MemOwnerTotal* y) { return x->peak > y->peak; });
        std::string peaks;
        for (std::size_t i = 0; i < by_peak.size() && i < 16; ++i) {
            std::snprintf(buf, sizeof(buf), "%s%s %llu", peaks.empty() ? "" : ", ", by_peak[i]->name.c_str(), mib(by_peak[i]->peak));
            peaks += buf;
        }
        std::snprintf(buf, sizeof(buf), "\n[bbhost]   video memory at most, by owner (MiB, each at its own highest): %s; every device-local heap "
                      "together at most %llu", peaks.c_str(), mib(device_peak));
        out += buf;
    }
    // The window imports' idle census (BBHOST_IMPORT_IDLE_CENSUS=1).
    if (const std::string idle = import_idle_report_locked(); !idle.empty()) out += "\n[bbhost]   " + idle;
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    std::snprintf(buf, sizeof(buf), " (the census took %lld us)", static_cast<long long>(us));
    out += buf;
    return out;
}

// Memory made ahead, off the command processor, and given back when idle.
// Upload staging is pinned host memory, which the driver pins page by page
// (~30 ms a 64 MiB chunk), and an image-heap block is 5-10 ms of VRAM; the
// frame the world first draws needs ~12 staging chunks in flight and ~5 heap
// blocks, and making them there was ~140 ms of that frame. That used to be
// paid by making 768 MiB of staging and six
// heap blocks at boot and keeping them for good - 1.5 GiB of VRAM the title
// used a fifth of, and pinned RAM Windows counts as GPU memory. The keeper
// holds less while the game plays and more while a loading screen is up
// (engine/loading.cpp tells it, host_gpu_set_loading), when the coming area
// will want it:
//   image heap spare  BBHOST_IMAGE_HEAP_SPARE blocks of free room (1; 0 with
//                     tight memory) in play, BBHOST_IMAGE_HEAP_SPARE_LOADING
//                     (4; 1) during a load; an empty block above that goes
//                     back after 10 s in play
//   upload staging    BBHOST_STAGING_FLOOR_MB (256; 128 on an integrated GPU
//                     under Windows, host/apu_memory.h) made at the start;
//                     BBHOST_STAGING_RESERVE_MB (768; 256) made ready during
//                     a load and kept in play for the next one, what is over
//                     it given back while a loading screen is up
// Staging stays in play because giving it back stalls the frames: each 64 MiB
// chunk's vkFreeMemory took 22-27 ms on the RTX 4070, and the frame it fell
// in ran 29-47 ms against 14-17 around it, one every two seconds for the 16 s
// after every load - after each death, lamp and area change (the frees of
// image heap blocks, 7-9 ms, show no such frame). The staging made again at
// the next load cost ~30 ms a chunk of pinning besides. BBHOST_STAGING_RETURN=1
// (or gpu.staging_return = true; host/apu_memory.h) gives it back in play as
// before: down to the floor once the world has settled (30 s after the latest
// load, a death restarting the count), a chunk every two seconds, each free
// timed for the exit report and logged with its flip.
// BBHOST_MEMORY_KEEPER=0 restores the old fixed reserve (six blocks and
// 768 MiB at boot, never given back) for comparison.
std::atomic<bool> g_loading_screen{false};
bool loading_screen_now() { return g_loading_screen.load(std::memory_order_relaxed); }
std::mutex g_keeper_mu;
std::condition_variable g_keeper_cv;
void keeper_wake() { g_keeper_cv.notify_one(); }

void start_memory_reserve() {
    const auto env = [](const char* name, std::uint64_t fallback) {
        const char* e = std::getenv(name);
        return e && *e ? static_cast<std::uint64_t>(std::strtoull(e, nullptr, 10)) : fallback;
    };
#if defined(_WIN32)
    constexpr bool kWindowsHost = true;
#else
    constexpr bool kWindowsHost = false;
#endif
    // With tight memory (memory_tight) only what is needed: the spares are
    // what a Steam Deck's world ran out of.
    const std::uint64_t reserve = env("BBHOST_STAGING_RESERVE_MB", g.tight ? 256 : 768) << 20;
    const std::uint64_t floor =
        std::min(apu_memory::staging_floor_mib(std::getenv("BBHOST_STAGING_FLOOR_MB"), g.integrated ? 1 : 0, kWindowsHost) << 20, reserve);
    // Small video memory (small_vram) too: a spare block is 256 MiB of it
    // the targets cannot have.
    const bool few = g.tight || g.small_vram;
    // A lean image heap (BBHOST_HEAP_LEAN; host/apu_memory.h) keeps none in
    // play either, once the world has settled (world_settled): until then the
    // area being entered may want the block.
    const std::uint64_t spare_play_full = apu_memory::spare_play_blocks(std::getenv("BBHOST_IMAGE_HEAP_SPARE"), few, false);
    const std::uint64_t spare_play = apu_memory::spare_play_blocks(std::getenv("BBHOST_IMAGE_HEAP_SPARE"), few, heap_lean_on());
    const std::uint64_t spare_load = std::max(env("BBHOST_IMAGE_HEAP_SPARE_LOADING", few ? 1 : 4), spare_play);
    g_spare_play_blocks.store(spare_play, std::memory_order_relaxed);
    const bool keeper = env("BBHOST_MEMORY_KEEPER", 1) != 0;
    // BBHOST_STAGING_RETURN, else gpu.staging_return, else the device's default (host/apu_memory.h).
    const std::string staging_return_cfg = config_value("gpu.staging_return");
    const bool staging_return =
        apu_memory::staging_return(std::getenv("BBHOST_STAGING_RETURN"), staging_return_cfg.c_str(), g.integrated ? 1 : 0, kWindowsHost);
    const bool scratch_return = env("BBHOST_SCRATCH_RETURN", 1) != 0;
    std::thread([=] {
        host_thread_set_name("bb-reserve");
        // The heap's blocks are of the memory type its first image asked for.
        std::uint32_t type = UINT32_MAX;
        const auto make_staging = [&](std::uint64_t upto) {
            std::uint64_t made = 0;
            for (;;) {
                {
                    std::lock_guard<GpuMutex> lk(g.mu);
                    if (g.staging_free_bytes + staging_chunk_bytes() > std::min(upto, kStagingKeepBytes)) break;
                }
                DevBuffer chunk;
                MemOwner own_mem("upload staging");
                if (!create_dev_buffer(chunk, staging_chunk_bytes(), true)) break;
                std::lock_guard<GpuMutex> lk(g.mu);
                g.staging_free.push_back(chunk);
                g.staging_free_bytes += chunk.size;
                g_staging_created.fetch_add(1, std::memory_order_relaxed);
                made += chunk.size;
            }
            return made;
        };
        const auto make_blocks = [&](std::uint64_t spare_blocks) {
            std::size_t added = 0;
            for (;;) {
                {
                    std::lock_guard<GpuMutex> lk(g.mu);
                    if (type == UINT32_MAX || heap_spare_room_locked(type) >= spare_blocks * kHeapBlock) break;
                }
                if (steady_ms() - g_vram_refused_ms.load(std::memory_order_relaxed) < 10000) break;  // video memory is full
                // No spare out of the reserve, during a load either: the
                // coming area's targets want that room more. None in play
                // over the caches' budget (a load's spares are still made:
                // the area's first frame would make them itself).
                if (vram_within_reserve(kHeapBlock)) break;
                if (!g_loading_screen.load(std::memory_order_relaxed) && cache_budget_pressing(kHeapBlock)) break;
                VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                mai.allocationSize = kHeapBlock;
                mai.memoryTypeIndex = type;
                VkDeviceMemory mem = VK_NULL_HANDLE;
                MemOwner own_mem("image heap blocks");
                if (vkAllocateMemory(g.device, &mai, nullptr, &mem) != VK_SUCCESS) {
                    g_vram_refused_ms.store(steady_ms(), std::memory_order_relaxed);
                    break;
                }
                std::lock_guard<GpuMutex> lk(g.mu);
                heap_add_block_locked(mem, type);
                ++added;
            }
            return added;
        };
        const auto find_type = [&] {
            std::lock_guard<GpuMutex> lk(g.mu);
            for (const HeapBlock& b : g_heap) {
                if (b.memory != VK_NULL_HANDLE && g.mem_props.memoryTypes[b.type].heapIndex == g.local_heap) return b.type;
            }
            return UINT32_MAX;
        };
        if (!keeper) {
            // The fixed reserve, as before: staging and six blocks once, kept.
            const auto t0 = std::chrono::steady_clock::now();
            const std::uint64_t made = make_staging(reserve);
            for (int i = 0; i < 1200 && type == UINT32_MAX; ++i) {
                type = find_type();
                if (type == UINT32_MAX) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::size_t added = 0;
            for (;;) {
                {
                    std::lock_guard<GpuMutex> lk(g.mu);
                    if (type == UINT32_MAX || heap_live_blocks_locked() >= 6) break;
                }
                added += make_blocks(heap_live_blocks_locked() + 1);
                if (!added) break;
            }
            host_log("gpu: memory made ahead in %.0f ms: %llu MiB of upload staging, %zu image heap blocks (BBHOST_MEMORY_KEEPER=0)",
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                     static_cast<unsigned long long>(made >> 20), added);
            return;
        }
        host_log("gpu: memory keeper: image heap spare %llu block(s) in play, %llu during loads; upload staging %llu MiB at the start, "
                 "%llu MiB from a load on%s",
                 static_cast<unsigned long long>(spare_play), static_cast<unsigned long long>(spare_load),
                 static_cast<unsigned long long>(floor >> 20), static_cast<unsigned long long>(reserve >> 20),
                 staging_return ? ", given back to the start's in play (BBHOST_STAGING_RETURN, gpu.staging_return)"
                                : ", kept in play (BBHOST_STAGING_RETURN=1 or gpu.staging_return = true gives it back)");
        make_staging(floor);
        shadow_keeper_started();
        // One piece of memory goes back at a time, two seconds apart: freeing
        // half a gigabyte of upload staging in one go (eight vkFreeMemory of
        // pinned, mapped memory) stalled the frames behind it ~100 ms in every
        // soak, the command processor's next driver calls waiting on it.
        auto last_given = std::chrono::steady_clock::time_point{};
        // Upload staging over the floor goes back a chunk at a time, one 64
        // MiB chunk every two seconds, a smaller one (BBHOST_STAGING_CHUNK_MB)
        // as much sooner: each vkFreeMemory is a stall in the driver that the
        // frame behind it can meet, and a smaller chunk's is shorter.
        std::uint64_t last_staging_given_ms = 0;
        const std::uint64_t staging_pace_ms = std::max<std::uint64_t>(250, 2000 * (staging_chunk_bytes() >> 20) / 64);
        bool was_loading = false;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(g_keeper_mu);
                g_keeper_cv.wait_for(lk, std::chrono::milliseconds(250));
            }
            shadow_keeper_work();  // the buffer shadow's mirrors, windows and spare block
            if (type == UINT32_MAX) type = find_type();
            const bool loading = g_loading_screen.load(std::memory_order_relaxed);
            const auto now = std::chrono::steady_clock::now();
            if (loading != was_loading) {
                was_loading = loading;
                if (loading) {
                    const auto t0 = now;
                    const std::uint64_t made = make_staging(reserve);
                    const std::size_t added = make_blocks(spare_load);
                    if (made || added) {
                        host_log("gpu: memory keeper: a load - %llu MiB of upload staging and %zu image heap block(s) made ready in %.0f ms",
                                 static_cast<unsigned long long>(made >> 20), added,
                                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                    }
                }
            }
            // The lean heap's spare: none once the world has settled, the full one until then.
            const std::uint64_t spare_now = world_settled() ? spare_play : spare_play_full;
            g_spare_play_blocks.store(loading ? spare_load : spare_now, std::memory_order_relaxed);
            make_blocks(loading ? spare_load : spare_now);
            if (loading) {
                // Staging over the reserve (chunks the command processor made
                // in a burst, or one a large upload had to itself) goes back
                // while the loading screen hides the stall, a chunk a turn.
                DevBuffer extra;
                {
                    std::lock_guard<GpuMutex> lk(g.mu);
                    if (!staging_return && g.staging_free_bytes > reserve && !g.staging_free.empty()) {
                        extra = g.staging_free.back();
                        g.staging_free_bytes -= extra.size;
                        g.staging_free.pop_back();
                    }
                }
                if (extra.buffer) {
                    const auto t0 = std::chrono::steady_clock::now();
                    vkDestroyBuffer(g.device, extra.buffer, nullptr);
                    vkFreeMemory(g.device, extra.memory, nullptr);
                    host_log("gpu: memory keeper: gave back %llu MiB of upload staging over the load's reserve in %.1f ms, during the load",
                             static_cast<unsigned long long>(extra.size >> 20),
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                }
                continue;
            }
            // In play: empty blocks above the spare go back after 10 s, one a
            // turn, two seconds apart (with BBHOST_STAGING_RETURN=1 also
            // staging over the floor once the world has settled - 30 s after
            // the latest load - down to it, a chunk a turn).
            std::vector<VkDeviceMemory> gone;
            std::vector<DevBuffer> chunks;
            DevBuffer scratch_gone;
            // Down to the reserve, an empty block goes back at once, spare or
            // not: the targets want the room more.
            // Over the caches' budget the same for the image heap's empty
            // blocks; upload staging and untile scratch still go back two
            // seconds apart (their frees are what stalls the frames).
            const bool reserve_short = vram_within_reserve();
            const bool short_now = reserve_short || cache_budget_pressing();
            const bool may_give = short_now || now - last_given >= std::chrono::seconds(2);
            const bool may_give_paced = reserve_short || now - last_given >= std::chrono::seconds(2);
            {
                std::lock_guard<GpuMutex> lk(g.mu);
                if (type != UINT32_MAX && may_give) {
                    const VkDeviceSize free = heap_spare_room_locked(type);
                    for (HeapBlock& b : g_heap) {
                        if (b.memory == VK_NULL_HANDLE || b.used != 0) continue;
                        if (!short_now && now - b.empty_since < std::chrono::seconds(10)) continue;
                        // A block of host memory, made while video memory was
                        // full, goes back once empty; one of video memory only
                        // above the spare.
                        if (b.type == type && !short_now && free < (spare_now + 1) * kHeapBlock) continue;
                        gone.push_back(b.memory);
                        b.memory = VK_NULL_HANDLE;
                        b.free.clear();
                        ++g_heap_released;
                        break;
                    }
                }
                // Untile scratch over one chunk, kept for a load's burst of
                // uploads, goes back in play once no upload has taken a chunk
                // for 10 s, a chunk a turn (BBHOST_SCRATCH_RETURN=0 keeps up
                // to 256 MiB, as before).
                if (scratch_return && gone.empty() && may_give_paced && g.scratch_free_bytes > kScratchChunkBytes && !g.scratch_free.empty() &&
                    steady_ms() - g_scratch_used_ms.load(std::memory_order_relaxed) >= 10000) {
                    scratch_gone = g.scratch_free.back();
                    g.scratch_free_bytes -= scratch_gone.size;
                    g.scratch_free.pop_back();
                    g_scratch_returned.fetch_add(1, std::memory_order_relaxed);
                }
                if (apu_memory::staging_give_now(staging_return, g.staging_free_bytes, floor, world_settled(), reserve_short,
                                                 static_cast<std::uint64_t>(steady_ms()), last_staging_given_ms, staging_pace_ms) &&
                    gone.empty() && !g.staging_free.empty()) {
                    chunks.push_back(g.staging_free.back());
                    g.staging_free_bytes -= g.staging_free.back().size;
                    g.staging_free.pop_back();
                }
            }
            if (scratch_gone.buffer) {
                vkDestroyBuffer(g.device, scratch_gone.buffer, nullptr);
                vkFreeMemory(g.device, scratch_gone.memory, nullptr);
                last_given = std::chrono::steady_clock::now();
                static int said = 0;
                if (said++ < 4)
                    host_log("gpu: memory keeper: gave back %llu MiB of untile scratch no upload had used for 10 s (%llu MiB kept)",
                             static_cast<unsigned long long>(scratch_gone.size >> 20), static_cast<unsigned long long>(g.scratch_free_bytes >> 20));
            }
            if (!gone.empty() || !chunks.empty()) {
                const auto t0 = std::chrono::steady_clock::now();
                for (VkDeviceMemory m : gone) vkFreeMemory(g.device, m, nullptr);
                // Each step of a staging chunk's free timed apart (where the
                // driver's stall is): its buffer, its mapping, its memory.
                double ms_buffer = 0, ms_unmap = 0, ms_memory = 0;
                std::uint64_t chunk_bytes = 0;
                for (DevBuffer& c : chunks) {
                    chunk_bytes += c.size;
                    const auto a = std::chrono::steady_clock::now();
                    vkDestroyBuffer(g.device, c.buffer, nullptr);
                    const auto b = std::chrono::steady_clock::now();
                    if (c.map) vkUnmapMemory(g.device, c.memory);
                    const auto d = std::chrono::steady_clock::now();
                    vkFreeMemory(g.device, c.memory, nullptr);
                    const auto e = std::chrono::steady_clock::now();
                    ms_buffer += std::chrono::duration<double, std::milli>(b - a).count();
                    ms_unmap += std::chrono::duration<double, std::milli>(d - b).count();
                    ms_memory += std::chrono::duration<double, std::milli>(e - d).count();
                }
                last_given = std::chrono::steady_clock::now();
                if (!chunks.empty()) last_staging_given_ms = static_cast<std::uint64_t>(steady_ms());
                if (!chunks.empty()) {
                    // Counted for the exit report (staging_pool_stats): what
                    // the frees cost is the question giving it back answers.
                    const std::uint64_t us = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(last_given - t0).count());
                    g_staging_returned.fetch_add(chunks.size(), std::memory_order_relaxed);
                    g_staging_return_us.fetch_add(us, std::memory_order_relaxed);
                    std::uint64_t max = g_staging_return_us_max.load(std::memory_order_relaxed);
                    while (us > max && !g_staging_return_us_max.compare_exchange_weak(max, us, std::memory_order_relaxed)) {
                    }
                }
                std::size_t live;
                {
                    std::lock_guard<GpuMutex> lk(g.mu);
                    live = heap_live_blocks_locked();
                }
                char what[96];
                if (gone.empty())
                    std::snprintf(what, sizeof(what), "%llu MiB of upload staging (its buffer %.1f, mapping %.1f, memory %.1f ms)",
                                  static_cast<unsigned long long>(chunk_bytes >> 20), ms_buffer, ms_unmap, ms_memory);
                else
                    std::snprintf(what, sizeof(what), "an image heap block");
                host_log("gpu: memory keeper: gave back %s in %.1f ms at flip %llu (%zu blocks live, %llu MiB of staging kept)", what,
                         std::chrono::duration<double, std::milli>(last_given - t0).count(),
                         static_cast<unsigned long long>(hle_video_flip_count()), live, static_cast<unsigned long long>(g.staging_free_bytes >> 20));
            }
        }
    }).detach();
}

void defer_destroy_image(VkImage image, VkDeviceMemory memory) {
    bump_view_epoch();
    if (image) g.slots[g.slot].dead_images.push_back(image);
    if (memory) g.slots[g.slot].dead_memory.push_back(memory);
}

void defer_destroy_view(VkImageView view) {
    bump_view_epoch();
    if (!view) return;
    set_cache_forget_view_locked(view);
    forget_view_locked(view);
    bindless_view_retired_locked(view);
    g.slots[g.slot].dead_views.push_back(view);
}

// A view no memo, cached set or bindless slot ever held - one a pass makes for
// itself - goes without moving the view epoch. The depth snapshot's two views
// a frame went through defer_destroy_view, and the two bumps emptied both
// view memos every frame: on a Steam Deck building views was 9.5% of the
// command processor, the words memo hit 47%.
void defer_destroy_private_view(VkImageView view) {
    if (view) g.slots[g.slot].dead_views.push_back(view);
}

namespace {
struct CopyBinding {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, range = 0;
    std::uint64_t guest = 0;
};
CopyBinding g_copy_bindings[64];  // under g.mu
std::size_t g_copy_binding_next = 0;
}  // namespace

void note_copy_binding_locked(const VkDescriptorBufferInfo& info, std::uint64_t guest) {
    // Only the draw and dispatch captures ask (copy_binding_guest_locked).
    // Written at every constant-buffer snapshot - ~4,300 a frame at the
    // Nightmare of Mensis on a Steam Deck - its lines were among the command
    // processor's cache misses there.
    static const bool wanted = [] {
        const char* d = std::getenv("BBHOST_CAPTURE_DRAW");
        const char* c = std::getenv("BBHOST_CAPTURE_DISPATCH");
        return (d && *d) || (c && *c);
    }();
    if (!wanted) return;
    g_copy_bindings[g_copy_binding_next++ % 64] = {info.buffer, info.offset, info.range, guest};
}
std::uint64_t copy_binding_guest_locked(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
    for (std::size_t k = 1; k <= 64 && k <= g_copy_binding_next; ++k) {  // the latest first: the scratch ring reuses offsets
        const CopyBinding& c = g_copy_bindings[(g_copy_binding_next - k) % 64];
        if (c.buffer == buffer && c.offset == offset && c.range == range) return c.guest;
    }
    return 0;
}

const std::uint8_t* imported_bytes_locked(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize bytes, std::uint64_t* guest_va) {
    if (guest_va) *guest_va = 0;
    if (!buffer) return nullptr;
    for (std::uint32_t k = 0; k < gcn::kDmemChunks; ++k) for (const auto& imp : g.dmem[k].imports) {
        const Chunk& c = *imp;
        if (c.buffer != buffer) continue;
        const std::uint64_t phys = (static_cast<std::uint64_t>(k) << gcn::kDmemChunkShift) + c.lo + offset;
        if (!g.mirror || offset > c.size || bytes > c.size - offset || phys + bytes > g.mirror_size) return nullptr;
        if (guest_va) {
            for (const GuestMapInfo& mi : g.maps) {
                const std::uint64_t mphys = static_cast<std::uint64_t>(mi.phys);
                if (mi.dmem && phys >= mphys && phys - mphys < mi.len) {
                    *guest_va = mi.va + (phys - mphys);
                    break;
                }
            }
        }
        return static_cast<const std::uint8_t*>(g.mirror) + phys;
    }
    for (const auto& [va, c] : g.anon_imports) {
        if (c.buffer != buffer) continue;
        if (offset > c.size || bytes > c.size - offset) return nullptr;
        if (guest_va) *guest_va = va + offset;
        return reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(va + offset));
    }
    return nullptr;
}

bool page_in_gpu_table_locked(std::uint64_t va) {
    if ((va >> 32) >= gcn::kL1Entries) return false;
    auto it = g.l2.find(static_cast<std::uint32_t>(va >> 32));
    if (it == g.l2.end()) return false;
    return static_cast<const std::uint64_t*>(it->second.map)[(va >> gcn::kPageShift) & (gcn::kL2Entries - 1)] != g.sink.address;
}

namespace {
// BBHOST_GPU_PROFILE=2: the distinct recording sites since the last pair
// ended (the profiler's own calls left out), the first four.
std::array<void*, 4> g_gap_sites{};
bool g_in_profiler = false;
}  // namespace

void profile_note_site_locked(void* site) {
    if (g_in_profiler) return;
    for (void*& s : g_gap_sites) {
        if (s == site) return;
        if (!s) {
            s = site;
            return;
        }
    }
}

void profile_begin_locked(const std::string* name) {
    if (!g.profile) return;
    Gpu::Slot& sl = g.slots[g.slot];
    if (sl.qnames.size() * 2 + 4 > kQueriesPerSlot) return;  // the last pair is the command-buffer span
    g_in_profiler = true;
    if (!sl.qreset) {
        vkCmdResetQueryPool(g_cmd(), sl.qpool, 0, kMaxQueued * kStageSlots * 2);
        if (g.profile_stats) vkCmdResetQueryPool(g_cmd(), sl.stat_pool, 0, kMaxQueued * kStageSlots);
        sl.qreset = true;
    }
    vkCmdWriteTimestamp(g_cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sl.qpool, static_cast<std::uint32_t>(sl.qnames.size() * 2));
    if (g.profile_stats && !sl.stat_open) {
        vkCmdBeginQuery(g_cmd(), sl.stat_pool, static_cast<std::uint32_t>(sl.qnames.size()), 0);
        sl.stat_open = true;
    }
    g_in_profiler = false;
    sl.qnames.push_back(name);
    if (g.profile_gaps) {
        sl.qgaps.push_back(g_gap_sites);
        g_gap_sites = {};
    }
}

void profile_end_locked() {
    if (!g.profile) return;
    Gpu::Slot& sl = g.slots[g.slot];
    if (sl.qnames.empty()) return;
    g_in_profiler = true;
    vkCmdWriteTimestamp(g_cmd(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sl.qpool, static_cast<std::uint32_t>(sl.qnames.size() * 2 - 1));
    if (sl.stat_open) {
        vkCmdEndQuery(g_cmd(), sl.stat_pool, static_cast<std::uint32_t>(sl.qnames.size() - 1));
        sl.stat_open = false;
    }
    g_in_profiler = false;
    g_gap_sites = {};  // what the pair itself recorded is its own time
}

void collect_profile_locked(Gpu::Slot& sl) {
    std::uint64_t span[2] = {};
    bool have_span = false;
    if (g.profile && sl.span) {
        if (vkGetQueryPoolResults(g.device, sl.qpool, kQueriesPerSlot - 2, 2, sizeof(span), span, 8,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS &&
            span[1] > span[0]) {
            g.cmdbuf_ns += static_cast<std::uint64_t>((span[1] - span[0]) * g.timestamp_period_ns);
            ++g.cmdbuf_count;
            have_span = true;
        }
    }
    sl.span = false;
    const auto charge_gap = [&](const std::array<void*, 4>& sites, std::uint64_t from, std::uint64_t to) {
        auto& e = g.profile_gap_ns[sites];
        e.first += to > from ? static_cast<std::uint64_t>((to - from) * g.timestamp_period_ns) : 0;
        e.second += 1;
    };
    if (!g.profile || sl.qnames.empty()) {
        if (g.profile_gaps && have_span) charge_gap(sl.tail_gap, span[0], span[1]);
        sl.qnames.clear();
        sl.qgaps.clear();
        sl.qreset = false;
        return;
    }
    std::vector<std::uint64_t> ts(sl.qnames.size() * 2);
    if (vkGetQueryPoolResults(g.device, sl.qpool, 0, static_cast<std::uint32_t>(ts.size()), ts.size() * 8, ts.data(), 8,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        for (std::size_t i = 0; i < sl.qnames.size(); ++i) {
            const std::uint64_t dt = ts[i * 2 + 1] > ts[i * 2] ? ts[i * 2 + 1] - ts[i * 2] : 0;
            auto& e = g.profile_ns[*sl.qnames[i]];
            e.first += static_cast<std::uint64_t>(dt * g.timestamp_period_ns);
            e.second += 1;
        }
        std::vector<std::uint64_t> stats(sl.qnames.size() * 6);
        if (g.profile_stats &&
            vkGetQueryPoolResults(g.device, sl.stat_pool, 0, static_cast<std::uint32_t>(sl.qnames.size()), stats.size() * 8, stats.data(),
                                  6 * 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            for (std::size_t i = 0; i < sl.qnames.size(); ++i) {
                auto& e = g.profile_stats_sum[*sl.qnames[i]];
                for (int k = 0; k < 6; ++k) e[k] += stats[i * 6 + k];
            }
        }
        // BBHOST_GPU_SLOW_MS (8): a command buffer that ran that long on the
        // GPU, with what took its time - the frames where the command
        // processor waits for a slot and the game's render thread for it.
        static const double slow_ms = [] {
            const char* e = std::getenv("BBHOST_GPU_SLOW_MS");
            return e ? std::atof(e) : 8.0;
        }();
        const double span_ms = have_span ? (span[1] - span[0]) * g.timestamp_period_ns / 1e6 : 0.0;
        if (slow_ms > 0 && span_ms >= slow_ms) {
            std::vector<std::pair<const std::string*, std::uint64_t>> top;
            std::uint64_t inside = 0;
            for (std::size_t i = 0; i < sl.qnames.size(); ++i) {
                const std::uint64_t dt = ts[i * 2 + 1] > ts[i * 2] ? ts[i * 2 + 1] - ts[i * 2] : 0;
                const std::uint64_t ns = static_cast<std::uint64_t>(dt * g.timestamp_period_ns);
                inside += ns;
                auto it = std::find_if(top.begin(), top.end(), [&](const auto& t) { return *t.first == *sl.qnames[i]; });
                if (it == top.end()) top.emplace_back(sl.qnames[i], ns);
                else it->second += ns;
            }
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            std::string list;
            for (std::size_t i = 0; i < top.size() && i < 6; ++i) {
                char b[96];
                std::snprintf(b, sizeof(b), " %s=%.1f", top[i].first->c_str(), top[i].second / 1e6);
                list += b;
            }
            host_log("gpu: a slow command buffer: %.1f ms on the GPU (submission %llu, flip %llu, %zu timed entries %.1f ms, "
                     "the rest outside them):%s",
                     span_ms, static_cast<unsigned long long>(sl.serial), static_cast<unsigned long long>(hle_video_flip_count()),
                     sl.qnames.size(), inside / 1e6, list.c_str());
        }
        // Each gap: from the pair before (or the command buffer's start) to
        // this pair's start; the tail from the last pair to the end.
        if (g.profile_gaps && have_span && sl.qgaps.size() == sl.qnames.size()) {
            for (std::size_t i = 0; i < sl.qnames.size(); ++i) charge_gap(sl.qgaps[i], i ? ts[i * 2 - 1] : span[0], ts[i * 2]);
            charge_gap(sl.tail_gap, ts[ts.size() - 1], span[1]);
        }
    }
    sl.qnames.clear();
    sl.qgaps.clear();
    sl.tail_gap = {};
    sl.qreset = false;
}

// Submits the recorded work and moves on to the next command buffer,
// waiting only if that one is still executing.
void submit_locked() {
    if (!g.recording) {
        return;
    }
    // The occlusion counters this recording dumped land before its labels,
    // as the PS4's DBs write them before the end-of-pipe event that follows.
    occlusion_submit_locked();
    if (landing_on()) hand_labels_to_landing_locked();  // written by the host once this submission has run
    flush_deferred_writes_locked();  // (its transfer batch puts deferred copy-backs in place first)
    transfer_flush_locked();
    render_end_pass_locked();
    view_clears_flush_locked(kVcwSubmit);  // the presenter and the next submission's readers see every clear made so far
    stall_test_record_locked();  // BBHOST_TEST_GPU_STALL
    copy_versions_flush_locked();
    // A render target a shortage put in host memory goes back into video
    // memory once there is room: its copy recorded here, after everything
    // this recording wrote it with (vram_return_on; one at most, once a
    // second).
    render_return_targets_locked();
    drain_target_blocks_locked();  // a targets' block holding little, emptied into room elsewhere (one turn a second)
    compact_image_heap_locked();  // in place of the drains where on: the emptiest block of either class, copies a turn a second
    textures_relocate_pinned_locked();  // a textures' block's few GPU-written surfaces, moved so it can drain (lean thresholds)
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (g.profile && g.slots[g.slot].span) {
        g.slots[g.slot].tail_gap = g_gap_sites;
        g_gap_sites = {};
        g_in_profiler = true;
        rec().write_timestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.slots[g.slot].qpool, kQueriesPerSlot - 1);
        g_in_profiler = false;
    }
    gpu_checkpoint(kGpuMarkSubmission, g.flushes * 4 + 2);
    busy_end_locked(g.slots[g.slot]);
    bump(g_stream_stats.submits);
    // With the stream the recorder ends the command buffer and hands it to
    // bb-submit (kEndSubmit, below); otherwise the command processor does.
    const bool streamed = stream_on();
    if (!streamed) {
        const VkCommandBuffer cmd = g_cmd();
        const std::uint64_t t0 = host_clock_monotonic_ns();
        vkEndCommandBuffer(cmd);
        bump(g_stream_stats.endbegin_ns, host_clock_monotonic_ns() - t0);
        bump(g_stream_stats.endbegin_n);
    }
    g.recording = false;
    // The vertex and index copies this recording's draws read go first.
    const VkCommandBuffer cmds[2] = {shadow_end_uploads_locked(), g.cmd_};
    const VkSemaphore bind_sem = shadow_bind_sparse_locked();
    const VkPipelineStageFlags bind_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = cmds[0] ? 2 : 1;
    si.pCommandBuffers = cmds[0] ? cmds : &g.cmd_;
    if (bind_sem) {
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &bind_sem;
        si.pWaitDstStageMask = &bind_stage;
    }
    Gpu::Slot& cur = g.slots[g.slot];
    cur.serial = g.flushes;
    {
        // What it holds, for the line a wait for it that runs long writes.
        static std::uint64_t last_draws = 0, last_dispatches = 0, last_tess = 0;
        const std::uint64_t draws = g.draws.load(std::memory_order_relaxed), dispatches = g.dispatches.load(std::memory_order_relaxed);
        const std::uint64_t tess = render_tess_draws();
        cur.held_draws = draws - last_draws;
        cur.held_dispatches = dispatches - last_dispatches;
        cur.held_tess = tess - last_tess;
        cur.held_flip = hle_video_flip_count();
        cur.said_stuck = false;
        last_draws = draws;
        last_dispatches = dispatches;
        last_tess = tess;
    }
    textures_before_submit();  // the staging spans its copies read are filled
    // BBHOST_TEST_DEVICE_LOST=<flip>: from that flip on, act as if the device
    // were lost (it is not; the GPU only gets no more work) - the reports, the
    // note for the next starts and the window's ending, tested.
    static const std::uint64_t test_lost = [] {
        const char* e = std::getenv("BBHOST_TEST_DEVICE_LOST");
        return e ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    if (test_lost && g.ok && hle_video_flip_count() >= test_lost) device_lost_locked("BBHOST_TEST_DEVICE_LOST");
    take_submit_failure_locked();
    if (g_device_lost.load(std::memory_order_acquire)) {
        // Lost, or taken for stuck (fence_wait_locked): nothing more goes to
        // the GPU. Work queued behind a submission it never finished would
        // reuse what that one may still read. The recording is dropped; bbhost
        // is ending.
        g.queued = 0;
        return;
    }
    if (streamed) {
        stream::EndSubmit es{};
        es.ncmds = si.commandBufferCount;
        es.cmds[0] = si.pCommandBuffers[0];
        if (es.ncmds > 1) es.cmds[1] = si.pCommandBuffers[1];
        es.items = g.queued;
        es.wait = bind_sem;
        es.wait_stage = bind_stage;
        es.fence = cur.fence;
        es.serial = cur.serial;
        busy_submitted_locked(cur);  // before the recorder can hand it over
        rec().end_submit(es);        // published now: the recorder ends it, bb-submit submits it
        cur.in_flight = true;
    } else if (g_sub_running) {
        SubmitJob job;
        job.ncmds = si.commandBufferCount;
        job.cmds[0] = si.pCommandBuffers[0];
        if (job.ncmds > 1) job.cmds[1] = si.pCommandBuffers[1];
        job.wait = bind_sem;
        job.fence = cur.fence;
        job.items = g.queued;
        job.serial = cur.serial;
        busy_submitted_locked(cur);  // before the thread can submit it
        enqueue_submit(job);
        cur.in_flight = true;
    } else {
        busy_submitted_locked(cur);
        TimelineSignal tl;
        tl.add(si, cur.serial);
        const VkResult r = vkQueueSubmit(g.queue, 1, &si, cur.fence);
        g_stream_submitted.store(cur.serial + 1, std::memory_order_release);
        if (r != VK_SUCCESS) {
            host_log("gpu: submit of %u items failed (%d)%s", g.queued, r,
                     r == VK_ERROR_DEVICE_LOST ? "; device lost, GPU execution disabled" : "");
            g.failures.fetch_add(g.queued);
            if (r == VK_ERROR_DEVICE_LOST) device_lost_locked("a submit");
            busy_retired_locked(cur, false);  // it never went in
        } else {
            cur.in_flight = true;
        }
    }
    // An image found video memory full since the last submission: the
    // mirrors are a cache and the images are not, so they go - freed once
    // this submission, the last that reads them, retires.
    if (g.vram_short.exchange(false, std::memory_order_relaxed)) shadow_give_back_locked(cur);
    else shadow_return_locked();
    shadow_flush_at_load_locked(cur);
    textures_submitted(g.slot);
    g.queued = 0;
    ++g.flushes;
    g.slot = (g.slot + 1) % g.slot_count;
    retire_slot_locked(g.slot);
    stall_test_wait_locked();  // BBHOST_TEST_GPU_STALL
    g.cmd_ = g.slots[g.slot].cmd;
    g.fence = g.slots[g.slot].fence;
    g.desc_pool = g.slots[g.slot].pool;
}

// Submits and waits for everything in flight (frame dumps, resource
// re-purposing, BBHOST_SYNC_GNM).
void flush_locked() {
    submit_locked();
    retire_through_locked(g.flushes - 1);
}

void wait_submission_locked(std::uint64_t serial) {
    if (g.completed_submits > serial || serial >= g.flushes) return;
    const int k = static_cast<int>(serial % static_cast<std::uint64_t>(g.slot_count));
    Gpu::Slot& sl = g.slots[k];
    if (!sl.in_flight || sl.serial != serial) return;  // retired, and its slot taken again
    stream_wait_submitted(serial);                      // its fence has gone to the queue
    // Until it has run (a step's wait used to give up after ten seconds, and
    // the caller reuse what the GPU might still read - the tessellation
    // ring's regions), or the GPU is taken for stuck, after which nothing
    // recorded goes to it.
    std::uint64_t us = 0;
    fence_wait_locked(k, us);
    g.gpu_us.fetch_add(us);
}

void retire_through_locked(std::uint64_t serial) {
    if (g.flushes == 0) return;
    serial = std::min(serial, g.flushes - 1);
    const std::uint64_t n = static_cast<std::uint64_t>(g.slot_count);
    for (std::uint64_t s = g.flushes > n ? g.flushes - n : 0; s <= serial; ++s) {
        const int k = static_cast<int>(s % n);
        if (g.slots[k].in_flight && g.slots[k].serial == s) retire_slot_locked(k);
    }
}

}  // namespace gpu

using namespace gpu;

void host_gpu_gx_buffer_made(std::uint64_t memory, std::uint64_t bytes, std::uint8_t usage, std::uint8_t bind, std::uint64_t gen) {
    twins_buffer_made(memory, bytes, usage, bind, gen);
}

void host_gpu_gx_buffer_mapped_own(std::uint64_t memory, std::uint64_t, std::uint64_t gen) { twins_buffer_mapped_own(memory, gen); }

void host_gpu_gx_buffer_gone(std::uint64_t memory, std::uint64_t gen) { twins_buffer_gone(memory, gen); }

void host_gpu_import_prefetch(std::uint64_t va, std::uint64_t bytes, std::uint8_t usage, std::uint8_t bind) {
    if (imports_whole() || !g_prefetch_on || !bytes) return;
    // What the GPU reads in place (host/import_prefetch.h): an unordered-access
    // buffer, a dynamic one, a vertex or index buffer without the buffer
    // shadow - not one kept as a device-local twin (BBHOST_GX_TWINS), whose
    // twin is filled through staging - and, with the shadow, the meshes it
    // reads in place. Textures never: their uploads go through staging.
    // (Not the meshes while GX on Vulkan binds GX's buffers by handle,
    // BBHOST_GXVK_BUFFERS: its own copies serve them, and nothing reads them in place.)
    const import_prefetch::Ask ask = import_prefetch::ask(usage, bind, buffer_shadow_on(), twins_level(), g_prefetch_meshes && !gxvk::buffers_on());
    if (ask == import_prefetch::Ask::kNone) return;
    const bool meshes = ask == import_prefetch::Ask::kMeshes;
    // The meshes only while the shadow reads told ones in place, which is
    // known once the device is made (the thread drops every job before then):
    // on a discrete GPU with window imports nothing more is queued.
    if (meshes && !(g.ok && shadow_in_place_on())) {
        g_pf_mesh_skipped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::lock_guard<std::mutex> lk(g_pf_mu);
    g_pf_asked.fetch_add(1, std::memory_order_relaxed);
    if (meshes) g_pf_mesh_asked.fetch_add(1, std::memory_order_relaxed);
    if (g_pf_jobs.size() >= 8192) {
        g_pf_dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_pf_jobs.push_back({va, bytes, meshes});
    if (!g_pf_started) {
        g_pf_started = true;
        std::thread(prefetch_thread).detach();
    }
    g_pf_cv.notify_one();
}

void host_gpu_import_waits(std::uint64_t* n, std::uint64_t* us) {
    std::uint64_t cn = 0, cus = 0;
    for (const int who : {kIwBinding, kIwAddress, kIwWalk}) {
        cn += g_import_who_n[who].load(std::memory_order_relaxed);
        cus += g_import_who_us[who].load(std::memory_order_relaxed);
    }
    if (n) *n = cn;
    if (us) *us = cus;
}

bool host_gpu_init(const char* const* instance_exts, std::uint32_t n_exts, bool want_present) {
    if (g_imports_whole.load(std::memory_order_relaxed) < 0) imports_whole_decide();  // the config is read by now
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.tried) {
        for (std::uint32_t i = 0; i < n_exts; ++i) g_instance_exts.push_back(instance_exts[i]);
        g_want_present = want_present;
    }
    return init_locked();
}

namespace {
// What the host asks of the device before it makes it - tight memory, small
// video memory, an integrated GPU - from an instance of its own, without
// extensions, choosing the device as the real one will: one probe for all
// three, and each one's first answer holds (the device's own, when it is
// made first).
void probe_before_device() {
    static std::once_flag once;
    std::call_once(once, [] {
        bool tight = false, small = false, integrated = false, room = false;
        if (g_enabled) {
            obs_capture_policy();  // before the first instance, this one
            VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            app.pApplicationName = "bbhost";
            app.apiVersion = VK_API_VERSION_1_3;
            VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            ici.pApplicationInfo = &app;
            VkInstance instance = VK_NULL_HANDLE;
            if (vkCreateInstance(&ici, nullptr, &instance) == VK_SUCCESS) {
                std::vector<Candidate> usable;
                const int chosen = choose_device(instance, false, usable);
                if (chosen >= 0) {
                    const Candidate& c = usable[static_cast<std::size_t>(chosen)];
                    tight = memory_tight(c);
                    small = small_vram(c);
                    integrated = c.type_rank == 3;
                    room = integrated_room(c);
                }
                vkDestroyInstance(instance, nullptr);
            }
        }
        int unknown = -1;
        g_tight.compare_exchange_strong(unknown, tight ? 1 : 0);
        unknown = -1;
        g_small.compare_exchange_strong(unknown, small ? 1 : 0);
        unknown = -1;
        g_integrated.compare_exchange_strong(unknown, integrated ? 1 : 0);
        unknown = -1;
        g_integrated_room.compare_exchange_strong(unknown, room ? 1 : 0);
    });
}
}  // namespace

bool host_gpu_memory_tight() {
    if (const int t = g_tight.load(); t >= 0) return t == 1;
    probe_before_device();
    return g_tight.load() == 1;
}

bool host_gpu_small_vram() {
    if (const int s = g_small.load(); s >= 0) return s == 1;
    probe_before_device();
    return g_small.load() == 1;
}

bool host_gpu_integrated() {
    if (const int i = g_integrated.load(); i >= 0) return i == 1;
    probe_before_device();
    return g_integrated.load() == 1;
}

bool host_gpu_integrated_with_room() {
    if (const int r = g_integrated_room.load(); r >= 0) return r == 1;
    probe_before_device();
    return g_integrated_room.load() == 1;
}

bool host_gpu_available() {
    std::lock_guard<GpuMutex> lock(g.mu);
    return init_locked();
}

GpuStats host_gpu_stats() {
    GpuStats st{g.draws.load(),      g.dispatches.load(),        g.flushes, g.gpu_us.load(),
                g.gfx_pipelines.load(), g.transfers.load(), g.transfer_batches.load(), {}};
    for (int i = 0; i < 8; ++i) st.phase_ns[i] = g.phase_ns[i].load(std::memory_order_relaxed);
    st.draw_calls = g.draw_calls.load();
    st.draw_failures = g.draw_failures.load();
    st.draws_empty = g.draws_empty.load();
    for (int i = 0; i < kDrawPsCount; ++i) st.draws_ps[i] = g.draws_ps[i].load(std::memory_order_relaxed);
    for (int i = 0; i < kDrawVsCount; ++i) st.draws_vs[i] = g.draws_vs[i].load(std::memory_order_relaxed);
    for (int i = 0; i < kVsWhyCount; ++i) st.draws_vs_why[i] = g.draws_vs_why[i].load(std::memory_order_relaxed);
    for (int i = 0; i < kDrawTessCount; ++i) st.draws_tess[i] = g.draws_tess[i].load(std::memory_order_relaxed);
    for (int i = 0; i < 16; ++i) st.draw_fail_why[i] = g.draw_fail_why[i].load(std::memory_order_relaxed);
    st.shader_dispatches = g.shader_dispatches.load(std::memory_order_relaxed);
    st.lifted_dispatches = g.lifted_dispatches.load(std::memory_order_relaxed);
    st.draws_clip_discard = g.draws_clip_discard.load(std::memory_order_relaxed);
    if (g.oob.map) {
        st.oob_on = true;
        const std::vector<std::uint64_t> totals = oob_totals();
        for (std::uint32_t slot = 0; slot < gcn::kOobSlots; ++slot) {
            for (std::uint32_t k = 0; k < gcn::kOobKinds; ++k) st.oob[k] += totals[slot * gcn::kOobKinds + k];
        }
    }
    return st;
}

static_assert(sizeof(GpuStats::oob) / sizeof(std::uint64_t) == gcn::kOobKinds, "GpuStats::oob holds every counter kind");
namespace {
std::mutex g_oob_mu;
std::vector<std::uint64_t> g_oob_total;  // under g_oob_mu: by slot and kind
std::vector<std::uint32_t> g_oob_seen;   // under g_oob_mu: each counter as last read
}  // namespace

std::vector<std::uint64_t> gpu::oob_totals() {
    if (!g.oob.map) return {};
    const std::size_t n = std::size_t{gcn::kOobSlots} * gcn::kOobKinds;
    std::lock_guard<std::mutex> lk(g_oob_mu);
    if (g_oob_total.size() != n) {
        g_oob_total.assign(n, 0);
        g_oob_seen.assign(n, 0);
    }
    // Each 32-bit counter's change since the last read, modulo 2^32: a counter
    // that wraps between two reads (2^32 lanes in a second) still adds up.
    const auto* c = static_cast<const volatile std::uint32_t*>(g.oob.map);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t now = c[i];
        g_oob_total[i] += static_cast<std::uint32_t>(now - g_oob_seen[i]);
        g_oob_seen[i] = now;
    }
    return g_oob_total;
}

const char* host_gpu_draw_fail_name(int reason) {
    static const char* const why[kFailCount] = {"tessellation off",  "tessellation plan",  "tessellation LS pass",
                                                "primitive type",    "no vertex shader",   "vertex shader",
                                                "pixel shader",      "fetch shader",       "pipeline build",
                                                "descriptor set",    "set allocation",     "fallback bindings",
                                                "pipeline creation", "index buffer",       "indirect arguments",
                                                "no GPU (device lost)"};
    return reason >= 0 && reason < kFailCount ? why[reason] : nullptr;
}
const char* host_gpu_vs_why_name(int why) {
    static const char* const names[kVsWhyCount] = {"tessellation (host)", "tessellation (emulated domain)", "no fetch shader",
                                                    "no GX geometry",      "vertex input off",               "the game's own vertex table",
                                                    "another fetch shape", "a missing record",               "no vertex format",
                                                    "a short stride",      "data outside one buffer"};
    return why >= 0 && why < kVsWhyCount ? names[why] : nullptr;
}
void host_gpu_phase_add(int phase, std::uint64_t ns) { g.phase_ns[phase & 7].fetch_add(ns, std::memory_order_relaxed); }

GpuHandles host_gpu_handles() {
    GpuHandles h;
    h.instance = g.instance;
    h.physical = g.phys;
    h.device = g.device;
    h.queue = g.queue;
    h.present_queue = g.present_queue;
    h.family = g.family;
    h.fifo_latest_ready = g.fifo_latest_ready_ext != nullptr;
    return h;
}

void host_gpu_lock() { g.mu.lock(); }
bool host_gpu_device_lost() { return g_device_lost.load(std::memory_order_acquire); }
void host_gpu_unlock() { g.mu.unlock(); }
void host_gpu_queue_lock() {
    queue_drain();
    g.queue_mu.lock();
}
void host_gpu_queue_unlock() { g.queue_mu.unlock(); }
void host_gpu_queue_lock_only() { g.queue_mu.lock(); }

std::uint64_t host_gpu_submit_presenter(void* cmd, void* wait, std::uint32_t wait_stage, void* signal, void* fence) {
    if (!g_sub_running) return 0;
    if (stream_on()) {
        // Into the stream behind the frame's kEndSubmit: a job given straight
        // to bb-submit could pass a submission the recorder has not handed
        // over yet. Published with a fence (R2): this thread recorded it.
        return kForeignTicket | rec().foreign_submit(static_cast<VkCommandBuffer>(cmd), static_cast<VkSemaphore>(wait), wait_stage,
                                                     static_cast<VkSemaphore>(signal), static_cast<VkFence>(fence));
    }
    SubmitJob job;
    job.cmds[0] = static_cast<VkCommandBuffer>(cmd);
    job.ncmds = 1;
    job.wait = static_cast<VkSemaphore>(wait);
    job.wait_stage = wait_stage;
    job.signal = static_cast<VkSemaphore>(signal);
    job.fence = static_cast<VkFence>(fence);
    return enqueue_submit(job);
}

void host_gpu_wait_submitted(std::uint64_t ticket) {
    if (!ticket) return;
    std::unique_lock<std::mutex> lk(g_sub_mu);
    if (ticket & kForeignTicket) {
        // A kForeignSubmit: once the recorder has replayed up to it and
        // bb-submit has submitted it. Its publish woke the recorder.
        const std::uint64_t id = ticket & ~kForeignTicket;
        g_sub_idle.wait(lk, [id] { return g_foreign_submitted.load(std::memory_order_acquire) >= id; });
        return;
    }
    g_sub_idle.wait(lk, [ticket] { return g_sub_submitted.load(std::memory_order_acquire) >= ticket; });
}

void host_gpu_wait_work_submitted(std::uint64_t need) {
    if (!need || need == ~0ull) return;
    // Its serial is need - 1: in the driver once stream_submitted_serial() is
    // past it. Bounded, and over at a lost device, whose recording is dropped
    // and never submitted: the presenter must not sit here for good.
    const std::uint64_t t0 = host_clock_monotonic_ns();
    for (std::uint64_t spins = 0; stream_submitted_serial() < need; ++spins) {
        if (g_device_lost.load(std::memory_order_acquire)) return;
        if (spins < 4096) {
            std::this_thread::yield();
        } else {
            if (host_clock_monotonic_ns() - t0 > 2'000'000'000ull) {
                static std::atomic<int> said{0};
                if (said.fetch_add(1) < 4)
                    host_log("present: the frame's submissions (to %llu) not in the driver after 2 s (%llu are); the copy goes anyway",
                             static_cast<unsigned long long>(need), static_cast<unsigned long long>(stream_submitted_serial()));
                return;
            }
            host_sleep_us(20);
        }
    }
}

bool host_gpu_submit_for_flip(std::uint64_t need) {
    {
        // The command processor submits after every job (gnm_exec.cpp,
        // BBHOST_SUBMIT_EVERY_JOB) and queues the flip after that, so the
        // flip's frame is normally in a submission already, and whatever is
        // being recorded now is the next frame's.
        std::lock_guard<GpuMutex> lock(g.mu);
        if (!g.ok || g.flushes >= need) return false;
    }
    host_gpu_submit();
    return true;
}

namespace {

// extract_program() scans up to a megabyte for the OrbShdr footer and hashes
// the whole shader; the draw path caches that per code address and the
// dispatch path used to repeat it for every dispatch, which was most of the
// command processor's "dispatch" time during a world load. Same shape of
// cache: keyed by address, revalidated against the first 64 bytes.
struct CachedCompute {
    std::vector<std::uint32_t> words;
    std::string name;
    std::uint64_t hash = 0;
    std::uint32_t head[16];
};
std::unordered_map<std::uint64_t, CachedCompute> g_compute_programs;

const CachedCompute* compute_program_at(std::uint64_t va) {
    if (!hle_kernel_va_mapped(va, 64)) return nullptr;
    const auto* p = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va));
    auto it = g_compute_programs.find(va);
    if (it != g_compute_programs.end() && std::memcmp(it->second.head, p, 64) == 0) return &it->second;
    CachedCompute cp;
    if (!extract_program(va, cp.words, cp.name)) return nullptr;
    cp.hash = fnv1a(cp.words.data(), cp.words.size() * 4);
    std::memcpy(cp.head, p, 64);
    return &(g_compute_programs[va] = cp);
}

}  // namespace

// BBHOST_DISPATCH_COST=1: where a dispatch's time goes. DISPATCH_DIRECT is
// 119,319 packets a run at 25 us each by the per-opcode timing - the largest
// item after the host-draw token - and nearly all of them are GX's copy and
// clear passes.
const bool g_dispatch_cost = [] {
    const char* e = std::getenv("BBHOST_DISPATCH_COST");
    return e && e[0] == '1';
}();
// The rest of the dispatch: its inputs prepared (recognised shapes, images
// prefetched, imports, the counters' starts), the open render pass and the
// pending transfers closed, its parameters and descriptor set, its commands
// recorded, and what it wrote noted (store shadows, writebacks, the barrier).
enum DispatchPart { kDpLock, kDpProgram, kDpTables, kDpPaths, kDpPipeline, kDpPrepare, kDpFlush, kDpSet, kDpRecord, kDpAfter, kDpParts };
const char* const kDispatchPartName[kDpParts] = {"lock", "program lookup", "page tables", "paths and dims", "key and pipeline",
                                                 "inputs and counters", "pass end and transfers", "parameters and set", "recording",
                                                 "what it wrote"};
// What the dispatches actually are. The whole class is at L0,
// and slice 2 was worth measuring before building - it turned out mostly done.
// Counted by shader, and separately the ones that never reach a shader at all
// because they are recognised as a fill or a copy.
std::mutex g_dispatch_census_mu;
std::map<std::string, std::uint64_t> g_dispatch_census;
std::atomic<std::uint64_t> g_dispatch_fills{0}, g_dispatch_copies{0}, g_dispatch_shaders{0};
std::atomic<std::uint64_t> g_dispatch_image_blits{0}, g_dispatch_buf_to_image{0};
std::atomic<std::uint64_t> g_dispatch_ns[kDpParts] = {}, g_dispatch_n{0}, g_dispatch_pt_ns{0}, g_dispatch_pt_slow{0},
    g_dispatch_pt_slow_ns{0};
struct DispatchStamp {
    std::chrono::steady_clock::time_point at = std::chrono::steady_clock::now();
    void to(DispatchPart p) {
        if (!g_dispatch_cost) return;
        const auto now = std::chrono::steady_clock::now();
        g_dispatch_ns[p].fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - at).count()),
                                   std::memory_order_relaxed);
        at = now;
    }
};
// The compute shaders by how they ran (BBHOST_LIFT_COMPUTE): programs and
// dispatches lifted, and the rest by why their lift was refused, or not asked
// for. Takes g.mu (the pipelines' counts).
std::string compute_lift_report() {
    // A rejection without the instruction offset it names (as tools/area_check.py strips it).
    const auto reason_of = [](std::string r) {
        if (r.size() > 8 && r[6] == ':' && r[7] == ' ') r = r.substr(8);
        return r;
    };
    std::map<std::string, std::string> decided;
    {
        std::lock_guard<std::mutex> lk(g_lift_decided_mu);
        decided = g_lift_decided;
    }
    std::lock_guard<GpuMutex> lock(g.mu);
    std::map<std::string, std::pair<std::set<std::string>, std::uint64_t>> why;  // reason -> programs, dispatches
    std::map<std::string, std::uint64_t> by_program;                             // dispatched programs' dispatches
    std::set<std::string> lifted_dispatched;
    std::uint64_t dispatches = 0, lifted_dispatches = 0;
    for (const auto& [key, pl] : g.pipelines) {
        if (pl.failed || pl.name.empty()) continue;
        by_program[pl.name] += pl.dispatches;
        dispatches += pl.dispatches;
        if (pl.lifted) {
            lifted_dispatches += pl.dispatches;
            continue;
        }
        auto& w = why[reason_of(pl.lift_why.empty() ? std::string("not asked for (BBHOST_LIFT_COMPUTE=0 or BBHOST_DECOMP)") : pl.lift_why)];
        w.first.insert(pl.name);
        w.second += pl.dispatches;
    }
    std::size_t made_lifted = 0;
    for (const auto& [name, reason] : decided) {
        if (reason.empty()) {
            ++made_lifted;
        } else if (!by_program.count(name)) {
            why[reason_of(reason)].first.insert(name);  // made at GX's creation, never dispatched
        }
    }
    for (const auto& [name, n] : by_program) {
        bool all = true;
        for (const auto& [reason, w] : why) all = all && !w.first.count(name);
        if (all) lifted_dispatched.insert(name);
    }
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "compute shader lifts: %zu of %zu programs lifted (their pipelines made at GX's creation or a dispatch); %zu of %zu "
                  "dispatched programs, %llu of %llu dispatches ran a lift",
                  made_lifted, decided.size(), lifted_dispatched.size(), by_program.size(), static_cast<unsigned long long>(lifted_dispatches),
                  static_cast<unsigned long long>(dispatches));
    std::string out = buf;
    for (const auto& [reason, w] : why) {
        std::string names;
        for (const std::string& n : w.first) names += (names.empty() ? "" : " ") + n;
        std::snprintf(buf, sizeof(buf), "; translated (%zu programs, %llu dispatches): ", w.first.size(), static_cast<unsigned long long>(w.second));
        out += buf + reason + " [" + names + "]";
    }
    return out;
}

void dispatch_cost_report() {
    const std::uint64_t n = g_dispatch_n.load();
    if (!g_dispatch_cost || !n) return;
    std::string parts;
    for (int k = 0; k < kDpParts; ++k) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " %s %llu;", kDispatchPartName[k],
                      static_cast<unsigned long long>(g_dispatch_ns[k].load() / n));
        parts += buf;
    }
    host_log("gpu: dispatch split over %llu dispatches, ns each:%s the page-table call alone %llu",
             static_cast<unsigned long long>(n), parts.c_str(), static_cast<unsigned long long>(g_dispatch_pt_ns.load() / n));
    const std::uint64_t slow = g_dispatch_pt_slow.load();
    host_log("gpu:   of those page-table calls %llu took over a microsecond (%.2f%%) and are %llu ms of the %llu ms total",
             static_cast<unsigned long long>(slow), 100.0 * slow / n,
             static_cast<unsigned long long>(g_dispatch_pt_slow_ns.load() / 1000000),
             static_cast<unsigned long long>(g_dispatch_pt_ns.load() / 1000000));
}

namespace {
std::atomic<std::uint64_t> g_unresolved_stores{0};
// BBHOST_STORE_LOG=1: the first 400 buffer ranges compute stores are taken to write.
const bool g_store_log = std::getenv("BBHOST_STORE_LOG") != nullptr;
std::atomic<std::uint64_t> g_store_logs{0};
// GDS counters (GpuDispatch::gds): dispatches that counted in at least one
// bound word, ones whose program counts in GDS with none bound, and the
// starts written or copied into a word first.
std::atomic<std::uint64_t> g_gds_dispatches{0}, g_gds_unbound{0}, g_gds_sets{0}, g_gds_copies{0}, g_gds_start_failed{0};
// BBHOST_GDS_TRACE=1: the first 64 dispatches that count in GDS, with their words.
const bool g_gds_trace = [] {
    const char* e = std::getenv("BBHOST_GDS_TRACE");
    return e && e[0] == '1';
}();
std::atomic<int> g_gds_traced{0};
bool counter_start_locked(const GpuDispatch::GdsCounter& c);
}

bool host_gpu_gds_counters() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_GDS_COUNTERS");
        return !(e && e[0] == '0');
    }();
    return on;
}

ParticlePath gpu::particle_path() {
    static const ParticlePath p = [] {
        const char* e = std::getenv("BBHOST_GX_PARTICLES");
        return e && std::strcmp(e, "gs") == 0 ? ParticlePath::kGs : ParticlePath::kVs;
    }();
    return p;
}

// BBHOST_GDS_SUBGROUP (on; =0 off): a lifted compute shader steps a GDS
// counter once a subgroup, as GCN does once a wave (gcn::TranslateOptions::
// gds_subgroup), not once an invocation.
bool host_gpu_gds_subgroup() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_GDS_SUBGROUP");
        return !(e && e[0] == '0');
    }();
    return on;
}

thread_local std::uint32_t t_dispatch_user_reads = 0xffff;
std::uint32_t host_gpu_dispatch_user_reads() { return t_dispatch_user_reads; }

// Dispatch tokens from GX's compute commit (GpuDispatch::cs_records): user
// data built from their records, and those whose records the builder did not
// take (their user data stays the token's).
std::atomic<std::uint64_t> g_cs_records_built{0}, g_cs_records_failed{0};

namespace {
// What the last host_gpu_dispatch on this thread turned out to be, and its
// program's name (BBHOST_CP_CENSUS sorts the dispatches by it).
thread_local GpuDispatchClass t_dispatch_class = kDcNone;
thread_local const std::string* t_dispatch_name = nullptr;
}  // namespace

GpuDispatchClass host_gpu_last_dispatch_class(const char** program) {
    if (program) *program = t_dispatch_name ? t_dispatch_name->c_str() : "-";
    return t_dispatch_class;
}

bool host_gpu_dispatch(const GpuDispatch& d_in) {
    t_dispatch_user_reads = 0xffff;
    t_dispatch_class = kDcFailed;
    t_dispatch_name = nullptr;
    const ResolveKind resolve_kind("dispatch");
    DispatchStamp stamp;
    GpuPhaseTimer timer(kPhaseLock);
    std::lock_guard<GpuMutex> lock(g.mu);
    timer.next(kPhaseDispatch);
    g_dispatch_n.fetch_add(1, std::memory_order_relaxed);
    stamp.to(kDpLock);
    if (!init_locked()) {
        return false;
    }
    // A token's compute records: the user data its descriptors write, built
    // as GX's compute commit writes it, the tables in the host ring. The rest
    // of the user data (dwords no descriptor writes) stays the token's.
    GpuDispatch built;
    const GpuDispatch* dp = &d_in;
    if (d_in.cs_records) {
        built = d_in;
        std::uint16_t set = 0;
        if (render_cs_user_data_locked(*d_in.cs_records, built.user_data, &set)) {
            g_cs_records_built.fetch_add(1, std::memory_order_relaxed);
            dp = &built;
        } else if (g_cs_records_failed.fetch_add(1, std::memory_order_relaxed) < 4) {
            host_log("gpu: a dispatch token's compute records name a descriptor the builder does not take: its user data is the token's");
        }
    }
    const GpuDispatch& d = *dp;
    const CachedCompute* cc = compute_program_at(d.code_va);
    if (!cc) {
        static std::atomic<int> logs{0};
        if (logs.fetch_add(1) < 4) {
            host_log("gpu: dispatch with unreadable shader at 0x%llx", static_cast<unsigned long long>(d.code_va));
        }
        g.failures.fetch_add(1);
        return false;
    }
    const std::vector<std::uint32_t>& words = cc->words;
    const std::string& name = cc->name;
    const std::uint64_t code_hash = cc->hash;
    t_dispatch_name = &cc->name;
    stamp.to(kDpProgram);
    if (host_tooling()) {
        g_dispatch_shaders.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(g_dispatch_census_mu);
        ++g_dispatch_census[name];
    }
    const auto t_pt = std::chrono::steady_clock::now();
    const bool pt_ok = rebuild_page_tables();
    {
        const std::uint64_t ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_pt).count());
        g_dispatch_pt_ns.fetch_add(ns, std::memory_order_relaxed);
        // Is the mean work, or a few stalls? Count the calls over a microsecond
        // and what they alone contribute.
        if (ns > 1000) {
            g_dispatch_pt_slow.fetch_add(1, std::memory_order_relaxed);
            g_dispatch_pt_slow_ns.fetch_add(ns, std::memory_order_relaxed);
        }
    }
    if (!pt_ok) {
        g.failures.fetch_add(1);
        return false;
    }
    stamp.to(kDpTables);
    // Image dimensions from the resolved T#s decide the translated image types.
    const std::vector<std::pair<std::uint32_t, bool>> dims =
        image_dims_for(paths_for(code_hash, words, gcn::Stage::Compute, d.rsrc1, d.rsrc2), d.user_data);
    const auto sampler_modes = sampler_modes_for(paths_for(code_hash, words, gcn::Stage::Compute, d.rsrc1, d.rsrc2), d.user_data);
    stamp.to(kDpPaths);
    const std::uint64_t key = compute_key(code_hash, d.rsrc1, d.rsrc2, d.threads, dims, sampler_modes);
    ComputePipeline& pl = pipeline_for(d, key, words, name, dims, sampler_modes);
    if (g.oob.map && !pl.failed && !pl.oob_named) {  // BBHOST_OOB_COUNT: the slot's name, once
        pl.oob_named = true;
        note_oob_compute_slot(pl.meta.oob_slot, pl.name);
    }
    t_dispatch_user_reads = 0xffff;
    if (d.want_user_reads && !pl.failed) {
        if (!pl.user_reads_known) {
            // What the program's translation touches, and what the host reads
            // to bind its resources: each path's root - the descriptor's words
            // when they sit in user data, the pointer when the path loads
            // through memory.
            std::uint32_t tables = 0, any = 0xffff;
            std::uint32_t reads = table_slots_read(pl.meta, tables, &any) ? (any & 0xffff) : 0xffff;
            const auto root = [&](const gcn::ResourcePath& p, int words) {
                if (p.user_sgpr < 0 || p.user_sgpr >= 16) return;
                const int n = p.immediate ? words : 2;
                for (int k = 0; k < n && p.user_sgpr + k < 16; ++k) reads |= 1u << (p.user_sgpr + k);
            };
            for (const gcn::ImageBinding& b : pl.meta.images) root(b.path, b.r128 ? 4 : 8);
            for (const gcn::SamplerBinding& b : pl.meta.samplers) root(b.path, 4);
            for (const gcn::BufferBinding& b : pl.meta.buffers) root(b.path, b.pointer ? 2 : 4);
            for (const gcn::ResourcePath& p : pl.meta.store_buffers) root(p, 4);
            pl.user_reads = reads;
            pl.user_reads_known = true;
        }
        t_dispatch_user_reads = pl.user_reads;
    }
    if (pl.failed) {
        g.failures.fetch_add(1);
        return false;
    }
    if (!rebuild_page_tables()) {
        g.failures.fetch_add(1);
        return false;
    }
    stamp.to(kDpPipeline);
    // Counted before anything acts on them: the first question is whether the
    // shapes are recognised on exactly the dispatches the census said, and
    // nothing else.
    if (pl.image_blit) g_dispatch_image_blits.fetch_add(1, std::memory_order_relaxed);
    if (pl.buffer_to_image) g_dispatch_buf_to_image.fetch_add(1, std::memory_order_relaxed);
    t_dispatch_class = pl.image_blit ? kDcImageBlit : pl.buffer_to_image ? kDcBufferToImage : pl.fill ? kDcFillShader : pl.copy ? kDcCopyShader : kDcShader;
    if (pl.fill) {
        const std::uint32_t* v = &d.user_data[pl.fill_dst_sgpr];
        // V# BASE_ADDRESS is a 40-bit byte address. T#/CB_COLOR are in 256-byte
        // units; if the byte address is unmapped, try << 8.
        std::uint64_t base = static_cast<std::uint64_t>(v[0]) | (static_cast<std::uint64_t>(v[1] & 0xff) << 32);
        if (!hle_kernel_va_mapped(base, 16) && hle_kernel_va_mapped(base << 8, 16)) base <<= 8;
        const std::uint32_t stride = (v[1] >> 16) & 0x3fff;
        const std::size_t bytes = static_cast<std::size_t>(v[2]) * (stride ? stride : 1);
        const std::uint32_t* c = &d.user_data[pl.fill_cb_sgpr];
        const std::uint64_t cb = (static_cast<std::uint64_t>(c[0]) | (static_cast<std::uint64_t>(c[1] & 0xff) << 32)) + pl.fill_cb_dw * 4;
        float rgba[4] = {};
        // The fill takes the value as given: what a buffer_store_format_xyzw
        // through a V# of 32_32_32_32 with the select X Y Z W writes. Another
        // format or select would convert it and put each component through the
        // select's inverse on the console (GX's own fill, gx_fill_resource,
        // builds its V# so: 0xfac40e); logged once if one is ever seen.
        if (pl.fill_format && (((v[3] >> 15) & 0xf) != 14 || (v[3] & 0xfff) != 0xfac)) {
            static std::atomic<bool> logged{false};
            if (!logged.exchange(true)) {
                host_log("gpu: fill dispatch %s stores through a V# whose format or select is not 32_32_32_32 X Y Z W (word 3 %08x): the value "
                         "is filled as given",
                         pl.name.c_str(), v[3]);
            }
        }
        if (hle_kernel_va_mapped(cb, 16)) {
            std::memcpy(rgba, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cb)), 16);
            static std::set<std::uint64_t> seen_bases;
            if (seen_bases.size() < 80 && seen_bases.insert(base).second) {
                host_log("gpu: fill dispatch %s: 0x%llx %zu bytes (stride %u) with %g %g %g %g -> %s", pl.name.c_str(),
                         static_cast<unsigned long long>(base), bytes, stride, rgba[0], rgba[1], rgba[2], rgba[3],
                         find_render_target(base) ? "render target" : "pending");
            }
            hle_gnm_warn_if_command_buffer("fill dispatch", base, bytes);
            // Only what lands on a texture or a render target: the rest are
            // buffers, and hundreds a frame would push the history out.
            const bool on_rt = find_render_target(base) != nullptr;
            if (mark_surfaces_dirty_locked(base, bytes, false) || on_rt) {
                tex_event(base, bytes, "fill dispatch %s 0x%llx 0x%zx bytes with %g %g %g %g%s", pl.name.c_str(),
                          static_cast<unsigned long long>(base), bytes, rgba[0], rgba[1], rgba[2], rgba[3],
                          on_rt ? " (render target)" : "");
            }
            if (render_handle_fill_locked(base, bytes, rgba, d.depth_clear, d.stencil_clear)) {
                t_dispatch_class = kDcFill;
                particle_args_after_dispatch_locked();  // a fill may reset a particle counter
                g.dispatches.fetch_add(1);
                g_dispatch_fills.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
        }
    }
    if (pl.copy) {
        const std::uint32_t* sv = &d.user_data[pl.copy_src_sgpr];
        const std::uint32_t* dv = &d.user_data[pl.copy_dst_sgpr];
        const std::uint64_t src = static_cast<std::uint64_t>(sv[0]) | (static_cast<std::uint64_t>(sv[1] & 0xff) << 32);
        const std::uint64_t dst = static_cast<std::uint64_t>(dv[0]) | (static_cast<std::uint64_t>(dv[1] & 0xff) << 32);
        const std::uint32_t stride = (dv[1] >> 16) & 0x3fff;
        const std::size_t bytes = static_cast<std::size_t>(dv[2]) * (stride ? stride : 1);
        const std::uint64_t watch_lo = g_watch_cb_lo.load(std::memory_order_relaxed);
        const std::uint64_t watch_hi = g_watch_cb_hi.load(std::memory_order_relaxed);
        const std::uint64_t watch_alias = watch_hi ? hle_kernel_gpu_alias(watch_lo) : 0;
        const bool into_watch = watch_hi && ((dst < watch_hi && dst + bytes > watch_lo) ||
                                             (watch_alias && dst < watch_alias + (watch_hi - watch_lo) && dst + bytes > watch_alias));
        if (into_watch) {
            static std::atomic<std::uint64_t> hits{0};
            const std::uint64_t hit = hits.fetch_add(1);
            if (hit < 12 || (hit & (hit + 1)) == 0) {
                host_log("gpu: copy dispatch %s writes the watched constant buffer [0x%llx,0x%llx): 0x%llx -> 0x%llx %zu bytes "
                         "(copy %llu, flip %llu)",
                         pl.name.c_str(), static_cast<unsigned long long>(watch_lo), static_cast<unsigned long long>(watch_hi),
                         static_cast<unsigned long long>(src), static_cast<unsigned long long>(dst), bytes,
                         static_cast<unsigned long long>(hit + 1), static_cast<unsigned long long>(hle_video_flip_count()));
            }
        }
        static std::set<std::uint64_t> seen;
        if (seen.size() < 60 && seen.insert(src).second) {
            host_log("gpu: copy dispatch %s: 0x%llx -> 0x%llx %zu bytes (stride %u)%s", pl.name.c_str(), static_cast<unsigned long long>(src),
                     static_cast<unsigned long long>(dst), bytes, stride, find_render_target(src) ? " (render target)" : "");
        }
        hle_gnm_warn_if_command_buffer("copy dispatch", dst, bytes);
        const bool as_image = render_copy_target_locked(src, dst, bytes, "copy dispatch");
        const int on_surfaces = mark_surfaces_dirty_locked(dst, bytes, !as_image, "copy dispatch");  // an image copy writes no memory
        if (as_image) {
            tex_event(dst, bytes, "copy dispatch %s 0x%llx -> 0x%llx 0x%zx bytes: render target copied as an image", pl.name.c_str(),
                      static_cast<unsigned long long>(src), static_cast<unsigned long long>(dst), bytes);
            t_dispatch_class = kDcCopyImage;
            g.dispatches.fetch_add(1);
            g_dispatch_copies.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        if (on_surfaces || find_render_target(src)) {
            tex_event(dst, bytes, "copy dispatch %s 0x%llx -> 0x%llx 0x%zx bytes: as memory%s", pl.name.c_str(),
                      static_cast<unsigned long long>(src), static_cast<unsigned long long>(dst), bytes,
                      find_render_target(src) ? " (FROM A RENDER TARGET)" : "");
        }
    }
    static thread_local StageImages stage_images;
    prefetch_stage_images(pl.meta, d.user_data, stage_images);
    // On-demand imports (window imports): what the dispatch's shader reaches
    // through the page table is imported before it runs - its stores and its
    // counter words here, what it reads once the variant that runs is known
    // (below: the lean one binds its buffers).
    const auto import_path = [&](const gcn::ResourcePath& path, bool pointer, std::uint64_t min_bytes) {
        std::uint32_t v[4]{};
        if (!resolve_resource(path, d.user_data, pointer ? 2 : 4, v)) return;
        const std::uint64_t base = v[0] | (static_cast<std::uint64_t>(pointer ? v[1] : (v[1] & 0xff)) << 32);
        const std::uint32_t stride = pointer ? 0 : (v[1] >> 16) & 0x3fff;
        const std::uint64_t bytes = pointer ? 0 : stride ? std::uint64_t{v[2]} * stride : v[2];
        const std::uint64_t span = std::min<std::uint64_t>(std::max(bytes, min_bytes), 64ull << 20);
        // A GX buffer a device-local twin holds (or will, at its binding below:
        // BBHOST_GX_TWINS) needs no import - its stores go to the twin, also
        // when the V# claims more than the buffer holds (a 12-byte argument
        // buffer stored as 16); a binding that lets the twin go imports what
        // it needs at its locate.
        if (base && !twin_candidate_holds(base, 1)) {
            if (twins_on()) twin_note_import_near_locked(base, span, t_import_site);
            import_windows(base, span);
        }
    };
    if (!imports_whole()) {
        {
            const ImportSite site("a dispatch's stores");
            for (const gcn::ResourcePath& p : pl.meta.store_buffers) import_path(p, false, 16);
        }
        // BBHOST_VB_RACE=1: what the dispatch reads and never stores to, as
        // the draws' constants are checked (render.cpp).
        if (vb_race_on()) {
            for (const gcn::BufferBinding& b : pl.meta.buffers) {
                std::uint32_t v[4]{};
                if (b.written || b.pointer || !resolve_resource(b.path, d.user_data, 4, v)) continue;
                const std::uint64_t base = v[0] | (static_cast<std::uint64_t>(v[1] & 0xff) << 32);
                const std::uint32_t stride = (v[1] >> 16) & 0x3fff;
                const std::uint64_t bytes = std::min<std::uint64_t>(stride ? std::uint64_t{v[2]} * stride : v[2], b.max_dw * 4ull);
                if (!base || !bytes || gpu_writes_pending_locked(base, bytes)) continue;
                char tag[40];
                std::snprintf(tag, sizeof(tag), "d:%s", pl.name.c_str());  // a dispatch's buffer (render.cpp: c:, v:, i:)
                vb_race_note_locked(tag, base, bytes);
            }
        }
        if (pl.meta.gds_counters) {
            const ImportSite site("a GDS counter word");
            for (const GpuDispatch::GdsCounter& c : d.gds) {
                if (c.word && !twin_candidate_holds(c.word, 4)) import_windows(c.word, 4);
                if (c.start == GpuDispatch::kGdsCopy && c.from && !twin_candidate_holds(c.from, 4)) import_windows(c.from, 4);
            }
        }
    }
    if (g.queued + 1 >= kMaxQueued * kStageSlots) {
        submit_locked();
    }
    begin_recording_locked();
    // BBHOST_PARTICLE_TWINS: the buffers a dispatch that counts binds - its
    // counter words' (the records, the counter at their end), what it stores
    // to and what it reads - become device-local twins from the next
    // recording on.
    if (pl.meta.gds_counters && twins_level() == 3) {
        for (const GpuDispatch::GdsCounter& c : d.gds) {
            if (c.word) twins_want_locked(c.word);
        }
        const auto want = [&](const gcn::ResourcePath& path, bool pointer) {
            std::uint32_t v[4]{};
            if (!resolve_resource(path, d.user_data, pointer ? 2 : 4, v)) return;
            twins_want_locked(v[0] | (static_cast<std::uint64_t>(pointer ? v[1] : (v[1] & 0xff)) << 32));
        };
        for (const gcn::ResourcePath& path : pl.meta.store_buffers) want(path, false);
        for (const gcn::BufferBinding& b : pl.meta.buffers) want(b.path, b.pointer);
    }
    // The counters' starts go first, on the GPU's timeline: a set count (an
    // initial count of 0 for the buffer the dispatch appends to) written into
    // the word, a count from another word copied in.
    bool gds_bound = false;
    if (pl.meta.gds_counters) {
        for (const GpuDispatch::GdsCounter& c : d.gds) {
            if (!c.word) continue;
            gds_bound = true;
            if (c.start == GpuDispatch::kGdsKeep) continue;
            if (counter_start_locked(c)) {
                (c.start == GpuDispatch::kGdsSet ? g_gds_sets : g_gds_copies).fetch_add(1, std::memory_order_relaxed);
            } else if (g_gds_start_failed.fetch_add(1, std::memory_order_relaxed) < 4) {
                host_log("gpu: dispatch %s: the start of the GDS counter at 0x%llx (%s) could not be recorded: it counts on from what it holds",
                         pl.name.c_str(), static_cast<unsigned long long>(c.word), c.start == GpuDispatch::kGdsSet ? "a set count" : "a copy");
            }
        }
        if (gds_bound) {
            g_gds_dispatches.fetch_add(1, std::memory_order_relaxed);
            // The dispatch counts in the word while it runs, where GCN counted
            // in GDS and left the word as it was until the store after: a V#
            // of the program's over a counter word would see the count move.
            // GX reads the live count from the indirect arguments instead;
            // said once per program if one ever does.
            if (!pl.gds_overlap_checked) {
                pl.gds_overlap_checked = true;
                const auto covers = [&](const gcn::ResourcePath& path) -> std::uint64_t {
                    std::uint32_t v[4]{};
                    if (!resolve_resource(path, d.user_data, 4, v)) return 0;
                    const std::uint64_t base = v[0] | (static_cast<std::uint64_t>(v[1] & 0xff) << 32);
                    const std::uint32_t stride = (v[1] >> 16) & 0x3fff;
                    const std::uint64_t bytes = stride ? std::uint64_t{v[2]} * stride : v[2];
                    for (const GpuDispatch::GdsCounter& c : d.gds) {
                        if (c.word && c.word >= base && c.word < base + bytes) return c.word;
                    }
                    return 0;
                };
                for (const gcn::BufferBinding& b : pl.meta.buffers) {
                    if (const std::uint64_t w = b.pointer ? 0 : covers(b.path)) {
                        host_log("gpu: dispatch %s reads %s, which holds the GDS counter word 0x%llx it counts in: it sees the count move",
                                 pl.name.c_str(), b.path.str().c_str(), static_cast<unsigned long long>(w));
                    }
                }
                for (const gcn::ResourcePath& path : pl.meta.store_buffers) {
                    if (const std::uint64_t w = covers(path)) {
                        host_log("gpu: dispatch %s stores through %s, which holds the GDS counter word 0x%llx it counts in", pl.name.c_str(),
                                 path.str().c_str(), static_cast<unsigned long long>(w));
                    }
                }
            }
        } else if (g_gds_unbound.fetch_add(1, std::memory_order_relaxed) < 8) {
            host_log("gpu: dispatch %s counts in GDS (%u site(s)) and the command buffer bound no counter for it: its ds_append / ds_consume "
                     "return 0 and count nothing",
                     pl.name.c_str(), pl.meta.gds_counters);
        }
        if (g_gds_trace && g_gds_traced.fetch_add(1, std::memory_order_relaxed) < 64) {
            std::string words;
            for (std::uint32_t k = 0; k < GpuDispatch::kGdsCounters; ++k) {
                const GpuDispatch::GdsCounter& c = d.gds[k];
                if (!c.word) continue;
                char t[160];
                std::uint32_t now = 0;
                if (hle_kernel_va_mapped(c.word, 4)) std::memcpy(&now, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(c.word)), 4);
                std::snprintf(t, sizeof(t), " [GDS 0x%x] 0x%llx (holds %u on the CPU) %s", (d.gds_base + k) * 4, static_cast<unsigned long long>(c.word), now,
                              c.start == GpuDispatch::kGdsKeep ? "kept" : c.start == GpuDispatch::kGdsSet ? "set to" : "copied from");
                words += t;
                if (c.start == GpuDispatch::kGdsSet) words += " " + std::to_string(c.value);
                if (c.start == GpuDispatch::kGdsCopy) {
                    std::snprintf(t, sizeof(t), " 0x%llx", static_cast<unsigned long long>(c.from));
                    words += t;
                }
            }
            host_log("gpu: GDS dispatch %s %ux%ux%u%s%s at flip %llu:%s", pl.name.c_str(), d.dim[0], d.dim[1], d.dim[2], d.indirect_va ? " indirect at " : "",
                     d.indirect_va ? std::to_string(d.indirect_va).c_str() : "", static_cast<unsigned long long>(hle_video_flip_count()),
                     words.empty() ? " none bound" : words.c_str());
        }
    }
    stamp.to(kDpPrepare);
    transfer_flush_locked();
    render_end_pass_locked();
    view_clears_flush_locked(kVcwDispatch);  // a compute shader may read or write any image: no clear stays owed past it
    copy_versions_flush_locked();  // a compute shader reads memory without looking at them
    if (g.depth_layouts) depth_layouts_for_images_locked(&stage_images, 1, nullptr);  // the depth targets it samples, read-only
    stamp.to(kDpFlush);
    // BBHOST_CAPTURE_DISPATCH (draw_capture.h): run everything recorded so
    // far - the counters' starts too - before this dispatch's set exists, so
    // memory and images hold its inputs.
    const bool capture = capture_dispatch_candidate(pl.name, hle_video_flip_count());
    if (capture) {
        capture_settle_locked();
        begin_recording_locked();
    }
    gcn::StageParams params{};
    params.l1_table = g.l1.address;
    params.lds_address = g.oob.address;  // BBHOST_OOB_COUNT=1: the counters (gcn::set_oob_count); 0 otherwise
    std::memcpy(params.user_sgpr, d.user_data, sizeof(params.user_sgpr));
    if (gds_bound) {
        params.gds_base = d.gds_base;
        // The words by their device addresses: the shader takes them as such,
        // not through the page table - their place in the import, or in a GX
        // buffer's device-local twin (gx twins, below). 0 (not reachable): the
        // counter does nothing, where the page table gave the sink page.
        const ImportSite site("a GDS counter word");
        for (std::uint32_t k = 0; k < GpuDispatch::kGdsCounters; ++k)
            params.gds_counter[k] = d.gds[k].word ? guest_device_address(d.gds[k].word, 4) : 0;
    }
    // BBHOST_CS_BUFFERS: the variant with its buffers pushed, when every one
    // of them binds; otherwise the pipeline that walks the page table. Not
    // under a capture, which records the walking translation's options.
    const ComputeLean* lean = !capture && pl.lean && pl.lean->pipeline ? pl.lean.get() : nullptr;
    static thread_local std::vector<VkDescriptorBufferInfo> lean_infos;
    if (lean) {
        // What the dispatch writes - through its stores and its counter
        // words - is a GPU write from here, so no binding over it is a copy
        // taken now: the shader would read the copy where its writes go to
        // memory.
        for (const gcn::ResourcePath& path : lean->meta.store_buffers) {
            std::uint32_t v[4]{};
            if (!resolve_resource(path, d.user_data, 4, v)) continue;
            const std::uint64_t base = v[0] | (static_cast<std::uint64_t>(v[1] & 0xff) << 32);
            const std::uint32_t stride = (v[1] >> 16) & 0x3fff;
            if (base) gpu_writes_note_locked(base, stride ? std::uint64_t{v[2]} * stride : v[2]);
        }
        if (gds_bound) {
            for (const GpuDispatch::GdsCounter& c : d.gds) {
                if (c.word) gpu_writes_note_locked(c.word, 4);
            }
        }
        lean_infos.clear();
        resolve_dispatch_buffers(lean->meta, pl.name.c_str(), d.user_data, params, lean_infos);
        const std::size_t n = lean->meta.buffers.size();
        const std::uint32_t all = n >= 32 ? ~0u : (1u << n) - 1;
        if ((params.cb_valid & all) != all) {
            lean = nullptr;
            g_cs_lean_fallbacks.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_cs_lean_dispatches.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // Window imports: what the variant that runs reads through the page
    // table - every buffer of the walking one; of the lean one, whose buffers
    // bound above (a copy as recorded, or a binding's own import), only the
    // walks left in it (TranslateResult::read_buffers: a V#'s records, a
    // pointer's dwords up to the highest a load reads, 64 KiB when no load's
    // offset is known), and its stores.
    // GX buffers' device-local twins (BBHOST_GX_TWINS=1) a page-table walk
    // reaches - every buffer of the walking variant, the walks left in the
    // lean one - are copied back and let go first: the walk reads guest memory.
    if (twins_on()) {
        const auto release = [&](const gcn::ResourcePath& path, bool pointer, std::uint64_t min_bytes) {
            std::uint32_t v[4]{};
            if (!resolve_resource(path, d.user_data, pointer ? 2 : 4, v)) return;
            const std::uint64_t base = v[0] | (static_cast<std::uint64_t>(pointer ? v[1] : (v[1] & 0xff)) << 32);
            const std::uint32_t stride = pointer ? 0 : (v[1] >> 16) & 0x3fff;
            const std::uint64_t bytes = pointer ? 0 : stride ? std::uint64_t{v[2]} * stride : v[2];
            if (base) twins_release_locked(base, std::min<std::uint64_t>(std::max(bytes, min_bytes), 64ull << 20), 0);
        };
        const gcn::TranslateResult& walks = lean ? lean->meta : pl.meta;
        if (!lean) {
            for (const gcn::BufferBinding& b : pl.meta.buffers) release(b.path, b.pointer, std::max<std::uint64_t>(b.max_dw * 4ull, 16));
            for (const gcn::ResourcePath& p : pl.meta.store_buffers) release(p, false, 16);
        }
        for (const gcn::TranslateResult::BufferRead& r : walks.read_buffers) {
            release(r.path, r.pointer, r.pointer ? (r.max_dw ? r.max_dw * 4ull : 64ull << 10) : 16);
        }
    }
    if (!imports_whole()) {
        const gcn::TranslateResult& walks = lean ? lean->meta : pl.meta;
        if (!lean) {
            const ImportSite site("a walking dispatch's buffers");
            for (const gcn::BufferBinding& b : pl.meta.buffers) import_path(b.path, b.pointer, std::max<std::uint64_t>(b.max_dw * 4ull, 16));
            // (Its stores too: a twin they would have gone to was let go above.)
            for (const gcn::ResourcePath& p : pl.meta.store_buffers) import_path(p, false, 16);
        } else {
            const ImportSite site("a dispatch's stores");
            for (const gcn::ResourcePath& p : lean->meta.store_buffers) import_path(p, false, 16);
        }
        for (const gcn::TranslateResult::BufferRead& r : walks.read_buffers) {
            import_path(r.path, r.pointer, r.pointer ? (r.max_dw ? r.max_dw * 4ull : 64ull << 10) : 16);
        }
        g_dispatch_read_imports.fetch_add(walks.read_buffers.size(), std::memory_order_relaxed);
        // A load or store whose base does not trace to user data reaches
        // memory nothing here imports: counted, the fallback's own reason.
        if (walks.untraced_reads || walks.untraced_stores) g_dispatch_untraced.fetch_add(1, std::memory_order_relaxed);
    }
    VkDescriptorBufferInfo ubi{};
    alloc_params_slot_locked(params, ubi);
    VkDescriptorSet set = alloc_set_locked();
    if (!set) {
        host_log("gpu: descriptor set allocation failed");
        g.failures.fetch_add(1);
        return false;
    }
    std::vector<VkWriteDescriptorSet> writes;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set;
    w.dstBinding = gcn::kBindingParams;
    w.descriptorCount = 1;
    w.descriptorType = params_descriptor_type();
    w.pBufferInfo = set_cache_on() ? &g_params_desc : &ubi;
    writes.push_back(w);
    std::vector<VkDescriptorImageInfo> infos;
    infos.reserve(pl.meta.images.size() + pl.meta.samplers.size() + 1);
    bind_stage_images(set, pl.meta, stage_images, writes, infos, pl.name.c_str());
    vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    // An indirect dispatch's arguments, checked on the GPU just before it
    // reads them (indirect_clamp.cpp): a group count past 65,535 runs nothing.
    VkBuffer args_buffer = VK_NULL_HANDLE;
    VkDeviceSize args_offset = 0;
    if (d.indirect_va) {
        if (g_import_audit) audit_range(d.indirect_va, 12, kUseIndex);
        const ImportSite site("a dispatch's indirect arguments");
        const Located loc = locate(d.indirect_va, 12);
        if (!loc.buffer || loc.avail < 12) {
            g.failures.fetch_add(1);
            return false;
        }
        args_buffer = loc.buffer;
        args_offset = loc.offset;
        if (!g_render_min) indirect_clamp_locked(loc.buffer, loc.offset, d.indirect_va, name, args_buffer, args_offset);
    }
    stamp.to(kDpSet);
    rec().bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, lean ? lean->pipeline : pl.pipeline);
    const std::uint32_t params_offset = static_cast<std::uint32_t>(ubi.offset);
    const VkPipelineLayout layout = lean ? g.cs_buffers_layout : g.pipe_layout;
    rec().bind_sets(VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, set_cache_on() ? 1 : 0, &params_offset);
    if (lean) {
        std::uint32_t bindings[gcn::kMaxBuffers];
        const std::uint32_t n = static_cast<std::uint32_t>(std::min<std::size_t>(lean->meta.buffers.size(), gcn::kMaxBuffers));
        for (std::uint32_t k = 0; k < n; ++k) bindings[k] = lean->meta.buffers[k].binding;
        rec().push_buffers(VK_PIPELINE_BIND_POINT_COMPUTE, layout, 1, n, bindings, lean_infos.data());
    }
    CapturePtr capture_session;
    if (capture) {
        CaptureDispatch cd;
        cd.pipeline = pl.name;
        cd.flip = hle_video_flip_count();
        cd.dispatch_index = g.dispatches.load();
        cd.meta = &pl.meta;
        cd.spirv = &pl.meta.spirv;
        cd.gcn_words = &words;
        cd.params = params;
        // The counters as the guest names them: a replay puts them on its own
        // pages (tools/drawreplay.cpp), where these device addresses mean nothing.
        for (std::uint32_t k = 0; k < GpuDispatch::kGdsCounters; ++k) cd.params.gds_counter[k] = gds_bound ? d.gds[k].word : 0;
        cd.images = &stage_images;
        std::memcpy(cd.dim, d.dim, sizeof(cd.dim));
        std::memcpy(cd.threads, d.threads, sizeof(cd.threads));
        cd.indirect_va = d.indirect_va;
        if (gds_bound) {
            for (const GpuDispatch::GdsCounter& c : d.gds) {
                if (c.word) cd.counters.push_back(c.word);
            }
        }
        cd.translate = json::Value::make_object();
        cd.translate.set("stage", "compute");
        cd.translate.set("rsrc1", json::hex(d.rsrc1));
        cd.translate.set("rsrc2", json::hex(d.rsrc2));
        cd.translate.set("gds_counters", host_gpu_gds_counters());
        cd.translate.set("gds_subgroup", host_gpu_gds_subgroup());
        cd.translate.set("exec_known", g.exec_known);
        cd.translate.set("max_lds_bytes", g.max_lds_bytes);
        cd.translate.set("lifted", pl.lifted);  // the captured module is the program's lift (gcn::lift_compute_shader)
        cd.translate.set("gds_by_address", true);  // its GDS counters by device address (params hold the guest words: drawreplay maps them)
        {
            // The image dimensions the pipeline was made with (with the texel
            // kind in bits 8-9), for a tool to rebuild its options (gcnlift).
            json::Value dj = json::Value::make_array();
            for (const auto& [dim, arrayed] : dims) {
                json::Value e = json::Value::make_array();
                e.push(dim);
                e.push(arrayed);
                dj.push(e);
            }
            cd.translate.set("image_dims", dj);
        }
        capture_session = capture_dispatch_begin_locked(std::move(cd));
    }
    profile_begin_locked(&pl.name);
    note_dispatch_name(g.dispatches.load(), pl.name.c_str());
    gpu_checkpoint(2, g.dispatches.load());
    if (d.indirect_va) {
        if (!g_render_min) rec().dispatch_indirect(args_buffer, args_offset);
        note_indirect_locked(d.indirect_va, 0, pl.meta.gds_counters != 0);
    } else {
        if (!g_render_min) rec().dispatch(d.dim[0], d.dim[1], d.dim[2]);
    }
    profile_end_locked();
    stamp.to(kDpRecord);
    // The dispatches by how their shader ran (BBHOST_LIFT_COMPUTE): per
    // pipeline for the exit report, and in all for the second's detail line.
    ++pl.dispatches;
    g.shader_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (pl.lifted) g.lifted_dispatches.fetch_add(1, std::memory_order_relaxed);
    // Buffers the shader writes: the buffer shadow's copies of them are stale
    // from here on. A store whose V# the host cannot find is logged; the
    // shadow's verify mode (BBHOST_SHADOW_VERIFY) finds what it changed.
    for (const gcn::ResourcePath& path : pl.meta.store_buffers) {
        std::uint32_t v[4]{};
        if (!resolve_resource(path, d.user_data, 4, v)) {
            if (g_unresolved_stores.fetch_add(1, std::memory_order_relaxed) < 8) {
                host_log("gpu: dispatch %s stores through %s, which did not resolve", pl.name.c_str(), path.str().c_str());
            }
            continue;
        }
        const std::uint64_t base = v[0] | (static_cast<std::uint64_t>(v[1] & 0xff) << 32);
        const std::uint32_t stride = (v[1] >> 16) & 0x3fff;
        const std::uint64_t bytes = stride ? std::uint64_t{v[2]} * stride : v[2];
        if (g_store_log && g_store_logs.fetch_add(1, std::memory_order_relaxed) < 400) {
            host_log("gpu: store %s %s -> 0x%llx %llu bytes (stride %u, records %u)", pl.name.c_str(), path.str().c_str(),
                     static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes), stride, v[2]);
        }
        // (A live twin takes the stores: guest memory stays as it was until
        // the twin is copied back, which notes its own write.)
        if (!twin_holds_live_locked(base, bytes)) shadow_written_locked(base, bytes, true);
    }
    if (pl.meta.untraced_stores && !pl.untraced_stores_logged) {
        pl.untraced_stores_logged = true;
        host_log("gpu: dispatch %s has %u store(s) through a V# not traced to user data", pl.name.c_str(), pl.meta.untraced_stores);
    }
    bool any_storage = false;
    for (const gcn::ImageBinding& b : pl.meta.images) any_storage |= b.storage;
    for (std::size_t k = 0; k < pl.meta.images.size(); ++k) {
        const gcn::ImageBinding& b = pl.meta.images[k];
        // What the dispatch bound (prefetch_stage_images), which follows T#s
        // loaded from memory too.
        if (k >= stage_images.images.size() || !stage_images.images[k].resolved) continue;
        const std::uint32_t* tw = stage_images.images[k].w;
        if (any_storage) {
            tex_event(tsharp_base(tw), texture_src_bytes_locked(tsharp_base(tw)), "dispatch %s %ux%ux%u %s 0x%llx dfmt %u nfmt %u %ux%u type %u tiling %u levels %u..%u%s", pl.name.c_str(),
                      d.dim[0], d.dim[1], d.dim[2], b.storage ? "WRITES" : "reads", static_cast<unsigned long long>(tsharp_base(tw)),
                      (tw[1] >> 20) & 0x3f, (tw[1] >> 26) & 0xf, (tw[2] & 0x3fff) + 1, ((tw[2] >> 14) & 0x3fff) + 1,
                      (tw[3] >> 28) & 0xf, (tw[3] >> 20) & 0x1f, (tw[3] >> 12) & 0xf, (tw[3] >> 16) & 0xf,
                      find_render_target(tsharp_base(tw)) ? " (render target)" : "");
        }
        if (b.storage) surface_queue_writeback(tsharp_base(tw));
        static const std::uint64_t probe = [] {
            const char* e = std::getenv("BBHOST_TEX_PROBE");
            return e ? std::strtoull(e, nullptr, 0) : 0ull;
        }();
        if (probe && tsharp_base(tw) == probe) {
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 24) {
                std::string bufs;
                for (const gcn::ResourcePath& path : pl.meta.store_buffers) bufs += " store " + path.str();
                for (std::size_t j = 0; j < pl.meta.buffers.size(); ++j) {
                    std::uint32_t v[4]{};
                    if (!resolve_resource(pl.meta.buffers[j].path, d.user_data, 4, v)) continue;
                    const std::uint64_t vb = v[0] | (static_cast<std::uint64_t>(v[1] & 0xff) << 32);
                    std::uint64_t head[2] = {};
                    if (vb && hle_kernel_va_mapped(vb, sizeof(head))) std::memcpy(head, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(vb)), sizeof(head));
                    char t[160];
                    std::snprintf(t, sizeof(t), "; buffer %s 0x%llx: %016llx %016llx", pl.meta.buffers[j].path.str().c_str(),
                                  static_cast<unsigned long long>(vb), static_cast<unsigned long long>(head[0]), static_cast<unsigned long long>(head[1]));
                    bufs += t;
                }
                {
                    char t[512];
                    const std::uint32_t* u = d.user_data;
                    const std::uint64_t vb = u[4] | (static_cast<std::uint64_t>(u[5] & 0xff) << 32);
                    std::uint8_t src[16] = {};
                    if (vb && hle_kernel_va_mapped(vb, sizeof(src))) std::memcpy(src, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(vb)), sizeof(src));
                    const std::uint64_t table = u[2] | (static_cast<std::uint64_t>(u[3]) << 32);
                    std::uint32_t cbv[4] = {}, cb[4] = {};
                    if (table && hle_kernel_va_mapped(table, sizeof(cbv))) std::memcpy(cbv, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(table)), sizeof(cbv));
                    const std::uint64_t cbb = cbv[0] | (static_cast<std::uint64_t>(cbv[1] & 0xff) << 32);
                    if (cbb && hle_kernel_va_mapped(cbb, sizeof(cb))) std::memcpy(cb, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cbb)), sizeof(cb));
                    std::snprintf(t, sizeof(t), "; user[4:7] %08x %08x %08x %08x, bytes at 0x%llx:", u[4], u[5], u[6], u[7], static_cast<unsigned long long>(vb));
                    bufs += t;
                    for (std::uint8_t b8 : src) {
                        std::snprintf(t, sizeof(t), " %02x", b8);
                        bufs += t;
                    }
                    std::snprintf(t, sizeof(t), "; cb 0x%llx: %u %u %u %u", static_cast<unsigned long long>(cbb), cb[0], cb[1], cb[2], cb[3]);
                    bufs += t;
                }
                host_log("texture probe: 0x%llx dispatch %s %ux%ux%u %s it, T# %08x %08x %08x %08x (flip %llu)%s",
                         static_cast<unsigned long long>(tsharp_base(tw)), pl.name.c_str(), d.dim[0], d.dim[1], d.dim[2], b.storage ? "WRITES" : "reads",
                         tw[0], tw[1], tw[2], tw[3], static_cast<unsigned long long>(hle_video_flip_count()), bufs.c_str());
            }
        }
    }
    if (gds_bound) {
        for (const GpuDispatch::GdsCounter& c : d.gds) {
            if (c.word) shadow_written_locked(c.word, 4, true);
        }
    }
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    // What follows may also take indirect arguments a shader wrote (the
    // particles' counter passes write the next dispatch's) or copy out of
    // memory it wrote (a counter, into the arguments) - and fetch vertices
    // from a buffer it wrote, or draw into or copy over an image it stored
    // (BBHOST_BARRIER_MASKS: every access; =0 the narrower set, which left
    // vertex fetch, attachments and transfer writes out).
    mb.dstAccessMask = barrier_masks_fixed() ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT
                                             : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDEX_READ_BIT |
                                                   VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    particle_args_after_dispatch_locked();  // the particles' draws' live counts, as this dispatch may have changed them
    g.dispatches.fetch_add(1);
    if (capture_session) capture_dispatch_finish_locked(std::move(capture_session));
    stamp.to(kDpAfter);
    return true;
}

void host_gpu_flush() {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (g.ok) {
        flush_locked();
    }
}

// BBHOST_GPU_PROFILE=1: the pipelines with the most GPU time since the
// last call, "name=ms(count)" for the top eight.
std::string host_gpu_profile_report() {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.profile) return "";
    std::vector<std::pair<std::string, std::pair<std::uint64_t, std::uint64_t>>> v(g.profile_ns.begin(), g.profile_ns.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    std::string out;
    std::uint64_t total = 0, graphics = 0, compute = 0;
    for (const auto& e : v) {
        total += e.second.first;
        // Graphics names are VS+PS, compute names are a single shader hash.
        // This classifies whole operations, not time spent in individual stages.
        (e.first.find('+') != std::string::npos ? graphics : compute) += e.second.first;
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf), " total=%llums cmdbuf=%llums(%llu)", static_cast<unsigned long long>(total / 1000000),
                  static_cast<unsigned long long>(g.cmdbuf_ns / 1000000), static_cast<unsigned long long>(g.cmdbuf_count));
    g.cmdbuf_ns = 0;
    g.cmdbuf_count = 0;
    out += buf;
    std::snprintf(buf, sizeof(buf), " graphics=%llums compute=%llums",
                  static_cast<unsigned long long>(graphics / 1000000),
                  static_cast<unsigned long long>(compute / 1000000));
    out += buf;
    // BBHOST_GPU_PROFILE_TOP=<n>: the top n rather than eight.
    static const std::size_t top = [] {
        const char* e = std::getenv("BBHOST_GPU_PROFILE_TOP");
        return e && *e ? static_cast<std::size_t>(std::max(1, std::atoi(e))) : std::size_t{8};
    }();
    for (std::size_t i = 0; i < v.size(); ++i) {
        // The top eight, and the host's own named work (rt-copy-*) wherever it ranks.
        if (i >= top && v[i].first.compare(0, 8, "rt-copy-") != 0) continue;
        std::snprintf(buf, sizeof(buf), " %s=%.1fms(%llu)", v[i].first.c_str(), v[i].second.first / 1e6,
                      static_cast<unsigned long long>(v[i].second.second));
        out += buf;
    }
    if (g.profile_passes) {
        // =3: every entry - the passes by their first draw's pipeline and
        // size, the dispatches, copies and uploads - on lines of their own,
        // six a line with their statistics (k per pair, as below): the top
        // line ends at 1 KiB, and a frame has ~45 passes.
        std::string line;
        int listed = 0;
        for (const auto& e : v) {
            char b[320];
            int k = std::snprintf(b, sizeof(b), " %s=%.3fms(%llu)", e.first.c_str(), e.second.first / 1e6,
                                  static_cast<unsigned long long>(e.second.second));
            const auto it = g.profile_stats_sum.find(e.first);
            if (k > 0 && k < static_cast<int>(sizeof(b)) && it != g.profile_stats_sum.end() && e.second.second) {
                const double n = static_cast<double>(e.second.second) * 1000.0;
                const auto& a = it->second;
                std::snprintf(b + k, sizeof(b) - k, "[%.1f/%.1f/%.1f/%.1f/%.1f/%.1f]", a[0] / n, a[2] / n, a[1] / n, a[3] / n, a[4] / n,
                              a[5] / n);
            }
            line += b;
            if (++listed % 6 == 0) {
                host_log("gpu profile all (since the last; k per pair: verts/vs/prims/clip-in/clip-out/fs):%s", line.c_str());
                line.clear();
            }
        }
        if (!line.empty()) host_log("gpu profile all (since the last; k per pair: verts/vs/prims/clip-in/clip-out/fs):%s", line.c_str());
    }
    if (g.profile_stats) {
        // The same entries' pipeline statistics, per pair, in thousands:
        // vertices in / vertex-shader invocations / primitives in /
        // primitives into clipping / out of it / fragment-shader invocations.
        // Eight entries a line, so a long list is not cut.
        std::string stats;
        int listed = 0;
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i >= top && v[i].first.compare(0, 8, "rt-copy-") != 0) continue;
            const auto it = g.profile_stats_sum.find(v[i].first);
            if (it == g.profile_stats_sum.end() || !v[i].second.second) continue;
            const double n = static_cast<double>(v[i].second.second) * 1000.0;
            const auto& a = it->second;
            char b[192];
            std::snprintf(b, sizeof(b), " %s=%.1f/%.1f/%.1f/%.1f/%.1f/%.1f", v[i].first.c_str(), a[0] / n, a[2] / n, a[1] / n, a[3] / n,
                          a[4] / n, a[5] / n);
            stats += b;
            if (++listed % 8 == 0) {
                host_log("gpu profile stats (k per pair: verts/vs/prims/clip-in/clip-out/fs):%s", stats.c_str());
                stats.clear();
            }
        }
        if (!stats.empty()) host_log("gpu profile stats (k per pair: verts/vs/prims/clip-in/clip-out/fs):%s", stats.c_str());
        g.profile_stats_sum.clear();
    }
    {
        // The game's compute programs (a name of eight hex digits) on lines of
        // their own, every one of them: the line above lists the top entries
        // and a log line ends at 1 KiB, so the cheap programs fell off it.
        std::string line;
        int listed = 0;
        for (const auto& e : v) {
            if (e.first.size() != 8 || e.first.find_first_not_of("0123456789abcdef") != std::string::npos) continue;
            std::snprintf(buf, sizeof(buf), " %s=%.2fms(%llu)", e.first.c_str(), e.second.first / 1e6, static_cast<unsigned long long>(e.second.second));
            line += buf;
            if (++listed % 16 == 0) {
                host_log("gpu profile compute (since the last):%s", line.c_str());
                line.clear();
            }
        }
        if (!line.empty()) host_log("gpu profile compute (since the last):%s", line.c_str());
    }
    {
        // The GPU particles' draws ("~particles", render.cpp) and the pass
        // that writes their arguments (particle-args), every one of them, on
        // lines of their own: the paths' time a draw (BBHOST_GX_PARTICLES).
        std::string line;
        int listed = 0;
        for (const auto& e : v) {
            if (e.first.find("~particles") == std::string::npos && e.first != "particle-args") continue;
            std::snprintf(buf, sizeof(buf), " %s=%.3fms(%llu)", e.first.c_str(), e.second.first / 1e6, static_cast<unsigned long long>(e.second.second));
            line += buf;
            if (++listed % 8 == 0) {
                host_log("gpu profile particles (since the last):%s", line.c_str());
                line.clear();
            }
        }
        if (!line.empty()) host_log("gpu profile particles (since the last):%s", line.c_str());
    }
    g.profile_ns.clear();
    if (g.profile_gaps) {
        // The gaps by what was recorded in them, the top twelve, as site
        // addresses (bbhost's own; tools/gpu_gaps.py names them).
        std::vector<std::pair<std::array<void*, 4>, std::pair<std::uint64_t, std::uint64_t>>> gv(g.profile_gap_ns.begin(), g.profile_gap_ns.end());
        std::sort(gv.begin(), gv.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
        std::uint64_t gaps = 0;
        for (const auto& e : gv) gaps += e.second.first;
        std::snprintf(buf, sizeof(buf), "\n[bbhost]   gpu gaps: %llums in all;", static_cast<unsigned long long>(gaps / 1000000));
        out += buf;
        for (std::size_t i = 0; i < gv.size() && i < 12; ++i) {
            out += " ";
            bool any = false;
            for (void* site : gv[i].first) {
                if (!site) continue;
                std::snprintf(buf, sizeof(buf), "%s%llx", any ? "+" : "",
                              static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(site) - host_image_base()));
                out += buf;
                any = true;
            }
            if (!any) out += "none";
            std::snprintf(buf, sizeof(buf), "=%.1fms(%llu)", gv[i].second.first / 1e6, static_cast<unsigned long long>(gv[i].second.second));
            out += buf;
        }
        g.profile_gap_ns.clear();
    }
    return out;
}

namespace {

// One save at a time: the periodic one's thread, or the one at exit.
std::atomic<bool> g_cache_saving{false};
std::size_t g_cache_size_hint = 0;  // the last serialized size; under g_cache_saving
bool g_cache_save_every = false;     // BBHOST_PIPELINE_CACHE_SAVE_S set

// Writes the file through a temporary one, 8 MiB at a time, each chunk on the
// device before the next is written (HostPacedFile). Written at once, the
// cache (313 MB) was seconds of write-back that other threads' file reads and
// log writes queued behind: a 2.9 s main-loop frame at boot and a 1 s stall in
// play, over NFS.
bool write_file_paced(const std::string& path, const char* data, std::size_t size) {
    HostPacedFile f(path, path + ".tmp");
    return f.write(data, size) && f.commit();
}

// Serializes the cache and writes it if it grew. The cache is ~230 MB, and
// vkGetPipelineCacheData serializes all of it on every call, the size query
// included; with a buffer sized from the last save it is usually one call.
// A periodic save that would add little (under 4 MiB or 2%) leaves it to a
// later one or the exit: the file is rewritten whole, 313 MB for the 2-39 KiB
// a soak's saves added.
void save_pipeline_cache_now(const std::string& path, bool periodic) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> data(g_cache_size_hint ? g_cache_size_hint + (g_cache_size_hint >> 3) + (1u << 20) : 0);
    std::size_t size = data.size();
    VkResult r = size ? vkGetPipelineCacheData(g.device, g.cache, &size, data.data()) : VK_INCOMPLETE;
    if (r == VK_INCOMPLETE) {  // grew past the guess, or no guess yet
        size = 0;
        if (vkGetPipelineCacheData(g.device, g.cache, &size, nullptr) != VK_SUCCESS || !size) return;
        data.resize(size);
        r = vkGetPipelineCacheData(g.device, g.cache, &size, data.data());
    }
    if (r != VK_SUCCESS) return;
    data.resize(size);
    g_cache_size_hint = size;
    const auto t1 = std::chrono::steady_clock::now();
    const auto ms = [](auto a, auto b) {
        return static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count());
    };
    // The driver appends what it compiles: pipelines that all came from the
    // cache leave its size alone, and there is nothing new to write.
    const std::size_t saved = g_pipeline_cache_bytes.load();
    if (size == saved) {
        host_log("gpu: pipeline cache unchanged, %zu KiB (serialized in %lld ms)", size >> 10, ms(t0, t1));
        return;
    }
    if (periodic && !g_cache_save_every && size < saved + std::max<std::size_t>(4u << 20, saved / 50)) {
        host_log("gpu: pipeline cache grew %lld KiB (serialized in %lld ms); left for a later save",
                 (static_cast<long long>(size) - static_cast<long long>(saved)) / 1024, ms(t0, t1));
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    if (write_file_paced(path, data.data(), data.size())) {
        g_pipeline_cache_bytes.store(data.size());
        host_log("gpu: pipeline cache saved, %zu KiB (serialized in %lld ms, written in %lld ms)", data.size() >> 10, ms(t0, t1),
                 ms(t1, std::chrono::steady_clock::now()));
    } else {
        host_log("gpu: saving the pipeline cache to %s failed", path.c_str());
    }
}

}  // namespace

// Called with the 300-flip report, on the game's render thread. Serializing
// the cache there held that thread ~100 ms every five seconds whenever a
// pipeline had been created - the command processor's backlog filled behind it
// and the main loop, waiting on the render thread, hitched. So the whole save
// runs on a thread of its own, at most every three minutes (serializing may
// hold the cache against the command processor's pipeline creation meanwhile),
// and host_gpu_save_pipeline_cache_at_exit() keeps what came after.
namespace {
// With g_cache_divert set and no creation left against the main cache (whose
// merge needs it to itself - Vulkan's dstCache is externally synchronized, the
// sources are not): the side cache's pipelines and the compute optimizer's
// (g.opt_cache) join it.
void merge_side_caches() {
    VkPipelineCache sources[2];
    std::uint32_t n = 0;
    if (g.side_cache != VK_NULL_HANDLE) sources[n++] = g.side_cache;
    if (g.opt_cache != VK_NULL_HANDLE) sources[n++] = g.opt_cache;
    std::lock_guard<std::mutex> lk(g_opt_cache_mu);  // not a destination meanwhile
    if (n) vkMergePipelineCaches(g.device, g.cache, n, sources);
}
}  // namespace

void host_gpu_save_pipeline_cache() {
    static std::uint64_t saved_pipelines = 0;
    static std::chrono::steady_clock::time_point last{};
    if (!g.ok || g.cache == VK_NULL_HANDLE) return;
    stage_manifest_save_async(stage_manifest_path());
    pipeline_manifest_save_async(pipeline_manifest_path());
    translation_cache_save_async(translation_cache_path());
    // Compute pipelines the optimizer built count: they join the cache at a save (merge_side_caches).
    const std::uint64_t pipelines = host_gpu_stats().pipelines + compute_optimized_built();
    if (pipelines == saved_pipelines) return;
    const auto now = std::chrono::steady_clock::now();
    // BBHOST_PIPELINE_CACHE_SAVE_S: the interval, and every save that finds the
    // cache grown writes it - for testing saves under load.
    static const std::chrono::seconds interval{[] {
        const char* e = std::getenv("BBHOST_PIPELINE_CACHE_SAVE_S");
        g_cache_save_every = e && std::atoi(e) > 0;
        return g_cache_save_every ? std::atoi(e) : 180;
    }()};
    if (last != std::chrono::steady_clock::time_point{} && now - last < interval) return;
    std::string path = pipeline_cache_path();
    if (path.empty() || g_cache_saving.exchange(true)) return;
    saved_pipelines = pipelines;
    last = now;
    std::thread([path = std::move(path)] {
        const std::uint64_t diverted = g_cache_diverted.load();
        g_cache_divert.store(true);  // creations go to the side cache while the main one serializes
        save_pipeline_cache_now(path, true);
        // What was created meanwhile joins the main cache, which the merge
        // needs to itself: the creations that took it before the divert end first.
        while (g_cache_users.load() != 0) host_sleep_us(200);
        merge_side_caches();
        g_cache_divert.store(false);
        if (const std::uint64_t n = g_cache_diverted.load() - diverted) {
            host_log("gpu: %llu pipelines created against the side cache while the cache saved", static_cast<unsigned long long>(n));
        }
        g_cache_saving.store(false);
    }).detach();
}

void host_gpu_save_pipeline_cache_at_exit() {
    if (!g.ok || g.cache == VK_NULL_HANDLE) return;
    const std::string path = pipeline_cache_path();
    if (path.empty()) return;
    while (g_cache_saving.exchange(true)) std::this_thread::sleep_for(std::chrono::milliseconds(10));  // a periodic save first
    // What the side cache and the compute optimizer's own cache took since the
    // last periodic save joins first - the optimized compute pipelines built
    // late in a run would otherwise be built again at the next start. The merge
    // needs the main cache to itself: when a pipeline still compiling against
    // it does not end within a second, they are left for the next run to build.
    {
        g_cache_divert.store(true);
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (g_cache_users.load() != 0 && std::chrono::steady_clock::now() < until) host_sleep_us(200);
        if (g_cache_users.load() == 0) merge_side_caches();
        else host_log("gpu: pipeline cache: a pipeline still compiling at exit; what the side caches hold is left out of the save");
        g_cache_divert.store(false);
    }
    save_pipeline_cache_now(path, false);  // the flag stays set: no save after this one
    stage_manifest_save(stage_manifest_path());
    pipeline_manifest_save(pipeline_manifest_path());
    translation_cache_save(translation_cache_path(), true);
}

std::string host_gpu_compile_report() {
    std::string out = compute_optimize_window();
    const std::string tc = translation_cache_window();
    if (!tc.empty()) out += (out.empty() ? "" : "; ") + tc;
    return out;
}

void host_gpu_submit() {
    textures_prewait_for_submit();  // the texture untiling it copies, waited for without the mutex
    std::lock_guard<GpuMutex> lock(g.mu);
    if (g.ok) {
        submit_locked();
    }
}

namespace {

// Guest-memory transfers recorded in command order: the CP's label writes
// and DMA land after the shader work queued before them, as on hardware.
// Returns false when the range is not imported (caller writes it itself).
bool locate_range(std::uint64_t va, std::size_t bytes, Located& loc) {
    if (bytes == 0 || (va & 3) || (bytes & 3)) return false;
    if (!hle_kernel_va_mapped(va, bytes)) return false;
    if (!rebuild_page_tables()) return false;
    if (g_import_audit) audit_range(va, bytes, kUseTransfer);
    loc = locate(va, bytes);
    return loc.buffer != VK_NULL_HANDLE && loc.avail >= bytes;
}

// A guest-memory transfer (a label write, a CP DMA) has to leave the render
// pass and fence itself against the shader work it follows. Doing that per
// write cost three full pipeline barriers and a render-pass restart each, and
// the world load does about five hundred of them a frame. Consecutive
// transfers now share one barrier pair: the opening barrier is emitted once
// and the closing one is deferred to transfer_flush_locked(), which every
// path that records other work calls first.
// The buffer bytes the open transfer batch has read and written. Its copies
// and fills have no barrier between them, so they run in any order: one that
// reads bytes an earlier one wrote, or writes bytes an earlier one read or
// wrote, waits for the batch so far (transfer_order_locked). Two copy tokens
// to one address in a batch were the validation layer's WRITE_AFTER_WRITE,
// and the older copy could land last.
struct BatchSpan {
    VkBuffer buffer;
    VkDeviceSize lo, hi;
    bool write;
};
std::vector<BatchSpan> g_batch_spans;  // under g.mu
std::atomic<std::uint64_t> g_batch_orders{0};

void transfer_begin_locked() {
    begin_recording_locked();
    if (g.in_transfer) return;
    render_end_pass_locked();
    copy_versions_flush_locked();  // copy-backs still in their copies (BBHOST_COPY_VERSIONS=2) land before this batch
    // Into the open packet when it can take it (a copy token's own).
    DrawCmds* c = DrawCmds::open();
    if (!c || !c->transfer_begin()) rec().transfer_barrier(true);
    g.in_transfer = true;
    g_batch_spans.clear();
    bump(g.transfer_batches);
}

// Before a transfer into the open batch that reads src [src_off, +bytes) (no
// src: a fill) and writes dst [dst_off, +bytes). `in_place`: the caller
// records it through rec(), not into the open packet; the barrier goes there too.
void transfer_order_locked(VkBuffer src, VkDeviceSize src_off, VkBuffer dst, VkDeviceSize dst_off, VkDeviceSize bytes,
                           bool in_place = false) {
    bool wait = false;
    for (const BatchSpan& s : g_batch_spans) {
        if ((s.buffer == dst && s.lo < dst_off + bytes && dst_off < s.hi) ||
            (s.write && src && s.buffer == src && s.lo < src_off + bytes && src_off < s.hi)) {
            wait = true;
            break;
        }
    }
    if (wait) {
        DrawCmds* c = in_place ? nullptr : DrawCmds::open();
        if (!c || !c->copy_order()) rec().copy_order_barrier();
        g_batch_spans.clear();
        g_batch_orders.fetch_add(1, std::memory_order_relaxed);
    }
    if (src) g_batch_spans.push_back({src, src_off, src_off + bytes, false});
    g_batch_spans.push_back({dst, dst_off, dst_off + bytes, true});
}

void transfer_end_locked() { g.transfers.fetch_add(1); }

}  // namespace

namespace gpu {

// ---- GX buffers the GPU writes, as device resources (BBHOST_GX_TWINS=1) ----
// A GX buffer the GPU writes - a particle system's state, an append buffer
// and its counter, indirect arguments a dispatch fills: type buffer, usage
// default (no CPU access, D3D11's rule; the 12-place census: 0 Maps of them),
// unordered-access bind - lived in imported guest memory, so every dispatch
// wrote it and every draw read it across the bus on a discrete GPU, and its
// pages had to stay imported (phase B's census: the in-place users left once
// copies need no import). Such a buffer gets a device-local twin, keyed by
// its memory and the registry's generation (gx_resources: a resource made
// later over the same memory is another), made and filled from guest memory
// on its first GPU use; from then on locate() and guest_device_address()
// hand out the twin, and the guest bytes go stale - so every CPU copy of
// them is refused (twin_covers_locked: constant-buffer snapshots, the buffer
// shadow's mirrors, snapshots and stages, copy versions' sources, the index
// scan). What cannot use the twin lets it go after copying it back: a range
// across its edge, a page-table walk over it. A twin whose resource died
// (the registry no longer has that generation there) is dropped. Off by
// default while it is measured; the exit report counts all of it.
//
// BBHOST_GX_TWINS=2 adds the buffers GX's GPU copies write: a dynamic buffer's
// renamed bytes written back to its own address when a context's map
// resolves, a static (constant) buffer's UpdateSubresource - their own
// memory is written by those copies alone (the CPU writes the renamed copy;
// `memory moved by a map` 0), and read in place by the draws after them (a
// world: ~3,800 constant buffers, ~2.8 MiB, and ~90 dynamic ones). Each
// becomes a candidate at its first copy (twin_note_copy_target_locked), and
// copy versions put their bytes into the twin. A buffer a Map hands its own
// memory to (the CPU writes it in place) is never one.
namespace {
// The level: BBHOST_GX_TWINS=1 or 2 the broad versions above; 3 the GPU
// particles' buffers alone (BBHOST_PARTICLE_TWINS, below); 0 none
// (BBHOST_GX_TWINS=0 or BBHOST_PARTICLE_TWINS=0: guest memory serves every
// buffer, as before twins); -1 until the device is known (twins_decide).
//
// BBHOST_PARTICLE_TWINS (unset: on for a discrete GPU; 1 on any; 0 off):
// the buffers a dispatch that counts in GDS binds - the GPU particles'
// records, append and consume buffers with their counter words at the end,
// and whatever else it reads and stores - become device-local twins. In
// guest memory, across the bus on a discrete GPU, the particle update read
// and wrote its 176-byte records at about 0.2 us a particle: 8 ms of an RTX
// 4070 a frame in Vicar Amelia's room. A candidate is made unarmed; the
// first counting dispatch that binds it wants it, and it is armed when the
// next recording begins (twins_arm_wanted_locked) - so, as with the broad
// version, its first use in a recording is the one that makes its twin,
// filled before that recording's work, and nothing earlier in the recording
// wrote guest memory there. An integrated GPU's device-local memory is the
// host's own, so there it stays off; so it does under a capture
// (BBHOST_CAPTURE_DISPATCH, BBHOST_CAPTURE_DRAW), which reads its inputs
// from guest memory.
std::atomic<int> g_twins{[] {
    const char* e = std::getenv("BBHOST_GX_TWINS");
    if (e && (e[0] == '1' || e[0] == '2')) return e[0] - '0';
    if (e && e[0] == '0') return 0;
    const char* p = std::getenv("BBHOST_PARTICLE_TWINS");
    if (p && p[0] == '0') return 0;
    return -1;
}()};
struct TwinCandidate {
    std::uint64_t bytes = 0, gen = 0;
    bool let_go = false;  // copied back and let go: guest memory serves it from then on
    bool armed = true;    // a twin may be made (level 3: once a counting dispatch has wanted it)
};
std::vector<std::uint64_t> g_twin_want;  // level 3: candidates' memory to arm when the next recording begins, under g.mu
std::atomic<std::uint64_t> g_twin_armed{0};
// Level 3: the armed candidates' ranges (memory -> end), under g.mu - a filter
// in front of the candidates' lock, so a binding that meets none of them (most
// of a frame's) costs one lookup in a short map. It holds every armed
// candidate's range, perhaps more (one let go or gone since). Its inserts and
// erases move its shape, as the live twins' do: the miss filter's rebuild.
twins::ShapedMap<std::uint64_t> g_twin_armed_ranges;
bool twin_armed_meets_locked(std::uint64_t va, std::uint64_t bytes) { return twins::ranges_meet(g_twin_armed_ranges, va, bytes); }
// Its range out of the filter, unless an armed candidate (a newer resource's)
// still holds that memory. Not under g_twin_cand_mu (it takes it).
void twin_armed_forget_locked(std::uint64_t memory);
std::mutex g_twin_cand_mu;  // GX's threads add candidates; the renderer reads them (lock order: g.mu, then this)
std::map<std::uint64_t, TwinCandidate> g_twin_cand;  // by memory, under g_twin_cand_mu
std::unordered_set<std::uint64_t> g_twin_cpu_mapped;  // generations a Map handed their own memory, under g_twin_cand_mu
std::atomic<std::size_t> g_twin_cand_n{0};
// A twin's place: a slice of a pool block (block >= 0), or a buffer of its own.
struct Twin {
    std::uint64_t bytes = 0, gen = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0, size = 0;
    VkDeviceAddress address = 0;
    int block = -1;
    DevBuffer own;
};
twins::ShapedMap<Twin> g_twin_live;  // by memory, under g.mu (inserts and erases move its shape: twin_filter.h)
enum { kTwinGoneWalk, kTwinGoneEdge, kTwinGoneDead, kTwinGoneFill, kTwinGoneTexture, kTwinGoneCpuMapped, kTwinGoneWhys };
std::atomic<std::uint64_t> g_twin_cands{0}, g_twin_made{0}, g_twin_made_bytes{0}, g_twin_served{0}, g_twin_addresses{0},
    g_twin_refused{0}, g_twin_partial{0}, g_twin_edge_in_pass{0}, g_twin_edge_ended_pass{0}, g_twin_gone[kTwinGoneWhys], g_twin_copy_cands{0},
    g_twin_cpu_mapped_n{0};
std::uint64_t g_twin_live_bytes = 0;  // under g.mu

// Twins are placed in pool blocks - device-local buffers of kTwinBlockBytes,
// first fit at 256-byte steps - so a world's thousands of small constant
// buffers take no allocation each; one bigger than kTwinOwnBytes gets a
// buffer of its own. A dropped twin's slice is free again once the work
// recorded with it has run (the recording it was dropped in has completed).
constexpr VkDeviceSize kTwinBlockBytes = 4u << 20, kTwinOwnBytes = 1u << 20, kTwinAlign = 256;
struct TwinBlock {
    DevBuffer buf;
    std::map<VkDeviceSize, VkDeviceSize> free;  // offset -> size: disjoint, merged with their neighbours
};
std::vector<TwinBlock> g_twin_blocks;  // under g.mu
struct DeadTwinSlice {
    std::uint64_t serial;
    int block;
    VkDeviceSize offset, size;
};
std::deque<DeadTwinSlice> g_twin_dead;  // under g.mu, oldest first

void twin_slice_free_locked(int block, VkDeviceSize offset, VkDeviceSize size) {
    auto& fr = g_twin_blocks[static_cast<std::size_t>(block)].free;
    auto it = fr.emplace(offset, size).first;
    if (const auto nx = std::next(it); nx != fr.end() && it->first + it->second == nx->first) {
        it->second += nx->second;
        fr.erase(nx);
    }
    if (it != fr.begin()) {
        if (const auto pv = std::prev(it); pv->first + pv->second == it->first) {
            pv->second += it->second;
            fr.erase(it);
        }
    }
}

bool twin_place_locked(VkDeviceSize bytes, Twin& t) {
    const VkDeviceSize need = (bytes + kTwinAlign - 1) & ~(kTwinAlign - 1);
    if (need > kTwinOwnBytes) {
        MemOwner own_mem("particle twins");
        if (!create_dev_buffer(t.own, need, false)) return false;
        t.buffer = t.own.buffer, t.offset = 0, t.size = t.own.size, t.address = t.own.address, t.block = -1;
        return true;
    }
    while (!g_twin_dead.empty() && g_twin_dead.front().serial < g.completed_submits) {
        const DeadTwinSlice& d = g_twin_dead.front();
        twin_slice_free_locked(d.block, d.offset, d.size);
        g_twin_dead.pop_front();
    }
    for (int round = 0; round < 2; ++round) {
        for (std::size_t b = 0; b < g_twin_blocks.size(); ++b) {
            auto& fr = g_twin_blocks[b].free;
            for (auto it = fr.begin(); it != fr.end(); ++it) {
                if (it->second < need) continue;
                const VkDeviceSize off = it->first, rest = it->second - need;
                fr.erase(it);
                if (rest) fr.emplace(off + need, rest);
                const DevBuffer& bb = g_twin_blocks[b].buf;
                t.buffer = bb.buffer, t.offset = off, t.size = need, t.address = bb.address + off, t.block = static_cast<int>(b);
                return true;
            }
        }
        if (round) break;
        TwinBlock nb;
        MemOwner own_mem("particle twins");
        if (!create_dev_buffer(nb.buf, kTwinBlockBytes, false)) return false;
        nb.free.emplace(0, nb.buf.size);
        g_twin_blocks.push_back(std::move(nb));
    }
    return false;
}

void twin_unplace_locked(Twin& t) {
    if (t.block < 0) {
        defer_destroy(t.own);  // after the work recorded with it has run
    } else {
        g_twin_dead.push_back({g.flushes, t.block, t.offset, t.size});
    }
    t.buffer = VK_NULL_HANDLE;
}

// The copy of `bytes` from (src, so) to (dst, do) at this point in the
// stream, as the copy tokens' transfers go (the batch's barriers order it).
bool twin_copy_locked(VkBuffer src, VkDeviceSize so, VkBuffer dst, VkDeviceSize dso, VkDeviceSize bytes) {
    if (!src || !dst || !bytes) return false;
    transfer_begin_locked();
    transfer_order_locked(src, so, dst, dso, bytes);
    const VkBufferCopy region{so, dso, bytes};
    DrawCmds* c = DrawCmds::open();
    if (!c || !c->copy_buffer(src, dst, region)) rec().copy_buffer(src, dst, 1, &region);
    transfer_end_locked();
    return true;
}

// A new twin's bytes from guest memory, in the recording's uploads (the
// buffer shadow's command buffer, run before the recording's own work: the
// first GPU use of the resource is the one that asked, so nothing earlier in
// the recording wrote it), never inside a render pass. With window imports
// the bytes come through upload staging, unless a GPU write is still to
// land on them (then from the import, after that write in queue order).
bool twin_fill_locked(std::uint64_t memory, std::uint64_t bytes, VkBuffer twin, VkDeviceSize to) {
    begin_recording_locked();
    const VkCommandBuffer cmd = shadow_upload_cmd_locked();
    if (!cmd) return false;
    VkBuffer src = VK_NULL_HANDLE;
    VkDeviceSize so = 0;
    if (!imports_whole() && !gpu_writes_pending_locked(memory, bytes) && hle_kernel_va_mapped(memory, bytes)) {
        DevBuffer sb;
        if (acquire_staging_locked(sb, bytes, so)) {
            std::memcpy(sb.map, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(memory)), bytes);  // sb.map: the placement's own
            src = sb.buffer;
        }
    }
    if (!src) {
        const Located guest = locate_guest(memory, bytes);
        if (!guest.buffer || guest.avail < bytes) return false;
        src = guest.buffer;
        so = guest.offset;
    }
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;  // earlier submissions' writes of the source
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    const VkBufferCopy region{so, to, bytes};
    vkCmdCopyBuffer(cmd, src, twin, 1, &region);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;  // ... before every use of the twin in the recording
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    return true;
}

// Drops the twin at `memory` (copied back into guest memory first when
// `copy_back`), and marks its candidate let go (a later use reads guest
// memory) - or, its resource dead (`why` kTwinGoneDead, generation `gen`),
// forgets the candidate unless a newer resource's has taken its place.
void twin_drop_locked(std::uint64_t memory, bool copy_back, int why, std::uint64_t gen = 0) {
    auto it = g_twin_live.find(memory);
    if (it != g_twin_live.end()) {
        Twin& t = it->second;
        // Copy versions still to be put into it go first (their copies name its slice).
        if (copy_versions_into_locked(t.buffer, t.offset, t.offset + t.size)) copy_versions_put_in_place_locked(kCvPutTwin);
        if (copy_back) {
            const Located guest = locate_guest(memory, t.bytes);
            if (guest.buffer && guest.avail >= t.bytes && twin_copy_locked(t.buffer, t.offset, guest.buffer, guest.offset, t.bytes)) {
                shadow_written_locked(memory, t.bytes, true);  // lands on the GPU's timeline
            }
        }
        g_twin_live_bytes -= t.size;
        twin_unplace_locked(t);
        g_twin_live.erase(it);
    }
    g_twin_gone[why].fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(g_twin_cand_mu);
        if (auto c = g_twin_cand.find(memory); c != g_twin_cand.end()) {
            if (why == kTwinGoneDead) {
                if (c->second.gen == gen) {
                    g_twin_cand.erase(c);
                    g_twin_cand_n.store(g_twin_cand.size(), std::memory_order_relaxed);
                }
            } else {
                c->second.let_go = true;
            }
        }
    }
    twin_armed_forget_locked(memory);
}
void twin_armed_forget_locked(std::uint64_t memory) {
    const auto r = g_twin_armed_ranges.find(memory);
    if (r == g_twin_armed_ranges.end()) return;
    {
        std::lock_guard<std::mutex> lk(g_twin_cand_mu);
        if (const auto c = g_twin_cand.find(memory); c != g_twin_cand.end() && c->second.armed && !c->second.let_go) return;
    }
    g_twin_armed_ranges.erase(r);
}
// Live twins of buffers a Map has since handed their own memory (the CPU
// writes them in place): dropped without a copy back, which would overwrite
// the CPU's bytes. Checked when the count of such Maps has moved.
std::uint64_t g_twin_cpu_mapped_seen = 0;  // under g.mu
// GX buffers destroyed (the registry's host_gpu_gx_buffer_gone, or a buffer
// made over a candidate's memory), whose twins the renderer drops: a live
// twin is valid until then, so its uses ask the registry nothing.
std::vector<std::pair<std::uint64_t, std::uint64_t>> g_twin_gone_q;  // (memory, generation), under g_twin_cand_mu
std::atomic<std::uint64_t> g_twin_gone_n{0};
std::uint64_t g_twin_gone_seen = 0;  // under g.mu
void twins_drain_gone_locked() {
    if (g_twin_gone_n.load(std::memory_order_acquire) == g_twin_gone_seen) return;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> q;
    {
        std::lock_guard<std::mutex> lk(g_twin_cand_mu);
        q.swap(g_twin_gone_q);
        g_twin_gone_seen = g_twin_gone_n.load(std::memory_order_relaxed);
    }
    for (const auto& [memory, gen] : q) {
        if (const auto it = g_twin_live.find(memory); it != g_twin_live.end() && it->second.gen == gen) twin_drop_locked(memory, false, kTwinGoneDead, gen);
        twin_armed_forget_locked(memory);
    }
}
void twins_drop_cpu_mapped_locked() {
    const std::uint64_t n = g_twin_cpu_mapped_n.load(std::memory_order_relaxed);
    if (n == g_twin_cpu_mapped_seen || g_twin_live.empty()) return;
    g_twin_cpu_mapped_seen = n;
    std::vector<std::uint64_t> gone;
    {
        std::lock_guard<std::mutex> lk(g_twin_cand_mu);
        for (const auto& [memory, t] : g_twin_live) {
            const auto c = g_twin_cand.find(memory);
            if (c != g_twin_cand.end() && c->second.gen == t.gen && c->second.let_go && g_twin_cpu_mapped.count(t.gen)) gone.push_back(memory);
        }
    }
    for (const std::uint64_t memory : gone) twin_drop_locked(memory, false, kTwinGoneCpuMapped);
}
// A drop the next lookup must make first: a gone resource's twin
// (twins_drain_gone_locked) or a CPU-mapped one (twins_drop_cpu_mapped_locked)
// - exactly when either of those does any work.
bool twins_drop_pending_locked() {
    return g_twin_gone_n.load(std::memory_order_acquire) != g_twin_gone_seen ||
           (g_twin_cpu_mapped_n.load(std::memory_order_relaxed) != g_twin_cpu_mapped_seen && !g_twin_live.empty());
}
// The miss filter (twin_filter.h): the live twins' and armed ranges' union,
// under g.mu. A lookup that meets none of it has the answer the maps would
// give it - no live twin, refused by the armed filter - without walking them.
// BBHOST_TWIN_FAST_MISS=0: every lookup walks the maps, as before; =check:
// both, each "no" of the filter checked against the maps' (the exit report
// counts any that differ).
enum { kFastMissOff, kFastMissOn, kFastMissCheck };
const int g_twin_fast_miss = [] {
    const char* e = std::getenv("BBHOST_TWIN_FAST_MISS");
    if (e && e[0] == '0') return int{kFastMissOff};
    if (e && std::strcmp(e, "check") == 0) return int{kFastMissCheck};
    return int{kFastMissOn};
}();
twins::MissFilter g_twin_filter;
std::uint64_t g_twin_filter_misses = 0, g_twin_filter_checked = 0, g_twin_filter_wrong = 0;  // under g.mu
// [va, va + bytes) meets no live twin and no armed range (bytes > 0).
bool twin_filter_misses_locked(std::uint64_t va, std::uint64_t bytes) {
    if (g_twin_fast_miss == kFastMissOff) return false;
    // A run part's draw (BBHOST_RUN_COMMIT): the filter made current at the
    // fork (twins_settled_locked), and nothing counted.
    const bool part = run_part();
    if (!part) g_twin_filter.refresh(g_twin_live, g_twin_armed_ranges);
    if (g_twin_filter.meets(va, bytes)) return false;
    if (part) return true;
    ++g_twin_filter_misses;
    if (g_twin_fast_miss == kFastMissCheck) {
        ++g_twin_filter_checked;
        if (twins::live_meets(g_twin_live, va, bytes) || twins::ranges_meet(g_twin_armed_ranges, va, bytes)) {
            if (++g_twin_filter_wrong <= 8) {
                host_log("gpu: twins: the miss filter said 0x%llx + 0x%llx meets nothing, and the maps disagree (%zu spans, %zu live, %zu armed)",
                         static_cast<unsigned long long>(va), static_cast<unsigned long long>(bytes), g_twin_filter.spans(), g_twin_live.size(),
                         g_twin_armed_ranges.size());
            }
            return false;  // the maps' answer, as without the filter
        }
    }
    return true;
}
}  // namespace

bool twins_on() { return g_twins.load(std::memory_order_relaxed) != 0; }
bool twins_settled_locked() {
    if (!g_twins) return true;
    if (twins_drop_pending_locked()) return false;
    if (g_twin_fast_miss != kFastMissOff) g_twin_filter.refresh(g_twin_live, g_twin_armed_ranges);
    return true;
}
bool import_census_on() { return g_census; }
int twins_level() { return g_twins.load(std::memory_order_relaxed); }

void twins_decide(bool discrete) {
    int undecided = -1;
    const char* p = std::getenv("BBHOST_PARTICLE_TWINS");
    const char* cap_d = std::getenv("BBHOST_CAPTURE_DISPATCH");
    const char* cap_w = std::getenv("BBHOST_CAPTURE_DRAW");
    const bool capture = (cap_d && cap_d[0]) || (cap_w && cap_w[0]);
    const bool on = !capture && (p && p[0] == '1' ? true : discrete);
    if (!g_twins.compare_exchange_strong(undecided, on ? 3 : 0)) return;  // set already (BBHOST_GX_TWINS, or =0)
    host_log("gpu: the GPU particles' buffers %s (BBHOST_PARTICLE_TWINS%s)",
             on ? "as device-local twins once a dispatch that counts binds them" : "in guest memory",
             p && p[0] ? (p[0] == '1' ? "=1" : "=0") : capture ? ": off under a capture" : discrete ? ": on for a discrete GPU" : ": off on this GPU");
}

void twins_want_locked(std::uint64_t va) {
    if (g_twins.load(std::memory_order_relaxed) != 3 || !va || !g_twin_cand_n.load(std::memory_order_relaxed)) return;
    std::uint64_t memory = 0;
    {
        std::lock_guard<std::mutex> lk(g_twin_cand_mu);
        auto it = g_twin_cand.upper_bound(va);
        if (it == g_twin_cand.begin()) return;
        --it;
        if (va >= it->first + it->second.bytes || it->second.armed || it->second.let_go) return;
        memory = it->first;
    }
    if (std::find(g_twin_want.begin(), g_twin_want.end(), memory) == g_twin_want.end()) g_twin_want.push_back(memory);
}

void twins_arm_wanted_locked() {
    if (g_twin_want.empty()) return;
    std::lock_guard<std::mutex> lk(g_twin_cand_mu);
    for (const std::uint64_t memory : g_twin_want) {
        if (auto c = g_twin_cand.find(memory); c != g_twin_cand.end() && !c->second.armed && !c->second.let_go) {
            c->second.armed = true;
            g_twin_armed_ranges.assign(memory, memory + c->second.bytes);
            g_twin_armed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    g_twin_want.clear();
}

bool twin_locate_locked(std::uint64_t va, std::uint64_t bytes, Located& out, VkDeviceAddress* address) {
    if (!g_twins || !bytes || !g_twin_cand_n.load(std::memory_order_relaxed)) return false;
    // Level 3 (the particle twins), no drop pending: a range that meets no live
    // twin and no armed range is refused below by the armed filter, after both
    // walks - answered here from the filter instead. A run part's draw
    // (BBHOST_RUN_COMMIT) owes no drop: the fork settled them, and one owed
    // since is for a buffer let go after this run was submitted.
    const bool part = run_part();
    if (g_twins.load(std::memory_order_relaxed) == 3 && (part || !twins_drop_pending_locked()) && twin_filter_misses_locked(va, bytes))
        return false;
    if (!part) {
        twins_drain_gone_locked();
        twins_drop_cpu_mapped_locked();
    }
    // Inside a live twin (the common case): served without the candidates'
    // lock or the registry - a twin lives until its buffer is gone, let go or
    // mapped in place by the CPU, which drop it.
    if (auto lt = g_twin_live.upper_bound(va); lt != g_twin_live.begin()) {
        --lt;
        const std::uint64_t m = lt->first;
        if (va >= m && va + bytes <= m + lt->second.bytes) {
            out.buffer = lt->second.buffer;
            out.offset = lt->second.offset + (va - m);
            out.avail = lt->second.bytes - (va - m);
            if (address) {
                *address = lt->second.address + (va - m);
                g_twin_addresses.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_twin_served.fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        }
    }
    if (g_twins.load(std::memory_order_relaxed) == 3 && !twin_armed_meets_locked(va, bytes)) return false;
    // A candidate's twin made, let go or dropped, a pass ended: in order.
    if (run_part_stop(kRunStopBuffers)) return false;
    std::uint64_t memory = 0, cbytes = 0, gen = 0;
    std::uint64_t edge[2] = {0, 0};  // twins a range across their edge must let go first
    int n_edge = 0;
    bool inside = false;
    {
        std::lock_guard<std::mutex> lk(g_twin_cand_mu);
        // The candidate holding va, and any starting inside the range after it.
        auto it = g_twin_cand.upper_bound(va);
        if (it != g_twin_cand.begin()) {
            auto prev = std::prev(it);
            if (va < prev->first + prev->second.bytes && !prev->second.let_go && prev->second.armed) {
                if (va + bytes <= prev->first + prev->second.bytes) {
                    inside = true;
                    memory = prev->first;
                    cbytes = prev->second.bytes;
                    gen = prev->second.gen;
                } else {
                    edge[n_edge++] = prev->first;
                }
            }
        }
        for (; it != g_twin_cand.end() && it->first < va + bytes && n_edge < 2; ++it) {
            if (!it->second.let_go && it->second.armed) edge[n_edge++] = it->first;
        }
    }
    if (n_edge) {
        // A range across a twin's edge: the twin cannot serve it whole, and
        // guest memory may not be read past a live twin - copied back and let
        // go. With a render pass open, one live twin holding the range's start
        // serves what it holds (the rest reads past it: a V# with more records
        // than its buffer); anything else ends the pass for a binding (below)
        // and, for a device address, is counted and read from guest memory as
        // it stands.
        const bool pass = render_pass_open_locked();
        if (pass && n_edge == 1 && edge[0] <= va && g_twin_live.count(edge[0])) {
            std::lock_guard<std::mutex> lk(g_twin_cand_mu);
            const auto c = g_twin_cand.find(edge[0]);
            if (c != g_twin_cand.end()) {
                memory = edge[0];
                cbytes = c->second.bytes;
                gen = c->second.gen;
                inside = true;
                g_twin_partial.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<int> logs{0};
                if (logs.fetch_add(1, std::memory_order_relaxed) < 12) {
                    GxResourceInfo info;
                    const bool gx = gx_resource_at(memory, static_cast<std::size_t>(cbytes), &info);
                    host_log("gpu: twins: %s asks for 0x%llx + 0x%llx, past the end of a twin at 0x%llx + 0x%llx (usage %d, bind 0x%x) inside a "
                             "render pass: the twin serves what it holds",
                             t_import_site ? t_import_site : "a path", static_cast<unsigned long long>(va), static_cast<unsigned long long>(bytes),
                             static_cast<unsigned long long>(memory), static_cast<unsigned long long>(cbytes), gx ? info.usage : -1, gx ? info.bind : 0);
                }
            }
        }
        if (!inside) {
            // A binding (no device address asked) may end the pass, as a
            // binding across a copy version's edge does: the twins are copied
            // back and let go before the draw, so guest memory is current.
            const bool end_pass = pass && !address;
            bool ended = false;
            for (int k = 0; k < n_edge; ++k) {
                if (const auto lt = g_twin_live.find(edge[k]); lt != g_twin_live.end()) {
                    // Which path asks across an edge, and how (logged a few times).
                    static std::atomic<int> logs{0};
                    if (logs.fetch_add(1, std::memory_order_relaxed) < 12) {
                        host_log("gpu: twins: %s asks for 0x%llx + 0x%llx across the edge of a twin at 0x%llx + 0x%llx%s", t_import_site ? t_import_site : "a path",
                                 static_cast<unsigned long long>(va), static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(edge[k]),
                                 static_cast<unsigned long long>(lt->second.bytes),
                                 end_pass ? " inside a render pass: the pass ends, copied back, let go"
                                 : pass   ? " inside a render pass: read from guest memory"
                                          : ": copied back, let go");
                    }
                    if (end_pass && !ended) {
                        render_end_pass_locked();
                        ended = true;
                    }
                    if (pass && !end_pass) {
                        g_twin_edge_in_pass.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        if (pass) g_twin_edge_ended_pass.fetch_add(1, std::memory_order_relaxed);
                        twin_drop_locked(edge[k], true, kTwinGoneEdge);
                    }
                } else {
                    std::lock_guard<std::mutex> lk(g_twin_cand_mu);
                    if (auto c = g_twin_cand.find(edge[k]); c != g_twin_cand.end()) c->second.let_go = true;
                }
            }
            return false;
        }
    }
    if (!inside) return false;
    // Still the resource the candidate was made for?
    GxResourceInfo info;
    if (!gx_resource_at(memory, static_cast<std::size_t>(cbytes), &info) || info.memory != memory || info.gen != gen) {
        twin_drop_locked(memory, false, kTwinGoneDead, gen);
        return false;
    }
    auto t = g_twin_live.find(memory);
    if (t != g_twin_live.end() && t->second.gen != gen) {
        // An older resource's twin at the same memory (its candidate was replaced).
        twin_drop_locked(memory, false, kTwinGoneDead, t->second.gen);
        t = g_twin_live.end();
    }
    if (t == g_twin_live.end()) {
        // Made and filled from guest memory here, before the binding that asked.
        Twin nt;
        nt.bytes = cbytes;
        nt.gen = gen;
        const bool placed = twin_place_locked(cbytes, nt);
        if (!placed || !twin_fill_locked(memory, cbytes, nt.buffer, nt.offset)) {
            if (placed) twin_unplace_locked(nt);
            g_twin_gone[kTwinGoneFill].fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(g_twin_cand_mu);
            if (auto c = g_twin_cand.find(memory); c != g_twin_cand.end()) c->second.let_go = true;
            return false;
        }
        g_twin_made.fetch_add(1, std::memory_order_relaxed);
        g_twin_made_bytes.fetch_add(nt.size, std::memory_order_relaxed);
        g_twin_live_bytes += nt.size;
        gxvk::heap_written(gxvk::kHeapGuardTwin, memory, cbytes);  // GX's upload heap is no GX buffer's own memory (stage 1d.1)
        t = g_twin_live.emplace(memory, nt).first;
    }
    out.buffer = t->second.buffer;
    out.offset = t->second.offset + (va - memory);
    out.avail = cbytes - (va - memory);
    if (address) {
        *address = t->second.address + (va - memory);
        g_twin_addresses.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_twin_served.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

bool twin_covers_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!g_twins || g_twin_live.empty() || !bytes) return false;
    const bool part = run_part();  // (BBHOST_RUN_COMMIT: as twin_locate_locked)
    if ((part || !twins_drop_pending_locked()) && twin_filter_misses_locked(va, bytes)) return false;  // meets no live twin
    // A twin whose buffer died is dropped first (its memory may be a CPU ring
    // now, which must keep its copies): the registry tells (twins_buffer_gone).
    if (!part) {
        twins_drain_gone_locked();
        twins_drop_cpu_mapped_locked();
    }
    auto it = g_twin_live.upper_bound(va);
    const bool any = (it != g_twin_live.begin() && va < std::prev(it)->first + std::prev(it)->second.bytes) ||
                     (it != g_twin_live.end() && it->first < va + bytes);
    if (any) g_twin_refused.fetch_add(1, std::memory_order_relaxed);
    return any;
}

bool twin_holds_live_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!g_twins || g_twin_live.empty() || !bytes) return false;
    if (twin_filter_misses_locked(va, bytes)) return false;  // meets no live twin
    auto it = g_twin_live.upper_bound(va);
    if (it == g_twin_live.begin()) return false;
    --it;
    return va >= it->first && va + bytes <= it->first + it->second.bytes;
}

bool twin_candidate_straddles(std::uint64_t va, std::uint64_t bytes) {
    if (!g_twins || !bytes || !g_twin_cand_n.load(std::memory_order_relaxed)) return false;
    if (g_twins.load(std::memory_order_relaxed) == 3 && (twin_filter_misses_locked(va, bytes) || !twin_armed_meets_locked(va, bytes)))
        return false;  // (callers hold g.mu)
    std::lock_guard<std::mutex> lk(g_twin_cand_mu);
    auto it = g_twin_cand.upper_bound(va);
    if (it != g_twin_cand.begin()) {
        const auto prev = std::prev(it);
        if (!prev->second.let_go && prev->second.armed && va < prev->first + prev->second.bytes && va + bytes > prev->first + prev->second.bytes) {
            return true;
        }
    }
    for (; it != g_twin_cand.end() && it->first < va + bytes; ++it) {
        if (!it->second.let_go && it->second.armed) return true;
    }
    return false;
}

void twin_note_copy_target_locked(std::uint64_t dst, std::uint64_t bytes) {
    if (g_twins.load(std::memory_order_relaxed) != 2 || !bytes) return;
    GxResourceInfo info;
    if (!gx_resource_at(dst, static_cast<std::size_t>(bytes), &info) || info.type != 1 || (info.usage != 0 && info.usage != 2)) return;
    if (dst < info.memory || dst + bytes > info.memory + info.bytes) return;
    // Not one smaller than a binding (16 bytes): a counter copy's 12-byte
    // indirect arguments are bound as 16 by the shaders that read them, which
    // runs across the twin's edge at every use (let go after a few).
    if (info.bytes < 16) return;
    // A new twin is filled before the recording's own work (twin_fill_locked):
    // not while this recording has a GPU write on its pages still to come
    // before this copy (a later copy makes it one).
    // Nor while copy versions still to be put in place write it: their
    // copies go to guest memory, after the twin's fill.
    if (gpu_wrote_since_locked(info.memory, info.bytes, g.flushes) || copy_versions_overlap_locked(info.memory, info.memory + info.bytes)) return;
    std::lock_guard<std::mutex> lk(g_twin_cand_mu);
    if (const auto it = g_twin_cand.find(info.memory); it != g_twin_cand.end() && it->second.gen == info.gen) return;  // one already (or let go)
    if (g_twin_cpu_mapped.count(info.gen)) return;
    g_twin_cand[info.memory] = TwinCandidate{info.bytes, info.gen, false};
    g_twin_cand_n.store(g_twin_cand.size(), std::memory_order_relaxed);
    g_twin_copy_cands.fetch_add(1, std::memory_order_relaxed);
}

void twins_buffer_mapped_own(std::uint64_t memory, std::uint64_t gen) {
    if (!g_twins) return;
    std::lock_guard<std::mutex> lk(g_twin_cand_mu);
    g_twin_cpu_mapped.insert(gen);
    if (auto it = g_twin_cand.find(memory); it != g_twin_cand.end() && it->second.gen == gen && !it->second.let_go) {
        // The CPU writes it in place from now on: guest memory serves it (a
        // live twin is dropped at its next use, without copying back over
        // the CPU's bytes - counted, as what it held since is lost).
        it->second.let_go = true;
        g_twin_cpu_mapped_n.fetch_add(1, std::memory_order_relaxed);
    }
}

void twin_note_import_near_locked(std::uint64_t va, std::uint64_t bytes, const char* site) {
    static std::atomic<int> logs{0};
    if (!g_twins || logs.load(std::memory_order_relaxed) >= 8) return;
    GxResourceInfo info;
    const bool gx = gx_resource_overlapping(va, 1, &info);
    std::uint64_t pm = 0, pb = 0, nm = 0;
    bool plet = false;
    {
        std::lock_guard<std::mutex> lk(g_twin_cand_mu);
        auto it = g_twin_cand.upper_bound(va);
        if (it != g_twin_cand.end()) nm = it->first;
        if (it != g_twin_cand.begin()) {
            --it;
            pm = it->first, pb = it->second.bytes, plet = it->second.let_go;
        }
    }
    if (!gx || info.type != 1 || info.usage != 0 || !(info.bind & 0x80)) return;  // only a candidate's kind
    logs.fetch_add(1, std::memory_order_relaxed);
    host_log("gpu: twins: %s at 0x%llx + 0x%llx imports a window over a GX buffer 0x%llx + 0x%x (bind 0x%x); the candidate before it 0x%llx + 0x%llx%s, "
             "the next at 0x%llx",
             site ? site : "a path", static_cast<unsigned long long>(va), static_cast<unsigned long long>(bytes),
             static_cast<unsigned long long>(info.memory), info.bytes, info.bind, static_cast<unsigned long long>(pm),
             static_cast<unsigned long long>(pb), plet ? " (let go)" : "", static_cast<unsigned long long>(nm));
}

bool twin_candidate_holds(std::uint64_t va, std::uint64_t bytes) {
    if (!g_twins || !bytes || !g_twin_cand_n.load(std::memory_order_relaxed)) return false;
    if (g_twins.load(std::memory_order_relaxed) == 3 && (twin_filter_misses_locked(va, bytes) || !twin_armed_meets_locked(va, bytes)))
        return false;  // (callers hold g.mu)
    std::lock_guard<std::mutex> lk(g_twin_cand_mu);
    auto it = g_twin_cand.upper_bound(va);
    if (it == g_twin_cand.begin()) return false;
    --it;
    return !it->second.let_go && it->second.armed && va + bytes <= it->first + it->second.bytes;
}

void twins_release_locked(std::uint64_t va, std::uint64_t bytes, int why) {
    if (!g_twins || g_twin_live.empty() || !bytes) return;
    std::uint64_t hit[8];
    int n = 0;
    auto it = g_twin_live.upper_bound(va);
    if (it != g_twin_live.begin() && va < std::prev(it)->first + std::prev(it)->second.bytes) hit[n++] = std::prev(it)->first;
    for (; it != g_twin_live.end() && it->first < va + bytes && n < 8; ++it) hit[n++] = it->first;
    for (int k = 0; k < n; ++k) twin_drop_locked(hit[k], true, why == 1 ? kTwinGoneTexture : kTwinGoneWalk);
}

void twins_keep_in_guest_locked(std::uint64_t va, std::uint64_t bytes) {
    if (!g_twins || !bytes || !g_twin_cand_n.load(std::memory_order_relaxed)) return;
    twins_release_locked(va, bytes, 0);
    std::lock_guard<std::mutex> lk(g_twin_cand_mu);
    auto it = g_twin_cand.upper_bound(va);
    if (it != g_twin_cand.begin() && va < std::prev(it)->first + std::prev(it)->second.bytes) --it;
    for (; it != g_twin_cand.end() && it->first < va + bytes; ++it) it->second.let_go = true;
}

std::string twins_report() {
    const int level = g_twins.load(std::memory_order_relaxed);
    if (level <= 0) return {};
    std::size_t cands = 0, live = 0, blocks = 0, cpu_mapped = 0;
    std::uint64_t live_bytes = 0;
    {
        std::lock_guard<std::mutex> lk(g_twin_cand_mu);
        cands = g_twin_cand.size();
        cpu_mapped = g_twin_cpu_mapped.size();
    }
    std::uint64_t filter_misses = 0, filter_checked = 0, filter_wrong = 0, filter_builds = 0;
    std::size_t filter_spans = 0;
    {
        std::lock_guard<GpuMutex> lock(g.mu);  // the command processor may still be making twins
        live = g_twin_live.size();
        live_bytes = g_twin_live_bytes;
        blocks = g_twin_blocks.size();
        filter_misses = g_twin_filter_misses, filter_checked = g_twin_filter_checked, filter_wrong = g_twin_filter_wrong;
        filter_builds = g_twin_filter.builds(), filter_spans = g_twin_filter.spans();
    }
    using ull = unsigned long long;
    char mode[96];
    if (level == 3) {
        std::snprintf(mode, sizeof(mode), "the GPU particles' alone, BBHOST_PARTICLE_TWINS: %llu candidates armed by a counting dispatch",
                      static_cast<ull>(g_twin_armed.load()));
    } else {
        std::snprintf(mode, sizeof(mode), "BBHOST_GX_TWINS=%d", level);
    }
    char b[1000];
    std::snprintf(b, sizeof(b),
                  "gpu: GX buffers the GPU writes as device resources (%s): %llu candidates seen made, %llu more at a GPU copy, %zu kept; %llu "
                  "twins made (%llu KiB), %zu live (%llu KiB) in %zu pool blocks; served %llu bindings (%llu across an edge, in part) and %llu device addresses; CPU copies refused %llu; "
                  "let go after copying back: page-table walks %llu, ranges across an edge %llu (%llu of them ending a pass; %llu more inside a pass read from guest memory), a texture over one %llu; dropped with "
                  "their resource %llu; never made (no guest bytes or no memory) %llu; mapped in place by the CPU %llu (%llu of them candidates)",
                  mode, static_cast<ull>(g_twin_cands.load()), static_cast<ull>(g_twin_copy_cands.load()), cands, static_cast<ull>(g_twin_made.load()),
                  static_cast<ull>(g_twin_made_bytes.load() >> 10), live, static_cast<ull>(live_bytes >> 10), blocks,
                  static_cast<ull>(g_twin_served.load()), static_cast<ull>(g_twin_partial.load()), static_cast<ull>(g_twin_addresses.load()),
                  static_cast<ull>(g_twin_refused.load()),
                  static_cast<ull>(g_twin_gone[kTwinGoneWalk].load()), static_cast<ull>(g_twin_gone[kTwinGoneEdge].load()),
                  static_cast<ull>(g_twin_edge_ended_pass.load()), static_cast<ull>(g_twin_edge_in_pass.load()),
                  static_cast<ull>(g_twin_gone[kTwinGoneTexture].load()), static_cast<ull>(g_twin_gone[kTwinGoneDead].load()),
                  static_cast<ull>(g_twin_gone[kTwinGoneFill].load()), static_cast<ull>(cpu_mapped), static_cast<ull>(g_twin_cpu_mapped_n.load()));
    std::string out = b;
    // The miss filter (twin_filter.h): lookups it answered without walking the maps.
    char f[300];
    if (g_twin_fast_miss == kFastMissOff) {
        std::snprintf(f, sizeof(f), "; miss filter off (BBHOST_TWIN_FAST_MISS=0)");
    } else {
        std::snprintf(f, sizeof(f), "; miss filter: %llu lookups answered from %zu spans (%llu builds)", static_cast<ull>(filter_misses), filter_spans,
                      static_cast<ull>(filter_builds));
        if (g_twin_fast_miss == kFastMissCheck) {
            const std::size_t at = std::strlen(f);
            std::snprintf(f + at, sizeof(f) - at, ", each checked against the maps: %llu checked, %llu differ", static_cast<ull>(filter_checked),
                          static_cast<ull>(filter_wrong));
        }
    }
    return out + f;
}

void twins_buffer_gone(std::uint64_t memory, std::uint64_t gen) {
    if (!g_twins || !memory) return;
    std::lock_guard<std::mutex> lk(g_twin_cand_mu);
    if (const auto it = g_twin_cand.find(memory); it != g_twin_cand.end() && it->second.gen == gen) {
        g_twin_cand.erase(it);
        g_twin_cand_n.store(g_twin_cand.size(), std::memory_order_relaxed);
    }
    g_twin_gone_q.emplace_back(memory, gen);
    g_twin_gone_n.fetch_add(1, std::memory_order_release);
}

void twins_buffer_made(std::uint64_t memory, std::uint64_t bytes, std::uint8_t usage, std::uint8_t bind, std::uint64_t gen) {
    if (!g_twins || !memory || !bytes) return;
    {
        // A candidate of an older resource at this memory (its destructor
        // unseen): that resource is gone.
        std::lock_guard<std::mutex> lk(g_twin_cand_mu);
        if (const auto it = g_twin_cand.find(memory); it != g_twin_cand.end() && it->second.gen != gen) {
            g_twin_gone_q.emplace_back(memory, it->second.gen);
            g_twin_gone_n.fetch_add(1, std::memory_order_release);
            g_twin_cand.erase(it);
            g_twin_cand_n.store(g_twin_cand.size(), std::memory_order_relaxed);
        }
    }
    // Type buffer (the caller's), usage default, unordered-access bind.
    if (usage != 0 || !(bind & 0x80)) return;
    g_twin_cands.fetch_add(1, std::memory_order_relaxed);
    const int level = g_twins.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_twin_cand_mu);
    g_twin_cand[memory] = TwinCandidate{bytes, gen, false, level == 1 || level == 2};
    g_twin_cand_n.store(g_twin_cand.size(), std::memory_order_relaxed);
}

// BBHOST_ARENA_WATCH: the last deferred writes replayed, with where each
// resolved, so a value that turns up somewhere it should not can be traced
// back to the write that carried it.
struct ReplayedWrite {
    std::uint64_t va, value, buffer, offset;
    std::uint32_t bytes;
};
std::mutex g_replayed_mu;
ReplayedWrite g_replayed[4096];
std::uint64_t g_replayed_next = 0;
const bool g_replay_watch = [] {
    const char* e = std::getenv("BBHOST_ARENA_WATCH");
    return e && e[0] == '1';
}();
// BBHOST_ARENA_RETIRE_ORDER (on unless 0): a command-arena chunk's retirement
// waits for the labels written into its chunk before it in the same fills
// (flush_deferred_writes_locked); counted at exit.
const bool g_arena_retire_order = [] {
    const char* e = std::getenv("BBHOST_ARENA_RETIRE_ORDER");
    return !(e && e[0] == '0');
}();
std::atomic<std::uint64_t> g_arena_retires_ordered{0};

// CP writes recorded as GPU fills (BBHOST_LABEL_LANDING=0) into memory the game
// had already written again when the fill was recorded (host_lists.h
// rounds_since), and those of them within what its new round had written.
std::atomic<std::uint64_t> g_fills_reused{0}, g_fills_reused_over{0}, g_fills_landed_over{0}, g_fills_landed_unsubmitted{0}, g_fills_dropped{0};
// Whether `w` goes into memory the game has written again (counted, the first
// logged) and is to be left out (`drop`, but never in a GX arena chunk: see
// label_landing.cpp); one that does not is kept to be looked at again once it
// has run.
bool note_fill_reused(const Gpu::SmallWrite& w, bool drop) {
    if (!w.epoch) return false;
    host_lists::Reuse at;
    if (host_lists::rounds_since(w.va, w.bytes, w.epoch, &at) <= 0) {
        if (g.slot_count) g.slots[g.slot].fill_checks.push_back(w);  // fills_retired_locked
        return false;
    }
    drop = drop && !at.arena;
    const bool over = at.written_to && w.va <= at.written_to;
    if (over) g_fills_reused_over.fetch_add(1, std::memory_order_relaxed);
    if (g_fills_reused.fetch_add(1, std::memory_order_relaxed) < 12) {
        std::uint64_t value = 0, there = 0;
        std::memcpy(&value, w.data, w.bytes < 8 ? w.bytes : 8);
        std::memcpy(&there, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(w.va)), w.bytes < 8 ? w.bytes : 8);
        host_log("gpu: a command-processor write (opcode 0x%02x) recorded as a fill into memory the game has written again since its "
                 "submission: 0x%llx <- 0x%llx (%u bytes), there now 0x%llx; the command buffer 0x%llx..0x%llx %d round(s) on, %s",
                 w.op, static_cast<unsigned long long>(w.va), static_cast<unsigned long long>(value), w.bytes,
                 static_cast<unsigned long long>(there), static_cast<unsigned long long>(at.chunk_lo),
                 static_cast<unsigned long long>(at.chunk_end), at.rounds_later,
                 over ? "within what the new round has written" : "past what the new round has written");
    }
    return drop;
}

// A retired submission's tagged fills: one whose chunk has begun a round since
// its submission and that the GPU wrote at a place the new round had written
// before - its value there now, where the new round's packets were - landed
// over them, as a late label would.
void fills_retired_locked(Gpu::Slot& sl) {
    for (const Gpu::SmallWrite& w : sl.fill_checks) {
        host_lists::Reuse at;
        if (host_lists::rounds_since(w.va, w.bytes, w.epoch, &at) <= 0 || !at.written_to || w.va > at.written_to) continue;
        if (!hle_kernel_va_mapped(w.va, w.bytes) ||
            std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(w.va)), w.data, w.bytes) != 0)
            continue;
        // The new round not submitted up to that place even now: the fill
        // landed before its submission, so the walk reads it.
        if (at.unsubmitted) g_fills_landed_unsubmitted.fetch_add(1, std::memory_order_relaxed);
        if (g_fills_landed_over.fetch_add(1, std::memory_order_relaxed) < 12) {
            std::uint64_t value = 0;
            std::memcpy(&value, w.data, w.bytes < 8 ? w.bytes : 8);
            host_log("gpu: a command-processor write (opcode 0x%02x) the GPU filled over a command buffer the game had written again: "
                     "0x%llx holds its 0x%llx (%u bytes) inside the new round (its last packet at 0x%llx), the command buffer "
                     "0x%llx..0x%llx %d round(s) on",
                     w.op, static_cast<unsigned long long>(w.va), static_cast<unsigned long long>(value), w.bytes,
                     static_cast<unsigned long long>(at.written_to), static_cast<unsigned long long>(at.chunk_lo),
                     static_cast<unsigned long long>(at.chunk_end), at.rounds_later);
        }
    }
    sl.fill_checks.clear();
}

// Replays the fence writes held back by host_gpu_mem_write at the end of the
// command buffer, in the order the command processor made them.
//
// They are **fills**, not vkCmdUpdateBuffer. Replayed as updates, a label
// would now and then land holding the data of a write queued a few places
// after it: a command-arena chunk's marker got the per-frame counter written
// to 0x20b800000 (the chunk was never free again, and in the Stats screen the
// arena ran dry within minutes, 0x2ab0a98), and a GX heap release node's state
// got the counter or another label's value (0x2aaa97f). Every recorded write
// resolved to the right buffer and offset; the data was what went astray. The
// same replay as fills: none in a Stats run that turned ~200 markers bad in
// ten seconds before.
//
// A fill carries at most one 32-bit pattern, so an 8-byte write whose halves
// differ is two. Transfers are unordered without a barrier, so a write that
// every later write in the batch overwrites is dropped, and a write that
// overlaps an earlier one still standing gets a barrier first: the last value
// wins, as it did on the command processor.
void flush_deferred_writes_locked() {
    if (g.deferred_writes.empty()) return;
    const ImportSite site("a label or CP write, deferred");
    if (label_landing::drop_reused()) {
        // Not filled at all: what it was written for is gone (label_landing.h).
        auto& all = g.deferred_writes;
        const std::size_t before = all.size();
        all.erase(std::remove_if(all.begin(), all.end(), [](const Gpu::SmallWrite& w) { return note_fill_reused(w, true); }), all.end());
        g_fills_dropped.fetch_add(before - all.size(), std::memory_order_relaxed);
        if (all.empty()) return;
    } else {
        for (const Gpu::SmallWrite& w : g.deferred_writes) note_fill_reused(w, false);
    }
    // Each ordinary run of transfers shares one barrier pair; this one starts
    // its own, so nothing recorded before it can land after it.
    transfer_flush_locked();
    static std::unordered_map<std::uint64_t, std::size_t> last;  // dword -> index of the last write to it
    static std::unordered_set<std::uint64_t> filled;             // dwords filled since the last barrier
    const auto& ws = g.deferred_writes;
    // A submission's few writes (a label or two, at most 8 bytes each) are
    // compared with each other directly: the maps' node per dword was an
    // allocation and a free, at every submission (~1.8% of the command
    // processor at Mensis). Past kSmall the maps, as before.
    constexpr std::size_t kSmall = 32;
    const bool small = ws.size() <= kSmall;
    std::uint64_t filled_small[2 * kSmall];
    std::size_t nfilled = 0;
    if (!small) {
        last.clear();
        filled.clear();
        for (std::size_t i = 0; i < ws.size(); ++i) {
            for (std::uint32_t k = 0; k < ws[i].bytes; k += 4) last[ws[i].va + k] = i;
        }
    }
    // Whether a write after the i-th puts a dword at `at`, as `last` keys them.
    const auto written_later = [&](std::size_t i, std::uint64_t at) {
        for (std::size_t j = i + 1; j < ws.size(); ++j) {
            if (at >= ws[j].va && at < ws[j].va + ws[j].bytes && (at - ws[j].va) % 4 == 0) return true;
        }
        return false;
    };
    // The command-arena chunks filled into since the last barrier.
    std::uint64_t filled_chunks[8];
    int nfilled_chunks = 0;
    bool filled_chunks_lost = false;
    for (std::size_t i = 0; i < ws.size(); ++i) {
        const Gpu::SmallWrite& w = ws[i];
        bool live = false, clash = false;
        for (std::uint32_t k = 0; k < w.bytes; k += 4) {
            if (small) {
                live |= !written_later(i, w.va + k);
                for (std::size_t f = 0; f < nfilled && !clash; ++f) clash = filled_small[f] == w.va + k;
            } else {
                live |= last[w.va + k] == i;
                clash |= filled.count(w.va + k) != 0;
            }
        }
        if (!live) continue;
        Located loc;
        if (!locate_range(w.va, w.bytes, loc)) continue;
        // A command-arena chunk's retirement (its marker's 0) hands the chunk
        // back to the game, which writes new packets into it at once - a
        // thread waiting for a chunk takes it the moment it reads 0. Fills
        // with no barrier between them land in any order, so a label written
        // into that chunk earlier in this batch (an embedded one: YEBIS's
        // compute utilities wait on theirs) could land after the 0, on top of
        // the new packets. The hardware's end-of-pipe events retire in order:
        // the retirement waits for the fills into its chunk before it.
        std::uint64_t chunk = 0;
        const bool in_arena = hle_gx_arena_overlap(w.va, w.bytes, &chunk);
        if (g_arena_retire_order && in_arena && !clash && w.bytes == 8 && hle_gx_arena_marker(w.va)) {
            bool earlier = filled_chunks_lost;
            for (int k = 0; k < nfilled_chunks && !earlier; ++k) earlier = filled_chunks[k] == chunk;
            if (earlier) {
                clash = true;
                g_arena_retires_ordered.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (g_replay_watch) {
            std::uint64_t v = 0;
            std::memcpy(&v, w.data, w.bytes < 8 ? w.bytes : 8);
            std::lock_guard<std::mutex> lk(g_replayed_mu);
            g_replayed[g_replayed_next++ % 4096] = {w.va, v, reinterpret_cast<std::uint64_t>(loc.buffer), loc.offset, w.bytes};
        }
        transfer_begin_locked();
        if (clash) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            rec().pipeline_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0,
                                 nullptr);
            filled.clear();
            nfilled = 0;
            nfilled_chunks = 0;
            filled_chunks_lost = false;
        }
        std::uint32_t d[2] = {0, 0};
        std::memcpy(d, w.data, w.bytes);
        if (w.bytes == 8 && d[0] == d[1]) {
            rec().fill_buffer(loc.buffer, loc.offset, 8, d[0]);
        } else {
            for (std::uint32_t k = 0; k < w.bytes; k += 4) rec().fill_buffer(loc.buffer, loc.offset + k, 4, d[k / 4]);
        }
        for (std::uint32_t k = 0; k < w.bytes; k += 4) {
            if (small) {
                filled_small[nfilled++] = w.va + k;  // at most two dwords a write, kSmall writes
            } else {
                filled.insert(w.va + k);
            }
        }
        if (in_arena) {
            bool known = false;
            for (int k = 0; k < nfilled_chunks && !known; ++k) known = filled_chunks[k] == chunk;
            if (!known) {
                if (nfilled_chunks < 8) {
                    filled_chunks[nfilled_chunks++] = chunk;
                } else {
                    filled_chunks_lost = true;  // more than eight: every retirement after them is ordered
                }
            }
        }
    }
    g.deferred_writes.clear();
}

void transfer_flush_locked() {
    if (!g.in_transfer) return;
    g.in_transfer = false;
    g_batch_spans.clear();
    // Into the open packet when it can take it (a draw's, before its pass).
    DrawCmds* c = DrawCmds::open();
    if (!c || !c->transfer_end()) rec().transfer_barrier(false);
}

bool barrier_masks_fixed() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_BARRIER_MASKS");
        return !(e && e[0] == '0');
    }();
    return on;
}

void record_transfer_barrier(VkCommandBuffer cmd, bool begin) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    if (begin) {
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        return;
    }
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &mb, 0, nullptr, 0, nullptr);
}

void record_copy_order_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// The device address of guest bytes [va, va + bytes) when all of them lie in
// one imported mapping (a direct-memory chunk or a flexible mapping), for a
// shader that reads them by address (the GPU untile, gpu_untile.cpp); 0 when
// they do not.
VkDeviceAddress guest_device_address(std::uint64_t va, std::size_t bytes) {
    if (!bytes || !hle_kernel_va_mapped(va, bytes) || !rebuild_page_tables()) return 0;
    if (twins_on()) {
        Located t;
        VkDeviceAddress at = 0;
        if (twin_locate_locked(va, bytes, t, &at)) return at;
    }
    for (const GuestMapInfo& mi : g.maps) {
        if (va < mi.va || va >= mi.va + mi.len) continue;
        if (g_import_audit) {
            import_audit(mi, 1);
            audit_range(va, bytes, kUseUntile);
        }
        const std::uint64_t off = va - mi.va;
        if (bytes > mi.len - off) return 0;
        if (mi.dmem) {
            const std::uint64_t phys = static_cast<std::uint64_t>(mi.phys) + off;
            const std::uint64_t chunk = phys >> gcn::kDmemChunkShift;
            const std::uint64_t in = phys & (kChunkBytes - 1);
            const int who = t_import_who;
            t_import_who = kIwAddress;
            t_import_va = va;
            const Chunk* c = chunk < gcn::kDmemChunks ? dmem_import(static_cast<std::uint32_t>(chunk), in, bytes, true) : nullptr;
            t_import_va = 0;
            t_import_who = who;
            if (c) note_site_use_locked(bytes);
            return c ? c->address + (in - c->lo) : 0;
        }
        auto it = g.anon_imports.find(mi.va);
        return it == g.anon_imports.end() || !it->second.address ? 0 : it->second.address + off;
    }
    return 0;
}

}  // namespace gpu

void host_gpu_set_loading(bool loading) {
    if (const bool was = gpu::g_loading_screen.exchange(loading); was != loading) {
        gpu::g_loading_ended_ms.store(apu_memory::latest_load_end(gpu::g_loading_ended_ms.load(std::memory_order_relaxed), was, loading,
                                                                  static_cast<std::uint64_t>(gpu::steady_ms())),
                                      std::memory_order_relaxed);
        gpu::g_keeper_cv.notify_one();
    }
    gpu::precompile_set_loading(loading);
}

void host_gpu_world_reached() { gpu::precompile_set_world_reached(); }
void host_gpu_set_compile_session(bool on) {
    gpu::g_compile_session.store(on);
    gpu::hlsl_set_inline(on);
}
HostShaderCompileProgress host_gpu_compile_progress() {
    HostShaderCompileProgress p;
    gpu::compile_session_progress(p.total, p.done, p.failed, p.listed, p.finished, p.reasons, p.recipes_total, p.recipes_done);
    return p;
}
void host_gpu_compile_cancel() { gpu::compile_session_cancel(); }
std::string host_gpu_device_identity() {
    using namespace gpu;
    if (!g.phys) return {};
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g.phys, &props);
    char uuid[2 * VK_UUID_SIZE + 1] = {};
    for (std::uint32_t k = 0; k < VK_UUID_SIZE; ++k) std::snprintf(uuid + 2 * k, 3, "%02x", props.pipelineCacheUUID[k]);
    char buf[384];
    std::snprintf(buf, sizeof(buf), "%08x:%08x:%08x:%s %s", props.vendorID, props.deviceID, props.driverVersion, uuid, props.deviceName);
    return buf;
}
std::size_t host_gpu_shader_backlog() {
    return (gpu::translation_cache_cold() ? gpu::precompile_backlog() : 0) + gpu::shader_sweep_left();
}

void host_gpu_resources_report() {
    std::string r;
    {
        std::lock_guard<GpuMutex> lock(g.mu);
        if (!g.ok) return;
        r = gpu::textures_census_window_locked();
    }
    for (std::size_t at = 0; at < r.size();) {
        const std::size_t nl = r.find('\n', at);
        const std::string line = r.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        host_log("  %s", line.c_str() + (line.rfind("  ", 0) == 0 ? 2 : 0));
        if (nl == std::string::npos) break;
        at = nl + 1;
    }
}

std::string host_gpu_image_heap_report() {
    std::lock_guard<GpuMutex> lock(gpu::g.mu);
    return gpu::image_heap_report();
}

std::string host_gpu_video_memory_report(bool at_exit) {
    if (!gpu::g.ok || !gpu::vram_report_on()) return {};
    std::lock_guard<GpuMutex> lock(gpu::g.mu);
    return gpu::video_memory_report_locked(at_exit);
}

void host_gpu_memory_census_refresh() { (void)host_gpu_video_memory_report(false); }

bool host_gpu_memory_glance(GpuMemoryGlance& out) {
    using namespace gpu;
    if (!g.ok || !g.phys) return false;
    std::uint64_t ours[3] = {};
    for (const MemOwnerTotal& o : bb_memory_owner_totals()) {
        for (int c = 0; c < 3; ++c) ours[c] += o.bytes[c];
    }
    out.vram = ours[0];
    if (g.has_memory_budget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT mb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        mp.pNext = &mb;
        vkGetPhysicalDeviceMemoryProperties2(g.phys, &mp);
        if (g.local_heap < mp.memoryProperties.memoryHeapCount && mb.heapUsage[g.local_heap]) out.vram = mb.heapUsage[g.local_heap];
    }
    out.shared = ours[2];
    out.cache = g_glance_cache.load(std::memory_order_relaxed);
    out.cache_known = g_glance_census.load(std::memory_order_relaxed);
    out.integrated = g.integrated;
    return true;
}

void host_gpu_log_video_memory(bool at_exit) {
    const std::string r = host_gpu_video_memory_report(at_exit);
    if (r.empty()) return;
    // A line a call: host_log keeps 1 KiB of one, and the lines together pass it.
    static const std::string sep = "\n[bbhost]   ";
    std::size_t at = 0;
    for (int k = 0;; ++k) {
        const std::size_t nl = r.find(sep, at);
        const std::string line = r.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        host_log(at_exit && k == 0 ? "%s" : "  %s", line.c_str());
        if (nl == std::string::npos) break;
        at = nl + sep.size();
    }
}

// Per heap: usage and budget as the driver reports them (VK_EXT_memory_budget),
// device-local heaps marked. Empty without the extension.
std::string host_gpu_memory_budget_report() {
    using namespace gpu;
    if (!g.has_memory_budget || !g.phys) return {};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    mp.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(g.phys, &mp);
    std::string out = "gpu: memory by heap:";
    char buf[96];
    for (std::uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount; ++i) {
        const bool local = (mp.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        std::snprintf(buf, sizeof(buf), " heap %u%s %llu/%llu MiB;", i, local ? " (device)" : "",
                      static_cast<unsigned long long>(budget.heapUsage[i] >> 20),
                      static_cast<unsigned long long>(budget.heapBudget[i] >> 20));
        out += buf;
    }
    out += "\n[bbhost]   " + bb_memory_sites_report();
    if (const std::string rings = rings_placement_report(); !rings.empty()) out += "\n[bbhost]   " + rings;
    if (g_import_audit) out += "\n[bbhost]   " + import_audit_report() + "\n[bbhost]   " + audit_pages_report();
    return out;
}
std::uint64_t host_gpu_texture_uploads() { return textures_upload_count(); }
std::uint64_t host_gpu_texture_hashes() { return textures_hash_count(); }
std::uint64_t host_gpu_texture_hash_us() { return textures_hash_us(); }
void host_gpu_texture_time_us(std::uint64_t out[5]) { textures_time_us(out); }
std::uint64_t host_gpu_surface_replacements() { return textures_replacements() + g.rt_replacements.load(); }
void host_gpu_note_shader_created(int stage, const std::uint8_t* container, std::size_t size, std::uint64_t flip) {
    note_shader_created(stage, container, size, flip);
}

// For BBHOST_ARENA_WATCH: where va resolves, and the replayed deferred writes
// that carried `value` (logged).
void host_gpu_explain_value(std::uint64_t va, std::uint64_t value) {
    std::uint64_t buf = 0, off = 0;
    {
        std::lock_guard<GpuMutex> lock(g.mu);
        Located loc;
        if (locate_range(va, 8, loc)) {
            buf = reinterpret_cast<std::uint64_t>(loc.buffer);
            off = loc.offset;
        }
    }
    host_log("arena watch:   0x%llx resolves to buffer 0x%llx offset 0x%llx", static_cast<unsigned long long>(va),
             static_cast<unsigned long long>(buf), static_cast<unsigned long long>(off));
    std::lock_guard<std::mutex> lk(gpu::g_replayed_mu);
    int shown = 0;
    for (std::uint64_t i = 0; i < 4096 && shown < 6; ++i) {
        const auto& r = gpu::g_replayed[i];
        if (r.bytes && (r.value == value || (r.buffer == buf && r.offset <= off && off < r.offset + r.bytes))) {
            ++shown;
            host_log("arena watch:   replayed write 0x%llx <- 0x%llx (%u bytes) resolved to buffer 0x%llx offset 0x%llx",
                     static_cast<unsigned long long>(r.va), static_cast<unsigned long long>(r.value), r.bytes,
                     static_cast<unsigned long long>(r.buffer), static_cast<unsigned long long>(r.offset));
        }
    }
    if (!shown) host_log("arena watch:   no replayed write carried that value or hit that location");
}

std::uint64_t host_gpu_completed_submits() {
    std::lock_guard<GpuMutex> lock(g.mu);
    return g.completed_submits;
}

std::atomic<std::uint64_t> gpu::g_completed_submits_seen{0};
std::uint64_t host_gpu_completed_submits_seen() { return gpu::g_completed_submits_seen.load(std::memory_order_relaxed); }


// A host-GPU write whose destination covers a command-arena chunk's marker
// (BBHOST_ARENA_MONITOR): nothing the renderer does should, so the first few
// are logged with where they came from.
void note_gpu_write_over_marker(const char* what, std::uint64_t dst, std::uint64_t src, std::size_t bytes) {
    if (!hle_gnm_arena_monitor() || !hle_gx_arena_covers_marker(dst, bytes)) return;
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 32) {
        host_log("arena: host GPU %s writes 0x%llx..0x%llx (%zu bytes, from 0x%llx) over a chunk marker", what,
                 static_cast<unsigned long long>(dst), static_cast<unsigned long long>(dst + bytes), bytes,
                 static_cast<unsigned long long>(src));
    }
}

// The command processor's tag for the writes it makes (host_gpu_set_write_tag).
thread_local std::uint64_t t_write_epoch = 0;
thread_local const std::uint32_t* t_write_op = nullptr;
void host_gpu_set_write_tag(std::uint64_t epoch, const std::uint32_t* op) {
    t_write_epoch = epoch;
    t_write_op = op;
}

bool host_gpu_mem_write(std::uint64_t va, const void* data, std::size_t bytes) { return host_gpu_mem_write_serial(va, data, bytes, nullptr); }

bool host_gpu_mem_write_serial(std::uint64_t va, const void* data, std::size_t bytes, std::uint64_t* serial) {
    std::lock_guard<GpuMutex> lock(g.mu);
    const ImportSite site("a label or CP write");
    if (!g.ok || bytes > 65536) return false;
    if (serial) *serial = g.flushes;  // the submission this write is recorded into
    if (hle_gnm_arena_monitor() && hle_gx_arena_covers_marker(va, bytes)) {
        std::uint64_t v = 0;
        std::memcpy(&v, data, bytes < 8 ? bytes : 8);
        if (v > 2) note_gpu_write_over_marker("write", va, v, bytes);
    }
    // A label the host writes once this submission has run
    // (label_landing.h): no import, no fill. Whatever reads it in place reads
    // it as behind this submission, as it did a GPU fill's.
    if (bytes <= 8 && landing_on()) {
        if ((va & 3) || (bytes & 3) || !hle_kernel_va_mapped(va, bytes)) return false;
        Gpu::SmallWrite w{va, static_cast<std::uint32_t>(bytes), {}, t_write_epoch, t_write_op ? *t_write_op : 0u};
        std::memcpy(w.data, data, bytes);
        g.deferred_writes.push_back(w);
        begin_recording_locked();  // the submission whose completion lands it
        shadow_written_locked(va, bytes, true);
        label_check_note_locked(va, data, bytes, g.flushes);
        return true;
    }
    Located loc;
    if (!locate_range(va, bytes, loc)) return false;
    // A four- or eight-byte write is a fence, not data any shader in this
    // command buffer reads. Recording it here would mean leaving the render
    // pass and draining the pipeline - about five hundred times a frame during
    // a world load, which was most of the GPU time. Hold it back to the end of
    // the command buffer instead: it still lands after every draw it fences,
    // only later, and the command processor no longer stalls on these because
    // WAIT_REG_MEM consults the recorded value.
    if (bytes <= 8) {
        Gpu::SmallWrite w{va, static_cast<std::uint32_t>(bytes), {}, t_write_epoch, t_write_op ? *t_write_op : 0u};
        std::memcpy(w.data, data, bytes);
        g.deferred_writes.push_back(w);
        begin_recording_locked();
        shadow_written_locked(va, bytes, true);  // lands on the GPU's timeline: the mirror's copies run before it
        g.transfers.fetch_add(1);
        return true;
    }
    // Staged and copied rather than vkCmdUpdateBuffer, which misplaced label
    // data (flush_deferred_writes_locked).
    DevBuffer staging;
    VkDeviceSize staging_offset = 0;
    if (!acquire_staging_locked(staging, bytes, staging_offset)) return false;
    std::memcpy(staging.map, data, bytes);
    // Into a packet of the recorder's, as the copy tokens' are.
    std::optional<DrawCmds> own;
    if (!DrawCmds::open() && !stream_on()) own.emplace(!g.profile && !g.has_checkpoints);  // with the stream, ops of their own
    transfer_begin_locked();
    transfer_order_locked(VK_NULL_HANDLE, 0, loc.buffer, loc.offset, bytes);  // the staging bytes are new: only the destination can clash
    VkBufferCopy region{staging_offset, loc.offset, bytes};
    DrawCmds* c = DrawCmds::open();
    if (!c || !c->copy_buffer(staging.buffer, loc.buffer, region)) rec().copy_buffer(staging.buffer, loc.buffer, 1, &region);
    transfer_end_locked();
    shadow_written_locked(va, bytes, true);
    note_pending_write_locked(va, static_cast<const std::uint8_t*>(data), bytes);
    return true;
}

bool host_gpu_mem_write_ordered(std::uint64_t va, const void* data, std::size_t bytes, std::uint64_t* serial) {
    std::lock_guard<GpuMutex> lock(g.mu);
    const ImportSite site("a label or CP write, ordered");
    if (!g.ok || !bytes || bytes > 65536) return false;
    if (surfaces_may_cover_locked(va, bytes) || find_render_target(va)) return false;
    Located loc;
    if (!locate_range(va, bytes, loc)) return false;
    DevBuffer staging;
    VkDeviceSize staging_offset = 0;
    if (!acquire_staging_locked(staging, bytes, staging_offset)) return false;
    std::memcpy(staging.map, data, bytes);
    if (serial) *serial = g.flushes;  // the submission this write is recorded into
    std::optional<DrawCmds> own;
    if (!DrawCmds::open() && !stream_on()) own.emplace(!g.profile && !g.has_checkpoints);  // with the stream, ops of their own
    transfer_begin_locked();
    transfer_order_locked(VK_NULL_HANDLE, 0, loc.buffer, loc.offset, bytes);  // after what the work before reads and writes there
    VkBufferCopy region{staging_offset, loc.offset, bytes};
    DrawCmds* c = DrawCmds::open();
    if (!c || !c->copy_buffer(staging.buffer, loc.buffer, region)) rec().copy_buffer(staging.buffer, loc.buffer, 1, &region);
    transfer_end_locked();
    shadow_written_locked(va, bytes, true);  // a constant buffer over it is read in place, behind the write
    note_pending_write_locked(va, static_cast<const std::uint8_t*>(data), bytes);
    return true;
}

void host_gpu_read_guest(std::uint64_t va, std::size_t bytes, void* out) {
    std::lock_guard<GpuMutex> lock(g.mu);
    read_guest_locked(va, bytes, out);
}

bool host_gpu_writes_pending(std::uint64_t va, std::size_t bytes) {
    std::lock_guard<GpuMutex> lock(g.mu);
    return gpu_writes_pending_locked(va, bytes);
}

bool host_gpu_clear_target(std::uint64_t va, std::uint32_t value, std::size_t bytes) {
    std::lock_guard<GpuMutex> lock(g.mu);
    return g.ok && render_clear_by_fill_locked(va, bytes, value);
}

std::atomic<std::uint64_t> g_token_fills{0}, g_token_fills_unplaced{0};

bool host_gpu_fill(std::uint64_t va, std::size_t bytes, const float rgba[4], std::uint32_t depth_clear, std::uint32_t stencil_clear) {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok || !bytes) return false;
    hle_gnm_warn_if_command_buffer("fill token", va, bytes);
    const bool on_rt = find_render_target(va) != nullptr;
    if (mark_surfaces_dirty_locked(va, bytes, false) || on_rt) {
        tex_event(va, bytes, "fill token 0x%llx 0x%zx bytes with %g %g %g %g%s", static_cast<unsigned long long>(va), bytes, rgba[0],
                  rgba[1], rgba[2], rgba[3], on_rt ? " (render target)" : "");
    }
    if (render_handle_fill_locked(va, bytes, rgba, depth_clear, stencil_clear)) {
        g_token_fills.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    g_token_fills_unplaced.fetch_add(1, std::memory_order_relaxed);
    return false;
}

bool host_gpu_upload_region(std::uint64_t base, const std::uint32_t* tsharp, std::uint32_t mip, std::uint32_t layer, std::uint32_t x,
                            std::uint32_t y, std::uint32_t w, std::uint32_t h, const void* data, std::size_t bytes, std::uint32_t row_bytes) {
    std::lock_guard<GpuMutex> lock(g.mu);
    return g.ok && textures_upload_region_locked(base, tsharp, mip, layer, x, y, w, h, data, bytes, row_bytes);
}

bool host_gpu_copy_image_region(const GpuImageCopy& c) {
    std::lock_guard<GpuMutex> lock(g.mu);
    return g.ok && textures_copy_image_region_locked(c);
}

bool host_gpu_texture_create_ahead(const std::uint32_t tsharp[8], std::uint64_t gen) {
    std::lock_guard<GpuMutex> lock(g.mu);
    return g.ok && textures_create_ahead_locked(tsharp, gen);
}

void host_gpu_texture_retire(std::uint64_t base, std::size_t bytes, std::uint64_t after_submitted, std::uint64_t gen) {
    textures_retire(base, bytes, after_submitted, gen);  // its own lock: callable from the kernel's unmap
}

void host_gpu_texture_written(std::uint64_t base, std::size_t bytes) {
    textures_gx_written(base, bytes);  // its own lock, as the retirement's
}

bool host_gpu_mem_fill(std::uint64_t va, std::uint32_t value, std::size_t bytes) {
    std::lock_guard<GpuMutex> lock(g.mu);
    const ImportSite site("a CP fill");
    if (!g.ok) return false;
    Located loc;
    if (!locate_range(va, bytes, loc)) return false;
    note_gpu_write_over_marker("fill", va, value, bytes);
    transfer_begin_locked();
    transfer_order_locked(VK_NULL_HANDLE, 0, loc.buffer, loc.offset, bytes, true);
    rec().fill_buffer(loc.buffer, loc.offset, bytes, value);
    transfer_end_locked();
    return true;
}

bool host_gpu_mem_copy(std::uint64_t dst, std::uint64_t src, std::size_t bytes) {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok) return false;
    if (dst < src + bytes && src < dst + bytes) return false;  // overlapping: memmove on the host
    Located d, sr;
    {
        const ImportSite site("a CP copy's destination");
        if (!locate_range(dst, bytes, d)) return false;
    }
    {
        const ImportSite site("a CP copy's source");
        if (!locate_range(src, bytes, sr)) return false;
    }
    note_gpu_write_over_marker("copy", dst, src, bytes);
    transfer_begin_locked();
    transfer_order_locked(sr.buffer, sr.offset, d.buffer, d.offset, bytes, true);
    VkBufferCopy region{sr.offset, d.offset, bytes};
    rec().copy_buffer(sr.buffer, d.buffer, 1, &region);
    transfer_end_locked();
    return true;
}

namespace {

// A GDS counter's start before the dispatch that counts in it
// (GpuDispatch::gds): a set count filled into the word, or another word's
// copied in, in the open transfer batch.
bool counter_start_locked(const GpuDispatch::GdsCounter& c) {
    const ImportSite site("a GDS counter's start");
    Located dst;
    if (!locate_range(c.word, 4, dst)) return false;
    Located src;
    if (c.start == GpuDispatch::kGdsCopy && !locate_range(c.from, 4, src)) return false;
    transfer_begin_locked();
    if (c.start == GpuDispatch::kGdsSet) {
        transfer_order_locked(VK_NULL_HANDLE, 0, dst.buffer, dst.offset, 4, true);
        rec().fill_buffer(dst.buffer, dst.offset, 4, c.value);
    } else {
        transfer_order_locked(src.buffer, src.offset, dst.buffer, dst.offset, 4, true);
        VkBufferCopy region{src.offset, dst.offset, 4};
        rec().copy_buffer(src.buffer, dst.buffer, 1, &region);
    }
    transfer_end_locked();
    shadow_written_locked(c.word, 4, true);
    return true;
}

}  // namespace

std::atomic<std::uint64_t> g_counter_copies{0}, g_counter_fills{0};

bool host_gpu_counter_copy(std::uint64_t dst, std::uint64_t src, std::size_t bytes) {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok || !bytes) return false;
    if (dst < src + bytes && src < dst + bytes) return false;
    Located d, sr;
    {
        const ImportSite site("a counter copy's source");
        if (!twin_candidate_holds(src, bytes)) import_windows(src, bytes);
        if (!locate_range(src, bytes, sr)) return false;
    }
    {
        const ImportSite site("a counter copy's destination");
        twin_note_copy_target_locked(dst, bytes);  // a GX buffer the GPU writes (BBHOST_GX_TWINS=2)
        if (!twin_candidate_holds(dst, bytes)) import_windows(dst, bytes);
        if (!locate_range(dst, bytes, d)) return false;
    }
    note_gpu_write_over_marker("counter copy", dst, src, bytes);
    transfer_begin_locked();
    transfer_order_locked(sr.buffer, sr.offset, d.buffer, d.offset, bytes, true);
    VkBufferCopy region{sr.offset, d.offset, bytes};
    rec().copy_buffer(sr.buffer, d.buffer, 1, &region);
    transfer_end_locked();
    if (!twin_holds_live_locked(dst, bytes)) shadow_written_locked(dst, bytes, true);  // (a twin's write leaves guest memory)
    g_counter_copies.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool host_gpu_counter_fill(std::uint64_t dst, std::uint32_t value, std::size_t bytes) {
    std::lock_guard<GpuMutex> lock(g.mu);
    const ImportSite site("a counter fill");
    if (!g.ok || !bytes) return false;
    twin_note_copy_target_locked(dst, bytes);  // a GX buffer the GPU writes (BBHOST_GX_TWINS=2)
    if (!twin_candidate_holds(dst, bytes)) import_windows(dst, bytes);
    Located d;
    if (!locate_range(dst, bytes, d)) return false;
    note_gpu_write_over_marker("counter fill", dst, value, bytes);
    transfer_begin_locked();
    transfer_order_locked(VK_NULL_HANDLE, 0, d.buffer, d.offset, bytes, true);
    rec().fill_buffer(d.buffer, d.offset, bytes, value);
    transfer_end_locked();
    if (!twin_holds_live_locked(dst, bytes)) shadow_written_locked(dst, bytes, true);  // (a twin's write leaves guest memory)
    g_counter_fills.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Copy tokens whose source was a render target, copied as images.
std::atomic<std::uint64_t> g_guest_copy_targets{0};
std::uint64_t host_gpu_guest_copy_targets() { return g_guest_copy_targets.load(std::memory_order_relaxed); }
// Copy tokens' sources copied into upload staging (window imports), at most
// this much a token (a bigger one reads its import).
constexpr std::size_t kStagedCopyMax = 4u << 20;
std::atomic<std::uint64_t> g_copy_src_staged{0}, g_copy_src_staged_bytes{0};

bool host_gpu_copy_guest(std::uint64_t dst, std::uint64_t src, std::size_t bytes, std::uint64_t src_gen, std::uint64_t dst_gen) {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok) return false;
    if (!bytes) return true;
    if (dst < src + bytes && src < dst + bytes) return false;
    note_gpu_write_over_marker("guest copy", dst, src, bytes);
    bool skip_target = false, by_resource = src_gen && copy_by_resource_on();
    if (by_resource) {
        HostImageRef ref;
        if (host_image_by_resource_locked(src, src_gen, &ref)) {
            if (ref.target) {
                g_copy_res[0].fetch_add(1, std::memory_order_relaxed);
            } else if (ref.gpu_written && g_copy_by_resource < 2) {
                g_copy_res[6].fetch_add(1, std::memory_order_relaxed);  // as before: by address (=2: as its image)
            } else {
                skip_target = true;  // the source is its surface: a target at its memory is another resource's
                if (textures_copy_resource_locked(src, src_gen, dst, dst_gen)) {
                    // Image to image: the destination's memory is not written, so
                    // its surfaces are not marked (its own image took the copy).
                    g_copy_res[1].fetch_add(1, std::memory_order_relaxed);
                    return true;
                } else {
                    g_copy_res[3].fetch_add(1, std::memory_order_relaxed);
                }
            }
        } else if (RtImage* rt = find_render_target(src); rt && rt->gx_gen && static_cast<std::uint32_t>(rt->gx_gen) != static_cast<std::uint32_t>(src_gen)) {
            skip_target = true;
            g_copy_res[2].fetch_add(1, std::memory_order_relaxed);
        } else {
            g_copy_res[3].fetch_add(1, std::memory_order_relaxed);
        }
    }
    // Copied image to image, the guest memory is not written: the surfaces
    // over it are not marked (mark_surfaces_dirty_locked's memory_written).
    const bool as_image = !skip_target && render_copy_target_locked(src, dst, bytes, "copy token");
    if (by_resource && dst_gen) {
        if (RtImage* into = find_render_target(dst); into && into->image) {
            if (as_image && static_cast<std::uint32_t>(into->gx_gen) != static_cast<std::uint32_t>(dst_gen)) {
                into->gx_gen = dst_gen;  // what the copy wrote is the destination resource's
                into->dead_ms = into->dead_after = into->dead_memory = 0;
                g_copy_res[5].fetch_add(1, std::memory_order_relaxed);
            } else if (!as_image && static_cast<std::uint32_t>(into->gx_gen) == static_cast<std::uint32_t>(dst_gen)) {
                g_copy_res[4].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    // A GX buffer GPU copies write (BBHOST_GX_TWINS=2): its twin takes the
    // copy, and guest memory - which the buffer shadow would note as written -
    // stays as it was.
    if (!as_image) twin_note_copy_target_locked(dst, bytes);
    // A destination across a twin's edge: the twin is copied back and let go
    // at the locate below, which a render pass would stop (and the copy would
    // write guest memory past a live twin) - the pass ends here, as the copy's
    // transfer would end it anyway.
    if (!as_image && twin_candidate_straddles(dst, bytes)) render_end_pass_locked();
    const int on_surfaces = !as_image && twin_candidate_holds(dst, bytes) ? 0 : mark_surfaces_dirty_locked(dst, bytes, !as_image, "copy token");
    if (as_image) {
        tex_event(dst, bytes, "copy token 0x%llx -> 0x%llx 0x%zx bytes: render target copied as an image",
                  static_cast<unsigned long long>(src), static_cast<unsigned long long>(dst), bytes);
        g_guest_copy_targets.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (on_surfaces || find_render_target(src)) {
        tex_event(dst, bytes, "copy token 0x%llx -> 0x%llx 0x%zx bytes: as memory%s", static_cast<unsigned long long>(src),
                  static_cast<unsigned long long>(dst), bytes, find_render_target(src) ? " (FROM A RENDER TARGET)" : "");
    }
    if (!hle_kernel_va_mapped(dst, bytes) || !hle_kernel_va_mapped(src, bytes) || !rebuild_page_tables()) return false;
    if (g_import_audit) {
        audit_range(src, bytes, kUseTransfer);
        audit_range(dst, bytes, kUseTransfer);
    }
    // The token's pass end, barrier and copies go to the recorder
    // in a packet of their own; recorded in place, each waited for the recorder
    // to finish every draw before it (~1.5% of the command processor).
    std::optional<DrawCmds> own;
    if (!DrawCmds::open() && !stream_on()) own.emplace(!g.profile && !g.has_checkpoints);  // with the stream, ops of their own
    // Window imports: a source the CPU wrote with no GPU write still to land
    // on it (GX's renamed ring bytes copied back, copies too big for a copy
    // version) is copied into upload staging here, as a copy version's is -
    // its bytes as this point in the stream has them, the pending versions
    // laid over guest memory's - and needs no import.
    Located staged;
    if (!imports_whole() && bytes <= kStagedCopyMax && !twin_covers_locked(src, bytes) && !gpu_writes_pending_locked(src, bytes) &&
        !find_render_target(src) && !surfaces_may_cover_locked(src, bytes)) {
        DevBuffer sb;
        VkDeviceSize so = 0;
        if (acquire_staging_locked(sb, bytes, so)) {
            auto* to = static_cast<std::uint8_t*>(sb.map);
            std::memcpy(to, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(src)), bytes);
            copy_versions_overlay_locked(src, bytes, to);
            staged = {sb.buffer, so, bytes};
            g_copy_src_staged.fetch_add(1, std::memory_order_relaxed);
            g_copy_src_staged_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }
    }
    for (std::size_t done = 0; done < bytes;) {
        Located d, s;
        {
            const ImportSite site("a copy token's destination");
            d = locate(dst + done, bytes - done);
            if (!done) note_copied_locked(dst);
        }
        if (staged.buffer) {
            s = {staged.buffer, staged.offset + done, bytes - done};
        } else {
            const ImportSite site("a copy token's source");
            s = locate(src + done, bytes - done);
        }
        if (!d.buffer || !s.buffer) return false;
        const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(bytes - done, std::min<std::uint64_t>(d.avail, s.avail)));
        if (!n) return false;
        if (d.buffer == s.buffer && d.offset < s.offset + n && s.offset < d.offset + n) return false;
        transfer_begin_locked();
        transfer_order_locked(s.buffer, s.offset, d.buffer, d.offset, n);
        VkBufferCopy region{s.offset, d.offset, n};
        DrawCmds* c = DrawCmds::open();
        if (!c || !c->copy_buffer(s.buffer, d.buffer, region)) rec().copy_buffer(s.buffer, d.buffer, 1, &region);
        transfer_end_locked();
        done += n;
    }
    return true;
}

// ---- Copy versions ----
// GX copies a dynamic buffer's renamed bytes back to its own address with a
// compute pass between draws: ~30 copy tokens a frame, ~200 copies, half of
// the tokens in the middle of a render pass (a dozen in the G-buffer pass
// alone). As GPU copies each token ended the pass and drained the pipeline
// twice: 6% of the RTX 4070's frame. Now the bytes go into a staging copy at
// the token - the CPU wrote them (the game renames, then writes) - the draws
// after it bind that copy wherever a binding falls inside one (the same frame's
// later passes read them, not the pass they were made in), and the copies into
// place wait for the next transfer batch, dispatch or submission: one batch
// of them, one pair of barriers. A binding across a copy's edge, a page-table
// read over one, a pipeline that walks the page table, or a source an earlier
// copy is still to write puts them in place first.
namespace gpu {
namespace {
struct CopyVersion {
    std::uint64_t dst = 0, bytes = 0;
    DevBuffer copy;  // its map is the bytes
    VkDeviceSize offset = 0;
    Located canon;   // where they belong
};
std::vector<CopyVersion> g_copy_versions;  // under g.mu: the pending ones, oldest first
// The same by destination, for lookups at every binding: a scan of the list
// was a tenth of the command processor (BBHOST_COPY_VERSIONS=2, ~80 pending).
struct CopyVersionKey {
    std::uint64_t dst, end;
    std::uint32_t index;  // into g_copy_versions: the larger, the newer
};
std::vector<CopyVersionKey> g_cv_sorted;
std::uint64_t g_cv_lo = ~0ull, g_cv_hi = 0;  // what they cover, from the lowest start to the highest end
constexpr std::uint64_t kCopyVersionMax = 64u << 10;
constexpr std::size_t kCopyVersionsPending = 512;
std::atomic<std::uint64_t> g_cv_made{0}, g_cv_bound{0}, g_cv_batches{0}, g_cv_copies{0}, g_cv_straddled{0}, g_cv_refused{0},
    g_cv_from_version{0}, g_cv_overwritten{0};
enum { kCvSize, kCvSurfaces, kCvTarget, kCvImport, kCvSource, kCvOther, kCvReasons };
std::atomic<std::uint64_t> g_cv_refused_why[kCvReasons] = {};
// 2 (default): the copies into place before the next transfer batch,
// dispatch or submission, and copies outside a pass versioned too - the
// draws of later passes read them, from their copies (1.6M bindings in a
// soak). 1: at the pass's end. 0: GPU copies at the token.
const int g_copy_versions_mode = [] {
    const char* e = std::getenv("BBHOST_COPY_VERSIONS");
    return e && *e ? std::atoi(e) : 2;
}();

// The newest version that shares bytes with [base, base + bytes).
const CopyVersion* copy_version_over(std::uint64_t base, std::uint64_t bytes) {
    const std::uint64_t end = base + bytes;
    if (base >= g_cv_hi || end <= g_cv_lo) return nullptr;
    // None is longer than kCopyVersionMax, so none starting before this reaches base.
    const std::uint64_t from = base > kCopyVersionMax ? base - kCopyVersionMax : 0;
    auto it = std::lower_bound(g_cv_sorted.begin(), g_cv_sorted.end(), from,
                               [](const CopyVersionKey& k, std::uint64_t v) { return k.dst < v; });
    const CopyVersionKey* best = nullptr;
    for (; it != g_cv_sorted.end() && it->dst < end; ++it) {
        if (it->end > base && (!best || it->index > best->index)) best = &*it;
    }
    return best ? &g_copy_versions[best->index] : nullptr;
}

void copy_version_add(const CopyVersion& v) {
    const CopyVersionKey k{v.dst, v.dst + v.bytes, static_cast<std::uint32_t>(g_copy_versions.size())};
    g_copy_versions.push_back(v);
    g_cv_sorted.insert(std::upper_bound(g_cv_sorted.begin(), g_cv_sorted.end(), k.dst,
                                        [](std::uint64_t d, const CopyVersionKey& e) { return d < e.dst; }),
                       k);
    g_cv_lo = std::min(g_cv_lo, k.dst);
    g_cv_hi = std::max(g_cv_hi, k.end);
}
}  // namespace

bool copy_versions_pending_locked() { return !g_copy_versions.empty(); }

void copy_versions_pass_end_locked() {
    if (g_copy_versions_mode == 1) copy_versions_flush_locked();
}

std::atomic<std::uint64_t> g_cv_put[kCvPutReasons] = {};
void copy_versions_put_in_place_locked(int why) {
    if (g_copy_versions.empty()) return;
    g_cv_put[why].fetch_add(1, std::memory_order_relaxed);
    render_end_pass_locked();
    copy_versions_flush_locked();
}

bool copy_versions_overlap_locked(std::uint64_t lo, std::uint64_t hi) {
    if (g_copy_versions.empty() || hi <= g_cv_lo || lo >= g_cv_hi) return false;
    for (const CopyVersion& v : g_copy_versions) {
        if (v.dst < hi && v.dst + v.bytes > lo) return true;
    }
    return false;
}

bool copy_versions_into_locked(VkBuffer buffer, VkDeviceSize lo, VkDeviceSize hi) {
    for (const CopyVersion& v : g_copy_versions) {
        if (v.canon.buffer == buffer && v.canon.offset < hi && v.canon.offset + v.bytes > lo) return true;
    }
    return false;
}

bool copy_versions_name_buffer_locked(VkBuffer buffer) {
    for (const CopyVersion& v : g_copy_versions) {
        if (v.canon.buffer == buffer) return true;
    }
    return false;
}

int copy_version_lookup_locked(std::uint64_t base, std::uint64_t bytes, Located* out) {
    if (g_copy_versions.empty() || !bytes) return kCopyVersionNone;
    const CopyVersion* v = copy_version_over(base, bytes);
    if (!v) return kCopyVersionNone;
    if (base < v->dst || base + bytes > v->dst + v->bytes) {
        g_cv_straddled.fetch_add(1, std::memory_order_relaxed);
        return kCopyVersionStraddles;
    }
    if (out) {
        *out = {v->copy.buffer, v->offset + (base - v->dst), v->bytes - (base - v->dst)};
        g_cv_bound.fetch_add(1, std::memory_order_relaxed);
    }
    return kCopyVersionInside;
}

void copy_versions_overlay_locked(std::uint64_t base, std::uint64_t bytes, std::uint8_t* to) {
    if (base >= g_cv_hi || base + bytes <= g_cv_lo) return;
    for (const CopyVersion& v : g_copy_versions) {  // oldest first: the newest bytes win
        const std::uint64_t lo = std::max(base, v.dst), hi = std::min(base + bytes, v.dst + v.bytes);
        if (lo < hi) std::memcpy(to + (lo - base), static_cast<const std::uint8_t*>(v.copy.map) + (lo - v.dst), hi - lo);
    }
}

bool copy_version_make_locked(std::uint64_t dst, std::uint64_t src, std::uint64_t bytes) {
    // Outside a pass the copy ends nothing; inside an open transfer batch it
    // costs nothing more, and a later transfer of the batch could read it.
    if (!g_copy_versions_mode || (g_copy_versions_mode == 1 && !render_pass_open_locked()) || g.in_transfer) return false;
    const auto refuse = [](int why) {
        bump(g_cv_refused);
        bump(g_cv_refused_why[why]);
        return false;
    };
    if (!bytes || bytes > kCopyVersionMax || ((dst | src | bytes) & 3)) return refuse(kCvSize);
    // A source kept as a device-local twin: its guest bytes are stale (the
    // version's are read on the CPU), and the copy goes to the GPU as it
    // stands. A destination in one takes the version's bytes into the twin.
    if (twin_covers_locked(src, bytes) || twin_candidate_straddles(dst, bytes)) return refuse(kCvOther);
    if ((dst < src + bytes && src < dst + bytes) || g_copy_versions.size() >= kCopyVersionsPending) return refuse(kCvOther);
    if (!hle_kernel_va_mapped(dst, bytes) || !hle_kernel_va_mapped(src, bytes) || !rebuild_page_tables()) return refuse(kCvOther);
    if (surfaces_may_cover_locked(dst, bytes)) return refuse(kCvSurfaces);
    const ImportSite site("a copy version's destination");
    if (find_render_target(src) || find_render_target(dst)) return refuse(kCvTarget);
    twin_note_copy_target_locked(dst, bytes);
    const Located canon = locate(dst, bytes);
    if (!canon.buffer || canon.avail < bytes) return refuse(kCvImport);
    note_copied_locked(dst);
    // The source as this point in the stream has it: an earlier version
    // holding all of it, else guest memory. Part of one: the GPU copy, after
    // the pass's own copies.
    const auto* from = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(src));
    if (const CopyVersion* v = copy_version_over(src, bytes)) {
        if (src < v->dst || src + bytes > v->dst + v->bytes) return refuse(kCvSource);
        from = static_cast<const std::uint8_t*>(v->copy.map) + (src - v->dst);
        bump(g_cv_from_version);
    }
    CopyVersion v;
    v.dst = dst;
    v.bytes = bytes;
    v.canon = canon;
    if (!acquire_staging_locked(v.copy, bytes, v.offset)) return refuse(kCvOther);
    std::memcpy(v.copy.map, from, bytes);
    note_gpu_write_over_marker("guest copy", dst, src, bytes);
    gxvk::heap_written(gxvk::kHeapGuardCopyVersion, dst, bytes);  // a copy-back's destination is never GX's upload heap (stage 1d.1)
    copy_version_add(v);
    bump(g_cv_made);  // not a locked add: that waits for the copy to leave the write-combining buffers
    return true;
}

// A batch of its own, closed here: the callers that end a pass go on to
// record other work without looking for an open transfer batch.
void copy_versions_flush_locked() {
    if (g_copy_versions.empty()) return;
    begin_recording_locked();  // made between recordings, their copies go in the next
    std::vector<CopyVersion> pending;
    pending.swap(g_copy_versions);
    g_cv_sorted.clear();
    g_cv_lo = ~0ull;
    g_cv_hi = 0;
    std::optional<DrawCmds> own;
    if (!DrawCmds::open() && !stream_on()) own.emplace(!g.profile && !g.has_checkpoints);  // with the stream, ops of their own
    if (DrawCmds* c = DrawCmds::open(); !c || !c->transfer_begin()) rec().transfer_barrier(true);
    // Newest first, each byte from the newest version that has it: copies in
    // one batch run in any order, and a range GX copied back twice since the
    // last batch (most of them, a constant buffer a draw) was two writes of
    // the same bytes, the older of which could land last.
    static std::map<std::uint64_t, std::uint64_t> covered;  // start -> end, disjoint, of the newer versions
    covered.clear();
    std::size_t copies = 0;
    for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
        const CopyVersion& v = *it;
        // The buffer shadow sees the GPU write - unless it lands in a twin,
        // which leaves guest memory as it was.
        if (!twin_holds_live_locked(v.dst, v.bytes)) mark_surfaces_dirty_locked(v.dst, v.bytes, true, "copy version");
        const std::uint64_t end = v.dst + v.bytes;
        auto copy_piece = [&](std::uint64_t lo, std::uint64_t hi) {
            const VkBufferCopy region{v.offset + (lo - v.dst), v.canon.offset + (lo - v.dst), hi - lo};
            DrawCmds* c = DrawCmds::open();
            if (!c || !c->copy_buffer(v.copy.buffer, v.canon.buffer, region)) rec().copy_buffer(v.copy.buffer, v.canon.buffer, 1, &region);
            transfer_end_locked();
            ++copies;
        };
        auto c = covered.upper_bound(v.dst);
        if (c != covered.begin() && std::prev(c)->second > v.dst) --c;
        for (std::uint64_t at = v.dst; at < end;) {
            if (c == covered.end() || c->first >= end) {
                copy_piece(at, end);
                break;
            }
            if (c->first > at) copy_piece(at, c->first);
            at = std::max(at, c->second);
            ++c;
        }
        std::uint64_t lo = v.dst, hi = end;
        auto m = covered.upper_bound(lo);
        if (m != covered.begin() && std::prev(m)->second >= lo) --m;
        while (m != covered.end() && m->first <= hi) {
            lo = std::min(lo, m->first);
            hi = std::max(hi, m->second);
            m = covered.erase(m);
        }
        covered.emplace(lo, hi);
    }
    g_cv_overwritten.fetch_add(pending.size() > copies ? pending.size() - copies : 0, std::memory_order_relaxed);
    if (DrawCmds* c = DrawCmds::open(); !c || !c->transfer_end()) rec().transfer_barrier(false);
    g_cv_batches.fetch_add(1, std::memory_order_relaxed);
    g_cv_copies.fetch_add(copies, std::memory_order_relaxed);
}

std::string copy_versions_report() {
    if (!g_cv_made.load() && !g_cv_refused.load()) return "";
    using ull = unsigned long long;
    char b[800];
    std::snprintf(b, sizeof(b),
                  "copy versions: %llu made (%llu read an earlier one), %llu bindings took one, %llu bindings across one's edge, "
                  "%llu copies refused (size %llu, surfaces %llu, targets %llu, not imported %llu, source %llu, other %llu); "
                  "%llu copies into place in %llu batches (%llu versions all overwritten by newer ones), of them early for a binding "
                  "across an edge %llu, a page-table read %llu, a page-table pipeline %llu, a retired import %llu, a twin let go %llu; "
                  "%llu transfers waited for an earlier one in their batch",
                  static_cast<ull>(g_cv_made.load()), static_cast<ull>(g_cv_from_version.load()), static_cast<ull>(g_cv_bound.load()),
                  static_cast<ull>(g_cv_straddled.load()), static_cast<ull>(g_cv_refused.load()), static_cast<ull>(g_cv_refused_why[kCvSize].load()),
                  static_cast<ull>(g_cv_refused_why[kCvSurfaces].load()), static_cast<ull>(g_cv_refused_why[kCvTarget].load()),
                  static_cast<ull>(g_cv_refused_why[kCvImport].load()), static_cast<ull>(g_cv_refused_why[kCvSource].load()),
                  static_cast<ull>(g_cv_refused_why[kCvOther].load()), static_cast<ull>(g_cv_copies.load()),
                  static_cast<ull>(g_cv_batches.load()), static_cast<ull>(g_cv_overwritten.load()), static_cast<ull>(g_cv_put[kCvStraddle].load()),
                  static_cast<ull>(g_cv_put[kCvPageTable].load()), static_cast<ull>(g_cv_put[kCvWalks].load()),
                  static_cast<ull>(g_cv_put[kCvRetired].load()), static_cast<ull>(g_cv_put[kCvPutTwin].load()),
                  static_cast<ull>(g_batch_orders.load()));
    return b;
}

}  // namespace gpu

bool host_gpu_copy_back(std::uint64_t dst, std::uint64_t src, std::size_t bytes) {
    {
        std::lock_guard<GpuMutex> lock(g.mu);
        if (!g.ok) return false;
        if (!bytes) return true;
        if (copy_version_make_locked(dst, src, bytes)) return true;
    }
    return host_gpu_copy_guest(dst, src, bytes);
}

std::size_t host_gpu_copy_backs(const std::uint64_t* dst, const std::uint64_t* src, const std::uint64_t* bytes, std::size_t n) {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok) return 0;
    std::size_t i = 0;
    for (; i < n; ++i) {
        if (bytes[i] && !copy_version_make_locked(dst[i], src[i], bytes[i])) break;
    }
    return i;
}

std::uint64_t host_gpu_work_needs() {
    std::lock_guard<GpuMutex> lock(g.mu);
    // Submissions are numbered from 0 in `flushes`; the recording under way
    // becomes the next one.
    return g.recording ? g.flushes + 1 : g.flushes;
}

std::uint64_t host_gpu_submissions_completed() {
    if (g_timeline_on.load(std::memory_order_acquire)) {
        std::uint64_t v = 0;
        if (g_device_lost.load(std::memory_order_relaxed) || vkGetSemaphoreCounterValue(g.device, g.timeline, &v) != VK_SUCCESS)
            v = ~0ull;  // nothing more will run: nothing to wait for
        // What counts as run has its labels in memory (label_landing.h).
        if (landing_on()) label_landing::land_through(v);
        return v;
    }
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok) return ~0ull;  // nothing more will run: nothing to wait for
    std::uint64_t done = g.completed_submits;
    for (const Gpu::Slot& sl : g.slots) {
        // Only a fence already submitted (bb-submit may be submitting it now).
        if (sl.in_flight && sl.serial + 1 > done && sl.serial < stream_submitted_serial() && vkGetFenceStatus(g.device, sl.fence) == VK_SUCCESS) {
            done = sl.serial + 1;
        }
    }
    return done;
}

void* host_gpu_timeline() { return g_timeline_on.load(std::memory_order_acquire) ? g.timeline : VK_NULL_HANDLE; }
void* host_gpu_present_timeline() { return g_timeline_on.load(std::memory_order_acquire) ? g.present_timeline : VK_NULL_HANDLE; }
std::uint32_t host_gpu_vendor_id() { return g.vendor_id; }
void host_gpu_present_reads_locked(std::uint64_t target) {
    if (!g.slot_count) return;
    Gpu::Slot& sl = g.slots[g.slot];
    sl.present_target = std::max(sl.present_target, target);
}

bool host_gpu_wait_completed(std::uint64_t need, std::uint64_t timeout_ns) {
    if (!g_timeline_on.load(std::memory_order_acquire)) return false;
    if (g_device_lost.load(std::memory_order_relaxed)) {
        if (landing_on()) label_landing::land_through(~0ull);
        return true;
    }
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &g.timeline;
    wi.pValues = &need;
    const bool done = vkWaitSemaphores(g.device, &wi, timeout_ns) == VK_SUCCESS;
    if (done && landing_on()) label_landing::land_through(need);  // its labels in memory before the caller reads it
    return done;
}

void host_gpu_report() {
    dispatch_cost_report();

    if (!g.tried) {
        return;
    }
    log_validation_census();
    {
        // Guest memory the GPU reaches as imports: how much, made by what, and
        // with window imports what replaced what and what the prefetch did.
        std::uint64_t live = 0, flex = 0;
        std::size_t nlive = 0;
        {
            std::lock_guard<GpuMutex> lock(g.mu);
            for (const DmemChunk& d : g.dmem) {
                for (const auto& c : d.imports) live += c->size;
                nlive += d.imports.size();
            }
            for (const auto& kv : g.anon_imports) flex += kv.second.size;
        }
        using ull = unsigned long long;
        const auto mib = [](const std::atomic<std::uint64_t>& v) { return static_cast<ull>(v.load() >> 20); };
        host_log("gpu: guest memory imports (%s): %llu made, %llu MiB in %llu ms of the driver's time - bindings %llu (%llu MiB), device "
                 "addresses %llu (%llu MiB), walks %llu (%llu MiB), prefetch %llu (%llu MiB), whole spans %llu (%llu MiB); %llu replaced "
                 "others, %llu retired (%llu MiB); live %llu MiB in %zu imports, flexible %llu MiB",
                 imports_whole() ? "whole spans" : "by window: BBHOST_IMPORT_WINDOWS=1, gpu.import_windows or an integrated GPU under Windows",
                 static_cast<ull>(g_imports_made.load()),
                 mib(g_import_bytes), static_cast<ull>(g_import_us.load() / 1000), static_cast<ull>(g_import_who_n[kIwBinding].load()),
                 mib(g_import_who_bytes[kIwBinding]), static_cast<ull>(g_import_who_n[kIwAddress].load()), mib(g_import_who_bytes[kIwAddress]),
                 static_cast<ull>(g_import_who_n[kIwWalk].load()), mib(g_import_who_bytes[kIwWalk]),
                 static_cast<ull>(g_import_who_n[kIwPrefetch].load()), mib(g_import_who_bytes[kIwPrefetch]),
                 static_cast<ull>(g_import_who_n[kIwSpan].load()), mib(g_import_who_bytes[kIwSpan]), static_cast<ull>(g_imports_merged.load()),
                 static_cast<ull>(g_imports_retired.load()), mib(g_import_retired_bytes), static_cast<ull>(live >> 20), nlive,
                 static_cast<ull>(flex >> 20));
        if (const std::string fb = shadow_import_fallback_report(); g_import_failed.load() || g_import_fail_test.mode || !fb.empty())
            host_log("gpu: guest memory imports failed %llu (%llu MiB asked%s) - bindings %llu, device addresses %llu, walks %llu, prefetch %llu, "
                     "whole spans %llu; a window asked again within %llu ms of a failure, read through the page table instead %llu%s",
                     static_cast<ull>(g_import_failed.load()), mib(g_import_failed_bytes),
                     g_import_fail_test.mode ? (", " + std::to_string(g_import_forced.load()) + " of them by BBHOST_TEST_IMPORT_FAIL").c_str() : "",
                     static_cast<ull>(g_import_failed_who[kIwBinding].load()), static_cast<ull>(g_import_failed_who[kIwAddress].load()),
                     static_cast<ull>(g_import_failed_who[kIwWalk].load()), static_cast<ull>(g_import_failed_who[kIwPrefetch].load()),
                     static_cast<ull>(g_import_failed_who[kIwSpan].load()), static_cast<ull>(kImportBackoffMs),
                     static_cast<ull>(g_import_backed_off.load()), fb.c_str());
        // Who waited for the driver: the import thread's imports keep nobody waiting.
        const auto ms = [](const std::atomic<std::uint64_t>& us) { return static_cast<double>(us.load()) / 1e3; };
        host_log("gpu: guest memory imports waited for (the driver's time of the thread that asked): by the paths that needed them %llu in %.0f ms "
                 "(bindings %llu in %.0f, device addresses %llu in %.0f, walks %llu in %.0f; the longest %.1f ms), whole spans %llu in %.0f ms; "
                 "the import thread's %llu in %.0f ms",
                 static_cast<ull>(g_import_who_n[kIwBinding].load() + g_import_who_n[kIwAddress].load() + g_import_who_n[kIwWalk].load()),
                 ms(g_import_who_us[kIwBinding]) + ms(g_import_who_us[kIwAddress]) + ms(g_import_who_us[kIwWalk]),
                 static_cast<ull>(g_import_who_n[kIwBinding].load()), ms(g_import_who_us[kIwBinding]), static_cast<ull>(g_import_who_n[kIwAddress].load()),
                 ms(g_import_who_us[kIwAddress]), static_cast<ull>(g_import_who_n[kIwWalk].load()), ms(g_import_who_us[kIwWalk]),
                 ms(g_import_wait_max_us), static_cast<ull>(g_import_who_n[kIwSpan].load()), ms(g_import_who_us[kIwSpan]),
                 static_cast<ull>(g_import_who_n[kIwPrefetch].load()), ms(g_import_who_us[kIwPrefetch]));
        if (!imports_whole()) {
            host_log("gpu: import prefetch: %llu asked, %llu already imported, %llu looked at, %llu imports made (%llu MiB), %llu dropped as a "
                     "binding imported first, %llu not queued; dispatches: %llu reads imported, %llu with a base that does not trace "
                     "(their walks reach what nothing imports for them); meshes the buffer shadow reads in place "
                     "(BBHOST_IMPORT_PREFETCH_MESHES): %llu asked, %llu dropped while it does not, %llu imports made (%llu MiB)",
                     static_cast<ull>(g_pf_asked.load()), static_cast<ull>(g_pf_known.load()), static_cast<ull>(g_pf_checked.load()),
                     static_cast<ull>(g_pf_imports.load()), mib(g_pf_bytes), static_cast<ull>(g_pf_raced.load()), static_cast<ull>(g_pf_dropped.load()),
                     static_cast<ull>(g_dispatch_read_imports.load()), static_cast<ull>(g_dispatch_untraced.load()),
                     static_cast<ull>(g_pf_mesh_asked.load()), static_cast<ull>(g_pf_mesh_skipped.load()), static_cast<ull>(g_pf_mesh_imports.load()),
                     mib(g_pf_mesh_bytes));
        }
        // (A copy taken under the lock: the command processor may still be
        // making imports while the exit report runs.)
        std::vector<std::pair<std::string, ImportAsker>> askers;
        if (!imports_whole()) {
            std::lock_guard<GpuMutex> lock(g.mu);
            askers.assign(g_import_askers.begin(), g_import_askers.end());
        }
        if (!askers.empty()) {
            // The most bytes first, a line each; the mapping the first import of a kind lay in.
            std::sort(askers.begin(), askers.end(), [](const auto& x, const auto& y) { return x.second.bytes > y.second.bytes; });
            host_log("gpu: window imports by the path that asked (binding, device address, walk, prefetch; the site) and what lies at its "
                     "address, %zu kinds:",
                     askers.size());
            for (const auto& [kind, a] : askers) {
                host_log("gpu:   %s: %llu (%llu MiB), the first at 0x%llx in the mapping 0x%llx + %llu MiB (prot 0x%x)", kind.c_str(),
                         static_cast<ull>(a.n), static_cast<ull>(a.bytes >> 20), static_cast<ull>(a.first_va), static_cast<ull>(a.map_va),
                         static_cast<ull>(a.map_len >> 20), a.map_prot);
            }
        }
        if (!g_census) {
            host_log("gpu: guest memory census off (BBHOST_IMPORT_CENSUS=1 counts the imports by the path that asked, what reads or writes "
                     "guest memory in place and the GX buffers GPU copies write)");
        } else {
            std::vector<SiteUse> uses;
            std::vector<std::pair<std::string, std::pair<std::uint64_t, std::uint64_t>>> copied;  // kind -> buffers, bytes
            std::uint64_t copied_other = 0, copies_total = 0;
            {
                std::lock_guard<GpuMutex> lock(g.mu);
                for (const SiteUse& u : g_site_uses) {
                    if (u.site) uses.push_back(u);
                }
                std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> by;
                for (const auto& [gen, c] : g_copied_buffers) {
                    static const char* const usages[] = {"default", "immutable", "dynamic", "staging"};
                    char k[48];
                    std::snprintf(k, sizeof(k), "%s bind 0x%x", c.usage < 4 ? usages[c.usage] : "?", c.bind);
                    auto& e = by[k];
                    ++e.first;
                    e.second += c.bytes;
                    copies_total += c.copies;
                }
                copied.assign(by.begin(), by.end());
                copied_other = g_copied_other;
            }
            std::sort(uses.begin(), uses.end(), [](const SiteUse& a, const SiteUse& b) { return a.n > b.n; });
            if (!uses.empty()) {
                host_log("gpu: guest memory served in place (a locate or device address an import answered), by the path that asked, %zu paths:",
                         uses.size());
                for (const SiteUse& u : uses) {
                    host_log("gpu:   %s: %llu (%llu MiB asked)", u.site, static_cast<ull>(u.n), static_cast<ull>(u.bytes >> 20));
                }
            }
            std::string cb;
            for (const auto& [k, e] : copied) {
                char b[96];
                std::snprintf(b, sizeof(b), "%s%s %llu (%llu KiB)", cb.empty() ? "" : "; ", k.c_str(), static_cast<ull>(e.first),
                              static_cast<ull>(e.second >> 10));
                cb += b;
            }
            host_log("gpu: GX buffers GPU copies wrote (copy tokens, copy versions; %llu copies): %s; copies into no GX buffer %llu",
                     static_cast<ull>(copies_total), cb.empty() ? "none" : cb.c_str(), static_cast<ull>(copied_other));
        }
        if (!imports_whole()) {
            host_log("gpu: copy tokens' sources copied into upload staging (window imports): %llu (%llu MiB)",
                     static_cast<ull>(g_copy_src_staged.load()), static_cast<ull>(g_copy_src_staged_bytes.load() >> 20));
        }
        if (const std::string tw = twins_report(); !tw.empty()) host_log("%s", tw.c_str());
        // GX buffers by handle (host/gxvk.h): one line each.
        for (std::string gx = gxvk_report(); !gx.empty();) {
            const std::size_t nl = gx.find('\n');
            host_log("%s", gx.substr(0, nl).c_str());
            gx = nl == std::string::npos ? std::string() : gx.substr(nl + 1);
        }
    }
    {
        // Before the census lock: compute_lift_report takes g.mu, and a
        // dispatch takes the census lock under g.mu (host_gpu_dispatch) -
        // the other order hung a timed exit with the CP still dispatching.
        const std::string lift = compute_lift_report();
        const std::string landing = label_landing::report();
        const std::string landing_reused = label_landing::on() ? label_landing::reuse_report() : std::string();
        const std::string label_reads = label_check_report();
        std::lock_guard<std::mutex> lk(g_dispatch_census_mu);
        std::vector<std::pair<std::uint64_t, std::string>> by_count;
        for (const auto& [n, c] : g_dispatch_census) by_count.emplace_back(c, n);
        std::sort(by_count.rbegin(), by_count.rend());
        host_log("gpu:   recognised as an image copy: %llu image-to-image, %llu buffer-to-image (they still run their shader)",
                 static_cast<unsigned long long>(g_dispatch_image_blits.load()),
                 static_cast<unsigned long long>(g_dispatch_buf_to_image.load()));
        host_log("gpu: %s", indirect_clamp_report().c_str());
        host_log("gpu: command-arena chunk retirements that came after a label into their own chunk in the same unordered "
                 "fills, given a barrier ahead of the marker's 0: %llu",
                 static_cast<unsigned long long>(g_arena_retires_ordered.load()));
        host_log("gpu: %s", landing.c_str());
        if (!landing_reused.empty()) host_log("gpu: %s", landing_reused.c_str());
        if (!landing_on()) {
            host_log("gpu: command-processor writes recorded as fills into memory the game had written again since their submission (a "
                     "command buffer begun again): %llu, %llu of them within what its new round had written; filled by the GPU over "
                     "what a new round had written there (found once their submission retired): %llu, %llu of them before that round's "
                     "submission; not filled %llu (BBHOST_LABEL_REUSED_DROP=%d)",
                     static_cast<unsigned long long>(g_fills_reused.load()), static_cast<unsigned long long>(g_fills_reused_over.load()),
                     static_cast<unsigned long long>(g_fills_landed_over.load()),
                     static_cast<unsigned long long>(g_fills_landed_unsubmitted.load()), static_cast<unsigned long long>(g_fills_dropped.load()),
                     label_landing::drop_reused() ? 1 : 0);
        }
        if (!label_reads.empty()) host_log("gpu: %s", label_reads.c_str());
        host_log("gpu: %s", lift.c_str());
        host_log("gpu: dispatch census: %llu reached a shader over %zu programs, %llu were a fill and %llu a copy (no shader runs "
                 "for those)",
                 static_cast<unsigned long long>(g_dispatch_shaders.load()), g_dispatch_census.size(),
                 static_cast<unsigned long long>(g_dispatch_fills.load()), static_cast<unsigned long long>(g_dispatch_copies.load()));
        for (std::size_t k = 0; k < by_count.size() && k < 16; ++k) {
            host_log("gpu:   %8llu %s", static_cast<unsigned long long>(by_count[k].first), by_count[k].second.c_str());
        }
    }
    if (const std::uint64_t n = g_cs_records_built.load() + g_cs_records_failed.load()) {
        host_log("gpu: dispatch tokens with compute records: %llu, user data built from them %llu, not built (the token's kept) %llu",
                 static_cast<unsigned long long>(n), static_cast<unsigned long long>(g_cs_records_built.load()),
                 static_cast<unsigned long long>(g_cs_records_failed.load()));
    }
    if (host_gpu_gds_counters()) {
        host_log("gpu: GDS counters: %llu dispatches counted in bound words, %llu counted in GDS with none bound; %llu counts set and %llu copied "
                 "in first (%llu could not be); %llu counter copies and %llu fills from the command processor",
                 static_cast<unsigned long long>(g_gds_dispatches.load()), static_cast<unsigned long long>(g_gds_unbound.load()),
                 static_cast<unsigned long long>(g_gds_sets.load()), static_cast<unsigned long long>(g_gds_copies.load()),
                 static_cast<unsigned long long>(g_gds_start_failed.load()), static_cast<unsigned long long>(g_counter_copies.load()),
                 static_cast<unsigned long long>(g_counter_fills.load()));
    }
    if (g.cmd_buffer_marker && g.marker_buf.map) {
        std::uint32_t started = 0, finished = 0;
        std::memcpy(&started, g.marker_buf.map, 4);
        std::memcpy(&finished, static_cast<const std::uint8_t*>(g.marker_buf.map) + 4, 4);
        host_log("gpu: AMD markers at exit: last started %s %u, everything finished before %s %u",
                 (started >> 30) == 1 ? "draw" : (started >> 30) == 2 ? "dispatch" : "-", started & 0x3fffffff,
                 (finished >> 30) == 1 ? "draw" : (finished >> 30) == 2 ? "dispatch" : "-", finished & 0x3fffffff);
    }
    host_log("gpu: dispatches=%llu draws=%llu draw-failures=%llu flushes=%llu failures=%llu pipelines=%llu/%llu dummy-images=%llu gpu-time=%llu ms",
             static_cast<unsigned long long>(g.dispatches.load()), static_cast<unsigned long long>(g.draws.load()),
             static_cast<unsigned long long>(g.draw_failures.load()), static_cast<unsigned long long>(g.flushes),
             static_cast<unsigned long long>(g.failures.load()), static_cast<unsigned long long>(g.translated.load()),
             static_cast<unsigned long long>(g.gfx_pipelines.load()), static_cast<unsigned long long>(g.dummy_images.load()),
             static_cast<unsigned long long>(g.gpu_us.load() / 1000));
    if (const std::string busy = busy_exit_report(); !busy.empty()) host_log("%s", busy.c_str());
    if (g.draw_failures.load()) {
        std::string line;
        for (int k = 0; k < kFailCount; ++k) {
            if (const std::uint64_t n = g.draw_fail_why[k].load()) {
                line += (line.empty() ? "" : ", ") + std::string(host_gpu_draw_fail_name(k)) + " " + std::to_string(n);
            }
        }
        host_log("gpu: draw failures by reason: %s", line.c_str());
    }
    if (glitch_on()) host_log("%s", glitch_report().c_str());
    if (occlusion_on()) host_log("%s", occlusion_report(true).c_str());
    host_gpu_log_video_memory(true);
    if (vb_race_on()) {
        std::lock_guard<GpuMutex> lock(g.mu);
        vb_race_exit_report_locked();
    }
    render_report();
}
