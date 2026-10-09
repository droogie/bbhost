#include "core/config.h"
#include "core/host_clock.h"
#include "core/portable.h"
#include "host/foreign_hooks.h"
#include "host/gpu_internal.h"
#include "host/shader_patch.h"
#include "host/translation_cache.h"
#include "bbhost_build_id.h"

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

void device_lost_locked(const char* where);  // below, with the reports it makes
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

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void*) {
    if (severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        return VK_FALSE;
    }
    g_validation_messages.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 40) {
        host_log("vk validation: %s", data && data->pMessage ? data->pMessage : "?");
    }
    return VK_FALSE;
}

// BBHOST_IMPORT_WINDOWS=1: direct memory imported a window at a time, on
// demand. Off by default: a tenth of the draws keep page-table walks through
// V#s and pointers the host does not resolve (the exit report's "walk" lines),
// and what they reach must already be imported or they read the sink. Off,
// each chunk's GPU-visible span is imported whole when the mappings are made.
const bool g_import_whole = [] {
    const char* e = std::getenv("BBHOST_IMPORT_WINDOWS");
    return !(e && e[0] == '1');
}();
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
constexpr std::uint64_t kForeignTicket = 1ull << 63;  // host_gpu_submit_presenter's tickets for a kForeignSubmit

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

void submit_thread() {
    host_thread_set_name("bb-submit");
    host_thread_set_class(HostThreadClass::GpuFeed, "bb-submit");  // core/host_clock.h
    for (;;) {
        SubmitJob job;
        {
            std::unique_lock<std::mutex> lk(g_sub_mu);
            g_sub_cv.wait(lk, [] { return !g_sub_jobs.empty(); });
            job = g_sub_jobs.front();
            g_sub_jobs.pop_front();
            g_sub_busy = true;
        }
        submit_job_now(job);
        std::lock_guard<std::mutex> lk(g_sub_mu);
        g_sub_busy = false;
        g_sub_submitted.store(job.ticket, std::memory_order_release);
        g_sub_idle.notify_all();  // a drain, or a wait for this ticket
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
constexpr std::uint64_t kDmabufPiece = 64ull << 20;

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
    const bool sparse = size > kDmabufPiece;
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
    if (req.size != size || (sparse && kDmabufPiece % req.alignment)) {
        host_log("gpu: a buffer over %llu bytes of direct memory wants %llu (alignment %llu)", static_cast<unsigned long long>(size),
                 static_cast<unsigned long long>(req.size), static_cast<unsigned long long>(req.alignment));
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::vector<VkSparseMemoryBind> binds;
    for (std::uint64_t at = 0; at < size; at += kDmabufPiece) {
        const std::uint64_t n = std::min(kDmabufPiece, size - at);
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

// Imports [lo, hi) of direct-memory chunk `index` (offsets in the chunk) as a
// new import of the chunk's. Earlier imports of the same bytes stay alive:
// descriptor sets and recordings made with them keep working - they map the
// same host pages.
Chunk* import_span(std::uint32_t index, std::uint64_t lo, std::uint64_t hi) {
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
#if defined(_WIN32)
    const VkResult r = host_span(*c, ptr, size);
#else
    const VkResult r = g.dmem_dmabuf ? dmabuf_span(*c, offset, size) : host_span(*c, ptr, size);
#endif
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    if (r != VK_SUCCESS) {
        host_log("gpu: importing dmem chunk %u (%llu MiB from +%llu MiB) failed after %lld us: %d", index,
                 static_cast<unsigned long long>(size >> 20), static_cast<unsigned long long>(lo >> 20), static_cast<long long>(us), r);
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
    if (g_imports_made.fetch_add(1, std::memory_order_relaxed) < 8 || size > (256ull << 20)) {
        host_log("gpu: dmem chunk %u: +%llu MiB, %llu MiB imported at device address 0x%llx in %lld us", index,
                 static_cast<unsigned long long>(lo >> 20), static_cast<unsigned long long>(size >> 20),
                 static_cast<unsigned long long>(c->address), static_cast<long long>(us));
    }
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
            host_log("gpu: the driver refuses direct memory as a host pointer (%d); importing it as dma-bufs (/dev/udmabuf)%s", r,
                     g.dmabuf_sparse ? "" : ", 64 MiB at most: no sparse binding");
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

const Chunk* dmem_import(std::uint32_t index, std::uint64_t in, std::uint64_t bytes) {
    if (index >= gcn::kDmemChunks) return nullptr;
    DmemChunk& d = g.dmem[index];
    if (!bytes) bytes = 1;
    if (in >= kChunkBytes || bytes > kChunkBytes - in) return nullptr;
    const std::uint32_t w0 = static_cast<std::uint32_t>(in >> kImportWindowShift);
    if (const Chunk* c = d.window[w0]; c && chunk_covers(*c, in, bytes)) return c;
    if (in < d.span_lo || in + bytes > d.span_hi) return nullptr;  // not GPU-visible: the sink
    const std::uint32_t w1 = static_cast<std::uint32_t>((in + bytes - 1) >> kImportWindowShift);
    const std::uint64_t lo = std::max(d.span_lo, static_cast<std::uint64_t>(w0) << kImportWindowShift);
    const std::uint64_t hi = std::min(d.span_hi, static_cast<std::uint64_t>(w1 + 1) << kImportWindowShift);
    bool merged = false;
    for (std::uint32_t w = w0; w <= w1; ++w) merged |= d.window[w] != nullptr;
    Chunk* c = import_span(index, lo, hi);
    if (!c) return nullptr;
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

void import_windows(std::uint64_t va, std::uint64_t bytes) {
    const std::uint64_t end = va + bytes;
    for (const GuestMapInfo& mi : g.maps) {
        if (!mi.dmem) continue;
        const std::uint64_t lo = std::max(va, mi.va), hi = std::min(end, mi.va + mi.len);
        for (std::uint64_t at = lo; at < hi;) {
            const std::uint64_t phys = static_cast<std::uint64_t>(mi.phys) + (at - mi.va);
            dmem_import(static_cast<std::uint32_t>(phys >> gcn::kDmemChunkShift), phys & (kChunkBytes - 1), 1);
            at += (1ull << kImportWindowShift) - (phys & ((1ull << kImportWindowShift) - 1));  // the next window
        }
    }
}

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

// The 2D dummy above covers most bindings; shaders also declare 1D, 3D, cube
// and arrayed images, and Vulkan requires the view type to match exactly. Make
// a 1x1 image for each so no declared binding is ever left unwritten.
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
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = k.flags;
        ici.imageType = k.type;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {1, 1, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = k.layers;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
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
        // Undefined contents are fine for a binding the shader should not be
        // reading; what matters is that the descriptor is defined. Move it to
        // GENERAL so the layout matches what the descriptor claims.
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
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &b);
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
        g.dummy_dim_images[key] = image;
        g.dummy_dim_mem[key] = mem;
        g.dummy_dim_views[key] = view;
    }
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
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g.device, &ici, nullptr, &g.dummy_image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, g.dummy_image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, 0);
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
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
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

// <data root>/bbhost/translation-cache.bin (host/translation_cache.h), the same.
std::string translation_cache_path() {
    std::string path = pipeline_cache_path();
    if (path.empty()) return path;
    return path.substr(0, path.rfind('/') + 1) + "translation-cache.bin";
}

// The shader caches belong to the build that made them. A translation is keyed
// by the translator's sources and the options hash_options names, a pipeline
// by the driver - neither by the rest of the host, so a change there (a lift,
// an option missing from the key) would leave the next build running the last
// one's shaders. A build other than the one named in
// <data root>/bbhost/shader-cache-build.txt starts both empty. The stage
// manifest stays: it holds the guest's inputs, which the running build
// translates again on the precompile workers, filling the caches before the
// title. BBHOST_KEEP_SHADER_CACHE=1 keeps them across builds.
static void shader_caches_check_build() {
    const std::string pipelines = pipeline_cache_path();
    if (pipelines.empty()) return;
    const std::string dir = pipelines.substr(0, pipelines.rfind('/') + 1);
    const std::string stamp = dir + "shader-cache-build.txt";
    const std::string ours = BBHOST_BUILD_ID;
    std::string theirs;
    {
        std::ifstream in(stamp);
        std::getline(in, theirs);
    }
    if (theirs == ours) return;
    std::error_code ec;
    if (const char* e = std::getenv("BBHOST_KEEP_SHADER_CACHE"); e && e[0] == '1') {
        host_log("gpu: shader caches made by %s kept for %s (BBHOST_KEEP_SHADER_CACHE=1)", theirs.empty() ? "an earlier build" : theirs.c_str(),
                 ours.c_str());
    } else {
        std::uintmax_t freed = 0;
        int removed = 0;
        for (const std::string& path : {pipelines, translation_cache_path()}) {
            const std::uintmax_t bytes = std::filesystem::file_size(path, ec);
            if (ec) continue;
            if (std::filesystem::remove(path, ec)) freed += bytes, ++removed;
        }
        if (removed)
            host_log("gpu: shader caches made by %s, this is %s: cleared (%.1f MiB); the precompile builds them again", theirs.empty() ? "an earlier build" : theirs.c_str(),
                     ours.c_str(), freed / 1048576.0);
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
        if (g_import_whole && !(d.window[lo >> kImportWindowShift] && chunk_covers(*d.window[lo >> kImportWindowShift], lo, hi - lo))) {
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
            if (Chunk* c = import_span(k, lo, ext)) {
                for (std::uint64_t w = lo >> kImportWindowShift; w <= (hi - 1) >> kImportWindowShift; ++w) d.window[w] = c;
            }
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

void audit_range(std::uint64_t va, std::uint64_t bytes, ImportUse use) {
    if (!bytes) return;
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
        const Chunk* c = dmem_import(chunk, in, std::min<std::uint64_t>({bytes, mi.len - off, kChunkBytes - in}));
        if (!c) return out;
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
        bool host_import = false;
        for (const VkExtensionProperties& e : exts) {
            if (std::strcmp(e.extensionName, "VK_EXT_external_memory_host") == 0) host_import = true;
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
            if ((end && *end == 0 && n == static_cast<long>(index)) || lname.find(lwant) != std::string::npos) chosen = static_cast<int>(usable.size());
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
            mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
            mci.pfnUserCallback = debug_callback;
            create(g.instance, &mci, nullptr, &g.messenger);
        }
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
        g.tight = memory_tight(c);
        if (g_tight.exchange(g.tight ? 1 : 0) == (g.tight ? 0 : 1))
            host_log("gpu: the device chosen has %s memory than the one the guest patches were sized for", g.tight ? "less" : "more");
        if (g.tight) host_log("gpu: %llu MiB in all heaps: tight memory (BBHOST_TIGHT_MEMORY=0 says otherwise) - no buffer shadow, fewer spares during loads, live resolution up to 1080p's pixels",
                              static_cast<unsigned long long>(c.heaps >> 20));
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
        // pipeline cache): these starts run with whichever this driver has.
        // BBHOST_GPU_CHECKPOINTS=0 keeps them off.
        std::string why;
        if (!e && (c.checkpoints || c.buffer_marker) && take_marked_start(why)) {
            (c.checkpoints ? want_nv : want_amd) = true;
            host_log("gpu: the last run lost the device (%s): this start runs with the GPU's progress markers (%s), so another loss "
                     "names the draw the GPU was in; BBHOST_GPU_CHECKPOINTS=0 turns them off",
                     why.c_str(), c.checkpoints ? "NVIDIA's checkpoints" : "AMD's buffer markers");
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
        if (const char* e = std::getenv("BBHOST_TEST_VRAM_MB"); e && std::strtoull(e, nullptr, 10)) {
            std::uint32_t types = 0;
            for (std::uint32_t i = 0; i < g.mem_props.memoryTypeCount; ++i) {
                if (g.mem_props.memoryTypes[i].heapIndex == g.local_heap) types |= 1u << i;
            }
            bb_memory_test_limit(types, std::strtoull(e, nullptr, 10) << 20);
        }
    }
    g.timestamp_period_ns = props.limits.timestampPeriod;
    g.max_lds_bytes = props.limits.maxComputeSharedMemorySize;
    host_log("gpu: stage interface limits: vertex outputs %u, geometry outputs %u, fragment inputs %u components",
             props.limits.maxVertexOutputComponents, props.limits.maxGeometryOutputComponents, props.limits.maxFragmentInputComponents);
    g.max_anisotropy = props.limits.maxSamplerAnisotropy;
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
        if (supported.shaderInt16) f2.features.shaderInt16 = VK_TRUE;
        g.has_f16_math = q12.shaderFloat16 && supported.shaderInt16;
        gcn::set_native_half_rtz(native);
        host_log("gpu: f16 rounding toward zero by the device: %s (shaderFloat16 %d, RTZ %d, denorm preserve %d)", native ? "yes" : "no",
                 static_cast<int>(q12.shaderFloat16), static_cast<int>(p12.shaderRoundingModeRTZFloat16),
                 static_cast<int>(p12.shaderDenormPreserveFloat16));
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
    // build of bbhost throws it away itself (and its translations with it).
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
    host_log("gpu: params ring in memory type %u (flags 0x%x)", mai.memoryTypeIndex,
             mai.memoryTypeIndex < g.mem_props.memoryTypeCount ? g.mem_props.memoryTypes[mai.memoryTypeIndex].propertyFlags : 0u);
    if (vkAllocateMemory(g.device, &mai, nullptr, &g.ubo_mem) != VK_SUCCESS) {
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
    if (!create_dev_buffer(g.l1, gcn::kL1Entries * 8, true) || !create_dev_buffer(g.sink, 1ull << gcn::kPageShift, false) ||
        !create_dev_buffer(g.sink_l2, gcn::kL2Entries * 8, true)) {
        host_log("gpu: page table buffers failed");
        return false;
    }
    for (std::uint32_t k = 0; k < gcn::kL2Entries; ++k) static_cast<std::uint64_t*>(g.sink_l2.map)[k] = g.sink.address;
    for (std::uint32_t k = 0; k < gcn::kL1Entries; ++k) static_cast<std::uint64_t*>(g.l1.map)[k] = g.sink_l2.address;

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
    translation_cache_load(translation_cache_path());
    g.ok = true;
    start_submit_thread();
    start_hang_watchdog();
    start_memory_reserve();
    stage_manifest_load(stage_manifest_path());
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
        busy_begin_locked(g.slots[g.slot]);
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

std::uint32_t alloc_params_slot_locked(const gcn::StageParams& params, VkDescriptorBufferInfo& out) {
    const std::uint32_t slot = g.queued++;
    const std::uint32_t index = static_cast<std::uint32_t>(g.slot) * kMaxQueued * kStageSlots + slot;
    std::memcpy(static_cast<std::uint8_t*>(g.ubo_map) + static_cast<std::size_t>(index) * kParamsStride, &params, sizeof(params));
    out = {g.ubo, static_cast<VkDeviceSize>(index) * kParamsStride, sizeof(gcn::StageParams)};
    return slot;
}

void rewrite_params_slot_locked(const gcn::StageParams& params, const VkDescriptorBufferInfo& slot) {
    std::memcpy(static_cast<std::uint8_t*>(g.ubo_map) + static_cast<std::size_t>(slot.offset), &params, sizeof(params));
}

// Sets come from the current slot's pool eight at a time: a draw takes two,
// and one vkAllocateDescriptorSets per set was ~3% of the command processor.
// The spares go with the pool's reset (retire_slot_locked); the pool has room
// for them (kSpareSets).
VkDescriptorSet alloc_set_locked(VkDescriptorSetLayout layout) {
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

VkImageView dummy_view_for(std::uint32_t dim, bool arrayed, bool zero) {
    if (zero && dim == 1 && !arrayed && g.dummy_zero_view) return g.dummy_zero_view;
    if (dim == 1 && !arrayed) return g.dummy_view;
    auto it = g.dummy_dim_views.find((dim << 1) | (arrayed ? 1u : 0u));
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
    bool other_memory = false;
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

gcn::TranslateOptions compute_options(std::uint32_t rsrc1, std::uint32_t rsrc2, const std::uint32_t* threads,
                                      const std::vector<std::pair<std::uint32_t, bool>>& dims, const std::vector<bool>& modes) {
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Compute;
    o.rsrc1 = rsrc1;
    o.rsrc2 = rsrc2;
    o.image_dims = dims;
    o.sampler_force_unnormalized = modes;
    o.max_lds_bytes = g.max_lds_bytes;
    o.cs_threads[0] = threads[0];
    o.cs_threads[1] = threads[1];
    o.cs_threads[2] = threads[2];
    o.exec_known = g.exec_known;
    return o;
}

struct PrecompiledCompute {
    struct Entry {
        bool done = false, failed = false;  // failed: nothing to take (a failed build, taken, or claimed by a dispatch)
        bool gave_up = false;               // a dispatch stopped waiting for it: its dispatches skip until it is done
        gcn::TranslateResult meta;
        VkShaderModule module = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        bool standin = false;  // unoptimized, the optimized one queued (first_compute_pipeline)
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
                          VkPipelineCache own_cache = VK_NULL_HANDLE) {
    if (how == CsBuild::kOptimizedCached && !g.cache_control) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.layout = g.pipe_layout;
    cpci.flags = how == CsBuild::kUnoptimized        ? VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT
                 : how == CsBuild::kOptimizedCached ? VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT
                                                     : 0;
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r;
    if (how == CsBuild::kOptimizedCached) {
        r = vkCreateComputePipelines(g.device, PipelineCacheUse().cache, 1, &cpci, nullptr, &p);  // VK_PIPELINE_COMPILE_REQUIRED: not there
    } else if (own_cache != VK_NULL_HANDLE) {
        CompilingScope compiling(watch, spirv);
        r = vkCreateComputePipelines(g.device, own_cache, 1, &cpci, nullptr, &p);
    } else {
        CompilingScope compiling(watch, spirv);
        r = vkCreateComputePipelines(g.device, PipelineCacheUse().cache, 1, &cpci, nullptr, &p);
    }
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
            p = create_compute(module, CsBuild::kOptimized, job.name + watch, &job.spirv, own);
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

void queue_compute_optimize(std::uint64_t key, const std::string& name, const std::vector<std::uint32_t>& spirv, std::uint64_t standin_us) {
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
        g_cs_opt.jobs.push_back(std::move(job));
    }
    g_cs_opt.cv.notify_one();
}

}  // namespace

VkPipeline first_compute_pipeline(VkShaderModule module, std::uint64_t opt_key, const std::string& name,
                                  const std::vector<std::uint32_t>& spirv, bool& standin, const std::string& watch) {
    standin = false;
    switch (cs_optimize_mode()) {
    case CsOptimize::kNever:
        return create_compute(module, CsBuild::kUnoptimized, watch, &spirv);
    case CsOptimize::kAlways:
        return create_compute(module, CsBuild::kOptimized, watch, &spirv);
    case CsOptimize::kBackground:
        break;
    }
    if (g_cs_opt.keep_unoptimized.count(name)) {
        g_cs_opt.kept.fetch_add(1, std::memory_order_relaxed);
        return create_compute(module, CsBuild::kUnoptimized, watch, &spirv);
    }
    {
        // Built this run already, for an earlier pipeline of the same key.
        std::lock_guard<std::mutex> lk(g_cs_opt.mu);
        if (const auto it = g_cs_opt.ready.find(opt_key); it != g_cs_opt.ready.end()) return it->second;
    }
    if (const VkPipeline p = create_compute(module, CsBuild::kOptimizedCached, watch, &spirv)) {
        g_cs_opt.from_cache.fetch_add(1, std::memory_order_relaxed);
        return p;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const VkPipeline p = create_compute(module, CsBuild::kUnoptimized, watch, &spirv);
    if (!p) return p;
    standin = true;
    g_cs_opt.standins.fetch_add(1, std::memory_order_relaxed);
    if (!t_compute_worker) g_cs_opt.standins_cp.fetch_add(1, std::memory_order_relaxed);
    queue_compute_optimize(opt_key, name, spirv, cs_us_since(t0));
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
        return pl;
    case Precompiled::kPending:
        pl.pending = pl.failed = true;
        g_cs_pre.skipped_dispatches.fetch_add(1, std::memory_order_relaxed);
        return pl;
    case Precompiled::kBuildHere:
        break;
    }
    pl.meta = translate_cached(prog, o);
    if (!pl.meta.ok()) {
        host_log("gpu: shader %s: %s", name.c_str(), pl.meta.errors[0].c_str());
        pl.failed = true;
        return pl;
    }
    if (const char* e = std::getenv("BBHOST_DUMP_SPIRV"); e && e[0] == '1') {
        host_mkdir("build/spv");
        const std::string path = "build/spv/" + name + "-cs.spv";
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(pl.meta.spirv.data(), 4, pl.meta.spirv.size(), f);
            std::fclose(f);
        }
        host_log("gpu: creating compute pipeline %s", name.c_str());
    }
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = pl.meta.spirv.size() * 4;
    smci.pCode = pl.meta.spirv.data();
    if (vkCreateShaderModule(g.device, &smci, nullptr, &pl.module) != VK_SUCCESS) {
        host_log("gpu: shader %s: module creation failed", name.c_str());
        pl.failed = true;
        return pl;
    }
    const auto t0 = std::chrono::steady_clock::now();
    pl.pipeline = first_compute_pipeline(pl.module, key, name, pl.meta.spirv, pl.standin, name + " (at its dispatch)");
    if (!pl.pipeline) {
        host_log("gpu: shader %s: pipeline creation failed", name.c_str());
        pl.failed = true;
        return pl;
    }
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
    return pl;
}

}  // namespace

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
    const std::uint64_t key = compute_key(fnv1a(code.words.data(), code.words.size() * 4), rsrc1, rsrc2, threads, dims, modes);
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
            return;
        }
        r.state = PrecompiledCompute::kCompiling;
    }
    // From here every way out marks the entry done: a dispatch may be waiting on it.
    PrecompiledCompute::Entry e;
    std::string why;
    e.meta = translate_cached(prog, compute_options(rsrc1, rsrc2, threads, dims, modes));
    if (!e.meta.ok()) why = "translation: " + e.meta.errors[0];
    if (why.empty()) {
        if (const char* env = std::getenv("BBHOST_DUMP_SPIRV"); env && env[0] == '1') {
            host_mkdir("build/spv");
            const std::string path = "build/spv/" + name + "-cs.spv";
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(e.meta.spirv.data(), 4, e.meta.spirv.size(), f);
                std::fclose(f);
            }
            // The program too, for `gcndis --raw` and `gcn2spv --raw ... cs <rsrc1> <rsrc2>`.
            if (FILE* f = std::fopen(("build/spv/" + name + "-cs.bin").c_str(), "wb")) {
                std::fwrite(code.words.data(), 4, code.words.size(), f);
                std::fclose(f);
            }
            if (FILE* f = std::fopen(("build/spv/" + name + "-cs.txt").c_str(), "w")) {
                std::fprintf(f, "rsrc1 0x%08x rsrc2 0x%08x threads %u %u %u\n", rsrc1, rsrc2, threads[0], threads[1], threads[2]);
                std::fclose(f);
            }
        }
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = e.meta.spirv.size() * 4;
        smci.pCode = e.meta.spirv.data();
        if (vkCreateShaderModule(g.device, &smci, nullptr, &e.module) != VK_SUCCESS) why = "shader module";
    }
    if (why.empty()) {
        // BBHOST_CS_COMPILE_DELAY_MS=<ms> (checks): a compiler that takes that
        // long over every compute pipeline built at creation - the dispatch's
        // bounded wait and the watchdog's line, tried on a fast driver.
        static const int delay_ms = [] {
            const char* d = std::getenv("BBHOST_CS_COMPILE_DELAY_MS");
            return d ? std::atoi(d) : 0;
        }();
        if (delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        t_compute_worker = true;
        e.pipeline = first_compute_pipeline(e.module, key, name, e.meta.spirv, e.standin, name);
        if (!e.pipeline) why = "pipeline";
    }
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
    if (!n) return;
    std::vector<VkCheckpointDataNV> data(n, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
    get_data(g.queue, &n, data.data());
    host_log("gpu: %u checkpoint(s) at the device loss (the last work each stage reached):", n);
    for (std::uint32_t i = 0; i < n && i < 16; ++i) {
        const std::uint64_t m = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data[i].pCheckpointMarker));
        const unsigned kind = static_cast<unsigned>(m >> 56);
        const std::uint64_t which = m & 0xffffffffffffffull;
        host_log("  stage 0x%x: %s %llu", data[i].stage, kind == 1 ? "draw" : kind == 2 ? "dispatch" : "?",
                 static_cast<unsigned long long>(which));
        if (kind == 1) describe_draw_record(which);
    }
}

// What the driver knows about the fault, when VK_EXT_device_fault is there.
void report_device_fault() {
    if (!g.has_device_fault) return;
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
void device_lost_locked(const char* where) {
    g.ok = false;
    // Nothing recorded will run now: land the CP writes still queued.
    for (const auto& [va, span] : g.pending_writes) {
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(va)), span.data.data(), span.data.size());
    }
    g.pending_writes.clear();
    if (g_device_lost.exchange(true)) return;
    const unsigned long long flip = static_cast<unsigned long long>(hle_video_flip_count());
    host_log("gpu: the device is lost (%s, flip %llu): the GPU stopped responding and the system reset it", where, flip);
    report_device_fault();
    report_checkpoints();
    if (!g.has_checkpoints) host_log("gpu: no progress markers this run; the next %d starts run with them", kMarkedRuns);
    // Everything stops after this, so say what was in flight: the last draws
    // name the pipeline that faulted or hung.
    host_gpu_hang_report();  // with the command stream's last op (recorder.cpp)
    const std::string hooks = host_foreign_hooks();
    if (!hooks.empty()) host_log("gpu: hooked into this process: %s - such hooks are a common cause of lost devices", hooks.c_str());
    const std::string path = device_lost_note_path();
    if (!path.empty()) {
        std::ofstream out(path, std::ios::trunc);
        out << "starts-left " << kMarkedRuns << '\n' << "flip " << flip << ", " << where << (hooks.empty() ? "" : ", with ") << hooks << '\n';
    }
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
    const auto t0 = std::chrono::steady_clock::now();
    g.wait_started_ns.store(static_cast<std::uint64_t>(t0.time_since_epoch().count()), std::memory_order_relaxed);
    g.waiting_slot.store(k, std::memory_order_relaxed);
    const VkResult r = vkWaitForFences(g.device, 1, &sl.fence, VK_TRUE, 10000000000ull);
    g.waiting_slot.store(-1, std::memory_order_relaxed);
    const auto fence_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    g.gpu_us.fetch_add(fence_us);
    bump(g_stream_stats.fence_ns, static_cast<std::uint64_t>(fence_us) * 1000);
    if (r != VK_SUCCESS) {
        host_log("gpu: waiting for submission %d failed (%d)%s", k, r,
                 r == VK_ERROR_DEVICE_LOST ? "; device lost, GPU execution disabled" : "");
        if (r == VK_ERROR_DEVICE_LOST) device_lost_locked("a wait for a submission");
    }
    vkResetFences(g.device, 1, &sl.fence);
    vkResetDescriptorPool(g.device, sl.pool, 0);
    sl.spare_sets.clear();
    for (DevBuffer& b : sl.garbage) {
        vkDestroyBuffer(g.device, b.buffer, nullptr);
        vkFreeMemory(g.device, b.memory, nullptr);
    }
    sl.garbage.clear();
    untile_checks_run_locked(sl.untile_checks);  // before their staging is reused
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
    // recorded into this submission or an earlier one is in memory now.
    g.completed_submits = std::max(g.completed_submits, sl.serial + 1);
    for (auto it = g.pending_writes.begin(); it != g.pending_writes.end();) {
        it = it->second.serial < g.completed_submits ? g.pending_writes.erase(it) : std::next(it);
    }
    collect_profile_locked(sl);
    busy_retired_locked(sl, r == VK_SUCCESS);
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
        const std::uint64_t want = std::max(need, kStagingChunkBytes);
        DevBuffer chunk;
        auto it = std::find_if(g.staging_free.begin(), g.staging_free.end(), [&](const DevBuffer& c) { return c.size >= want; });
        if (it != g.staging_free.end()) {
            chunk = *it;
            g.staging_free_bytes -= it->size;
            g.staging_free.erase(it);
        } else {
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
void scratch_let_go_locked(const DevBuffer& b) {
    if (g.scratch_free_bytes + b.size <= kScratchKeepBytes) {
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
    char buf[160];
    std::snprintf(buf, sizeof(buf), "chunks created %llu, uploads placed %llu; device-local untile scratch: chunks created %llu, spans placed %llu",
                  static_cast<unsigned long long>(g_staging_created.load()), static_cast<unsigned long long>(g_staging_placed.load()),
                  static_cast<unsigned long long>(g_scratch_created.load()), static_cast<unsigned long long>(g_scratch_placed.load()));
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
};
std::vector<HeapBlock> g_heap;  // under g.mu; an image's ImageMemory::block indexes it
std::uint64_t g_heap_allocs = 0, g_heap_own = 0, g_heap_released = 0, g_heap_cp_blocks = 0;

// A new block's slot: a released one if any, so the indices live images
// hold never move.
void heap_add_block_locked(VkDeviceMemory memory, std::uint32_t type) {
    HeapBlock nb;
    nb.memory = memory;
    nb.type = type;
    nb.free[0] = kHeapBlock;
    nb.empty_since = std::chrono::steady_clock::now();
    for (HeapBlock& b : g_heap) {
        if (b.memory == VK_NULL_HANDLE) {
            b = std::move(nb);
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

void heap_release(const ImageMemory& m) {
    HeapBlock& b = g_heap[static_cast<std::size_t>(m.block)];
    b.used -= m.size;
    if (b.used == 0) b.empty_since = std::chrono::steady_clock::now();
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
bool heap_place_locked(const VkMemoryRequirements& req, std::uint32_t type, bool grow, ImageMemory& out) {
    const VkDeviceSize align = std::max<VkDeviceSize>(req.alignment, 256);
    if (req.size <= kHeapLargest) {
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t k = 0; k < g_heap.size(); ++k) {
                HeapBlock& b = g_heap[k];
                if (b.memory == VK_NULL_HANDLE || b.type != type) continue;
                for (auto it = b.free.begin(); it != b.free.end(); ++it) {
                    const VkDeviceSize at = (it->first + align - 1) & ~(align - 1);
                    if (at + req.size > it->first + it->second) continue;
                    const VkDeviceSize lo = it->first, hi = it->first + it->second;
                    b.free.erase(it);
                    if (at > lo) b.free[lo] = at - lo;
                    if (at + req.size < hi) b.free[at + req.size] = hi - (at + req.size);
                    b.used += req.size;
                    out = {b.memory, at, req.size, static_cast<std::int32_t>(k)};
                    ++g_heap_allocs;
                    return true;
                }
            }
            if (pass || !grow) break;
            // No room: another block, made here on the command processor
            // (5-10 ms) because the keeper's spare ran out (memory_keeper).
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = kHeapBlock;
            mai.memoryTypeIndex = type;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            if (vkAllocateMemory(g.device, &mai, nullptr, &mem) != VK_SUCCESS) break;
            heap_add_block_locked(mem, type);
            ++g_heap_cp_blocks;
            host_log("gpu: image heap block made on the command processor (%zu live, %llu MiB each)%s", heap_live_blocks_locked(),
                     static_cast<unsigned long long>(kHeapBlock >> 20),
                     g.mem_props.memoryTypes[type].heapIndex == g.local_heap ? "" : " in host memory");
        }
    }
    if (!grow) return false;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (vkAllocateMemory(g.device, &mai, nullptr, &out.memory) != VK_SUCCESS) return false;
    out.size = req.size;
    ++g_heap_own;
    return true;
}
}  // namespace

bool image_memory_alloc(const VkMemoryRequirements& req, ImageMemory& out) {
    out = ImageMemory{};
    const std::uint32_t local = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const std::int64_t now = steady_ms();
    const bool may_grow = now - g_vram_refused_ms.load(std::memory_order_relaxed) >= 1000;
    if (local != UINT32_MAX && heap_place_locked(req, local, may_grow, out)) return true;
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
    if (!heap_place_locked(req, host, true, out)) return false;
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

std::string image_heap_report() {
    VkDeviceSize used = 0;
    for (const HeapBlock& b : g_heap) used += b.used;
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "image heap: %zu blocks, %llu MiB in use; %llu textures placed in it, %llu with memory of their own, %llu in host memory "
                  "for want of video memory; blocks made on the command processor %llu, released %llu; upload staging free %llu MiB; view epoch %llu",
                  heap_live_blocks_locked(), static_cast<unsigned long long>(used >> 20), static_cast<unsigned long long>(g_heap_allocs),
                  static_cast<unsigned long long>(g_heap_own), static_cast<unsigned long long>(g_heap_host),
                  static_cast<unsigned long long>(g_heap_cp_blocks), static_cast<unsigned long long>(g_heap_released),
                  static_cast<unsigned long long>(g.staging_free_bytes >> 20), static_cast<unsigned long long>(view_epoch()));
    if (const std::string sites = view_epoch_sites_report(); !sites.empty()) host_log("  %s", sites.c_str());
    return buf;
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
//   upload staging    BBHOST_STAGING_FLOOR_MB (256) kept in play, chunks
//                     over it freed after 30 s; BBHOST_STAGING_RESERVE_MB
//                     (768; 256) made ready during a load
// BBHOST_MEMORY_KEEPER=0 restores the old fixed reserve (six blocks and
// 768 MiB at boot, never given back) for comparison.
std::atomic<bool> g_loading_screen{false};
std::mutex g_keeper_mu;
std::condition_variable g_keeper_cv;

void start_memory_reserve() {
    const auto env = [](const char* name, std::uint64_t fallback) {
        const char* e = std::getenv(name);
        return e && *e ? static_cast<std::uint64_t>(std::strtoull(e, nullptr, 10)) : fallback;
    };
    // With tight memory (memory_tight) only what is needed: the spares are
    // what a Steam Deck's world ran out of.
    const std::uint64_t reserve = env("BBHOST_STAGING_RESERVE_MB", g.tight ? 256 : 768) << 20;
    const std::uint64_t floor = std::min(env("BBHOST_STAGING_FLOOR_MB", 256) << 20, reserve);
    const std::uint64_t spare_play = env("BBHOST_IMAGE_HEAP_SPARE", g.tight ? 0 : 1);
    const std::uint64_t spare_load = std::max(env("BBHOST_IMAGE_HEAP_SPARE_LOADING", g.tight ? 1 : 4), spare_play);
    const bool keeper = env("BBHOST_MEMORY_KEEPER", 1) != 0;
    std::thread([=] {
        host_thread_set_name("bb-reserve");
        // The heap's blocks are of the memory type its first image asked for.
        std::uint32_t type = UINT32_MAX;
        const auto make_staging = [&](std::uint64_t upto) {
            std::uint64_t made = 0;
            for (;;) {
                {
                    std::lock_guard<GpuMutex> lk(g.mu);
                    if (g.staging_free_bytes + kStagingChunkBytes > std::min(upto, kStagingKeepBytes)) break;
                }
                DevBuffer chunk;
                if (!create_dev_buffer(chunk, kStagingChunkBytes, true)) break;
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
                    if (type == UINT32_MAX || heap_free_locked(type, nullptr) >= spare_blocks * kHeapBlock) break;
                }
                if (steady_ms() - g_vram_refused_ms.load(std::memory_order_relaxed) < 10000) break;  // video memory is full
                VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                mai.allocationSize = kHeapBlock;
                mai.memoryTypeIndex = type;
                VkDeviceMemory mem = VK_NULL_HANDLE;
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
        host_log("gpu: memory keeper: image heap spare %llu block(s) in play, %llu during loads; upload staging %llu MiB kept, %llu MiB "
                 "during loads",
                 static_cast<unsigned long long>(spare_play), static_cast<unsigned long long>(spare_load),
                 static_cast<unsigned long long>(floor >> 20), static_cast<unsigned long long>(reserve >> 20));
        make_staging(floor);
        auto over_floor_since = std::chrono::steady_clock::time_point{};
        // One piece of memory goes back at a time, two seconds apart: freeing
        // half a gigabyte of upload staging in one go (eight vkFreeMemory of
        // pinned, mapped memory) stalled the frames behind it ~100 ms in every
        // soak, the command processor's next driver calls waiting on it.
        auto last_given = std::chrono::steady_clock::time_point{};
        bool was_loading = false;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(g_keeper_mu);
                g_keeper_cv.wait_for(lk, std::chrono::milliseconds(250));
            }
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
            make_blocks(loading ? spare_load : spare_play);
            if (loading) continue;
            // In play: empty blocks above the spare go back after 10 s; staging
            // over the floor for 30 s goes back to it - one block or chunk a
            // turn, two seconds apart.
            std::vector<VkDeviceMemory> gone;
            std::vector<DevBuffer> chunks;
            const bool may_give = now - last_given >= std::chrono::seconds(2);
            {
                std::lock_guard<GpuMutex> lk(g.mu);
                if (type != UINT32_MAX && may_give) {
                    const VkDeviceSize free = heap_free_locked(type, nullptr);
                    for (HeapBlock& b : g_heap) {
                        if (b.memory == VK_NULL_HANDLE || b.used != 0) continue;
                        if (now - b.empty_since < std::chrono::seconds(10)) continue;
                        // A block of host memory, made while video memory was
                        // full, goes back once empty; one of video memory only
                        // above the spare.
                        if (b.type == type && free < (spare_play + 1) * kHeapBlock) continue;
                        gone.push_back(b.memory);
                        b.memory = VK_NULL_HANDLE;
                        b.free.clear();
                        ++g_heap_released;
                        break;
                    }
                }
                if (g.staging_free_bytes > floor) {
                    if (over_floor_since == std::chrono::steady_clock::time_point{}) over_floor_since = now;
                    if (now - over_floor_since >= std::chrono::seconds(30) && may_give && gone.empty() && !g.staging_free.empty()) {
                        chunks.push_back(g.staging_free.back());
                        g.staging_free_bytes -= g.staging_free.back().size;
                        g.staging_free.pop_back();
                    }
                } else {
                    over_floor_since = {};
                }
            }
            if (!gone.empty() || !chunks.empty()) {
                const auto t0 = std::chrono::steady_clock::now();
                for (VkDeviceMemory m : gone) vkFreeMemory(g.device, m, nullptr);
                for (DevBuffer& c : chunks) {
                    vkDestroyBuffer(g.device, c.buffer, nullptr);
                    vkFreeMemory(g.device, c.memory, nullptr);
                }
                last_given = std::chrono::steady_clock::now();
                std::size_t live;
                {
                    std::lock_guard<GpuMutex> lk(g.mu);
                    live = heap_live_blocks_locked();
                }
                host_log("gpu: memory keeper: gave back %s in %.1f ms (%zu blocks live, %llu MiB of staging kept)",
                         gone.empty() ? "64 MiB of upload staging" : "an image heap block",
                         std::chrono::duration<double, std::milli>(last_given - t0).count(), live,
                         static_cast<unsigned long long>(g.staging_free_bytes >> 20));
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
    flush_deferred_writes_locked();  // (its transfer batch puts deferred copy-backs in place first)
    transfer_flush_locked();
    render_end_pass_locked();
    copy_versions_flush_locked();
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
    textures_submitted(g.slot);
    g.queued = 0;
    ++g.flushes;
    g.slot = (g.slot + 1) % g.slot_count;
    retire_slot_locked(g.slot);
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
    Gpu::Slot& sl = g.slots[serial % static_cast<std::uint64_t>(g.slot_count)];
    if (!sl.in_flight || sl.serial != serial) return;  // retired, and its slot taken again
    stream_wait_submitted(serial);                      // its fence has gone to the queue
    const auto t0 = std::chrono::steady_clock::now();
    vkWaitForFences(g.device, 1, &sl.fence, VK_TRUE, 10000000000ull);
    g.gpu_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
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

bool host_gpu_init(const char* const* instance_exts, std::uint32_t n_exts, bool want_present) {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.tried) {
        for (std::uint32_t i = 0; i < n_exts; ++i) g_instance_exts.push_back(instance_exts[i]);
        g_want_present = want_present;
    }
    return init_locked();
}

bool host_gpu_memory_tight() {
    if (const int t = g_tight.load(); t >= 0) return t == 1;
    bool tight = false;
    if (g_enabled) {
        obs_capture_policy();  // before the first instance, this one
        // Before the device: an instance of its own, without extensions.
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "bbhost";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&ici, nullptr, &instance) == VK_SUCCESS) {
            std::vector<Candidate> usable;
            const int chosen = choose_device(instance, false, usable);
            if (chosen >= 0) tight = memory_tight(usable[static_cast<std::size_t>(chosen)]);
            vkDestroyInstance(instance, nullptr);
        }
    }
    int unknown = -1;
    g_tight.compare_exchange_strong(unknown, tight ? 1 : 0);
    return g_tight.load() == 1;
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
    for (int i = 0; i < 16; ++i) st.draw_fail_why[i] = g.draw_fail_why[i].load(std::memory_order_relaxed);
    return st;
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
enum DispatchPart { kDpLock, kDpProgram, kDpTables, kDpPaths, kDpPipeline, kDpWork, kDpParts };
const char* const kDispatchPartName[kDpParts] = {"lock", "program lookup", "page tables", "paths and dims", "key and pipeline",
                                                 "the work"};
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
}

bool host_gpu_dispatch(const GpuDispatch& d) {
    DispatchStamp stamp;
    GpuPhaseTimer timer(kPhaseLock);
    std::lock_guard<GpuMutex> lock(g.mu);
    timer.next(kPhaseDispatch);
    g_dispatch_n.fetch_add(1, std::memory_order_relaxed);
    stamp.to(kDpLock);
    if (!init_locked()) {
        return false;
    }
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
    stamp.to(kDpProgram);
    {
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
            if (mark_surfaces_dirty_locked(base, bytes) || on_rt) {
                tex_event(base, bytes, "fill dispatch %s 0x%llx 0x%zx bytes with %g %g %g %g%s", pl.name.c_str(),
                          static_cast<unsigned long long>(base), bytes, rgba[0], rgba[1], rgba[2], rgba[3],
                          on_rt ? " (render target)" : "");
            }
            if (render_handle_fill_locked(base, bytes, rgba, d.depth_clear, d.stencil_clear)) {
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
        const int on_surfaces = mark_surfaces_dirty_locked(dst, bytes);
        if (render_copy_target_locked(src, dst, bytes)) {
            tex_event(dst, bytes, "copy dispatch %s 0x%llx -> 0x%llx 0x%zx bytes: render target copied as an image", pl.name.c_str(),
                      static_cast<unsigned long long>(src), static_cast<unsigned long long>(dst), bytes);
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
    // On-demand imports: a dispatch binds no buffers - its shader walks the
    // page table for every one - so what it reads and writes is imported first.
    if (!g_import_whole) {
        const auto import_path = [&](const gcn::ResourcePath& path, bool pointer, std::uint64_t min_bytes) {
            std::uint32_t v[4]{};
            if (!resolve_resource(path, d.user_data, pointer ? 2 : 4, v)) return;
            const std::uint64_t base = v[0] | (static_cast<std::uint64_t>(pointer ? v[1] : (v[1] & 0xff)) << 32);
            const std::uint32_t stride = pointer ? 0 : (v[1] >> 16) & 0x3fff;
            const std::uint64_t bytes = pointer ? 0 : stride ? std::uint64_t{v[2]} * stride : v[2];
            if (base) import_windows(base, std::min<std::uint64_t>(std::max(bytes, min_bytes), 64ull << 20));
        };
        for (const gcn::BufferBinding& b : pl.meta.buffers) import_path(b.path, b.pointer, std::max<std::uint64_t>(b.max_dw * 4ull, 16));
        for (const gcn::ResourcePath& p : pl.meta.store_buffers) import_path(p, false, 16);
    }
    if (g.queued + 1 >= kMaxQueued * kStageSlots) {
        submit_locked();
    }
    begin_recording_locked();
    transfer_flush_locked();
    render_end_pass_locked();
    copy_versions_flush_locked();  // a compute shader reads memory without looking at them
    gcn::StageParams params{};
    params.l1_table = g.l1.address;
    std::memcpy(params.user_sgpr, d.user_data, sizeof(params.user_sgpr));
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
    rec().bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pl.pipeline);
    const std::uint32_t params_offset = static_cast<std::uint32_t>(ubi.offset);
    rec().bind_sets(VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe_layout, 0, 1, &set, set_cache_on() ? 1 : 0, &params_offset);
    profile_begin_locked(&pl.name);
    note_dispatch_name(g.dispatches.load(), pl.name.c_str());
    gpu_checkpoint(2, g.dispatches.load());
    if (d.indirect_va) {
        if (g_import_audit) audit_range(d.indirect_va, 12, kUseIndex);
        const Located loc = locate(d.indirect_va, 12);
        if (!loc.buffer || loc.avail < 12) {
            profile_end_locked();
            g.failures.fetch_add(1);
            return false;
        }
        if (!g_render_min) rec().dispatch_indirect(loc.buffer, loc.offset);
    } else {
        if (!g_render_min) rec().dispatch(d.dim[0], d.dim[1], d.dim[2]);
    }
    profile_end_locked();
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
        shadow_written_locked(base, bytes, true);
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
    }
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDEX_READ_BIT;
    rec().pipeline_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    g.dispatches.fetch_add(1);
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
// device before the next is written. Written at once, the cache (313 MB) was
// seconds of write-back that other threads' file reads and log writes queued
// behind: a 2.9 s main-loop frame at boot and a 1 s stall in play, over NFS.
bool write_file_paced(const std::string& path, const char* data, std::size_t size) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    constexpr std::size_t kChunk = 8u << 20;
    bool ok = true;
    for (std::size_t at = 0; ok && at < size; at += kChunk) {
        const std::size_t n = std::min(kChunk, size - at);
        ok = std::fwrite(data + at, 1, n, f) == n && std::fflush(f) == 0;
#if defined(_WIN32)
        ok = ok && _commit(_fileno(f)) == 0;
#else
        ok = ok && ::fdatasync(fileno(f)) == 0;
#endif
    }
    ok = std::fclose(f) == 0 && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);
    return ok && !ec;
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
    // Each ordinary run of transfers shares one barrier pair; this one starts
    // its own, so nothing recorded before it can land after it.
    transfer_flush_locked();
    static std::unordered_map<std::uint64_t, std::size_t> last;  // dword -> index of the last write to it
    static std::unordered_set<std::uint64_t> filled;             // dwords filled since the last barrier
    last.clear();
    filled.clear();
    const auto& ws = g.deferred_writes;
    for (std::size_t i = 0; i < ws.size(); ++i) {
        for (std::uint32_t k = 0; k < ws[i].bytes; k += 4) last[ws[i].va + k] = i;
    }
    for (std::size_t i = 0; i < ws.size(); ++i) {
        const Gpu::SmallWrite& w = ws[i];
        bool live = false, clash = false;
        for (std::uint32_t k = 0; k < w.bytes; k += 4) {
            live |= last[w.va + k] == i;
            clash |= filled.count(w.va + k) != 0;
        }
        if (!live) continue;
        Located loc;
        if (!locate_range(w.va, w.bytes, loc)) continue;
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
        }
        std::uint32_t d[2] = {0, 0};
        std::memcpy(d, w.data, w.bytes);
        if (w.bytes == 8 && d[0] == d[1]) {
            rec().fill_buffer(loc.buffer, loc.offset, 8, d[0]);
        } else {
            for (std::uint32_t k = 0; k < w.bytes; k += 4) rec().fill_buffer(loc.buffer, loc.offset + k, 4, d[k / 4]);
        }
        for (std::uint32_t k = 0; k < w.bytes; k += 4) filled.insert(w.va + k);
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
            const Chunk* c = chunk < gcn::kDmemChunks ? dmem_import(static_cast<std::uint32_t>(chunk), in, bytes) : nullptr;
            return c ? c->address + (in - c->lo) : 0;
        }
        auto it = g.anon_imports.find(mi.va);
        return it == g.anon_imports.end() || !it->second.address ? 0 : it->second.address + off;
    }
    return 0;
}

}  // namespace gpu

void host_gpu_set_loading(bool loading) {
    if (gpu::g_loading_screen.exchange(loading) != loading) gpu::g_keeper_cv.notify_one();
    gpu::precompile_set_loading(loading);
}

void host_gpu_world_reached() { gpu::precompile_set_world_reached(); }

std::string host_gpu_image_heap_report() {
    std::lock_guard<GpuMutex> lock(gpu::g.mu);
    return gpu::image_heap_report();
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

bool host_gpu_mem_write(std::uint64_t va, const void* data, std::size_t bytes) { return host_gpu_mem_write_serial(va, data, bytes, nullptr); }

bool host_gpu_mem_write_serial(std::uint64_t va, const void* data, std::size_t bytes, std::uint64_t* serial) {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok || bytes > 65536) return false;
    if (serial) *serial = g.flushes;  // the submission this write is recorded into
    if (hle_gnm_arena_monitor() && hle_gx_arena_covers_marker(va, bytes)) {
        std::uint64_t v = 0;
        std::memcpy(&v, data, bytes < 8 ? bytes : 8);
        if (v > 2) note_gpu_write_over_marker("write", va, v, bytes);
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
        Gpu::SmallWrite w{va, static_cast<std::uint32_t>(bytes), {}};
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
    if (mark_surfaces_dirty_locked(va, bytes) || on_rt) {
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

bool host_gpu_texture_create_ahead(const std::uint32_t tsharp[8]) {
    std::lock_guard<GpuMutex> lock(g.mu);
    return g.ok && textures_create_ahead_locked(tsharp);
}

void host_gpu_texture_retire(std::uint64_t base, std::size_t bytes, std::uint64_t after_submitted) {
    textures_retire(base, bytes, after_submitted);  // its own lock: callable from the kernel's unmap
}

bool host_gpu_mem_fill(std::uint64_t va, std::uint32_t value, std::size_t bytes) {
    std::lock_guard<GpuMutex> lock(g.mu);
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
    if (!locate_range(dst, bytes, d) || !locate_range(src, bytes, sr)) return false;
    note_gpu_write_over_marker("copy", dst, src, bytes);
    transfer_begin_locked();
    transfer_order_locked(sr.buffer, sr.offset, d.buffer, d.offset, bytes, true);
    VkBufferCopy region{sr.offset, d.offset, bytes};
    rec().copy_buffer(sr.buffer, d.buffer, 1, &region);
    transfer_end_locked();
    return true;
}

// Copy tokens whose source was a render target, copied as images.
std::atomic<std::uint64_t> g_guest_copy_targets{0};
std::uint64_t host_gpu_guest_copy_targets() { return g_guest_copy_targets.load(std::memory_order_relaxed); }

bool host_gpu_copy_guest(std::uint64_t dst, std::uint64_t src, std::size_t bytes) {
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok) return false;
    if (!bytes) return true;
    if (dst < src + bytes && src < dst + bytes) return false;
    note_gpu_write_over_marker("guest copy", dst, src, bytes);
    const int on_surfaces = mark_surfaces_dirty_locked(dst, bytes);
    if (render_copy_target_locked(src, dst, bytes)) {
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
    for (std::size_t done = 0; done < bytes;) {
        const Located d = locate(dst + done, bytes - done), s = locate(src + done, bytes - done);
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
        g_cv_refused.fetch_add(1, std::memory_order_relaxed);
        g_cv_refused_why[why].fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    if (!bytes || bytes > kCopyVersionMax || ((dst | src | bytes) & 3)) return refuse(kCvSize);
    if ((dst < src + bytes && src < dst + bytes) || g_copy_versions.size() >= kCopyVersionsPending) return refuse(kCvOther);
    if (!hle_kernel_va_mapped(dst, bytes) || !hle_kernel_va_mapped(src, bytes) || !rebuild_page_tables()) return refuse(kCvOther);
    if (surfaces_may_cover_locked(dst, bytes)) return refuse(kCvSurfaces);
    if (find_render_target(src) || find_render_target(dst)) return refuse(kCvTarget);
    const Located canon = locate(dst, bytes);
    if (!canon.buffer || canon.avail < bytes) return refuse(kCvImport);
    // The source as this point in the stream has it: an earlier version
    // holding all of it, else guest memory. Part of one: the GPU copy, after
    // the pass's own copies.
    const auto* from = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(src));
    if (const CopyVersion* v = copy_version_over(src, bytes)) {
        if (src < v->dst || src + bytes > v->dst + v->bytes) return refuse(kCvSource);
        from = static_cast<const std::uint8_t*>(v->copy.map) + (src - v->dst);
        g_cv_from_version.fetch_add(1, std::memory_order_relaxed);
    }
    CopyVersion v;
    v.dst = dst;
    v.bytes = bytes;
    v.canon = canon;
    if (!acquire_staging_locked(v.copy, bytes, v.offset)) return refuse(kCvOther);
    std::memcpy(v.copy.map, from, bytes);
    note_gpu_write_over_marker("guest copy", dst, src, bytes);
    copy_version_add(v);
    g_cv_made.fetch_add(1, std::memory_order_relaxed);
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
        mark_surfaces_dirty_locked(v.dst, v.bytes);  // the buffer shadow sees the GPU write
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
                  "across an edge %llu, a page-table read %llu, a page-table pipeline %llu; %llu transfers waited for an earlier one in "
                  "their batch",
                  static_cast<ull>(g_cv_made.load()), static_cast<ull>(g_cv_from_version.load()), static_cast<ull>(g_cv_bound.load()),
                  static_cast<ull>(g_cv_straddled.load()), static_cast<ull>(g_cv_refused.load()), static_cast<ull>(g_cv_refused_why[kCvSize].load()),
                  static_cast<ull>(g_cv_refused_why[kCvSurfaces].load()), static_cast<ull>(g_cv_refused_why[kCvTarget].load()),
                  static_cast<ull>(g_cv_refused_why[kCvImport].load()), static_cast<ull>(g_cv_refused_why[kCvSource].load()),
                  static_cast<ull>(g_cv_refused_why[kCvOther].load()), static_cast<ull>(g_cv_copies.load()),
                  static_cast<ull>(g_cv_batches.load()), static_cast<ull>(g_cv_overwritten.load()), static_cast<ull>(g_cv_put[kCvStraddle].load()),
                  static_cast<ull>(g_cv_put[kCvPageTable].load()), static_cast<ull>(g_cv_put[kCvWalks].load()),
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

std::uint64_t host_gpu_work_needs() {
    std::lock_guard<GpuMutex> lock(g.mu);
    // Submissions are numbered from 0 in `flushes`; the recording under way
    // becomes the next one.
    return g.recording ? g.flushes + 1 : g.flushes;
}

std::uint64_t host_gpu_submissions_completed() {
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

void host_gpu_report() {
    dispatch_cost_report();

    if (!g.tried) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_dispatch_census_mu);
        std::vector<std::pair<std::uint64_t, std::string>> by_count;
        for (const auto& [n, c] : g_dispatch_census) by_count.emplace_back(c, n);
        std::sort(by_count.rbegin(), by_count.rend());
        host_log("gpu:   recognised as an image copy: %llu image-to-image, %llu buffer-to-image (they still run their shader)",
                 static_cast<unsigned long long>(g_dispatch_image_blits.load()),
                 static_cast<unsigned long long>(g_dispatch_buf_to_image.load()));
        host_log("gpu: dispatch census: %llu reached a shader over %zu programs, %llu were a fill and %llu a copy (no shader runs "
                 "for those)",
                 static_cast<unsigned long long>(g_dispatch_shaders.load()), g_dispatch_census.size(),
                 static_cast<unsigned long long>(g_dispatch_fills.load()), static_cast<unsigned long long>(g_dispatch_copies.load()));
        for (std::size_t k = 0; k < by_count.size() && k < 16; ++k) {
            host_log("gpu:   %8llu %s", static_cast<unsigned long long>(by_count[k].first), by_count[k].second.c_str());
        }
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
    render_report();
}
