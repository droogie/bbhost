// The GX resource registry; see the header.
//
// Hooks (hash-gated through the eboot check the caller makes, prologue hooks
// of engine/graphics_patch.h):
//   creators  0x2565c10 CreateBuffer, 0x2565f90 CreateTexture (1D/2D),
//             0x2566300 CreateTexture3D - (ctx, holder** out, desc*,
//             initial data*, ...). The holder exists only when the method
//             returns, so the hook swaps the return address for gx_res_ret,
//             which reads *out and registers it.
//   Map       0x2569c30 (ctx, ?, holder, subresource, type, mapped*): a CPU
//             write is coming. Type 1/2 write the memory in place after a
//             GPU wait, 3 appends into a renamed copy, 4 discards - the
//             holder's memory may move, so the return hook re-reads +0x20.
//   shaders   the four shader creators (hooked by gx_trace.cpp) and the
//             input-layout creator 0x25657a0 - (ctx, object** out, ...):
//             the object gets an id at the return.
//   views     0x25667a0 render target, 0x2566610 depth-stencil (ctx, view**
//             out, texture holder, description, ...): an id, and the
//             texture's extent at the view's mip. The view keeps the holder
//             at +0x08 (a reference); the description is D3D11's - a render
//             target's format / dimension / mip / first slice / slices at
//             +0x3c..+0x4c, a depth-stencil view's format / dimension / flags
//             / mip at +0x44..+0x50.
//   destruct  the deleting and the in-place destructors of a texture holder
//             (0x256d1a0 / 0x256d1e0, 3D 0x256d310 / 0x256d350; their
//             prologue starts with a rip-relative lea of the vtable, run in
//             the stub as a movabs), and a buffer's resource release
//             0x256d690 (rdi = holder + 0x18).
#include "engine/gx_resources.h"

#include "core/elf.h"
#include "core/portable.h"
#include "core/write_watch.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "guest_abi.h"
#include "hle/hle.h"
#include "hle/modules.h"
#include "host/gpu.h"
#include "log.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#include <link.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

extern "C" void gx_res_ret();
extern "C" GUEST_ABI std::uint64_t gx_res_after(std::uint64_t result);

namespace {

constexpr std::uint64_t kCreateBuffer = 0x2565c10, kCreateTexture = 0x2565f90, kCreateTexture3D = 0x2566300;
constexpr std::uint64_t kMap = 0x2569c30;
constexpr std::uint64_t kCreateInputLayout = 0x25657a0;
constexpr std::uint64_t kCreateRtv = 0x25667a0, kCreateDsv = 0x2566610;
// Shader-resource views (the T# GX built at +0x10) and samplers
// (the S# at +0x0), (ctx, object** out, ...).
constexpr std::uint64_t kCreateSrv = 0x25669a0, kCreateSampler = 0x2565b40;
// push rbp; mov rbp, rsp; push r15; push r14; push rbx; sub rsp, 0x48; mov rbx, rcx
constexpr std::uint8_t kSamplerPrologue[16] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                               0x53, 0x48, 0x83, 0xec, 0x48, 0x48, 0x89, 0xcb};
// push rbp; mov rbp, rsp; push r15; push r14; push r12; push rbx; sub rsp, 0x70
constexpr std::uint8_t kInputLayoutPrologue[15] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                   0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x70};
constexpr std::uint64_t kRing = 0x2adb1b0;  // the dynamic ring allocator: (ring, out*, ctx, bytes) -> *out
constexpr std::uint64_t kBufferRelease = 0x256d690;
constexpr std::uint64_t kTexDelete = 0x256d1a0, kTexDestroy = 0x256d1e0, kTexVtable = 0x575f3e0;
constexpr std::uint64_t kTex3dDelete = 0x256d310, kTex3dDestroy = 0x256d350, kTex3dVtable = 0x575f400;
// The T# builder the 1D/2D creator runs (0x2571d60 -> 0x2fa6af0(size*, tsharp*,
// desc, subresource info, scratch)): its rsi is the T# GX gives the texture,
// before the memory is allocated, so the base is 0 until we fill it in.
constexpr std::uint64_t kTsharpBuild = 0x2fa6af0;
// 1D textures have no holder creator of their own: 0x2ad5e90 builds the
// holder inline around the 1D resource creator 0x2571790(resource* out, ctx,
// desc, initial data, ...), whose T# builder is 0x2fa64a0 (rsi = T# out).
// The resource struct is registered from the creator's return under a key
// made from its memory; the holder's destructors (0x256d030 / 0x256d070,
// vtable 0x575f3c0) find it by the memory at holder+0x20.
constexpr std::uint64_t kCreate1D = 0x2571790, kTsharpBuild1D = 0x2fa64a0;
constexpr std::uint64_t kTex1dDelete = 0x256d030, kTex1dDestroy = 0x256d070, kTex1dVtable = 0x575f3c0;
constexpr std::uint64_t kMemoryKey = 0x8000000000000000ull;
constexpr std::uint8_t kTsharpBuildPrologue[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                                   0x41, 0x54, 0x53, 0x48, 0x81, 0xe4, 0xe0, 0xff, 0xff, 0xff};
constexpr std::uint8_t kCreate1DPrologue[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                                0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xc8, 0x00, 0x00, 0x00};

#define BB_PUSH6 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53
constexpr std::uint8_t kPrologue58[17] = {BB_PUSH6, 0x48, 0x83, 0xec, 0x58};  // the three creators and Map
constexpr std::uint8_t kRingPrologue[17] = {BB_PUSH6, 0x48, 0x83, 0xec, 0x28};
constexpr std::uint8_t kDsvPrologue[17] = {BB_PUSH6, 0x48, 0x83, 0xec, 0x68};  // the render-target view creator's is kPrologue58
#undef BB_PUSH6
// push rbp; mov rbp, rsp; push r15; push r14; push rbx; push rax; mov r14, rdi; mov rdi, [r14]
constexpr std::uint8_t kBufferReleasePrologue[16] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                     0x53, 0x50, 0x49, 0x89, 0xfe, 0x49, 0x8b, 0x3e};
// push rbp; mov rbp, rsp; push rbx; push rax; mov rbx, rdi; lea rax, [rip + vtable]
constexpr std::uint8_t kDtorHead[9] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb};

enum Kind : std::uint8_t { kBuffer = 1, kTexture = 2, kTexture3D = 3, kMapCall = 4, kTsharp = 5, kResource1D = 6, kObject = 7 };

struct Pending {
    std::uint64_t ret;
    Kind kind;
    std::uint64_t out;      // creators: holder**; Map: mapped* (the address the caller gets)
    std::uint64_t holder;   // Map: the holder; creators: the description
    std::uint64_t type;     // Map: the map type; creators: initial data pointer
    bool ring = false;
    std::uint64_t ring_bytes = 0;
};
// Creators that returned an error (0 is success; 9 an allocation that failed):
// a render target that did not fit is the one live_resolution.cpp watches for.
std::atomic<std::uint64_t> g_create_failures{0};
thread_local std::vector<Pending> t_pending;
// The T# the builder produced on this thread, for the creator's return.
thread_local std::uint32_t t_tsharp[8];
thread_local bool t_tsharp_valid = false;
// BBHOST_GX_TEX_AHEAD=0: do not create textures ahead of their first bind.
const bool g_tex_ahead = [] {
    const char* e = std::getenv("BBHOST_GX_TEX_AHEAD");
    return !e || e[0] != '0';
}();
std::atomic<std::uint64_t> g_ahead_created{0}, g_ahead_bytes{0}, g_ahead_refused{0}, g_ahead_no_tsharp{0}, g_ahead_dropped{0},
    g_ahead_gone{0};
std::atomic<std::uint64_t> g_retire_queued{0};

struct Entry {
    GxResourceInfo info;
    std::uint64_t holder = 0;
    std::uint64_t created_flip = 0;
    std::uint32_t tsharp[8] = {};
    bool has_tsharp = false;
};

std::uint64_t g_slide = 0;
std::mutex g_mu;
std::unordered_map<std::uint64_t, Entry> g_by_holder;      // under g_mu
std::map<std::uint64_t, std::uint64_t> g_by_memory;        // memory base -> holder, live entries only
std::atomic<std::uint64_t> g_write_seq{1};

// Counters.
std::atomic<std::uint64_t> g_created[5] = {}, g_created_with_data[5] = {}, g_created_bytes[5] = {};
std::atomic<std::uint64_t> g_destroyed{0}, g_destroyed_unknown{0}, g_replaced_holder{0}, g_memory_reused{0};
std::atomic<std::uint64_t> g_released_under_holder{0}, g_release_events{0}, g_protected{0}, g_protect_events{0};
std::atomic<std::uint64_t> g_maps[5][5] = {};  // [resource type][map type]
std::atomic<std::uint64_t> g_map_unknown{0}, g_map_moved{0};
std::atomic<std::uint64_t> g_lookups{0}, g_lookup_hits{0};
std::atomic<std::uint64_t> g_peak_live{0};

// Object -> id for shader and input-layout objects. Written at
// creation (rare: area loads) under g_obj_mu, read at every draw on the GX
// threads without a lock: a slot's id is stored before its object, and a
// re-created object's new id before the creator returns - before the game
// can hand the object to a draw.
constexpr std::uint32_t kObjSlots = 1u << 17, kObjProbes = 64;
struct ObjSlot {
    std::atomic<std::uint64_t> object{0};
    std::atomic<std::uint32_t> id{0};
    std::atomic<std::uint32_t> extent{0};  // views: gx_view_extent
    std::atomic<std::uint64_t> words{0};   // shader-resource views and samplers: their T#/S# hashed at creation
};
ObjSlot g_obj[kObjSlots];
std::mutex g_obj_mu;
std::atomic<std::uint32_t> g_obj_next{1};
// No counter on the lookup: six GX threads look up three objects a draw, and
// a shared one would be a contended line; the renderer counts draws without ids.
std::atomic<std::uint64_t> g_obj_created{0}, g_obj_recreated{0}, g_obj_full{0};
std::uint32_t obj_slot(std::uint64_t object) {
    return static_cast<std::uint32_t>(((object >> 3) * 0x9e3779b97f4a7c15ull) >> 47) & (kObjSlots - 1);
}
void register_object(std::uint64_t object, std::uint32_t extent = 0, std::uint64_t words = 0) {
    if (!object) return;
    const std::uint32_t id = g_obj_next.fetch_add(1, std::memory_order_relaxed);
    g_obj_created.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_obj_mu);
    const std::uint32_t h = obj_slot(object);
    for (std::uint32_t probe = 0; probe < kObjProbes; ++probe) {
        ObjSlot& s = g_obj[(h + probe) & (kObjSlots - 1)];
        const std::uint64_t o = s.object.load(std::memory_order_relaxed);
        if (o == object) {
            g_obj_recreated.fetch_add(1, std::memory_order_relaxed);
            s.extent.store(extent, std::memory_order_relaxed);
            s.words.store(words, std::memory_order_relaxed);
            s.id.store(id, std::memory_order_release);
            return;
        }
        if (!o) {
            s.id.store(id, std::memory_order_relaxed);
            s.extent.store(extent, std::memory_order_relaxed);
            s.words.store(words, std::memory_order_relaxed);
            s.object.store(object, std::memory_order_release);
            return;
        }
    }
    g_obj_full.fetch_add(1, std::memory_order_relaxed);  // its draws find no id and take the address path
}

bool g_trace = false;  // BBHOST_GX_MAP_TRACE
std::map<std::uint64_t, std::uint64_t> g_trace_callers;  // return address (Binary Ninja) -> calls, under g_mu
std::atomic<int> g_trace_logged{0};

template <typename T>
bool rd(std::uint64_t at, T* out) {
    return host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), out, sizeof(T));
}

std::uint64_t bn(std::uint64_t host_va) { return host_va - g_slide + kPreferredGuestSlide; }

bool old_holder_alive(std::uint64_t holder) { return g_by_holder.count(holder) != 0; }

// `key` names the entry (the holder, or kMemoryKey | memory for a 1D
// resource registered from its struct); `holder` is where the fields are
// read from as a holder lays them out (the struct at +0x18).
// BBHOST_GX_RATE=1: a line for each wall-clock second resources were made
// in, with how many and how many bytes - when a warp's or a walk's
// streaming settles. The second is the epoch's (the map probe's clock).
const bool g_rate_log = std::getenv("BBHOST_GX_RATE") != nullptr;
std::mutex g_rate_mu;
std::int64_t g_rate_sec = 0;
std::uint64_t g_rate_n = 0, g_rate_bytes = 0;
void note_rate(std::uint32_t bytes) {
    const std::int64_t sec =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lk(g_rate_mu);
    if (sec != g_rate_sec) {
        if (g_rate_n) {
            host_log("gx-resources: rate: %llu made (%llu KiB) in second %lld, flip %llu", static_cast<unsigned long long>(g_rate_n),
                     static_cast<unsigned long long>(g_rate_bytes >> 10), static_cast<long long>(g_rate_sec),
                     static_cast<unsigned long long>(hle_video_flip_count()));
        }
        g_rate_sec = sec;
        g_rate_n = 0;
        g_rate_bytes = 0;
    }
    ++g_rate_n;
    g_rate_bytes += bytes;
}

void register_holder(std::uint64_t key, std::uint64_t holder, Kind kind, bool initial_data) {
    if (!holder) return;
    Entry e;
    e.holder = key;
    e.created_flip = hle_video_flip_count();
    e.info.created_flip = e.created_flip;
    std::uint32_t id = 0;
    if (!rd(holder + 0x20, &e.info.memory) || !rd(holder + 0x28, &id) || !rd(holder + 0x2c, &e.info.bytes) ||
        !rd(holder + 0x38, &e.info.type) || !rd(holder + 0x39, &e.info.usage)) {
        return;
    }
    e.info.id = id & 0xffffff;
    e.info.initial_data = initial_data;
    e.info.write_seq = initial_data ? g_write_seq.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
    const unsigned t = e.info.type < 5 ? e.info.type : 0;
    g_created[t].fetch_add(1, std::memory_order_relaxed);
    if (initial_data) g_created_with_data[t].fetch_add(1, std::memory_order_relaxed);
    g_created_bytes[t].fetch_add(e.info.bytes, std::memory_order_relaxed);
    if (g_rate_log) note_rate(e.info.bytes);
    (void)kind;
    if (key & kMemoryKey) key = kMemoryKey | e.info.memory;
    std::lock_guard<std::mutex> lk(g_mu);
    if (auto old = g_by_holder.find(key); old != g_by_holder.end()) {
        // The holder's address reused before its destructor was seen.
        g_replaced_holder.fetch_add(1, std::memory_order_relaxed);
        if (auto m = g_by_memory.find(old->second.info.memory); m != g_by_memory.end() && m->second == key) g_by_memory.erase(m);
        g_by_holder.erase(old);
    }
    if (e.info.memory) {
        if (auto m = g_by_memory.find(e.info.memory); m != g_by_memory.end() && m->second != key) {
            // The memory of a resource whose destructor was not seen, now a
            // new resource's: the old one is dead as far as the memory goes.
            if (g_memory_reused.fetch_add(1, std::memory_order_relaxed) < 8) {
                const auto old = g_by_holder.find(m->second);
                host_log("gx-resources: 0x%llx reused by resource %u (type %u) at flip %llu while resource %u (type %u, created at flip %llu, "
                         "holder 0x%llx) still held it",
                         static_cast<unsigned long long>(e.info.memory), e.info.id, e.info.type, static_cast<unsigned long long>(e.created_flip),
                         old != g_by_holder.end() ? old->second.info.id : 0u, old != g_by_holder.end() ? old->second.info.type : 0u,
                         static_cast<unsigned long long>(old != g_by_holder.end() ? old->second.created_flip : 0), static_cast<unsigned long long>(m->second));
            }
            if (old_holder_alive(m->second)) g_by_holder.erase(m->second);
        }
        g_by_memory[e.info.memory] = key;
    }
    g_by_holder.emplace(key, e);
    const std::uint64_t live = g_by_memory.size();
    if (live > g_peak_live.load(std::memory_order_relaxed)) g_peak_live.store(live, std::memory_order_relaxed);
}

// Textures made ahead of their first bind are made on a thread of their own,
// not the one that created them: that is the game's main loop while it
// streams, and making the image takes the render lock, which the command
// processor holds through every draw. A lock profile over a soak had the main
// loop wait 449 ms for it (1,952 times) and hold it 127 ms more, in the
// frames that stream. A texture a draw binds before its turn here is made by
// that draw, as it was before textures were made ahead; this one then finds it.
struct AheadJob {
    std::array<std::uint32_t, 8> tsharp;
    std::uint64_t memory = 0, bytes = 0;
};
constexpr std::size_t kAheadQueue = 4096;
std::mutex g_ahead_mu;
std::condition_variable g_ahead_cv;
std::deque<AheadJob>* const g_ahead_jobs = new std::deque<AheadJob>;  // under g_ahead_mu; never destroyed

void ahead_thread() {
    host_thread_set_name("bb-tex-ahead");
    for (;;) {
        AheadJob job;
        {
            std::unique_lock<std::mutex> lk(g_ahead_mu);
            g_ahead_cv.wait(lk, [] { return !g_ahead_jobs->empty(); });
            job = g_ahead_jobs->front();
            g_ahead_jobs->pop_front();
        }
        // Destroyed while it waited: its memory may be another resource's by now.
        if (!gx_resource_at(job.memory, std::max<std::uint64_t>(job.bytes, 1), nullptr)) {
            g_ahead_gone.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (host_gpu_texture_create_ahead(job.tsharp.data())) {
            g_ahead_created.fetch_add(1, std::memory_order_relaxed);
            g_ahead_bytes.fetch_add(job.bytes, std::memory_order_relaxed);
        } else {
            g_ahead_refused.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// A 2D texture created with initial data: its image is made off this thread,
// from the T# GX built for it (the command processor asks the registry while
// it holds the render lock, so the job takes neither lock here).
void create_ahead(const Entry& e) {
    if (!g_tex_ahead || (e.info.type != 3 && e.info.type != 2) || !e.info.initial_data || !e.info.memory) return;
    if (!t_tsharp_valid) {
        g_ahead_no_tsharp.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    AheadJob job;
    std::memcpy(job.tsharp.data(), t_tsharp, sizeof(t_tsharp));
    job.tsharp[0] = static_cast<std::uint32_t>(e.info.memory >> 8);
    job.tsharp[1] = (job.tsharp[1] & ~0xffu) | static_cast<std::uint32_t>((e.info.memory >> 40) & 0xff);
    job.memory = e.info.memory;
    job.bytes = e.info.bytes;
    static std::once_flag started;
    std::call_once(started, [] { std::thread(ahead_thread).detach(); });
    {
        std::lock_guard<std::mutex> lk(g_ahead_mu);
        if (g_ahead_jobs->size() >= kAheadQueue) {
            g_ahead_dropped.fetch_add(1, std::memory_order_relaxed);  // made at its first bind instead
            return;
        }
        g_ahead_jobs->push_back(job);
    }
    g_ahead_cv.notify_one();
}

// A resource's memory is going: its surfaces retire once the command
// processor has retired everything submitted so far.
// The memory goes back to GX's heap and the next resource made there writes
// it at once (the internal creators 0x256d420 and 0x2571d60 copy a buffer's
// or a texture's initial data in): with the dead resource's pages still
// write-protected - by the texture cache until its surface retires behind the
// GPU, by the buffer shadow until its mirror pages are next checked - each of
// them faulted on the game's thread, the main loop's when it streams. The
// resource's whole pages are released now (writable, and counted as written,
// so any watch on them sees the change); the pages it shares with its
// neighbours keep their protection.
void release_pages(std::uint64_t memory, std::uint32_t bytes) {
    const std::uint64_t lo = (memory + 0xfff) & ~0xfffull, hi = (memory + bytes) & ~0xfffull;
    if (hi > lo) write_watch_release(reinterpret_cast<void*>(static_cast<std::uintptr_t>(lo)), hi - lo);
}

void retire_memory(std::uint64_t memory, std::uint32_t bytes) {
    if (!memory || !bytes) return;
    g_retire_queued.fetch_add(1, std::memory_order_relaxed);
    release_pages(memory, bytes);
    host_gpu_texture_retire(memory, bytes, hle_gnm_submitted_total());
}

void unregister_holder(std::uint64_t holder) {
    std::uint64_t memory = 0, buffer_memory = 0;
    std::uint32_t bytes = 0, buffer_bytes = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_by_holder.find(holder);
        if (it == g_by_holder.end()) {
            g_destroyed_unknown.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        g_destroyed.fetch_add(1, std::memory_order_relaxed);
        if (it->second.info.type != 1 && it->second.info.alive) {
            memory = it->second.info.memory;
            bytes = it->second.info.bytes;
        } else if (it->second.info.type == 1 && it->second.info.alive) {
            buffer_memory = it->second.info.memory;
            buffer_bytes = it->second.info.bytes;
        }
        if (auto m = g_by_memory.find(it->second.info.memory); m != g_by_memory.end() && m->second == holder) g_by_memory.erase(m);
        g_by_holder.erase(it);
    }
    retire_memory(memory, bytes);
    if (buffer_memory && buffer_bytes) release_pages(buffer_memory, buffer_bytes);
}

// A Map on the holder: the resource is about to be written by the CPU. Its
// memory may have been renamed (type 3/4): re-read +0x20 and re-index.
void note_map(std::uint64_t holder, std::uint64_t type) {
    std::uint64_t memory = 0;
    rd(holder + 0x20, &memory);
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_by_holder.find(holder);
    if (it == g_by_holder.end()) {
        g_map_unknown.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Entry& e = it->second;
    const unsigned rt = e.info.type < 5 ? e.info.type : 0, mt = type < 5 ? static_cast<unsigned>(type) : 0;
    g_maps[rt][mt].fetch_add(1, std::memory_order_relaxed);
    e.info.write_seq = g_write_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    if (memory && memory != e.info.memory) {
        g_map_moved.fetch_add(1, std::memory_order_relaxed);
        if (auto m = g_by_memory.find(e.info.memory); m != g_by_memory.end() && m->second == holder) g_by_memory.erase(m);
        e.info.memory = memory;
        g_by_memory[memory] = holder;
    }
}

// The creators: rsi = holder**, rcx = initial data. The holder exists at the return.
GUEST_ABI std::int64_t create_hook(std::uint64_t id, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], static_cast<Kind>(id), saved[4], saved[3], saved[2]});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// A render-target or depth-stencil view just made: its 2D texture's extent at
// the view's mip (0 for anything else).
std::atomic<std::uint64_t> g_views[4] = {}, g_views_extent[2] = {};  // render target, depth-stencil, shader resource, sampler
std::atomic<int> g_views_logged{0};
std::uint32_t view_extent(std::uint64_t view, bool depth) {
    std::uint64_t holder = 0;
    std::uint8_t type = 0;
    std::uint16_t w = 0, h = 0;
    std::uint32_t format = 0, dim = 0, mip = 0;
    if (!rd(view + 0x08, &holder) || !holder || !rd(holder + 0x38, &type) || !rd(holder + 0x44, &w) || !rd(holder + 0x46, &h) ||
        !rd(view + (depth ? 0x44 : 0x3c), &format) || !rd(view + (depth ? 0x48 : 0x40), &dim) || !rd(view + (depth ? 0x50 : 0x44), &mip)) {
        return 0;
    }
    // D3D11's dimensions: render target 4 TEXTURE2D, 5 2D array; depth-stencil
    // 3 TEXTURE2D, 4 2D array.
    const bool two_d = type == 3 && (depth ? dim == 3 || dim == 4 : dim == 4 || dim == 5);
    const std::uint32_t mw = mip < 16 ? std::max<std::uint32_t>(1, w >> mip) : 0, mh = mip < 16 ? std::max<std::uint32_t>(1, h >> mip) : 0;
    if (g_views_logged.fetch_add(1, std::memory_order_relaxed) < 24) {
        std::uint32_t regs[6] = {};
        for (int k = 0; k < 6; ++k) rd(view + 0x10 + 4 * static_cast<std::uint64_t>(k), &regs[k]);
        host_log("gx-resources: %s view 0x%llx of texture 0x%llx (type %u, %ux%u): format %u, dimension %u, mip %u -> %ux%u; "
                 "registers %08x %08x %08x %08x %08x %08x",
                 depth ? "depth-stencil" : "render-target", static_cast<unsigned long long>(view), static_cast<unsigned long long>(holder), type,
                 w, h, format, dim, mip, mw, mh, regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
    }
    return two_d && mw && mh ? (mw << 16) | mh : 0;
}

// The input-layout creator: rsi = object**. (The shader creators come through
// gx_objects_watch_creator.)
GUEST_ABI std::int64_t object_create_hook(std::uint64_t, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], kObject, saved[4], 0, 0});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// The view creators: rsi = view**; `id` 1 render target, 2 depth-stencil.
GUEST_ABI std::int64_t view_create_hook(std::uint64_t id, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], kObject, saved[4], 0, id});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// Map: rdx = holder, r8 = type, r9 = mapped*.
GUEST_ABI std::int64_t map_hook(std::uint64_t, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], kMapCall, saved[0], saved[3], saved[1]});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// The T# builder: rsi = the T# out.
GUEST_ABI std::int64_t tsharp_hook(std::uint64_t, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], kTsharp, saved[4], 0, 0});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// The ring allocator (trace only): rsi = out*, rcx = bytes.
GUEST_ABI std::int64_t ring_hook(std::uint64_t, const std::uint64_t* saved) {
    Pending p{saved[6], kMapCall, saved[4], 0, 0};
    p.ring = true;
    p.ring_bytes = saved[2];
    t_pending.push_back(p);
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// A texture holder's destructor: rdi = holder.
GUEST_ABI std::int64_t texture_dtor_hook(std::uint64_t, const std::uint64_t* saved) {
    unregister_holder(saved[5]);
    return 0;
}

// A 1D texture holder's destructor: its entry is keyed by the memory.
GUEST_ABI std::int64_t texture1d_dtor_hook(std::uint64_t, const std::uint64_t* saved) {
    std::uint64_t memory = 0;
    if (rd(saved[5] + 0x20, &memory) && memory) unregister_holder(kMemoryKey | memory);
    return 0;
}

// The 1D resource creator: rdi = the resource struct out, rcx = initial data.
GUEST_ABI std::int64_t create1d_hook(std::uint64_t, const std::uint64_t* saved) {
    t_pending.push_back({saved[6], kResource1D, saved[5], 0, saved[2]});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
    return 0;
}

// A buffer's resource release: rdi = holder + 0x18.
GUEST_ABI std::int64_t buffer_release_hook(std::uint64_t, const std::uint64_t* saved) {
    unregister_holder(saved[5] - 0x18);
    return 0;
}

void trace_map(const Pending& p) {
    if (p.ring) {
        std::uint64_t ptr = 0;
        if (p.out) rd(p.out, &ptr);
        // The 128-byte constant-buffer allocations are thousands a frame, and
        // the menus' quads come before the world: the log starts at
        // BBHOST_GX_MAP_FROM_FLIP (default 3000) and takes 600 of the rest.
        static const std::uint64_t from_flip = [] {
            const char* e = std::getenv("BBHOST_GX_MAP_FROM_FLIP");
            return e && e[0] ? std::strtoull(e, nullptr, 10) : 3000ull;
        }();
        if (p.ring_bytes != 128 && hle_video_flip_count() >= from_flip) {
            static std::mutex mu;
            static std::map<std::uint64_t, int> per_caller;  // the first eight calls of each caller
            const std::uint64_t caller = bn(p.ret);
            int n;
            {
                std::lock_guard<std::mutex> lk(mu);
                n = per_caller[caller]++;
            }
            if (n < 8) {
                host_log("gx-ring: caller 0x%llx takes %llu bytes -> 0x%llx", static_cast<unsigned long long>(caller),
                         static_cast<unsigned long long>(p.ring_bytes), static_cast<unsigned long long>(ptr));
            }
        }
        return;
    }
    std::uint64_t data = 0;
    if (p.out) rd(p.out, &data);
    const std::uint64_t caller = bn(p.ret);
    bool first = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        first = g_trace_callers[caller]++ == 0;
    }
    if (first && g_trace_logged.fetch_add(1) < 64) {
        char who[160] = "";
#if !defined(_WIN32)
        if (p.ret > 0x100000000000ull) {  // a host caller: name it
            Dl_info info{};
            if (dladdr(reinterpret_cast<void*>(static_cast<std::uintptr_t>(p.ret)), &info) && info.dli_sname) {
                std::snprintf(who, sizeof(who), " (host %s+0x%llx)", info.dli_sname,
                              static_cast<unsigned long long>(p.ret - reinterpret_cast<std::uintptr_t>(info.dli_saddr)));
            } else {
                struct Find { std::uint64_t a; std::uint64_t base; const char* name; bool hit; } f{p.ret, 0, nullptr, false};
                dl_iterate_phdr(
                    [](dl_phdr_info* i, std::size_t, void* d) {
                        auto* f = static_cast<Find*>(d);
                        for (int k = 0; k < i->dlpi_phnum; ++k) {
                            const auto& ph = i->dlpi_phdr[k];
                            const std::uint64_t lo = i->dlpi_addr + ph.p_vaddr;
                            if (ph.p_type == PT_LOAD && f->a >= lo && f->a < lo + ph.p_memsz) {
                                *f = {f->a, i->dlpi_addr, i->dlpi_name, true};
                                return 1;
                            }
                        }
                        return 0;
                    },
                    &f);
                if (f.hit) std::snprintf(who, sizeof(who), " (host %s+0x%llx)", f.name && f.name[0] ? f.name : "bbhost",
                                         static_cast<unsigned long long>(p.ret - f.base));
                else std::snprintf(who, sizeof(who), " (host, generated code)");
            }
        }
#endif
        host_log("gx-map: caller 0x%llx%s maps resource 0x%llx type %llu -> 0x%llx", static_cast<unsigned long long>(caller), who,
                 static_cast<unsigned long long>(p.holder), static_cast<unsigned long long>(p.type),
                 static_cast<unsigned long long>(data));
    }
}

}  // namespace

extern "C" GUEST_ABI std::uint64_t gx_res_after(std::uint64_t result) {
    const Pending p = t_pending.back();
    t_pending.pop_back();
    if ((p.kind == kBuffer || p.kind == kTexture || p.kind == kTexture3D) && static_cast<std::uint32_t>(result) != 0) {
        const std::uint64_t n = g_create_failures.fetch_add(1, std::memory_order_relaxed);
        std::uint32_t wh[2] = {0, 0};
        if (n < 8 && p.holder && rd(p.holder, &wh))
            host_log("gx-resources: a %s of %ux%u was not made (error %u)", p.kind == kBuffer ? "buffer" : "texture", wh[0], wh[1],
                     static_cast<std::uint32_t>(result));
    }
    if (p.kind == kMapCall) {
        if (!p.ring) note_map(p.holder, p.type);
        if (g_trace) trace_map(p);
    } else if (p.kind == kObject) {
        std::uint64_t object = 0;
        if (p.out && rd(p.out, &object) && object) {
            std::uint32_t extent = 0;
            std::uint64_t words = 0;
            if (p.type == 1 || p.type == 2) {  // a view
                const bool depth = p.type == 2;
                extent = view_extent(object, depth);
                g_views[depth].fetch_add(1, std::memory_order_relaxed);
                if (extent) g_views_extent[depth].fetch_add(1, std::memory_order_relaxed);
            } else if (p.type == 3 || p.type == 4) {  // a shader-resource view's T#, a sampler's S#
                std::uint8_t w[32];
                if (host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(object + (p.type == 3 ? 0x10 : 0))), w, 32)) {
                    words = gx_words_hash(w);
                }
                g_views[p.type - 1].fetch_add(1, std::memory_order_relaxed);
            }
            register_object(object, extent, words);
        }
    } else if (p.kind == kTsharp) {
        t_tsharp_valid = p.out && rd(p.out, &t_tsharp);
    } else {
        // The holder (creators write it through their out pointer), or the 1D
        // resource struct itself, read as if it sat in a holder.
        std::uint64_t holder = 0, key = 0;
        if (p.kind == kResource1D) {
            std::uint8_t type = 0;
            if (p.out && rd(p.out + 0x20, &type) && type == 2) {
                holder = p.out - 0x18;
                key = kMemoryKey;
            }
        } else if (p.out && rd(p.out, &holder)) {
            key = holder;
        }
        if (holder) {
            register_holder(key, holder, p.kind, p.type != 0);
            if (key & kMemoryKey) {
                std::uint64_t memory = 0;
                rd(holder + 0x20, &memory);
                key = kMemoryKey | memory;
            }
            Entry e;
            bool found = false;
            {
                std::lock_guard<std::mutex> lk(g_mu);
                if (auto it = g_by_holder.find(key); it != g_by_holder.end()) {
                    if (t_tsharp_valid && (it->second.info.type == 2 || it->second.info.type == 3)) {
                        std::memcpy(it->second.tsharp, t_tsharp, sizeof(t_tsharp));
                        it->second.has_tsharp = true;
                    }
                    e = it->second;
                    found = true;
                }
            }
            if (found) create_ahead(e);
        }
        t_tsharp_valid = false;
    }
    return p.ret;
}

extern "C" void* g_gx_res_after_thunk = nullptr;
#if defined(_WIN32)
#define GR_TYPE_(n)
#else
#define GR_TYPE_(n) ".type " #n ", @function\n"
#endif
asm(".text\n.globl gx_res_ret\n" GR_TYPE_(gx_res_ret) "gx_res_ret:\n"
    "push %rax\n push %rdx\n mov %rax, %rdi\n call *g_gx_res_after_thunk(%rip)\n"
    "mov %rax, %r11\n pop %rdx\n pop %rax\n jmp *%r11\n");

std::uint64_t gx_resources_create_failures() { return g_create_failures.load(std::memory_order_relaxed); }

void gx_resources_install(ElfImage* image) {
    g_slide = image->mem.slide;
    g_trace = [] {
        const char* e = std::getenv("BBHOST_GX_MAP_TRACE");
        return e && e[0] == '1';
    }();
    if (const char* e = std::getenv("BBHOST_GX_RESOURCES"); e && e[0] == '0') {
        host_log("gx-resources: off (BBHOST_GX_RESOURCES=0)");
        return;
    }
    g_gx_res_after_thunk = hle_wrap_fn(reinterpret_cast<void*>(&gx_res_after));
    const auto at = [&](std::uint64_t bn_va) { return image->mem.slide + (bn_va - kPreferredGuestSlide); };
    int hooked = 0, refused = 0;
    const auto hook = [&](std::uint64_t va, const std::uint8_t* prologue, std::size_t n, void* host, std::uint64_t id) {
        if (engine_prologue_hook(image, at(va), prologue, n, host, id)) ++hooked;
        else {
            ++refused;
            host_log("gx-resources: 0x%llx is not the prologue expected; not hooked", static_cast<unsigned long long>(va));
        }
    };
    hook(kCreateBuffer, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&create_hook), kBuffer);
    hook(kCreateTexture, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&create_hook), kTexture);
    hook(kCreateTexture3D, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&create_hook), kTexture3D);
    hook(kMap, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&map_hook), 0);
    if (g_tex_ahead) hook(kTsharpBuild, kTsharpBuildPrologue, sizeof(kTsharpBuildPrologue), reinterpret_cast<void*>(&tsharp_hook), 0);
    hook(kBufferRelease, kBufferReleasePrologue, sizeof(kBufferReleasePrologue), reinterpret_cast<void*>(&buffer_release_hook), 0);
    // The texture destructors: the prologue's lea of the vtable is
    // rip-relative, so the stub runs a movabs of the slid vtable instead.
    const auto dtor_with = [&](std::uint64_t va, std::uint64_t vtable, void* handler) {
        std::uint8_t prologue[16], exec[19];
        std::memcpy(prologue, kDtorHead, sizeof(kDtorHead));
        prologue[9] = 0x48;
        prologue[10] = 0x8d;
        prologue[11] = 0x05;
        const std::uint32_t disp = static_cast<std::uint32_t>(vtable - (va + 16));
        std::memcpy(prologue + 12, &disp, 4);
        std::memcpy(exec, kDtorHead, sizeof(kDtorHead));
        exec[9] = 0x48;
        exec[10] = 0xb8;  // movabs rax, imm64
        const std::uint64_t slid = at(vtable);
        std::memcpy(exec + 11, &slid, 8);
        if (engine_prologue_hook_exec(image, at(va), prologue, sizeof(prologue), exec, sizeof(exec), handler, 0)) {
            ++hooked;
        } else {
            ++refused;
            host_log("gx-resources: 0x%llx is not the destructor prologue expected; not hooked", static_cast<unsigned long long>(va));
        }
    };
    const auto dtor = [&](std::uint64_t va, std::uint64_t vtable) { dtor_with(va, vtable, reinterpret_cast<void*>(&texture_dtor_hook)); };
    const auto dtor1d = [&](std::uint64_t va) { dtor_with(va, kTex1dVtable, reinterpret_cast<void*>(&texture1d_dtor_hook)); };
    dtor(kTexDelete, kTexVtable);
    dtor(kTexDestroy, kTexVtable);
    dtor(kTex3dDelete, kTex3dVtable);
    dtor(kTex3dDestroy, kTex3dVtable);
    hook(kCreate1D, kCreate1DPrologue, sizeof(kCreate1DPrologue), reinterpret_cast<void*>(&create1d_hook), kResource1D);
    if (g_tex_ahead) hook(kTsharpBuild1D, kTsharpBuildPrologue, sizeof(kTsharpBuildPrologue), reinterpret_cast<void*>(&tsharp_hook), 0);
    dtor1d(kTex1dDelete);
    dtor1d(kTex1dDestroy);
    if (g_trace) hook(kRing, kRingPrologue, sizeof(kRingPrologue), reinterpret_cast<void*>(&ring_hook), 0);
    hook(kCreateInputLayout, kInputLayoutPrologue, sizeof(kInputLayoutPrologue), reinterpret_cast<void*>(&object_create_hook), 0);
    hook(kCreateRtv, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&view_create_hook), 1);
    hook(kCreateDsv, kDsvPrologue, sizeof(kDsvPrologue), reinterpret_cast<void*>(&view_create_hook), 2);
    hook(kCreateSrv, kPrologue58, sizeof(kPrologue58), reinterpret_cast<void*>(&view_create_hook), 3);
    hook(kCreateSampler, kSamplerPrologue, sizeof(kSamplerPrologue), reinterpret_cast<void*>(&view_create_hook), 4);
    host_log("gx-resources: registry on: %d hooks (creators, Map, destructors)%s%s", hooked, refused ? ", some refused" : "",
             g_trace ? "; Map and ring callers logged" : "");
}

std::uint64_t gx_guest_to_bn(std::uint64_t va) { return g_slide ? bn(va) : va; }

void gx_objects_watch_creator(std::uint64_t* frame) {
    if (!g_gx_res_after_thunk) return;
    t_pending.push_back({frame[6], kObject, frame[4], 0, 0});
    frame[6] = reinterpret_cast<std::uint64_t>(&gx_res_ret);
}

std::uint32_t gx_view_extent(std::uint64_t view) {
    if (!view) return 0;
    const std::uint32_t h = obj_slot(view);
    for (std::uint32_t probe = 0; probe < kObjProbes; ++probe) {
        const ObjSlot& s = g_obj[(h + probe) & (kObjSlots - 1)];
        const std::uint64_t o = s.object.load(std::memory_order_acquire);
        if (o == view) return s.extent.load(std::memory_order_relaxed);
        if (!o) break;
    }
    return 0;
}

std::uint64_t gx_words_hash(const std::uint8_t* w) {
    std::uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 32; i += 8) {
        std::uint64_t v;
        std::memcpy(&v, w + i, 8);
        h = (h ^ v) * 1099511628211ull;
    }
    return h | 1;
}

std::uint32_t gx_object_lookup(std::uint64_t object, std::uint64_t* words) {
    if (!object) return 0;
    const std::uint32_t h = obj_slot(object);
    for (std::uint32_t probe = 0; probe < kObjProbes; ++probe) {
        const ObjSlot& s = g_obj[(h + probe) & (kObjSlots - 1)];
        const std::uint64_t o = s.object.load(std::memory_order_acquire);
        if (o == object) {
            const std::uint32_t id = s.id.load(std::memory_order_acquire);
            if (words) *words = s.words.load(std::memory_order_relaxed);
            return id;
        }
        if (!o) break;
    }
    return 0;
}

std::uint32_t gx_object_id(std::uint64_t object) {
    if (!object) return 0;
    const std::uint32_t h = obj_slot(object);
    for (std::uint32_t probe = 0; probe < kObjProbes; ++probe) {
        const ObjSlot& s = g_obj[(h + probe) & (kObjSlots - 1)];
        const std::uint64_t o = s.object.load(std::memory_order_acquire);
        if (o == object) return s.id.load(std::memory_order_acquire);
        if (!o) break;
    }
    return 0;
}

bool gx_resource_tsharp(std::uint64_t memory, std::uint32_t out[8]) {
    std::lock_guard<std::mutex> lk(g_mu);
    const auto m = g_by_memory.find(memory);
    if (m == g_by_memory.end()) return false;
    const auto h = g_by_holder.find(m->second);
    if (h == g_by_holder.end() || !h->second.has_tsharp) return false;
    std::memcpy(out, h->second.tsharp, 32);
    out[0] = static_cast<std::uint32_t>(memory >> 8);
    out[1] = (out[1] & ~0xffu) | static_cast<std::uint32_t>((memory >> 40) & 0xff);
    return true;
}

void gx_resources_memory_protected(std::uint64_t va, std::size_t len, int prot) {
    if (!g_gx_res_after_thunk || !len) return;
    std::lock_guard<std::mutex> lk(g_mu);
    std::uint64_t n = 0;
    std::uint32_t first_id = 0;
    std::uint8_t first_type = 0;
    for (auto it = g_by_memory.lower_bound(va); it != g_by_memory.end() && it->first < va + len; ++it) {
        if (auto h = g_by_holder.find(it->second); h != g_by_holder.end() && !n) {
            first_id = h->second.info.id;
            first_type = h->second.info.type;
        }
        ++n;
    }
    if (!n) return;
    g_protected.fetch_add(n, std::memory_order_relaxed);
    if (g_protect_events.fetch_add(1, std::memory_order_relaxed) < 6) {
        host_log("gx-resources: mprotect of [0x%llx, +0x%zx) to %d at flip %llu covers %llu live resources (first: id %u type %u)",
                 static_cast<unsigned long long>(va), len, prot, static_cast<unsigned long long>(hle_video_flip_count()),
                 static_cast<unsigned long long>(n), first_id, first_type);
    }
}

void gx_resources_memory_released(std::uint64_t va, std::size_t len) {
    if (!g_gx_res_after_thunk || !len) return;
    std::uint64_t n = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_by_memory.empty()) return;
        for (auto it = g_by_memory.lower_bound(va); it != g_by_memory.end() && it->first < va + len;) {
            auto h = g_by_holder.find(it->second);
            if (h != g_by_holder.end()) h->second.info.alive = false;  // the holder may still be destroyed later
            it = g_by_memory.erase(it);
            ++n;
        }
    }
    retire_memory(va, static_cast<std::uint32_t>(std::min<std::size_t>(len, 0xffffffffu)));
    if (n) {
        g_released_under_holder.fetch_add(n, std::memory_order_relaxed);
        if (g_release_events.fetch_add(1, std::memory_order_relaxed) < 6) {
            host_log("gx-resources: unmap of [0x%llx, +0x%zx) at flip %llu released the memory of %llu live resources", static_cast<unsigned long long>(va),
                     len, static_cast<unsigned long long>(hle_video_flip_count()), static_cast<unsigned long long>(n));
        }
    }
}

bool gx_resource_at(std::uint64_t va, std::size_t bytes, GxResourceInfo* out) {
    g_lookups.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_by_memory.upper_bound(va);
    if (it == g_by_memory.begin()) return false;
    --it;
    auto h = g_by_holder.find(it->second);
    if (h == g_by_holder.end()) return false;
    const GxResourceInfo& r = h->second.info;
    if (va + bytes > r.memory + r.bytes) return false;
    g_lookup_hits.fetch_add(1, std::memory_order_relaxed);
    if (out) *out = r;
    return true;
}

bool gx_resource_newer_overlapping(std::uint64_t va, std::size_t bytes, std::uint64_t after_flip) {
    if (!bytes) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    // Live resources do not overlap each other: those that start inside the
    // range, and the one before it if it reaches in.
    auto it = g_by_memory.lower_bound(va);
    const auto newer = [&](std::map<std::uint64_t, std::uint64_t>::const_iterator m) {
        auto h = g_by_holder.find(m->second);
        if (h == g_by_holder.end()) return false;
        const GxResourceInfo& r = h->second.info;
        return r.alive && r.created_flip > after_flip && r.memory < va + bytes && r.memory + r.bytes > va;
    };
    if (it != g_by_memory.begin() && newer(std::prev(it))) return true;
    for (; it != g_by_memory.end() && it->first < va + bytes; ++it) {
        if (newer(it)) return true;
    }
    return false;
}

void gx_resources_report() {
    if (!g_gx_res_after_thunk) return;
    std::size_t live = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        live = g_by_memory.size();
    }
    using ull = unsigned long long;
    static const char* const kType[5] = {"other", "buffers", "1D textures", "2D textures", "3D textures"};
    std::string created;
    for (unsigned t = 0; t < 5; ++t) {
        if (!g_created[t].load()) continue;
        char buf[160];
        std::snprintf(buf, sizeof(buf), " %s %llu (%llu with initial data, %llu MiB)", kType[t], static_cast<ull>(g_created[t].load()),
                      static_cast<ull>(g_created_with_data[t].load()), static_cast<ull>(g_created_bytes[t].load() >> 20));
        created += buf;
    }
    host_log("gx-resources: created%s; destroyed %llu (%llu unknown, %llu holders reused before their destructor, %llu memories reused "
             "before their holder's); memory unmapped under a live holder %llu resources in %llu unmaps, re-protected %llu in %llu mprotects; "
             "live %zu, peak %llu",
             created.c_str(), static_cast<ull>(g_destroyed.load()), static_cast<ull>(g_destroyed_unknown.load()),
             static_cast<ull>(g_replaced_holder.load()), static_cast<ull>(g_memory_reused.load()), static_cast<ull>(g_released_under_holder.load()),
             static_cast<ull>(g_release_events.load()), static_cast<ull>(g_protected.load()), static_cast<ull>(g_protect_events.load()), live,
             static_cast<ull>(g_peak_live.load()));
    std::string maps;
    for (unsigned t = 0; t < 5; ++t) {
        ull n[5];
        ull sum = 0;
        for (unsigned m = 0; m < 5; ++m) sum += n[m] = g_maps[t][m].load();
        if (!sum) continue;
        char buf[200];
        std::snprintf(buf, sizeof(buf), " %s: read %llu, write %llu, no-overwrite %llu, discard %llu;", kType[t], n[1], n[2], n[3], n[4]);
        maps += buf;
    }
    host_log("gx-resources: Map by resource and type%s unknown holders %llu, memory moved by a map %llu; lookups %llu (%llu hits)",
             maps.empty() ? " none;" : maps.c_str(), static_cast<ull>(g_map_unknown.load()), static_cast<ull>(g_map_moved.load()),
             static_cast<ull>(g_lookups.load()), static_cast<ull>(g_lookup_hits.load()));
    host_log("gx-resources: shader and input-layout objects created %llu (%llu at an address an earlier one had), not registered (table "
             "full) %llu",
             static_cast<ull>(g_obj_created.load()), static_cast<ull>(g_obj_recreated.load()), static_cast<ull>(g_obj_full.load()));
    host_log("gx-resources: views made: render target %llu (%llu of a 2D texture with its extent), depth-stencil %llu (%llu), shader "
             "resource %llu; samplers %llu",
             static_cast<ull>(g_views[0].load()), static_cast<ull>(g_views_extent[0].load()), static_cast<ull>(g_views[1].load()),
             static_cast<ull>(g_views_extent[1].load()), static_cast<ull>(g_views[2].load()), static_cast<ull>(g_views[3].load()));
    host_log("gx-resources: textures created ahead of their first bind %llu (%llu MiB), refused by the cache %llu, no T# seen %llu, "
             "destroyed before their turn %llu, queue full %llu; retirements queued %llu%s",
             static_cast<ull>(g_ahead_created.load()), static_cast<ull>(g_ahead_bytes.load() >> 20), static_cast<ull>(g_ahead_refused.load()),
             static_cast<ull>(g_ahead_no_tsharp.load()), static_cast<ull>(g_ahead_gone.load()), static_cast<ull>(g_ahead_dropped.load()),
             static_cast<ull>(g_retire_queued.load()), g_tex_ahead ? "" : " (BBHOST_GX_TEX_AHEAD=0)");
    if (g_trace) {
        std::lock_guard<std::mutex> lk(g_mu);
        host_log("gx-map: %zu distinct callers", g_trace_callers.size());
        for (const auto& [caller, n] : g_trace_callers) {
            host_log("gx-map:   0x%llx x%llu", static_cast<ull>(caller), static_cast<ull>(n));
        }
    }
}
