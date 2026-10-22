// GX's upload heap's move (BBHOST_GXVK_ARENA=2) on a GPU, outside the game:
// the steps bbhost runs (host/arena_move.h over host/arena_move_vk.h), on a
// heap like the game's - 160 MiB of host memory at an address below 1 TiB,
// a pattern in it - beside a mapped 32 MiB buffer where bbhost's params ring
// is, while four threads write the heap all the while (a write that faults
// while the move runs waits and retries, as in bbhost's fault handler).
//   1. the undo: a move whose placing fails at one piece must leave the heap
//      host memory again, writable, holding the pattern and every write;
//   2. the move: every piece placed; the GPU must read the pattern and every
//      write from each piece's buffer (a copy of every piece into host memory,
//      compared whole), and the CPU the same through the heap (every write's
//      slot and a sample of the rest: reading video memory from the CPU runs at
//      tens of MB/s, so not every byte).
// It prints what the driver reports and exits 0 when both hold. Seconds long.
//   arena_move_check [--piece-mib N] [--fail-at K]
#include <cstdio>

#if defined(__linux__)
#include <vulkan/vulkan.h>

#include "host/arena_move_vk.h"

#include <signal.h>
#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(VK_EXT_map_memory_placed) && defined(VK_KHR_map_memory2)
namespace {

std::atomic<std::uint64_t> g_lo{0}, g_hi{0};
std::atomic<bool> g_moving{false};

void on_moving(bool on) { g_moving.store(on, std::memory_order_release); }

// bbhost's gxvk::arena_move_fault: a fault on the heap while it moves waits for the move, then retries.
void on_segv(int sig, siginfo_t* si, void*) {
    const std::uint64_t a = reinterpret_cast<std::uint64_t>(si->si_addr);
    if (a >= g_lo.load() && a < g_hi.load() && g_moving.load(std::memory_order_acquire)) {
        while (g_moving.load(std::memory_order_acquire)) __builtin_ia32_pause();
        return;
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

struct Writers {
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    // Slots of 8 bytes, one in every 2 KiB of the heap, writer w owning slot i when i % 4 == w; last[i] its last value,
    // init[i] what the slot held before the writers started.
    std::vector<std::uint64_t> last, init;
    std::uint64_t lo = 0, slots = 0;
    void start(std::uint64_t heap, std::uint64_t bytes) {
        lo = heap;
        slots = bytes / 2048;
        last.assign(slots, 0);
        init.assign(slots, 0);
        for (std::uint64_t i = 0; i < slots; ++i) std::memcpy(&init[i], reinterpret_cast<const void*>(lo + i * 2048), 8);
        stop = false;
        for (int w = 0; w < 4; ++w) {
            threads.emplace_back([this, w] {
                std::mt19937_64 rng(1234 + w);
                std::uint64_t v = static_cast<std::uint64_t>(w + 1) << 56;
                while (!stop.load(std::memory_order_relaxed)) {
                    const std::uint64_t i = (rng() % (slots / 4)) * 4 + static_cast<std::uint64_t>(w);
                    ++v;
                    *reinterpret_cast<volatile std::uint64_t*>(lo + i * 2048) = v;
                    last[i] = v;
                }
            });
        }
    }
    void finish() {
        stop = true;
        for (auto& t : threads) t.join();
        threads.clear();
    }
};

// The pattern a heap byte holds where no writer's slot is.
std::uint8_t pattern(std::uint64_t off) { return static_cast<std::uint8_t>((off * 2654435761u) >> 13); }

// Every byte of the heap as the pattern and the writers left it; the first difference's offset, or ~0.
std::uint64_t check_cpu(const std::uint8_t* heap, std::uint64_t bytes, const Writers& w) {
    for (std::uint64_t off = 0; off < bytes; ++off) {
        if (off % 2048 < 8) continue;  // a slot
        if (heap[off] != pattern(off)) return off;
    }
    for (std::uint64_t i = 0; i < w.slots; ++i) {
        std::uint64_t v = 0;
        std::memcpy(&v, heap + i * 2048, 8);
        const std::uint64_t want = w.last[i] ? w.last[i] : w.init[i];
        if (v != want) return i * 2048;
    }
    return ~0ull;
}

// What heap byte `off` must hold: a writer's slot's last value (or what it held before them), else the pattern.
std::uint8_t expected(std::uint64_t off, const Writers& w) {
    if (off % 2048 < 8) {
        const std::uint64_t i = off / 2048, v = w.last[i] ? w.last[i] : w.init[i];
        return static_cast<std::uint8_t>(v >> (8 * (off % 2048)));
    }
    return pattern(off);
}

struct Fail : arena_move::VkOps {
    std::uint64_t fail_at = ~0ull;
    bool place(std::uint64_t k, std::uint64_t at) { return k == fail_at ? false : arena_move::VkOps::place(k, at); }
};

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::uint64_t piece = arena_move::kPieceDefault, fail_at = 2;
    for (int i = 1; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--piece-mib")) piece = arena_move::piece_bytes(argv[++i]);
        else if (!std::strcmp(argv[i], "--fail-at")) fail_at = std::strtoull(argv[++i], nullptr, 10);
    }
    const std::uint64_t kHeap = 160ull << 20, kRing = 32ull << 20;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return std::puts("no Vulkan instance"), 2;
    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    for (VkPhysicalDevice d : pds) {
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            pd = d;
            break;
        }
    }
    if (!pd) return std::puts("no discrete GPU"), 2;
    VkPhysicalDeviceMapMemoryPlacedFeaturesEXT q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 qf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    qf.pNext = &q;
    vkGetPhysicalDeviceFeatures2(pd, &qf);
    std::printf("%s, driver 0x%x: memoryMapPlaced %u, memoryUnmapReserve %u\n", props.deviceName, props.driverVersion, q.memoryMapPlaced,
                q.memoryUnmapReserve);
    if (!q.memoryMapPlaced) return std::puts("no placed mappings"), 2;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char* exts[] = {VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME, VK_KHR_MAP_MEMORY_2_EXTENSION_NAME, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceMapMemoryPlacedFeaturesEXT en{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
    en.pNext = &v12;
    en.memoryMapPlaced = VK_TRUE;
    en.memoryUnmapReserve = q.memoryUnmapReserve;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &en;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 3;
    dci.ppEnabledExtensionNames = exts;
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(pd, &dci, nullptr, &dev) != VK_SUCCESS) return std::puts("no device"), 2;
    VkQueue queue;
    vkGetDeviceQueue(dev, 0, 0, &queue);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    // The memory type: as bbhost picks it for a piece's buffer.
    std::uint32_t type = UINT32_MAX;
    {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = piece;
        bci.usage = arena_move::kPieceUsage;
        VkBuffer probe;
        vkCreateBuffer(dev, &bci, nullptr, &probe);
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, probe, &req);
        vkDestroyBuffer(dev, probe, nullptr);
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t t = 0; t < mp.memoryTypeCount && type == UINT32_MAX; ++t)
            if ((req.memoryTypeBits >> t & 1) && (mp.memoryTypes[t].propertyFlags & want) == want) type = t;
    }
    if (type == UINT32_MAX) return std::puts("no video memory the CPU writes"), 2;
    const std::uint32_t heap = mp.memoryTypes[type].heapIndex;
    const auto heap_now = [&] {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT mb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 mp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        mp2.pNext = &mb;
        vkGetPhysicalDeviceMemoryProperties2(pd, &mp2);
        arena_memory::Heap h;
        h.size = mp.memoryHeaps[heap].size;
        h.flags = mp.memoryHeaps[heap].flags;
        h.have_budget = mb.heapBudget[heap] != 0;
        h.budget = mb.heapBudget[heap];
        h.used = mb.heapUsage[heap];
        return h;
    };
    const auto say_heap = [&](const char* when) {
        const arena_memory::Heap h = heap_now();
        std::printf("  heap %u (type %u): %llu MiB, %llu in use, budget %llu - %s\n", heap, type, static_cast<unsigned long long>(h.size >> 20),
                    static_cast<unsigned long long>(h.used >> 20), static_cast<unsigned long long>(h.budget >> 20), when);
    };
    say_heap("at the start");
    // bbhost's params ring, mapped, where it is when the heap moves.
    VkDeviceMemory ring = VK_NULL_HANDLE;
    void* ring_map = nullptr;
    {
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = kRing;
        mai.memoryTypeIndex = type;
        if (vkAllocateMemory(dev, &mai, nullptr, &ring) != VK_SUCCESS) return std::puts("no params ring"), 2;
        vkMapMemory(dev, ring, 0, VK_WHOLE_SIZE, 0, &ring_map);
    }
    say_heap("with the params ring");
    std::printf("  room for the heap by bbhost's rule: %s\n", arena_move::bar_room(heap_now(), kHeap) ? "yes" : "NO");
    // The heap, as hle_kernel_map_host_pieces maps it.
    std::uint64_t lo = 0;
    for (std::uint64_t hint = 0xF000000000ull; hint >= 0xC000000000ull && !lo; hint -= 0x40000000ull) {
        void* m = mmap(reinterpret_cast<void*>(hint), kHeap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (m == reinterpret_cast<void*>(hint)) lo = hint;
        else if (m != MAP_FAILED) munmap(m, kHeap);
    }
    if (!lo) return std::puts("no heap mapping"), 2;
    std::uint8_t* const h8 = reinterpret_cast<std::uint8_t*>(lo);
    for (std::uint64_t off = 0; off < kHeap; ++off) h8[off] = pattern(off);
    g_lo = lo;
    g_hi = lo + kHeap;
    struct sigaction sa {};
    sa.sa_sigaction = on_segv;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, nullptr);
    const arena_move::Pieces layout{lo, kHeap, piece};
    std::printf("  the heap at 0x%llx, %llu pieces of up to %llu MiB\n", static_cast<unsigned long long>(lo),
                static_cast<unsigned long long>(layout.count()), static_cast<unsigned long long>(piece >> 20));
    std::vector<arena_move::VkPiece> adopted;
    static std::vector<arena_move::VkPiece>* s_adopted = nullptr;
    s_adopted = &adopted;
    const auto ops_for = [&](Fail& o) {
        o.dev = dev;
        o.p = &layout;
        o.type = type;
        o.map2 = reinterpret_cast<PFN_vkMapMemory2KHR>(vkGetDeviceProcAddr(dev, "vkMapMemory2KHR"));
        o.unmap2 = reinterpret_cast<PFN_vkUnmapMemory2KHR>(vkGetDeviceProcAddr(dev, "vkUnmapMemory2KHR"));
        o.unmap_reserve = q.memoryUnmapReserve == VK_TRUE;
        o.on_moving = on_moving;
        o.on_adopt = [](std::uint64_t, const arena_move::VkPiece& pc) { s_adopted->push_back(pc); };
    };
    int failed = 0;
    // 1. The undo.
    {
        Writers w;
        w.start(lo, kHeap);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Fail o;
        ops_for(o);
        o.fail_at = fail_at;
        const auto t0 = std::chrono::steady_clock::now();
        const arena_move::Result r = arena_move::move(o, layout);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        w.finish();
        const std::uint64_t bad = check_cpu(h8, kHeap, w);
        // Writable again: one more write through every page.
        for (std::uint64_t off = 4096 - 8; off < kHeap; off += 4096) h8[off] = pattern(off);
        const bool ok = !r.ok && r.piece == fail_at && bad == ~0ull && adopted.empty();
        std::printf("1. the undo (placing fails at piece %llu): %s in %.1f ms - \"%s\"; the heap %s, writable; %s\n",
                    static_cast<unsigned long long>(fail_at), r.ok ? "MOVED" : "undone", ms, r.why,
                    bad == ~0ull ? "holds the pattern and every write" : ("DIFFERS at +" + std::to_string(bad)).c_str(),
                    ok ? "ok" : "WRONG");
        say_heap("after the undo");
        if (!ok) ++failed;
    }
    // 2. The move.
    Writers w;
    w.start(lo, kHeap);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Fail o;
    ops_for(o);
    const auto t0 = std::chrono::steady_clock::now();
    const arena_move::Result r = arena_move::move(o, layout);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    w.finish();
    std::printf("2. the move: %s in %.1f ms (%s), %zu pieces adopted\n", r.ok ? "moved" : "FAILED", ms, r.why, adopted.size());
    say_heap("after the move");
    if (!r.ok || adopted.size() != layout.count()) ++failed;
    if (r.ok) {
        // The GPU's view: each piece copied into host memory and compared whole with what the CPU wrote.
        VkBufferCreateInfo rb{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        rb.size = piece;
        rb.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBuffer back;
        vkCreateBuffer(dev, &rb, nullptr, &back);
        VkMemoryRequirements rr{};
        vkGetBufferMemoryRequirements(dev, back, &rr);
        std::uint32_t ht = UINT32_MAX;
        for (std::uint32_t t = 0; t < mp.memoryTypeCount && ht == UINT32_MAX; ++t)
            if ((rr.memoryTypeBits >> t & 1) && (mp.memoryTypes[t].propertyFlags & 0xe) == 0xe && !(mp.memoryTypes[t].propertyFlags & 1)) ht = t;
        VkMemoryAllocateInfo ra{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ra.allocationSize = rr.size;
        ra.memoryTypeIndex = ht;
        VkDeviceMemory bm;
        vkAllocateMemory(dev, &ra, nullptr, &bm);
        vkBindBufferMemory(dev, back, bm, 0);
        void* bmap = nullptr;
        vkMapMemory(dev, bm, 0, VK_WHOLE_SIZE, 0, &bmap);
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        VkCommandPool pool;
        vkCreateCommandPool(dev, &pci, nullptr, &pool);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cb;
        vkAllocateCommandBuffers(dev, &cai, &cb);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        vkCreateFence(dev, &fci, nullptr, &fence);
        std::uint64_t same = 0, first_bad = ~0ull;
        const auto g0 = std::chrono::steady_clock::now();
        for (std::uint64_t k = 0; k < adopted.size(); ++k) {
            vkResetCommandBuffer(cb, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(cb, &bi);
            VkBufferCopy cp{0, 0, adopted[k].size};
            vkCmdCopyBuffer(cb, adopted[k].buffer, back, 1, &cp);
            vkEndCommandBuffer(cb);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            vkResetFences(dev, 1, &fence);
            vkQueueSubmit(queue, 1, &si, fence);
            vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull);
            const std::uint8_t* got = static_cast<const std::uint8_t*>(bmap);
            const std::uint64_t base = layout.start(k) - lo;
            bool ok = true;
            for (std::uint64_t i = 0; i < adopted[k].size && ok; ++i)
                if (got[i] != expected(base + i, w)) {
                    ok = false;
                    if (first_bad == ~0ull) first_bad = base + i;
                }
            if (ok) ++same;
        }
        const double gms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count();
        std::printf("   the GPU reads the pattern and every write in %llu of %zu pieces%s (%.0f ms)\n", static_cast<unsigned long long>(same),
                    adopted.size(), first_bad == ~0ull ? "" : (" - first difference at +" + std::to_string(first_bad)).c_str(), gms);
        if (same != adopted.size()) ++failed;
        // The CPU's view through the heap: every writer slot and 4096 bytes elsewhere.
        const auto c0 = std::chrono::steady_clock::now();
        std::uint64_t cpu_bad = 0, reads = 0;
        for (std::uint64_t i = 0; i < w.slots; ++i, ++reads) {
            std::uint64_t v = 0;
            std::memcpy(&v, h8 + i * 2048, 8);
            if (v != (w.last[i] ? w.last[i] : w.init[i])) ++cpu_bad;
        }
        std::mt19937_64 rng(99);
        for (int i = 0; i < 4096; ++i, ++reads) {
            const std::uint64_t off = rng() % kHeap;
            if (h8[off] != expected(off, w)) ++cpu_bad;
        }
        const double cms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
        std::printf("   the CPU reads them through the heap: %llu of %llu reads differ (%.0f ms)\n", static_cast<unsigned long long>(cpu_bad),
                    static_cast<unsigned long long>(reads), cms);
        if (cpu_bad) ++failed;
        vkDestroyFence(dev, fence, nullptr);
        vkDestroyCommandPool(dev, pool, nullptr);
        vkDestroyBuffer(dev, back, nullptr);
        vkFreeMemory(dev, bm, nullptr);
    }
    // Clean up: the pieces unmapped (the heap's range with them), then freed.
    for (const arena_move::VkPiece& pc : adopted) {
        vkUnmapMemory(dev, pc.memory);
        vkDestroyBuffer(dev, pc.buffer, nullptr);
        vkFreeMemory(dev, pc.memory, nullptr);
    }
    vkUnmapMemory(dev, ring);
    vkFreeMemory(dev, ring, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    std::printf(failed ? "arena_move_check: %d failed\n" : "arena_move_check: ok\n", failed);
    return failed ? 1 : 0;
}
#else
int main() {
    std::puts("arena_move_check: this build's Vulkan headers have no VK_EXT_map_memory_placed");
    return 2;
}
#endif
#else
int main() {
    std::puts("arena_move_check: Linux only");
    return 2;
}
#endif
