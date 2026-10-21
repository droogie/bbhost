// Device memory by allocation site (see gpu_internal.h). This file includes
// the Vulkan header directly, without the macros, so the real calls are made.
#include <vulkan/vulkan.h>

#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "host/gpu_memtrack.h"

namespace {
struct Site {
    std::uint64_t bytes = 0;
    std::uint64_t count = 0;
    std::uint64_t calls = 0, us = 0;  // every allocation made here, and the driver's time for them
};
struct Block {
    std::uint64_t bytes = 0;
    std::uint32_t type = 0;
    bool limited = false;  // counted against BBHOST_TEST_VRAM_MB
    bool local = false;    // counted in g_local_bytes
    int heap_class = 2;    // bb_memory_heap_class of its type when it was made
    std::string site;
    const char* owner = nullptr;  // MemOwner's name when it was made (nullptr: none, the site stands for it)
};
std::mutex g_mu;
std::unordered_map<std::uint64_t, Block> g_blocks;  // by the handle's value
std::map<std::string, Site> g_sites;
std::map<std::uint32_t, std::uint64_t> g_by_type;
// BBHOST_TEST_VRAM_MB (bb_memory_test_limit): the memory types of the
// device-local heap, and what they may hold in all before an allocation in
// them fails as a full card's does. 0: no limit.
std::uint32_t g_limit_types = 0;
std::uint64_t g_limit_bytes = 0, g_limited_bytes = 0;
// The memory types of the device-local heap (bb_memory_set_local_types) and
// what our allocations hold in them: the video memory this process has taken
// itself, and the report's sites outside it marked "@bar" (another
// device-local heap) or "@host", so the report no longer adds host and device
// memory together.
std::uint32_t g_local_types = 0;
std::uint64_t g_local_bytes = 0, g_local_peak = 0;
// The other device-local heaps' types (a BAR window, an integrated GPU's
// second heap): bb_memory_set_heap_classes.
std::uint32_t g_other_device_types = 0;
// What our allocations hold by owner (MemOwner's name, or the site without
// one) and heap class, and the most the device-local classes have held.
struct OwnerSum {
    std::uint64_t bytes[3] = {};
    std::uint64_t count = 0;
    std::uint64_t peak = 0;  // the most it held in classes 0 and 1 together
};
std::map<std::string, OwnerSum> g_owners;
std::uint64_t g_device_bytes = 0, g_device_peak = 0;  // classes 0 and 1

thread_local const char* t_owner = nullptr;

std::string site_name(const char* file, int line) {
    const char* base = file;
    for (const char* p = file; *p; ++p) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s:%d", base, line);
    return buf;
}

int heap_class_locked(std::uint32_t type) {
    if ((g_local_types >> type) & 1) return 0;
    if ((g_other_device_types >> type) & 1) return 1;
    return 2;
}

const std::string& owner_key(const Block& b) {
    static thread_local std::string key;
    key = b.owner ? std::string(b.owner) : b.site;
    return key;
}
}  // namespace

MemOwner::MemOwner(const char* owner) : prev_(t_owner) { t_owner = owner; }
MemOwner::~MemOwner() { t_owner = prev_; }

void bb_memory_test_limit(std::uint32_t types, std::uint64_t bytes) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_limit_types = types;
    g_limit_bytes = bytes;
}

void bb_memory_set_local_types(std::uint32_t types) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_local_types = types;
}

void bb_memory_set_heap_classes(std::uint32_t other_device_types) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_other_device_types = other_device_types & ~g_local_types;
}

std::uint64_t bb_memory_local_bytes() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_local_bytes;
}

bool bb_memory_test_limited(std::uint64_t* limit, std::uint64_t* used) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (limit) *limit = g_limit_bytes;
    if (used) *used = g_limited_bytes;
    return g_limit_bytes != 0;
}

VkResult bb_alloc_memory(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* cb,
                         VkDeviceMemory* out, const char* file, int line) {
    const bool limited = g_limit_bytes && ((g_limit_types >> info->memoryTypeIndex) & 1);
    if (limited) {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_limited_bytes + info->allocationSize > g_limit_bytes) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        g_limited_bytes += info->allocationSize;  // held for it now, so two threads cannot both pass
    }
    const auto t0 = std::chrono::steady_clock::now();
    const VkResult r = vkAllocateMemory(device, info, cb, out);
    const auto us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    if (limited && !(r == VK_SUCCESS && out && *out != VK_NULL_HANDLE)) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_limited_bytes -= info->allocationSize;
    }
    if (r == VK_SUCCESS && out && *out != VK_NULL_HANDLE) {
        std::lock_guard<std::mutex> lk(g_mu);
        Block b;
        b.bytes = info->allocationSize;
        b.type = info->memoryTypeIndex;
        b.limited = limited;
        b.site = site_name(file, line);
        b.owner = t_owner;
        b.heap_class = heap_class_locked(b.type);
        if (g_local_types) {
            b.local = ((g_local_types >> b.type) & 1) != 0;
            if (b.local) {
                g_local_bytes += b.bytes;
                g_local_peak = std::max(g_local_peak, g_local_bytes);
            } else {
                // Another device-local heap (a BAR window: video memory the
                // CPU writes) or host memory.
                b.site += b.heap_class == 1 ? "@bar" : "@host";
            }
        }
        Site& s = g_sites[b.site];
        s.bytes += b.bytes;
        ++s.count;
        ++s.calls;
        s.us += us;
        // A slow one is a frame's worth when it lands in play (host-visible
        // memory is pinned page by page): say where and when.
        if (us >= 4000) {
            static int said = 0;
            if (said++ < 64) {
                host_log("gpu: a %llu MiB device allocation at %s%s%s took %.1f ms",
                         static_cast<unsigned long long>(info->allocationSize >> 20), b.site.c_str(), b.owner ? ", " : "",
                         b.owner ? b.owner : "", us / 1000.0);
            }
        }
        OwnerSum& o = g_owners[owner_key(b)];
        o.bytes[b.heap_class] += b.bytes;
        ++o.count;
        if (b.heap_class < 2) {
            g_device_bytes += b.bytes;
            g_device_peak = std::max(g_device_peak, g_device_bytes);
            o.peak = std::max(o.peak, o.bytes[0] + o.bytes[1]);
        }
        g_by_type[b.type] += b.bytes;
        g_blocks[reinterpret_cast<std::uint64_t>(*out)] = std::move(b);
    }
    return r;
}

void bb_free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* cb) {
    if (memory != VK_NULL_HANDLE) {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_blocks.find(reinterpret_cast<std::uint64_t>(memory));
        if (it != g_blocks.end()) {
            Site& s = g_sites[it->second.site];
            s.bytes -= it->second.bytes;
            --s.count;
            OwnerSum& o = g_owners[owner_key(it->second)];
            o.bytes[it->second.heap_class] -= it->second.bytes;
            --o.count;
            if (it->second.heap_class < 2) g_device_bytes -= it->second.bytes;
            g_by_type[it->second.type] -= it->second.bytes;
            if (it->second.limited) g_limited_bytes -= it->second.bytes;
            if (it->second.local) g_local_bytes -= it->second.bytes;
            g_blocks.erase(it);
        }
    }
    vkFreeMemory(device, memory, cb);
}

std::vector<MemOwnerTotal> bb_memory_owner_totals() {
    std::lock_guard<std::mutex> lk(g_mu);
    std::vector<MemOwnerTotal> v;
    v.reserve(g_owners.size());
    for (const auto& [name, o] : g_owners) {
        if (!o.count && !o.peak) continue;
        MemOwnerTotal t;
        t.name = name;
        for (int c = 0; c < 3; ++c) t.bytes[c] = o.bytes[c];
        t.count = o.count;
        t.peak = o.peak;
        v.push_back(std::move(t));
    }
    std::sort(v.begin(), v.end(), [](const MemOwnerTotal& a, const MemOwnerTotal& b) {
        const std::uint64_t da = a.bytes[0] + a.bytes[1], db = b.bytes[0] + b.bytes[1];
        return da != db ? da > db : a.bytes[2] > b.bytes[2];
    });
    return v;
}

void bb_memory_peaks(std::uint64_t* local_peak, std::uint64_t* device_peak) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (local_peak) *local_peak = g_local_peak;
    if (device_peak) *device_peak = g_device_peak;
}

std::string bb_memory_sites_report() {
    std::lock_guard<std::mutex> lk(g_mu);
    std::vector<std::pair<std::string, Site>> v(g_sites.begin(), g_sites.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
    std::string out = "gpu: memory by site (MiB, blocks live; allocations made, driver ms):";
    char buf[128];
    std::uint64_t total = 0;
    for (const auto& [t, bytes] : g_by_type) total += bytes;
    std::snprintf(buf, sizeof(buf), " total %llu;", static_cast<unsigned long long>(total >> 20));
    out += buf;
    if (g_local_types) {
        std::snprintf(buf, sizeof(buf), " video memory %llu;", static_cast<unsigned long long>(g_local_bytes >> 20));
        out += buf;
    }
    if (g_limit_bytes) {
        std::snprintf(buf, sizeof(buf), " BBHOST_TEST_VRAM_MB %llu/%llu;", static_cast<unsigned long long>(g_limited_bytes >> 20),
                      static_cast<unsigned long long>(g_limit_bytes >> 20));
        out += buf;
    }
    int n = 0;
    for (const auto& [name, s] : v) {
        if ((s.bytes < (1ull << 20) && s.us < 1000) || n++ >= 16) continue;
        std::snprintf(buf, sizeof(buf), " %s %llu/%llu (%llu, %.1f);", name.c_str(), static_cast<unsigned long long>(s.bytes >> 20),
                      static_cast<unsigned long long>(s.count), static_cast<unsigned long long>(s.calls), s.us / 1000.0);
        out += buf;
    }
    return out;
}
