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

namespace {
struct Site {
    std::uint64_t bytes = 0;
    std::uint64_t count = 0;
    std::uint64_t calls = 0, us = 0;  // every allocation made here, and the driver's time for them
    std::uint64_t addressed = 0;      // of `count`, those made for buffers with device addresses
};
struct Block {
    std::uint64_t bytes = 0;
    std::uint32_t type = 0;
    bool limited = false;  // counted against BBHOST_TEST_VRAM_MB
    bool local = false;    // counted in g_local_bytes
    bool addressed = false;  // VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    std::string site;
};
std::mutex g_mu;
std::unordered_map<std::uint64_t, Block> g_blocks;  // by the handle's value
std::map<std::string, Site> g_sites;
std::map<std::uint32_t, std::uint64_t> g_by_type;
std::uint64_t g_addressed = 0;  // live blocks with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT

bool for_device_address(const VkMemoryAllocateInfo* info) {
    for (auto* p = static_cast<const VkBaseInStructure*>(info->pNext); p; p = p->pNext) {
        if (p->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO)
            return (reinterpret_cast<const VkMemoryAllocateFlagsInfo*>(p)->flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) != 0;
    }
    return false;
}
// BBHOST_TEST_VRAM_MB (bb_memory_test_limit): the memory types of the
// device-local heap, and what they may hold in all before an allocation in
// them fails as a full card's does. 0: no limit.
std::uint32_t g_limit_types = 0;
std::uint64_t g_limit_bytes = 0, g_limited_bytes = 0;
// The memory types of the device-local heap (bb_memory_set_local_types) and
// what our allocations hold in them: the video memory this process has taken
// itself, and the report's sites outside it marked "@host", so the report no
// longer adds host and device memory together.
std::uint32_t g_local_types = 0;
std::uint64_t g_local_bytes = 0;

std::string site_name(const char* file, int line) {
    const char* base = file;
    for (const char* p = file; *p; ++p) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s:%d", base, line);
    return buf;
}
}  // namespace

void bb_memory_test_limit(std::uint32_t types, std::uint64_t bytes) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_limit_types = types;
    g_limit_bytes = bytes;
}

void bb_memory_set_local_types(std::uint32_t types) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_local_types = types;
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
        b.addressed = for_device_address(info);
        b.site = site_name(file, line);
        if (g_local_types) {
            b.local = ((g_local_types >> b.type) & 1) != 0;
            if (b.local) g_local_bytes += b.bytes;
            else b.site += "@host";
        }
        Site& s = g_sites[b.site];
        s.bytes += b.bytes;
        ++s.count;
        ++s.calls;
        s.us += us;
        s.addressed += b.addressed;
        g_addressed += b.addressed;
        // A slow one is a frame's worth when it lands in play (host-visible
        // memory is pinned page by page): say where and when.
        if (us >= 4000) {
            static int said = 0;
            if (said++ < 64) {
                host_log("gpu: a %llu MiB device allocation at %s took %.1f ms",
                         static_cast<unsigned long long>(info->allocationSize >> 20), b.site.c_str(), us / 1000.0);
            }
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
            s.addressed -= it->second.addressed;
            g_addressed -= it->second.addressed;
            g_by_type[it->second.type] -= it->second.bytes;
            if (it->second.limited) g_limited_bytes -= it->second.bytes;
            if (it->second.local) g_local_bytes -= it->second.bytes;
            g_blocks.erase(it);
        }
    }
    vkFreeMemory(device, memory, cb);
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
    std::snprintf(buf, sizeof(buf), " for device addresses %llu blocks", static_cast<unsigned long long>(g_addressed));
    out += buf;
    return out;
}

std::uint64_t bb_memory_address_blocks() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_addressed;
}

std::string bb_memory_address_report() {
    std::lock_guard<std::mutex> lk(g_mu);
    std::vector<std::pair<std::string, std::uint64_t>> v;
    for (const auto& [name, s] : g_sites) {
        if (s.addressed) v.emplace_back(name, s.addressed);
    }
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(g_addressed));
    std::string out = buf;
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::snprintf(buf, sizeof(buf), "%s%s %llu", i ? ", " : " (by allocation site: ", v[i].first.c_str(),
                      static_cast<unsigned long long>(v[i].second));
        out += buf;
    }
    if (!v.empty()) out += ')';
    return out;
}
