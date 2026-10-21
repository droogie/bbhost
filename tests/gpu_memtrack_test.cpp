// The device memory tracker's owners and heap classes (host/gpu_memtrack.cpp),
// without a GPU: vkAllocateMemory and vkFreeMemory are this file's fakes, so
// the tracker's bookkeeping is all that runs.
#include <vulkan/vulkan.h>

#include "host/gpu_memtrack.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

VkResult bb_alloc_memory(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* cb, VkDeviceMemory* out,
                         const char* file, int line);
void bb_free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* cb);
void bb_memory_set_local_types(std::uint32_t types);
std::uint64_t bb_memory_local_bytes();
void bb_memory_test_limit(std::uint32_t types, std::uint64_t bytes);

namespace {
std::uint64_t g_next = 0x1000;
int g_failures = 0;
}  // namespace

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*,
                                                           VkDeviceMemory* out) {
    *out = reinterpret_cast<VkDeviceMemory>(g_next);
    g_next += 0x1000;
    return VK_SUCCESS;
}
extern "C" VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}

namespace {
constexpr std::uint64_t kMiB = 1ull << 20;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

VkDeviceMemory alloc(std::uint32_t type, std::uint64_t bytes, int line) {
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = bytes;
    mai.memoryTypeIndex = type;
    VkDeviceMemory m = VK_NULL_HANDLE;
    if (bb_alloc_memory(VK_NULL_HANDLE, &mai, nullptr, &m, "src/host/fake.cpp", line) != VK_SUCCESS) return VK_NULL_HANDLE;
    return m;
}

const MemOwnerTotal* find(const std::vector<MemOwnerTotal>& v, const std::string& name) {
    for (const MemOwnerTotal& t : v) {
        if (t.name == name) return &t;
    }
    return nullptr;
}
}  // namespace

int main() {
    // Type 0 is the video memory heap's, type 1 another device-local heap's
    // (a BAR window), type 2 host memory's.
    bb_memory_set_local_types(1u << 0);
    bb_memory_set_heap_classes((1u << 1) | (1u << 0));  // the video memory heap's own type is left out of class 1

    VkDeviceMemory a0, a1, a2, b0, s0;
    {
        MemOwner own("rings");
        a0 = alloc(0, 10 * kMiB, 10);
        a1 = alloc(1, 2 * kMiB, 11);
        {
            MemOwner inner("mirrors");
            b0 = alloc(0, 64 * kMiB, 12);
        }
        a2 = alloc(2, 5 * kMiB, 13);  // back to "rings" once the inner scope ends
    }
    s0 = alloc(0, 1 * kMiB, 42);  // no owner: its site stands for it

    std::vector<MemOwnerTotal> v = bb_memory_owner_totals();
    const MemOwnerTotal* rings = find(v, "rings");
    const MemOwnerTotal* mirrors = find(v, "mirrors");
    const MemOwnerTotal* site = find(v, "fake.cpp:42");
    check(rings && rings->bytes[0] == 10 * kMiB && rings->bytes[1] == 2 * kMiB && rings->bytes[2] == 5 * kMiB && rings->count == 3,
          "an owner's bytes by heap class");
    check(mirrors && mirrors->bytes[0] == 64 * kMiB && mirrors->count == 1, "a nested owner");
    check(site && site->bytes[0] == 1 * kMiB, "no owner: the site");
    check(!v.empty() && v[0].name == "mirrors", "largest in video memory first");
    check(bb_memory_local_bytes() == 75 * kMiB, "video memory bytes: classes 0 only");
    std::uint64_t local_peak = 0, device_peak = 0;
    bb_memory_peaks(&local_peak, &device_peak);
    check(local_peak == 75 * kMiB && device_peak == 77 * kMiB, "the peaks");

    // Freed: the owner's bytes go, its peak stays.
    bb_free_memory(VK_NULL_HANDLE, b0, nullptr);
    bb_free_memory(VK_NULL_HANDLE, a0, nullptr);
    v = bb_memory_owner_totals();
    mirrors = find(v, "mirrors");
    rings = find(v, "rings");
    check(!mirrors || (mirrors->bytes[0] == 0 && mirrors->peak == 64 * kMiB), "a freed owner keeps its peak");
    check(rings && rings->bytes[0] == 0 && rings->bytes[1] == 2 * kMiB && rings->peak == 12 * kMiB, "an owner after a free");
    check(bb_memory_local_bytes() == 1 * kMiB, "video memory bytes after the frees");
    bb_memory_peaks(&local_peak, &device_peak);
    check(local_peak == 75 * kMiB, "the peak after the frees");

    // BBHOST_TEST_VRAM_MB's limit still refuses past it, and counts nothing.
    bb_memory_test_limit(1u << 0, 4 * kMiB);
    check(alloc(0, 2 * kMiB, 50) != VK_NULL_HANDLE, "within the test limit");
    check(alloc(0, 3 * kMiB, 51) == VK_NULL_HANDLE, "past the test limit");
    check(bb_memory_local_bytes() == 3 * kMiB, "a refused allocation is not counted");

    bb_free_memory(VK_NULL_HANDLE, a1, nullptr);
    bb_free_memory(VK_NULL_HANDLE, a2, nullptr);
    bb_free_memory(VK_NULL_HANDLE, s0, nullptr);
    if (g_failures) return 1;
    std::printf("gpu_memtrack_test: ok\n");
    return 0;
}
