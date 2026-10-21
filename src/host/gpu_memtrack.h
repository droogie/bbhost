#pragma once

// Who holds the device memory our allocations take (gpu_memtrack.cpp): every
// vkAllocateMemory in the files that include gpu_internal.h goes through
// bb_alloc_memory, which keeps its site (file:line) and, when one is set on
// the allocating thread, its owner - the report's name for what the memory is
// for ("untile scratch", "image heap blocks"). Allocations made in a helper
// that many callers share (create_dev_buffer) are told apart that way.

#include <cstdint>
#include <string>
#include <vector>

// The owner the allocations made on this thread inside its scope are counted
// under. `owner` must outlive the allocation (a string literal).
class MemOwner {
public:
    explicit MemOwner(const char* owner);
    ~MemOwner();
    MemOwner(const MemOwner&) = delete;
    MemOwner& operator=(const MemOwner&) = delete;

private:
    const char* prev_;
};

// What our allocations hold for one owner (or one site without an owner), by
// heap class: [0] the video memory heap (bb_memory_set_local_types), [1] other
// device-local heaps (a BAR window: video memory the CPU writes), [2] host
// memory (imports, staging). `peak`: the most it held in [0] and [1].
struct MemOwnerTotal {
    std::string name;
    std::uint64_t bytes[3] = {};
    std::uint64_t count = 0;
    std::uint64_t peak = 0;
};
std::vector<MemOwnerTotal> bb_memory_owner_totals();  // largest in video memory first
// The memory types of the device-local heaps other than the video memory
// heap's (a BAR window): what our allocations hold there is class 1.
void bb_memory_set_heap_classes(std::uint32_t other_device_types);
// The most our allocations have held in the video memory heap, and in every
// device-local heap together.
void bb_memory_peaks(std::uint64_t* local_peak, std::uint64_t* device_peak);
