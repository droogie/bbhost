#pragma once

#include <cstddef>
#include <cstdint>

struct GuestMemory {
    void* base = nullptr;
    std::size_t size = 0;
    std::uint64_t slide = 0;
};

bool guest_alloc(GuestMemory* mem, std::uint64_t slide, std::size_t size);
void guest_free(GuestMemory* mem);
bool guest_protect_rx(GuestMemory* mem, std::uint64_t va, std::size_t size);
bool guest_protect_rw(GuestMemory* mem, std::uint64_t va, std::size_t size);
bool guest_protect_rwx(GuestMemory* mem, std::uint64_t va, std::size_t size);
// Keeps the page holding va read-write-execute for good: a later
// guest_protect_rx or guest_protect_rw over it leaves it so. For values in
// the image's read-only part written every frame (engine/frame_rate.cpp),
// whose pages other writers put back to read-only after their own writes
// (live resolution's plate sizes share one).
void guest_keep_writable(GuestMemory* mem, std::uint64_t va);
// The image's executable segment (core/elf.cpp, once it is sealed): a
// guest_protect_rw over any of it keeps execute there. The segment holds the
// read-only data too, and code is written into it - a patch manifest's cave
// over unused strings (the Restore Debug Dash patch's) - which a later write of a
// value on the same page (the debug menu's font paths, a plugin's write) left
// unexecutable: the game faulted when it next ran the cave.
void guest_mark_executable(GuestMemory* mem, std::uint64_t va, std::size_t size);

inline void* guest_ptr(const GuestMemory& mem, std::uint64_t va) {
    return static_cast<std::uint8_t*>(mem.base) + (va - mem.slide);
}
