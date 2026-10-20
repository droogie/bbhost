// The guest's address space on macOS.
//
// The eboot wants fixed addresses from 4 GiB up (hle/kernel.cpp). Rosetta 2
// maps its own regions at the first free address >= 4 GiB before main, and
// mapping over them hangs the process; a larger __PAGEZERO makes the kernel
// refuse those addresses to everyone. So the low windows are a zero-fill
// section of bbhost's own image (core/mac_reserve.cpp), linked low and fixed.
//
// At start the windows become placeholders (tagged PROT_NONE memory). A
// mapping in a window replaces them with MAP_FIXED and an unmapping puts them
// back, so no host mapping lands in a window; mach_vm_region is the only
// record of which pages are free.
//
// Cost of the low image: a guest near-null access hits bbhost's image instead
// of faulting.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__APPLE__)

// The image window is kernel.cpp's kImageFallbackBase (0x400000 is inside
// bbhost here); the high ones end at 1 TiB, the GPU's 40-bit reach.
constexpr std::uint64_t kMacVmImageBase = 0x300000000ull;
constexpr std::uint64_t kMacVmImageSize = 0x10000000ull;
constexpr std::uint64_t kMacVmHighBase = 0xC000000000ull;
constexpr std::uint64_t kMacVmHighEnd = 0x10000000000ull;

// core/mac_reserve.cpp's section; only bbhost links it.
void mac_vm_image_reservation(std::uint64_t* lo, std::uint64_t* hi);

// Makes the image's reservation a window. False (logged) on failure.
bool mac_vm_adopt(std::uint64_t lo, std::uint64_t hi);

// Call before anything else maps. False (logged) when any of it is taken.
bool mac_vm_reserve(std::uint64_t base, std::uint64_t len);

enum class MacVmPlace { Anywhere, Exact, NoReplace };

// mmap that keeps the windows. Exact: MAP_FIXED, replacing anything, so the
// caller must own the range. NoReplace: only over placeholders or free space.
// Anywhere: `at` is a hint. `flags` without MAP_FIXED. nullptr on failure.
void* mac_vm_map(void* at, std::uint64_t len, int prot, int flags, int fd, std::uint64_t offset, MacVmPlace place);

// Every byte of [at, at+len) lies in one window.
bool mac_vm_in_window(std::uint64_t at, std::uint64_t len);

// What lies in a window becomes placeholders again.
bool mac_vm_unmap(void* at, std::uint64_t len);

// Anonymous memory from a window's placeholders within rel32 reach of all of
// [lo, hi), both ways; nullptr when there is no such room.
void* mac_vm_map_near(std::uint64_t lo, std::uint64_t hi, std::uint64_t len, int prot);

#endif
