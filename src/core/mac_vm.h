// The guest's address space on macOS.
//
// The eboot wants fixed addresses (hle/kernel.cpp): its direct memory at
// 4 GiB (up to 0x2f0000000), the GPU alias of it at 0xff00000000, the sync
// arena at 0xe000000000, host mappings the GPU reaches from 0xc000000000.
// Rosetta 2, which runs bbhost on Apple Silicon, maps its own regions (its
// runtime, a 128 MiB JIT area) at the first free address at or above 4 GiB
// before main runs - in a plain executable, on the direct memory window - and
// mapping over them hangs the process. A larger __PAGEZERO does not help: the
// kernel then refuses those addresses to everyone.
//
// So the low windows are part of bbhost's own image: a zero-fill section at
// the end of its data (core/mac_reserve.cpp), bbhost linked low and fixed
// (0x10000, no ASLR; CMakeLists.txt) so that the section covers 4 GiB to
// past the eboot's window. Rosetta and the dylibs land above it. At start
// (hle_kernel_reserve_guest_windows) the section becomes placeholders -
// PROT_NONE anonymous memory with a tag of ours - and so do the high windows,
// which nothing has taken that early. A mapping in a window replaces
// placeholders with MAP_FIXED, where Linux has MAP_FIXED_NOREPLACE take free
// space, and an unmapping puts placeholders back, so nothing of the host's
// ever lands in a window. The kernel's map, read with mach_vm_region, says
// which pages are still placeholders: no bookkeeping beside it.
//
// The eboot's image goes to 0x300000000, where Windows puts it when 0x400000
// is taken (core/memory.cpp): bbhost's own image, from 0x10000, covers
// 0x400000 here. Stubs that must be within a rel32 of the game's code come
// from the windows' unused space (mac_vm_map_near).
//
// The price of the low, fixed image: a guest access through a near-null
// pointer (0x10000 up to the end of bbhost's data, some tens of MiB) reads
// bbhost's image, and a write past its code lands in its data, where the
// console would fault.
//
// Nothing here exists on Linux or Windows.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__APPLE__)

// The eboot's image window (kernel.cpp's kImageFallbackBase, 256 MiB as on
// Windows) and the high windows, from the host mappings' lowest hint up to
// the end of the GPU alias window at 1 TiB, the GPU's 40-bit reach.
constexpr std::uint64_t kMacVmImageBase = 0x300000000ull;
constexpr std::uint64_t kMacVmImageSize = 0x10000000ull;
constexpr std::uint64_t kMacVmHighBase = 0xC000000000ull;
constexpr std::uint64_t kMacVmHighEnd = 0x10000000000ull;

// The zero-fill section of bbhost's image that holds the low windows
// (core/mac_reserve.cpp; only bbhost has one).
void mac_vm_image_reservation(std::uint64_t* lo, std::uint64_t* hi);

// Takes [lo, hi), the image's reservation, as a window: placeholders from
// here on. False (and a log line) when it cannot be remapped.
bool mac_vm_adopt(std::uint64_t lo, std::uint64_t hi);

// Reserves [base, base+len) as a window of placeholders. Call before
// anything else maps. False (and a log line) when any of it is taken.
bool mac_vm_reserve(std::uint64_t base, std::uint64_t len);

enum class MacVmPlace { Anywhere, Exact, NoReplace };

// mmap that keeps the windows. Exact: MAP_FIXED at `at`, which replaces
// whatever is there, the host's own memory too: the caller owns the range
// (in a window, mac_vm_in_window, or a mapping of its own). NoReplace: at `at` only when every page there is a placeholder, or
// every page is free (then as a hint, which the kernel takes when it can);
// fails otherwise. Anywhere: `at` as a hint. `flags` as for mmap, without
// MAP_FIXED. Returns the address or nullptr.
void* mac_vm_map(void* at, std::uint64_t len, int prot, int flags, int fd, std::uint64_t offset, MacVmPlace place);

// Every byte of [at, at+len) lies in one window.
bool mac_vm_in_window(std::uint64_t at, std::uint64_t len);

// Unmaps [at, at+len): what lies in a window becomes placeholders again,
// the rest is unmapped.
bool mac_vm_unmap(void* at, std::uint64_t len);

// `len` bytes of anonymous memory with `prot`, taken from a window's
// placeholders where a rel32 anywhere in [lo, hi) reaches all of them and
// back: above hi first, nearest first, then below lo. nullptr when there is
// no such room (or no window).
void* mac_vm_map_near(std::uint64_t lo, std::uint64_t hi, std::uint64_t len, int prot);

#endif
