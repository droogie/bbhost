// The guest's address space on Windows.
//
// The eboot wants its direct memory at fixed addresses (kernel.cpp: CPU
// window at 4 GiB, GPU alias window at 0xff00000000) and maps 16 KiB pages
// of a 6 GiB backing at 64 KiB-aligned offsets. Linux does that with
// MAP_FIXED over a memfd. Windows has no fixed-address replace: a view of a
// file mapping lands only where the space is free, and anything else may
// have taken the address first (under wine the process heap lands at
// exactly 4 GiB). So the windows are reserved as *placeholders* before
// anything else runs (VirtualAlloc2 + MEM_RESERVE_PLACEHOLDER), a mapping
// carves the placeholder to its exact range and replaces it with a view
// (MapViewOfFile3 + MEM_REPLACE_PLACEHOLDER), and an unmapping gives the
// placeholder back (UnmapViewOfFile2 + MEM_PRESERVE_PLACEHOLDER). Partial
// unmaps of a view unmap it whole and remap the kept parts from the same
// section, so the guest's munmap of part of a mapping behaves.
//
// Flexible (anonymous) guest memory is a private pagefile section per
// mapping, mapped the same way, so it too can be partially released.
// Everything is 64 KiB granular: Windows requires view offsets and bases on
// the allocation granularity, and every mapping the game makes is (the
// Linux smoke: 14 direct maps, all 64 KiB aligned and linear).
//
// Nothing here exists on Linux; kernel.cpp's Windows branches call it.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)

// True when VirtualAlloc2/MapViewOfFile3 are available (Windows 10 1803+,
// wine 7+). Without them the layer falls back to MapViewOfFileEx at the
// hint, which works only while nothing else took the address.
bool win_vm_available();

// Reserves [base, base+len) as a placeholder. Call before anything else
// allocates. False (and a log line) when the range is not free.
bool win_vm_reserve(std::uint64_t base, std::uint64_t len);

enum class WinVmPlace { Anywhere, Exact, NoReplace };

// Maps [offset, offset+len) of `section` (a file-mapping HANDLE) at `at`
// (Exact: unmapping our own views there first; NoReplace: the range must be
// a placeholder) or anywhere below `limit` (Anywhere, at ignored; the base a
// multiple of `align` where that is above the 64 KiB granularity) with the
// PAGE_* protection. Returns the base or nullptr.
void* win_vm_map(void* section, void* at, std::uint64_t len, std::uint64_t offset, unsigned long protect,
                 WinVmPlace place, std::uint64_t limit, std::uint64_t align = 0);

// Anonymous memory: a private section of `len`, mapped as above.
void* win_vm_alloc(void* at, std::uint64_t len, unsigned long protect, WinVmPlace place, std::uint64_t limit);

// Committed private memory at `at` (Exact: inside a placeholder) or, with
// at == nullptr, anywhere below `limit`; for the eboot image, which is
// never partially released, so no section behind it.
void* win_vm_commit(void* at, std::uint64_t len, unsigned long protect, std::uint64_t limit);

// Placeholder only (a guest reservation): the range becomes inaccessible
// and can later be mapped Exact. Returns the base or nullptr.
void* win_vm_place(void* at, std::uint64_t len, WinVmPlace place, std::uint64_t limit);

// Logs every view this layer holds, and flags two views of one section whose
// offsets overlap (memory visible at two addresses that should be distinct).
void win_vm_dump_views();

// Unmaps whatever of ours lies in [at, at+len), whole views or parts of
// them; the range becomes placeholder again.
bool win_vm_unmap(void* at, std::uint64_t len);

#endif
