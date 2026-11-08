// The guest's fixed address windows on Linux.
//
// The eboot's direct memory lives at fixed addresses (hle/kernel.cpp: the CPU
// window at 4 GiB, the GPU alias window at 0xff00000000), mapped there with
// MAP_FIXED_NOREPLACE, so whatever the process mapped there first makes the
// game's mapping fail.
// NVIDIA's driver reserves 4 GiB at 8 GiB once Vulkan and NGX are up: a heap
// sized for a 2560x1440 display reached past it, failed to map, and the game
// crashed in its allocator. So the windows are reserved before anything else
// runs (main), PROT_NONE and MAP_NORESERVE - no memory and no commit. A
// mapping in a window takes its part of the reservation (MAP_FIXED: the range
// is ours to replace), and an unmapping there puts the reservation back
// rather than leave a hole something else could map into. A mapping asked
// not to replace still fails where an earlier mapping made here is, as
// MAP_FIXED_NOREPLACE would (the windows' mapped parts are recorded).
// Outside the windows everything is the plain call.
//
// The Windows counterpart is core/win_vm.h (placeholders).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if !defined(_WIN32)

// Reserves [base, base + len) as a window, all of it or nothing: false (and
// errno EEXIST) when any of it is mapped already.
bool linux_vm_reserve(std::uint64_t base, std::uint64_t len);

// Reserves what nothing is mapped in of [base, base + len), each free piece a
// window of its own, for when the whole range cannot be had. Returns the
// bytes reserved.
std::uint64_t linux_vm_reserve_free(std::uint64_t base, std::uint64_t len);

// The process's mappings (/proc/self/maps lines) that overlap
// [base, base + len), for the log.
std::vector<std::string> linux_vm_occupants(std::uint64_t base, std::uint64_t len);

// Whether all of [p, p + len) lies inside one window.
bool linux_vm_owns(std::uint64_t p, std::uint64_t len);

enum class LinuxVmFix { None, Exact, NoReplace };

// mmap(at, len, prot, flags, fd, off), `flags` without MAP_FIXED*: None
// takes `at` as a hint (the kernel places it, never in a window); Exact
// replaces what is there (MAP_FIXED); NoReplace fails with EEXIST where
// something is mapped already (MAP_FIXED_NOREPLACE) - inside a window, where
// an earlier mapping made here is: the reservation itself does not count.
// Returns what mmap returns.
void* linux_vm_mmap(void* at, std::size_t len, int prot, int flags, int fd, std::int64_t off, LinuxVmFix fix);

// munmap(at, len), except that the parts of the range inside a window go
// back to the reservation. Returns 0, or -1 when a part failed.
int linux_vm_munmap(void* at, std::size_t len);

// The bytes the windows hold, and of them the bytes mapped now.
std::uint64_t linux_vm_reserved_bytes();
std::uint64_t linux_vm_mapped_bytes();

#endif
