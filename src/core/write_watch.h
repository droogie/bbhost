#pragma once

// Write tracking for guest memory the host keeps a copy of: the texture
// cache's surfaces that no GX texture holds (a GX texture's changes come
// through GX itself: host/textures.cpp, by_identity) and the buffer shadow's
// vertex and index pages. A surface's pages are made read-only when it is uploaded,
// the first write to one of them faults, and the fault makes that page
// writable again and records that it was written. The next bind of the
// surface sees the written page and uploads it again - on the next frame,
// not whenever a hash of its memory is next due.
//
// The hashes it replaces could only poll, and polling cost what it read, so
// a surface that had sat unchanged was rechecked every 2 s: a glyph drawn
// into an idle menu atlas took up to that long to appear. The shadPS4 texture
// cache works this way too (its page manager write-protects the pages behind
// cached images and a write fault marks them dirty).
//
// Each 4 KiB host page has one entry: 0 when no surface was ever armed on it,
// otherwise the sequence number of its last write (or of the last change to
// its mapping). A watch records the sequence number it was armed at and is
// dirty when any of its pages carries a later one. Every step is ordered so
// that no write goes unseen:
//
//   arm:    load the change count, load the sequence, then protect - and only
//           then does the caller read the memory it uploads;
//   fault:  make the page writable, take a sequence number, store it in the
//           entry, then bump the change count;
//   check:  load the change count - unchanged since the last check means no
//           page anywhere was written - else scan the watch's pages.
//
// Only direct memory is armed (hle_kernel_write_watch): Vulkan imports it
// through its own mirror mapping, so the guest's view can be protected without
// touching an import. Writes that do not go through the watched view are not
// seen: GX and the command processor write through the GPU alias. The buffer
// shadow arms the alias as well (hle_kernel_write_watch_alias); the texture
// cache keeps a slow hash as the backstop instead.
//
// BBHOST_WRITE_WATCH=0 turns it off. On Windows the protection is
// VirtualProtect and the fault a vectored exception handler
// (write_watch_install, first in line); the section views the guest memory
// lives in take page protections like any private mapping, and Vulkan reads
// the memory through its own mirror view, which stays writable.

#include <cstddef>
#include <cstdint>
#include <cstdio>

bool write_watch_enabled();

// Who arms a watch: the pages it protects are tagged with it, so the faults
// and releases they take are counted by owner (write_watch_stats).
enum : std::uint8_t { kWatchOwnerTexture = 1, kWatchOwnerMirror = 2 };

struct WriteWatch {
    std::uint32_t armed = 0;  // the sequence at arming; 0 when not armed
    std::uint64_t seen = 0;   // the change count at the last check
    std::uint8_t owner = 0;   // kWatchOwner*, set by the owner before arming
};

// Protects [va, va+len)'s pages read-only. Called by hle_kernel_write_watch,
// under the kernel's map lock, once it has checked the range is direct memory
// mapped CPU read/write. False (and `w` unarmed) when it could not.
bool write_watch_arm_locked(std::uint64_t va, std::size_t len, WriteWatch& w);

// The number of pages of the range written (or remapped) since `w` was armed;
// 0 when none were. Cheap when nothing anywhere was written since the last
// clean answer. Asking again gives the same answer until `w` is re-armed.
std::uint32_t write_watch_dirty(std::uint64_t va, std::size_t len, WriteWatch& w);

// The SIGSEGV handler's first question: whether a fault at `addr` was a write
// to a watched page. If so the page is writable again, the write recorded,
// and the faulting instruction can simply be retried. Touches no lock and no
// thread-local state: it runs on guest threads, whose FS is the guest's.
// `frame`: the faulting pc and up to seven return addresses (the top of the
// stack and [rbp+8]) the handler read, for the writer census; may be null.
bool write_watch_on_fault(std::uint64_t addr, bool is_write, const std::uint64_t* frame = nullptr);
// Return addresses recorded with faults into [va, va+len) (a ring of the
// last 16384 faults, eight addresses each): who writes a watched surface.
// Returns how many distinct ones were written to `pcs`.
int write_watch_recent_faults(std::uint64_t va, std::size_t len, std::uint64_t* pcs, int max);
constexpr int kWriteWatchFrame = 8;
// The faults into [va, va+len) still in the ring, with the earliest and
// latest fault sequence numbers among them (0 when none).
int write_watch_recent_fault_span(std::uint64_t va, std::size_t len, std::uint64_t* first_seq, std::uint64_t* last_seq);
std::uint64_t write_watch_fault_seq();  // the count of faults so far
// The writers behind the faults still in the ring, most first: the first
// guest address of each fault's frame (the writing instruction or, for a
// write in a host leaf, its guest caller) and the guest address after it.
// Returns how many distinct pairs were written to pcs/callers/counts.
int write_watch_fault_writers(std::uint64_t* pcs, std::uint64_t* callers, std::uint32_t* counts, int max);

#if defined(_WIN32)
// Installs the vectored exception handler that asks write_watch_on_fault.
// Called once from main before guest code runs. False when the watch is off
// or the handler could not be added (then no surface is armed).
bool write_watch_install();
#endif

// Before the host has the *kernel* write guest memory (read(2) into a guest
// buffer): a protected page makes the syscall fail with EFAULT instead of
// faulting, so the range's watched pages are made writable and counted as
// written first. True when any were. Also before a copy that will write the
// whole range anyway (a destroyed resource's memory, the game's memcpy):
// one call instead of a fault a page. Safe in a decomp leaf - no lock, no
// thread-local, no stack protector.
bool write_watch_release(void* p, std::size_t n);

// fread(3) into guest memory, released first.
std::size_t write_watch_fread(void* p, std::size_t size, std::size_t n, std::FILE* f);

// The mapping under [va, va+len) changed (mapped over, unmapped, protected):
// every watched page in it counts as written.
void write_watch_forget(std::uint64_t va, std::size_t len);

// Diagnostics: the watch's arming, and each page's entry, as text.
void write_watch_describe(std::uint64_t va, std::size_t len, const WriteWatch& w, char* out, std::size_t cap);

struct WriteWatchStats {
    std::uint64_t arms = 0, faults = 0, releases = 0, forgets = 0, refused = 0;
    std::uint64_t ahead_pages = 0;  // pages a fault released past its own (a streaming writer's)
    // By the owners that armed the page ([kWatchOwnerTexture], [kWatchOwnerMirror],
    // [3] both, [0] none recorded): faults, pages a release made writable, arms.
    std::uint64_t faults_by_owner[4] = {}, released_pages_by_owner[4] = {}, arms_by_owner[4] = {};
    // Time the fault handler spent in its own work (the page's mprotect
    // included, the kernel's signal delivery not), and arming's mprotects, ns.
    std::uint64_t fault_handler_ns = 0, arm_ns = 0;
};
WriteWatchStats write_watch_stats();
