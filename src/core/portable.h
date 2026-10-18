// The few host-OS calls the port makes outside the platform layers, behind
// one name each, so a source file is the same on Linux and Windows. Nothing
// here touches guest state.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// mkdir -p for one level; true when the directory exists afterwards.
bool host_mkdir(const char* path);
// The names in a directory (no . and ..), unsorted; false when it cannot be read.
bool host_list_dir(const char* path, std::vector<std::string>* names);
// This process's CPU time, user plus system, every thread, in milliseconds.
std::uint64_t host_process_cpu_ms();
// The calling thread's CPU time, user plus system, in nanoseconds
// (CLOCK_THREAD_CPUTIME_ID; on Windows GetThreadTimes, which moves in the
// scheduler's ticks - sum it over many frames, not one).
std::uint64_t host_thread_cpu_ns();
// Copies len bytes from addr, failing (false) instead of faulting when the
// page is not mapped: on Linux a plain copy whose fault the SIGSEGV handler
// hands back (host_read_safe_recover), on Windows ReadProcessMemory. Not from
// a signal handler: a fault there, with SIGSEGV blocked, ends the process.
bool host_read_safe(const void* addr, void* out, std::size_t len);
// For the SIGSEGV handler (Linux): a fault inside host_read_safe's copy
// returns to it (false); anything else returns here.
void host_read_safe_recover();
// The calling thread's own stack, low and high address, from the thread
// library. False when the platform will not say. A read inside it cannot
// fault, so it needs no system call to be safe: the GX layer hands out
// pointers to structures that live there, and reading one through
// host_read_safe cost 8.6 million system calls and 25 seconds in a three-
// minute soak (hle/gx_trace.cpp, safe_read).
bool host_thread_stack(std::uint64_t* lo, std::uint64_t* hi);
// The calling thread's OS id (gettid / GetCurrentThreadId), for logs and keys.
// macOS: pthread_threadid_np, which fits 32 bits in practice.
std::uint32_t host_thread_id();
// Names the calling thread for the tools that list threads: pthread_setname_np
// (comm, 15 characters) on Linux; on Windows winpthreads' copy and
// SetThreadDescription, which the sampler and debuggers read. macOS: 63 chars.
void host_thread_set_name(const char* name);
// The largest width and height among the desktop's displays, in physical
// pixels (each display's current mode). False when the platform does not say
// (only Windows answers today).
bool host_desktop_max_size(int* w, int* h);
// The name Linux can hold (15 characters): leading dotted parts dropped
// first ("Core.Res.PostProcessor_0" -> "PostProcessor_0"), then the tail.
void host_thread_fit_name(const char* name, char (&out)[16]);
// Where this executable is loaded: code addresses minus this are what
// addr2line takes (the link addresses, PIE's start at 0). macOS: for atos -l.
std::uintptr_t host_image_base();
#if defined(__APPLE__)
// macOS: this executable's path, symbolic links resolved (_NSGetExecutablePath
// and realpath), or empty - where Linux reads /proc/self/exe.
std::string host_exe_path();
// macOS: the loaded image (this executable or a library) one of whose
// segments holds addr - an executable segment when code_only - with its path
// and where its Mach-O header is loaded (addr minus that is what atos -l
// takes with the header's link address). dyld's image list, for what Linux
// asks dl_iterate_phdr; false for any address outside them, the guest's
// included. (bbhost's data segment holds the guest's low windows,
// core/mac_vm.h: for an address there only a code lookup says false.)
bool host_image_of(std::uint64_t addr, bool code_only, const char** name, std::uint64_t* base);
// macOS: [addr, addr+len) is mapped readable. Fault- and lock-free, so the
// fault handler may ask; Rosetta 2's stacks can't be told by address.
bool host_readable(std::uint64_t addr, std::size_t len);
#endif
// Windows: the threads host_thread_set_name named, with their CPU time
// (kernel + user, 100 ns units), for frame statistics. A registry, not a
// Toolhelp snapshot: that walks every thread on the system, tens of
// milliseconds on the laptop, and frame statistics run on the game's render
// thread once a second. Threads that have exited are dropped. Empty on
// Linux, where /proc lists them.
struct HostThreadTime {
    std::uint32_t tid;
    std::uint64_t ticks;
    char name[32];
};
std::vector<HostThreadTime> host_thread_times();
// Sleeps for about `usec` microseconds: usleep on Linux; on Windows a
// high-resolution waitable timer (Sleep's granularity is the 15.6 ms system
// tick without timeBeginPeriod, and the guest's polling loops sleep 100 us);
// macOS: a kqueue timer, as usleep wakes up to a quarter of the sleep late.
void host_sleep_us(std::uint64_t usec);
// Sleeps until `t` on the steady clock through host_sleep_us. Use these, not
// std::this_thread::sleep_for/sleep_until, for anything under a millisecond:
// the Windows build's libstdc++ sleeps through winpthreads' nanosleep, which
// truncates to whole milliseconds, so a 500 us sleep_for returns at once and
// a loop around it spins a core (the audio pacing loop did, 2026-10-04).
void host_sleep_until(std::chrono::steady_clock::time_point t);
// memmem: the first occurrence of needle in hay, or nullptr.
const void* host_memmem(const void* hay, std::size_t hay_len, const void* needle, std::size_t needle_len);

// The guest heap. One family on each platform: Windows' aligned allocations
// must be freed by their own free, so every guest block goes through
// _aligned_* there, and through the C library here. Zeroed where the caller
// asks (memalign and the nothrow new promise it).
void* host_guest_malloc(std::size_t n);
void* host_guest_calloc(std::size_t n, std::size_t size);
void* host_guest_realloc(void* p, std::size_t n);
void* host_guest_memalign(std::size_t align, std::size_t n);  // zeroed
void host_guest_free(void* p);

// Pages for generated code: RWX on request (the thunk stubs), RW otherwise.
void* host_page_alloc(std::size_t size, bool exec);
void host_page_free(void* p, std::size_t size);
// Makes [p, p+size) inaccessible: a guard below a stack.
void host_page_guard(void* p, std::size_t size);
