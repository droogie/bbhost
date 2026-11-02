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
std::uint32_t host_thread_id();
// Names the calling thread for the tools that list threads: pthread_setname_np
// (comm, 15 characters) on Linux; on Windows winpthreads' copy and
// SetThreadDescription, which the sampler and debuggers read.
void host_thread_set_name(const char* name);
// Puts the calling thread at the lowest priority there is (idle on Windows,
// nice 19 on Linux), for good: work that must never take a core the game wants.
void host_thread_background();
// The largest width and height among the desktop's displays, in physical
// pixels (each display's current mode). False when the platform does not say:
// a headless run (host_set_headless), or a build without SDL off Windows.
bool host_desktop_max_size(int* w, int* h);
// The refresh rate of the display the window opens on (the primary one), as
// the display gives it (59.94, 143.998), asked before the window shell starts:
// the Uncapped frame rate's (engine/frame_rate.cpp). 0 when nothing says: a
// headless run, or a build without SDL off Windows.
float host_desktop_refresh_hz();
// Whether this run opens no window (main, once the configuration is final):
// host_desktop_max_size and host_desktop_refresh_hz then ask no display server.
void host_set_headless(bool headless);
// The name Linux can hold (15 characters): leading dotted parts dropped
// first ("Core.Res.PostProcessor_0" -> "PostProcessor_0"), then the tail.
void host_thread_fit_name(const char* name, char (&out)[16]);
// Where this executable is loaded: code addresses minus this are what
// addr2line takes (the link addresses, PIE's start at 0).
std::uintptr_t host_image_base();
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
// tick without timeBeginPeriod, and the guest's polling loops sleep 100 us).
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
