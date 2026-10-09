#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#include <asm/prctl.h>
#include <sys/prctl.h>
#ifndef ARCH_SET_FS
#define ARCH_SET_FS 0x1002
#define ARCH_GET_FS 0x1003
#endif
#ifndef ARCH_SET_GS
#define ARCH_SET_GS 0x1001
#endif
#endif

// PT_TLS memsz on this eboot. TCB sits at FS:0; .tbss is at negative offsets.
constexpr std::size_t kGuestTls = 0x750;
constexpr std::int64_t kTlsHeapOff = -0x748;
constexpr std::uint64_t kTcbMagic = 0x4242544c53475354ull;

// Guest uses FS:0 as self. glibc stack_chk reads FS:0x28 if a host call is
// ever made with guest FS still installed, so keep a copy of the host canary.
struct GuestTcb {
    GuestTcb* self;
    void* host_fs;
    std::uint64_t magic;
    std::uint32_t multiple_threads;
    std::uint32_t gscope_flag;
    std::uint64_t sysinfo;
    std::uint64_t stack_guard;
    // The thread's gsync::Thread, for the HLE entries the guest calls without
    // the thunk (hle/pthread.cpp), which cannot reach host TLS: sync_thread
    // once gsync::current_thread() has run on the thread, pthread_self once
    // pthread's self_thread() has.
    void* sync_thread;
    void* pthread_self;
};

static_assert(offsetof(GuestTcb, stack_guard) == 0x28, "stack_chk at fs:0x28");
static_assert(offsetof(GuestTcb, magic) == 0x10 && offsetof(GuestTcb, sync_thread) == 0x30 &&
                  offsetof(GuestTcb, pthread_self) == 0x38,
              "the raw entries read fs:0x10, fs:0x30 and fs:0x38");

inline thread_local void* t_guest_fs = nullptr;
