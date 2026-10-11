#include "core/thunk.h"
#include <unordered_set>

#include "core/tls_rewrite.h"

#include "hle/guest_fs.h"
#include "host/frame_stats.h"
#include "log.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <xmmintrin.h>

#include "core/portable.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <signal.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef HWCAP2_FSGSBASE
#define HWCAP2_FSGSBASE (1 << 1)
#endif
#endif

// Plain byte emission, the same on every platform; see the header.
std::uint8_t* thunk_emit_prologue_stub(std::uint8_t* c, std::uint64_t id, void* host, std::uint64_t resume,
                                      const std::uint8_t* displaced, std::size_t n, bool keep_rax) {
    // Below the six registers: a return slot (zeroed; what a skipped call
    // returns, saved[-1] to the host) and xmm0-7, which carry a hooked
    // function's float arguments and which host code would otherwise clobber.
    // 48 + 136 bytes over the entry's 8 keeps the host call 16-byte aligned.
    static constexpr std::uint8_t kSave[] = {
        0x57, 0x56, 0x52, 0x51, 0x41, 0x50, 0x41, 0x51,                    // push rdi rsi rdx rcx r8 r9
        0x48, 0x81, 0xec, 0x88, 0x00, 0x00, 0x00,                          // sub rsp, 136
        0xf3, 0x0f, 0x7f, 0x44, 0x24, 0x00,                                // movdqu [rsp], xmm0
        0xf3, 0x0f, 0x7f, 0x4c, 0x24, 0x10,                                // movdqu [rsp+16], xmm1
        0xf3, 0x0f, 0x7f, 0x54, 0x24, 0x20,                                // movdqu [rsp+32], xmm2
        0xf3, 0x0f, 0x7f, 0x5c, 0x24, 0x30,                                // movdqu [rsp+48], xmm3
        0xf3, 0x0f, 0x7f, 0x64, 0x24, 0x40,                                // movdqu [rsp+64], xmm4
        0xf3, 0x0f, 0x7f, 0x6c, 0x24, 0x50,                                // movdqu [rsp+80], xmm5
        0xf3, 0x0f, 0x7f, 0x74, 0x24, 0x60,                                // movdqu [rsp+96], xmm6
        0xf3, 0x0f, 0x7f, 0x7c, 0x24, 0x70,                                // movdqu [rsp+112], xmm7
        0x48, 0xc7, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00, 0, 0, 0, 0,        // mov qword [rsp+128], 0
        0x48, 0x8d, 0xb4, 0x24, 0x88, 0x00, 0x00, 0x00,                    // lea rsi, [rsp+136]
    };
    static constexpr std::uint8_t kRestore[] = {
        0x4c, 0x8b, 0x94, 0x24, 0x80, 0x00, 0x00, 0x00,                    // mov r10, [rsp+128]
        0xf3, 0x0f, 0x6f, 0x44, 0x24, 0x00,                                // movdqu xmm0, [rsp]
        0xf3, 0x0f, 0x6f, 0x4c, 0x24, 0x10,                                // movdqu xmm1, [rsp+16]
        0xf3, 0x0f, 0x6f, 0x54, 0x24, 0x20,                                // movdqu xmm2, [rsp+32]
        0xf3, 0x0f, 0x6f, 0x5c, 0x24, 0x30,                                // movdqu xmm3, [rsp+48]
        0xf3, 0x0f, 0x6f, 0x64, 0x24, 0x40,                                // movdqu xmm4, [rsp+64]
        0xf3, 0x0f, 0x6f, 0x6c, 0x24, 0x50,                                // movdqu xmm5, [rsp+80]
        0xf3, 0x0f, 0x6f, 0x74, 0x24, 0x60,                                // movdqu xmm6, [rsp+96]
        0xf3, 0x0f, 0x6f, 0x7c, 0x24, 0x70,                                // movdqu xmm7, [rsp+112]
        0x48, 0x81, 0xc4, 0x88, 0x00, 0x00, 0x00,                          // add rsp, 136
        0x41, 0x59, 0x41, 0x58, 0x59, 0x5a, 0x5e, 0x5f,                    // pop r9 r8 rcx rdx rsi rdi
    };
    // The same, with rax pushed where the alignment pad was: seven pushes are
    // 56 bytes, which aligns the call as the pad did.
    static constexpr std::uint8_t kSaveRax[] = {
        0x57, 0x56, 0x52, 0x51, 0x41, 0x50, 0x41, 0x51,  // push rdi rsi rdx rcx r8 r9
        0x50,                                            // push rax
        0x48, 0x8d, 0x74, 0x24, 0x08,                    // lea rsi, [rsp+8]
    };
    static constexpr std::uint8_t kRestoreRax[] = {
        0x58,                                            // pop rax (after mov r11, rax)
        0x41, 0x59, 0x41, 0x58, 0x59, 0x5a, 0x5e, 0x5f,  // pop r9 r8 rcx rdx rsi rdi
    };
    if (keep_rax) {
        std::memcpy(c, kSaveRax, sizeof(kSaveRax));
        c += sizeof(kSaveRax);
    } else {
        std::memcpy(c, kSave, sizeof(kSave));
        c += sizeof(kSave);
    }
    *c++ = 0x48;
    *c++ = 0xbf;  // movabs rdi, id
    std::memcpy(c, &id, 8);
    c += 8;
    const std::uint64_t fn = reinterpret_cast<std::uint64_t>(host);
    *c++ = 0x48;
    *c++ = 0xb8;  // movabs rax, host
    std::memcpy(c, &fn, 8);
    c += 8;
    *c++ = 0xff;
    *c++ = 0xd0;  // call rax
    static constexpr std::uint8_t kKeep[] = {0x49, 0x89, 0xc3};  // mov r11, rax (the pops leave r11 alone)
    std::memcpy(c, kKeep, sizeof(kKeep));
    c += sizeof(kKeep);
    if (keep_rax) {
        std::memcpy(c, kRestoreRax, sizeof(kRestoreRax));
        c += sizeof(kRestoreRax);
    } else {
        std::memcpy(c, kRestore, sizeof(kRestore));
        c += sizeof(kRestore);
    }
    static constexpr std::uint8_t kReturn[] = {
        0x4d, 0x85, 0xdb,  // test r11, r11
        0x74, 0x04,        // jz +4: run the method
        0x4c, 0x89, 0xd0,  // mov rax, r10: the return slot
        0xc3,              // ret to the method's caller
    };
    static constexpr std::uint8_t kReturnRax[] = {
        0x4d, 0x85, 0xdb,  // test r11, r11
        0x74, 0x03,        // jz +3: run the method
        0x31, 0xc0,        // xor eax, eax
        0xc3,              // ret to the method's caller
    };
    if (keep_rax) {
        std::memcpy(c, kReturnRax, sizeof(kReturnRax));
        c += sizeof(kReturnRax);
    } else {
        std::memcpy(c, kReturn, sizeof(kReturn));
        c += sizeof(kReturn);
    }
    if (n) std::memcpy(c, displaced, n);  // none for a redirected tail call
    c += n;
    const std::uint8_t jmp[6] = {0xff, 0x25, 0, 0, 0, 0};  // jmp [rip+0]
    std::memcpy(c, jmp, 6);
    c += 6;
    std::memcpy(c, &resume, 8);
    return c + 8;
}


namespace {

#if !defined(_WIN32)
// Linux >= 5.9 exposes FSGSBASE to user mode; wrfsbase is a few cycles
// instead of a syscall per HLE call. Probed once from the aux vector.
const bool g_fsgsbase = (getauxval(AT_HWCAP2) & HWCAP2_FSGSBASE) != 0;

inline void set_fs(void* base) {
    if (g_fsgsbase) {
        asm volatile("wrfsbase %0" : : "r"(base) : "memory");
    } else {
        syscall(SYS_arch_prctl, ARCH_SET_FS, reinterpret_cast<unsigned long>(base));
    }
}

inline void* get_fs() {
    if (g_fsgsbase) {
        void* v;
        asm volatile("rdfsbase %0" : "=r"(v));
        return v;
    }
    unsigned long v = 0;
    syscall(SYS_arch_prctl, ARCH_GET_FS, &v);
    return reinterpret_cast<void*>(v);
}
#else
// Windows: the guest reads its TCB through GS (core/tls_rewrite.h), FS is
// never touched, and the slot is the process's own TLS slot.
constexpr bool g_fsgsbase = false;
inline void set_fs(void*) {}
inline void* get_fs() { return nullptr; }
#endif

// Host stack for HLE. Guest frames must not share a stack with host C++
// (host locals leaked into uninitialised guest locals once), and the guest
// boot stack is a raw mmap without glibc's guard handling. Allocated per
// thread on first HLE call; freed in guest_thread_leave.
constexpr std::size_t kHostStack = 1u << 20;
constexpr std::size_t kHostGuard = 0x1000;
thread_local std::uint8_t* t_host_stack = nullptr;
thread_local std::uint8_t* t_guest_teb = nullptr;  // GS mode: the TEB stand-in
#if defined(_WIN32)
// Windows dispatches an exception only while rsp lies inside the TEB's
// stack limits, so the switch to the host stack moves those limits with it
// (hle_host_stack_enter) and back on the outermost return
// (hle_host_stack_leave); raw longjmp restores the saved limits and depth
// when it bypasses that return path. The guest's own are kept here meanwhile.
thread_local void* t_guest_stack_base = nullptr;
thread_local void* t_guest_stack_limit = nullptr;
thread_local int t_host_stack_depth = 0;
#endif

std::uint8_t* host_stack_alloc() {
    void* m = host_page_alloc(kHostStack + kHostGuard, false);
    if (!m) {
        return nullptr;
    }
    auto* base = static_cast<std::uint8_t*>(m) + kHostGuard;
    host_page_guard(m, kHostGuard);  // the page below the stack faults instead of overwriting
    return base;
}

void host_stack_free() {
    if (t_host_stack) {
        host_page_free(t_host_stack - kHostGuard, kHostStack + kHostGuard);
        t_host_stack = nullptr;
    }
}

}  // namespace

extern "C" GUEST_ABI void hle_fs_host() {
    // GS mode: FS is glibc's (or Windows' own) throughout; nothing to switch.
    if (tls_gs_mode()) return;
#if !defined(_WIN32)
    GuestTcb* tcb = nullptr;
    asm volatile("mov %%fs:0, %0" : "=r"(tcb));
    if (tcb && tcb->self == tcb && tcb->magic == kTcbMagic && tcb->host_fs) {
        set_fs(tcb->host_fs);
    }
#endif
}

// BBHOST_HLE_COUNT=1: every HLE call counted by thread and function
// (hle_thunk_common passes its target), reported with the 300-flip report
// (hle_call_counts_report): what the game's threads spend their HLE calls on.
extern "C" bool g_hle_count_on;  // read by hle_thunk_common
bool g_hle_count_on = [] {
    const char* e = std::getenv("BBHOST_HLE_COUNT");
    return e && e[0] == '1';
}();
namespace {
struct CallTable {
    static constexpr int kSlots = 512;
    std::atomic<void*> fn[kSlots];
    std::atomic<std::uint64_t> count[kSlots];
    char name[16] = {};
};
std::mutex g_call_tables_mu;
std::vector<CallTable*> g_call_tables;
thread_local CallTable* t_calls = nullptr;
}  // namespace

// frame: the trampoline's rbp, on the guest stack - [0] the guest's rbp,
// [1] its return address. The guest keeps frame pointers, so two more of its
// callers follow from there.
const char* (*g_thunk_name_lookup)(void*) = nullptr;
std::uint64_t (*g_thunk_slide_lookup)() = nullptr;
void thunk_set_lookups(const char* (*name)(void*), std::uint64_t (*slide)()) {
    g_thunk_name_lookup = name;
    g_thunk_slide_lookup = slide;
}

extern "C" GUEST_ABI void hle_count_call(void* fn, const std::uint64_t* frame) {
    std::uint64_t chain[3] = {frame[1], 0, 0};
    const std::uint64_t* f = frame;
    for (int i = 1; i < 3; ++i) {
        const auto* up = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(f[0]));
        if (up <= f || reinterpret_cast<std::uintptr_t>(up) - reinterpret_cast<std::uintptr_t>(f) > (1u << 20) ||
            (reinterpret_cast<std::uintptr_t>(up) & 7)) {
            break;
        }
        chain[i] = up[1];
        f = up;
    }
    main_wait_note_site(chain);
    // BBHOST_HLE_FIRST=1 (with BBHOST_HLE_COUNT=1): every HLE function the
    // first time it is called, with the guest return address as an ELF VA.
    static const bool first_log = [] {
        const char* e = std::getenv("BBHOST_HLE_FIRST");
        return e && e[0] == '1';
    }();
    if (first_log) {
        static std::mutex mu;
        static std::unordered_set<void*> seen;
        std::lock_guard<std::mutex> lk(mu);
        if (seen.insert(fn).second) {
            const char* name = g_thunk_name_lookup ? g_thunk_name_lookup(fn) : nullptr;
            host_log("hle first: %s from 0x%llx", name ? name : "?",
                     static_cast<unsigned long long>(chain[0] - (g_thunk_slide_lookup ? g_thunk_slide_lookup() : 0)));
        }
    }
    if (!t_calls) {
        t_calls = new CallTable();  // kept for the report after the thread ends
#if !defined(_WIN32)
        pthread_getname_np(pthread_self(), t_calls->name, sizeof(t_calls->name));
#endif
        std::lock_guard<std::mutex> lk(g_call_tables_mu);
        g_call_tables.push_back(t_calls);
    }
    std::size_t i = (reinterpret_cast<std::uintptr_t>(fn) >> 4) % CallTable::kSlots;
    for (int probe = 0; probe < CallTable::kSlots; ++probe, i = (i + 1) % CallTable::kSlots) {
        void* have = t_calls->fn[i].load(std::memory_order_relaxed);
        if (have == fn || (!have && (t_calls->fn[i].store(fn, std::memory_order_relaxed), true))) {
            t_calls->count[i].store(t_calls->count[i].load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
            return;
        }
    }
}

std::vector<HleCallCount> hle_call_counts_take() {
    std::vector<HleCallCount> out;
    std::lock_guard<std::mutex> lk(g_call_tables_mu);
    for (CallTable* t : g_call_tables) {
        for (int i = 0; i < CallTable::kSlots; ++i) {
            void* fn = t->fn[i].load(std::memory_order_relaxed);
            if (!fn) continue;
            const std::uint64_t n = t->count[i].exchange(0, std::memory_order_relaxed);
            if (n) out.push_back({t->name, fn, n});
        }
    }
    return out;
}

extern "C" GUEST_ABI void hle_fs_guest() {
    if (t_guest_fs && !tls_gs_mode()) {
        set_fs(t_guest_fs);
    }
}

#if defined(_WIN32)
// The trampoline's return path: the outermost level restores the guest's
// stack limits, then the segment switch (none in GS mode).
extern "C" GUEST_ABI void hle_host_stack_leave() {
    if (t_host_stack_depth > 0 && --t_host_stack_depth == 0) {
        auto* tib = reinterpret_cast<NT_TIB*>(NtCurrentTeb());
        tib->StackBase = t_guest_stack_base;
        tib->StackLimit = t_guest_stack_limit;
    }
    hle_fs_guest();
}
#endif

// Called from the trampoline with host FS installed. cur is the trampoline's
// save area on whatever stack the guest was running on. Returns the 16-byte
// aligned stack pointer the HLE function should run on. If the guest was
// already on this thread's host stack (guest callback nested inside HLE),
// continue below the current frame instead of jumping to the top.
extern "C" GUEST_ABI void* hle_host_stack_enter(std::uint8_t* cur) {
    if (!t_host_stack) {
        t_host_stack = host_stack_alloc();
        if (!t_host_stack) {
            host_log("host stack mmap failed; running HLE on the guest stack");
            return reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(cur - 256) & ~15ull);
        }
    }
    if (cur >= t_host_stack && cur < t_host_stack + kHostStack) {
#if defined(_WIN32)
        ++t_host_stack_depth;
#endif
        return reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(cur - 256) & ~15ull);
    }
#if defined(_WIN32)
    auto* tib = reinterpret_cast<NT_TIB*>(NtCurrentTeb());
    t_guest_stack_base = tib->StackBase;
    t_guest_stack_limit = tib->StackLimit;
    tib->StackBase = t_host_stack + kHostStack;
    tib->StackLimit = t_host_stack;
    t_host_stack_depth = 1;
#endif
    return t_host_stack + kHostStack - 64;
}

// Shared trampoline. Entered with r11 = real function, on the guest stack,
// guest FS installed, SysV arguments in place.
//
// Its CFI keeps the frame on rbp, which stays on the guest stack while the
// HLE function runs on the host one, so an unwinder (host/sampler.cpp, a
// debugger) steps from the HLE function back to the guest's return address
// instead of reading the host stack as if it were the caller's.
//
// Save area layout (rbx), 16-byte aligned:
//   +0..+40  rdi rsi rdx rcx r8 r9      +48 rax (AL = vector count)
//   +56      r11 (real fn)              +64..+191 xmm0..xmm7
#if defined(_WIN32)
asm(R"(
    .text
    .globl hle_thunk_common
hle_thunk_common:
    .cfi_startproc
    push %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    mov %rsp, %rbp
    .cfi_def_cfa_register %rbp
    push %r12
    .cfi_offset %r12, -24
    push %rbx
    .cfi_offset %rbx, -32
    sub $192, %rsp
    mov %rdi, 0(%rsp)
    mov %rsi, 8(%rsp)
    mov %rdx, 16(%rsp)
    mov %rcx, 24(%rsp)
    mov %r8, 32(%rsp)
    mov %r9, 40(%rsp)
    mov %rax, 48(%rsp)
    mov %r11, 56(%rsp)
    movdqa %xmm0, 64(%rsp)
    movdqa %xmm1, 80(%rsp)
    movdqa %xmm2, 96(%rsp)
    movdqa %xmm3, 112(%rsp)
    movdqa %xmm4, 128(%rsp)
    movdqa %xmm5, 144(%rsp)
    movdqa %xmm6, 160(%rsp)
    movdqa %xmm7, 176(%rsp)
    mov %rsp, %rbx
    call hle_fs_host
    cmpb $0, g_hle_count_on(%rip)
    je 1f
    mov 56(%rbx), %rdi
    mov %rbp, %rsi
    call hle_count_call
1:
    mov %rbx, %rdi
    call hle_host_stack_enter
    mov %rax, %r12
    mov %r12, %rsp
    sub $128, %rsp
    # The caller's stack arguments (up to 16 qwords), in xmm8-15 - caller-
    # saved, so free here - rather than rep movsq, whose start-up cost every
    # HLE call paid (the GX workers make hundreds of thousands a second).
    vmovdqu 16(%rbp), %xmm8
    vmovdqu 32(%rbp), %xmm9
    vmovdqu 48(%rbp), %xmm10
    vmovdqu 64(%rbp), %xmm11
    vmovdqu 80(%rbp), %xmm12
    vmovdqu 96(%rbp), %xmm13
    vmovdqu 112(%rbp), %xmm14
    vmovdqu 128(%rbp), %xmm15
    vmovdqu %xmm8, 0(%rsp)
    vmovdqu %xmm9, 16(%rsp)
    vmovdqu %xmm10, 32(%rsp)
    vmovdqu %xmm11, 48(%rsp)
    vmovdqu %xmm12, 64(%rsp)
    vmovdqu %xmm13, 80(%rsp)
    vmovdqu %xmm14, 96(%rsp)
    vmovdqu %xmm15, 112(%rsp)
    mov 0(%rbx), %rdi
    mov 8(%rbx), %rsi
    mov 16(%rbx), %rdx
    mov 24(%rbx), %rcx
    mov 32(%rbx), %r8
    mov 40(%rbx), %r9
    mov 48(%rbx), %rax
    mov 56(%rbx), %r11
    movdqa 64(%rbx), %xmm0
    movdqa 80(%rbx), %xmm1
    movdqa 96(%rbx), %xmm2
    movdqa 112(%rbx), %xmm3
    movdqa 128(%rbx), %xmm4
    movdqa 144(%rbx), %xmm5
    movdqa 160(%rbx), %xmm6
    movdqa 176(%rbx), %xmm7
    call *%r11
    mov %rax, 0(%rbx)
    mov %rdx, 8(%rbx)
    movdqa %xmm0, 64(%rbx)
    movdqa %xmm1, 80(%rbx)
    call hle_host_stack_leave
    mov 0(%rbx), %rax
    mov 8(%rbx), %rdx
    movdqa 64(%rbx), %xmm0
    movdqa 80(%rbx), %xmm1
    lea 192(%rbx), %rsp
    pop %rbx
    pop %r12
    pop %rbp
    .cfi_def_cfa %rsp, 8
    ret
    .cfi_endproc

    .globl hle_call_guest6
hle_call_guest6:
    .cfi_startproc
    push %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    mov %rsp, %rbp
    .cfi_def_cfa_register %rbp
    push %rbx
    .cfi_offset %rbx, -24
    push %r12
    .cfi_offset %r12, -32
    mov %rdi, %rbx
    mov %rsi, %rdi
    mov %rdx, %rsi
    mov %rcx, %rdx
    mov %r8, %rcx
    mov %r9, %r8
    mov 16(%rbp), %r9
    push %r9
    push %r8
    push %rcx
    push %rdx
    push %rsi
    push %rdi
    call hle_fs_guest
    pop %rdi
    pop %rsi
    pop %rdx
    pop %rcx
    pop %r8
    pop %r9
    xor %eax, %eax
    call *%rbx
    push %rax
    push %rdx
    call hle_fs_host
    pop %rdx
    pop %rax
    pop %r12
    pop %rbx
    pop %rbp
    .cfi_def_cfa %rsp, 8
    ret
    .cfi_endproc
)");
#else
asm(R"(
    .text
    .globl hle_thunk_common
    .type hle_thunk_common,@function
hle_thunk_common:
    .cfi_startproc
    push %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    mov %rsp, %rbp
    .cfi_def_cfa_register %rbp
    push %r12
    .cfi_offset %r12, -24
    push %rbx
    .cfi_offset %rbx, -32
    sub $192, %rsp
    mov %rdi, 0(%rsp)
    mov %rsi, 8(%rsp)
    mov %rdx, 16(%rsp)
    mov %rcx, 24(%rsp)
    mov %r8, 32(%rsp)
    mov %r9, 40(%rsp)
    mov %rax, 48(%rsp)
    mov %r11, 56(%rsp)
    movdqa %xmm0, 64(%rsp)
    movdqa %xmm1, 80(%rsp)
    movdqa %xmm2, 96(%rsp)
    movdqa %xmm3, 112(%rsp)
    movdqa %xmm4, 128(%rsp)
    movdqa %xmm5, 144(%rsp)
    movdqa %xmm6, 160(%rsp)
    movdqa %xmm7, 176(%rsp)
    mov %rsp, %rbx
    call hle_fs_host
    cmpb $0, g_hle_count_on(%rip)
    je 1f
    mov 56(%rbx), %rdi
    mov %rbp, %rsi
    call hle_count_call
1:
    mov %rbx, %rdi
    call hle_host_stack_enter
    mov %rax, %r12
    mov %r12, %rsp
    sub $128, %rsp
    # The caller's stack arguments (up to 16 qwords), in xmm8-15 - caller-
    # saved, so free here - rather than rep movsq, whose start-up cost every
    # HLE call paid (the GX workers make hundreds of thousands a second).
    vmovdqu 16(%rbp), %xmm8
    vmovdqu 32(%rbp), %xmm9
    vmovdqu 48(%rbp), %xmm10
    vmovdqu 64(%rbp), %xmm11
    vmovdqu 80(%rbp), %xmm12
    vmovdqu 96(%rbp), %xmm13
    vmovdqu 112(%rbp), %xmm14
    vmovdqu 128(%rbp), %xmm15
    vmovdqu %xmm8, 0(%rsp)
    vmovdqu %xmm9, 16(%rsp)
    vmovdqu %xmm10, 32(%rsp)
    vmovdqu %xmm11, 48(%rsp)
    vmovdqu %xmm12, 64(%rsp)
    vmovdqu %xmm13, 80(%rsp)
    vmovdqu %xmm14, 96(%rsp)
    vmovdqu %xmm15, 112(%rsp)
    mov 0(%rbx), %rdi
    mov 8(%rbx), %rsi
    mov 16(%rbx), %rdx
    mov 24(%rbx), %rcx
    mov 32(%rbx), %r8
    mov 40(%rbx), %r9
    mov 48(%rbx), %rax
    mov 56(%rbx), %r11
    movdqa 64(%rbx), %xmm0
    movdqa 80(%rbx), %xmm1
    movdqa 96(%rbx), %xmm2
    movdqa 112(%rbx), %xmm3
    movdqa 128(%rbx), %xmm4
    movdqa 144(%rbx), %xmm5
    movdqa 160(%rbx), %xmm6
    movdqa 176(%rbx), %xmm7
    call *%r11
    mov %rax, 0(%rbx)
    mov %rdx, 8(%rbx)
    movdqa %xmm0, 64(%rbx)
    movdqa %xmm1, 80(%rbx)
    call hle_fs_guest
    mov 0(%rbx), %rax
    mov 8(%rbx), %rdx
    movdqa 64(%rbx), %xmm0
    movdqa 80(%rbx), %xmm1
    lea 192(%rbx), %rsp
    pop %rbx
    pop %r12
    pop %rbp
    .cfi_def_cfa %rsp, 8
    ret
    .cfi_endproc
    .size hle_thunk_common, .-hle_thunk_common

    .globl hle_call_guest6
    .type hle_call_guest6,@function
hle_call_guest6:
    .cfi_startproc
    push %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    mov %rsp, %rbp
    .cfi_def_cfa_register %rbp
    push %rbx
    .cfi_offset %rbx, -24
    push %r12
    .cfi_offset %r12, -32
    mov %rdi, %rbx
    mov %rsi, %rdi
    mov %rdx, %rsi
    mov %rcx, %rdx
    mov %r8, %rcx
    mov %r9, %r8
    mov 16(%rbp), %r9
    push %r9
    push %r8
    push %rcx
    push %rdx
    push %rsi
    push %rdi
    call hle_fs_guest
    pop %rdi
    pop %rsi
    pop %rdx
    pop %rcx
    pop %r8
    pop %r9
    xor %eax, %eax
    call *%rbx
    push %rax
    push %rdx
    call hle_fs_host
    pop %rdx
    pop %rax
    pop %r12
    pop %rbx
    pop %rbp
    .cfi_def_cfa %rsp, 8
    ret
    .cfi_endproc
    .size hle_call_guest6, .-hle_call_guest6
)");
#endif

extern "C" void hle_thunk_common();

// jmp_buf: [0] rbx [1] rbp [2] r12 [3] r13 [4] r14 [5] r15 [6] rsp-after-ret
// [7] return address [8] mxcsr (low 32) + x87 cw (next 16). The first 72
// bytes have the same layout on both platforms. Windows uses the remaining
// three qwords of FreeBSD's 96-byte buffer for TEB stack bounds and HLE depth.
#if defined(_WIN32)
extern "C" GUEST_ABI void hle_jmp_stack_save(std::uint64_t* buf) {
    auto* tib = reinterpret_cast<NT_TIB*>(NtCurrentTeb());
    buf[9] = reinterpret_cast<std::uint64_t>(tib->StackBase);
    buf[10] = reinterpret_cast<std::uint64_t>(tib->StackLimit);
    buf[11] = static_cast<std::uint64_t>(t_host_stack_depth);
}

extern "C" GUEST_ABI void hle_jmp_stack_restore(const std::uint64_t* buf) {
    auto* tib = reinterpret_cast<NT_TIB*>(NtCurrentTeb());
    tib->StackBase = reinterpret_cast<void*>(buf[9]);
    tib->StackLimit = reinterpret_cast<void*>(buf[10]);
    t_host_stack_depth = static_cast<int>(buf[11]);
    // At a nonzero depth the destination is still on the HLE stack; the
    // outer thunk's guest bounds remain in t_guest_stack_base/limit for its
    // eventual normal return. At depth zero the next enter saves them anew.
}

asm(R"(
    .text
    .globl hle_setjmp_raw
hle_setjmp_raw:
    mov %rbx, 0(%rdi)
    mov %rbp, 8(%rdi)
    mov %r12, 16(%rdi)
    mov %r13, 24(%rdi)
    mov %r14, 32(%rdi)
    mov %r15, 40(%rdi)
    lea 8(%rsp), %rax
    mov %rax, 48(%rdi)
    mov (%rsp), %rax
    mov %rax, 56(%rdi)
    stmxcsr 64(%rdi)
    fnstcw 68(%rdi)
    push %rdi
    call hle_jmp_stack_save
    pop %rdi
    xor %eax, %eax
    ret

    .globl hle_longjmp_raw
hle_longjmp_raw:
    # Keep both SysV arguments across the helper, with a 16-byte-aligned call.
    sub $24, %rsp
    mov %rdi, 0(%rsp)
    mov %esi, 8(%rsp)
    call hle_jmp_stack_restore
    mov 0(%rsp), %rdi
    mov 8(%rsp), %esi
    add $24, %rsp
    mov 0(%rdi), %rbx
    mov 8(%rdi), %rbp
    mov 16(%rdi), %r12
    mov 24(%rdi), %r13
    mov 32(%rdi), %r14
    mov 40(%rdi), %r15
    ldmxcsr 64(%rdi)
    fldcw 68(%rdi)
    mov %esi, %eax
    test %eax, %eax
    jnz 1f
    mov $1, %eax
1:
    mov 48(%rdi), %rsp
    jmp *56(%rdi)
)");
#else
asm(R"(
    .text
    .globl hle_setjmp_raw
    .type hle_setjmp_raw,@function
hle_setjmp_raw:
    mov %rbx, 0(%rdi)
    mov %rbp, 8(%rdi)
    mov %r12, 16(%rdi)
    mov %r13, 24(%rdi)
    mov %r14, 32(%rdi)
    mov %r15, 40(%rdi)
    lea 8(%rsp), %rax
    mov %rax, 48(%rdi)
    mov (%rsp), %rax
    mov %rax, 56(%rdi)
    stmxcsr 64(%rdi)
    fnstcw 68(%rdi)
    xor %eax, %eax
    ret
    .size hle_setjmp_raw, .-hle_setjmp_raw

    .globl hle_longjmp_raw
    .type hle_longjmp_raw,@function
hle_longjmp_raw:
    mov 0(%rdi), %rbx
    mov 8(%rdi), %rbp
    mov 16(%rdi), %r12
    mov 24(%rdi), %r13
    mov 32(%rdi), %r14
    mov 40(%rdi), %r15
    ldmxcsr 64(%rdi)
    fldcw 68(%rdi)
    mov %esi, %eax
    test %eax, %eax
    jnz 1f
    mov $1, %eax
1:
    mov 48(%rdi), %rsp
    jmp *56(%rdi)
    .size hle_longjmp_raw, .-hle_longjmp_raw
)");
#endif

// hle_call_guest_regs: GuestCallRegs (thunk.h) - four stack arguments
// copied below the call, the six integer and eight vector argument registers
// loaded, al = 8 (a variadic callee's count), the call made under guest FS,
// rax and xmm0 stored back.
#if defined(_WIN32)
asm(R"(
    .text
    .globl hle_call_guest_regs
hle_call_guest_regs:
    .cfi_startproc
    push %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    mov %rsp, %rbp
    .cfi_def_cfa_register %rbp
    push %rbx
    .cfi_offset %rbx, -24
    push %r12
    .cfi_offset %r12, -32
    mov %rdi, %rbx
    call hle_fs_guest
    sub $32, %rsp
    mov 120(%rbx), %r11
    mov %r11, 0(%rsp)
    mov 128(%rbx), %r11
    mov %r11, 8(%rsp)
    mov 136(%rbx), %r11
    mov %r11, 16(%rsp)
    mov 144(%rbx), %r11
    mov %r11, 24(%rsp)
    mov 8(%rbx), %rdi
    mov 16(%rbx), %rsi
    mov 24(%rbx), %rdx
    mov 32(%rbx), %rcx
    mov 40(%rbx), %r8
    mov 48(%rbx), %r9
    movq 56(%rbx), %xmm0
    movq 64(%rbx), %xmm1
    movq 72(%rbx), %xmm2
    movq 80(%rbx), %xmm3
    movq 88(%rbx), %xmm4
    movq 96(%rbx), %xmm5
    movq 104(%rbx), %xmm6
    movq 112(%rbx), %xmm7
    mov $8, %eax
    call *0(%rbx)
    add $32, %rsp
    mov %rax, 152(%rbx)
    movq %xmm0, 160(%rbx)
    call hle_fs_host
    pop %r12
    pop %rbx
    pop %rbp
    .cfi_def_cfa %rsp, 8
    ret
    .cfi_endproc
)");
#else
asm(R"(
    .text
    .globl hle_call_guest_regs
    .type hle_call_guest_regs,@function
hle_call_guest_regs:
    .cfi_startproc
    push %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    mov %rsp, %rbp
    .cfi_def_cfa_register %rbp
    push %rbx
    .cfi_offset %rbx, -24
    push %r12
    .cfi_offset %r12, -32
    mov %rdi, %rbx
    call hle_fs_guest
    sub $32, %rsp
    mov 120(%rbx), %r11
    mov %r11, 0(%rsp)
    mov 128(%rbx), %r11
    mov %r11, 8(%rsp)
    mov 136(%rbx), %r11
    mov %r11, 16(%rsp)
    mov 144(%rbx), %r11
    mov %r11, 24(%rsp)
    mov 8(%rbx), %rdi
    mov 16(%rbx), %rsi
    mov 24(%rbx), %rdx
    mov 32(%rbx), %rcx
    mov 40(%rbx), %r8
    mov 48(%rbx), %r9
    movq 56(%rbx), %xmm0
    movq 64(%rbx), %xmm1
    movq 72(%rbx), %xmm2
    movq 80(%rbx), %xmm3
    movq 88(%rbx), %xmm4
    movq 96(%rbx), %xmm5
    movq 104(%rbx), %xmm6
    movq 112(%rbx), %xmm7
    mov $8, %eax
    call *0(%rbx)
    add $32, %rsp
    mov %rax, 152(%rbx)
    movq %xmm0, 160(%rbx)
    call hle_fs_host
    pop %r12
    pop %rbx
    pop %rbp
    .cfi_def_cfa %rsp, 8
    ret
    .cfi_endproc
    .size hle_call_guest_regs, .-hle_call_guest_regs
)");
#endif

namespace {

std::mutex g_stub_mu;
std::uint8_t* g_stub_page = nullptr;
std::size_t g_stub_used = 0;
constexpr std::size_t g_stub_page_size = 0x40000;
constexpr std::size_t kStub = 32;  // 23 bytes used

}  // namespace

void* thunk_wrap(void* fn) {
    if (!fn) {
        return fn;
    }
    std::lock_guard<std::mutex> lock(g_stub_mu);
    if (!g_stub_page) {
        void* m = host_page_alloc(g_stub_page_size, true);
        if (!m) {
            host_log("thunk page alloc failed; HLE runs unthunked");
            return fn;
        }
        g_stub_page = static_cast<std::uint8_t*>(m);
    }
    if (g_stub_used + kStub > g_stub_page_size) {
        host_log("thunk page full; HLE runs unthunked");
        return fn;
    }
    std::uint8_t* p = g_stub_page + g_stub_used;
    g_stub_used += kStub;
    std::uint8_t* c = p;
    *c++ = 0x49;
    *c++ = 0xbb;  // movabs r11, fn
    std::uint64_t a = reinterpret_cast<std::uint64_t>(fn);
    std::memcpy(c, &a, 8);
    c += 8;
    *c++ = 0xe9;  // jmp rel32 hle_thunk_common
    std::int64_t rel = reinterpret_cast<std::int64_t>(reinterpret_cast<void*>(&hle_thunk_common)) -
                       reinterpret_cast<std::int64_t>(c + 4);
    if (rel > INT32_MAX || rel < INT32_MIN) {
        // Fall back to an absolute jump through rax (clobbered anyway before AL matters? No:
        // AL carries the vector count, so use r10 instead).
        c -= 1;
        *c++ = 0x49;
        *c++ = 0xba;  // movabs r10, common
        std::uint64_t t = reinterpret_cast<std::uint64_t>(reinterpret_cast<void*>(&hle_thunk_common));
        std::memcpy(c, &t, 8);
        c += 8;
        *c++ = 0x41;
        *c++ = 0xff;
        *c++ = 0xe2;  // jmp r10
        g_stub_used += 8;  // 23 bytes; keep 16-byte spacing plus slack
    } else {
        std::int32_t r32 = static_cast<std::int32_t>(rel);
        std::memcpy(c, &r32, 4);
    }
    return p;
}

// A thunk that also hands the HLE function the guest's stack and frame
// pointers, as SysV arguments 4 and 5. Only for entry points patched in
// place at a function's first byte: rsp still points at the return address
// the guest's call pushed, and rbp is still the caller's frame, so the
// handler can walk the guest's frames. fn takes at most three arguments of
// its own; hle_thunk_common saves and restores rcx and r8 around the call.
void* thunk_wrap_capture_frame(void* fn) {
    void* inner = thunk_wrap(fn);
    if (!inner || inner == fn) {
        return inner;
    }
    std::lock_guard<std::mutex> lock(g_stub_mu);
    if (g_stub_used + kStub > g_stub_page_size) {
        host_log("thunk page full; the frame capture is skipped");
        return inner;
    }
    std::uint8_t* p = g_stub_page + g_stub_used;
    g_stub_used += kStub;
    std::uint8_t* c = p;
    static const std::uint8_t kHead[] = {0x48, 0x89, 0xe1,   // mov %rsp, %rcx
                                         0x49, 0x89, 0xe8,   // mov %rbp, %r8
                                         0x49, 0xba};        // movabs r10, inner
    std::memcpy(c, kHead, sizeof(kHead));
    c += sizeof(kHead);
    const std::uint64_t a = reinterpret_cast<std::uint64_t>(inner);
    std::memcpy(c, &a, 8);
    c += 8;
    *c++ = 0x41;
    *c++ = 0xff;
    *c++ = 0xe2;  // jmp *%r10
    return p;
}

// The console's floating-point environment, the one every Orbis thread starts
// with (shadPS4's ORBIS_MXCSR and ORBIS_FPUCW): MXCSR 0x9fc0 - denormals flushed
// to zero and read as zero (FTZ, DAZ), round to nearest, every exception masked
// - and the x87 control word 0x37f (64-bit precision, nearest). The game is
// built and tested under it, and a host thread starts with 0x1f80 (FTZ and DAZ
// off): there an underflowing vector length survives the code's "== 0" test,
// rsqrtps or rcpps of it is inf, and inf * 0 is NaN - a hit list sorted on NaN
// distances runs off its array (sub_2a0c5f0, a Steam Deck crash) - and FMOD's
// reverb and filter tails decay through denormals, each operation on which
// takes x86 ~100x as long. Every guest context starts here: the game's threads,
// its main thread, the AV player's callback thread; host threads keep their
// own. BBHOST_GUEST_MXCSR=<hex> sets another (0x1f80: the host's, as before).
void guest_fp_enter() {
    static const unsigned mxcsr = [] {
        const char* e = std::getenv("BBHOST_GUEST_MXCSR");
        const unsigned v = e && *e ? static_cast<unsigned>(std::strtoul(e, nullptr, 16)) : 0x9fc0u;
        host_log("guest: MXCSR 0x%04x for guest threads%s", v, e && *e ? " (BBHOST_GUEST_MXCSR)" : " (the console's: FTZ, DAZ)");
        return v;
    }();
    _mm_setcsr(mxcsr);
    const std::uint16_t cw = 0x037f;
    asm volatile("fldcw %0" : : "m"(cw));
}

void* guest_thread_enter() {
    guest_fp_enter();
#if !defined(_WIN32)
    // Signal handlers must not run on a guest stack that may be exhausted.
    {
        stack_t ss{};
        ss.ss_size = 64 * 1024;
        ss.ss_sp = std::malloc(ss.ss_size);
        if (ss.ss_sp) {
            sigaltstack(&ss, nullptr);
        }
    }
#endif
    auto* block = new unsigned char[kGuestTls + sizeof(GuestTcb) + 32]();
    auto* tcb = reinterpret_cast<GuestTcb*>(block + kGuestTls);
    tcb->self = tcb;
    tcb->magic = kTcbMagic;
    tcb->host_fs = get_fs();
    std::uint64_t guard = 0;
#if !defined(_WIN32)
    asm volatile("mov %%fs:0x28, %0" : "=r"(guard));
#endif
    tcb->stack_guard = guard;
    // fs:-0x748 stays zero: _init constructs the Dantelion TLS heap there.
    static int logs = 0;
    if (logs < 6) {
        ++logs;
        host_log("guest tcb %p guard=0x%llx %s", static_cast<void*>(tcb), static_cast<unsigned long long>(guard),
#if defined(_WIN32)
                 "gs-slot=TlsAlloc");
#else
                 tls_gs_mode() ? (g_fsgsbase ? "gs-slot=wrgsbase" : "gs-slot=arch_prctl")
                               : (g_fsgsbase ? "fs-switch=wrfsbase" : "fs-switch=arch_prctl"));
#endif
    }
    t_guest_fs = tcb;
    if (tls_gs_mode()) {
#if defined(_WIN32)
        // The real TEB: the slot the rewritten sites read is this process's
        // TLS slot (core/tls_rewrite.cpp took it from TlsAlloc).
        TlsSetValue(tls_gs_slot(), tcb);
#else
        // The Windows shape on Linux: the rewritten sites read
        // gs:[TlsSlots+8n], so GS gets a TEB-sized block whose slot holds the
        // TCB; FS stays glibc's.
        auto* teb = static_cast<std::uint8_t*>(std::calloc(1, kTebStandInSize));
        std::memcpy(teb + tls_gs_disp(), &tcb, sizeof(tcb));
        t_guest_teb = teb;
        if (g_fsgsbase) {
            asm volatile("wrgsbase %0" : : "r"(teb) : "memory");
        } else {
            syscall(SYS_arch_prctl, ARCH_SET_GS, reinterpret_cast<unsigned long>(teb));
        }
#endif
        return tcb;
    }
    set_fs(tcb);
    return tcb;
}

void guest_thread_leave(void* tcb_v) {
    auto* tcb = static_cast<GuestTcb*>(tcb_v);
    if (tcb && tcb->host_fs && !tls_gs_mode()) {
        set_fs(tcb->host_fs);
    }
#if defined(_WIN32)
    if (tls_gs_mode()) TlsSetValue(tls_gs_slot(), nullptr);
#else
    if (t_guest_teb) {
        if (g_fsgsbase) {
            asm volatile("wrgsbase %0" : : "r"(static_cast<std::uint8_t*>(nullptr)) : "memory");
        } else {
            syscall(SYS_arch_prctl, ARCH_SET_GS, 0ul);
        }
        std::free(t_guest_teb);
        t_guest_teb = nullptr;
    }
#endif
    t_guest_fs = nullptr;
    host_stack_free();
    if (tcb) {
        delete[] (reinterpret_cast<unsigned char*>(tcb) - kGuestTls);
    }
}
