// Phase 0.1: the guest->host thunk must forward register args, stack args,
// varargs, vector args, preserve return registers, nest, and switch FS.
#include "core/asm_sym.h"
#include "core/thunk.h"
#include "core/tls_rewrite.h"
#include "guest_abi.h"
#include "hle/guest_fs.h"
#include "hle/sysv_va.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <pthread.h>

namespace {

int g_fail = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                  \
        }                                                              \
    } while (0)

GUEST_ABI std::int64_t nine(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d,
                            std::int64_t e, std::int64_t f, std::int64_t g, std::int64_t h,
                            std::int64_t i) {
    return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h + 9 * i;
}

}  // namespace

// The variadic entries take the guest va_list shape the HLE uses (6.2a):
// an asm shim over a body that walks it.
extern "C" GUEST_ABI int sum_var_body(int n, SysvVaList* ap) {
    int s = 0;
    for (int i = 0; i < n; ++i) {
        s += static_cast<int>(sysv_va_gp(ap));
    }
    return s;
}
extern "C" GUEST_ABI double mix_var_body(int n, SysvVaList* ap) {
    double s = 0;
    for (int i = 0; i < n; ++i) {
        s += sysv_va_fp(ap);
    }
    return s;
}
SYSV_VA_ENTRY(sum_var, sum_var_body, 1, rsi)
SYSV_VA_ENTRY(mix_var, mix_var_body, 1, rsi)

namespace {

GUEST_ABI double fma3(double a, double b, int c) { return a * b + c; }

struct Pair {
    std::int64_t lo, hi;
};
GUEST_ABI Pair pair_ret(std::int64_t a, std::int64_t b) { return Pair{a * 3, b * 5}; }

using GuestCb = std::int64_t(GUEST_ABI*)(std::int64_t, std::int64_t);
void* g_inner_thunk = nullptr;

GUEST_ABI std::int64_t eight(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d,
                             std::int64_t e, std::int64_t f, std::int64_t g, std::int64_t h) {
    return a + b + c + d + e + f + g + h;
}

// "Guest" callback: calls back into HLE (thunked) with eight args.
GUEST_ABI std::int64_t guest_cb(std::int64_t x, std::int64_t y) {
    using Fn8 = std::int64_t(GUEST_ABI*)(std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                                          std::int64_t, std::int64_t, std::int64_t, std::int64_t);
    return reinterpret_cast<Fn8>(g_inner_thunk)(x, y, 1, 2, 3, 4, 5, 6);
}

// HLE that calls a guest callback (nesting HLE -> guest -> HLE).
GUEST_ABI std::int64_t outer(void* cb, std::int64_t x, std::int64_t y) {
    return hle_call_guest<std::int64_t>(cb, x, y) + 100;
}

// The TCB as the guest's rewritten sites read it: gs:[disp] in GS mode
// (always on Windows, BBHOST_TLS_GS=1 here), fs:0 otherwise.
std::uint64_t fs0_magic() {
    GuestTcb* tcb = nullptr;
    if (tls_gs_mode()) {
        asm volatile("mov %%gs:(%1), %0" : "=r"(tcb) : "r"(static_cast<std::uint64_t>(tls_gs_disp())));
    } else {
#if defined(_WIN32)
        return 0;
#else
        asm volatile("mov %%fs:0, %0" : "=r"(tcb));
#endif
    }
    return tcb ? tcb->magic : 0;
}

std::uint64_t g_magic_in_hle = 1;
std::uint64_t g_magic_in_cb = 0;

GUEST_ABI std::int64_t fs_probe_cb(std::int64_t, std::int64_t) {
    g_magic_in_cb = fs0_magic();
    return 7;
}
GUEST_ABI std::int64_t fs_probe(void* cb) {
    g_magic_in_hle = fs0_magic();
    return hle_call_guest<std::int64_t>(cb, 1, 2);
}

void* fs_thread(void*) {
    using Fn = std::int64_t(GUEST_ABI*)(void*);
    auto* fn = reinterpret_cast<Fn>(thunk_wrap(reinterpret_cast<void*>(&fs_probe)));
    void* tcb = guest_thread_enter();
    // Guest FS installed: no libc from here until leave.
    std::uint64_t before = fs0_magic();
    std::int64_t r = fn(reinterpret_cast<void*>(&fs_probe_cb));
    std::uint64_t after = fs0_magic();
    guest_thread_leave(tcb);
    CHECK(before == kTcbMagic);
    CHECK(after == kTcbMagic);
    // FS mode switches the segment for host code; GS mode leaves it, so the
    // HLE body sees the TCB there too.
    if (tls_gs_mode()) {
        CHECK(g_magic_in_hle == kTcbMagic);
    } else {
        CHECK(g_magic_in_hle != kTcbMagic);
    }
    CHECK(g_magic_in_cb == kTcbMagic);
    CHECK(r == 7);
    return nullptr;
}


// setjmp/longjmp across a thunked HLE call and a guest callback.
alignas(16) std::uint64_t g_jb[12];
GUEST_ABI std::int64_t jumper(std::int64_t, std::int64_t) {
    hle_longjmp_raw(g_jb, 42);
}
GUEST_ABI std::int64_t call_jumper(void* cb) {
    hle_call_guest<std::int64_t>(cb, 0, 0);
    return -1;  // never
}

// thunk_wrap_capture_frame(): the DL_PANIC path. The loader patches the
// guest function's first byte, so the stub runs before anything touches the
// stack and the handler sees the guest's own rsp (pointing at the return
// address the call pushed) and rbp (the calling frame).
std::uint64_t g_cap_rsp = 0;
std::uint64_t g_cap_rbp = 0;
GUEST_ABI void capture_probe(std::int64_t a, std::int64_t b, std::int64_t c, std::uint64_t rsp,
                             std::uint64_t rbp) {
    CHECK(a == 1);
    CHECK(b == 2);
    CHECK(c == 3);
    g_cap_rsp = rsp;
    g_cap_rbp = rbp;
}

}  // namespace

// Two frames shaped like the eboot's: frame pointers, and a call through the
// patched entry. The return addresses are recorded so the test can name them.
extern "C" {
void* g_cap_stub = nullptr;
void* g_cap_inner_ret = nullptr;
void* g_cap_outer_ret = nullptr;
void cap_outer();
}
asm(".text\n"
    "    .globl " ASM_SYM(cap_inner) "\n"
    ASM_TYPE_FUNCTION(cap_inner)
    ASM_SYM(cap_inner) ":\n"
    "    push %rbp\n"
    "    mov %rsp, %rbp\n"
    "    mov $1, %edi\n"
    "    mov $2, %esi\n"
    "    mov $3, %edx\n"
    "    call *" ASM_SYM(g_cap_stub) "(%rip)\n"
    "1:  lea 1b(%rip), %rax\n"
    "    mov %rax, " ASM_SYM(g_cap_inner_ret) "(%rip)\n"
    "    pop %rbp\n"
    "    ret\n"
    ASM_SIZE(cap_inner)

    "    .globl " ASM_SYM(cap_outer) "\n"
    ASM_TYPE_FUNCTION(cap_outer)
    ASM_SYM(cap_outer) ":\n"
    "    push %rbp\n"
    "    mov %rsp, %rbp\n"
    "    call " ASM_SYM(cap_inner) "\n"
    "2:  lea 2b(%rip), %rax\n"
    "    mov %rax, " ASM_SYM(g_cap_outer_ret) "(%rip)\n"
    "    pop %rbp\n"
    "    ret\n"
    ASM_SIZE(cap_outer));

namespace {
void check_capture_frame() {
    g_cap_stub = thunk_wrap_capture_frame(reinterpret_cast<void*>(&capture_probe));
    CHECK(g_cap_stub != nullptr);
    cap_outer();
    CHECK(g_cap_rsp != 0);
    CHECK(g_cap_rbp != 0);
    if (g_cap_rsp && g_cap_rbp) {
        // [rsp] is the return into cap_inner; [rbp+8] the return into cap_outer.
        CHECK(*reinterpret_cast<void**>(g_cap_rsp) == g_cap_inner_ret);
        CHECK(reinterpret_cast<void**>(g_cap_rbp)[1] == g_cap_outer_ret);
        // rbp is cap_inner's frame, so it sits just above the pushed return.
        CHECK(g_cap_rbp > g_cap_rsp);
    }
}
}  // namespace

int main() {
    {
        int v = hle_setjmp_raw(g_jb);
        if (v == 0) {
            hle_longjmp_raw(g_jb, 5);
        }
        CHECK(v == 5);
        volatile int marker = 7;
        int w = hle_setjmp_raw(g_jb);
        if (w == 0) {
            using Fn = std::int64_t(GUEST_ABI*)(void*);
            auto* fn = reinterpret_cast<Fn>(thunk_wrap(reinterpret_cast<void*>(&call_jumper)));
            fn(reinterpret_cast<void*>(&jumper));
            CHECK(false);
        }
        CHECK(w == 42);
        CHECK(marker == 7);
    }
    using Fn9 = decltype(&nine);
    auto* n9 = reinterpret_cast<Fn9>(thunk_wrap(reinterpret_cast<void*>(&nine)));
    CHECK(n9(1, 2, 3, 4, 5, 6, 7, 8, 9) == nine(1, 2, 3, 4, 5, 6, 7, 8, 9));

    using FnV = int(GUEST_ABI*)(int, ...);
    auto* sv = reinterpret_cast<FnV>(thunk_wrap(reinterpret_cast<void*>(&sum_var)));
    CHECK(sv(12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) == 78);

    using FnMV = double(GUEST_ABI*)(int, ...);
    auto* mv = reinterpret_cast<FnMV>(thunk_wrap(reinterpret_cast<void*>(&mix_var)));
    CHECK(mv(10, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0) == 55.0);

    using FnD = decltype(&fma3);
    auto* fd = reinterpret_cast<FnD>(thunk_wrap(reinterpret_cast<void*>(&fma3)));
    CHECK(fd(2.5, 4.0, 3) == 13.0);

    using FnP = decltype(&pair_ret);
    auto* fp = reinterpret_cast<FnP>(thunk_wrap(reinterpret_cast<void*>(&pair_ret)));
    Pair pr = fp(2, 3);
    CHECK(pr.lo == 6 && pr.hi == 15);

    g_inner_thunk = thunk_wrap(reinterpret_cast<void*>(&eight));
    using FnO = decltype(&outer);
    auto* fo = reinterpret_cast<FnO>(thunk_wrap(reinterpret_cast<void*>(&outer)));
    CHECK(fo(reinterpret_cast<void*>(&guest_cb), 10, 20) == 10 + 20 + 21 + 100);

    check_capture_frame();

    pthread_t th;
    pthread_create(&th, nullptr, fs_thread, nullptr);
    pthread_join(th, nullptr);

    if (g_fail) {
        std::fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    std::puts("thunk_args ok");
    return 0;
}
