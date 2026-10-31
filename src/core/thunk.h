#pragma once

// Guest <-> host transition machinery (Linux x86-64).
//
// The eboot runs with its own FS base (GuestTcb, see hle/guest_fs.h). Host
// code needs glibc's FS. Every HLE entry the guest can call is wrapped by
// thunk_wrap(): a shared asm trampoline that switches FS to the host,
// moves to a per-thread host stack, forwards all six register arguments,
// the vector argument registers, AL, and 16 stack-passed qwords, calls the
// real function, then restores guest FS and the guest stack.
//
// Host code that needs to call back into guest code uses hle_call_guest():
// it installs guest FS for the duration of the call.

#include "guest_abi.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Wraps fn (a SysV function) so the guest can call it. Returns fn unchanged on
// platforms without the trampoline.
void* thunk_wrap(void* fn);

// thunk_wrap(), plus the guest's rsp and rbp as SysV arguments 4 and 5. Only
// valid for an entry point patched at a function's first byte, where rsp
// still points at the return address the call pushed. fn takes at most three
// arguments of its own.
void* thunk_wrap_capture_frame(void* fn);

// A prologue hook's stub, for engine code that must see a guest function's
// arguments as it is entered. It saves the six argument registers, calls
// host(id, saved) - host already thunk_wrap()ed, `saved` pointing at r9, r8,
// rcx, rdx, rsi, rdi in that order - and restores them. If host returned
// nonzero it returns 0 to the function's caller (host did the work);
// otherwise it runs the `n` displaced prologue bytes, which must not be
// rip-relative, and jumps to `resume`. Returns the end of what it wrote.
//
// `keep_rax` saves rax around the host call as well, for a hook on a variadic
// function: SysV passes the number of vector registers used in al, and the
// call to host otherwise leaves the host's return value there - which the
// displaced prologue then reads as that count. Nothing else needs it, because
// rax is call-clobbered everywhere else. `saved` still points at the six
// argument registers; writing to them is how a hook rewrites an argument.
// Without keep_rax the stub also keeps xmm0-7 (a float argument) across the
// host call - saved - 136 is xmm0, 16 bytes each, rewritable the same way -
// and saved[-1] is a slot, zero to start, that a skipped call returns.
//
// `site` is for a hook inside a function body rather than at its entry: there
// rsp is 16-byte aligned (an entry's is 8 past), so the stub pads it by 8
// first and gives the pad back before the displaced bytes. Such a hook's host
// must return 0 - the skip's ret would leave the function from its middle.
std::uint8_t* thunk_emit_prologue_stub(std::uint8_t* code, std::uint64_t id, void* host, std::uint64_t resume,
                                      const std::uint8_t* displaced, std::size_t n, bool keep_rax = false,
                                      bool site = false);

// Per-thread guest TLS block + TCB. Installs guest FS. Returns the TCB.
void* guest_thread_enter();
// Restores host FS and frees the TLS block.
void guest_thread_leave(void* tcb);

// Everything here is asm written for SysV registers, or is called from that
// asm, so each carries GUEST_ABI: a Windows caller must not use the MS
// convention with them (found by running tests/thunk_args under wine).
extern "C" {
// Switch FS: host <-> guest. Safe to call from either state.
GUEST_ABI void hle_fs_host();
GUEST_ABI void hle_fs_guest();
// FreeBSD-layout setjmp/longjmp for the guest. Registered raw (no thunk):
// they save and restore the guest frame itself. 12 qwords like jmp_buf.
GUEST_ABI int hle_setjmp_raw(void* buf);
[[noreturn]] GUEST_ABI void hle_longjmp_raw(void* buf, int val);
// A guest call with every argument register - the integer ones and the low
// 64 bits of xmm0-7 (a float in the low 32) - and four stack arguments. rax
// and xmm0 come back in ret and ret_xmm0. For functions with float
// arguments, more than six integer ones, or a float result.
struct GuestCallRegs {
    void* fn;                 // +0
    std::uint64_t ints[6];    // +8: rdi rsi rdx rcx r8 r9
    std::uint64_t xmm[8];     // +56
    std::uint64_t stack[4];   // +120: the 7th.. integer arguments, as the callee finds them at [rsp+8]..
    std::uint64_t ret;        // +152: rax
    std::uint64_t ret_xmm0;   // +160
};
static_assert(sizeof(GuestCallRegs) == 168, "hle_call_guest_regs reads these offsets");
GUEST_ABI void hle_call_guest_regs(GuestCallRegs* r);
// The guest's return address of the outermost HLE call this thread is in,
// and `depth` frames up the guest's rbp chain from it (0 before any call, or
// past the chain's end): for logs naming the guest code that called an HLE
// function.
std::uint64_t hle_guest_return(int depth = 0);
// Raw guest call with up to six integer/pointer args. Host FS on return.
GUEST_ABI std::int64_t hle_call_guest6(void* fn, std::int64_t a0, std::int64_t a1, std::int64_t a2,
                                       std::int64_t a3, std::int64_t a4, std::int64_t a5);
}

namespace detail {
template <typename T>
inline std::int64_t to_i64(T v) {
    if constexpr (std::is_pointer_v<T>) {
        return reinterpret_cast<std::int64_t>(v);
    } else {
        return static_cast<std::int64_t>(v);
    }
}
}  // namespace detail

template <typename R = std::int64_t, typename... A>
R hle_call_guest(void* fn, A... a) {
    static_assert(sizeof...(A) <= 6, "hle_call_guest supports six args");
    std::int64_t v[6] = {0, 0, 0, 0, 0, 0};
    std::int64_t tmp[] = {0, detail::to_i64(a)...};
    for (unsigned i = 0; i + 1 < sizeof(tmp) / sizeof(tmp[0]); ++i) {
        v[i] = tmp[i + 1];
    }
    std::int64_t r = hle_call_guest6(fn, v[0], v[1], v[2], v[3], v[4], v[5]);
    return static_cast<R>(r);
}

// The name of an HLE function and the image slide, for the BBHOST_HLE_FIRST
// log (set by the runtime; the thunk itself knows neither).
void thunk_set_lookups(const char* (*name)(void*), std::uint64_t (*slide)());
// BBHOST_HLE_COUNT=1: the HLE calls made since the last take, by thread name
// and target (thunk.cpp).
#include <string>
#include <vector>
struct HleCallCount {
    std::string thread;
    void* fn;
    std::uint64_t calls;
};
std::vector<HleCallCount> hle_call_counts_take();
