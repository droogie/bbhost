// The machine the legacy-CPU tests run code on (tests/legacy_cpu_test.cpp,
// tests/legacy_cpu_sites_test.cpp): a block of executable memory, and around
// the code under test a harness that loads every general register and the
// arithmetic flags from a State, switches to a stack of the test's own,
// runs the code, and saves the registers and flags back without touching
// that stack - so the 128 bytes under rsp (the red zone) are compared as the
// code left them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#include <cpuid.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace harness {

// EFLAGS' arithmetic bits, as lahf and seto capture them.
constexpr std::uint32_t kCF = 0x1, kPF = 0x4, kAF = 0x10, kZF = 0x40, kSF = 0x80, kOF = 0x800;
constexpr std::uint32_t kArith = kCF | kPF | kAF | kZF | kSF | kOF;

struct State {
    std::uint64_t r[16];        // rax rcx rdx rbx rsp rbp rsi rdi r8-r15
    std::uint64_t flags;        // the arithmetic bits
    std::uint8_t x[16][16];     // xmm0-15
};
constexpr std::uint32_t kXmmAt = 8 * 17;

inline std::uint8_t* alloc_exec(std::size_t size) {
#if defined(_WIN32)
    return static_cast<std::uint8_t*>(VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
#else
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<std::uint8_t*>(p);
#endif
}

// What the processor has: the native side of a comparison runs only then.
struct Features {
    bool bmi1 = false, lzcnt = false, movbe = false;
};
inline Features features() {
    Features f;
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid(1, &a, &b, &c, &d)) f.movbe = (c >> 22) & 1;
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) f.bmi1 = (b >> 3) & 1;
    if (__get_cpuid(0x80000001u, &a, &b, &c, &d)) f.lzcnt = (c >> 5) & 1;
    if (const char* e = std::getenv("LEGACY_CPU_TEST_NO_NATIVE"); e && e[0] == '1') f = Features{};
    return f;
}

// Saved across a run: the host's rsp, and the state in and out.
struct Slots {
    std::uint64_t host_rsp;
    std::uint64_t ax;  // the out-state's rax, before lahf takes it
    std::uint16_t flagword;
    State in, out;
};

class Writer {
public:
    Writer(std::uint8_t* p) : p_(p) {}
    void b(std::uint8_t v) { *p_++ = v; }
    void bs(std::initializer_list<std::uint8_t> v) {
        for (std::uint8_t x : v) b(x);
    }
    void q(std::uint64_t v) {
        std::memcpy(p_, &v, 8);
        p_ += 8;
    }
    std::uint8_t* at() const { return p_; }
    void skip(std::size_t n) { p_ += n; }

private:
    std::uint8_t* p_;
};

// mov reg, [rax + off] (64-bit), reg 0-15 but rax.
inline void load_from_rax(Writer& w, int reg, std::uint32_t off) {
    w.b(static_cast<std::uint8_t>(0x48 | ((reg & 8) ? 4 : 0)));
    w.b(0x8b);
    w.b(static_cast<std::uint8_t>(0x80 | ((reg & 7) << 3)));  // [rax + disp32]
    std::memcpy(w.at(), &off, 4);
    w.skip(4);
}
// movdqu xmmN, [rax + off] / movdqu [rax + off], xmmN.
inline void xmm_rax(Writer& w, int reg, std::uint32_t off, bool store) {
    w.b(0xf3);
    if (reg & 8) w.b(0x44);
    w.b(0x0f);
    w.b(store ? 0x7f : 0x6f);
    w.b(static_cast<std::uint8_t>(0x80 | ((reg & 7) << 3)));
    std::memcpy(w.at(), &off, 4);
    w.skip(4);
}
inline void store_to_rax(Writer& w, int reg, std::uint32_t off) {
    w.b(static_cast<std::uint8_t>(0x48 | ((reg & 8) ? 4 : 0)));
    w.b(0x89);
    w.b(static_cast<std::uint8_t>(0x80 | ((reg & 7) << 3)));
    std::memcpy(w.at(), &off, 4);
    w.skip(4);
}

// The harness's code before the code under test. Returns where that goes.
inline std::uint8_t* prologue(std::uint8_t* code, Slots* s) {
    Writer w(code);
    w.bs({0x53, 0x55, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57});  // push rbx rbp r12 r13 r14 r15
    w.bs({0x48, 0xb8});
    w.q(reinterpret_cast<std::uint64_t>(&s->host_rsp));
    w.bs({0x48, 0x89, 0x20});  // mov [rax], rsp
    w.bs({0x48, 0xb8});
    w.q(reinterpret_cast<std::uint64_t>(&s->in.flags));
    w.bs({0xff, 0x30, 0x9d});  // push qword [rax]; popfq (on the host's stack)
    w.bs({0x48, 0xb8});
    w.q(reinterpret_cast<std::uint64_t>(&s->in));
    for (int r = 0; r < 16; ++r) xmm_rax(w, r, kXmmAt + 16u * static_cast<std::uint32_t>(r), false);
    for (int r = 1; r < 16; ++r) load_from_rax(w, r, static_cast<std::uint32_t>(8 * r));  // rsp included
    load_from_rax(w, 0, 0);                                                               // rax last
    return w.at();
}

// After it: save the state without the stack, then back to the host.
inline std::uint8_t* epilogue(std::uint8_t* code, Slots* s) {
    Writer w(code);
    w.bs({0x48, 0xa3});  // mov [moffs64], rax
    w.q(reinterpret_cast<std::uint64_t>(&s->ax));
    w.bs({0x9f});              // lahf
    w.bs({0x0f, 0x90, 0xc0});  // seto al
    w.bs({0x66, 0xa3});        // mov [moffs64], ax
    w.q(reinterpret_cast<std::uint64_t>(&s->flagword));
    w.bs({0x48, 0xb8});
    w.q(reinterpret_cast<std::uint64_t>(&s->out));
    for (int r = 1; r < 16; ++r) store_to_rax(w, r, static_cast<std::uint32_t>(8 * r));
    for (int r = 0; r < 16; ++r) xmm_rax(w, r, kXmmAt + 16u * static_cast<std::uint32_t>(r), true);
    w.bs({0x48, 0xb8});
    w.q(reinterpret_cast<std::uint64_t>(&s->host_rsp));
    w.bs({0x48, 0x8b, 0x20});  // mov rsp, [rax]
    w.bs({0xfc});              // cld
    w.bs({0x41, 0x5f, 0x41, 0x5e, 0x41, 0x5d, 0x41, 0x5c, 0x5d, 0x5b, 0xc3});  // pop r15 r14 r13 r12 rbp rbx; ret
    return w.at();
}

#if defined(_WIN32)
using Entry = void(__attribute__((sysv_abi))*)();
#else
using Entry = void (*)();
#endif

// Runs code built with prologue/epilogue around it; fills s->out.
inline void run(std::uint8_t* code, Slots* s) {
    reinterpret_cast<Entry>(code)();
    s->out.r[0] = s->ax;
    const std::uint8_t ah = static_cast<std::uint8_t>(s->flagword >> 8), al = static_cast<std::uint8_t>(s->flagword);
    s->out.flags = (ah & (kCF | kPF | kAF | kZF | kSF)) | (al ? kOF : 0);
}

// A small deterministic generator (the tests must run the same everywhere).
struct Rng {
    std::uint64_t s;
    std::uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    std::uint64_t below(std::uint64_t n) { return n ? next() % n : 0; }
};

}  // namespace harness
