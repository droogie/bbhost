// The 1.09 eboot for tests that read or run the game's own code: the decomp
// list's differentials (docs/decomp.md, "How one is checked") - the game's
// function and ours on the same inputs, compared - and anything else that
// wants the game's bytes. A test using it skips (exit 77) where there is no
// eboot: BBHOST_EBOOT, or eboot-109-decrypted.bin in the checkout.
//
// load() maps the image the way bbhost does (core/elf.cpp: at 0x400000,
// relocated, its TLS reads moved to GS), with every import a trap that
// names itself if called - but the C library's memory and string functions
// (memset, memcpy, memmove, memcmp, strlen, strcmp, strncmp, strcpy), which
// are the host's as in the game - and the calling thread made a guest thread
// (a TLS block, the console's MXCSR and x87 control word). The game's static
// constructors do not run: a test sets up what the function under test
// reads, and stubs (stub()) what reaches further than it can set up.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

struct ElfImage;

namespace eboot_kit {

constexpr int kSkip = 77;
constexpr std::uint64_t kBase = 0x400000;  // Binary Ninja's image base, and the slide load() maps at

// The eboot's bytes, its SHA-256 checked; exits with kSkip, saying why, when
// there is none or it is not 1.09.
const std::vector<std::uint8_t>& bytes(const char* test);

// The functions .eh_frame describes - every function the eboot can unwind,
// 162,959 of them - by Binary Ninja address, in address order.
struct Function {
    std::uint64_t bn;
    std::uint32_t size;
};
const std::vector<Function>& functions(const char* test);

// The executable segment, by Binary Ninja address: [lo, hi). Code without an
// .eh_frame entry (static initializers, some small methods) lies in it too.
struct Range {
    std::uint64_t lo, hi;
};
Range text(const char* test);

// The file's bytes at a Binary Ninja address (any PT_LOAD), null outside them.
const std::uint8_t* file_at(const char* test, std::uint64_t bn);

// The image, loaded and relocated, for running the game's code (above).
// Exits with 1 when it cannot be mapped (0x400000 taken).
void load(const char* test);
// Before load(): bytes to reserve after the image for core/legacy_cpu.h's
// routines (core/elf.h's arena). After it: the loaded image.
void reserve_arena(std::size_t bytes);
::ElfImage* image();
// The pointers the loader relocated (R_X86_64_RELATIVE): where each sits
// and what it points at, by Binary Ninja address - every vtable slot among
// them. After load().
struct Relocated {
    std::uint64_t slot, target;
};
const std::vector<Relocated>& relocated();
// The calling thread's guest TCB (what the game's fs:[0] gives): its TLS
// block lies below it (PT_TLS, 0x750 bytes). After load().
void* tcb();
// The game's function at `bn` replaced by `host` (SysV) in the loaded image:
// both the game's code and ours then reach the stub - for what a test
// cannot set up (a param lookup, a heap search).
void stub(std::uint64_t bn, const void* host);
// An import the game's code calls, given a function of the test's (by the
// name bbhost binds it as, core/plt_names.inc) in place of its trap; false
// when there is none. After load().
bool bind(const char* name, const void* fn);
// A Binary Ninja address in the loaded image.
inline std::uint8_t* at(std::uint64_t bn) { return reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(bn)); }
template <class F>
F fn(std::uint64_t bn) {
    return reinterpret_cast<F>(at(bn));
}

// The console's floating-point environment (MXCSR 0x9fc0: flush-to-zero,
// denormals-are-zero; the x87 at 0x37f), which the game's code and ours
// both run under in the game; load() sets it on the calling thread.
void guest_fp();
// The host's default (MXCSR 0x1f80), for a test that wants both.
void host_fp();

// Seeded inputs, with the values a float compare should see now and then.
class Rng {
public:
    explicit Rng(std::uint64_t seed) : g_(seed) {}
    std::uint64_t next() { return g_(); }
    std::uint32_t u32() { return static_cast<std::uint32_t>(g_()); }
    int range(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g_); }
    bool chance(double p) { return std::uniform_real_distribution<double>(0, 1)(g_) < p; }
    float uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(g_); }
    // uniform(lo, hi), or - when specials are on, 15% of the time - zero of
    // either sign, a denormal, an infinity, a NaN, the extremes or a small
    // integer: where the bits, not the value, are the question.
    float value(float lo, float hi);
    bool specials = false;

private:
    std::mt19937_64 g_;
};

// The same float: the same bits, or both NaN (of whichever payload - an
// operand order the compiler was free to swap gives another).
inline bool same_float(float a, float b) {
    std::uint32_t x, y;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    return x == y || (std::isnan(a) && std::isnan(b));
}

}  // namespace eboot_kit
