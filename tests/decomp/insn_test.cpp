// decomp/insn.cpp: instruction lengths and rip-relative / branch fields
// against capstone's reading of the same bytes, and relocated instructions
// run from another page doing what they did where they were.
#include "decomp/insn.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

#include <sys/mman.h>

namespace {

int g_bad = 0;

struct Case {
    std::vector<std::uint8_t> bytes;
    int len, disp_at, rel_at;
    X86Insn::Kind kind;
};

// Lengths and offsets as capstone 5 reads them.
const Case kCases[] = {
    {{0x55}, 1, 0, 0, X86Insn::Plain},
    {{0x48, 0x89, 0xe5}, 3, 0, 0, X86Insn::Plain},
    {{0x41, 0x57}, 2, 0, 0, X86Insn::Plain},
    {{0x48, 0x83, 0xec, 0x30}, 4, 0, 0, X86Insn::Plain},
    {{0x48, 0x81, 0xec, 0x00, 0x01, 0x00, 0x00}, 7, 0, 0, X86Insn::Plain},
    {{0x48, 0x8d, 0x05, 0x10, 0, 0, 0}, 7, 3, 0, X86Insn::Plain},                  // lea rax, [rip+0x10]
    {{0x48, 0x8b, 0x05, 0x10, 0, 0, 0}, 7, 3, 0, X86Insn::Plain},                  // mov rax, [rip+0x10]
    {{0xc5, 0xfa, 0x10, 0x05, 0x10, 0, 0, 0}, 8, 4, 0, X86Insn::Plain},            // vmovss xmm0, [rip+0x10]
    {{0xc4, 0xe2, 0x79, 0x18, 0x05, 0x10, 0, 0, 0}, 9, 5, 0, X86Insn::Plain},      // vbroadcastss xmm0, [rip+0x10]
    {{0x80, 0x3d, 0x10, 0, 0, 0, 0x00}, 7, 2, 0, X86Insn::Plain},                  // cmp byte [rip+0x10], 0
    {{0xc7, 0x05, 0x10, 0, 0, 0, 1, 0, 0, 0}, 10, 2, 0, X86Insn::Plain},           // mov dword [rip+0x10], 1
    {{0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00}, 6, 0, 0, X86Insn::Plain},
    {{0x0f, 0x1f, 0x80, 0, 0, 0, 0}, 7, 0, 0, X86Insn::Plain},
    {{0x74, 0x05}, 2, 0, 1, X86Insn::Jcc},
    {{0x0f, 0x84, 0x00, 0x01, 0x00, 0x00}, 6, 0, 2, X86Insn::Jcc},
    {{0xe8, 0, 0, 0, 0}, 5, 0, 1, X86Insn::Call},
    {{0xe9, 0, 0, 0, 0}, 5, 0, 1, X86Insn::Jmp},
    {{0xeb, 0xfe}, 2, 0, 1, X86Insn::Jmp},
    {{0xe3, 0x00}, 2, 0, 1, X86Insn::Loop},                                        // jrcxz
    {{0xf6, 0xc1, 0x01}, 3, 0, 0, X86Insn::Plain},                                 // test cl, 1
    {{0xf6, 0xd1}, 2, 0, 0, X86Insn::Plain},                                       // not cl
    {{0xf7, 0xc1, 0x00, 0x01, 0x00, 0x00}, 6, 0, 0, X86Insn::Plain},               // test ecx, 0x100
    {{0xf7, 0xd9}, 2, 0, 0, X86Insn::Plain},                                       // neg ecx
    {{0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7, 8}, 10, 0, 0, X86Insn::Plain},              // movabs rax, imm64
    {{0x66, 0xb8, 0x34, 0x12}, 4, 0, 0, X86Insn::Plain},                           // mov ax, imm16
    {{0xc4, 0xe3, 0x79, 0x04, 0xc0, 0x00}, 6, 0, 0, X86Insn::Plain},               // vpermilps xmm0, xmm0, 0
    {{0x0f, 0xae, 0xf0}, 3, 0, 0, X86Insn::Plain},                                 // mfence
    {{0xc5, 0xf8, 0x77}, 3, 0, 0, X86Insn::Plain},                                 // vzeroupper
    {{0x48, 0x8b, 0x04, 0x25, 0x10, 0, 0, 0}, 8, 0, 0, X86Insn::Plain},            // mov rax, [0x10]: SIB, no base
    {{0x48, 0x8b, 0x44, 0x24, 0x08}, 5, 0, 0, X86Insn::Plain},
    {{0x48, 0x8b, 0x84, 0x24, 0x00, 0x01, 0x00, 0x00}, 8, 0, 0, X86Insn::Plain},
    {{0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0}, 9, 0, 0, X86Insn::Plain},         // mov rax, fs:[0]
    {{0xf3, 0x0f, 0x10, 0x05, 0x10, 0, 0, 0}, 8, 4, 0, X86Insn::Plain},            // movss xmm0, [rip+0x10]
    {{0xff, 0x15, 0x10, 0, 0, 0}, 6, 2, 0, X86Insn::Plain},                        // call [rip+0x10]
    {{0xff, 0x25, 0x10, 0, 0, 0}, 6, 2, 0, X86Insn::Plain},                        // jmp [rip+0x10]
    {{0x0f, 0x0b}, 2, 0, 0, X86Insn::Plain},                                       // ud2
    {{0x3e, 0x74, 0x02}, 3, 0, 2, X86Insn::Jcc},                                   // je with a branch hint
    {{0xc4, 0xe2, 0x7d, 0x18, 0x05, 0x10, 0, 0, 0}, 9, 5, 0, X86Insn::Plain},      // vbroadcastss ymm0, [rip+0x10]
    {{0x66, 0x41, 0x0f, 0x7e, 0xc0}, 5, 0, 0, X86Insn::Plain},                     // movd r8d, xmm0
    {{0x48, 0x0f, 0xaf, 0xc1}, 4, 0, 0, X86Insn::Plain},                           // imul rax, rcx
    {{0x0f, 0xb6, 0x05, 0x10, 0, 0, 0}, 7, 3, 0, X86Insn::Plain},                  // movzx eax, byte [rip+0x10]
    {{0x48, 0x63, 0xc7}, 3, 0, 0, X86Insn::Plain},                                 // movsxd rax, edi
    {{0xc4, 0xc1, 0x7a, 0x10, 0x44, 0x24, 0x08}, 7, 0, 0, X86Insn::Plain},         // vmovss xmm0, [r12+8]
    {{0x41, 0xf6, 0x44, 0x24, 0x08, 0x01}, 6, 0, 0, X86Insn::Plain},               // test byte [r12+8], 1
    {{0x66, 0x81, 0x3d, 0x10, 0, 0, 0, 0x34, 0x12}, 9, 3, 0, X86Insn::Plain},      // cmp word [rip+0x10], 0x1234
    {{0x48, 0x69, 0xc0, 0x10, 0, 0, 0}, 7, 0, 0, X86Insn::Plain},                  // imul rax, rax, 0x10
    {{0x6b, 0xc0, 0x10}, 3, 0, 0, X86Insn::Plain},
    {{0x0f, 0xba, 0xe0, 0x03}, 4, 0, 0, X86Insn::Plain},                           // bt eax, 3
    {{0xc1, 0xe0, 0x04}, 3, 0, 0, X86Insn::Plain},
    {{0xd1, 0xe0}, 2, 0, 0, X86Insn::Plain},
    {{0xd9, 0x05, 0x10, 0, 0, 0}, 6, 2, 0, X86Insn::Plain},                        // fld dword [rip+0x10]
    {{0xf3, 0x48, 0xab}, 3, 0, 0, X86Insn::Plain},                                 // rep stosq
    {{0x48, 0xa1, 1, 2, 3, 4, 5, 6, 7, 8}, 10, 0, 0, X86Insn::Plain},              // movabs rax, [moffs64]
    {{0x06}, 0, 0, 0, X86Insn::Plain},                                             // not an instruction in 64-bit mode
    {{0x48, 0x8b}, 0, 0, 0, X86Insn::Plain},                                       // cut short
};

void check_cases() {
    for (const Case& c : kCases) {
        const X86Insn x = x86_decode(c.bytes.data(), c.bytes.size());
        const bool rel_ok = c.len == 0 || x.rel_at == c.rel_at;
        if (x.len != c.len || (c.len && (x.disp_at != c.disp_at || x.kind != c.kind)) || !rel_ok) {
            std::printf("decode %02x %02x..: len %d disp %d rel %d kind %d, expected %d %d %d %d\n", c.bytes[0],
                        c.bytes.size() > 1 ? c.bytes[1] : 0, x.len, x.disp_at, x.rel_at, x.kind, c.len, c.disp_at, c.rel_at,
                        c.kind);
            ++g_bad;
        }
    }
}

// Two pages in the low 2 GiB, a rel32 apart at most.
std::uint8_t* low_page() {
#if defined(__APPLE__)
    // Nothing is free below 4 GiB on macOS (__PAGEZERO, core/mac_vm.h): one
    // mapping keeps the two pages close.
    static std::uint8_t* pages = nullptr;
    static int taken = 0;
    if (!pages) {
        void* p = mmap(nullptr, 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return nullptr;
        pages = static_cast<std::uint8_t*>(p);
    }
    return taken < 2 ? pages + 0x1000 * taken++ : nullptr;
#else
    void* p = mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<std::uint8_t*>(p);
#endif
}

void put(std::uint8_t* at, std::initializer_list<std::uint8_t> bytes) { std::memcpy(at, bytes.begin(), bytes.size()); }

void put_jmp(std::uint8_t* at, const std::uint8_t* target) {
    at[0] = 0xe9;
    const auto rel = static_cast<std::int32_t>(target - (at + 5));
    std::memcpy(at + 1, &rel, 4);
}

std::uint64_t addr(const void* p) { return reinterpret_cast<std::uint64_t>(p); }

// The first n bytes of `fn` moved to `to`, then a jump back after them.
template <class F>
F moved(std::uint8_t* fn, std::size_t n, std::uint8_t* to) {
    const std::size_t m = x86_relocate(fn, n, addr(fn), to, addr(to), 64);
    if (!m) return nullptr;
    put_jmp(to + m, fn + n);
    return reinterpret_cast<F>(to);
}

void expect(const char* what, long got, long want) {
    if (got != want) {
        std::printf("%s: %ld, expected %ld\n", what, got, want);
        ++g_bad;
    }
}

void check_runs() {
    std::uint8_t* a = low_page();
    std::uint8_t* b = low_page();
    if (!a || !b) {
        std::printf("no low pages\n");
        ++g_bad;
        return;
    }
    // lea rax, [rip+d] / mov eax, [rax] / ret, the value 0x100 bytes on.
    put(a, {0x48, 0x8d, 0x05, 0, 0, 0, 0, 0x8b, 0x00, 0xc3});
    const std::int32_t d = 0x100 - 7;
    std::memcpy(a + 3, &d, 4);
    const std::uint32_t value = 0x12345678;
    std::memcpy(a + 0x100, &value, 4);
    if (auto f = moved<int (*)()>(a, 7, b)) expect("lea moved", f(), static_cast<int>(value));
    else ++g_bad;

    // test edi, edi / je +6 / mov eax, 1 / ret / mov eax, 2 / ret: the short
    // jcc widened, still going where it went.
    std::uint8_t* f2 = a + 0x200;
    put(f2, {0x85, 0xff, 0x74, 0x06, 0xb8, 0x01, 0, 0, 0, 0xc3, 0xb8, 0x02, 0, 0, 0, 0xc3});
    if (auto f = moved<int (*)(int)>(f2, 4, b + 0x80)) {
        expect("jcc taken", f(0), 2);
        expect("jcc not taken", f(5), 1);
    } else {
        ++g_bad;
    }

    // call helper / add eax, 1 / ret, the helper returning 7.
    std::uint8_t* f3 = a + 0x300;
    std::uint8_t* helper = a + 0x380;
    put(helper, {0xb8, 0x07, 0, 0, 0, 0xc3});
    put_jmp(f3, helper);
    f3[0] = 0xe8;
    put(f3 + 5, {0x83, 0xc0, 0x01, 0xc3});
    if (auto f = moved<int (*)()>(f3, 5, b + 0x100)) expect("call moved", f(), 8);
    else ++g_bad;

    // A short jmp over a ret, widened.
    std::uint8_t* f4 = a + 0x400;
    put(f4, {0xeb, 0x01, 0xc3, 0xb8, 0x2a, 0, 0, 0, 0xc3});
    if (auto f = moved<int (*)()>(f4, 2, b + 0x180)) expect("jmp moved", f(), 42);
    else ++g_bad;

    // What cannot move: a branch back into the moved bytes, jrcxz, a cut
    // instruction.
    std::uint8_t out[64];
    const std::uint8_t loop_back[] = {0x85, 0xff, 0x74, 0xfc};
    expect("branch into moved bytes", static_cast<long>(x86_relocate(loop_back, 4, 0x1000, out, 0x2000, 64)), 0);
    const std::uint8_t jrcxz[] = {0xe3, 0x00, 0x90};
    expect("jrcxz", static_cast<long>(x86_relocate(jrcxz, 3, 0x1000, out, 0x2000, 64)), 0);
    const std::uint8_t cut[] = {0x48, 0x8d, 0x05, 0, 0, 0, 0};
    expect("cut instruction", static_cast<long>(x86_relocate(cut, 5, 0x1000, out, 0x2000, 64)), 0);
    // Out of a rel32's reach.
    const std::uint8_t lea[] = {0x48, 0x8d, 0x05, 0, 0, 0, 0};
    expect("out of reach", static_cast<long>(x86_relocate(lea, 7, 0x1000, out, 0x400000000ull, 64)), 0);
}

}  // namespace

int main() {
    check_cases();
    check_runs();
    std::printf("decomp_insn: %zu decodes and four relocated runs, %d wrong\n", std::size(kCases), g_bad);
    return g_bad ? 1 : 0;
}
