// The guest's va_list, walked by hand.
//
// The eboot is SysV x86-64: a va_list is {gp_offset, fp_offset,
// overflow_arg_area, reg_save_area}, and a variadic call puts the extras in
// rdi..r9 / xmm0-7 and then on the stack. On Linux the host's va_list is the
// same thing, but on Windows it is a pointer into an MS-ABI frame, and Clang
// refuses va_start in a sysv_abi function there. So the printf and scanf
// entries the guest calls go through one shape on both platforms: an asm
// shim (SYSV_VA_ENTRY) spills the argument registers into a reg_save_area,
// builds the guest va_list and calls a C++ body that takes SysvVaList*; the
// v-prefixed entries (vsnprintf, vfprintf...) receive the guest's own
// va_list pointer directly. The bodies parse the format themselves and hand
// each conversion, with its argument pulled from the list, to the host's
// snprintf/sscanf one at a time - so a guest `%ld` (64-bit) is right on a
// host where long is 32 bits, and the layout is proven on Linux where the
// host's own va_list can be compared against it (tests/sysv_va_test.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

struct SysvVaList {
    std::uint32_t gp_offset;
    std::uint32_t fp_offset;
    void* overflow_arg_area;
    void* reg_save_area;
};

// The next integer-class (int, long, pointer) and floating-class argument.
std::uint64_t sysv_va_gp(SysvVaList* ap);
double sysv_va_fp(SysvVaList* ap);

// printf/scanf over a guest va_list; return values as the C functions'.
// n == 0 with out == nullptr measures. Guest wide strings (%ls, %lc) are
// UTF-16 and are narrowed as ASCII.
int sysv_vsnprintf(char* out, std::size_t n, const char* fmt, SysvVaList* ap);
int sysv_vfprintf(std::FILE* f, const char* fmt, SysvVaList* ap);
// The text printf would write, as a string (the console's repeat filter takes it: hle/guest_console.h).
std::string sysv_vformat(const char* fmt, SysvVaList* ap);
// UTF-16 output, native multibyte %s/%c, UTF-16 %ls/%lc; counts code units.
// Unlike snprintf, truncation returns -1, leaving a terminated prefix.
int sysv_vswprintf(char16_t* out, std::size_t n, const char16_t* fmt, SysvVaList* ap);
int sysv_vsscanf(const char* s, const char* fmt, SysvVaList* ap);
int sysv_vfscanf(std::FILE* f, const char* fmt, SysvVaList* ap);

// Declares a variadic guest entry `name` in asm whose body is `target`, a
// SysV C++ function with the same fixed arguments followed by SysvVaList*.
// `slot` is the register the va_list pointer goes in: the one after the
// fixed arguments (1 fixed -> rsi, 2 -> rdx, 3 -> rcx, 4 -> r8). The frame:
//   +0x00 rdi..r9   +0x30 xmm0..xmm7   +0xb0 the va_list   +0xe0 caller's stack args
#if defined(_WIN32)
#define SYSV_VA_TYPE_(name)
#else
#define SYSV_VA_TYPE_(name) ".type " #name ", @function\n"
#endif
#define SYSV_VA_ENTRY(name, target, nfixed, slot)                                                        \
    extern "C" GUEST_ABI int name(...);                                                                  \
    asm(".text\n"                                                                                        \
        ".globl " #name "\n" SYSV_VA_TYPE_(name) #name ":\n"                                             \
        ".cfi_startproc\n"                                                                               \
        "subq $0xd8, %rsp\n"                                                                             \
        ".cfi_adjust_cfa_offset 0xd8\n"                                                                  \
        "movq %rdi, 0x00(%rsp)\n movq %rsi, 0x08(%rsp)\n movq %rdx, 0x10(%rsp)\n"                        \
        "movq %rcx, 0x18(%rsp)\n movq %r8, 0x20(%rsp)\n movq %r9, 0x28(%rsp)\n"                          \
        "movaps %xmm0, 0x30(%rsp)\n movaps %xmm1, 0x40(%rsp)\n movaps %xmm2, 0x50(%rsp)\n"               \
        "movaps %xmm3, 0x60(%rsp)\n movaps %xmm4, 0x70(%rsp)\n movaps %xmm5, 0x80(%rsp)\n"               \
        "movaps %xmm6, 0x90(%rsp)\n movaps %xmm7, 0xa0(%rsp)\n"                                          \
        "movl $" #nfixed "*8, 0xb0(%rsp)\n"                                                              \
        "movl $48, 0xb4(%rsp)\n"                                                                         \
        "leaq 0xe0(%rsp), %rax\n movq %rax, 0xb8(%rsp)\n"                                                \
        "movq %rsp, 0xc0(%rsp)\n"                                                                        \
        "leaq 0xb0(%rsp), %" #slot "\n"                                                                  \
        "call " #target "\n"                                                                             \
        "addq $0xd8, %rsp\n"                                                                             \
        ".cfi_adjust_cfa_offset -0xd8\n"                                                                 \
        "ret\n"                                                                                          \
        ".cfi_endproc\n");
