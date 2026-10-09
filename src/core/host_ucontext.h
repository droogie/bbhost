// UC_REG(uc, RIP): a signal handler's saved register as an lvalue, on Linux
// and macOS. macOS's ERR has the write bit 2, as on Linux.
#pragma once

#if defined(__APPLE__)
#include <cstdint>
#include <sys/ucontext.h>  // not the deprecated <ucontext.h>

#define UC_REG(uc, r) ((uc)->uc_mcontext->UC_APPLE_##r)
using host_greg_t = std::uint64_t;

#define UC_APPLE_RAX __ss.__rax
#define UC_APPLE_RBX __ss.__rbx
#define UC_APPLE_RCX __ss.__rcx
#define UC_APPLE_RDX __ss.__rdx
#define UC_APPLE_RSI __ss.__rsi
#define UC_APPLE_RDI __ss.__rdi
#define UC_APPLE_RBP __ss.__rbp
#define UC_APPLE_RSP __ss.__rsp
#define UC_APPLE_R8 __ss.__r8
#define UC_APPLE_R9 __ss.__r9
#define UC_APPLE_R10 __ss.__r10
#define UC_APPLE_R11 __ss.__r11
#define UC_APPLE_R12 __ss.__r12
#define UC_APPLE_R13 __ss.__r13
#define UC_APPLE_R14 __ss.__r14
#define UC_APPLE_R15 __ss.__r15
#define UC_APPLE_RIP __ss.__rip
#define UC_APPLE_EFL __ss.__rflags
#define UC_APPLE_ERR __es.__err
#else
#include <ucontext.h>

#define UC_REG(uc, r) ((uc)->uc_mcontext.gregs[REG_##r])
using host_greg_t = greg_t;
#endif
