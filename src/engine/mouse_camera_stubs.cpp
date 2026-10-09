// The stubs the mouse camera's hooks put in the game's follow camera
// (engine/mouse_camera.cpp places them; engine/mouse_camera.h): at the free
// camera's store, 0x183ce54, and at the fast turn's two exits. A file of their
// own so that a test can place them in the game's code and drive the same
// step (tests/decomp/follow_camera_test.cpp).
#include "engine/mouse_camera_stubs.h"

#include "core/asm_sym.h"

extern "C" {
void* g_mouse_camera_host = nullptr;
std::uint64_t g_mouse_camera_resume = 0;
std::uint8_t bb_mouse_camera_free_now = 0;
}

// The two stores the hook displaced, then the host with every register the
// game's code could still need around it - the SysV ones a call may clobber,
// and all sixteen xmm (the thunk uses xmm8-15 as scratch; the camera keeps its
// stick vector in xmm10) - and the new pitch back in xmm0, which the next
// block stores as the return-to-centre reference. The update is not a leaf,
// so nothing of its lives below rsp to be pushed over.
asm(".text\n"
    "    .globl " ASM_SYM(bb_mouse_camera_stub) "\n"
    ASM_SYM(bb_mouse_camera_stub) ":\n"
    "    vmovss %xmm0, 0x140(%r13)\n"
    "    vextractps $1, %xmm1, 0x144(%r13)\n"
    "    push %rax\n"
    "    push %rcx\n"
    "    push %rdx\n"
    "    push %rsi\n"
    "    push %rdi\n"
    "    push %r8\n"
    "    push %r9\n"
    "    push %r10\n"
    "    push %r11\n"
    "    push %rbx\n"
    "    mov %rsp, %rbx\n"
    "    and $-16, %rsp\n"
    "    sub $256, %rsp\n"
    "    vmovdqu %xmm0, 0(%rsp)\n"
    "    vmovdqu %xmm1, 16(%rsp)\n"
    "    vmovdqu %xmm2, 32(%rsp)\n"
    "    vmovdqu %xmm3, 48(%rsp)\n"
    "    vmovdqu %xmm4, 64(%rsp)\n"
    "    vmovdqu %xmm5, 80(%rsp)\n"
    "    vmovdqu %xmm6, 96(%rsp)\n"
    "    vmovdqu %xmm7, 112(%rsp)\n"
    "    vmovdqu %xmm8, 128(%rsp)\n"
    "    vmovdqu %xmm9, 144(%rsp)\n"
    "    vmovdqu %xmm10, 160(%rsp)\n"
    "    vmovdqu %xmm11, 176(%rsp)\n"
    "    vmovdqu %xmm12, 192(%rsp)\n"
    "    vmovdqu %xmm13, 208(%rsp)\n"
    "    vmovdqu %xmm14, 224(%rsp)\n"
    "    vmovdqu %xmm15, 240(%rsp)\n"
    "    mov %r13, %rdi\n"
    "    lea -0x3c4(%rbp), %rsi\n"
    "    vmovaps %xmm10, %xmm0\n"
    "    vmovshdup %xmm10, %xmm1\n"
    "    mov " ASM_SYM(g_mouse_camera_host) "(%rip), %rax\n"
    "    call *%rax\n"
    "    vmovdqu 0(%rsp), %xmm0\n"
    "    vmovdqu 16(%rsp), %xmm1\n"
    "    vmovdqu 32(%rsp), %xmm2\n"
    "    vmovdqu 48(%rsp), %xmm3\n"
    "    vmovdqu 64(%rsp), %xmm4\n"
    "    vmovdqu 80(%rsp), %xmm5\n"
    "    vmovdqu 96(%rsp), %xmm6\n"
    "    vmovdqu 112(%rsp), %xmm7\n"
    "    vmovdqu 128(%rsp), %xmm8\n"
    "    vmovdqu 144(%rsp), %xmm9\n"
    "    vmovdqu 160(%rsp), %xmm10\n"
    "    vmovdqu 176(%rsp), %xmm11\n"
    "    vmovdqu 192(%rsp), %xmm12\n"
    "    vmovdqu 208(%rsp), %xmm13\n"
    "    vmovdqu 224(%rsp), %xmm14\n"
    "    vmovdqu 240(%rsp), %xmm15\n"
    "    mov %rbx, %rsp\n"
    "    pop %rbx\n"
    "    pop %r11\n"
    "    pop %r10\n"
    "    pop %r9\n"
    "    pop %r8\n"
    "    pop %rdi\n"
    "    pop %rsi\n"
    "    pop %rdx\n"
    "    pop %rcx\n"
    "    pop %rax\n"
    "    vmovss 0x140(%r13), %xmm0\n"
    "    jmp *" ASM_SYM(g_mouse_camera_resume) "(%rip)\n"

    // The stick's fast turn - a stick held over, ramping up to the high speed -
    // is the free camera too, with its own two stores of pitch and a jump to
    // 0x183ce67. The stick has the camera there, so the mouse waits, as in DS3;
    // the stub only notes that the free camera ran.
    "    .globl " ASM_SYM(bb_mouse_camera_ramp_stub) "\n"
    ASM_SYM(bb_mouse_camera_ramp_stub) ":\n"
    "    vmovss %xmm0, 0x140(%r13)\n"
    "    movb $1, " ASM_SYM(bb_mouse_camera_free_now) "(%rip)\n"
    "    jmp *" ASM_SYM(g_mouse_camera_resume) "(%rip)\n");
