// Object-format differences for the hand-written x86-64 asm: Mach-O prefixes
// C names with '_'; .type and .size exist only in ELF.
#pragma once

#if defined(__APPLE__)
#define ASM_SYM(name) "_" #name
#else
#define ASM_SYM(name) #name
#endif

#if defined(__ELF__)
#define ASM_TYPE_FUNCTION(name) ".type " #name ", @function\n"
#define ASM_SIZE(name) ".size " #name ", .-" #name "\n"
#else
#define ASM_TYPE_FUNCTION(name)
#define ASM_SIZE(name)
#endif
