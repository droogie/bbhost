// The instructions a function's entry gives up to the decomp list's jump
// (decomp/decomp.cpp): their lengths, and where a rip-relative displacement
// or a relative branch's offset sits in each, so the trampoline can run them
// from another address. Enough of the x86-64 encoding for the eboot's code -
// legacy prefixes, REX, VEX, the one-, two- and three-byte maps - and what it
// does not know it says so (length 0): that entry is refused, never guessed.
#pragma once

#include <cstddef>
#include <cstdint>

struct X86Insn {
    std::uint8_t len = 0;      // 0: not an instruction this decodes
    std::uint8_t disp_at = 0;  // where a rip-relative disp32 sits in it, 0 if none
    std::uint8_t rel_at = 0;   // where a relative branch's offset sits, 0 if none
    std::uint8_t rel_size = 0; // that offset's size: 1 or 4
    enum Kind : std::uint8_t {
        Plain,
        Jmp,   // jmp rel8 / rel32
        Jcc,   // jcc rel8 / rel32, condition in cc
        Call,  // call rel32
        Loop,  // loop, loope, loopne, jrcxz: rel8 only, with no rel32 form to widen to
    } kind = Plain;
    std::uint8_t cc = 0;
    std::uint8_t modrm_at = 0;  // where its ModRM byte sits, 0 if it has none
    std::uint8_t rex = 0;       // REX's bits (or VEX's, as REX would give them): 0x40 | W R X B, 0 for none
};

// The instruction at p, reading at most `avail` bytes.
X86Insn x86_decode(const std::uint8_t* p, std::size_t avail);

// Copies the n bytes of whole instructions at `src` - which run at `from` -
// to `dst`, which will run at `to`, so that they do what they did there:
// rip-relative displacements and relative branches re-aimed at what they
// reached, a short branch widened to its rel32 form. Returns the bytes
// written, or 0 when they cannot be moved: an instruction this does not
// decode, a loop/jrcxz, a branch into the moved bytes themselves, a target
// out of a rel32's reach from `to`, n not ending on an instruction, or `cap`
// too small.
std::size_t x86_relocate(const std::uint8_t* src, std::size_t n, std::uint64_t from, std::uint8_t* dst, std::uint64_t to,
                         std::size_t cap);
