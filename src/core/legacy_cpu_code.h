// The game's BMI1, LZCNT, TZCNT and MOVBE instructions as code that needs
// none of them (core/legacy_cpu.h rewrites the eboot with it): a decoder for
// the four families, their results as the processor's manual defines them,
// and the routine standing in for one instruction.
//
// A routine computes exactly what the instruction does - the destination,
// and every flag the manual defines for it (ZF, SF, CF and OF as each sets
// them; MOVBE sets none, and its routine keeps all of them) - and changes no
// other register. It reaches its scratch registers by pushing them below the
// SysV red zone (rsp is stepped down 128 first), so the 128 bytes under the
// guest's rsp are never touched. Flags the manual leaves undefined (PF, AF,
// and SF for BEXTR, LZCNT, TZCNT) come out as the routine's last arithmetic
// leaves them.
//
// Processors before Haswell (Intel) and Excavator (AMD) lack these: BMI1 and
// MOVBE raise #UD, while LZCNT and TZCNT run as BSR and BSF - no fault, a
// different answer (lzcnt(1) is 31, bsr(1) is 0). The routines are built
// from BSR/BSF, BSWAP, shifts and the base instruction set.
#pragma once

#include <cstddef>
#include <cstdint>

enum class LegacyFamily : std::uint8_t { kAndn, kBextr, kBlsi, kBlsmsk, kBlsr, kLzcnt, kTzcnt, kMovbe };
const char* legacy_family_name(LegacyFamily f);

// A memory operand: [base + index*scale + disp], or [rip + disp] (base kRip,
// relative to the instruction's end).
struct LegacyMem {
    static constexpr std::int8_t kNone = -1, kRip = 16;
    std::int8_t base = kNone;   // 0-15, kNone or kRip
    std::int8_t index = kNone;  // 0-15 (never rsp), or kNone
    std::uint8_t scale = 1;
    std::int32_t disp = 0;
};

// Registers by their encoding: 0 rax, 1 rcx, 2 rdx, 3 rbx, 4 rsp, 5 rbp,
// 6 rsi, 7 rdi, 8-15 r8-r15.
struct LegacyInsn {
    LegacyFamily family = LegacyFamily::kAndn;
    std::uint8_t len = 0;    // the instruction's bytes
    std::uint8_t size = 0;   // operand bytes: 2 (LZCNT, TZCNT, MOVBE), 4 or 8
    std::uint8_t dst = 0;    // the register written; a MOVBE store's register read
    std::uint8_t vex = 0;    // VEX.vvvv: ANDN's inverted operand, BEXTR's control
    bool store = false;      // MOVBE m, r
    bool mem = false;        // the r/m operand is memory (always, for MOVBE)
    std::uint8_t rm = 0;     // the r/m register, when it is not memory
    LegacyMem m;             // the r/m operand, when it is
};

// The instruction at p, reading at most `avail` bytes: true when it is one of
// the families above in a form the routines cover (no segment override, no
// 32-bit addressing, no lock). BLSMSK is decoded and computed, and has no
// site in the 1.09 eboot.
bool legacy_decode(const std::uint8_t* p, std::size_t avail, LegacyInsn* out);

// The flags the manual defines for the instruction (an EFLAGS mask: CF 0x1,
// ZF 0x40, SF 0x80, OF 0x800). Zero for MOVBE.
std::uint32_t legacy_defined_flags(const LegacyInsn& in);

// What the instruction computes, as the manual defines it: `rm` the r/m
// operand's value (a register's, or the size bytes read from memory), `vex`
// the VEX.vvvv register's, `old` the destination's value before (a 16-bit
// form keeps its upper bits), `flags` EFLAGS before. Returns the destination
// (or, for a MOVBE store, the value to write) and sets *flags_out: the defined
// flags computed, the rest as they were.
std::uint64_t legacy_eval(const LegacyInsn& in, std::uint64_t rm, std::uint64_t vex, std::uint64_t old, std::uint32_t flags,
                          std::uint32_t* flags_out);

// Writes, at `out` (which runs at `out_va`, at most `cap` bytes), code doing
// what `in` did when it ended at `end_va` (where a rip-relative operand is
// reached from), then falling through. Returns the bytes written, or 0 when
// they do not fit or a rip-relative operand is out of a rel32's reach.
std::size_t legacy_emit(const LegacyInsn& in, std::uint64_t end_va, std::uint8_t* out, std::uint64_t out_va, std::size_t cap);

// The effective address of the instruction's memory operand, from a register
// file (regs[16] by encoding) and the instruction's end.
std::uint64_t legacy_address(const LegacyInsn& in, const std::uint64_t regs[16], std::uint64_t end_va);

// The instruction applied to a register file: regs[16] (rsp included),
// *flags (EFLAGS), memory read and written at the effective address. For the
// illegal-instruction path, which computes it in a faulting thread's context.
void legacy_apply(const LegacyInsn& in, std::uint64_t regs[16], std::uint64_t* flags, std::uint64_t end_va);

// x86_relocate (decomp/insn.h) with each BMI1, LZCNT, TZCNT or MOVBE
// instruction among the n bytes replaced by its routine: for code that moves
// the game's instructions elsewhere while they are rewritten (a hook's stub,
// a decomp trampoline). `to` 0: the copy's address is not known yet, and
// nothing in it may depend on it (no rip-relative operand, no relative
// branch). Returns the bytes written, 0 when they cannot be moved.
std::size_t legacy_relocate(const std::uint8_t* src, std::size_t n, std::uint64_t from, std::uint8_t* dst, std::uint64_t to,
                            std::size_t cap);
