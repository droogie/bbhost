#pragma once

// PM4 WAIT_REG_MEM (type-3 opcode 0x3c), as the command processor
// (hle/gnm_exec.cpp) reads it: after the header, [0] the compare function in
// bits 2:0 (0 always, 1 <, 2 <=, 3 ==, 4 !=, 5 >=, 6 >), memory rather than a
// register in bit 4, the engine in bit 8; [1] the address's low dword (or a
// register); [2] its high dword; [3] the reference; [4] the mask; [5] the poll
// interval. The order Gnm's writers use in the 1.09 eboot - the draw and
// compute command buffers' waits (0x1477b30, 0x148b430) put the reference at
// +4 of the packet and the mask at +5, a poll interval of 10 at +6 - and the
// amdgpu kernel's pm4__wait_reg_mem (reference, then mask).
//
// The compare is the polled dword under the mask against the reference, the
// reference as it is. Every wait the game writes has its reference inside
// its mask (its label waits a mask of 0xffffffff), so a reference the mask
// can never reach means the packet was not the game's as written: something
// wrote over it in memory.

#include <cstdint>

namespace wait_reg_mem {

inline bool passes(unsigned fn, std::uint32_t value, std::uint32_t ref, std::uint32_t mask) {
    const std::uint32_t v = value & mask;
    switch (fn) {
        case 1: return v < ref;
        case 2: return v <= ref;
        case 3: return v == ref;
        case 4: return v != ref;
        case 5: return v >= ref;
        case 6: return v > ref;
        default: return true;  // 0 always; 7 reserved
    }
}

// Whether any dword in memory can pass: the masked value ranges over the
// mask's subsets, from 0 to the mask itself.
inline bool can_pass(unsigned fn, std::uint32_t ref, std::uint32_t mask) {
    switch (fn) {
        case 1: return ref != 0;                  // 0 < ref
        case 2: return true;                      // 0 <= ref
        case 3: return (ref & ~mask) == 0;        // ref is one of the subsets
        case 4: return mask != 0 || ref != 0;     // two values to choose from, or 0 against ref
        case 5: return ref <= mask;               // the mask >= ref
        case 6: return ref < mask;                // the mask > ref
        default: return true;
    }
}

}  // namespace wait_reg_mem
