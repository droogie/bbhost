#pragma once

#include <cstdint>

#include "guest_abi.h"

// The effects' node emission, sub_2c4e6b0 (decomp/sfx/node_emit.cpp): a staged
// node's emitting state, as ours, so its emission events can follow a clock of
// their own (the emission clock, [video] effect_emission).
void decomp_sfx_node_emit_add();

namespace sfx_node {

// What emit() is asked to do.
enum class Emit : int {
    // sub_2c4e6b0 itself: (node, the instance's definition, render context)
    // -> the node's next state.
    Game,
    // The same, but where the game makes an emission event - a CPU particle
    // count added to the node's container, or its children - 1 is answered
    // and nothing done: the emission clock makes the events.
    Edges,
    // Only the emission event, at time t (not the node's own +0x7c): made
    // when the game would make one at t with no edge before it - a record
    // switch, a reset, a set-up, a linger, a 6 - and nothing done otherwise,
    // the edges waiting for the whole step. Children only when `children`
    // (a whole step's frame: the node trees change only there). 1 when made.
    // The next (+0xe4) is met within `slack`: the clock's sub-steps lie on
    // the node's whole steps (+0x7c moved by the whole step, as at 30), the
    // next on the last event's time and the interval (as at 60), and an add
    // rounding the other way must not skip a sub-step.
    Event,
};

GUEST_ABI __attribute__((no_stack_protector)) std::uint32_t emit(std::uint64_t node, std::uint64_t definition, std::uint64_t context,
                                                                  Emit mode, float t, bool children, float slack = 0.0f);

// The emission clock: 0 off - the game's one emission event an update -
// else the number of 1/60 s sub-steps in one of the game's whole steps (2
// for whole 1/30 s steps, at 30 fps too; 1 for whole 1/60 s steps). Set at
// start (engine/frame_rate.cpp), read by the node update.
void set_emission_clock(int subs);
int emission_clock();

}  // namespace sfx_node

// sub_2c4e6b0 itself, the game's signature.
GUEST_ABI __attribute__((no_stack_protector)) std::uint32_t sfx_node_emit(std::uint64_t node, std::uint64_t definition,
                                                                           std::uint64_t context);
