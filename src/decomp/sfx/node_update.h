#pragma once

#include <cstdint>

#include "guest_abi.h"

// The effects' node update, sub_2c4df80 (decomp/sfx/node_update.cpp): the
// game's sequence as ours, for the smooth steps to split.
void decomp_sfx_node_update_add();

namespace sfx_node {

// What one node update is given. The game gives both kinds one step, with no
// offsets, every update; the smooth effect steps' frame domain gives the
// counted parts (a plain node's definition switch, pick and definition time,
// a staged node's state machine and render items, the node's time) the whole
// step's on a frame with one and nothing on a frame between, and the
// container calls the frame's own step with the node's time put that far past
// its whole step for each call.
struct Steps {
    float counted = 0.0f;  // the counted parts' step (the whole step's, times the instance's scale)
    float frame = 0.0f;    // the container calls' step (the frame's, times the scale)
    float from = 0.0f;     // the node's time offset for the container calls: the frame's start past its whole step, scaled
    float into = 0.0f;     // a plain node's definition-time offset for its second call: the frame's end past it, scaled
    bool tick = true;      // a frame with a whole step: the counted parts run, a dead node is freed
    bool smooth = false;   // the smooth steps' frame domain: each container call is published (smooth_call)
    float sub = 0.0f;      // the emission clock's sub-step (1/60 s, scaled; node_emit.h): an emitting staged node makes the
                           // events of the sub-steps its time reaches this frame; 0: the game's one event a whole step
};

// The node update with those steps; false when the node is to be freed.
GUEST_ABI __attribute__((no_stack_protector)) std::uint8_t update(std::uint64_t node, const Steps& steps, std::uint64_t definition,
                                                                  std::uint64_t context);

// The container call the smooth steps' frame domain is making, for a module
// whose update counts what the game counts once an update (the FFX light's
// flicker phase): its node, that node's time of the whole step (+0x7c before
// the call's offset, its bits) and whether the frame has a whole step. No
// node (0) outside such a call. The game updates an effect manager's
// instances one after the other, so one call is made at a time; a reader
// takes the node, then the rest, then the node again, and a node that moved
// meanwhile reads as none.
struct SmoothCall {
    std::uint64_t node = 0;
    std::uint32_t time = 0;
    bool tick = false;
};
SmoothCall smooth_call();

}  // namespace sfx_node

// sub_2c4df80 itself, the game's signature: one step for both kinds, no offsets.
GUEST_ABI __attribute__((no_stack_protector)) std::uint8_t sfx_node_update(std::uint64_t node, float* step, std::uint64_t definition,
                                                                           std::uint64_t context);
