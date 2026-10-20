#pragma once

// The effects' frames between whole steps (decomp/sfx/step_gate.cpp):
// engine/frame_rate.cpp says each frame, before the game's effect update,
// whether bbhost put this frame between whole effect steps (and the gate is
// on). The game's own still frame (SprjSfx+0x8c) is told apart by the gate.
void decomp_sfx_gate_frame(bool between);
// Whether the gate's functions took the game's place.
bool decomp_sfx_gate_placed();
