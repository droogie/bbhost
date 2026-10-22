#pragma once

// The effects' frames between whole steps (decomp/sfx/step_gate.cpp):
// engine/frame_rate.cpp says each frame, before the game's effect update,
// whether bbhost put this frame between whole effect steps (and the gate is
// on). The game's own still frame (SprjSfx+0x8c) is told apart by the gate.
void decomp_sfx_gate_frame(bool between);
// Whether the frame is between whole effect steps at all, the gate on or off:
// what GPU particle type 0x2713's record keeps its whole step's time on.
void decomp_sfx_step_frame(bool between);
// The same frame's own time and the carry before and after it
// (engine/sfx_schedule.h's frame, from and into): what the smooth steps' frame
// domain steps the effects by and puts on their nodes' times.
void decomp_sfx_frame_time(float frame, float from, float into);
// The smooth effect steps on or off ([video] effect_steps = "smooth", the default): an
// instance whose classes are all on the gate's converted list steps what it
// counts in whole 1/30 s and what it integrates by the frame's own time.
void decomp_sfx_smooth(bool on);
// Whether the gate's functions took the game's place.
bool decomp_sfx_gate_placed();
