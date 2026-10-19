// Gnm's constant-engine writers as our source (decomp/gx/constant_engine.cpp,
// docs/decomp.md): what the command processor's setup tells them.
#pragma once

// hle_gx_trace_arm, before any of the game's code runs: whether the
// constant engine's counter step is written (true: it writes nothing -
// nothing dumps that RAM; its wait, RAM writes and dumps are the game's
// packets in every mode).
void gnm_ce_configure(bool dead);
