// Gnm's constant-engine writers as our source (decomp/gx/constant_engine.cpp,
// docs/decomp.md): what the command processor's setup tells them.
#pragma once

// hle_gx_trace_arm, before any of the game's code runs: whether the
// constant engine's packets are written (false: the constant RAM writes and
// the two counters write nothing - nothing dumps that RAM).
void gnm_ce_configure(bool dead);
