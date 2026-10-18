// Gnm's DMA_DATA writer as our source (decomp/gx/dma.cpp, docs/decomp.md):
// what the command processor's setup tells it, and what the census asks it.
#pragma once

#include <cstdint>

// hle_gx_trace_arm: whether the two moves through GDS around a counting
// dispatch are written (false: nothing - the counters ride in the dispatch
// tokens), and whether the CP census wants each packet's caller. Before any
// of the game's code runs.
void gnm_dma_configure(bool gds_moves_dead, bool census);

// Whether the writer is ours (on the decomp list and placed), after
// decomp_install.
bool gnm_dma_placed();

// BBHOST_CP_CENSUS: the caller (a Binary Ninja address) of the DMA_DATA
// packet at `packet_va`, 0 when it is not one of the last few thousand.
std::uint64_t gnm_dma_caller(std::uint64_t packet_va);
