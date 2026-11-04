// The GX layer's block reclaim as ours (decomp/gx/block_reclaim.cpp): what
// the discard rename's wait (decomp/gx/rename_wait.cpp) asks of it.
#pragma once

#include <cstdint>

// Walks this thread has made from the discard rename's empty path - sub_2ad0d30
// found no room for a renamed copy and asked for the pool's done blocks before
// trying again - and from any thread.
std::uint64_t decomp_gx_rename_reclaims_here();
std::uint64_t decomp_gx_rename_reclaims();
// One more of the GPU's submissions run (its labels landed): the oldest
// blocks come free first, and the GPU keeps the rest of what it was handed to
// run meanwhile - a wait for everything recorded (a flush) drained it and left
// it idle at each one. False when none handed to the GPU is still to finish
// (the recording under way is not one: that takes a flush), or after 50 ms.
bool decomp_gx_gpu_step();
