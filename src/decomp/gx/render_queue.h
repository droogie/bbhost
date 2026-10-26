// The render queue's census (decomp/gx/render_queue.cpp, BBHOST_RQ_CENSUS=1).
#pragma once

#include <cstdint>

// Each flip the game submits (frame_stats_on_flip): the flips so far
// (hle_video_flip_count), which the census's records and lines are cut by.
void render_queue_census_flip(std::uint32_t flips);
