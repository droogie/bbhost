// The scene's render queue (decomp/gx/render_queue.cpp): its cut
// (BBHOST_RQ_SPLIT) and its census (BBHOST_RQ_CENSUS=1).
#pragma once

#include <cstdint>

// Each flip the game submits (frame_stats_on_flip): the flips so far
// (hle_video_flip_count), which the census's records and lines are cut by,
// and the frame rate's limit (frame_rate_limit_fps), which the cut follows
// from the next partition on unless BBHOST_RQ_SPLIT says: on above 60 fps,
// the game's partitions at 60 and below.
void render_queue_flip(std::uint32_t flips, float limit_fps);
