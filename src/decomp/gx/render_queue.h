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

// A segment of a plugin's own in the scene (on_scene_category, plugin API
// 15): fn(command context, user) on the GX worker that records the first
// chunk of the scene at or after `pass` - a sort key's category and
// sub-category, (bits 22-27 << 6) | bits 16-21 - in a command context of its
// own with that chunk's pass set up, queued under the chunk's key just ahead
// of it. pass < 0 (or fn null) takes it away.
void render_queue_scene_insert(int pass, void (*fn)(std::uint64_t cc, void* user), void* user);
