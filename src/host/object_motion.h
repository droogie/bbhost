// SPDX-License-Identifier: GPL-2.0-or-later
// bbhost: GPU object motion vector resource and history tracking module.

#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>
#include "host/motion_history.h"

namespace gpu {

// Draw-scoped shader variant and attachment selection; reset on all draw exits.
extern thread_local bool t_object_motion_draw;
extern thread_local uint32_t t_object_motion_location;

/// Returns true if object motion vectors are enabled via config / options or environment override.
/// Captured at startup; toggling requires game restart to ensure consistent shader variant selection.
bool object_motion_enabled();

/// Initializes the object motion GPU module under host_gpu_lock().
/// Allocates the 128 MiB device-local ping-pong vertex clip position buffer and initializes history.
bool object_motion_init_locked();

/// Shuts down the module and frees GPU buffers and image resources under host_gpu_lock().
void object_motion_shutdown_locked();

/// Returns true if the module is initialized and ready for draw recording.
bool object_motion_is_enabled();

/// Returns the 64-bit Vulkan buffer device address for the ping-pong vertex clip positions buffer.
/// Element 0 is reserved.
uint64_t object_motion_positions_bda();

/// Allocates position buffer storage in the current frame ping-pong half and retrieves load base
/// from the previous frame for the given draw key.
motion::History::Allocation object_motion_prepare_draw_locked(const motion::Draw& draw);

/// Checks whether a small skeleton (weapon/prop) changed its bone palette since last frame.
bool object_motion_is_moving_locked(const motion::History::GateKey& key, uint64_t palette_hash);

/// Accessors for CPU history and index range cache.
motion::History& object_motion_history();
motion::IndexRangeCache& object_motion_index_cache();

/// Prepares the RGBA32F object motion target matching the current logical render dimensions.
void object_motion_prepare_target_locked(uint32_t width, uint32_t height, uint64_t depth_base);

/// Returns the RGBA32F image view for binding as MRT color attachment 7.
VkImageView object_motion_get_attachment_view(uint32_t width, uint32_t height);

/// Ensures target exists and records clear/transition to COLOR_ATTACHMENT_OPTIMAL once per frame.
void object_motion_attach_locked(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint64_t depth_base);

/// Transitions the object motion target from COLOR_ATTACHMENT_OUTPUT to COMPUTE_SHADER read in GENERAL layout.
/// Returns the readable image view if the target matches depth_base, width, and height without a reset/cut;
/// returns VK_NULL_HANDLE otherwise so compute pass falls back to camera motion.
VkImageView object_motion_read_locked(uint64_t depth_base, uint32_t w, uint32_t h, bool reject_history);

/// Called at every HDR anchor (including early returns).
/// If invalidate is true (camera cut, resolution change, or scene reset), clears history generations.
/// Advances CPU history generation and records a vertex-to-vertex memory barrier on cmd.
void object_motion_finish_frame_locked(bool invalidate);

} // namespace gpu
