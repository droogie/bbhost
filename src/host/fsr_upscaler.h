#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>
#include "host/dlaa.h"
#include "../../tools/ngx_bridge/ngx_bridge.h"

namespace gpu {

// Record FSR upscaling / native AA into cmd under renderer lock (g.mu).
// Uses g.flushes as recording_serial and g.completed_submits as completed_submits.
// Returns false if unsafe or unavailable.
bool fsr_upscale_locked(VkPhysicalDevice phys,
                        VkDevice device,
                        VkCommandBuffer cmd,
                        const NgxbImage& color,
                        const NgxbImage& output,
                        const DlssFgGuides& guides,
                        uint64_t recording_serial,
                        uint64_t completed_submits);

// Retire submissions whose fences have signaled (called on retire_slot_locked).
void fsr_retire_locked(uint64_t completed_submits);

// Tear down upscaler context and free resources.
void fsr_shutdown_locked();

} // namespace gpu
