// SPDX-License-Identifier: GPL-2.0-or-later
// bbhost: GPU object motion vector resource and history tracking module.

#include "host/object_motion.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/config.h"
#include "host/gpu_internal.h"
#include "host/options.h"
#include "log.h"

namespace gpu {

thread_local bool t_object_motion_draw = false;
thread_local uint32_t t_object_motion_location = 0;

namespace {

struct ObjectTarget {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t depth_base = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool frame_cleared = false;
    bool written_this_frame = false;
};

bool s_initialized = false;
bool s_failed = false;
DevBuffer s_positions_buf{};
motion::History s_history{motion::PositionsPerFrame};
motion::IndexRangeCache s_index_cache{};
ObjectTarget s_target{};
uint64_t s_frame_counter = 0;

void destroy_target(ObjectTarget& t) {
    if (t.view) {
        vkDestroyImageView(g.device, t.view, nullptr);
        t.view = VK_NULL_HANDLE;
    }
    if (t.image) {
        vkDestroyImage(g.device, t.image, nullptr);
        t.image = VK_NULL_HANDLE;
    }
    if (t.memory) {
        vkFreeMemory(g.device, t.memory, nullptr);
        t.memory = VK_NULL_HANDLE;
    }
    t.width = 0;
    t.height = 0;
    t.depth_base = 0;
    t.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    t.frame_cleared = false;
    t.written_this_frame = false;
}

bool create_target_image(ObjectTarget& t, uint32_t w, uint32_t h, uint64_t depth_base) {
    destroy_target(t);
    if (!w || !h) {
        return false;
    }

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(g.device, &ici, nullptr, &t.image) != VK_SUCCESS) {
        host_log("object_motion: failed to create RGBA32F motion target image (%ux%u)", w, h);
        return false;
    }

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, t.image, &req);

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX ||
        vkAllocateMemory(g.device, &mai, nullptr, &t.memory) != VK_SUCCESS) {
        vkDestroyImage(g.device, t.image, nullptr);
        t.image = VK_NULL_HANDLE;
        host_log("object_motion: failed to allocate memory for motion target");
        return false;
    }

    if (vkBindImageMemory(g.device, t.image, t.memory, 0) != VK_SUCCESS) {
        destroy_target(t);
        return false;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(g.device, &vci, nullptr, &t.view) != VK_SUCCESS) {
        vkDestroyImage(g.device, t.image, nullptr);
        vkFreeMemory(g.device, t.memory, nullptr);
        t.image = VK_NULL_HANDLE;
        t.memory = VK_NULL_HANDLE;
        host_log("object_motion: failed to create image view for motion target");
        return false;
    }

    t.width = w;
    t.height = h;
    t.depth_base = depth_base;
    t.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    t.frame_cleared = false;
    t.written_this_frame = false;
    return true;
}

void run_readback_diagnostic(VkCommandBuffer cmd, const ObjectTarget& t) {
    static const bool readback_enabled = []() {
        const char* env = std::getenv("BBHOST_OBJECT_MOTION_READBACK");
        return env && (env[0] == '1' || env[0] == 't' || env[0] == 'T');
    }();
    if (!readback_enabled || !t.image || !t.width || !t.height) {
        return;
    }

    static uint64_t diag_counter = 0;
    if (++diag_counter % 300 != 0) {
        return;
    }

    const VkDeviceSize bytes = static_cast<VkDeviceSize>(t.width) * t.height * 16;
    DevBuffer staging{};
    if (!create_dev_buffer(staging, bytes, /*host_visible=*/true, /*cached=*/true)) {
        return;
    }

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = t.width;
    region.bufferImageHeight = t.height;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {t.width, t.height, 1};

    VkMemoryBarrier read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    read.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &read, 0, nullptr, 0, nullptr);
    vkCmdCopyImageToBuffer(cmd, t.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &region);

    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = staging.buffer;
    bmb.offset = 0;
    bmb.size = bytes;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &bmb, 0, nullptr);

    // Flush and wait for execution on GPU so diagnostic can inspect host-visible data
    flush_locked();

    if (staging.map) {
        const float* pixels = static_cast<const float*>(staging.map);
        uint32_t valid_count = 0;
        uint32_t nonzero_count = 0;
        float max_motion = 0.0f;
        const size_t total_pixels = static_cast<size_t>(t.width) * t.height;

        for (size_t i = 0; i < total_pixels; ++i) {
            const float r = pixels[i * 4 + 0];
            const float g_val = pixels[i * 4 + 1];
            const float b = pixels[i * 4 + 2];
            if (b > 0.99f) {
                ++valid_count;
                if (!std::isnan(r) && !std::isnan(g_val) && !std::isinf(r) && !std::isinf(g_val)) {
                    const float mag = std::sqrt(r * r + g_val * g_val);
                    if (mag > 0.001f) {
                        ++nonzero_count;
                    }
                    max_motion = std::max(max_motion, mag);
                }
            }
        }
        host_log("object_motion: readback diag (frame %llu, %ux%u): valid %u (%.2f%%), nonzero %u, max_motion %.2f px",
                 static_cast<unsigned long long>(s_frame_counter), t.width, t.height,
                 valid_count, (100.0f * valid_count) / static_cast<float>(total_pixels),
                 nonzero_count, max_motion);
    }

    if (staging.map) vkUnmapMemory(g.device, staging.memory);
    if (staging.buffer) vkDestroyBuffer(g.device, staging.buffer, nullptr);
    if (staging.memory) vkFreeMemory(g.device, staging.memory, nullptr);
    begin_recording_locked();
}

} // namespace

bool object_motion_enabled() {
    static const bool enabled = []() {
        if (const char* env = std::getenv("BBHOST_OBJECT_MOTION")) {
            return env[0] == '1' || env[0] == 't' || env[0] == 'T';
        }
        if (const char* env_old = std::getenv("BB_OBJECT_MOTION")) {
            return env_old[0] == '1' || env_old[0] == 't' || env_old[0] == 'T';
        }
        return host_opt_get("object_motion");
    }();
    return enabled;
}

bool object_motion_init_locked() {
    if (s_initialized) {
        return true;
    }
    if (s_failed || !object_motion_enabled()) {
        return false;
    }
    if (!g.device) {
        return false;
    }

    // Allocate 128 MiB device-local buffer for ping-pong vertex clip positions
    // (1 scratch element + 2 halves of PositionsPerFrame vec4s)
    const uint64_t total_elements = 1ull + 2ull * motion::PositionsPerFrame;
    const uint64_t buffer_size = total_elements * 16ull;

    if (!create_dev_buffer(s_positions_buf, buffer_size, /*host_visible=*/false) || !s_positions_buf.address) {
        host_log("object_motion: failed to allocate %llu MiB device positions buffer",
                 static_cast<unsigned long long>(buffer_size >> 20));
        if (s_positions_buf.buffer) vkDestroyBuffer(g.device, s_positions_buf.buffer, nullptr);
        if (s_positions_buf.memory) vkFreeMemory(g.device, s_positions_buf.memory, nullptr);
        s_positions_buf = {};
        s_failed = true;
        return false;
    }

    s_initialized = true;
    s_frame_counter = 0;
    host_log("object_motion: initialized device buffer BDA 0x%llx (%u vertices/frame half, 128 MiB)",
             static_cast<unsigned long long>(s_positions_buf.address), motion::PositionsPerFrame);
    return true;
}

void object_motion_shutdown_locked() {
    if (!s_initialized) {
        return;
    }
    destroy_target(s_target);
    if (s_positions_buf.buffer) {
        vkDestroyBuffer(g.device, s_positions_buf.buffer, nullptr);
        s_positions_buf.buffer = VK_NULL_HANDLE;
    }
    if (s_positions_buf.memory) {
        vkFreeMemory(g.device, s_positions_buf.memory, nullptr);
        s_positions_buf.memory = VK_NULL_HANDLE;
    }
    s_positions_buf.address = 0;
    s_positions_buf.size = 0;
    s_index_cache.Clear();
    s_initialized = false;
    host_log("object_motion: shutdown completed");
}

bool object_motion_is_enabled() {
    return s_initialized && s_positions_buf.address != 0;
}

uint64_t object_motion_positions_bda() {
    return s_positions_buf.address;
}

motion::History::Allocation object_motion_prepare_draw_locked(const motion::Draw& draw) {
    if (!s_initialized) {
        return {};
    }
    return s_history.Prepare(draw);
}

bool object_motion_is_moving_locked(const motion::History::GateKey& key, uint64_t palette_hash) {
    if (!s_initialized) {
        return false;
    }
    return s_history.Moving(key, palette_hash);
}

motion::History& object_motion_history() {
    return s_history;
}

motion::IndexRangeCache& object_motion_index_cache() {
    return s_index_cache;
}

void object_motion_prepare_target_locked(uint32_t width, uint32_t height, uint64_t depth_base) {
    if (!s_initialized || !width || !height) {
        return;
    }
    if (s_target.image && s_target.width == width && s_target.height == height &&
        s_target.depth_base == depth_base) {
        return;
    }
    render_end_pass_locked();
    if (s_target.image && s_target.width == width && s_target.height == height) {
        s_target.depth_base = depth_base;
        s_target.frame_cleared = s_target.written_this_frame = false;
        s_history.Invalidate();
        return;
    }
    if (s_target.image) flush_locked();
    s_history.Invalidate();
    create_target_image(s_target, width, height, depth_base);
    begin_recording_locked();
}

VkImageView object_motion_get_attachment_view(uint32_t width, uint32_t height) {
    if (!s_initialized || !s_target.image || s_target.width != width || s_target.height != height) {
        return VK_NULL_HANDLE;
    }
    return s_target.view;
}

void object_motion_attach_locked(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint64_t depth_base) {
    if (!s_initialized || !width || !height) {
        return;
    }
    if (!s_target.image || s_target.width != width || s_target.height != height ||
        s_target.depth_base != depth_base) {
        object_motion_prepare_target_locked(width, height, depth_base);
        cmd = g_cmd();
    }
    if (!s_target.view) return;

    if (!s_target.frame_cleared) {
        render_end_pass_locked();
        // Publish a deferred pass end before issuing commands directly.
        cmd = g_cmd();
        if (!cmd) return;
        VkImageMemoryBarrier imb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        imb.oldLayout = s_target.layout;
        imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imb.image = s_target.image;
        imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        imb.srcAccessMask = (s_target.layout == VK_IMAGE_LAYOUT_UNDEFINED)
                                ? 0
                                : (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        imb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &imb);

        VkClearColorValue clear_color{};
        clear_color.float32[0] = 0.0f;
        clear_color.float32[1] = 0.0f;
        clear_color.float32[2] = 0.0f;
        clear_color.float32[3] = 0.0f;

        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, s_target.image, VK_IMAGE_LAYOUT_GENERAL,
                             &clear_color, 1, &range);
        imb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imb.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        imb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imb.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &imb);

        s_target.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        s_target.frame_cleared = true;
        s_target.written_this_frame = true;
    }
}

VkImageView object_motion_read_locked(uint64_t depth_base, uint32_t w, uint32_t h, bool reject_history) {
    if (!s_initialized || !s_target.image || reject_history) {
        return VK_NULL_HANDLE;
    }
    if (s_target.width != w || s_target.height != h) {
        return VK_NULL_HANDLE;
    }
    if (depth_base != 0 && s_target.depth_base != depth_base) {
        return VK_NULL_HANDLE;
    }
    if (!s_target.written_this_frame) {
        return VK_NULL_HANDLE;
    }

    VkCommandBuffer cmd = g_cmd();
    if (!cmd) {
        return VK_NULL_HANDLE;
    }

    // Transition attachment 7 from COLOR_ATTACHMENT_OPTIMAL to GENERAL layout for compute shader sampling
    VkImageMemoryBarrier imb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imb.oldLayout = s_target.layout;
    imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imb.image = s_target.image;
    imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    imb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imb);
    s_target.layout = VK_IMAGE_LAYOUT_GENERAL;

    // Run optional readback diagnostic if requested by BBHOST_OBJECT_MOTION_READBACK=1
    run_readback_diagnostic(cmd, s_target);

    return s_target.view;
}

void object_motion_finish_frame_locked(bool invalidate) {
    if (!s_initialized) {
        return;
    }

    const auto used = s_history.Used();
    if (invalidate) {
        s_history.Invalidate();
        s_target.written_this_frame = false;
    } else {
        s_history.NextFrame();
        ++s_frame_counter;
        if (s_frame_counter % motion::IndexRangeCache::Unused == 0) {
            s_index_cache.Trim(s_history.CurrentFrame());
        }
        if (s_frame_counter % 300 == 0 && s_history.stats.draws > 0) {
            const auto& st = s_history.stats;
            host_log("object_motion: frame %llu stats: draws %llu, stored %llu, loaded %llu, unmatched %llu, exhausted %llu, invalid %llu, still %llu, used %u/%u vertices",
                     static_cast<unsigned long long>(s_frame_counter),
                     static_cast<unsigned long long>(st.draws),
                     static_cast<unsigned long long>(st.stored),
                     static_cast<unsigned long long>(st.loaded),
                     static_cast<unsigned long long>(st.unmatched),
                     static_cast<unsigned long long>(st.exhausted),
                     static_cast<unsigned long long>(st.invalid),
                     static_cast<unsigned long long>(st.still),
                     used, s_history.Capacity());
            s_history.stats = {};
        }
    }

    s_target.frame_cleared = false;
    s_target.written_this_frame = false;

    render_end_pass_locked();
    begin_recording_locked();
    VkCommandBuffer cmd = g_cmd();
    if (cmd) {
        // Vertex-to-vertex memory barrier: ensure current frame stores are visible to next frame loads
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
    }
}

} // namespace gpu
