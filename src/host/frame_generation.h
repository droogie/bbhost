#pragma once

#include <chrono>
#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "host/dlaa.h"
#include "host/fg_timing.h"

namespace host {

struct FgStats {
    std::uint64_t real_presented = 0;
    std::uint64_t generated_presented = 0;
    std::uint64_t frames_skipped = 0;
    std::uint64_t evaluations = 0;
    std::uint64_t evaluations_failed = 0;
    std::uint64_t interpolation_disabled = 0;
};

// Whether Frame Generation (DLSS or FSR3) is enabled by configuration or environment variable.
bool fg_enabled();

// Whether Frame Generation is available on the hardware/driver.
bool fg_available();

// Statistics for telemetry/PresentMon correlation.
FgStats fg_get_stats();

void fg_note_real_presented();
void fg_note_generated_presented();
void fg_note_interpolation_disabled();

// Manages per-frame Vulkan resources and evaluation for DLSS/FSR3 Frame Generation.
class FrameGenerator {
public:
    static constexpr unsigned kFrameSlots = 3;
    FrameGenerator() = default;
    ~FrameGenerator();

    bool init(VkInstance instance, VkPhysicalDevice phys, VkDevice device, std::uint32_t queue_family, VkQueue queue);
    void shutdown();

    // Ensure output images and FG feature match the given dimensions and format.
    // Must be called with a valid command buffer (e.g. presenter's cmd under GPU lock).
    bool ensure_feature(VkCommandBuffer cmd, std::uint32_t width, std::uint32_t height, VkFormat format);

    // Evaluate frame generation into the owned generated image using the provided input and guides.
    // 'color_image', 'depth', 'motion' must be in VK_IMAGE_LAYOUT_GENERAL.
    // Writes generated frame to generated_image(), in VK_IMAGE_LAYOUT_GENERAL.
    // Returns true if evaluation command was recorded successfully.
    bool evaluate(VkCommandBuffer cmd, VkImage color_image, VkImageView color_view, VkFormat color_format,
                  std::uint32_t width, std::uint32_t height, const gpu::DlssFgGuides& guides, unsigned slot = 0);

    // Separate command submit and fence for evaluation. Called under GPU lock.
    bool evaluate_submit(VkImage color_image, VkImageView color_view, VkFormat color_format,
                         std::uint32_t width, std::uint32_t height, const gpu::DlssFgGuides& guides, unsigned slot = 0);
    bool evaluate_submit(VkImage color_image, VkImageView color_view, VkFormat color_format,
                         std::uint32_t width, std::uint32_t height, std::uint32_t full_width, std::uint32_t full_height,
                         const gpu::DlssFgGuides& guides, unsigned slot = 0);

    // Wait for the separate evaluation fence to complete. Never call under GPU lock.
    bool wait_evaluation(std::uint64_t timeout_ns = 1000000000ull, unsigned slot = 0);
    bool evaluation_finished(unsigned slot) const;
    bool slot_available(unsigned slot) const;
    bool evaluation_times(unsigned slot, std::uint64_t* source_ns, std::uint64_t* ready_ns) const;
    VkSemaphore ready_semaphore(unsigned slot) const { return frames_[slot].ready; }
    void note_ready_waited(unsigned slot) { frames_[slot].ready_unconsumed = false; }

    // Check if the most recent evaluation had interpolation disabled (read from staging buffer).
    bool is_interpolation_disabled(unsigned slot = 0);

    // Accessors for owned generated resources.
    VkImage generated_image(unsigned index = 0, unsigned slot = 0) const { return frames_[slot].generated[index].image; }
    VkImageView generated_view(unsigned index = 0, unsigned slot = 0) const { return frames_[slot].generated[index].view; }
    VkFormat generated_format() const { return format_; }
    unsigned generated_count() const { return generated_count_; }

    // Accessors for owned real copy resources (if OutputReal was used).
    VkImage real_image(unsigned slot = 0) const { return frames_[slot].real_copy.image; }
    VkImageView real_view(unsigned slot = 0) const { return frames_[slot].real_copy.view; }

    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }
    bool ready() const { return feature_created_ && !recreate_; }

private:
    struct OwnedImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::uint32_t width = 0, height = 0;
    };

    bool create_image(OwnedImage& im, std::uint32_t w, std::uint32_t h, VkFormat fmt, VkImageUsageFlags usage);
    void destroy_image(OwnedImage& im);
    bool ensure_hud_pipeline();
    void destroy_hud_pipeline();
    bool record_fsr_fg(VkCommandBuffer cmd, std::uint32_t width, std::uint32_t height,
                       const gpu::DlssFgGuides& guides, unsigned slot);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;

    std::uint32_t width_ = 0, height_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    bool feature_created_ = false;
    bool recreate_ = false;
    bool fsr_support_checked_ = false;
    bool fsr_supported_ = false;
    int active_backend_ = 0; // 0: DLSS, 1: FSR3
    FgHistory history_;

    std::uint32_t fsr_max_render_w_ = 0, fsr_max_render_h_ = 0;

    unsigned generated_count_ = 1;
    VkCommandPool eval_pool_ = VK_NULL_HANDLE;
    VkQueryPool times_ = VK_NULL_HANDLE;
    double timestamp_period_ns_ = 0;

    std::uint64_t fsr_frame_id_ = 1;
    void* fsr_fg_context_ = nullptr; // FfxVkFsr3_3_1_6FrameGenerationContext*
    VkDescriptorSetLayout hud_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout hud_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline hud_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool hud_pool_ = VK_NULL_HANDLE;

    struct FrameSlot {
        std::array<OwnedImage, 3> generated{};
        OwnedImage real_copy, input;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore ready = VK_NULL_HANDLE;
        bool ready_unconsumed = false;
        std::uint64_t ticket = 0;
        bool pending = false;
        std::uint64_t fsr_frame_id = 0;
        VkBuffer disable_buf = VK_NULL_HANDLE;
        VkDeviceMemory disable_mem = VK_NULL_HANDLE;
        void* disable_mapped = nullptr;
        OwnedImage hud_mask;
        VkDescriptorSet hud_desc_set = VK_NULL_HANDLE;
    };
    std::array<FrameSlot, kFrameSlots> frames_{};
};

// Global instance helper used by the presenter in window.cpp
FrameGenerator& fg_get();

}  // namespace host
