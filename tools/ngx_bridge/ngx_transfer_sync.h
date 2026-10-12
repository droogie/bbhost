// NGX 310.9.1 records shader-only image barriers around transfers inside
// DLSS-G evaluation. Supply the SDK's documented Vulkan dispatch callbacks
// to complete those dependencies without altering NVIDIA's binaries.
// Included inside ngx_bridge.cpp's anonymous namespace.
struct NgxTransferSync {
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;
    PFN_vkCmdPipelineBarrier barrier = nullptr;
    PFN_vkCmdFillBuffer fill = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    bool enabled = true;
};
NgxTransferSync g_transfer_sync;
thread_local VkCommandBuffer g_transfer_cmd = VK_NULL_HANDLE;

struct TransferScope {
    VkCommandBuffer previous = g_transfer_cmd;
    explicit TransferScope(VkCommandBuffer cmd) {
        g_transfer_cmd = g_transfer_sync.enabled ? cmd : VK_NULL_HANDLE;
    }
    ~TransferScope() { g_transfer_cmd = previous; }
};

void VKAPI_CALL transfer_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src,
    VkPipelineStageFlags dst, VkDependencyFlags flags, uint32_t memory_count,
    const VkMemoryBarrier* memory, uint32_t buffer_count,
    const VkBufferMemoryBarrier* buffers, uint32_t image_count,
    const VkImageMemoryBarrier* images) {
    if (cmd == g_transfer_cmd && image_count) {
        // Preserve the SDK's layouts, subresource ranges, pNext chains and
        // dependencies. Only extend the image access/stage scopes for clears
        // and copies; do not add a separate all-commands barrier per transfer.
        std::vector<VkImageMemoryBarrier> complete(images, images + image_count);
        for (auto& image : complete) {
            image.srcAccessMask |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            image.dstAccessMask |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        g_transfer_sync.barrier(cmd, src | VK_PIPELINE_STAGE_TRANSFER_BIT,
            dst | VK_PIPELINE_STAGE_TRANSFER_BIT, flags, memory_count, memory,
            buffer_count, buffers, image_count, complete.data());
    } else {
        g_transfer_sync.barrier(cmd, src, dst, flags, memory_count, memory,
            buffer_count, buffers, image_count, images);
    }
}

void VKAPI_CALL transfer_fill(VkCommandBuffer cmd, VkBuffer buffer,
    VkDeviceSize offset, VkDeviceSize size, uint32_t data) {
    if (cmd == g_transfer_cmd) {
        // The runtime also issues overlapping fills without an intervening
        // dependency. Scope this dependency to the actual destination range.
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        b.srcAccessMask = b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = buffer;
        b.offset = offset;
        b.size = size;
        g_transfer_sync.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
    }
    g_transfer_sync.fill(cmd, buffer, offset, size, data);
}

PFN_vkVoidFunction wrap_transfer_proc(const char* name, PFN_vkVoidFunction original) {
    if (!original || !name || !g_transfer_sync.enabled) return original;
    if (!std::strcmp(name, "vkCmdPipelineBarrier"))
        return reinterpret_cast<PFN_vkVoidFunction>(transfer_barrier);
    if (!std::strcmp(name, "vkCmdFillBuffer"))
        return reinterpret_cast<PFN_vkVoidFunction>(transfer_fill);
    return original;
}

PFN_vkVoidFunction VKAPI_CALL transfer_gdpa(VkDevice device, const char* name) {
    auto original = g_transfer_sync.gdpa(device, name);
    return device == g_transfer_sync.device ? wrap_transfer_proc(name, original) : original;
}

PFN_vkVoidFunction VKAPI_CALL transfer_gipa(VkInstance instance, const char* name) {
    auto original = g_transfer_sync.gipa(instance, name);
    if (original && name && !std::strcmp(name, "vkGetDeviceProcAddr"))
        return reinterpret_cast<PFN_vkVoidFunction>(transfer_gdpa);
    return wrap_transfer_proc(name, original);
}
