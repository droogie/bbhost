#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>
#include <atomic>
#include <thread>
#include <iostream>

namespace {
#include "ngx_transfer_sync.h"

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::cerr << "FAIL: " << msg << " (" << __LINE__ << ")\n"; ++g_failures; } } while (0)

struct BarrierCall {
    VkCommandBuffer cmd{};
    VkPipelineStageFlags src{}, dst{};
    VkDependencyFlags flags{};
    std::vector<VkBufferMemoryBarrier> buffers;
    std::vector<VkImageMemoryBarrier> images;
};
struct FillCall {
    VkCommandBuffer cmd{};
    VkBuffer buffer{};
    VkDeviceSize offset{}, size{};
    uint32_t data{};
};

static std::vector<BarrierCall> g_barriers;
static std::vector<FillCall> g_fills;

void VKAPI_CALL mock_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src,
    VkPipelineStageFlags dst, VkDependencyFlags flags, uint32_t,
    const VkMemoryBarrier*, uint32_t buf_cnt,
    const VkBufferMemoryBarrier* bufs, uint32_t img_cnt,
    const VkImageMemoryBarrier* imgs) {
    BarrierCall call{cmd, src, dst, flags, {}, {}};
    if (buf_cnt && bufs) call.buffers.assign(bufs, bufs + buf_cnt);
    if (img_cnt && imgs) call.images.assign(imgs, imgs + img_cnt);
    g_barriers.push_back(call);
}

void VKAPI_CALL mock_fill(VkCommandBuffer cmd, VkBuffer buf,
    VkDeviceSize offset, VkDeviceSize size, uint32_t data) {
    g_fills.push_back({cmd, buf, offset, size, data});
}

PFN_vkVoidFunction VKAPI_CALL mock_gdpa(VkDevice, const char* name) {
    if (!name) return nullptr;
    if (!std::strcmp(name, "vkCmdPipelineBarrier"))
        return reinterpret_cast<PFN_vkVoidFunction>(mock_barrier);
    if (!std::strcmp(name, "vkCmdFillBuffer"))
        return reinterpret_cast<PFN_vkVoidFunction>(mock_fill);
    if (!std::strcmp(name, "missing_proc"))
        return nullptr;
    return reinterpret_cast<PFN_vkVoidFunction>(0x1234);
}

PFN_vkVoidFunction VKAPI_CALL mock_gipa(VkInstance, const char* name) {
    if (name && !std::strcmp(name, "vkGetDeviceProcAddr"))
        return reinterpret_cast<PFN_vkVoidFunction>(mock_gdpa);
    return mock_gdpa(VK_NULL_HANDLE, name);
}

void reset(bool enabled = true) {
    g_barriers.clear();
    g_fills.clear();
    g_transfer_sync = {mock_gipa, mock_gdpa, mock_barrier, mock_fill,
                       reinterpret_cast<VkDevice>(0x1000), enabled};
    g_transfer_cmd = VK_NULL_HANDLE;
}

void test_forwarding_outside_scope() {
    reset();
    VkCommandBuffer cmd = reinterpret_cast<VkCommandBuffer>(0x2001);
    VkImageMemoryBarrier imb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    transfer_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                     0, nullptr, 0, nullptr, 1, &imb);
    transfer_fill(cmd, reinterpret_cast<VkBuffer>(0x3001), 16, 64, 0x11);

    CHECK(g_barriers.size() == 1, "outside scope barrier count");
    if (!g_barriers.empty()) {
        CHECK(g_barriers[0].src == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "src unmodified");
        CHECK(g_barriers[0].dst == VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, "dst unmodified");
        CHECK(g_barriers[0].images.size() == 1 &&
              g_barriers[0].images[0].srcAccessMask == VK_ACCESS_SHADER_READ_BIT, "img access unmodified");
    }
    CHECK(g_fills.size() == 1 && g_fills[0].offset == 16, "outside scope fill forwarded");
}

void test_scoped_image_barrier() {
    reset();
    VkCommandBuffer cmd = reinterpret_cast<VkCommandBuffer>(0x2002);
    int dummy_pnext = 42;
    VkImageMemoryBarrier imb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imb.pNext = &dummy_pnext;
    imb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imb.srcQueueFamilyIndex = 3;
    imb.dstQueueFamilyIndex = 4;
    imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 2, 3, 4, 5};

    {
        TransferScope scope(cmd);
        transfer_barrier(cmd, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_DEPENDENCY_BY_REGION_BIT,
                         0, nullptr, 0, nullptr, 1, &imb);
    }

    CHECK(g_barriers.size() == 1, "scoped barrier count");
    if (!g_barriers.empty()) {
        const auto& b = g_barriers[0];
        CHECK(b.src == (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT), "src extended");
        CHECK(b.dst == (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT), "dst extended");
        CHECK(b.flags == VK_DEPENDENCY_BY_REGION_BIT, "flags preserved");
        CHECK(b.images.size() == 1, "image barrier count");
        if (!b.images.empty()) {
            const auto& img = b.images[0];
            CHECK(img.pNext == &dummy_pnext, "pNext preserved");
            CHECK(img.oldLayout == VK_IMAGE_LAYOUT_GENERAL, "layout preserved");
            CHECK(img.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, "layout preserved");
            CHECK(img.subresourceRange.baseMipLevel == 2, "range preserved");
            CHECK(img.subresourceRange.levelCount == 3 && img.subresourceRange.baseArrayLayer == 4 &&
                  img.subresourceRange.layerCount == 5, "full subresource range preserved");
            CHECK(img.srcQueueFamilyIndex == 3 && img.dstQueueFamilyIndex == 4, "queue families preserved");
            CHECK(img.srcAccessMask == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT), "src access extended");
            CHECK(img.dstAccessMask == (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT), "dst access extended");
        }
    }
    CHECK(imb.srcAccessMask == VK_ACCESS_SHADER_READ_BIT && imb.dstAccessMask == VK_ACCESS_SHADER_WRITE_BIT,
          "caller barrier remains immutable");

    g_barriers.clear();
    {
        TransferScope scope(cmd);
        transfer_barrier(reinterpret_cast<VkCommandBuffer>(0x2012), VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imb);
    }
    CHECK(g_barriers.size() == 1 && g_barriers[0].src == VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
          "another command buffer is forwarded during evaluation");
}

void test_scoped_fill() {
    reset();
    VkCommandBuffer cmd = reinterpret_cast<VkCommandBuffer>(0x2003);
    VkBuffer buf = reinterpret_cast<VkBuffer>(0x3003);
    {
        TransferScope scope(cmd);
        transfer_fill(cmd, buf, 128, 512, 0xbeef);
    }
    CHECK(g_barriers.size() == 1, "scoped fill emits barrier");
    if (!g_barriers.empty()) {
        const auto& b = g_barriers[0];
        CHECK(b.src == VK_PIPELINE_STAGE_TRANSFER_BIT && b.dst == VK_PIPELINE_STAGE_TRANSFER_BIT, "fill stage");
        CHECK(b.buffers.size() == 1, "fill buffer barrier count");
        if (!b.buffers.empty()) {
            CHECK(b.buffers[0].buffer == buf, "buffer handle");
            CHECK(b.buffers[0].offset == 128 && b.buffers[0].size == 512, "precise buffer range");
            CHECK(b.buffers[0].srcAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT, "fill access");
        }
    }
    CHECK(g_fills.size() == 1 && g_fills[0].data == 0xbeef, "fill captured");
}

void test_disabled_scope() {
    reset(false);
    VkCommandBuffer cmd = reinterpret_cast<VkCommandBuffer>(0x2004);
    VkImageMemoryBarrier imb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    {
        TransferScope scope(cmd);
        CHECK(g_transfer_cmd == VK_NULL_HANDLE, "disabled scope sets null cmd");
        transfer_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &imb);
        transfer_fill(cmd, reinterpret_cast<VkBuffer>(0x3004), 0, 32, 0);
    }
    CHECK(g_barriers.size() == 1 && g_barriers[0].src == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "disabled unmodified");
    auto p = wrap_transfer_proc("vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction>(mock_barrier));
    CHECK(p == reinterpret_cast<PFN_vkVoidFunction>(mock_barrier), "wrap proc when disabled returns original");
}

void test_nested_scope() {
    reset();
    VkCommandBuffer c1 = reinterpret_cast<VkCommandBuffer>(0x11);
    VkCommandBuffer c2 = reinterpret_cast<VkCommandBuffer>(0x22);
    {
        TransferScope s1(c1);
        CHECK(g_transfer_cmd == c1, "outer set");
        {
            TransferScope s2(c2);
            CHECK(g_transfer_cmd == c2, "inner set");
        }
        CHECK(g_transfer_cmd == c1, "outer restored");
    }
    CHECK(g_transfer_cmd == VK_NULL_HANDLE, "null restored");
}

void test_dispatch_resolution() {
    reset();
    CHECK(wrap_transfer_proc("vkCmdPipelineBarrier", nullptr) == nullptr, "missing proc null");
    CHECK(transfer_gdpa(g_transfer_sync.device, "missing_proc") == nullptr, "gdpa missing proc null");

    VkDevice other_dev = reinterpret_cast<VkDevice>(0x9999);
    auto our_p = transfer_gdpa(g_transfer_sync.device, "vkCmdPipelineBarrier");
    auto other_p = transfer_gdpa(other_dev, "vkCmdPipelineBarrier");
    CHECK(our_p == reinterpret_cast<PFN_vkVoidFunction>(transfer_barrier), "our device wrapped");
    CHECK(other_p == reinterpret_cast<PFN_vkVoidFunction>(mock_barrier), "other device forwarded unchanged");

    auto gipa_gdpa = transfer_gipa(VK_NULL_HANDLE, "vkGetDeviceProcAddr");
    CHECK(gipa_gdpa == reinterpret_cast<PFN_vkVoidFunction>(transfer_gdpa), "gipa returns transfer_gdpa");
}

void test_thread_isolation() {
    reset();
    VkCommandBuffer main_cmd = reinterpret_cast<VkCommandBuffer>(0x5555);
    TransferScope main_scope(main_cmd);

    std::atomic<bool> worker_done{false};
    std::thread t([&]() {
        CHECK(g_transfer_cmd == VK_NULL_HANDLE, "worker sees null initially");
        VkCommandBuffer worker_cmd = reinterpret_cast<VkCommandBuffer>(0x6666);
        {
            TransferScope worker_scope(worker_cmd);
            CHECK(g_transfer_cmd == worker_cmd, "worker sees own cmd");
        }
        CHECK(g_transfer_cmd == VK_NULL_HANDLE, "worker restores null");
        worker_done.store(true);
    });
    t.join();
    CHECK(worker_done.load(), "worker finished");
    CHECK(g_transfer_cmd == main_cmd, "main thread unchanged");
}

} // namespace

int main() {
    test_forwarding_outside_scope();
    test_scoped_image_barrier();
    test_scoped_fill();
    test_disabled_scope();
    test_nested_scope();
    test_dispatch_resolution();
    test_thread_isolation();

    if (g_failures != 0) {
        std::cerr << g_failures << " test failure(s)\n";
        return 1;
    }
    std::cout << "All ngx_transfer_sync tests passed\n";
    return 0;
}
