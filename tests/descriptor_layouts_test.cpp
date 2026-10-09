// host/descriptor_layouts.h against device limits, without a device: per
// stage within MoltenVK's, unchanged on devices the old layouts fit.
#include "host/descriptor_layouts.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

void expect_within(const std::vector<gpu::LayoutSet>& sets, const VkPhysicalDeviceLimits& limits, const std::string& what) {
    const std::string over = gpu::layout_over_limits(sets, limits);
    expect(over.empty(), what + ": " + over);
}

// MoltenVK's (vulkaninfo, M1 Pro): Metal's 16 samplers and 31 buffers a stage.
VkPhysicalDeviceLimits moltenvk_limits() {
    VkPhysicalDeviceLimits l{};
    l.maxPerStageDescriptorSamplers = 16;
    l.maxPerStageDescriptorUniformBuffers = 31;
    l.maxPerStageDescriptorStorageBuffers = 31;
    l.maxPerStageDescriptorSampledImages = 256;
    l.maxPerStageDescriptorStorageImages = 256;
    l.maxPerStageResources = 287;
    l.maxDescriptorSetSamplers = 80;
    l.maxDescriptorSetUniformBuffers = 155;
    l.maxDescriptorSetUniformBuffersDynamic = 155;
    l.maxDescriptorSetStorageBuffers = 155;
    l.maxDescriptorSetStorageBuffersDynamic = 155;
    l.maxDescriptorSetSampledImages = 1280;
    l.maxDescriptorSetStorageImages = 1280;
    return l;
}

// A desktop driver's (NVIDIA's on Linux): limits far past what bbhost binds.
VkPhysicalDeviceLimits desktop_limits() {
    VkPhysicalDeviceLimits l{};
    l.maxPerStageDescriptorSamplers = l.maxPerStageDescriptorUniformBuffers = l.maxPerStageDescriptorStorageBuffers = 1048576;
    l.maxPerStageDescriptorSampledImages = l.maxPerStageDescriptorStorageImages = 1048576;
    l.maxPerStageResources = 4294967295u;
    l.maxDescriptorSetSamplers = l.maxDescriptorSetUniformBuffers = l.maxDescriptorSetStorageBuffers = 1048576;
    l.maxDescriptorSetUniformBuffersDynamic = 15;
    l.maxDescriptorSetStorageBuffersDynamic = 16;
    l.maxDescriptorSetSampledImages = l.maxDescriptorSetStorageImages = 1048576;
    return l;
}

// Just large enough for the old layouts.
VkPhysicalDeviceLimits exact_limits() {
    VkPhysicalDeviceLimits l{};
    l.maxPerStageDescriptorSamplers = 64;
    l.maxPerStageDescriptorUniformBuffers = 2;
    l.maxPerStageDescriptorStorageBuffers = 32;
    l.maxPerStageDescriptorSampledImages = l.maxPerStageDescriptorStorageImages = 64;
    l.maxPerStageResources = 2 + 64 + 64 + 32;
    l.maxDescriptorSetSamplers = 64;
    l.maxDescriptorSetUniformBuffers = l.maxDescriptorSetUniformBuffersDynamic = 2;
    l.maxDescriptorSetStorageBuffers = 32;
    l.maxDescriptorSetStorageBuffersDynamic = 0;
    l.maxDescriptorSetSampledImages = l.maxDescriptorSetStorageImages = 64;
    return l;
}

// The most a translated stage binds in its own set; images numbered as the
// translator does, one index across sampled and storage.
gcn::TranslateResult busiest_stage(std::uint32_t samplers, std::uint32_t cb_base, bool bindless) {
    gcn::TranslateResult r;
    r.bindless = bindless;
    for (std::uint32_t k = 0; k < gpu::kMaxImages; ++k) {
        gcn::ImageBinding b;
        b.storage = k >= gpu::kMaxImages / 2;
        b.binding = (b.storage ? gcn::kBindingStorageImage0 : gcn::kBindingImage0) + k;
        r.images.push_back(b);
    }
    for (std::uint32_t k = 0; k < samplers; ++k) {
        gcn::SamplerBinding b;
        b.binding = gcn::kBindingSampler0 + k;
        r.samplers.push_back(b);
    }
    for (std::uint32_t k = 0; k < gcn::kMaxBuffers; ++k) {
        gcn::BufferBinding b;
        b.binding = cb_base + k;
        r.buffers.push_back(b);
    }
    return r;
}

// Every binding of `binds` read by `stages`.
bool all_read_by(const std::vector<VkDescriptorSetLayoutBinding>& binds, VkShaderStageFlags stages) {
    for (const VkDescriptorSetLayoutBinding& b : binds) {
        if (b.stageFlags != stages) return false;
    }
    return !binds.empty();
}

// The params block is a dynamic uniform buffer with the set cache, a plain
// one without; the constant buffers are pushed (set 2) only with it.
constexpr VkDescriptorType kParams[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

}  // namespace

int main() {
    using Binds = std::vector<VkDescriptorSetLayoutBinding>;

    // MoltenVK: the old layouts are over, so they go per stage.
    const VkPhysicalDeviceLimits mvk = moltenvk_limits();
    const std::uint32_t samplers = gpu::shared_set_samplers(true, mvk.maxPerStageDescriptorSamplers);
    expect(samplers == 16, "MoltenVK's shared sets declare 16 samplers");
    for (const VkDescriptorType params : kParams) {
        const bool push = params == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        const std::string mode = push ? " (constant buffers pushed)" : " (constant buffers in the stage sets)";
        const std::string old_over = gpu::old_layouts_over_limits(params, push, mvk);
        expect(old_over.find("fragment: 64 samplers against 16") != std::string::npos, "the old layouts are over MoltenVK's" + mode);
        if (push) {
            expect(old_over.find("vertex: 32 storage buffers against 31") != std::string::npos,
                   "the old pushed buffers are over MoltenVK's" + mode);
        }
        const Binds compute = gpu::shared_set_bindings(params, samplers, !push, VK_SHADER_STAGE_ALL);
        expect_within({{&compute}}, mvk, "the compute layout" + mode);
        const Binds cbs = push ? gpu::cb_push_bindings(true) : Binds{};
        const Binds gfx[2] = {gpu::shared_set_bindings(params, samplers, !push, gpu::gfx_set_stages(true, 0)),
                              gpu::shared_set_bindings(params, samplers, !push, gpu::gfx_set_stages(true, 1))};
        expect_within({{&gfx[0]}, {&gfx[1]}, {&cbs}}, mvk, "the graphics layout" + mode);
        // Bindless: the stage sets hold the params block alone; the global
        // set is update-after-bind, under limits of its own.
        const Binds params_only[2] = {{{gcn::kBindingParams, params, 1, gpu::kGfxSetStages[0], nullptr}},
                                      {{gcn::kBindingParams, params, 1, gpu::kGfxSetStages[1], nullptr}}};
        const Binds global = {{gcn::kBindlessImages, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u << 18, VK_SHADER_STAGE_ALL, nullptr},
                              {gcn::kBindlessSamplers, VK_DESCRIPTOR_TYPE_SAMPLER, 1024, VK_SHADER_STAGE_ALL, nullptr}};
        expect_within({{&params_only[0]}, {&params_only[1]}, {&cbs}, {&global, true}}, mvk, "the bindless graphics layout" + mode);
        // A pipeline's own sets (BBHOST_SET_LAYOUTS), both stages binding the
        // most they can.
        for (const bool bindless : {false, true}) {
            const Binds vs = gpu::stage_set_bindings(busiest_stage(samplers, gcn::kBindingStorageBuffer0, bindless), params, !push,
                                                     gpu::gfx_set_stages(true, 0));
            const Binds ps = gpu::stage_set_bindings(
                busiest_stage(samplers, gcn::kBindingStorageBuffer0 + (push ? gcn::kMaxBuffers : 0), bindless), params, !push,
                gpu::gfx_set_stages(true, 1));
            expect(vs.size() == 1 + (bindless ? 0 : gpu::kMaxImages + samplers) + (push ? 0 : gcn::kMaxBuffers),
                   "a vertex stage's set has its params, images, samplers and unpushed buffers" + mode);
            expect_within({{&vs}, {&ps}, {&cbs}, {bindless ? &global : nullptr, true}}, mvk,
                          std::string("a pipeline's own layout") + (bindless ? ", bindless" : "") + mode);
        }
        // Every pushed buffer is read by exactly one of the two stage groups.
        bool split = !push || cbs.size() == 2 * gcn::kMaxBuffers;
        for (std::size_t k = 0; k < cbs.size(); ++k) {
            split = split && cbs[k].binding == gcn::kBindingStorageBuffer0 + k &&
                    cbs[k].stageFlags == gpu::kCbPushStages[k < gcn::kMaxBuffers ? 0 : 1];
        }
        expect(split, "per stage, the vertex stages read bindings 112-127 of set 2, the fragment stage 128-143" + mode);
        // MoltenVK puts a control stage's own buffers (tessellation levels)
        // above its bindings and the 8 argument buffers of its sets.
        std::uint32_t tcs_buffers = 0;
        for (const VkDescriptorSetLayoutBinding& b : cbs) {
            if (b.stageFlags & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) tcs_buffers += b.descriptorCount;
        }
        expect(tcs_buffers <= mvk.maxPerStageDescriptorStorageBuffers - 8 - 8,
               "a tessellation control stage leaves MoltenVK room for its tessellation levels buffer" + mode);
        expect(gfx[0][0].binding == gcn::kBindingParams && (gfx[0][0].stageFlags & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) &&
                   (params_only[0][0].stageFlags & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT),
               "the hull shader as a control stage still reads set 0's params block" + mode);
        expect((gpu::kCbPushStages[0] | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) == gpu::kGfxSetStages[0],
               "the vertex half of set 2 is read by set 0's stages but the control stage" + mode);
    }
    // One sampler short of the old layouts' 64 a stage is enough to go per stage.
    {
        VkPhysicalDeviceLimits l = exact_limits();
        l.maxPerStageDescriptorSamplers = 63;
        expect(!gpu::old_layouts_over_limits(kParams[0], true, l).empty(), "63 samplers a stage takes the layouts per stage");
        l = exact_limits();
        l.maxPerStageDescriptorStorageBuffers = 31;
        expect(!gpu::old_layouts_over_limits(kParams[0], true, l).empty(), "31 storage buffers a stage takes the layouts per stage");
    }

    // Devices the old layouts fit keep them: every binding in every stage.
    const VkPhysicalDeviceLimits fits[2] = {desktop_limits(), exact_limits()};
    for (int d = 0; d < 2; ++d) {
        const VkPhysicalDeviceLimits& l = fits[d];
        const std::string device = d == 0 ? " on a desktop device" : " on a device just large enough";
        for (const VkDescriptorType params : kParams) {
            const bool push = params == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            const std::string old_over = gpu::old_layouts_over_limits(params, push, l);
            expect(old_over.empty(), "the old layouts fit" + device + ": " + old_over);
            const std::uint32_t n = gpu::shared_set_samplers(false, l.maxPerStageDescriptorSamplers);
            expect(n == gpu::kMaxImages, "32 samplers a set" + device);
            const Binds shared = gpu::shared_set_bindings(params, n, !push, VK_SHADER_STAGE_ALL);
            expect(shared.size() == 1 + 3 * gpu::kMaxImages + (push ? 0 : gcn::kMaxBuffers), "the shared set's bindings" + device);
            const Binds gfx[2] = {gpu::shared_set_bindings(params, n, !push, gpu::gfx_set_stages(false, 0)),
                                  gpu::shared_set_bindings(params, n, !push, gpu::gfx_set_stages(false, 1))};
            expect(all_read_by(gfx[0], VK_SHADER_STAGE_ALL) && all_read_by(gfx[1], VK_SHADER_STAGE_ALL),
                   "the graphics sets in every stage" + device);
            const Binds cbs = push ? gpu::cb_push_bindings(false) : Binds{};
            expect(!push || (cbs.size() == 2 * gcn::kMaxBuffers && all_read_by(cbs, VK_SHADER_STAGE_ALL)),
                   "the pushed buffers in every stage" + device);
            const Binds vs = gpu::stage_set_bindings(busiest_stage(n, gcn::kBindingStorageBuffer0, false), params, !push,
                                                     gpu::gfx_set_stages(false, 0));
            expect(all_read_by(vs, VK_SHADER_STAGE_ALL), "a pipeline's own sets in every stage" + device);
            expect_within({{&shared}}, l, "the compute layout" + device);
            expect_within({{&gfx[0]}, {&gfx[1]}, {&cbs}}, l, "the graphics layout" + device);
        }
    }

    if (g_failures) return 1;
    std::printf("descriptor_layouts_test: per stage within MoltenVK's limits; as they were on devices they fit\n");
    return 0;
}
