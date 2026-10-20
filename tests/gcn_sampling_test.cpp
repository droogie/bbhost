// Execute synthetic GCN on Vulkan: check sampled and stored values, not just
// SPIR-V validity. No game files are needed. Exit 77 if no suitable Vulkan
// device is available.
#include "gcn/lift.h"
#include "gcn/translate.h"
#include <spirv-tools/libspirv.hpp>
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(VkResult r) {
    if (r != VK_SUCCESS) throw std::runtime_error("Vulkan error " + std::to_string(r));
}

struct Fixture {
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer cmd{};
    VkBuffer buffer{};
    VkDeviceMemory buffer_mem{};
    void* mapped{};
    VkImage source{}, dest{};
    VkDeviceMemory source_mem{}, dest_mem{};
    VkImageView source_view{}, dest_view{};
    VkSampler sampler{};
    VkDescriptorSetLayout layout{};
    VkPipelineLayout pipeline_layout{};
    VkDescriptorPool descriptors{};
    VkDescriptorSet set{};
    VkImage cube_image{};
    VkDeviceMemory cube_mem{};
    VkImageView cube_view{};
    // Guest memory for the buffer loads and stores: one 64 KiB page behind the
    // two-level page table StageParams::l1_table names - every L1 entry the
    // one L2 table, every L2 entry the page - so a guest address reads and
    // writes the page at (address & 0xffff).
    static constexpr VkDeviceSize kGuestL2 = 64 << 10, kGuestL1 = kGuestL2 + (std::uint64_t{gcn::kL2Entries} << 3),
                                  kGuestBytes = kGuestL1 + (gcn::kL1Entries << 3);
    static constexpr std::uint64_t kGuestData = 0x1'0000'0100ull;  // a guest address; the page's byte 0x100
    VkBuffer guest{};
    VkDeviceMemory guest_mem{};
    std::uint8_t* guest_map{};
    VkDeviceAddress guest_base{};  // the page's device address: what a GDS counter word is named by (StageParams::gds_counter)
    // A constant buffer bound as a storage buffer at binding 112 (the
    // translator's cb_binding_base): what the renderer binds over a V#.
    VkBuffer cb{};
    VkDeviceMemory cb_mem{};
    std::uint32_t* cb_map{};

    // Twelve 4x4 layers (two cubes of six faces), texel (x, y, layer, 1): what a
    // Sea Islands cube T# is bound as.
    void init_cube() {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        ci.extent = {4, 4, 1}; ci.mipLevels = 1; ci.arrayLayers = 12;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        check(vkCreateImage(device, &ci, nullptr, &cube_image));
        VkMemoryRequirements req{}; vkGetImageMemoryRequirements(device, cube_image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size; ai.memoryTypeIndex = memory_type(req.memoryTypeBits, 0);
        check(vkAllocateMemory(device, &ai, nullptr, &cube_mem));
        check(vkBindImageMemory(device, cube_image, cube_mem, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = cube_image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY; vi.format = ci.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 12};
        check(vkCreateImageView(device, &vi, nullptr, &cube_view));
        VkBuffer staging{}; VkDeviceMemory staging_mem{};
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = 12 * 256; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        check(vkCreateBuffer(device, &bi, nullptr, &staging));
        vkGetBufferMemoryRequirements(device, staging, &req);
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &ai, nullptr, &staging_mem));
        check(vkBindBufferMemory(device, staging, staging_mem, 0));
        void* map = nullptr;
        check(vkMapMemory(device, staging_mem, 0, VK_WHOLE_SIZE, 0, &map));
        auto* px = static_cast<float*>(map);
        for (int l = 0; l < 12; ++l) for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) {
            const int i = ((l * 4 + y) * 4 + x) * 4;
            px[i] = float(x); px[i+1] = float(y); px[i+2] = float(l); px[i+3] = 1;
        }
        begin();
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = cube_image; b.subresourceRange = vi.subresourceRange;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        std::vector<VkBufferImageCopy> regions(12);
        for (std::uint32_t l = 0; l < 12; ++l) {
            regions[l].bufferOffset = l * 256; regions[l].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, l, 1};
            regions[l].imageExtent = {4, 4, 1};
        }
        vkCmdCopyBufferToImage(cmd, staging, cube_image, VK_IMAGE_LAYOUT_GENERAL, 12, regions.data());
        barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT); submit();
        vkDestroyBuffer(device, staging, nullptr); vkFreeMemory(device, staging_mem, nullptr);
    }

    void init_guest() {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = kGuestBytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device, &bi, nullptr, &guest));
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device, guest, &req);
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.pNext = &flags; ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &ai, nullptr, &guest_mem)); check(vkBindBufferMemory(device, guest, guest_mem, 0));
        void* map = nullptr; check(vkMapMemory(device, guest_mem, 0, VK_WHOLE_SIZE, 0, &map));
        guest_map = static_cast<std::uint8_t*>(map);
        std::memset(guest_map, 0, kGuestBytes);
        VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; dai.buffer = guest;
        const VkDeviceAddress base = vkGetBufferDeviceAddress(device, &dai);
        guest_base = base;
        auto* l2 = reinterpret_cast<std::uint64_t*>(guest_map + kGuestL2);
        for (std::uint32_t i = 0; i < gcn::kL2Entries; ++i) l2[i] = base;
        auto* l1 = reinterpret_cast<std::uint64_t*>(guest_map + kGuestL1);
        for (std::uint32_t i = 0; i < gcn::kL1Entries; ++i) l1[i] = base + kGuestL2;
        static_cast<gcn::StageParams*>(mapped)->l1_table = base + kGuestL1;
    }

    // A compute program of `body`, then run once against the fixture's set:
    // the destination texel back as four floats.
    std::array<float, 4> dispatch_compute(const std::vector<gcn::Inst>& body, std::uint32_t user_sgprs, int cb = 0,
                                          gcn::TranslateResult* result = nullptr) {
        gcn::Program p;
        for (gcn::Inst in : body) { in.offset = uint32_t(p.insts.size() * 8); in.size = 2; p.insts.push_back(in); }
        gcn::Inst end; end.enc = gcn::Enc::SOPP; end.op = 1; end.offset = uint32_t(p.insts.size() * 8); end.size = 1; p.insts.push_back(end);
        gcn::TranslateOptions o; o.stage = gcn::Stage::Compute; o.rsrc2 = user_sgprs << 1; o.cs_threads[0] = 1;
        o.sampler_force_unnormalized = {false};
        // `cb` 1: constant buffers through storage-buffer bindings, with the
        // page-table fallback; 2: without it (the renderer's no-fallback variant).
        o.cb_ssbo = cb != 0;
        o.cb_no_fallback = cb == 2;
        const auto translated = gcn::translate(p, o);
        if (result) *result = translated;
        if (!translated.ok()) throw std::runtime_error(translated.errors[0]);
        spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_2);
        validator.SetMessageConsumer([](auto, auto, const auto&, const char* msg) { std::fprintf(stderr,"%s\n",msg); });
        if (!validator.Validate(translated.spirv)) throw std::runtime_error("invalid SPIR-V");
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = translated.spirv.size()*4; mi.pCode = translated.spirv.data(); VkShaderModule module{};
        check(vkCreateShaderModule(device, &mi, nullptr, &module));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.layout = pipeline_layout;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = module; ci.stage.pName = "main"; VkPipeline pipeline{};
        check(vkCreateComputePipelines(device, {}, 1, &ci, nullptr, &pipeline));
        begin(); barrier(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
        VkBufferImageCopy copy{}; copy.bufferOffset = 2048; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent = {1,1,1};
        vkCmdCopyImageToBuffer(cmd, dest, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &copy);
        barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT); submit();
        vkDestroyPipeline(device, pipeline, nullptr); vkDestroyShaderModule(device, module, nullptr);
        std::array<float, 4> r{};
        std::memcpy(r.data(), static_cast<char*>(mapped) + 2048, sizeof(r));
        return r;
    }
    // ---- GDS counters (TranslateOptions::gds_counters) ----
    // Programs in machine words (the decoder fills the fields), run as
    // `groups` workgroups of `threads` against counter words in guest memory:
    // each lane that counts adds 1 to the element its index names (marks, a
    // 32-bit V# in s[4:7]), so an index two lanes got, or one no lane got,
    // shows. M0 comes from s12, as GX's particle passes take it (0x200020:
    // GDS bytes 0x20.., 0x20 of them).
    static constexpr std::uint64_t kCounterA = 0x1'0000'0f00ull, kCounterB = 0x1'0000'0f04ull, kArgs = 0x1'0000'0f10ull,
                                   kMarks = 0x1'0000'1000ull;
    static constexpr std::uint32_t kMaxMarks = 4096;
    std::uint32_t& guest_u32(std::uint64_t va) { return *reinterpret_cast<std::uint32_t*>(guest_map + (va & 0xffff)); }
    std::uint32_t mark(std::uint32_t k) { return guest_u32(kMarks + 4ull * k); }
    // Which lanes count: all, those whose local id is below `limit`, or the odd ones.
    enum class Lanes { kAll, kBelow, kOdd };
    // ds_append (or ds_consume) at GDS offset `off`, the lane's index made as
    // the game's passes make it - v_mbcnt added (append), or subtracted with 1
    // more (consume) - then marks[index] += 1.
    static std::vector<std::uint32_t> counter_program(bool append, std::uint32_t off, Lanes lanes, std::uint32_t limit = 0) {
        std::vector<std::uint32_t> w = {0xBEFC030Cu};  // s_mov_b32 m0, s12
        if (lanes == Lanes::kBelow) {
            w.push_back(0x7DA800FFu);  // v_cmpx_gt_u32 vcc, limit, v0
            w.push_back(limit);
        } else if (lanes == Lanes::kOdd) {
            w.push_back(0x360A0081u);  // v_and_b32 v5, 1, v0
            w.push_back(0x7DA40A81u);  // v_cmpx_eq_u32 vcc, 1, v5
        }
        w.push_back((append ? 0xD8FA0000u : 0xD8F60000u) | off);  // ds_append / ds_consume v1, v0 offset:off gds
        w.push_back(1u << 24);
        w.push_back(0xD2480003u);  // v_mbcnt_hi_u32_b32 v3, exec_hi, 0
        w.push_back(0x0001007Fu);
        w.push_back(0x4604067Eu);  // v_mbcnt_lo_u32_b32 v2, exec_lo, v3
        if (append) {
            w.push_back(0x4A020501u);  // v_add_i32 v1, v1, v2
        } else {
            w.push_back(0x4C020501u);  // v_sub_i32 v1, v1, v2
            w.push_back(0x4E020281u);  // v_subrev_i32 v1, 1, v1
        }
        w.push_back(0x7E0C0281u);  // v_mov_b32 v6, 1
        w.push_back(0xE0C82000u);  // buffer_atomic_add v6, v1, s[4:7], 0 idxen
        w.push_back(0x80010601u);
        w.push_back(0xBF810000u);  // s_endpgm
        return w;
    }
    struct CounterPipeline {
        VkShaderModule module{};
        VkPipeline pipeline{};
        gcn::TranslateResult result;
    };
    // The counter programs as which module: the translation (its GDS step a
    // subgroup at a time), or the lift with an atomic an invocation or a
    // subgroup (TranslateOptions::gds_subgroup).
    enum class Path { kTranslation, kLiftInvocation, kLiftSubgroup };
    Path path = Path::kTranslation;
    static const char* path_name(Path p) {
        return p == Path::kTranslation ? "translation" : p == Path::kLiftInvocation ? "lift, an atomic an invocation" : "lift, an atomic a subgroup";
    }
    CounterPipeline counter_pipeline(const std::vector<std::uint32_t>& words, std::uint32_t threads, bool gds_counters = true) {
        const gcn::Program prog = gcn::decode(words.data(), words.size());
        if (!prog.errors.empty()) throw std::runtime_error("counter program: " + prog.errors[0].what);
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::Compute;
        o.rsrc2 = 13 << 1;
        o.cs_threads[0] = threads;
        o.gds_counters = gds_counters;
        o.gds_subgroup = path == Path::kLiftSubgroup;
        CounterPipeline cp;
        cp.result = gcn::translate(prog, o);
        if (!cp.result.ok()) throw std::runtime_error(cp.result.errors[0]);
        std::vector<std::uint32_t> spirv = cp.result.spirv;
        if (path != Path::kTranslation && gds_counters) {
            const gcn::LiftResult l = gcn::lift_compute_shader(prog, o, cp.result);
            if (!l.ok()) throw std::runtime_error("counter program not lifted: " + (l.rejections.empty() ? std::string("?") : l.rejections[0]));
            spirv = l.spirv;
            // The step it was asked for: an elected invocation's atomic a subgroup, or none.
            std::size_t elects = 0;
            for (std::size_t k = 5; k < spirv.size(); k += std::max<std::uint32_t>(1, spirv[k] >> 16)) elects += (spirv[k] & 0xffff) == 333;  // OpGroupNonUniformElect
            if ((elects != 0) != (path == Path::kLiftSubgroup)) throw std::runtime_error("the lift's GDS step is not the one asked for");
        }
        spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_2);
        validator.SetMessageConsumer([](auto, auto, const auto&, const char* msg) { std::fprintf(stderr, "%s\n", msg); });
        if (!validator.Validate(spirv)) throw std::runtime_error("invalid SPIR-V");
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = spirv.size() * 4;
        mi.pCode = spirv.data();
        check(vkCreateShaderModule(device, &mi, nullptr, &cp.module));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.layout = pipeline_layout;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = cp.module;
        ci.stage.pName = "main";
        check(vkCreateComputePipelines(device, {}, 1, &ci, nullptr, &cp.pipeline));
        return cp;
    }
    void destroy(CounterPipeline& cp) {
        vkDestroyPipeline(device, cp.pipeline, nullptr);
        vkDestroyShaderModule(device, cp.module, nullptr);
    }
    // The params a dispatch of these takes: M0's user SGPR, the marks V#, and
    // the counter words for GDS dwords 8.. (byte 0x20 on).
    void counter_params(std::uint64_t word0, std::uint64_t word1) {
        auto* params = static_cast<gcn::StageParams*>(mapped);
        std::memset(params->user_sgpr, 0, sizeof(params->user_sgpr));
        params->user_sgpr[4] = static_cast<std::uint32_t>(kMarks);
        params->user_sgpr[5] = static_cast<std::uint32_t>(kMarks >> 32) | 4u << 16;  // stride 4
        params->user_sgpr[6] = kMaxMarks;
        params->user_sgpr[7] = 4u << 15 | 4u << 12 | 0xfac;  // 32 uint, X Y Z W
        params->user_sgpr[12] = 0x200020;
        params->gds_base = 8;
        std::memset(params->gds_counter, 0, sizeof(params->gds_counter));
        // By device address, as the host hands them over (no page-table walk).
        params->gds_counter[0] = word0 ? guest_base + (word0 & 0xffff) : 0;
        params->gds_counter[1] = word1 ? guest_base + (word1 & 0xffff) : 0;
        std::memset(guest_map + (kMarks & 0xffff), 0, kMaxMarks * 4);
    }
    void record_counter_dispatch(const CounterPipeline& cp, std::uint32_t groups, bool indirect = false) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
        if (indirect) {
            vkCmdDispatchIndirect(cmd, guest, kArgs & 0xffff);
        } else {
            vkCmdDispatch(cmd, groups, 1, 1);
        }
    }
    // Every element in [lo, hi) marked once and nothing else: what a counter
    // that hands out each index of its range exactly once leaves.
    bool marks_once(std::uint32_t lo, std::uint32_t hi, std::string& why) {
        for (std::uint32_t k = 0; k < kMaxMarks; ++k) {
            const std::uint32_t want = k >= lo && k < hi ? 1 : 0;
            if (mark(k) != want) {
                why = "element " + std::to_string(k) + " marked " + std::to_string(mark(k)) + " times, expected " + std::to_string(want);
                return false;
            }
        }
        return true;
    }
    // `threads` x `groups` lanes (those `lanes` lets count) append to counter
    // A at GDS byte 0x20 + off from `start`: A ends at start + the lanes that
    // counted, each index in between went to one lane, B is untouched.
    bool run_gds_append(std::uint32_t threads, std::uint32_t groups, Lanes lanes, std::uint32_t limit, std::uint32_t off, std::uint32_t start,
                        const char* what) {
        const std::uint32_t total = threads * groups;
        std::uint32_t counting = 0;
        for (std::uint32_t t = 0; t < threads; ++t) counting += lanes == Lanes::kAll || (lanes == Lanes::kBelow && t < limit) || (lanes == Lanes::kOdd && (t & 1));
        counting *= groups;
        CounterPipeline cp = counter_pipeline(counter_program(true, off, lanes, limit), threads);
        // off 0 counts in gds_counter[0], off 4 in [1]: A goes where off names, B in the other.
        counter_params(off == 0 ? kCounterA : kCounterB, off == 0 ? kCounterB : kCounterA);
        guest_u32(kCounterA) = start;
        guest_u32(kCounterB) = 777;
        begin();
        barrier(VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        record_counter_dispatch(cp, groups);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        submit();
        std::string why;
        bool ok = cp.result.gds_counters == 1 && guest_u32(kCounterA) == start + counting && guest_u32(kCounterB) == 777 &&
                  marks_once(start, start + counting, why);
        if (cp.result.gds_counters != 1) why = std::to_string(cp.result.gds_counters) + " GDS counter sites, expected 1";
        std::printf("%s [%s] ds_append gds offset:%u, %u threads x %u groups (%u counting) from %u, %s: counter %u (expected %u), other %u%s%s\n",
                    ok ? "PASS" : "FAIL", path_name(path), off, threads, groups, counting, start, what, guest_u32(kCounterA), start + counting, guest_u32(kCounterB),
                    why.empty() ? "" : "; ", why.c_str());
        destroy(cp);
        (void)total;
        return ok;
    }
    // Lanes below `n` of `threads` x `groups` consume from counter A holding
    // `start`: A ends at start - those lanes, and the indices they read
    // (from the top down) are each read once.
    bool run_gds_consume(std::uint32_t threads, std::uint32_t groups, std::uint32_t n, std::uint32_t start, const char* what) {
        const std::uint32_t counting = std::min(threads, n) * groups;
        CounterPipeline cp = counter_pipeline(counter_program(false, 0, Lanes::kBelow, n), threads);
        counter_params(kCounterA, kCounterB);
        guest_u32(kCounterA) = start;
        guest_u32(kCounterB) = 777;
        begin();
        barrier(VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        record_counter_dispatch(cp, groups);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        submit();
        std::string why;
        const bool ok = guest_u32(kCounterA) == start - counting && guest_u32(kCounterB) == 777 && marks_once(start - counting, start, why);
        std::printf("%s [%s] ds_consume gds, %u threads x %u groups (%u counting) from %u, %s: counter %u (expected %u)%s%s\n", ok ? "PASS" : "FAIL",
                    path_name(path), threads, groups, counting, start, what, guest_u32(kCounterA), start - counting, why.empty() ? "" : "; ", why.c_str());
        destroy(cp);
        return ok;
    }
    // No word bound for the dword (gds_counter 0): nothing counts, the
    // program still runs, and every lane gets 0 back (its index is its rank).
    bool run_gds_unbound() {
        CounterPipeline cp = counter_pipeline(counter_program(true, 8, Lanes::kAll), 32);  // GDS byte 0x28: gds_counter[2], left 0
        counter_params(kCounterA, kCounterB);
        guest_u32(kCounterA) = 5;
        guest_u32(kCounterB) = 6;
        begin();
        barrier(VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        record_counter_dispatch(cp, 1);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        submit();
        std::uint32_t marked = 0;
        for (std::uint32_t k = 0; k < kMaxMarks; ++k) marked += mark(k);
        const bool ok = guest_u32(kCounterA) == 5 && guest_u32(kCounterB) == 6 && marked == 32 && mark(0) >= 1;
        std::printf("%s ds_append gds with no word bound: counters %u %u (expected 5 6), %u lanes marked from index 0 (expected 32)\n",
                    ok ? "PASS" : "FAIL", guest_u32(kCounterA), guest_u32(kCounterB), marked);
        destroy(cp);
        return ok;
    }
    // CopyStructureCount as the host does it: lanes below `limit` of `groups`
    // workgroups of 64 append to A, a copy of A into the X of a dispatch's
    // indirect arguments on the GPU's timeline (after the dispatch, before
    // the next), then that many workgroups of one lane consume from A. A ends
    // at 0, each index once: the second dispatch ran the count the first one
    // left, not the count memory held when it was recorded (0).
    bool run_counter_to_args(std::uint32_t groups, std::uint32_t limit) {
        const std::uint32_t n = groups * std::min(limit, 64u);
        CounterPipeline producer = counter_pipeline(counter_program(true, 0, Lanes::kBelow, limit), 64);
        CounterPipeline consumer = counter_pipeline(counter_program(false, 0, Lanes::kAll), 1);
        counter_params(kCounterA, kCounterB);
        guest_u32(kCounterA) = 0;
        guest_u32(kArgs) = 0;  // what the CPU sees when the dispatches are recorded
        guest_u32(kArgs + 4) = 1;
        guest_u32(kArgs + 8) = 1;
        begin();
        barrier(VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        record_counter_dispatch(producer, groups);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);  // the host's transfer-batch barriers
        VkBufferCopy region{kCounterA & 0xffff, kArgs & 0xffff, 4};
        vkCmdCopyBuffer(cmd, guest, guest, 1, &region);
        barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        // The consumer's marks start where the producer's are: clear them on the GPU's timeline too.
        vkCmdFillBuffer(cmd, guest, kMarks & 0xffff, kMaxMarks * 4, 0);
        barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
        record_counter_dispatch(consumer, 0, true);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        submit();
        std::string why;
        const bool ok = guest_u32(kArgs) == n && guest_u32(kCounterA) == 0 && marks_once(0, n, why);
        std::printf("%s counter into indirect arguments on the GPU's timeline: %u appended, arguments X %u, counter after the consumers %u "
                    "(expected 0)%s%s\n",
                    ok ? "PASS" : "FAIL", n, guest_u32(kArgs), guest_u32(kCounterA), why.empty() ? "" : "; ", why.c_str());
        destroy(producer);
        destroy(consumer);
        return ok;
    }
    // Off (BBHOST_GDS_COUNTERS=0) the program is the old one: no counter sites.
    bool run_gds_off() {
        CounterPipeline cp = counter_pipeline(counter_program(true, 4, Lanes::kAll), 64, false);
        const bool ok = cp.result.gds_counters == 0;
        std::printf("%s ds_append gds with gds_counters off: %u counter sites (expected 0)\n", ok ? "PASS" : "FAIL", cp.result.gds_counters);
        destroy(cp);
        return ok;
    }

    static gcn::Inst vmov(int reg, std::uint32_t bits) {
        gcn::Inst in; in.enc = gcn::Enc::VOP1; in.op = 1; in.dst = static_cast<std::uint16_t>(reg);
        in.src0 = gcn::kLiteral; in.has_literal = true; in.literal = bits; return in;
    }
    // A store of v[16:19] to the destination (0, 0) through the T# in s[0:7]
    // whose word 3 is `t_w3`; first a sample through the same T#, so the store
    // is the program's second image binding (81) as in run().
    static void image_store_body(std::vector<gcn::Inst>& body) {
        body.push_back(vmov(26, 0)); body.push_back(vmov(27, 0));
        gcn::Inst sample; sample.enc = gcn::Enc::MIMG; sample.op = 39;  // image_sample_lz
        sample.srsrc = 0; sample.ssamp = 8; sample.vaddr = 26; sample.vdata = 30; sample.dmask = 15; body.push_back(sample);
        body.push_back(vmov(24, 0)); body.push_back(vmov(25, 0));
        gcn::Inst store; store.enc = gcn::Enc::MIMG; store.op = 8;  // image_store
        store.srsrc = 0; store.vaddr = 24; store.vdata = 16; store.dmask = 15; body.push_back(store);
    }
    static bool near4(const std::array<float, 4>& got, const float (&want)[4]) {
        for (int k = 0; k < 4; ++k) if (std::abs(got[static_cast<std::size_t>(k)] - want[k]) > 0.002f) return false;
        return true;
    }
    // image_store goes through the inverse of the T#'s DST_SEL (memory
    // component c from the data component that selects it, 0 where none does):
    // the data (1, 2, 3, 4) through the select `t_w3`.
    bool run_store_select(std::uint32_t t_w3, const float (&want)[4], const char* what) {
        auto* params = static_cast<gcn::StageParams*>(mapped);
        params->user_sgpr[3] = t_w3; params->user_sgpr[8] = 0;
        std::vector<gcn::Inst> body;
        for (int k = 0; k < 4; ++k) body.push_back(vmov(16 + k, std::bit_cast<std::uint32_t>(float(k + 1))));
        image_store_body(body);
        const auto got = dispatch_compute(body, 12);
        const bool ok = near4(got, want);
        std::printf("%s image_store through DST_SEL %03x (%s): got (%g,%g,%g,%g), expected (%g,%g,%g,%g)\n", ok ? "PASS" : "FAIL", t_w3 & 0xfff, what,
                    got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
        return ok;
    }
    // The shape of GX's texture upload copy (gx_upload_initial_data, for a
    // texture created with data): buffer_load_format_xyzw of one 8-bit unorm
    // element, at an odd byte, through a V# with the A8 select (0, 0, 0, X) -
    // word 3 0x8800 - then image_store of the four components through a T#
    // with the same select. The load gives (0, 0, 0, a) and the inverse store
    // puts a in R: on the console the 4x4 dither texture holds its Bayer
    // values this way.
    bool run_upload_copy() {
        static const std::uint8_t bayer[16] = {0x00, 0x88, 0x22, 0xaa, 0xcc, 0x44, 0xee, 0x66, 0x33, 0xbb, 0x11, 0x99, 0xff, 0x77, 0xdd, 0x55};
        std::memcpy(guest_map + (kGuestData & 0xffff), bayer, sizeof(bayer));
        auto* params = static_cast<gcn::StageParams*>(mapped);
        params->user_sgpr[3] = 0x800; params->user_sgpr[8] = 0;
        params->user_sgpr[12] = static_cast<std::uint32_t>(kGuestData);
        params->user_sgpr[13] = static_cast<std::uint32_t>(kGuestData >> 32) | 1u << 16;  // stride 1
        params->user_sgpr[14] = 16;                                                     // records
        params->user_sgpr[15] = 0x8800;  // DFMT 8, NFMT unorm, DST_SEL (0, 0, 0, X)
        std::vector<gcn::Inst> body;
        body.push_back(vmov(0, 5));  // element 5: 0x44
        gcn::Inst load; load.enc = gcn::Enc::MUBUF; load.op = 3;  // buffer_load_format_xyzw
        load.vaddr = 0; load.vdata = 16; load.srsrc = 12; load.soffset = 128; load.idxen = true; body.push_back(load);
        image_store_body(body);
        const auto got = dispatch_compute(body, 16);
        const float want[4] = {0x44 / 255.0f, 0, 0, 0};
        const bool ok = near4(got, want);
        std::printf("%s A8 upload copy, buffer_load_format then image_store: got (%g,%g,%g,%g), expected (%g,0,0,0)\n", ok ? "PASS" : "FAIL",
                    got[0], got[1], got[2], got[3], want[0]);
        return ok;
    }
    // buffer_store_format goes through the inverse of the V#'s DST_SEL too:
    // (0, 0, 0, 0.5) through an A8 V# writes 128 into its one byte, at an odd
    // address, and leaves the bytes around it.
    bool run_buffer_store_select() {
        std::memset(guest_map + (kGuestData & 0xffff), 0x5a, 16);
        auto* params = static_cast<gcn::StageParams*>(mapped);
        params->user_sgpr[12] = static_cast<std::uint32_t>(kGuestData);
        params->user_sgpr[13] = static_cast<std::uint32_t>(kGuestData >> 32) | 1u << 16;
        params->user_sgpr[14] = 16;
        params->user_sgpr[15] = 0x8800;
        std::vector<gcn::Inst> body;
        body.push_back(vmov(0, 7));
        body.push_back(vmov(16, 0)); body.push_back(vmov(17, 0)); body.push_back(vmov(18, 0));
        body.push_back(vmov(19, std::bit_cast<std::uint32_t>(0.5f)));
        gcn::Inst store; store.enc = gcn::Enc::MUBUF; store.op = 7;  // buffer_store_format_xyzw
        store.vaddr = 0; store.vdata = 16; store.srsrc = 12; store.soffset = 128; store.idxen = true; body.push_back(store);
        dispatch_compute(body, 16);
        const std::uint8_t* b = guest_map + (kGuestData & 0xffff);
        const bool ok = b[7] == 128 && b[6] == 0x5a && b[8] == 0x5a;
        std::printf("%s A8 buffer_store_format: byte %u (expected 128), neighbours %02x %02x (expected 5a 5a)\n", ok ? "PASS" : "FAIL", b[7], b[6], b[8]);
        return ok;
    }

    // s_buffer_load_dwordx4 at a register offset (BufferBinding::ranged):
    // dword (offset >> 2) + k of the V#'s memory, 0 for a dword at or past
    // the V#'s range - num_records bytes, times the stride when that is not 0.
    // The V# is in s[12:15]; the offset is put in s20 by s_movk_i32, so the
    // load's offset is a register; the four dwords go to the destination
    // texel through v[16:19]. Bound (`bound`): the storage buffer at 112 holds
    // the V#'s memory and StageParams::cb_stride its range in dwords; walked:
    // guest memory, and the range from the V#'s own words.
    bool run_ranged_load(int cb, bool bound, std::uint32_t offset, std::uint32_t stride, std::uint32_t records, const float (&want)[4],
                         const char* what) {
        auto* params = static_cast<gcn::StageParams*>(mapped);
        params->user_sgpr[3] = 0xfac; params->user_sgpr[8] = 0;
        float* mem = reinterpret_cast<float*>(guest_map + (kGuestData & 0xffff));
        for (int k = 0; k < 64; ++k) {
            mem[k] = 100.0f + float(k);                                   // walked
            reinterpret_cast<float*>(cb_map)[k] = 100.0f + float(k);      // bound
        }
        params->user_sgpr[12] = static_cast<std::uint32_t>(kGuestData);
        params->user_sgpr[13] = static_cast<std::uint32_t>(kGuestData >> 32) | stride << 16;
        params->user_sgpr[14] = records;
        params->user_sgpr[15] = 0x77fac;  // 32_32_32_32 float, X Y Z W: what scalar loads ignore
        params->cb_valid = bound ? 1u : 0u;
        params->cb_bias_dw[0] = 0;
        params->cb_stride[0] = bound ? static_cast<std::uint32_t>(gcn::vsharp_range_dwords(params->user_sgpr[13], records)) : 0xdeadu;
        std::vector<gcn::Inst> body;
        gcn::Inst movk; movk.enc = gcn::Enc::SOPK; movk.op = 0; movk.dst = 20; movk.imm = static_cast<std::int32_t>(offset); body.push_back(movk);
        gcn::Inst load; load.enc = gcn::Enc::SMRD; load.op = 10;  // s_buffer_load_dwordx4 s[24:27], s[12:15], s20
        load.sdst = 24; load.src0 = 12; load.imm_flag = false; load.imm = 20; body.push_back(load);
        gcn::Inst wait; wait.enc = gcn::Enc::SOPP; wait.op = 12; wait.imm = 0; body.push_back(wait);
        for (int k = 0; k < 4; ++k) {
            gcn::Inst mov; mov.enc = gcn::Enc::VOP1; mov.op = 1; mov.dst = static_cast<std::uint16_t>(16 + k); mov.src0 = static_cast<std::uint16_t>(24 + k);
            body.push_back(mov);
        }
        image_store_body(body);
        gcn::TranslateResult tr;
        const auto got = dispatch_compute(body, 16, cb, &tr);
        params->cb_valid = 0;
        const bool ranged = tr.buffers.size() == 1 && tr.buffers[0].ranged && !tr.buffers[0].indexed && tr.buffer_at.count(8);
        const bool ok = ranged && near4(got, want);
        std::printf("%s s_buffer_load at a register offset (%s, %s): offset %u, stride %u, records %u: got (%g,%g,%g,%g), expected (%g,%g,%g,%g)%s\n",
                    ok ? "PASS" : "FAIL", cb == 2 ? "no fallback" : "with fallback", bound ? "bound" : "walked", offset, stride, records, got[0],
                    got[1], got[2], got[3], want[0], want[1], want[2], want[3], ranged ? "" : " - not a ranged binding");
        (void)what;
        return ok;
    }

    // buffer_store_dwordx4 v[16:19] through the V# in s[12:15] at its base
    // plus s20 (`offset`, or with `constant` the inline 0) plus offset12 16:
    // in the no-fallback variant (`cb` 2) the store goes through the V#'s
    // binding (BufferBinding::written, ranged), and each dword at or past the
    // V#'s range is not written, as GCN does not perform it; walked (`cb` 0)
    // it lands in guest memory with no range check (the translator's walk).
    bool run_bound_store(int cb, std::uint32_t offset, bool constant, std::uint32_t stride, std::uint32_t records, const char* what) {
        auto* params = static_cast<gcn::StageParams*>(mapped);
        params->user_sgpr[3] = 0xfac; params->user_sgpr[8] = 0;
        std::uint32_t* mem = reinterpret_cast<std::uint32_t*>(guest_map + (kGuestData & 0xffff));
        for (int k = 0; k < 64; ++k) mem[k] = cb_map[k] = 0x5555u;
        params->user_sgpr[12] = static_cast<std::uint32_t>(kGuestData);
        params->user_sgpr[13] = static_cast<std::uint32_t>(kGuestData >> 32) | stride << 16;
        params->user_sgpr[14] = records;
        params->user_sgpr[15] = 0x77fac;
        const std::uint32_t range = static_cast<std::uint32_t>(gcn::vsharp_range_dwords(params->user_sgpr[13], records));
        params->cb_valid = cb ? 1u : 0u;
        params->cb_bias_dw[0] = 0;
        params->cb_stride[0] = range;
        std::vector<gcn::Inst> body;
        gcn::Inst movk; movk.enc = gcn::Enc::SOPK; movk.op = 0; movk.dst = 20; movk.imm = static_cast<std::int32_t>(offset); body.push_back(movk);
        for (int k = 0; k < 4; ++k) body.push_back(vmov(16 + k, 0x1000u + static_cast<std::uint32_t>(k)));
        gcn::Inst st; st.enc = gcn::Enc::MUBUF; st.op = 30;  // buffer_store_dwordx4 v[16:19], off, s[12:15], s20 offset:16
        st.vdata = 16; st.srsrc = 12; st.soffset = static_cast<std::uint16_t>(constant ? 128 : 20); st.offset12 = 16;
        body.push_back(st);
        image_store_body(body);
        gcn::TranslateResult tr;
        dispatch_compute(body, 16, cb, &tr);
        params->cb_valid = 0;
        const bool written = cb == 2 && tr.buffers.size() == 1 && tr.buffers[0].written && tr.buffers[0].ranged && tr.walks.empty();
        const bool walked = cb == 0 && tr.buffers.empty() && !tr.walks.empty();
        const std::uint32_t* got = cb == 2 ? cb_map : mem;
        const std::uint32_t first = ((constant ? 0u : offset) + 16) >> 2;
        bool ok = written || walked;
        for (std::uint32_t k = 0; k < 64; ++k) {
            const bool stored = k >= first && k < first + 4 && (cb == 0 || k < range);
            const std::uint32_t want = stored ? 0x1000u + (k - first) : 0x5555u;
            if (got[k] != want) {
                std::printf("  dword %u holds %08x, expected %08x\n", k, got[k], want);
                ok = false;
            }
        }
        std::printf("%s buffer_store_dwordx4 at %s (%s): offset %u, stride %u, records %u (%s)%s\n", ok ? "PASS" : "FAIL",
                    constant ? "a constant offset" : "a register offset", cb == 2 ? "bound" : "walked", offset, stride, records, what,
                    written || walked ? "" : " - not the expected binding");
        return ok;
    }

    // tbuffer_load_format_* takes the instruction's format and the identity
    // select - X000, XY00, XYZ0 or XYZW by the format's components - whatever
    // the V#'s DST_SEL holds (the ISA's buffer instruction table); its VGPRs
    // go to the destination texel. Elements of four floats (1, 2, 3, 4) + 4i
    // at the guest page; the V# in s[12:15] with stride 16; v0 the index.
    // Walked (`cb` 0) or through the storage-buffer binding (`cb` 2).
    bool run_typed_load(int cb, std::uint32_t dfmt, int vgprs, std::uint32_t v_w3, const float (&want)[4], const char* what) {
        auto* params = static_cast<gcn::StageParams*>(mapped);
        params->user_sgpr[3] = 0xfac; params->user_sgpr[8] = 0;
        float* mem = reinterpret_cast<float*>(guest_map + (kGuestData & 0xffff));
        for (int k = 0; k < 64; ++k) mem[k] = reinterpret_cast<float*>(cb_map)[k] = float(k + 1);
        params->user_sgpr[12] = static_cast<std::uint32_t>(kGuestData);
        params->user_sgpr[13] = static_cast<std::uint32_t>(kGuestData >> 32) | 16u << 16;
        params->user_sgpr[14] = 4;
        params->user_sgpr[15] = v_w3;
        params->cb_valid = cb ? 1u : 0u;
        params->cb_bias_dw[0] = 0;
        params->cb_stride[0] = 16;
        params->cb_w3[0] = v_w3;
        std::vector<gcn::Inst> body;
        body.push_back(vmov(0, 1));  // element 1: (5, 6, 7, 8)
        for (int k = 0; k < 4; ++k) body.push_back(vmov(16 + k, std::bit_cast<std::uint32_t>(-1.0f)));
        gcn::Inst load; load.enc = gcn::Enc::MTBUF; load.op = static_cast<std::uint16_t>(vgprs - 1);  // tbuffer_load_format_x..xyzw
        load.vaddr = 0; load.vdata = 16; load.srsrc = 12; load.soffset = 128; load.idxen = true; load.dfmt = static_cast<std::uint8_t>(dfmt); load.nfmt = 7;
        body.push_back(load);
        image_store_body(body);
        const auto got = dispatch_compute(body, 16, cb);
        params->cb_valid = 0;
        const bool ok = near4(got, want);
        std::printf("%s tbuffer_load_format (%s), dfmt %u into %d VGPRs, V# select %03x (%s): got (%g,%g,%g,%g), expected (%g,%g,%g,%g)\n",
                    ok ? "PASS" : "FAIL", cb ? "bound" : "walked", dfmt, vgprs, v_w3 & 0xfff, what, got[0], got[1], got[2], got[3], want[0], want[1],
                    want[2], want[3]);
        return ok;
    }
    // buffer_load_format_* takes the V#'s format and its select: the same
    // element read through 32_32_32_32 float with Z Y X W.
    bool run_format_load_select() {
        auto* params = static_cast<gcn::StageParams*>(mapped);
        params->user_sgpr[3] = 0xfac; params->user_sgpr[8] = 0;
        float* mem = reinterpret_cast<float*>(guest_map + (kGuestData & 0xffff));
        for (int k = 0; k < 64; ++k) mem[k] = float(k + 1);
        params->user_sgpr[12] = static_cast<std::uint32_t>(kGuestData);
        params->user_sgpr[13] = static_cast<std::uint32_t>(kGuestData >> 32) | 16u << 16;
        params->user_sgpr[14] = 4;
        params->user_sgpr[15] = 14u << 15 | 7u << 12 | 0xf2eu;  // 32_32_32_32 float, Z Y X W
        std::vector<gcn::Inst> body;
        body.push_back(vmov(0, 1));
        gcn::Inst load; load.enc = gcn::Enc::MUBUF; load.op = 3;  // buffer_load_format_xyzw
        load.vaddr = 0; load.vdata = 16; load.srsrc = 12; load.soffset = 128; load.idxen = true; body.push_back(load);
        image_store_body(body);
        const auto got = dispatch_compute(body, 16);
        const float want[4] = {7, 6, 5, 8};
        const bool ok = near4(got, want);
        std::printf("%s buffer_load_format_xyzw through the V#'s Z Y X W: got (%g,%g,%g,%g), expected (7,6,5,8)\n", ok ? "PASS" : "FAIL", got[0], got[1],
                    got[2], got[3]);
        return ok;
    }
    // image_store's DMASK names the data components the consecutive VGPRs
    // hold, R first; the others are 0 (the ISA's MIMG DMASK field), before the
    // inverse of the T#'s select (X Y Z W here).
    bool run_store_dmask(std::uint32_t dmask, const float (&want)[4]) {
        auto* params = static_cast<gcn::StageParams*>(mapped);
        params->user_sgpr[3] = 0xfac; params->user_sgpr[8] = 0;
        std::vector<gcn::Inst> body;
        for (int k = 0; k < 4; ++k) body.push_back(vmov(16 + k, std::bit_cast<std::uint32_t>(float(k + 1))));
        image_store_body(body);
        body.back().dmask = static_cast<std::uint8_t>(dmask);
        const auto got = dispatch_compute(body, 12);
        const bool ok = near4(got, want);
        std::printf("%s image_store dmask %x of (1, 2, 3, 4): got (%g,%g,%g,%g), expected (%g,%g,%g,%g)\n", ok ? "PASS" : "FAIL", dmask, got[0], got[1],
                    got[2], got[3], want[0], want[1], want[2], want[3]);
        return ok;
    }

    void bind_sampled(VkImageView view) {
        VkDescriptorImageInfo info{{}, view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set; w.dstBinding = 16; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w.pImageInfo = &info;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }

    // A Sea Islands cube sample: (s, t) in [1, 2] on the face and the face id,
    // plus 8 per cube in an array, must read that face's texel as an array layer.
    bool run_cube(float s, float t, float face, int ex, int ey, int elayer) {
        gcn::Program p;
        auto add = [&](gcn::Inst in) { in.offset = uint32_t(p.insts.size() * 8); in.size = 2; p.insts.push_back(in); };
        auto mov = [&](int reg, float value) {
            gcn::Inst in; in.enc = gcn::Enc::VOP1; in.op = 1; in.dst = reg;
            in.src0 = gcn::kLiteral; in.has_literal = true; in.literal = std::bit_cast<uint32_t>(value); add(in);
        };
        mov(0, s); mov(1, t); mov(2, face);
        gcn::Inst sample; sample.enc = gcn::Enc::MIMG; sample.op = 39;  // image_sample_lz
        sample.srsrc = 0; sample.ssamp = 8; sample.vaddr = 0; sample.vdata = 16; sample.dmask = 15; add(sample);
        mov(24, 0); mov(25, 0);
        gcn::Inst store = sample; store.op = 8; store.vaddr = 24; add(store);
        gcn::Inst end; end.enc = gcn::Enc::SOPP; end.op = 1; add(end);
        gcn::TranslateOptions o; o.stage = gcn::Stage::Compute; o.rsrc2 = 12 << 1; o.cs_threads[0] = 1;
        o.sampler_force_unnormalized = {false};
        o.image_dims = {{3 /* DimCube */, false}};
        const auto translated = gcn::translate(p, o);
        if (!translated.ok()) throw std::runtime_error(translated.errors[0]);
        if (translated.images.empty() || !translated.images[0].cube || translated.images[0].dim != 1 || !translated.images[0].arrayed)
            throw std::runtime_error("a cube binding must be a 2D array");
        spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_2);
        validator.SetMessageConsumer([](auto, auto, const auto&, const char* msg) { std::fprintf(stderr,"%s\n",msg); });
        if (!validator.Validate(translated.spirv)) throw std::runtime_error("invalid SPIR-V");
        static_cast<gcn::StageParams*>(mapped)->user_sgpr[8] = 0;
        static_cast<gcn::StageParams*>(mapped)->user_sgpr[3] = 0xfac;
        bind_sampled(cube_view);
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = translated.spirv.size()*4; mi.pCode = translated.spirv.data(); VkShaderModule module{};
        check(vkCreateShaderModule(device, &mi, nullptr, &module));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.layout = pipeline_layout;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = module; ci.stage.pName = "main"; VkPipeline pipeline{};
        check(vkCreateComputePipelines(device, {}, 1, &ci, nullptr, &pipeline));
        begin(); barrier(VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{}; copy.bufferOffset = 2048; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent = {1,1,1};
        vkCmdCopyImageToBuffer(cmd, dest, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &copy);
        barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT); submit();
        bind_sampled(source_view);
        const float* r = reinterpret_cast<float*>(static_cast<char*>(mapped) + 2048);
        const bool ok = std::abs(r[0]-ex)<0.01f && std::abs(r[1]-ey)<0.01f && std::abs(r[2]-elayer)<0.01f && std::abs(r[3]-1)<0.01f;
        std::printf("%s cube (s,t,face)=(%g,%g,%g): got (%g,%g,layer %g), expected (%d,%d,layer %d)\n",
                    ok ? "PASS" : "FAIL", s, t, face, r[0], r[1], r[2], ex, ey, elayer);
        vkDestroyPipeline(device, pipeline, nullptr); vkDestroyShaderModule(device, module, nullptr);
        return ok;
    }

    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
        VkPhysicalDeviceMemoryProperties p{};
        vkGetPhysicalDeviceMemoryProperties(physical, &p);
        for (uint32_t i = 0; i < p.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (p.memoryTypes[i].propertyFlags & flags) == flags) return i;
        throw std::runtime_error("no matching memory type");
    }
    void make_image(uint32_t size, VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        ci.extent = {size, size, 1}; ci.mipLevels = 1; ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        check(vkCreateImage(device, &ci, nullptr, &image));
        VkMemoryRequirements req{}; vkGetImageMemoryRequirements(device, image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size; ai.memoryTypeIndex = memory_type(req.memoryTypeBits, 0);
        check(vkAllocateMemory(device, &ai, nullptr, &memory));
        check(vkBindImageMemory(device, image, memory, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = ci.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device, &vi, nullptr, &view));
    }
    void begin() {
        check(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cmd, &bi));
    }
    void submit() {
        check(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence{};
        check(vkCreateFence(device, &fi, nullptr, &fence));
        check(vkQueueSubmit(queue, 1, &si, fence));
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, 10'000'000'000ull));
        vkDestroyFence(device, fence, nullptr);
    }
    void barrier(VkAccessFlags src, VkAccessFlags dst) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = src; mb.dstAccessMask = dst;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    bool init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return false;
        uint32_t count = 0; check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
        uint32_t family = 0;
        for (auto candidate : devices) {
            VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f.pNext = &v12;
            vkGetPhysicalDeviceFeatures2(candidate, &f);
            VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; props.pNext = &subgroup;
            vkGetPhysicalDeviceProperties2(candidate, &props);
            const auto ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
            if (!f.features.shaderInt64 || !f.features.shaderStorageImageWriteWithoutFormat || !v12.bufferDeviceAddress ||
                props.properties.apiVersion < VK_API_VERSION_1_2 || !(subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
                (subgroup.supportedOperations & ops) != ops) continue;
            uint32_t n = 0; vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, nullptr);
            std::vector<VkQueueFamilyProperties> queues(n); vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, queues.data());
            for (uint32_t i = 0; i < n; ++i) if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physical = candidate; family = i; break;
            }
            if (physical) break;
        }
        if (!physical) return false;
        float priority = 1;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures features{}; features.shaderInt64 = VK_TRUE;
        features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        // The translator declares both formatless storage capabilities.
        VkPhysicalDeviceFeatures supported{}; vkGetPhysicalDeviceFeatures(physical, &supported);
        features.shaderStorageImageReadWithoutFormat = supported.shaderStorageImageReadWithoutFormat;
        if (!features.shaderStorageImageReadWithoutFormat) return false;
        VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; v12.bufferDeviceAddress = VK_TRUE;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.pNext = &v12;
        di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi; di.pEnabledFeatures = &features;
        check(vkCreateDevice(physical, &di, nullptr, &device)); vkGetDeviceQueue(device, family, 0, &queue);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = family; pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(device, &pi, nullptr, &pool));
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ca.commandPool = pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device, &ca, &cmd));
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = 4096;
        bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device, &bi, nullptr, &buffer));
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device, buffer, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &ai, nullptr, &buffer_mem)); check(vkBindBufferMemory(device, buffer, buffer_mem, 0));
        check(vkMapMemory(device, buffer_mem, 0, VK_WHOLE_SIZE, 0, &mapped));
        std::memset(mapped, 0, 4096);
        auto* pixels = reinterpret_cast<float*>(static_cast<char*>(mapped) + 256);
        for (int y = 0; y < 8; ++y) for (int x = 0; x < 8; ++x) {
            const int i = (y * 8 + x) * 4; pixels[i] = float(x); pixels[i+1] = float(y); pixels[i+2] = 0.25f; pixels[i+3] = 1;
        }
        make_image(8, source, source_mem, source_view); make_image(1, dest, dest_mem, dest_view);
        begin();
        std::array<VkImageMemoryBarrier, 2> transitions{};
        for (int i = 0; i < 2; ++i) {
            auto& b = transitions[i]; b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = i ? dest : source; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, 2, transitions.data());
        VkBufferImageCopy copy{}; copy.bufferOffset = 256; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent = {8,8,1};
        vkCmdCopyBufferToImage(cmd, buffer, source, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
        barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT); submit();
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_NEAREST; sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        check(vkCreateSampler(device, &sci, nullptr, &sampler));
        const std::array<VkDescriptorSetLayoutBinding,5> bindings = {{{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
            {16,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
            {48,VK_DESCRIPTOR_TYPE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
            {81,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
            {gcn::kBindingStorageBuffer0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}}};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.bindingCount = 5; li.pBindings = bindings.data();
        check(vkCreateDescriptorSetLayout(device, &li, nullptr, &layout));
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &layout;
        check(vkCreatePipelineLayout(device, &pli, nullptr, &pipeline_layout));
        std::array<VkDescriptorPoolSize,5> sizes{};
        for (int i = 0; i < 5; ++i) sizes[i] = {bindings[i].descriptorType,1};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 5; dpi.pPoolSizes = sizes.data();
        check(vkCreateDescriptorPool(device, &dpi, nullptr, &descriptors));
        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; da.descriptorPool = descriptors; da.descriptorSetCount = 1; da.pSetLayouts = &layout;
        check(vkAllocateDescriptorSets(device, &da, &set));
        VkDescriptorBufferInfo ub{buffer,0,sizeof(gcn::StageParams)};
        std::array<VkDescriptorImageInfo,3> images = {{{{},source_view,VK_IMAGE_LAYOUT_GENERAL},{sampler,{},VK_IMAGE_LAYOUT_UNDEFINED},{ {},dest_view,VK_IMAGE_LAYOUT_GENERAL}}};
        std::array<VkWriteDescriptorSet,4> writes{};
        for (int i = 0; i < 4; ++i) {
            auto& w = writes[i]; w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = set;
            w.dstBinding = bindings[i].binding; w.descriptorType = bindings[i].descriptorType; w.descriptorCount = 1;
            if (i == 0) w.pBufferInfo = &ub; else w.pImageInfo = &images[i-1];
        }
        vkUpdateDescriptorSets(device, 4, writes.data(), 0, nullptr);
        {
            VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = 4096; ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            check(vkCreateBuffer(device, &ci, nullptr, &cb));
            VkMemoryRequirements cr{}; vkGetBufferMemoryRequirements(device, cb, &cr);
            VkMemoryAllocateInfo ca2{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ca2.allocationSize = cr.size;
            ca2.memoryTypeIndex = memory_type(cr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            check(vkAllocateMemory(device, &ca2, nullptr, &cb_mem)); check(vkBindBufferMemory(device, cb, cb_mem, 0));
            void* map = nullptr; check(vkMapMemory(device, cb_mem, 0, VK_WHOLE_SIZE, 0, &map));
            cb_map = static_cast<std::uint32_t*>(map);
            std::memset(cb_map, 0, 4096);
            VkDescriptorBufferInfo cbi{cb, 0, 4096};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = set; w.dstBinding = gcn::kBindingStorageBuffer0; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.descriptorCount = 1;
            w.pBufferInfo = &cbi;
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        }
        return true;
    }
    bool run(bool unorm, bool force, bool gradients, float x, float y, int expected_x, int expected_y,
             bool branch = false) {
        gcn::Program p;
        auto add = [&](gcn::Inst in) { in.offset = uint32_t(p.insts.size() * 8); in.size = 2; p.insts.push_back(in); };
        auto mov = [&](int reg, float value) {
            gcn::Inst in; in.enc = gcn::Enc::VOP1; in.op = 1; in.dst = reg;
            in.src0 = gcn::kLiteral; in.has_literal = true; in.literal = std::bit_cast<uint32_t>(value); add(in);
        };
        int v = 0;
        if (gradients) { mov(v++, 0.0625f); mov(v++, 0); mov(v++, 0); mov(v++, 0.0625f); }
        mov(v++, x); mov(v++, y);
        if (branch) {
            gcn::Inst jump; jump.enc = gcn::Enc::SOPP; jump.op = 2; jump.imm = 3; add(jump);
            mov(v - 2, 0);  // unreachable poison; exercises the dispatcher fallback
        }
        gcn::Inst sample; sample.enc = gcn::Enc::MIMG; sample.op = gradients ? 34 : 39;
        sample.srsrc = 0; sample.ssamp = 8; sample.vaddr = 0; sample.vdata = 16; sample.dmask = 15; sample.unorm = unorm; add(sample);
        mov(24, 0); mov(25, 0);
        gcn::Inst store = sample; store.op = 8; store.vaddr = 24; store.unorm = false; add(store);
        gcn::Inst end; end.enc = gcn::Enc::SOPP; end.op = 1; add(end);
        gcn::TranslateOptions o; o.stage = gcn::Stage::Compute; o.rsrc2 = 12 << 1; o.cs_threads[0] = 1;
        o.sampler_force_unnormalized = {force};
        const auto translated = gcn::translate(p, o);
        if (!translated.ok()) throw std::runtime_error(translated.errors[0]);
        spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_2);
        validator.SetMessageConsumer([](auto, auto, const auto&, const char* msg) { std::fprintf(stderr,"%s\n",msg); });
        if (!validator.Validate(translated.spirv)) throw std::runtime_error("invalid SPIR-V");
        auto* params = static_cast<gcn::StageParams*>(mapped); params->user_sgpr[8] = force ? 1u << 15 : 0;
        params->user_sgpr[3] = 0xfac;  // the T#'s DST_SEL: X, Y, Z, W, which the store goes through
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = translated.spirv.size()*4; mi.pCode = translated.spirv.data(); VkShaderModule module{};
        check(vkCreateShaderModule(device, &mi, nullptr, &module));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.layout = pipeline_layout;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = module; ci.stage.pName = "main"; VkPipeline pipeline{};
        check(vkCreateComputePipelines(device, {}, 1, &ci, nullptr, &pipeline));
        begin(); barrier(VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{}; copy.bufferOffset = 2048; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent = {1,1,1};
        vkCmdCopyImageToBuffer(cmd, dest, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &copy);
        barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT); submit();
        const float* result = reinterpret_cast<float*>(static_cast<char*>(mapped) + 2048);
        const bool ok = std::abs(result[0]-expected_x)<0.01f && std::abs(result[1]-expected_y)<0.01f &&
                        std::abs(result[2]-0.25f)<0.01f && std::abs(result[3]-1)<0.01f;
        std::printf("%s unorm=%d force=%d gradients=%d uv=(%g,%g): got (%g,%g), expected (%d,%d)\n",
                    ok ? "PASS" : "FAIL",unorm,force,gradients,x,y,result[0],result[1],expected_x,expected_y);
        vkDestroyPipeline(device, pipeline, nullptr); vkDestroyShaderModule(device, module, nullptr);
        return ok;
    }
    ~Fixture() {
        if (device) {
            vkDeviceWaitIdle(device);
            vkDestroyDescriptorPool(device,descriptors,nullptr); vkDestroyPipelineLayout(device,pipeline_layout,nullptr);
            vkDestroyDescriptorSetLayout(device,layout,nullptr); vkDestroySampler(device,sampler,nullptr);
            vkDestroyImageView(device,source_view,nullptr); vkDestroyImageView(device,dest_view,nullptr);
            vkDestroyImageView(device,cube_view,nullptr); vkDestroyImage(device,cube_image,nullptr); vkFreeMemory(device,cube_mem,nullptr);
            if (cb_map) vkUnmapMemory(device,cb_mem);
            vkDestroyBuffer(device,cb,nullptr); vkFreeMemory(device,cb_mem,nullptr);
            if (guest_map) vkUnmapMemory(device,guest_mem);
            vkDestroyBuffer(device,guest,nullptr); vkFreeMemory(device,guest_mem,nullptr);
            vkDestroyImage(device,source,nullptr); vkDestroyImage(device,dest,nullptr);
            vkFreeMemory(device,source_mem,nullptr); vkFreeMemory(device,dest_mem,nullptr);
            if (mapped) vkUnmapMemory(device,buffer_mem);
            vkDestroyBuffer(device,buffer,nullptr); vkFreeMemory(device,buffer_mem,nullptr);
            vkDestroyCommandPool(device,pool,nullptr); vkDestroyDevice(device,nullptr);
        }
        if (instance) vkDestroyInstance(instance,nullptr);
    }
};

int main() try {
    Fixture f;
    if (!f.init()) { std::puts("SKIP: suitable Vulkan device unavailable"); return 77; }
    bool ok = true;
    ok &= f.run(false,false,false,0.3125f,0.6875f,2,5);
    ok &= f.run(false,false,false,1.3125f,-0.3125f,2,5);
    ok &= f.run(true,false,false,2.5f,5.5f,2,5);
    ok &= f.run(false,true,false,2.5f,5.5f,2,5);
    ok &= f.run(false,false,true,0.3125f,0.6875f,2,5);
    ok &= f.run(true,false,true,2.5f,5.5f,2,5);
    ok &= f.run(false,false,false,0.3125f,0.6875f,2,5,true);
    f.init_cube();
    ok &= f.run_cube(1.375f, 1.625f, 3.0f, 1, 2, 3);    // face 3 of the first cube
    ok &= f.run_cube(1.125f, 1.875f, 0.0f, 0, 3, 0);    // face 0, far corner row
    ok &= f.run_cube(1.875f, 1.125f, 11.0f, 3, 0, 9);   // second cube (8 + face 3) is layer 9
    ok &= f.run_store_select(0xfac, {1, 2, 3, 4}, "X Y Z W: as given");
    ok &= f.run_store_select(0x800, {4, 0, 0, 0}, "the A8 select 0 0 0 X: w to R, nothing else");
    ok &= f.run_store_select(0xf2e, {3, 2, 1, 4}, "Z Y X W: R and B swapped back");
    ok &= f.run_store_select(0x924, {1, 0, 0, 0}, "X X X X: the first data component");
    f.init_guest();
    ok &= f.run_upload_copy();
    ok &= f.run_buffer_store_select();
    // Typed loads: the identity select, whatever the V#'s (32_32_32_32 float
    // element (5, 6, 7, 8) read through four formats, walked and bound).
    for (const int cb : {0, 2}) {
        const std::uint32_t xyzw = 14u << 15 | 7u << 12 | 0xfac, zyxw = 14u << 15 | 7u << 12 | 0xf2e, zero = 14u << 15 | 7u << 12;
        ok &= f.run_typed_load(cb, 14, 4, xyzw, {5, 6, 7, 8}, "X Y Z W");
        ok &= f.run_typed_load(cb, 14, 4, zyxw, {5, 6, 7, 8}, "Z Y X W, ignored");
        ok &= f.run_typed_load(cb, 14, 4, zero, {5, 6, 7, 8}, "0 0 0 0, ignored");
        ok &= f.run_typed_load(cb, 13, 4, xyzw, {5, 6, 7, 0}, "a three-component format into four VGPRs: XYZ0");
        ok &= f.run_typed_load(cb, 11, 3, xyzw, {5, 6, 0, -1}, "two components into three: XY0");
        ok &= f.run_typed_load(cb, 4, 1, zyxw, {5, -1, -1, -1}, "one component");
    }
    ok &= f.run_format_load_select();
    ok &= f.run_store_dmask(0xf, {1, 2, 3, 4});
    ok &= f.run_store_dmask(0x1, {1, 0, 0, 0});
    ok &= f.run_store_dmask(0x9, {1, 0, 0, 2});
    ok &= f.run_store_dmask(0x6, {0, 1, 2, 0});
    // Ranged scalar loads: ten dwords of range (40 bytes, stride 0; 2 records
    // of 20 bytes); the offset's two low bits ignored; past the range 0.
    for (const int cb : {1, 2}) {
        for (const bool bound : {true, false}) {
            if (cb == 2 && !bound) continue;  // the no-fallback variant always reads the binding
            ok &= f.run_ranged_load(cb, bound, 16, 0, 40, {104, 105, 106, 107}, "inside");
            ok &= f.run_ranged_load(cb, bound, 33, 0, 40, {108, 109, 0, 0}, "low bits ignored, straddling the end");
            ok &= f.run_ranged_load(cb, bound, 40, 0, 40, {0, 0, 0, 0}, "past the end");
            ok &= f.run_ranged_load(cb, bound, 0, 20, 2, {100, 101, 102, 103}, "records times the stride");
            ok &= f.run_ranged_load(cb, bound, 32, 20, 2, {108, 109, 0, 0}, "records times the stride, straddling");
            ok &= f.run_ranged_load(cb, bound, 0, 0, 0, {0, 0, 0, 0}, "an empty V#");
            ok &= f.run_ranged_load(cb, bound, 36, 0, 38, {109, 0, 0, 0}, "a dword in range when it starts inside");
        }
    }
    // Buffer stores at a register or constant offset, bound (no fallback) and walked.
    ok &= f.run_bound_store(2, 0, false, 0, 64, "inside");
    ok &= f.run_bound_store(2, 24, false, 0, 48, "straddling the end");
    ok &= f.run_bound_store(2, 48, false, 0, 48, "past the end");
    ok &= f.run_bound_store(2, 3, false, 0, 64, "the low bits ignored");
    ok &= f.run_bound_store(2, 0, false, 16, 2, "records times the stride, straddling");
    ok &= f.run_bound_store(2, 99, true, 0, 64, "a constant offset");
    ok &= f.run_bound_store(0, 8, false, 0, 64, "walked");
    // GDS counters: one subgroup, several in a workgroup, several workgroups;
    // EXEC partly off; M0's base plus the offset picking the word; consume's
    // values from before; an unbound dword; the counter into indirect
    // arguments on the GPU's timeline.
    // Each as the translation and as the lift - an atomic an invocation, and
    // one a subgroup (TranslateOptions::gds_subgroup): the same counts, each
    // slot of the range handed out once, whichever.
    for (const Fixture::Path p : {Fixture::Path::kTranslation, Fixture::Path::kLiftInvocation, Fixture::Path::kLiftSubgroup}) {
        f.path = p;
        ok &= f.run_gds_append(32, 1, Fixture::Lanes::kAll, 0, 4, 0, "one subgroup");
        ok &= f.run_gds_append(256, 1, Fixture::Lanes::kAll, 0, 4, 10, "several subgroups");
        ok &= f.run_gds_append(256, 8, Fixture::Lanes::kAll, 0, 4, 3, "several workgroups");
        ok &= f.run_gds_append(64, 4, Fixture::Lanes::kOdd, 0, 4, 1, "the odd lanes only");
        ok &= f.run_gds_append(128, 2, Fixture::Lanes::kBelow, 45, 4, 0, "the first 45 lanes of each group");
        ok &= f.run_gds_append(64, 2, Fixture::Lanes::kAll, 0, 0, 100, "offset 0: the first word");
        ok &= f.run_gds_consume(256, 1, 256, 300, "every lane");
        ok &= f.run_gds_consume(64, 4, 37, 200, "37 lanes of each group");
        ok &= f.run_gds_unbound();
        ok &= f.run_counter_to_args(1, 37);
        ok &= f.run_counter_to_args(16, 64);
    }
    f.path = Fixture::Path::kTranslation;
    ok &= f.run_gds_off();
    return ok ? 0 : 1;
} catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
