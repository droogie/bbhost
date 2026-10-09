// The vertex stage on a device without the subgroup built-ins there
// (MoltenVK).
//
// TranslateOptions::lane_from_scan works the lane out from an exclusive add
// and a ballot instead of reading SubgroupLocalInvocationId; the SPIR-V must
// validate and read no such built-in, and the module without the option still
// reads it.
//
// Then, on a Vulkan device (MoltenVK is found through portability
// enumeration), the shaders run: points from a translated vertex shader whose
// colours come from a compare's mask bit, an EXEC-masked write and the lane's
// count must come out per vertex for every wave size, with the lane from the
// scan and - where the device lists the vertex stage - from the built-in.
// Without a device only the SPIR-V is checked.
#include "gcn/isa.h"
#include "gcn/spirv.h"
#include "gcn/translate.h"

#include <spirv-tools/libspirv.hpp>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_fail;
    }
}

// Sea Islands encodings of the few instructions the programs use.
using Words = std::vector<std::uint32_t>;
void vop1(Words& w, std::uint32_t op, std::uint32_t vdst, std::uint32_t src0) { w.push_back(0x7e000000u | vdst << 17 | op << 9 | src0); }
void vop2(Words& w, std::uint32_t op, std::uint32_t vdst, std::uint32_t vsrc1, std::uint32_t src0) {
    w.push_back(op << 25 | vdst << 17 | vsrc1 << 9 | src0);
}
void vopc(Words& w, std::uint32_t op, std::uint32_t vsrc1, std::uint32_t src0) { w.push_back(0x7c000000u | op << 17 | vsrc1 << 9 | src0); }
void vop3(Words& w, std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t src1) {
    w.push_back(0xd0000000u | op << 17 | vdst);
    w.push_back(src0 | src1 << 9);
}
void sop1(Words& w, std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc) { w.push_back(0xbe800000u | sdst << 16 | op << 8 | ssrc); }
void export_(Words& w, std::uint32_t target, std::uint32_t v0, std::uint32_t v1, std::uint32_t v2, std::uint32_t v3, bool done) {
    w.push_back(0xf8000000u | (1u << 12) | (done ? 1u << 11 : 0u) | target << 4 | 0xf);
    w.push_back(v0 | v1 << 8 | v2 << 16 | v3 << 24);
}
void mov_literal(Words& w, std::uint32_t vdst, float value) {
    vop1(w, 1, vdst, 255);
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    w.push_back(bits);
}
constexpr std::uint32_t kV = 256, kVcc = 106, kExec = 126, kZero = 128, kOne = 129, kHalf = 240, kOneF = 242, kMinusOneF = 243, kMinusOne = 193;
constexpr std::uint32_t kEndpgm = 0xbf810000u;
constexpr std::uint32_t kPointSize = (1u << 16) | (1u << 20);  // PA_CL_VS_OUT_CNTL: USE_VTX_POINT_SIZE, the misc vector

// Point i at pixel i of a width x 1 target, coloured (i odd, i odd under an
// EXEC of the odd lanes, i < 7, lane count < 64): the first two read a
// compare's mask bit at the lane, the second through s_and_saveexec.
Words lane_program(std::uint32_t width) {
    Words w;
    vop1(w, 6, 1, kV + 0);                         // v_cvt_f32_u32 v1, v0
    vop2(w, 3, 1, 1, kHalf);                       // v_add_f32 v1, 0.5, v1
    vop2(w, 8, 1, 1, 255);                         // v_mul_f32 v1, 2 / width, v1
    const float scale = 2.0f / static_cast<float>(width);
    std::uint32_t bits;
    std::memcpy(&bits, &scale, 4);
    w.push_back(bits);
    vop2(w, 3, 1, 1, kMinusOneF);                  // v_add_f32 v1, -1.0, v1
    vop1(w, 1, 4, kOneF);                          // v_mov_b32 v4, 1.0
    vop1(w, 1, 6, kZero);                          // v_mov_b32 v6, 0
    vop2(w, 27, 7, 0, kOne);                       // v_and_b32 v7, 1, v0
    vopc(w, 0xc2, 7, kOne);                        // v_cmp_eq_u32 vcc, 1, v7
    vop2(w, 0, 3, 4, kZero);                       // v_cndmask_b32 v3, 0, v4, vcc
    vop1(w, 1, 8, kZero);                          // v_mov_b32 v8, 0
    sop1(w, 36, 10, kVcc);                         // s_and_saveexec_b64 s[10:11], vcc
    vop1(w, 1, 8, kOneF);                          // v_mov_b32 v8, 1.0
    sop1(w, 4, kExec, 10);                         // s_mov_b64 exec, s[10:11]
    vopc(w, 0xc4, 0, 128 + 7);                     // v_cmp_gt_u32 vcc, 7, v0
    vop2(w, 0, 9, 4, kZero);                       // v_cndmask_b32 v9, 0, v4, vcc
    vop3(w, 0x123, 2, kMinusOne, kZero);           // v_mbcnt_lo_u32_b32 v2, -1, 0
    vop3(w, 0x124, 2, kMinusOne, kV + 2);          // v_mbcnt_hi_u32_b32 v2, -1, v2
    vopc(w, 0xc4, 2, 255);                         // v_cmp_gt_u32 vcc, 64, v2
    w.push_back(64);
    vop2(w, 0, 10, 4, kZero);                      // v_cndmask_b32 v10, 0, v4, vcc
    export_(w, 32, 3, 8, 9, 10, false);                // exp param0 v3, v8, v9, v10
    export_(w, 12, 1, 6, 6, 4, false);                 // exp pos0 v1, 0, 0, 1.0
    export_(w, 13, 4, 4, 4, 4, true);                  // exp pos1 (point size) 1.0 done
    w.push_back(kEndpgm);
    return w;
}

gcn::TranslateResult translate_vs(const Words& words, bool lane_from_scan, std::uint32_t vs_out_cntl) {
    const gcn::Program p = gcn::decode(words.data(), words.size());
    if (!p.errors.empty()) throw std::runtime_error("decode: " + p.errors[0].what);
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Vertex;
    o.vs_out_cntl = vs_out_cntl;
    o.descriptor_set = 0;
    o.lane_from_scan = lane_from_scan;
    gcn::TranslateResult r = gcn::translate(p, o);
    if (!r.ok()) throw std::runtime_error("translate: " + r.errors[0]);
    return r;
}

std::string validate(const std::vector<std::uint32_t>& spirv) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    return tools.Validate(spirv) ? std::string() : (msg.empty() ? "invalid" : msg);
}

// Whether the module decorates something BuiltIn `builtin`, and how many
// instructions of `op` it has.
bool has_builtin(const std::vector<std::uint32_t>& spv, std::uint32_t builtin) {
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t op = spv[i] & 0xffff, len = spv[i] >> 16;
        if (!len) break;
        if (op == 71 && len >= 4 && spv[i + 2] == 11 && spv[i + 3] == builtin) return true;  // OpDecorate BuiltIn
        i += len;
    }
    return false;
}
int count_op(const std::vector<std::uint32_t>& spv, std::uint32_t opcode) {
    int n = 0;
    for (std::size_t i = 5; i < spv.size();) {
        const std::uint32_t op = spv[i] & 0xffff, len = spv[i] >> 16;
        if (!len) break;
        n += op == opcode;
        i += len;
    }
    return n;
}

void spirv_checks() {
    const Words lanes = lane_program(128);
    const gcn::TranslateResult builtin = translate_vs(lanes, false, kPointSize), scan = translate_vs(lanes, true, kPointSize);
    expect(validate(builtin.spirv).empty(), "the vertex shader with the built-in lane validates: " + validate(builtin.spirv));
    expect(validate(scan.spirv).empty(), "the vertex shader with the lane from a scan validates: " + validate(scan.spirv));
    expect(has_builtin(builtin.spirv, 41), "without lane_from_scan the lane is SubgroupLocalInvocationId");
    expect(!has_builtin(scan.spirv, 41) && !has_builtin(scan.spirv, 36), "with lane_from_scan no subgroup built-in is read");
    expect(count_op(scan.spirv, 349) == 1 && count_op(builtin.spirv, 349) == 0, "the scan is one OpGroupNonUniformIAdd");
    // One ballot at entry serves the lane and EXEC.
    expect(count_op(scan.spirv, 339) == count_op(builtin.spirv, 339), "the scan takes no ballot of its own");
    // A pixel stage, where a discard runs before it, validates too.
    {
        gcn::TranslateOptions o;
        o.stage = gcn::Stage::Pixel;
        o.descriptor_set = 1;
        o.ps_clip_discard = 1 | 2u << 8;
        o.lane_from_scan = true;
        Words ps;
        vop3(ps, 0x123, 2, kMinusOne, kZero);
        export_(ps, 0, 2, 2, 2, 2, true);
        ps.push_back(kEndpgm);
        const gcn::TranslateResult r = gcn::translate(gcn::decode(ps.data(), ps.size()), o);
        expect(r.ok() && validate(r.spirv).empty(), "a pixel stage with lane_from_scan validates");
    }
    std::printf("gcn_vertex_stage_test: SPIR-V checks done\n");
}

// ---- the device ----
void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + ": VkResult " + std::to_string(r));
}

struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkBuffer params = VK_NULL_HANDLE, readback = VK_NULL_HANDLE;
    VkDeviceMemory params_mem = VK_NULL_HANDLE, readback_mem = VK_NULL_HANDLE;
    void* readback_map = nullptr;
    bool tessellation = false, geometry = false;
    std::uint32_t subgroup_stages = 0;
    std::string name;
    std::vector<VkShaderModule> modules;

    std::uint32_t memory_type(std::uint32_t bits, VkMemoryPropertyFlags flags) {
        VkPhysicalDeviceMemoryProperties p{};
        vkGetPhysicalDeviceMemoryProperties(phys, &p);
        for (std::uint32_t i = 0; i < p.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (p.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        throw std::runtime_error("no memory type");
    }
    void buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& b, VkDeviceMemory& mem, void** map) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = usage;
        check(vkCreateBuffer(device, &bi, nullptr, &b), "vkCreateBuffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, b, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &ai, nullptr, &mem), "vkAllocateMemory");
        check(vkBindBufferMemory(device, b, mem, 0), "vkBindBufferMemory");
        void* p = nullptr;
        check(vkMapMemory(device, mem, 0, VK_WHOLE_SIZE, 0, &p), "vkMapMemory");
        std::memset(p, 0, size);
        if (map) *map = p;
    }

    bool init() {
        std::uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> ie(n);
        vkEnumerateInstanceExtensionProperties(nullptr, &n, ie.data());
        bool portability = false;
        for (const VkExtensionProperties& e : ie) portability |= std::strcmp(e.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
        const char* exts[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (portability) {
            ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            ici.enabledExtensionCount = 1;
            ici.ppEnabledExtensionNames = exts;
        }
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return false;
        std::uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        std::uint32_t family = 0;
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f.pNext = &v12;
            vkGetPhysicalDeviceFeatures2(candidate, &f);
            VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            props.pNext = &sub;
            vkGetPhysicalDeviceProperties2(candidate, &props);
            const VkSubgroupFeatureFlags ops = VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
            if (props.properties.apiVersion < VK_API_VERSION_1_2 || !f.features.shaderInt64 || !v12.bufferDeviceAddress ||
                (sub.supportedOperations & ops) != ops) {
                continue;
            }
            std::uint32_t qn = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &qn, nullptr);
            std::vector<VkQueueFamilyProperties> queues(qn);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &qn, queues.data());
            for (std::uint32_t i = 0; i < qn && !phys; ++i) {
                if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    phys = candidate;
                    family = i;
                }
            }
            if (phys) {
                tessellation = f.features.tessellationShader == VK_TRUE;
                geometry = f.features.geometryShader == VK_TRUE;
                subgroup_stages = sub.supportedStages;
                name = props.properties.deviceName;
                break;
            }
        }
        if (!phys) return false;
        std::uint32_t dn = 0;
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &dn, nullptr);
        std::vector<VkExtensionProperties> de(dn);
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &dn, de.data());
        std::vector<const char*> dexts;
        for (const VkExtensionProperties& e : de) {
            if (std::strcmp(e.extensionName, "VK_KHR_portability_subset") == 0) dexts.push_back("VK_KHR_portability_subset");
        }
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        v12.bufferDeviceAddress = VK_TRUE;
        VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f.pNext = &v12;
        f.features.shaderInt64 = VK_TRUE;
        f.features.tessellationShader = tessellation ? VK_TRUE : VK_FALSE;
        f.features.geometryShader = geometry ? VK_TRUE : VK_FALSE;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.pNext = &f;
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        di.enabledExtensionCount = static_cast<std::uint32_t>(dexts.size());
        di.ppEnabledExtensionNames = dexts.data();
        check(vkCreateDevice(phys, &di, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, family, 0, &queue);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = family;
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(device, &pi, nullptr, &pool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ca.commandPool = pool;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device, &ca, &cmd), "vkAllocateCommandBuffers");
        // The vertex stage's params block (gcn::StageParams): nothing in it is read.
        buffer(sizeof(gcn::StageParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, params, params_mem, nullptr);
        buffer(64 * 64 * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, readback, readback_mem, &readback_map);
        VkDescriptorSetLayoutBinding b{gcn::kBindingParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL_GRAPHICS, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 1;
        li.pBindings = &b;
        check(vkCreateDescriptorSetLayout(device, &li, nullptr, &set_layout), "vkCreateDescriptorSetLayout");
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &set_layout;
        check(vkCreatePipelineLayout(device, &pli, nullptr, &layout), "vkCreatePipelineLayout");
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &size;
        check(vkCreateDescriptorPool(device, &dpi, nullptr, &descriptors), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        da.descriptorPool = descriptors;
        da.descriptorSetCount = 1;
        da.pSetLayouts = &set_layout;
        check(vkAllocateDescriptorSets(device, &da, &set), "vkAllocateDescriptorSets");
        VkDescriptorBufferInfo bi{params, 0, sizeof(gcn::StageParams)};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = gcn::kBindingParams;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        return true;
    }

    VkShaderModule module(const std::vector<std::uint32_t>& spirv) {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = spirv.size() * 4;
        mi.pCode = spirv.data();
        VkShaderModule m = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device, &mi, nullptr, &m), "vkCreateShaderModule");
        modules.push_back(m);
        return m;
    }

    struct Stages {
        VkShaderModule vs = VK_NULL_HANDLE, tcs = VK_NULL_HANDLE, tes = VK_NULL_HANDLE, gs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    };
    // Draws `vertices` into a width x height RGBA32F target cleared to -2 and
    // returns its texels. Null when the pipeline cannot be made.
    std::vector<float> draw(const Stages& st, VkPrimitiveTopology topology, std::uint32_t vertices, std::uint32_t width, std::uint32_t height,
                            VkCullModeFlags cull = VK_CULL_MODE_NONE, VkFrontFace front = VK_FRONT_FACE_COUNTER_CLOCKWISE) {
        const VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {width, height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VkImage image = VK_NULL_HANDLE;
        check(vkCreateImage(device, &ici, nullptr, &image), "vkCreateImage");
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits, 0);
        VkDeviceMemory image_mem = VK_NULL_HANDLE;
        check(vkAllocateMemory(device, &ai, nullptr, &image_mem), "vkAllocateMemory");
        check(vkBindImageMemory(device, image, image_mem, 0), "vkBindImageMemory");
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        check(vkCreateImageView(device, &vci, nullptr, &view), "vkCreateImageView");
        VkAttachmentDescription att{};
        att.format = format;
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &ref;
        VkSubpassDependency dep{};
        dep.srcSubpass = 0;
        dep.dstSubpass = VK_SUBPASS_EXTERNAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpi.attachmentCount = 1;
        rpi.pAttachments = &att;
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sub;
        rpi.dependencyCount = 1;
        rpi.pDependencies = &dep;
        VkRenderPass pass = VK_NULL_HANDLE;
        check(vkCreateRenderPass(device, &rpi, nullptr, &pass), "vkCreateRenderPass");
        VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbi.renderPass = pass;
        fbi.attachmentCount = 1;
        fbi.pAttachments = &view;
        fbi.width = width;
        fbi.height = height;
        fbi.layers = 1;
        VkFramebuffer fb = VK_NULL_HANDLE;
        check(vkCreateFramebuffer(device, &fbi, nullptr, &fb), "vkCreateFramebuffer");

        std::vector<VkPipelineShaderStageCreateInfo> stages;
        const auto stage = [&](VkShaderStageFlagBits bit, VkShaderModule m) {
            if (!m) return;
            VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            si.stage = bit;
            si.module = m;
            si.pName = "main";
            stages.push_back(si);
        };
        stage(VK_SHADER_STAGE_VERTEX_BIT, st.vs);
        stage(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, st.tcs);
        stage(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, st.tes);
        stage(VK_SHADER_STAGE_GEOMETRY_BIT, st.gs);
        stage(VK_SHADER_STAGE_FRAGMENT_BIT, st.fs);
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = st.tes ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : topology;
        VkPipelineTessellationStateCreateInfo ts{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
        ts.patchControlPoints = 3;
        VkViewport viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
        VkRect2D scissor{{0, 0}, {width, height}};
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.pViewports = &viewport;
        vp.scissorCount = 1;
        vp.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = cull;
        rs.frontFace = front;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState blend{};
        blend.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &blend;
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gpci.stageCount = static_cast<std::uint32_t>(stages.size());
        gpci.pStages = stages.data();
        gpci.pVertexInputState = &vi;
        gpci.pInputAssemblyState = &ia;
        if (st.tes) gpci.pTessellationState = &ts;
        gpci.pViewportState = &vp;
        gpci.pRasterizationState = &rs;
        gpci.pMultisampleState = &ms;
        gpci.pColorBlendState = &cb;
        gpci.layout = layout;
        gpci.renderPass = pass;
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::vector<float> texels;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline) == VK_SUCCESS) {
            check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
            VkClearValue clear{};
            for (float& c : clear.color.float32) c = -2.0f;
            VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rbi.renderPass = pass;
            rbi.framebuffer = fb;
            rbi.renderArea = scissor;
            rbi.clearValueCount = 1;
            rbi.pClearValues = &clear;
            vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
            vkCmdDraw(cmd, vertices, 1, 0, 0);
            vkCmdEndRenderPass(cmd);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {width, height, 1};
            vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &copy);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
            check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
            const float* px = static_cast<const float*>(readback_map);
            texels.assign(px, px + static_cast<std::size_t>(width) * height * 4);
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        vkDestroyFramebuffer(device, fb, nullptr);
        vkDestroyRenderPass(device, pass, nullptr);
        vkDestroyImageView(device, view, nullptr);
        vkDestroyImage(device, image, nullptr);
        vkFreeMemory(device, image_mem, nullptr);
        return texels;
    }

    ~Device() {
        if (device) {
            vkDeviceWaitIdle(device);
            for (VkShaderModule m : modules) vkDestroyShaderModule(device, m, nullptr);
            vkDestroyDescriptorPool(device, descriptors, nullptr);
            vkDestroyPipelineLayout(device, layout, nullptr);
            vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            vkDestroyBuffer(device, params, nullptr);
            vkFreeMemory(device, params_mem, nullptr);
            vkDestroyBuffer(device, readback, nullptr);
            vkFreeMemory(device, readback_mem, nullptr);
            vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

// The fragment stage: input 0 as the colour.
std::vector<std::uint32_t> passthrough_fs() {
    spv::Module m;
    m.capability(spv::CapShader);
    m.memory_model(0, 1);
    const spv::Id t_void = m.type_void(), t_v4f = m.type_vector(m.type_float(32), 4);
    const spv::Id in = m.global_variable(m.type_pointer(spv::ScInput, t_v4f), spv::ScInput);
    const spv::Id out = m.global_variable(m.type_pointer(spv::ScOutput, t_v4f), spv::ScOutput);
    m.decorate(in, spv::DecLocation, {0});
    m.decorate(out, spv::DecLocation, {0});
    const spv::Id fn = m.begin_function(t_void, m.type_function(t_void, {}));
    m.label();
    m.store(out, m.load(t_v4f, in));
    m.emit_void(spv::OpReturn, {});
    m.end_function();
    m.entry_point(spv::EmFragment, fn, "main", {in, out});
    m.execution_mode(fn, spv::ExOriginUpperLeft);
    return m.assemble();
}

void lane_checks(Device& d, VkShaderModule fs) {
    constexpr std::uint32_t kWidth = 128;
    for (int from_scan = 1; from_scan >= 0; --from_scan) {
        if (!from_scan && !(d.subgroup_stages & VK_SHADER_STAGE_VERTEX_BIT)) {
            std::printf("gcn_vertex_stage_test: the device has no subgroup operations in its vertex stage: built-in lane not run\n");
            continue;
        }
        const char* how = from_scan ? "lane from a scan" : "lane built-in";
        Device::Stages st;
        st.vs = d.module(translate_vs(lane_program(kWidth), from_scan != 0, kPointSize).spirv);
        st.fs = fs;
        int checked = 0;
        for (std::uint32_t n : {1u, 5u, 31u, 32u, 33u, 63u, 64u, 65u, 100u, 128u}) {
            const std::vector<float> px = d.draw(st, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, n, kWidth, 1);
            if (px.empty()) {
                expect(false, std::string(how) + ": the point pipeline was not created");
                break;
            }
            int bad = 0;
            for (std::uint32_t i = 0; i < kWidth; ++i) {
                const float* c = &px[i * 4];
                float want[4] = {-2, -2, -2, -2};
                if (i < n) {
                    want[0] = want[1] = (i & 1) ? 1.0f : 0.0f;
                    want[2] = i < 7 ? 1.0f : 0.0f;
                    want[3] = 1.0f;
                }
                if (c[0] != want[0] || c[1] != want[1] || c[2] != want[2] || c[3] != want[3]) {
                    if (bad++ < 4) {
                        std::printf("  %s, %u points: pixel %u is (%g, %g, %g, %g), not (%g, %g, %g, %g)\n", how, n, i, c[0], c[1], c[2], c[3],
                                    want[0], want[1], want[2], want[3]);
                    }
                }
            }
            expect(bad == 0, std::string(how) + ": " + std::to_string(n) + " points");
            ++checked;
        }
        std::printf("gcn_vertex_stage_test: %s: %d draws of points checked\n", how, checked);
    }
}

}  // namespace

int main() try {
    spirv_checks();
    Device d;
    if (!d.init()) {
        std::printf("gcn_vertex_stage_test: no suitable Vulkan device: device checks skipped\n");
    } else {
        std::printf("gcn_vertex_stage_test: %s (subgroup stages 0x%x)\n", d.name.c_str(), d.subgroup_stages);
        const VkShaderModule fs = d.module(passthrough_fs());
        lane_checks(d, fs);
    }
    if (g_fail) {
        std::printf("gcn_vertex_stage_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("gcn_vertex_stage_test: ok\n");
    return 0;
} catch (const std::exception& e) {
    std::printf("gcn_vertex_stage_test: %s\n", e.what());
    return 1;
}
