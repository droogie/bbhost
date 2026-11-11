// FSR 3 from bbhost's Linux build of AMD's FidelityFX library (src/ffx_vk), on a
// real Vulkan device (lavapipe in the tests), called as bbhost calls it for
// FSR 3: the library exports the FidelityFX API and nothing else, offers
// FSR 3.1.6 for upscaling before any device, and makes a context on this one
// (every pipeline from the generated shaders) that upscales a jittered
// synthetic picture to twice its size. The picture is drawn as a renderer
// draws it, each frame at the render size with that frame's jitter (+x: the
// picture moved right, +y: down, in render pixels, the convention the host
// hands the upscalers). After 64 frames FSR's output must be the same scene
// drawn at the output size, closer to it than FSR's own first frame (the
// frames were accumulated) and far closer than the same run with the jitter's
// sign turned round (the jitter is read as the host means it); the output is
// filled with a sentinel before the last dispatch, so a dispatch that writes
// nothing fails. With the Khronos validation layer installed, any validation
// error fails it too.
//   ffx_fsr3_test LIBRARY       exit 77 without a Vulkan device
#include <vulkan/vulkan.h>

#include "ffx_api.h"
#include "ffx_api_types.h"
#include "ffx_upscale.h"
#include "vk/ffx_api_vk.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_failures;
}

// FSR's own messages (its context's fpMessage), as the host logs them.
void on_ffx_message(uint32_t type, const wchar_t* message) {
    std::printf("FSR %s: %ls\n", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", message);
}

int g_validation_errors = 0;
VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                          const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        ++g_validation_errors;
        std::printf("validation error: %s\n", data->pMessage);
    }
    return VK_FALSE;
}

std::uint16_t to_half(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
    std::uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return static_cast<std::uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        std::uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) ++h;
        return static_cast<std::uint16_t>(sign | h);
    }
    std::uint32_t h = sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
    if (mant & 0x1000u) ++h;
    return static_cast<std::uint16_t>(h);
}

float from_half(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1f;
    std::uint32_t mant = h & 0x3ffu;
    std::uint32_t x;
    if (exp == 0) {
        if (!mant) {
            x = sign;
        } else {
            int e = -1;
            do {
                ++e;
                mant <<= 1;
            } while (!(mant & 0x400u));
            x = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | ((mant & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// The scene, in output pixels: smooth colour with detail near what the render
// size can carry.
void scene(double x, double y, float rgb[3]) {
    const double pi = 3.14159265358979;
    // (Detail at 5 and 7 output pixels: 2.5 and 3.5 render pixels, which one frame cannot hold.)
    const double fine = 0.15 * std::sin(2 * pi * x / 5.0) * std::sin(2 * pi * y / 7.0);
    rgb[0] = static_cast<float>(0.5 + 0.3 * std::sin(2 * pi * x / 11.0) * std::cos(2 * pi * y / 13.0) + fine);
    rgb[1] = static_cast<float>(0.4 + 0.25 * std::sin(2 * pi * (x + y) / 17.0) + fine);
    rgb[2] = static_cast<float>(0.3 + 0.2 * std::cos(2 * pi * (x - 0.5 * y) / 23.0));
}

float halton(std::uint32_t i, std::uint32_t base) {
    float f = 1.0f, r = 0.0f;
    for (; i; i /= base) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(i % base);
    }
    return r;
}

struct Vk {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem{};
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t w = 0, h = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* map = nullptr;
    VkDeviceSize size = 0;
};

std::uint32_t memory_type(const Vk& vk, std::uint32_t bits, VkMemoryPropertyFlags want) {
    for (std::uint32_t i = 0; i < vk.mem.memoryTypeCount; ++i)
        if ((bits >> i & 1) && (vk.mem.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

bool make_image(const Vk& vk, Image* im, VkFormat format, std::uint32_t w, std::uint32_t h, VkImageUsageFlags usage) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {w, h, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    if (vkCreateImage(vk.device, &ici, nullptr, &im->image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(vk.device, im->image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type(vk, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = memory_type(vk, req.memoryTypeBits, 0);
    if (vkAllocateMemory(vk.device, &mai, nullptr, &im->memory) != VK_SUCCESS) return false;
    vkBindImageMemory(vk.device, im->image, im->memory, 0);
    im->format = format;
    im->w = w;
    im->h = h;
    return true;
}

bool make_buffer(const Vk& vk, Buffer* b, VkDeviceSize size) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(vk.device, &bci, nullptr, &b->buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(vk.device, b->buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex =
        memory_type(vk, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(vk.device, &mai, nullptr, &b->memory) != VK_SUCCESS) return false;
    vkBindBufferMemory(vk.device, b->buffer, b->memory, 0);
    vkMapMemory(vk.device, b->memory, 0, VK_WHOLE_SIZE, 0, &b->map);
    b->size = size;
    return true;
}

void transition(VkCommandBuffer cb, Image* im, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = im->layout;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im->image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    im->layout = to;
}

void upload(VkCommandBuffer cb, Image* im, const Buffer& staging, VkDeviceSize offset) {
    transition(cb, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy r{};
    r.bufferOffset = offset;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {im->w, im->h, 1};
    vkCmdCopyBufferToImage(cb, staging.buffer, im->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
}

FfxApiResource resource(const Image& im, std::uint32_t usage, std::uint32_t state) {
    FfxApiResource r{};
    r.resource = reinterpret_cast<void*>(im.image);
    r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = ffxApiGetSurfaceFormatVK(im.format);
    r.description.width = im.w;
    r.description.height = im.h;
    r.description.depth = 1;
    r.description.mipCount = 1;
    r.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    r.description.usage = usage;
    r.state = state;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ffx_fsr3_test LIBRARY\n");
        return 2;
    }
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    check(lib != nullptr, "the library loads (RTLD_NOW: nothing left unresolved)");
    if (!lib) {
        std::printf("%s\n", dlerror());
        return 1;
    }
    auto create = reinterpret_cast<PfnFfxCreateContext>(dlsym(lib, "ffxCreateContext"));
    auto destroy = reinterpret_cast<PfnFfxDestroyContext>(dlsym(lib, "ffxDestroyContext"));
    auto configure = reinterpret_cast<PfnFfxConfigure>(dlsym(lib, "ffxConfigure"));
    auto query = reinterpret_cast<PfnFfxQuery>(dlsym(lib, "ffxQuery"));
    auto dispatch = reinterpret_cast<PfnFfxDispatch>(dlsym(lib, "ffxDispatch"));
    check(create && destroy && configure && query && dispatch, "the five FidelityFX API functions are exported");
    check(dlsym(lib, "ffxGetInterfaceVK") == nullptr && dlsym(lib, "_ZNSt6thread4joinEv") == nullptr,
          "nothing else is (no backend internals, no libstdc++)");
    if (!create || !destroy || !query || !dispatch) return 1;

    // The versions it offers for upscaling, before any device.
    {
        uint64_t count = 0;
        ffxQueryDescGetVersions q{};
        q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        q.outputCount = &count;
        const ffxReturnCode_t rc = query(nullptr, &q.header);
        std::vector<uint64_t> ids(count ? count : 1);
        std::vector<const char*> names(count ? count : 1);
        q.versionIds = ids.data();
        q.versionNames = names.data();
        const ffxReturnCode_t rc2 = count ? query(nullptr, &q.header) : rc;
        std::string list;
        for (uint64_t i = 0; i < count; ++i) list += std::string(i ? ", " : "") + (names[i] ? names[i] : "?");
        std::printf("versions: %s\n", list.c_str());
        check(rc == FFX_API_RETURN_OK && rc2 == FFX_API_RETURN_OK && count == 1 && list == "3.1.6", "FSR 3.1.6 is the one upscaler offered");
    }

    // A device: the first with a compute queue; the validation layer when it is installed.
    Vk vk;
    bool validation = false;
    {
        std::uint32_t n = 0;
        vkEnumerateInstanceLayerProperties(&n, nullptr);
        std::vector<VkLayerProperties> layers(n);
        vkEnumerateInstanceLayerProperties(&n, layers.data());
        for (const VkLayerProperties& l : layers) validation |= std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0;
        if (const char* e = std::getenv("FFX_TEST_VALIDATION"); e && e[0] == '0') validation = false;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* debug_ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    if (validation) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = &debug_ext;
    }
    if (vkCreateInstance(&ici, nullptr, &vk.instance) != VK_SUCCESS) {
        std::puts("SKIP: no Vulkan");
        return 77;
    }
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (validation) {
        auto make = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(vk.instance, "vkCreateDebugUtilsMessengerEXT"));
        VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        mci.pfnUserCallback = &on_message;
        if (make) make(vk.instance, &mci, nullptr, &messenger);
    }
    {
        std::uint32_t n = 0;
        vkEnumeratePhysicalDevices(vk.instance, &n, nullptr);
        std::vector<VkPhysicalDevice> phys(n);
        vkEnumeratePhysicalDevices(vk.instance, &n, phys.data());
        for (VkPhysicalDevice p : phys) {
            std::uint32_t q = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(p, &q, nullptr);
            std::vector<VkQueueFamilyProperties> qs(q);
            vkGetPhysicalDeviceQueueFamilyProperties(p, &q, qs.data());
            for (std::uint32_t k = 0; k < q && !vk.phys; ++k)
                if (qs[k].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    vk.phys = p;
                    vk.family = k;
                }
            if (vk.phys) break;
        }
    }
    if (!vk.phys) {
        std::puts("SKIP: no device with a compute queue");
        vkDestroyInstance(vk.instance, nullptr);
        return 77;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(vk.phys, &props);
    vkGetPhysicalDeviceMemoryProperties(vk.phys, &vk.mem);
    // The features bbhost turns on (host/gpu.cpp): 16-bit floats and integers where the device has them, which is what
    // makes the backend take FSR's 16-bit shaders, and compute shaders' derivative groups, which FSR 3's two pyramid
    // passes declare (DXC writes QuadReadAcross* that way).
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR have_derivatives{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    have12.pNext = &have_derivatives;
    VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    have.pNext = &have12;
    vkGetPhysicalDeviceFeatures2(vk.phys, &have);
    bool derivatives_ext = false;
    {
        std::uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(vk.phys, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> ext(n);
        vkEnumerateDeviceExtensionProperties(vk.phys, nullptr, &n, ext.data());
        for (const VkExtensionProperties& e : ext) derivatives_ext |= std::strcmp(e.extensionName, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME) == 0;
    }
    const bool derivatives = derivatives_ext && have_derivatives.computeDerivativeGroupLinear;
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR f_derivatives{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    f_derivatives.computeDerivativeGroupLinear = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.shaderFloat16 = have12.shaderFloat16;
    if (derivatives) f12.pNext = &f_derivatives;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    f2.features.shaderInt16 = have.features.shaderInt16;
    f2.features.shaderStorageImageReadWithoutFormat = have.features.shaderStorageImageReadWithoutFormat;
    f2.features.shaderStorageImageWriteWithoutFormat = have.features.shaderStorageImageWriteWithoutFormat;
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = vk.family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    const char* derivatives_name = VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qi;
    if (derivatives) {
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = &derivatives_name;
    }
    if (vkCreateDevice(vk.phys, &dci, nullptr, &vk.device) != VK_SUCCESS) {
        std::puts("SKIP: no device");
        return 77;
    }
    std::printf("device: %s (16-bit floats %s, compute derivatives %s, validation %s)\n", props.deviceName, f12.shaderFloat16 ? "on" : "off",
                derivatives ? "on" : "off", validation ? "on" : "off");
    vkGetDeviceQueue(vk.device, vk.family, 0, &vk.queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = vk.family;
    vkCreateCommandPool(vk.device, &pci, nullptr, &vk.pool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = vk.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(vk.device, &cai, &cb);

    // The upscale: 96x64 to 192x128 (FSR's Performance ratio), HDR colour with FSR's own exposure, as bbhost asks.
    const std::uint32_t rw = 96, rh = 64, ow = 192, oh = 128;
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    Image color, depth, motion, output, motion_out;
    bool made = make_image(vk, &color, VK_FORMAT_R16G16B16A16_SFLOAT, rw, rh, usage) &&
                make_image(vk, &depth, VK_FORMAT_R32_SFLOAT, rw, rh, usage) &&
                make_image(vk, &motion, VK_FORMAT_R16G16_SFLOAT, rw, rh, usage) &&
                make_image(vk, &output, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, usage) &&
                make_image(vk, &motion_out, VK_FORMAT_R16G16_SFLOAT, ow, oh, usage);
    Buffer staging, readback, readback0;
    made = made && make_buffer(vk, &staging, rw * rh * (8 + 4 + 4)) && make_buffer(vk, &readback, ow * oh * 8) &&
           make_buffer(vk, &readback0, ow * oh * 8);
    check(made, "the test's images and buffers");
    if (!made) return 1;

    // One run: a context, 64 frames, FSR's picture after the first and after the last; the jitter handed to FSR times
    // `sign` (the picture is always drawn with it as it is).
    const std::uint32_t phases = static_cast<std::uint32_t>(std::ceil(8.0 * (double(ow) / rw) * (double(ow) / rw)));
    const int frames = 2 * static_cast<int>(phases);
    std::vector<std::uint16_t> last_input(static_cast<std::size_t>(rw) * rh * 4);
    const auto run = [&](float sign) -> bool {
        ffxCreateBackendVKDesc backend{};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backend.deviceContext.vkDevice = vk.device;
        backend.deviceContext.vkPhysicalDevice = vk.phys;
        backend.deviceContext.vkDeviceProcAddr = vkGetDeviceProcAddr;
        ffxCreateContextDescUpscale cd{};
        cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        cd.header.pNext = &backend.header;
        cd.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
        cd.maxRenderSize = {rw, rh};
        cd.maxUpscaleSize = {ow, oh};
        cd.fpMessage = &on_ffx_message;
        ffxContext ctx = nullptr;
        const ffxReturnCode_t made_rc = create(&ctx, &cd.header, nullptr);
        if (made_rc != FFX_API_RETURN_OK || !ctx) {
            std::printf("ffxCreateContext: %u\n", made_rc);
            return false;
        }
        if (sign > 0) {
            ffxQueryGetProviderVersion pv{};
            pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
            check(query(&ctx, &pv.header) == FFX_API_RETURN_OK && pv.versionName && std::strcmp(pv.versionName, "3.1.6") == 0,
                  "the context runs FSR 3.1.6");
        }
        bool ok = true;
        for (int f = 0; f < frames && ok; ++f) {
            const float jx = halton(static_cast<std::uint32_t>(f) % phases + 1, 2) - 0.5f;
            const float jy = halton(static_cast<std::uint32_t>(f) % phases + 1, 3) - 0.5f;
            auto* rgba = static_cast<std::uint16_t*>(staging.map);
            auto* d = reinterpret_cast<float*>(static_cast<char*>(staging.map) + rw * rh * 8);
            auto* mv = reinterpret_cast<std::uint16_t*>(static_cast<char*>(staging.map) + rw * rh * 12);
            for (std::uint32_t y = 0; y < rh; ++y)
                for (std::uint32_t x = 0; x < rw; ++x) {
                    // The pixel's centre shows the point of the unjittered picture jitter pixels up and left of it.
                    float c[3];
                    scene((x + 0.5 - jx) * ow / rw, (y + 0.5 - jy) * oh / rh, c);
                    std::uint16_t* p = rgba + (static_cast<std::size_t>(y) * rw + x) * 4;
                    for (int k = 0; k < 3; ++k) p[k] = to_half(c[k]);
                    p[3] = to_half(1.0f);
                    d[y * rw + x] = 0.5f;
                    mv[(y * rw + x) * 2] = mv[(y * rw + x) * 2 + 1] = 0;
                }
            std::memcpy(last_input.data(), rgba, last_input.size() * 2);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkResetCommandBuffer(cb, 0);
            vkBeginCommandBuffer(cb, &bi);
            upload(cb, &color, staging, 0);
            upload(cb, &depth, staging, rw * rh * 8);
            upload(cb, &motion, staging, rw * rh * 12);
            transition(cb, &color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            transition(cb, &depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            transition(cb, &motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (f == frames - 1) {
                // A sentinel FSR's output must cover.
                transition(cb, &output, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                VkClearColorValue sentinel{};
                sentinel.float32[0] = sentinel.float32[1] = sentinel.float32[2] = sentinel.float32[3] = -4096.0f;
                const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cb, output.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &sentinel, 1, &all);
            }
            transition(cb, &output, VK_IMAGE_LAYOUT_GENERAL);
            ffxDispatchDescUpscale dd{};
            dd.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
            dd.commandList = cb;
            dd.color = resource(color, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
            dd.depth = resource(depth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
            dd.motionVectors = resource(motion, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
            dd.output = resource(output, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
            dd.jitterOffset = {jx * sign, jy * sign};
            dd.motionVectorScale = {1.0f, 1.0f};
            dd.renderSize = {rw, rh};
            dd.upscaleSize = {ow, oh};
            dd.frameTimeDelta = 16.7f;
            dd.preExposure = 1.0f;
            dd.reset = f == 0;
            dd.cameraNear = 0.1f;
            dd.cameraFar = 1000.0f;
            dd.cameraFovAngleVertical = 1.0f;
            dd.viewSpaceToMetersFactor = 1.0f;
            const ffxReturnCode_t rc = dispatch(&ctx, &dd.header);
            if (rc != FFX_API_RETURN_OK) {
                std::printf("ffxDispatch frame %d: %u\n", f, rc);
                ok = false;
            }
            if (f == 0 || f == frames - 1) {
                // (The first frame's too: FSR from one frame, before any history.)
                output.layout = VK_IMAGE_LAYOUT_GENERAL;  // (FidelityFX puts it back as it was handed over)
                transition(cb, &output, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                VkBufferImageCopy r{};
                r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                r.imageExtent = {ow, oh, 1};
                vkCmdCopyImageToBuffer(cb, output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, f ? readback.buffer : readback0.buffer, 1, &r);
            }
            vkEndCommandBuffer(cb);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(vk.queue);
        }
        ok = destroy(&ctx, nullptr) == FFX_API_RETURN_OK && ok;
        return ok;
    };

    // FSR's picture against the scene at the output size, inside a 4-pixel border.
    const auto measure = [&](const Buffer& b, double* mean, double* largest, std::size_t* sentinel) {
        const auto* out = static_cast<const std::uint16_t*>(b.map);
        double sum = 0.0;
        std::size_t n = 0;
        *largest = 0.0;
        *sentinel = 0;
        for (std::uint32_t y = 0; y < oh; ++y)
            for (std::uint32_t x = 0; x < ow; ++x) {
                const std::uint16_t* p = out + (static_cast<std::size_t>(y) * ow + x) * 4;
                if (!(from_half(p[0]) > -1000.0f)) ++*sentinel;
                if (x < 4 || y < 4 || x >= ow - 4 || y >= oh - 4) continue;
                float want[3];
                scene(x + 0.5, y + 0.5, want);
                for (int k = 0; k < 3; ++k) {
                    const double e = std::fabs(from_half(p[k]) - want[k]);
                    sum += e;
                    *largest = std::max(*largest, e);
                    ++n;
                }
            }
        *mean = sum / static_cast<double>(n);
    };
    // The last input brought up linearly, against the same: what one frame gives.
    const auto linear_error = [&]() {
        double sum = 0.0;
        std::size_t n = 0;
        // The output pixel's centre in the last input frame, whose pixel centres sit jitter away.
        const float jx = halton(static_cast<std::uint32_t>(frames - 1) % phases + 1, 2) - 0.5f;
        const float jy = halton(static_cast<std::uint32_t>(frames - 1) % phases + 1, 3) - 0.5f;
        for (std::uint32_t y = 4; y < oh - 4; ++y)
            for (std::uint32_t x = 4; x < ow - 4; ++x) {
                float want[3];
                scene(x + 0.5, y + 0.5, want);
                const double sx = std::clamp((x + 0.5) * rw / ow - 0.5 + jx, 0.0, rw - 1.0);
                const double sy = std::clamp((y + 0.5) * rh / oh - 0.5 + jy, 0.0, rh - 1.0);
                const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
                const int x1 = std::min(x0 + 1, static_cast<int>(rw) - 1), y1 = std::min(y0 + 1, static_cast<int>(rh) - 1);
                const double fx = sx - x0, fy = sy - y0;
                for (int k = 0; k < 3; ++k) {
                    const auto in = [&](int ix, int iy) { return double(from_half(last_input[(static_cast<std::size_t>(iy) * rw + ix) * 4 + k])); };
                    sum += std::fabs((in(x0, y0) * (1 - fx) + in(x1, y0) * fx) * (1 - fy) + (in(x0, y1) * (1 - fx) + in(x1, y1) * fx) * fy - want[k]);
                    ++n;
                }
            }
        return sum / static_cast<double>(n);
    };

    const bool ran = run(1.0f);
    check(ran, "an FSR 3 context on this device (every pipeline made from the generated shaders), 64 dispatches, destroyed");
    if (!ran) return 1;
    double err0 = 0.0, max0 = 0.0, err = 0.0, max_err = 0.0;
    std::size_t sentinel0 = 0, sentinel_left = 0;
    measure(readback0, &err0, &max0, &sentinel0);
    measure(readback, &err, &max_err, &sentinel_left);
    const double lin_err = linear_error();
    double wrong = 0.0, wrong_max = 0.0;
    std::size_t wrong_sentinel = 0;
    const bool ran_wrong = run(-1.0f);
    if (ran_wrong) measure(readback, &wrong, &wrong_max, &wrong_sentinel);
    std::printf("FSR's mean error: %.4f after the first frame (largest %.4f), %.4f after %d (largest %.4f), %.4f with the jitter's sign "
                "turned round; one frame brought up linearly %.4f\n",
                err0, max0, err, frames, max_err, wrong, lin_err);
    check(sentinel_left == 0, "FSR wrote every output pixel (no sentinel left)");
    check(err < 0.03, "FSR's picture is the scene at the output size (mean error under 0.03)");
    check(err < 0.8 * err0 && err < 0.8 * lin_err, "closer to it than one frame: the frames were accumulated");
    check(ran_wrong && err < 0.6 * wrong, "far closer than with the jitter's sign turned round: the jitter is read as bbhost means it");

    // Every other set of the context's flags the shaders are chosen by (HDR or not, depth inverted, motion at the output
    // size, motion with the jitter in it): a context made and one sharpened dispatch, so each of those pipelines is
    // made from its blobs and run.
    int combos = 0, combos_ok = 0;
    for (std::uint32_t bits = 0; bits < 16; ++bits) {
        const std::uint32_t flags = (bits & 1 ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0u) |
                                    (bits & 2 ? FFX_UPSCALE_ENABLE_DEPTH_INVERTED : 0u) |
                                    (bits & 4 ? FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS : 0u) |
                                    (bits & 8 ? FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION : 0u);
        ++combos;
        ffxCreateBackendVKDesc backend{};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backend.deviceContext.vkDevice = vk.device;
        backend.deviceContext.vkPhysicalDevice = vk.phys;
        backend.deviceContext.vkDeviceProcAddr = vkGetDeviceProcAddr;
        ffxCreateContextDescUpscale cd{};
        cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        cd.header.pNext = &backend.header;
        cd.flags = flags;
        cd.maxRenderSize = {rw, rh};
        cd.maxUpscaleSize = {ow, oh};
        cd.fpMessage = &on_ffx_message;
        ffxContext ctx = nullptr;
        if (create(&ctx, &cd.header, nullptr) != FFX_API_RETURN_OK || !ctx) {
            std::printf("flags 0x%x: no context\n", flags);
            continue;
        }
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &bi);
        if (motion_out.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            transition(cb, &motion_out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            const VkClearColorValue zero{};
            const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, motion_out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
            transition(cb, &motion_out, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        transition(cb, &output, VK_IMAGE_LAYOUT_GENERAL);
        ffxDispatchDescUpscale dd{};
        dd.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dd.commandList = cb;
        dd.color = resource(color, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dd.depth = resource(depth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dd.motionVectors = resource(bits & 4 ? motion_out : motion, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dd.output = resource(output, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dd.motionVectorScale = {1.0f, 1.0f};
        dd.renderSize = {rw, rh};
        dd.upscaleSize = {ow, oh};
        dd.enableSharpening = true;
        dd.sharpness = 0.5f;
        dd.frameTimeDelta = 16.7f;
        dd.preExposure = 1.0f;
        dd.reset = true;
        dd.cameraNear = 0.1f;
        dd.cameraFar = 1000.0f;
        dd.cameraFovAngleVertical = 1.0f;
        dd.viewSpaceToMetersFactor = 1.0f;
        const bool dispatched = dispatch(&ctx, &dd.header) == FFX_API_RETURN_OK;
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        vkQueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk.queue);
        if (destroy(&ctx, nullptr) == FFX_API_RETURN_OK && dispatched) ++combos_ok;
        else std::printf("flags 0x%x: the dispatch or the destroy failed\n", flags);
    }
    std::printf("contexts with each set of flags: %d of %d made, dispatched sharpened and destroyed\n", combos_ok, combos);
    check(combos_ok == combos, "a context of each set of flags, its dispatch with sharpening");
    vkDeviceWaitIdle(vk.device);
    if (validation) {
        std::printf("validation errors: %d\n", g_validation_errors);
        check(g_validation_errors == 0, "no validation errors");
    }
    vkDestroyCommandPool(vk.device, vk.pool, nullptr);
    for (Image* im : {&color, &depth, &motion, &output, &motion_out}) {
        vkDestroyImage(vk.device, im->image, nullptr);
        vkFreeMemory(vk.device, im->memory, nullptr);
    }
    for (Buffer* b : {&staging, &readback, &readback0}) {
        vkDestroyBuffer(vk.device, b->buffer, nullptr);
        vkFreeMemory(vk.device, b->memory, nullptr);
    }
    vkDestroyDevice(vk.device, nullptr);
    if (messenger) {
        auto kill = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(vk.instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (kill) kill(vk.instance, messenger, nullptr);
    }
    vkDestroyInstance(vk.instance, nullptr);
    std::printf("%s\n", g_failures ? "FAILED" : "all passed");
    return g_failures ? 1 : 0;
}
