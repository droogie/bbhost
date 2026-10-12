// Actual NGX SR and FG execution/readback tests, with independent synthetic inputs.
// A counter or a successful evaluate call is insufficient: inspect the output pixels.
#define VK_NO_PROTOTYPES
#include "ngx_bridge.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void check(VkResult r) { if (r != VK_SUCCESS) throw std::runtime_error("Vulkan operation failed"); }
void ngx(uint32_t r) { if (ngxb_failed(r)) { std::printf("NGX failure: 0x%08x\n",r); throw std::runtime_error("NGX operation failed"); } }

struct Image { NgxbImage api{}; VkDeviceMemory memory{}; bool initialized = false; };
struct Gpu {
    VkDevice device;
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer cmd{};
    VkBuffer staging{};
    VkDeviceMemory staging_memory{};
    void* mapped{};
    VkPhysicalDeviceMemoryProperties memory{};
    std::vector<Image> images;
#define PROC(name) PFN_##name name{}
    PROC(vkCreateImage); PROC(vkDestroyImage); PROC(vkGetImageMemoryRequirements);
    PROC(vkAllocateMemory); PROC(vkFreeMemory); PROC(vkBindImageMemory);
    PROC(vkCreateImageView); PROC(vkDestroyImageView); PROC(vkCreateBuffer); PROC(vkDestroyBuffer);
    PROC(vkGetBufferMemoryRequirements); PROC(vkBindBufferMemory); PROC(vkMapMemory); PROC(vkUnmapMemory);
    PROC(vkCreateCommandPool); PROC(vkDestroyCommandPool); PROC(vkAllocateCommandBuffers);
    PROC(vkResetCommandBuffer); PROC(vkBeginCommandBuffer); PROC(vkEndCommandBuffer);
    PROC(vkGetDeviceQueue); PROC(vkQueueSubmit); PROC(vkQueueWaitIdle);
    PROC(vkCmdPipelineBarrier); PROC(vkCmdCopyBufferToImage); PROC(vkCmdCopyImageToBuffer); PROC(vkCmdCopyImage);
#undef PROC
    Gpu(VkInstance instance, VkPhysicalDevice physical, VkDevice d, uint32_t family,
        PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa) : device(d) {
#define LOAD(name) name = reinterpret_cast<PFN_##name>(gdpa(device,#name))
        LOAD(vkCreateImage); LOAD(vkDestroyImage); LOAD(vkGetImageMemoryRequirements);
        LOAD(vkAllocateMemory); LOAD(vkFreeMemory); LOAD(vkBindImageMemory);
        LOAD(vkCreateImageView); LOAD(vkDestroyImageView); LOAD(vkCreateBuffer); LOAD(vkDestroyBuffer);
        LOAD(vkGetBufferMemoryRequirements); LOAD(vkBindBufferMemory); LOAD(vkMapMemory); LOAD(vkUnmapMemory);
        LOAD(vkCreateCommandPool); LOAD(vkDestroyCommandPool); LOAD(vkAllocateCommandBuffers);
        LOAD(vkResetCommandBuffer); LOAD(vkBeginCommandBuffer); LOAD(vkEndCommandBuffer);
        LOAD(vkGetDeviceQueue); LOAD(vkQueueSubmit); LOAD(vkQueueWaitIdle);
        LOAD(vkCmdPipelineBarrier); LOAD(vkCmdCopyBufferToImage); LOAD(vkCmdCopyImageToBuffer); LOAD(vkCmdCopyImage);
#undef LOAD
        auto props = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(gipa(instance,"vkGetPhysicalDeviceMemoryProperties"));
        props(physical,&memory);
        vkGetDeviceQueue(device,family,0,&queue);
        VkCommandPoolCreateInfo p{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        p.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; p.queueFamilyIndex = family;
        check(vkCreateCommandPool(device,&p,nullptr,&pool));
        VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        a.commandPool = pool; a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device,&a,&cmd));
        VkBufferCreateInfo b{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        b.size = 16*1024*1024; b.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        check(vkCreateBuffer(device,&b,nullptr,&staging));
        VkMemoryRequirements r{}; vkGetBufferMemoryRequirements(device,staging,&r);
        staging_memory = allocate(r,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
        check(vkBindBufferMemory(device,staging,staging_memory,0));
        check(vkMapMemory(device,staging_memory,0,VK_WHOLE_SIZE,0,&mapped));
    }
    VkDeviceMemory allocate(VkMemoryRequirements r, VkMemoryPropertyFlags flags, bool address = false) {
        uint32_t i = 0;
        while (i < memory.memoryTypeCount && (!(r.memoryTypeBits & (1u<<i)) || (memory.memoryTypes[i].propertyFlags & flags) != flags)) ++i;
        if (i == memory.memoryTypeCount) throw std::runtime_error("no suitable memory type");
        VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        VkMemoryAllocateFlagsInfo address_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        address_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        if (address) a.pNext = &address_flags;
        a.allocationSize = r.size; a.memoryTypeIndex = i;
        VkDeviceMemory m{}; check(vkAllocateMemory(device,&a,nullptr,&m)); return m;
    }
    size_t image(uint32_t w,uint32_t h,VkFormat format) {
        Image i; i.api.width = w; i.api.height = h; i.api.format = format; i.api.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        VkImageCreateInfo c{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        c.imageType = VK_IMAGE_TYPE_2D; c.format = format; c.extent = {w,h,1}; c.mipLevels = c.arrayLayers = 1;
        c.samples = VK_SAMPLE_COUNT_1_BIT; c.tiling = VK_IMAGE_TILING_OPTIMAL;
        c.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        check(vkCreateImage(device,&c,nullptr,&i.api.image));
        VkMemoryRequirements r{}; vkGetImageMemoryRequirements(device,i.api.image,&r);
        i.memory = allocate(r,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkBindImageMemory(device,i.api.image,i.memory,0));
        VkImageViewCreateInfo v{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        v.image = i.api.image; v.viewType = VK_IMAGE_VIEW_TYPE_2D; v.format = format; v.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        check(vkCreateImageView(device,&v,nullptr,&i.api.view));
        images.push_back(i); return images.size()-1;
    }
    void begin() {
        check(vkResetCommandBuffer(cmd,0));
        VkCommandBufferBeginInfo b{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; b.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd,&b));
        for (auto& i : images) if (!i.initialized) {
            VkImageMemoryBarrier r{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            r.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; r.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            r.srcQueueFamilyIndex = r.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            r.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            r.image = i.api.image; r.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&r);
            i.initialized = true;
        }
    }
    void barrier() {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&b,0,nullptr,0,nullptr);
    }
    void submit() {
        check(vkEndCommandBuffer(cmd));
        VkSubmitInfo s{VK_STRUCTURE_TYPE_SUBMIT_INFO}; s.commandBufferCount = 1; s.pCommandBuffers = &cmd;
        check(vkQueueSubmit(queue,1,&s,nullptr)); check(vkQueueWaitIdle(queue));
    }
    void upload(size_t image,const void* data,size_t bytes) {
        std::memcpy(mapped,data,bytes);
        begin(); barrier();
        VkBufferImageCopy c{}; c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; c.imageExtent = {images[image].api.width,images[image].api.height,1};
        vkCmdCopyBufferToImage(cmd,staging,images[image].api.image,VK_IMAGE_LAYOUT_GENERAL,1,&c);
        barrier(); submit();
    }
    std::vector<unsigned char> read(size_t image,size_t bytes) {
        begin(); barrier();
        VkBufferImageCopy c{}; c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; c.imageExtent = {images[image].api.width,images[image].api.height,1};
        vkCmdCopyImageToBuffer(cmd,images[image].api.image,VK_IMAGE_LAYOUT_GENERAL,staging,1,&c);
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&b,0,nullptr,0,nullptr);
        submit();
        std::vector<unsigned char> result(bytes); std::memcpy(result.data(),mapped,bytes); return result;
    }
    ~Gpu() {
        vkQueueWaitIdle(queue);
        for (auto& i : images) { vkDestroyImageView(device,i.api.view,nullptr); vkDestroyImage(device,i.api.image,nullptr); vkFreeMemory(device,i.memory,nullptr); }
        vkUnmapMemory(device,staging_memory); vkDestroyBuffer(device,staging,nullptr); vkFreeMemory(device,staging_memory,nullptr);
        vkDestroyCommandPool(device,pool,nullptr);
    }
};
}

int ngx_gpu_tests(VkInstance instance,VkPhysicalDevice physical,VkDevice device,uint32_t family,
                  PFN_vkGetInstanceProcAddr gipa,PFN_vkGetDeviceProcAddr gdpa) {
    try {
        Gpu g(instance,physical,device,family,gipa,gdpa);
        const uint32_t rw=640,rh=360,ow=960,oh=540;
        auto color=g.image(rw,rh,VK_FORMAT_R16G16B16A16_SFLOAT), depth=g.image(rw,rh,VK_FORMAT_R32_SFLOAT),
             motion=g.image(rw,rh,VK_FORMAT_R16G16_SFLOAT), result=g.image(ow,oh,VK_FORMAT_R16G16B16A16_SFLOAT);
        std::vector<uint16_t> green(size_t(rw)*rh*4);
        for (size_t i=0;i<green.size();i+=4) { green[i+1]=0x3c00; green[i+3]=0x3c00; }
        std::vector<float> z(size_t(rw)*rh,1);
        std::vector<uint16_t> mv(size_t(rw)*rh*2);
        g.upload(color,green.data(),green.size()*2); g.upload(depth,z.data(),z.size()*4); g.upload(motion,mv.data(),mv.size()*2);
        g.begin(); ngx(ngxb_create_sr(g.cmd,rw,rh,ow,oh,1,0,2|64)); g.submit();
        for (int frame=0;frame<4;++frame) {
            NgxbEval e{}; e.color=g.images[color].api; e.output=g.images[result].api; e.depth=g.images[depth].api; e.motion=g.images[motion].api;
            e.mv_scale_x=e.mv_scale_y=1; e.frame_ms=16.6667f; e.reset=frame==0;
            g.begin(); g.barrier(); ngx(ngxb_evaluate(g.cmd,&e)); g.barrier(); g.submit();
        }
        auto pixels=g.read(result,size_t(ow)*oh*8);
        auto center=reinterpret_cast<const uint16_t*>(pixels.data())+((oh/2)*ow+ow/2)*4;
        if (center[1] < 0x3800 || center[0] > 0x2800 || center[2] > 0x2800) throw std::runtime_error("SR output did not reconstruct the green input");
        std::puts("PASS SR: 640x360 input reconstructed to 960x540; output read back from GPU");
        ngxb_release();
        if (!ngxb_fg_available()) { std::puts("SKIP FG: NGX reports unavailable"); return 77; }
        const uint32_t w=1280,h=720;
        color=g.image(w,h,VK_FORMAT_R8G8B8A8_UNORM); depth=g.image(w,h,VK_FORMAT_R32_SFLOAT);
        motion=g.image(w,h,VK_FORMAT_R16G16_SFLOAT); result=g.image(w,h,VK_FORMAT_R8G8B8A8_UNORM);
        auto real=g.image(w,h,VK_FORMAT_R8G8B8A8_UNORM);
        std::vector<unsigned char> rgba(size_t(w)*h*4);
        z.assign(size_t(w)*h,.99f); mv.assign(size_t(w)*h*2,0);
        const auto max_generated = ngxb_fg_max_generated();
        std::printf("FG maximum generated frames: %u\n", max_generated);
        for (uint32_t count=1; count<=std::min(3u,max_generated); ++count) {
        g.begin(); ngx(ngxb_fg_create(g.cmd,w,h,VK_FORMAT_R8G8B8A8_UNORM)); g.submit();
        int verified=0;
        int verified_after_reset=0;
        for (int frame=0;frame<18;++frame) {
            const int x=200+frame*16;
            for (uint32_t yy=0;yy<h;++yy) for (uint32_t xx=0;xx<w;++xx) {
                size_t p=size_t(yy)*w+xx;
                const bool object=xx>=uint32_t(x) && xx<uint32_t(x+100) && yy>=240 && yy<480;
                rgba[p*4]=rgba[p*4+1]=rgba[p*4+2]=object?255:24; rgba[p*4+3]=255;
                z[p]=object?.5f:.99f; mv[p*2]=object?0xcc00:0; // half(-16)
            }
            g.upload(color,rgba.data(),rgba.size()); g.upload(depth,z.data(),z.size()*4); g.upload(motion,mv.data(),mv.size()*2);
            // Match the host: MFG retains one real endpoint explicitly and
            // leaves optional OutputReal unset for the entire index sequence.
            if (count > 1) {
                g.begin(); g.barrier();
                VkImageCopy copy{};
                copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
                copy.extent={w,h,1};
                g.vkCmdCopyImage(g.cmd,g.images[color].api.image,VK_IMAGE_LAYOUT_GENERAL,
                                g.images[real].api.image,VK_IMAGE_LAYOUT_GENERAL,1,&copy);
                g.barrier(); g.submit();
            }
            NgxbGenerate f{}; f.color=g.images[color].api; f.output=g.images[result].api;
            if (count == 1) f.real=g.images[real].api;
            f.depth=g.images[depth].api; f.motion=g.images[motion].api; f.disable_interpolation=g.staging;
            f.reset=frame==0 || frame==10;
            f.camera.near_plane=.1f; f.camera.far_plane=1000; f.camera.vertical_fov=1;
            f.camera.up[1]=f.camera.right[0]=f.camera.forward[2]=1;
            const float py=1/std::tan(.5f), px=py/(float(w)/h), a=1000/999.9f, b=-.1f*a;
            f.camera.view_to_clip[0]=px; f.camera.view_to_clip[5]=py; f.camera.view_to_clip[10]=a; f.camera.view_to_clip[11]=1; f.camera.view_to_clip[14]=b;
            f.camera.clip_to_view[0]=1/px; f.camera.clip_to_view[5]=1/py; f.camera.clip_to_view[11]=1/b; f.camera.clip_to_view[14]=1; f.camera.clip_to_view[15]=-a/b;
            for (int i=0;i<4;++i) f.camera.clip_to_previous[i*5]=f.camera.previous_to_clip[i*5]=1;
            *static_cast<uint32_t*>(g.mapped)=1;
            bool disabled = true;
            for (uint32_t index=1; index<=count; ++index) {
            g.begin(); g.barrier(); ngx(ngxb_fg_evaluate_index(g.cmd,&f,count,index,frame+1));
            VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; host.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT; host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
            g.vkCmdPipelineBarrier(g.cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&host,0,nullptr,0,nullptr);
            g.submit();
            if (index==1) disabled=*static_cast<uint32_t*>(g.mapped)!=0;
            if (f.reset && !disabled) throw std::runtime_error("FG did not suppress interpolation on a history reset");
            auto copied_real=g.read(real,rgba.size());
            if (copied_real != rgba) throw std::runtime_error("FG real endpoint did not preserve the rendered frame");
            if (frame<3 || disabled) { std::printf("FG frame %d: disabled=%d\n",frame,disabled); continue; }
            auto output=g.read(result,rgba.size());
            double error_mid=0,error_prev=0,error_current=0,error_other=0;
            const int expected_x = x-16+int(std::lround(16.0*index/(count+1)));
            const int other_x = x-16+int(std::lround(16.0*(index==1?count:1)/(count+1)));
            for (int yy=260;yy<460;++yy) for (int xx=x-24;xx<x+112;++xx) {
                double value=output[(size_t(yy)*w+xx)*4];
                auto expect=[&](int left){return double(xx>=left && xx<left+100?255:24);};
                error_mid+=std::pow(value-expect(expected_x),2); error_prev+=std::pow(value-expect(x-16),2); error_current+=std::pow(value-expect(x),2);
                error_other+=std::pow(value-expect(other_x),2);
            }
            std::printf("FG %ux frame %d index %u: expected %.0f previous %.0f current %.0f other %.0f\n",count+1,frame,index,error_mid,error_prev,error_current,error_other);
            if (error_mid < .7*std::min(error_prev,error_current) && (count==1 || error_mid < .8*error_other)) {
                ++verified;
                if (frame>12) ++verified_after_reset;
            }
            }
        }
        ngxb_fg_release();
        if (verified<int(6*count)) throw std::runtime_error("generated images did not consistently reconstruct distinct temporal positions");
        if (verified_after_reset<int(3*count)) throw std::runtime_error("FG did not recover its interpolation history after reset");
        std::printf("PASS FG %ux: %d distinct generated images verified at their temporal positions\n",count+1,verified);
        std::puts("PASS FG: reset suppresses interpolation, history recovers, real endpoint preserves every input frame");
        }
        return 0;
    } catch (const std::exception& e) { ngxb_fg_release(); ngxb_release(); std::printf("FAIL: %s\n",e.what()); return 1; }
}
