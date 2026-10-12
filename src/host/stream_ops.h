#pragma once
// The command stream's op records (recorder.cpp's header describes the
// stream).
//
// A block of the stream is an arena of 8-byte words. Each record is a Header
// followed by its plain struct and then its arrays, every part 8-byte
// aligned, and `bytes` covers the whole record. A record holds values only:
// handles, flags, the arrays it was given copied in - never a pointer into
// the caller's memory, which is gone (or reused) by the time the recorder
// replays it. The structs are trivially copyable, which the static_asserts
// below keep so.

#include <cstdint>
#include <type_traits>

#include <vulkan/vulkan.h>

namespace gpu::stream {

enum class Op : std::uint16_t {
    kBegin,              // vkBeginCommandBuffer: the recorder's command buffer from here
    kEndSubmit,          // vkEndCommandBuffer, then the submission handed to bb-submit (a kick)
    kForeignSubmit,      // another thread's command buffer, submitted in stream order (the presenter's blit; a kick)
    kDrawState,          // a draw packet replayed here (its descriptor writes, pass changes, binds, draw)
    kBarrier,            // vkCmdPipelineBarrier over copied arrays
    kTransferBarrier,    // record_transfer_barrier
    kCopyOrderBarrier,   // record_copy_order_barrier
    kPassBarrier,        // record_pass_barrier
    kEndRendering,       // record_end_rendering
    kBeginRendering,     // vkCmdBeginRendering, its attachments copied
    kFill,               // vkCmdFillBuffer
    kCopyBuffer,         // vkCmdCopyBuffer
    kCopyBufferToImage,  // vkCmdCopyBufferToImage
    kCopyImageToBuffer,  // vkCmdCopyImageToBuffer
    kClearColor,         // vkCmdClearColorImage
    kResetQueries,       // vkCmdResetQueryPool
    kBeginQuery,         // vkCmdBeginQuery
    kEndQuery,           // vkCmdEndQuery
    kCopyQueryResults,   // vkCmdCopyQueryPoolResults
    kTimestamp,          // vkCmdWriteTimestamp
    kBindPipeline,       // vkCmdBindPipeline
    kBindSets,           // vkCmdBindDescriptorSets
    kPushConstants,      // vkCmdPushConstants
    kUpdateSets,         // vkUpdateDescriptorSets at its place in the stream
    kDispatch,           // vkCmdDispatch
    kClearDepth,         // vkCmdClearDepthStencilImage
    kCopyImage,          // vkCmdCopyImage
    kDispatchIndirect,   // vkCmdDispatchIndirect
    kMarker,             // AMD's buffer markers around what follows (gpu_write_markers)
    kCount
};

const char* op_name(Op op);

struct Header {
    Op kind;
    std::uint16_t site;   // 0, or an index into the census table (BBHOST_CMD_CENSUS)
    std::uint32_t bytes;  // the whole record, this header included; a multiple of 8
};
static_assert(sizeof(Header) == 8, "a record's header is one word");

struct Begin {
    VkCommandBuffer cmd;
};
struct EndSubmit {
    VkCommandBuffer cmds[2];  // the buffer shadow's uploads (ended on the command processor) or null, then the recording's
    std::uint32_t ncmds, items;
    VkSemaphore wait;  // the buffer shadow's sparse bind, or null
    VkPipelineStageFlags wait_stage;
    std::uint32_t pad;
    VkFence fence;
    std::uint64_t reflex_frame_id; // simulation ID captured at recording start
    std::uint64_t serial;  // Gpu::flushes of this submission
};
struct ForeignSubmit {
    VkCommandBuffer cmd;
    VkSemaphore wait;
    VkPipelineStageFlags wait_stage;
    std::uint32_t pad;
    VkSemaphore signal;
    VkFence fence;
    std::uint64_t reflex_frame_id;
    std::uint64_t id;  // what host_gpu_wait_submitted waits for
};
struct DrawState {
    std::uint64_t packet;  // the packet's number (recorder.cpp's packet ring)
};
struct Barrier {
    VkPipelineStageFlags src, dst;
    VkDependencyFlags dep;
    std::uint32_t nmem, nbuf, nimg;
    // + VkMemoryBarrier[nmem], VkBufferMemoryBarrier[nbuf], VkImageMemoryBarrier[nimg]
};
struct Flag {
    std::uint32_t value;  // kTransferBarrier: begin; kEndRendering: barrier
    std::uint32_t pad;
};
struct BeginRendering {
    VkRenderingFlags flags;
    std::uint32_t layers, view_mask, ncolor;
    VkRect2D area;
    std::uint32_t has_depth, has_stencil;
    // + VkRenderingAttachmentInfo[ncolor], then depth, then stencil (when present)
};
struct Fill {
    VkBuffer buffer;
    VkDeviceSize offset, size;
    std::uint32_t data, pad;
};
struct CopyBuffer {
    VkBuffer src, dst;
    std::uint32_t n, pad;
    // + VkBufferCopy[n]
};
struct CopyBufferToImage {
    VkBuffer buffer;
    VkImage image;
    VkImageLayout layout;
    std::uint32_t n;
    // + VkBufferImageCopy[n]
};
struct CopyImageToBuffer {
    VkImage image;
    VkBuffer buffer;
    VkImageLayout layout;
    std::uint32_t n;
    // + VkBufferImageCopy[n]
};
struct ClearColor {
    VkImage image;
    VkImageLayout layout;
    std::uint32_t n;
    VkClearColorValue color;
    // + VkImageSubresourceRange[n]
};
struct Queries {
    VkQueryPool pool;
    std::uint32_t first, count;  // kBeginQuery / kEndQuery: first is the query
    VkQueryControlFlags flags;   // kBeginQuery
    std::uint32_t pad;
};
struct CopyQueryResults {
    VkQueryPool pool;
    std::uint32_t first, count;
    VkBuffer dst;
    VkDeviceSize offset, stride;
    VkQueryResultFlags flags;
    std::uint32_t pad;
};
struct Timestamp {
    VkQueryPool pool;
    VkPipelineStageFlagBits stage;
    std::uint32_t query;
};
struct BindPipeline {
    VkPipeline pipeline;
    VkPipelineBindPoint bind_point;
    std::uint32_t pad;
};
struct BindSets {
    VkPipelineLayout layout;
    VkPipelineBindPoint bind_point;
    std::uint32_t first, n, ndynamic;
    // + VkDescriptorSet[n], then std::uint32_t dynamic offsets[ndynamic]
};
struct PushConstants {
    VkPipelineLayout layout;
    VkShaderStageFlags stages;
    std::uint32_t offset, size, pad;
    // + size bytes
};
struct UpdateSets {
    std::uint32_t n, nbuffers, nimages, pad;
    // + VkWriteDescriptorSet[n] (pBufferInfo / pImageInfo null), then
    // std::int32_t first_buffer[n], std::int32_t first_image[n] (-1: none),
    // then VkDescriptorBufferInfo[nbuffers], VkDescriptorImageInfo[nimages]
};
struct Dispatch {
    std::uint32_t x, y, z, pad;
};
struct ClearDepth {
    VkImage image;
    VkImageLayout layout;
    std::uint32_t n;
    VkClearDepthStencilValue value;
    // + VkImageSubresourceRange[n]
};
struct CopyImage {
    VkImage src, dst;
    VkImageLayout src_layout, dst_layout;
    std::uint32_t n, pad;
    // + VkImageCopy[n]
};
struct DispatchIndirect {
    VkBuffer buffer;
    VkDeviceSize offset;
};
struct Marker {
    std::uint32_t value, pad;  // gpu_marker_value
};

template <class T> constexpr bool kRecordable = std::is_trivially_copyable_v<T> && alignof(T) <= 8;
static_assert(kRecordable<Begin> && kRecordable<EndSubmit> && kRecordable<ForeignSubmit> && kRecordable<DrawState> && kRecordable<Barrier> &&
                  kRecordable<Flag> && kRecordable<BeginRendering> && kRecordable<Fill> && kRecordable<CopyBuffer> &&
                  kRecordable<CopyBufferToImage> && kRecordable<CopyImageToBuffer> && kRecordable<ClearColor> && kRecordable<Queries> &&
                  kRecordable<CopyQueryResults> && kRecordable<Timestamp> && kRecordable<BindPipeline> && kRecordable<BindSets> &&
                  kRecordable<PushConstants> && kRecordable<UpdateSets> && kRecordable<Dispatch> && kRecordable<ClearDepth> &&
                  kRecordable<CopyImage> && kRecordable<DispatchIndirect> && kRecordable<Marker>,
              "op records are plain values");
static_assert(kRecordable<VkMemoryBarrier> && kRecordable<VkBufferMemoryBarrier> && kRecordable<VkImageMemoryBarrier> &&
                  kRecordable<VkBufferCopy> && kRecordable<VkBufferImageCopy> && kRecordable<VkImageSubresourceRange> &&
                  kRecordable<VkRenderingAttachmentInfo> && kRecordable<VkWriteDescriptorSet> && kRecordable<VkDescriptorBufferInfo> &&
                  kRecordable<VkDescriptorImageInfo>,
              "the arrays records carry are plain values");

}  // namespace gpu::stream
