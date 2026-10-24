// Exercise the production helper and immediate caller without a Vulkan device.
// Link gpu_memtrack.cpp too: the allocation macros and accounting stay real.
#include "../src/host/buffer_shadow.cpp"

#include <iostream>
#include <stdexcept>

#undef vkAllocateMemory
#undef vkFreeMemory

namespace {
std::uintptr_t next_handle = 100;
template<class T> T handle() { return reinterpret_cast<T>(++next_handle); }
struct Buffer { VkDeviceMemory memory{}; };
std::unordered_map<VkBuffer, Buffer> buffers;
std::unordered_map<VkDeviceMemory, VkDeviceSize> memories;
std::vector<std::string> calls;
VkResult create_result = VK_SUCCESS, allocate_result = VK_SUCCESS, bind_result = VK_SUCCESS;
bool poison = false;
VkMemoryRequirements requirements{(4ull << 20) + 256, 256, 1};
unsigned copies = 0;
const VkBuffer source = handle<VkBuffer>();
const std::uint64_t source_va = 0x10000;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void release(gpu::StageChunk c) {
    vkDestroyBuffer(gpu::g.device, c.buffer, nullptr);
    bb_free_memory(gpu::g.device, c.memory, nullptr);
}
void closed() {
    check(buffers.empty() && memories.empty() && bb_memory_local_bytes() == 0, "all acquired resources/accounting released");
}
void trace(const char* name, bool result, const gpu::StageChunk& c) {
    std::cout << name << " result=" << result << " buffer=" << c.buffer << " memory=" << c.memory
              << " chunks=" << gpu::g_stage_chunks << " local_bytes=" << bb_memory_local_bytes()
              << " buffers=" << buffers.size() << " allocations=" << memories.size() << " calls=";
    for (const auto& call : calls) std::cout << call << ',';
    std::cout << '\n';
}
}  // namespace

namespace gpu {
Gpu g;
const bool g_import_audit = false;
void import_audit(const GuestMapInfo&, int) {}
void audit_range(std::uint64_t, std::uint64_t, ImportUse) {}
const Chunk* dmem_import(std::uint32_t chunk, std::uint64_t off, std::uint64_t bytes) {
    static Chunk imported;
    return chunk == 0 && off + bytes <= 4096 ? &imported : nullptr;
}
std::uint32_t find_memory_type(std::uint32_t bits, VkMemoryPropertyFlags wanted) {
    check(wanted == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "device-local preference preserved");
    return bits & 1 ? 0 : UINT32_MAX;
}
}  // namespace gpu
void host_log(const char*, ...) {}

extern "C" {
VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice, const VkBufferCreateInfo* i, const VkAllocationCallbacks*, VkBuffer* out) {
    calls.push_back("create");
    std::cout << "raw create result=" << create_result << " poison=" << poison << '\n';
    check(i->sType == VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO && !i->pNext && !i->flags, "buffer create shape");
    check(i->size == (4ull << 20), "4 MiB chunk size unchanged");
    check(i->usage == (VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT), "usages unchanged");
    if (create_result != VK_SUCCESS) {
        if (poison) *out = reinterpret_cast<VkBuffer>(0xdead);
        return create_result;
    }
    *out = handle<VkBuffer>(); buffers[*out] = {};
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice, VkBuffer b, VkMemoryRequirements* out) {
    calls.push_back("requirements"); check(buffers.count(b), "requirements for acquired buffer"); *out = requirements;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo* i, const VkAllocationCallbacks*, VkDeviceMemory* out) {
    calls.push_back("allocate");
    std::cout << "raw allocate result=" << allocate_result << " poison=" << poison << '\n';
    check(i->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO && !i->pNext, "allocation shape");
    check(i->allocationSize == requirements.size && i->memoryTypeIndex == 0, "requirements size/type preserved");
    if (allocate_result != VK_SUCCESS) {
        if (poison) *out = reinterpret_cast<VkDeviceMemory>(0xbeef);
        return allocate_result;
    }
    *out = handle<VkDeviceMemory>(); memories[*out] = i->allocationSize;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice, VkBuffer b, VkDeviceMemory m, VkDeviceSize offset) {
    calls.push_back("bind");
    std::cout << "raw bind result=" << bind_result << " offset=" << offset << '\n';
    check(buffers.count(b) && memories.count(m), "bind only acquired handles");
    check(offset == 0, "zero binding offset preserved");
    if (bind_result == VK_SUCCESS) buffers.at(b).memory = m;
    return bind_result;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice, VkBuffer b, const VkAllocationCallbacks*) {
    calls.push_back("destroy"); check(buffers.erase(b) == 1, "destroy owned buffer exactly once");
}
VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice, VkDeviceMemory m, const VkAllocationCallbacks*) {
    calls.push_back("free");
    for (const auto& [b, state] : buffers) check(state.memory != m, "destroy buffer before freeing its bound memory");
    check(memories.erase(m) == 1, "free owned memory exactly once");
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool* out) {
    *out = handle<VkCommandPool>(); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer* out) {
    *out = handle<VkCommandBuffer>(); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer, VkBuffer src, VkBuffer dst, std::uint32_t n, const VkBufferCopy* r) {
    check(src == source && buffers.count(dst) && buffers.at(dst).memory, "caller copies original source to bound destination");
    check(n == 1 && r->srcOffset == 64 && r->dstOffset == 0 && r->size == 32, "caller copy range preserved"); ++copies;
}
}  // extern C

int main(int argc, char** argv) {
    using namespace gpu;
    try {
        const std::string name = argc == 2 ? argv[1] : "bind";
        bb_memory_set_local_types(1);
        g.device = handle<VkDevice>();
        if (name == "create" || name == "create-poison") create_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        if (name == "allocate" || name == "allocate-poison") allocate_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        if (name == "missing-type") requirements.memoryTypeBits = 2;
        if (name == "bind" || name == "caller-bind") bind_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        if (name == "bind-host") bind_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        if (name == "limit") g_stage_chunks = kStageMaxChunks;
        if (name == "given-back") g_given_back = true;
        poison = name == "create-poison" || name == "allocate-poison";
        const auto count_before = g_stage_chunks;
        StageChunk c;
        if (name == "caller-bind") {
            g.maps = {{source_va, 4096, 0, true, 0x30}};
            g.slot = 0; g.record_serial = 1;
            g_mirrors[0].pages.resize(1);
            // CTest selects BBHOST_BUFFER_SHADOW=1; keep the ordinary non-tight device state.
            g.integrated = false; g.tight = false;
            Located loc{source, 64, 128};
            const bool result = shadow_stage_locked(source_va + 64, 32, loc);
            trace(name.c_str(), result, {});
            check(!result, "caller refuses failed bind");
            check(loc.buffer == source && loc.offset == 64 && loc.avail == 128, "caller leaves original source untouched");
            check(copies == 0 && g_stage_held[0].empty() && g_stage_free.empty() && g_stage_fill[0] == 0, "caller publishes/copies nothing on refusal");
            check(g_stage_refused == 1 && g_staged == 0 && g_staged_bytes == 0 && g_stage_chunks == 0, "caller refusal/count state");
            closed();
            bind_result = VK_SUCCESS;
            check(shadow_stage_locked(source_va + 64, 32, loc), "caller succeeds after failed acquisition");
            check(copies == 1 && loc.buffer != source && loc.offset == 0 && loc.avail == 32, "caller publishes staged range only after success");
            check(g_stage_held[0].size() == 1 && g_stage_fill[0] == 256 && g_stage_chunks == 1 && g_staged == 1 && g_staged_bytes == 32, "caller successful pool/count state");
            release(g_stage_held[0].back()); g_stage_held[0].clear();
        } else {
            const bool result = new_stage_chunk(c);
            trace(name.c_str(), result, c);
            if (name == "success") {
                check(result && c.buffer && c.memory && buffers.at(c.buffer).memory == c.memory, "bound chunk published");
                check(g_stage_chunks == 1 && bb_memory_local_bytes() == requirements.size, "successful count/accounting");
                release(c);
            } else {
                check(!result && !c.buffer && !c.memory && g_stage_chunks == count_before, "failed acquisition unpublished/uncounted");
                const std::vector<std::string> expected = name == "limit" || name == "given-back" ? std::vector<std::string>{} :
                    name == "create" || name == "create-poison" ? std::vector<std::string>{"create"} :
                    name == "missing-type" ? std::vector<std::string>{"create", "requirements", "destroy"} :
                    name == "allocate" || name == "allocate-poison" ? std::vector<std::string>{"create", "requirements", "allocate", "destroy"} :
                    std::vector<std::string>{"create", "requirements", "allocate", "bind", "destroy", "free"};
                check(calls == expected, "failure cleanup order and exact call counts");
                closed();
                // A failed raw acquisition does not poison the next ordinary attempt.
                create_result = allocate_result = bind_result = VK_SUCCESS;
                requirements.memoryTypeBits = 1; g_given_back = false; g_stage_chunks = 0;
                check(new_stage_chunk(c) && c.buffer && c.memory && g_stage_chunks == 1, "valid attempt after failure");
                check(bb_memory_local_bytes() == requirements.size, "retry accounted exactly once");
                release(c);
            }
        }
        closed();
        std::cout << "PASS " << name << "; all owned resources closed; synthetic API only\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
