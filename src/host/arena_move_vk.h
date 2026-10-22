// The move's steps (host/arena_move.h) over Vulkan and the kernel: piece
// buffers in video memory the CPU writes, placed over the heap with
// VK_EXT_map_memory_placed, and the heap's own mapping protected, saved and
// mapped again with mprotect/mmap. gpu.cpp's arena_place_locked runs them in
// the game; tools/arena_move_check.cpp runs the same on a GPU outside it.
// Linux only: Windows has no placed mappings.
#pragma once

#include <vulkan/vulkan.h>

#if defined(VK_EXT_map_memory_placed) && defined(VK_KHR_map_memory2) && !defined(_WIN32)
#include "host/arena_move.h"

#include <sys/mman.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace arena_move {

struct VkPiece {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    std::uint64_t size = 0;
};

constexpr VkBufferUsageFlags kPieceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

struct VkOps {
    VkDevice dev = VK_NULL_HANDLE;
    const Pieces* p = nullptr;
    std::uint32_t type = UINT32_MAX;  // video memory the CPU writes
    PFN_vkMapMemory2KHR map2 = nullptr;
    PFN_vkUnmapMemory2KHR unmap2 = nullptr;
    bool unmap_reserve = false;               // the device's memoryUnmapReserve, enabled
    void (*on_moving)(bool) = nullptr;        // GX's writes wait while on (gxvk::arena_moving)
    void (*on_adopt)(std::uint64_t, const VkPiece&) = nullptr;  // piece as the import at its address
    std::vector<VkPiece> made;
    std::unique_ptr<std::uint8_t[]> saved;

    bool allocate(std::uint64_t, std::uint64_t len) {
        VkPiece c;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = len;
        bci.usage = kPieceUsage;
        if (vkCreateBuffer(dev, &bci, nullptr, &c.buffer) != VK_SUCCESS) return false;
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &flags;
        mai.allocationSize = len;  // a placed mapping takes its allocation whole
        mai.memoryTypeIndex = type;
        if (vkAllocateMemory(dev, &mai, nullptr, &c.memory) != VK_SUCCESS) {
            vkDestroyBuffer(dev, c.buffer, nullptr);
            return false;
        }
        vkBindBufferMemory(dev, c.buffer, c.memory, 0);
        VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        bai.buffer = c.buffer;
        c.address = vkGetBufferDeviceAddress(dev, &bai);
        c.size = len;
        made.push_back(c);
        return true;
    }
    void release(std::uint64_t) {
        for (VkPiece& c : made) {
            vkDestroyBuffer(dev, c.buffer, nullptr);
            vkFreeMemory(dev, c.memory, nullptr);
        }
        made.clear();
    }
    void moving(bool on) {
        if (on_moving) on_moving(on);
    }
    bool protect(std::uint64_t at, std::uint64_t len, bool writable) {
        return ::mprotect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(at)), len, writable ? PROT_READ | PROT_WRITE : PROT_READ) == 0;
    }
    bool save(std::uint64_t at, std::uint64_t len) {
        saved.reset(new (std::nothrow) std::uint8_t[len]);
        if (!saved) return false;
        std::memcpy(saved.get(), reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), len);
        return true;
    }
    bool fill(std::uint64_t k, std::uint64_t off, std::uint64_t len) {
        void* m = nullptr;
        if (vkMapMemory(dev, made[k].memory, 0, VK_WHOLE_SIZE, 0, &m) != VK_SUCCESS) return false;
        std::memcpy(m, saved.get() + off, len);
        vkUnmapMemory(dev, made[k].memory);
        return true;
    }
    bool place(std::uint64_t k, std::uint64_t at) {
        void* const want = reinterpret_cast<void*>(static_cast<std::uintptr_t>(at));
        VkMemoryMapPlacedInfoEXT placed{VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT};
        placed.pPlacedAddress = want;
        VkMemoryMapInfoKHR mi{VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR};
        mi.pNext = &placed;
        mi.flags = VK_MEMORY_MAP_PLACED_BIT_EXT;
        mi.memory = made[k].memory;
        mi.offset = 0;
        mi.size = VK_WHOLE_SIZE;
        void* got = nullptr;
        const VkResult r = map2(dev, &mi, &got);
        if (r == VK_SUCCESS && got != want) {  // mapped, but not there
            vkUnmapMemory(dev, made[k].memory);
            return false;
        }
        return r == VK_SUCCESS;
    }
    void read_back(std::uint64_t k, std::uint64_t off, std::uint64_t len) {
        std::memcpy(saved.get() + off, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(p->start(k))), len);
    }
    void unplace(std::uint64_t k, std::uint64_t) {
        if (unmap2) {
            VkMemoryUnmapInfoKHR ui{VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR};
            ui.memory = made[k].memory;
            ui.flags = unmap_reserve ? VK_MEMORY_UNMAP_RESERVE_BIT_EXT : 0;
            unmap2(dev, &ui);
        } else {
            vkUnmapMemory(dev, made[k].memory);
        }
    }
    bool restore(std::uint64_t at, std::uint64_t len, std::uint64_t off) {
        void* const want = reinterpret_cast<void*>(static_cast<std::uintptr_t>(at));
        // Over the range the driver kept reserved, or into the hole it left.
        const int fixed = unmap_reserve ? MAP_FIXED : MAP_FIXED_NOREPLACE;
        if (::mmap(want, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | fixed, -1, 0) != want) return false;
        std::memcpy(want, saved.get() + off, len);
        return true;
    }
    void adopt(std::uint64_t k, std::uint64_t at) {
        if (on_adopt) on_adopt(at, made[k]);
    }
    void drop_saved() { saved.reset(); }
};

}  // namespace arena_move
#endif
