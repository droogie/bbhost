// Where a direct mapping goes when the kernel chooses its address.
//
// sceKernelMapDirectMemory without SCE_KERNEL_MAP_FIXED leaves the address to
// the kernel. kernel.cpp maps direct memory at its identity address (the CPU
// window + phys) while that is free; when it is not - the game maps memory
// that is mapped already a second time, with another protection - the
// mapping goes wherever there is room, as the console's kernel puts it.
//
// A view of the memory's backing starts and ends on whole blocks of it: 64 KiB
// on Windows (the allocation granularity, for a view's offset into its
// section and for its base address alike), the page on Linux. So the view
// covers the blocks that hold [phys, phys + len), and the mapping starts
// inside it at phys's offset in its block: the address the game gets keeps
// that offset. The rest of the view's blocks is padding no other mapping
// uses.
#pragma once

#include <cstdint>

struct DmemPlacement {
    std::uint64_t view_offset = 0;  // the view's offset into the backing: phys rounded down to its block
    std::uint64_t view_len = 0;     // whole blocks, covering [phys, phys + len)
    std::uint64_t va_offset = 0;    // where the mapping starts in the view: phys's offset in its block
    std::uint64_t base_align = 0;   // what the view's base must be a multiple of: the block, or a larger alignment asked for
    bool ok = false;                // false: no base gives an address with the alignment asked for
};

// phys and len: the mapping (len above 0); align: the alignment the game asked
// for (a power of two, 0 for none); block: the views' granularity (a power of
// two). The address is the view's base + va_offset, and the base is a
// multiple of base_align, so it meets `align` exactly when phys's offset in
// its block is a multiple of it.
constexpr DmemPlacement dmem_placement(std::uint64_t phys, std::uint64_t len, std::uint64_t align, std::uint64_t block) {
    DmemPlacement p;
    p.view_offset = phys & ~(block - 1);
    p.view_len = ((phys + len + block - 1) & ~(block - 1)) - p.view_offset;
    p.va_offset = phys - p.view_offset;
    p.base_align = align > block ? align : block;
    p.ok = len != 0 && (align == 0 || p.va_offset % align == 0);
    return p;
}
