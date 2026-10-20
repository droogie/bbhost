// Holds the guest's low windows (core/mac_vm.h) as a zero-fill section of
// bbhost's own image (linked at 0x10000, no ASLR), so Rosetta 2, which maps
// at the first free address >= 4 GiB before main, cannot take them.
//
// ld puts large zero-fill objects at the end of __DATA in link order: this
// must stay the last bbhost source (CMakeLists.txt), or the others' land
// gigabytes from the code, out of rip-relative reach.
#include "core/mac_vm.h"

#if defined(__APPLE__)

// Checked at start by hle_kernel_reserve_guest_windows.
#define BB_GUEST_RESERVE_SIZE 0x310000000
#define BB_STR2(x) #x
#define BB_STR(x) BB_STR2(x)
static_assert(BB_GUEST_RESERVE_SIZE >= kMacVmImageBase + kMacVmImageSize, "the reservation must reach past the eboot's window");

asm(".globl _bb_guest_reserve\n"
    ".zerofill __DATA,__huge,_bb_guest_reserve," BB_STR(BB_GUEST_RESERVE_SIZE) ",14\n");
extern "C" char bb_guest_reserve[];

void mac_vm_image_reservation(std::uint64_t* lo, std::uint64_t* hi) {
    *lo = reinterpret_cast<std::uint64_t>(bb_guest_reserve);
    *hi = *lo + BB_GUEST_RESERVE_SIZE;
}

#endif
