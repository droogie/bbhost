// The guest's low windows on macOS (core/mac_vm.h), held by bbhost's own
// image from the moment the process exists: a zero-fill section, which costs
// no memory and no file space. bbhost is linked at 0x10000 without ASLR, so
// the section starts just past its data (about 26 MiB today) and covers the
// direct memory window at 4 GiB and the eboot's at 0x300000000; Rosetta 2's
// own regions, which take the first free address at or above 4 GiB before
// main runs, land above it.
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
