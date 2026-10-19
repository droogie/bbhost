// YEBIS's post-processing parameter setters (decomp/yebis/ppfx_setters.cpp)
// beside the game's own entries (tests/decomp/eboot_kit.h). Every CPostEffect
// method an entry hands on to is a recorder of its own here, taking every
// argument register (rdi..r9, xmm0..xmm7); the entries' arguments, the effect
// pointer and the methods' results are generated, garbage in what the method
// does not read. Ours must reach the game's method with what the game's entry
// gives it - the effect, each integer argument in the width the method reads
// (a zero-extended byte as 32 bits), the floats' bits - and return what it
// returns; the five that write in place must write the same bytes and return
// 0. Six deliberate mistakes must be caught, and the entries hold the
// instructions written against. Skips without the 1.09 eboot.
#define DECOMP_YEBIS_PPFX_SETTERS_MUTANTS
int g_ppfx_setters_mutant = 0;
#include "decomp/yebis/ppfx_setters.cpp"

#include "eboot_kit.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

eboot_kit::Rng g_rng(0x5e7);

struct Regs {
    int who = -1;
    u64 r[6] = {};
    u64 x[8] = {};
};
Regs g_seen;
u64 g_ret = 0;

double bits(u64 v) {
    double d;
    std::memcpy(&d, &v, 8);
    return d;
}
u64 bits(double d) {
    u64 v;
    std::memcpy(&v, &d, 8);
    return v;
}

template <int N>
GUEST_ABI u64 method(u64 rdi, u64 rsi, u64 rdx, u64 rcx, u64 r8, u64 r9, double x0, double x1, double x2, double x3, double x4,
                     double x5, double x6, double x7) {
    g_seen.who = N;
    const u64 r[6] = {rdi, rsi, rdx, rcx, r8, r9};
    const double x[8] = {x0, x1, x2, x3, x4, x5, x6, x7};
    for (int i = 0; i < 6; ++i) g_seen.r[i] = r[i];
    for (int i = 0; i < 8; ++i) g_seen.x[i] = bits(x[i]);
    return g_ret;
}

using All = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64, u64, double, double, double, double, double, double, double, double);

struct Entry {
    const char* name;
    u64 bn;          // the entry
    u64 target;      // the method, or the field's offset (kinds 3 and 4)
    int kind;        // 0 thunk, 1 thunk with a 0 second argument, 2 fb37d0's shuffle, 3 a float stored, 4 a byte stored
    const void* ours;
    const u8* code;  // the instructions written against
    std::size_t code_len;
    int widths[5];   // the method's integer arguments after rdi, in the width it reads (0: none)
    int floats;      // its float arguments
    All rec;         // its recorder
};

const Entry kEntries[] = {
    {"ppfx_set_render_scene_luminance_scale", 0xfb2710, 0xfd48d0, 0, reinterpret_cast<const void*>(&ppfx_set_render_scene_luminance_scale), kEntry_fb2710, sizeof kEntry_fb2710, {0, 0, 0, 0, 0}, 5, reinterpret_cast<All>(&method<0>)},
    {"ppfx_sub_fb2720", 0xfb2720, 0xfd85d0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb2720), kEntry_fb2720, sizeof kEntry_fb2720, {0, 0, 0, 0, 0}, 3, reinterpret_cast<All>(&method<1>)},
    {"ppfx_sub_fb2740", 0xfb2740, 0xfd4a40, 0, reinterpret_cast<const void*>(&ppfx_sub_fb2740), kEntry_fb2740, sizeof kEntry_fb2740, {0, 0, 0, 0, 0}, 3, reinterpret_cast<All>(&method<2>)},
    {"ppfx_sub_fb27d0", 0xfb27d0, 0xfd8cb0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb27d0), kEntry_fb27d0, sizeof kEntry_fb27d0, {64, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<3>)},
    {"ppfx_sub_fb27f0", 0xfb27f0, 0xfd90f0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb27f0), kEntry_fb27f0, sizeof kEntry_fb27f0, {64, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<4>)},
    {"ppfx_sub_fb3fb0", 0xfb3fb0, 0xff90a0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb3fb0), kEntry_fb3fb0, sizeof kEntry_fb3fb0, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<5>)},
    {"ppfx_set_auto_exposure_enable", 0xfb31c0, 0xfd4fc0, 0, reinterpret_cast<const void*>(&ppfx_set_auto_exposure_enable), kEntry_fb31c0, sizeof kEntry_fb31c0, {32, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<6>)},
    {"ppfx_set_auto_exposure_middle_gray", 0xfb32f0, 0xfd5550, 0, reinterpret_cast<const void*>(&ppfx_set_auto_exposure_middle_gray), kEntry_fb32f0, sizeof kEntry_fb32f0, {32, 0, 0, 0, 0}, 2, reinterpret_cast<All>(&method<7>)},
    {"ppfx_set_auto_exposure_range", 0xfb32e0, 0xfd53c0, 0, reinterpret_cast<const void*>(&ppfx_set_auto_exposure_range), kEntry_fb32e0, sizeof kEntry_fb32e0, {32, 0, 0, 0, 0}, 2, reinterpret_cast<All>(&method<8>)},
    {"ppfx_set_auto_exposure_metering_area", 0xfb3300, 0xfd5870, 0, reinterpret_cast<const void*>(&ppfx_set_auto_exposure_metering_area), kEntry_fb3300, sizeof kEntry_fb3300, {32, 0, 0, 0, 0}, 4, reinterpret_cast<All>(&method<9>)},
    {"ppfx_set_auto_exposure_delay", 0xfb3310, 0xfd56f0, 0, reinterpret_cast<const void*>(&ppfx_set_auto_exposure_delay), kEntry_fb3310, sizeof kEntry_fb3310, {32, 32, 0, 0, 0}, 1, reinterpret_cast<All>(&method<10>)},
    {"ppfx_set_auto_exposure_adjustment", 0xfb3320, 0xfd5130, 0, reinterpret_cast<const void*>(&ppfx_set_auto_exposure_adjustment), kEntry_fb3320, sizeof kEntry_fb3320, {32, 0, 0, 0, 0}, 7, reinterpret_cast<All>(&method<11>)},
    {"ppfx_set_tonemap_parameters", 0xfb3550, 0xfd4d40, 0, reinterpret_cast<const void*>(&ppfx_set_tonemap_parameters), kEntry_fb3550, sizeof kEntry_fb3550, {32, 0, 0, 0, 0}, 4, reinterpret_cast<All>(&method<12>)},
    {"ppfx_sub_fb3560", 0xfb3560, 0xfd4e60, 0, reinterpret_cast<const void*>(&ppfx_sub_fb3560), kEntry_fb3560, sizeof kEntry_fb3560, {0, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<13>)},
    {"ppfx_set_chromatic_aberration_enable", 0xfb37a0, 0xfd3320, 1, reinterpret_cast<const void*>(&ppfx_set_chromatic_aberration_enable), kEntry_fb37a0, sizeof kEntry_fb37a0, {32, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<14>)},
    {"ppfx_sub_fb37d0", 0xfb37d0, 0xfd33a0, 2, reinterpret_cast<const void*>(&ppfx_sub_fb37d0), kEntry_fb37d0, sizeof kEntry_fb37d0, {32, 32, 64, 64, 0}, 0, reinterpret_cast<All>(&method<15>)},
    {"ppfx_initialize_matrix", 0xfb4300, 0xfd6cb0, 0, reinterpret_cast<const void*>(&ppfx_initialize_matrix), kEntry_fb4300, sizeof kEntry_fb4300, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<16>)},
    {"ppfx_multiply_matrix_fb4360", 0xfb4360, 0xff6700, 0, reinterpret_cast<const void*>(&ppfx_multiply_matrix_fb4360), kEntry_fb4360, sizeof kEntry_fb4360, {32, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<17>)},
    {"ppfx_multiply_matrix_fb4390", 0xfb4390, 0xff6be0, 0, reinterpret_cast<const void*>(&ppfx_multiply_matrix_fb4390), kEntry_fb4390, sizeof kEntry_fb4390, {32, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<18>)},
    {"ppfx_multiply_matrix_fb43b0", 0xfb43b0, 0xff7290, 0, reinterpret_cast<const void*>(&ppfx_multiply_matrix_fb43b0), kEntry_fb43b0, sizeof kEntry_fb43b0, {32, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<19>)},
    {"ppfx_multiply_matrix_fb43e0", 0xfb43e0, 0xff6aa0, 0, reinterpret_cast<const void*>(&ppfx_multiply_matrix_fb43e0), kEntry_fb43e0, sizeof kEntry_fb43e0, {32, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<20>)},
    {"ppfx_sub_fb3670", 0xfb3670, 0xfd4f70, 0, reinterpret_cast<const void*>(&ppfx_sub_fb3670), kEntry_fb3670, sizeof kEntry_fb3670, {0, 0, 0, 0, 0}, 2, reinterpret_cast<All>(&method<21>)},
    {"ppfx_set_glare_parameters", 0xfb2840, 0xfd4b60, 0, reinterpret_cast<const void*>(&ppfx_set_glare_parameters), kEntry_fb2840, sizeof kEntry_fb2840, {32, 0, 0, 0, 0}, 3, reinterpret_cast<All>(&method<22>)},
    {"ppfx_sub_fb2860", 0xfb2860, 0xfd94d0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb2860), kEntry_fb2860, sizeof kEntry_fb2860, {64, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<23>)},
    {"ppfx_sub_fb2850", 0xfb2850, 0xfd4c10, 0, reinterpret_cast<const void*>(&ppfx_sub_fb2850), kEntry_fb2850, sizeof kEntry_fb2850, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<24>)},
    {"ppfx_set_glare_bloom_num_levels", 0xfb40e0, 0xff9030, 0, reinterpret_cast<const void*>(&ppfx_set_glare_bloom_num_levels), kEntry_fb40e0, sizeof kEntry_fb40e0, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<25>)},
    {"ppfx_sub_fb3c10", 0xfb3c10, 0xc3c, 3, reinterpret_cast<const void*>(&ppfx_sub_fb3c10), kEntry_fb3c10, sizeof kEntry_fb3c10, {0, 0, 0, 0, 0}, 1, nullptr},
    {"ppfx_sub_fb4130", 0xfb4130, 0xff91f0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb4130), kEntry_fb4130, sizeof kEntry_fb4130, {0, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<26>)},
    {"ppfx_sub_fb4100", 0xfb4100, 0xfd63d0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb4100), kEntry_fb4100, sizeof kEntry_fb4100, {0, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<27>)},
    {"ppfx_sub_fb40f0", 0xfb40f0, 0xff9130, 0, reinterpret_cast<const void*>(&ppfx_sub_fb40f0), kEntry_fb40f0, sizeof kEntry_fb40f0, {0, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<28>)},
    {"ppfx_sub_fb4120", 0xfb4120, 0xff91c0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb4120), kEntry_fb4120, sizeof kEntry_fb4120, {0, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<29>)},
    {"ppfx_sub_fb4110", 0xfb4110, 0xff9190, 0, reinterpret_cast<const void*>(&ppfx_sub_fb4110), kEntry_fb4110, sizeof kEntry_fb4110, {0, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<30>)},
    {"ppfx_sub_fb3c30", 0xfb3c30, 0xc40, 3, reinterpret_cast<const void*>(&ppfx_sub_fb3c30), kEntry_fb3c30, sizeof kEntry_fb3c30, {0, 0, 0, 0, 0}, 1, nullptr},
    {"ppfx_sub_fb41a0", 0xfb41a0, 0xfd99b0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb41a0), kEntry_fb41a0, sizeof kEntry_fb41a0, {0, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<31>)},
    {"ppfx_sub_fb4160", 0xfb4160, 0xfd4c20, 0, reinterpret_cast<const void*>(&ppfx_sub_fb4160), kEntry_fb4160, sizeof kEntry_fb4160, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<32>)},
    {"ppfx_sub_fb4180", 0xfb4180, 0xfd4c40, 0, reinterpret_cast<const void*>(&ppfx_sub_fb4180), kEntry_fb4180, sizeof kEntry_fb4180, {0, 0, 0, 0, 0}, 2, reinterpret_cast<All>(&method<33>)},
    {"ppfx_sub_fb4190", 0xfb4190, 0xfd4c50, 0, reinterpret_cast<const void*>(&ppfx_sub_fb4190), kEntry_fb4190, sizeof kEntry_fb4190, {64, 64, 0, 0, 0}, 0, reinterpret_cast<All>(&method<34>)},
    {"ppfx_set_active_light_shaft", 0xfb28a0, 0xfd33d0, 0, reinterpret_cast<const void*>(&ppfx_set_active_light_shaft), kEntry_fb28a0, sizeof kEntry_fb28a0, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<35>)},
    {"ppfx_set_light_shaft_source_enable", 0xfb28d0, 0xfd9ac0, 0, reinterpret_cast<const void*>(&ppfx_set_light_shaft_source_enable), kEntry_fb28d0, sizeof kEntry_fb28d0, {32, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<36>)},
    {"ppfx_sub_fb2940", 0xfb2940, 0xfd9f00, 0, reinterpret_cast<const void*>(&ppfx_sub_fb2940), kEntry_fb2940, sizeof kEntry_fb2940, {32, 0, 0, 0, 0}, 2, reinterpret_cast<All>(&method<37>)},
    {"ppfx_sub_fb2900", 0xfb2900, 0xfd9c50, 0, reinterpret_cast<const void*>(&ppfx_sub_fb2900), kEntry_fb2900, sizeof kEntry_fb2900, {32, 64, 32, 0, 0}, 6, reinterpret_cast<All>(&method<38>)},
    {"ppfx_sub_fb2920", 0xfb2920, 0xfd9dc0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb2920), kEntry_fb2920, sizeof kEntry_fb2920, {32, 64, 0, 0, 0}, 4, reinterpret_cast<All>(&method<39>)},
    {"ppfx_sub_fb2960", 0xfb2960, 0xfd2a90, 0, reinterpret_cast<const void*>(&ppfx_sub_fb2960), kEntry_fb2960, sizeof kEntry_fb2960, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<40>)},
    {"ppfx_set_auto_focus_enable", 0xfb3350, 0xfd5be0, 0, reinterpret_cast<const void*>(&ppfx_set_auto_focus_enable), kEntry_fb3350, sizeof kEntry_fb3350, {32, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<41>)},
    {"ppfx_set_auto_focus_adjustment", 0xfb34b0, 0xfd5d50, 0, reinterpret_cast<const void*>(&ppfx_set_auto_focus_adjustment), kEntry_fb34b0, sizeof kEntry_fb34b0, {32, 32, 0, 0, 0}, 2, reinterpret_cast<All>(&method<42>)},
    {"ppfx_set_auto_focus_delay", 0xfb3490, 0xfd6250, 0, reinterpret_cast<const void*>(&ppfx_set_auto_focus_delay), kEntry_fb3490, sizeof kEntry_fb3490, {32, 32, 0, 0, 0}, 1, reinterpret_cast<All>(&method<43>)},
    {"ppfx_set_auto_focus_distance_range", 0xfb3470, 0xfd5f10, 0, reinterpret_cast<const void*>(&ppfx_set_auto_focus_distance_range), kEntry_fb3470, sizeof kEntry_fb3470, {32, 0, 0, 0, 0}, 2, reinterpret_cast<All>(&method<44>)},
    {"ppfx_set_auto_focus_position", 0xfb3480, 0xfd60a0, 0, reinterpret_cast<const void*>(&ppfx_set_auto_focus_position), kEntry_fb3480, sizeof kEntry_fb3480, {32, 0, 0, 0, 0}, 2, reinterpret_cast<All>(&method<45>)},
    {"ppfx_set_auto_focus_adjustment_reset", 0xfb3340, 0xfdee90, 0, reinterpret_cast<const void*>(&ppfx_set_auto_focus_adjustment_reset), kEntry_fb3340, sizeof kEntry_fb3340, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<46>)},
    {"ppfx_sub_fb3f80", 0xfb3f80, 0xff9080, 0, reinterpret_cast<const void*>(&ppfx_sub_fb3f80), kEntry_fb3f80, sizeof kEntry_fb3f80, {0, 0, 0, 0, 0}, 1, reinterpret_cast<All>(&method<47>)},
    {"ppfx_sub_fb34a0", 0xfb34a0, 0xfd4d20, 0, reinterpret_cast<const void*>(&ppfx_sub_fb34a0), kEntry_fb34a0, sizeof kEntry_fb34a0, {32, 0, 0, 0, 0}, 2, reinterpret_cast<All>(&method<48>)},
    {"ppfx_sub_fb3fd0", 0xfb3fd0, 0xff2590, 0, reinterpret_cast<const void*>(&ppfx_sub_fb3fd0), kEntry_fb3fd0, sizeof kEntry_fb3fd0, {32, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<49>)},
    {"ppfx_sub_fb2970", 0xfb2970, 0xfdc470, 0, reinterpret_cast<const void*>(&ppfx_sub_fb2970), kEntry_fb2970, sizeof kEntry_fb2970, {32, 0, 0, 0, 0}, 5, reinterpret_cast<All>(&method<50>)},
    {"ppfx_set_depth_of_field_edge_quality", 0xfb45c0, 0xff8a40, 0, reinterpret_cast<const void*>(&ppfx_set_depth_of_field_edge_quality), kEntry_fb45c0, sizeof kEntry_fb45c0, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<51>)},
    {"ppfx_sub_fb3600", 0xfb3600, 0xfd4650, 0, reinterpret_cast<const void*>(&ppfx_sub_fb3600), kEntry_fb3600, sizeof kEntry_fb3600, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<52>)},
    {"ppfx_set_depth_of_field_depth_remap_lut_parameters", 0xfb3660, 0xfd4670, 0, reinterpret_cast<const void*>(&ppfx_set_depth_of_field_depth_remap_lut_parameters), kEntry_fb3660, sizeof kEntry_fb3660, {32, 32, 32, 0, 0}, 1, reinterpret_cast<All>(&method<53>)},
    {"ppfx_set_motion_blur_enable", 0xfb3870, 0xfd3b00, 0, reinterpret_cast<const void*>(&ppfx_set_motion_blur_enable), kEntry_fb3870, sizeof kEntry_fb3870, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<54>)},
    {"ppfx_sub_fb38b0", 0xfb38b0, 0x9e8, 4, reinterpret_cast<const void*>(&ppfx_sub_fb38b0), kEntry_fb38b0, sizeof kEntry_fb38b0, {8, 0, 0, 0, 0}, 0, nullptr},
    {"ppfx_sub_fb3890", 0xfb3890, 0xfd3b80, 0, reinterpret_cast<const void*>(&ppfx_sub_fb3890), kEntry_fb3890, sizeof kEntry_fb3890, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<55>)},
    {"ppfx_sub_fb38e0", 0xfb38e0, 0xfd3c10, 0, reinterpret_cast<const void*>(&ppfx_sub_fb38e0), kEntry_fb38e0, sizeof kEntry_fb38e0, {0, 0, 0, 0, 0}, 2, reinterpret_cast<All>(&method<56>)},
    {"ppfx_sub_fb27c0", 0xfb27c0, 0xfd8a70, 0, reinterpret_cast<const void*>(&ppfx_sub_fb27c0), kEntry_fb27c0, sizeof kEntry_fb27c0, {64, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<57>)},
    {"ppfx_sub_fb27e0", 0xfb27e0, 0xfd8eb0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb27e0), kEntry_fb27e0, sizeof kEntry_fb27e0, {64, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<58>)},
    {"ppfx_sub_fb2800", 0xfb2800, 0x6ac, 3, reinterpret_cast<const void*>(&ppfx_sub_fb2800), kEntry_fb2800, sizeof kEntry_fb2800, {0, 0, 0, 0, 0}, 1, nullptr},
    {"ppfx_sub_fb38f0", 0xfb38f0, 0xfd3b90, 0, reinterpret_cast<const void*>(&ppfx_sub_fb38f0), kEntry_fb38f0, sizeof kEntry_fb38f0, {32, 32, 0, 0, 0}, 3, reinterpret_cast<All>(&method<59>)},
    {"ppfx_sub_fb3900", 0xfb3900, 0xfd3be0, 0, reinterpret_cast<const void*>(&ppfx_sub_fb3900), kEntry_fb3900, sizeof kEntry_fb3900, {32, 0, 0, 0, 0}, 2, reinterpret_cast<All>(&method<60>)},
    {"ppfx_set_feedback_enable", 0xfb34c0, 0xfd40e0, 0, reinterpret_cast<const void*>(&ppfx_set_feedback_enable), kEntry_fb34c0, sizeof kEntry_fb34c0, {32, 0, 0, 0, 0}, 0, reinterpret_cast<All>(&method<61>)},
    {"ppfx_sub_fb34d0", 0xfb34d0, 0xfd4150, 0, reinterpret_cast<const void*>(&ppfx_sub_fb34d0), kEntry_fb34d0, sizeof kEntry_fb34d0, {0, 0, 0, 0, 0}, 3, reinterpret_cast<All>(&method<62>)},
    {"ppfx_sub_fb34e0", 0xfb34e0, 0x950, 3, reinterpret_cast<const void*>(&ppfx_sub_fb34e0), kEntry_fb34e0, sizeof kEntry_fb34e0, {0, 0, 0, 0, 0}, 1, nullptr},
    {"ppfx_set_antialias_enable", 0xfb37f0, 0xfd31f0, 1, reinterpret_cast<const void*>(&ppfx_set_antialias_enable), kEntry_fb37f0, sizeof kEntry_fb37f0, {32, 32, 0, 0, 0}, 0, reinterpret_cast<All>(&method<63>)},
};

int g_failed = 0;
bool g_quiet = false;
void fail(const char* what, int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: %s, case %d: %s\n", what, trial, why);
}

// The effect the five stores write into, from the same bytes for both.
alignas(16) u8 g_effect[0x1000];
u8 g_start[sizeof g_effect], g_after_game[sizeof g_effect];

// Whether the method saw the same argument: the bits it reads.
bool same_int(u64 a, u64 b, int width) {
    if (width == 64) return a == b;
    const u64 mask = width == 32 ? 0xffffffffull : 0xffull;
    return (a & mask) == (b & mask);
}

u64 call(const void* f, const u64 (&r)[6], const u64 (&x)[8]) {
    return reinterpret_cast<All>(const_cast<void*>(f))(r[0], r[1], r[2], r[3], r[4], r[5], bits(x[0]), bits(x[1]), bits(x[2]), bits(x[3]),
                                                       bits(x[4]), bits(x[5]), bits(x[6]), bits(x[7]));
}

long run(int cases) {
    long n = 0;
    for (const Entry& e : kEntries) {
        const bool in_place = e.kind >= 3;
        for (int c = 0; c < cases; ++c) {
            u64 r[6], x[8];
            for (u64& v : r) v = g_rng.next();
            for (u64& v : x) v = g_rng.next();
            // The bytes an entry zero-extends: often 0 or 1, now and then any.
            for (int i = 0; i < 5; ++i) {
                if (g_rng.chance(0.4)) r[1 + i] = (r[1 + i] & ~0xffull) | (g_rng.chance(0.5) ? (g_rng.u32() & 1) : (g_rng.u32() & 0xff));
            }
            for (u64& v : x) {
                if (g_rng.chance(0.5)) {
                    const float f = g_rng.value(-1000.0f, 1000.0f);
                    u32 b;
                    std::memcpy(&b, &f, 4);
                    v = (v & ~0xffffffffull) | b;
                }
            }
            store<u64>(kSinglePostEffect, in_place ? reinterpret_cast<u64>(g_effect) : g_rng.next());
            for (u8& b : g_start) b = static_cast<u8>(g_rng.u32());
            g_ret = g_rng.next();

            std::memcpy(g_effect, g_start, sizeof g_effect);
            g_seen = Regs{};
            const u64 game_ret = call(eboot_kit::at(e.bn), r, x);
            const Regs game = g_seen;
            std::memcpy(g_after_game, g_effect, sizeof g_effect);

            std::memcpy(g_effect, g_start, sizeof g_effect);
            g_seen = Regs{};
            const u64 our_ret = call(e.ours, r, x);
            const Regs ours = g_seen;
            ++n;

            if (our_ret != game_ret) fail(e.name, c, "the return value");
            if (in_place) {
                if (std::memcmp(g_effect, g_after_game, sizeof g_effect) != 0) fail(e.name, c, "the effect's bytes");
                if (ours.who != -1 || game.who != -1) fail(e.name, c, "a method was called");
                continue;
            }
            if (game.who < 0 || ours.who != game.who) {
                fail(e.name, c, "the method");
                continue;
            }
            if (ours.r[0] != game.r[0]) fail(e.name, c, "the effect");
            for (int i = 0; i < 5; ++i) {
                if (e.widths[i] && !same_int(ours.r[1 + i], game.r[1 + i], e.widths[i])) fail(e.name, c, "an integer argument");
            }
            for (int i = 0; i < e.floats; ++i) {
                if ((ours.x[i] & 0xffffffffull) != (game.x[i] & 0xffffffffull)) fail(e.name, c, "a float argument");
            }
        }
    }
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_ppfx_setters_mutant = mutant;
    g_quiet = true;
    run(200);
    g_quiet = false;
    g_ppfx_setters_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("yebis_ppfx_setters_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 2000;
    for (const Entry& e : kEntries) {
        if (e.kind < 3) eboot_kit::stub(e.target, reinterpret_cast<const void*>(e.rec));
        const u8* at = eboot_kit::file_at("yebis_ppfx_setters_test", e.bn);
        if (!at || std::memcmp(at, e.code, e.code_len) != 0) fail(e.name, -1, "the entry's instructions");
    }
    const long total = run(cases);
    for (const int mutant : {1, 2, 3, 4, 5, 6}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("yebis_ppfx_setters_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("yebis_ppfx_setters_test: ok - %zu entries, %ld cases, 6 mutants caught\n", sizeof kEntries / sizeof kEntries[0], total);
    return 0;
}

