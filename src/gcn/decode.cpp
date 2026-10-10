// Sea Islands ISA decoder. Opcode tables are indexed by the encoding-local
// opcode; a null entry is an opcode the ISA leaves undefined (CI additions
// over SI are included, VI-only ones are not).
#include "gcn/isa.h"

#include <cstdio>
#include <cstring>

namespace gcn {
namespace {

constexpr const char* kSop2[] = {
    "s_add_u32", "s_sub_u32", "s_add_i32", "s_sub_i32", "s_addc_u32", "s_subb_u32", "s_min_i32", "s_min_u32",
    "s_max_i32", "s_max_u32", "s_cselect_b32", "s_cselect_b64", nullptr, nullptr, "s_and_b32", "s_and_b64",
    "s_or_b32", "s_or_b64", "s_xor_b32", "s_xor_b64", "s_andn2_b32", "s_andn2_b64", "s_orn2_b32", "s_orn2_b64",
    "s_nand_b32", "s_nand_b64", "s_nor_b32", "s_nor_b64", "s_xnor_b32", "s_xnor_b64", "s_lshl_b32", "s_lshl_b64",
    "s_lshr_b32", "s_lshr_b64", "s_ashr_i32", "s_ashr_i64", "s_bfm_b32", "s_bfm_b64", "s_mul_i32", "s_bfe_u32",
    "s_bfe_i32", "s_bfe_u64", "s_bfe_i64", "s_cbranch_g_fork", "s_absdiff_i32",
};
constexpr const char* kSopk[] = {
    "s_movk_i32", nullptr, "s_cmovk_i32", "s_cmpk_eq_i32", "s_cmpk_lg_i32", "s_cmpk_gt_i32", "s_cmpk_ge_i32",
    "s_cmpk_lt_i32", "s_cmpk_le_i32", "s_cmpk_eq_u32", "s_cmpk_lg_u32", "s_cmpk_gt_u32", "s_cmpk_ge_u32",
    "s_cmpk_lt_u32", "s_cmpk_le_u32", "s_addk_i32", "s_mulk_i32", "s_cbranch_i_fork", "s_getreg_b32",
    "s_setreg_b32", nullptr, "s_setreg_imm32_b32",
};
constexpr const char* kSop1[] = {
    nullptr, nullptr, nullptr, "s_mov_b32", "s_mov_b64", "s_cmov_b32", "s_cmov_b64", "s_not_b32", "s_not_b64",
    "s_wqm_b32", "s_wqm_b64", "s_brev_b32", "s_brev_b64", "s_bcnt0_i32_b32", "s_bcnt0_i32_b64", "s_bcnt1_i32_b32",
    "s_bcnt1_i32_b64", "s_ff0_i32_b32", "s_ff0_i32_b64", "s_ff1_i32_b32", "s_ff1_i32_b64", "s_flbit_i32_b32",
    "s_flbit_i32_b64", "s_flbit_i32", "s_flbit_i32_i64", "s_sext_i32_i8", "s_sext_i32_i16", "s_bitset0_b32",
    "s_bitset0_b64", "s_bitset1_b32", "s_bitset1_b64", "s_getpc_b64", "s_setpc_b64", "s_swappc_b64", "s_rfe_b64",
    nullptr, "s_and_saveexec_b64", "s_or_saveexec_b64", "s_xor_saveexec_b64", "s_andn2_saveexec_b64",
    "s_orn2_saveexec_b64", "s_nand_saveexec_b64", "s_nor_saveexec_b64", "s_xnor_saveexec_b64", "s_quadmask_b32",
    "s_quadmask_b64", "s_movrels_b32", "s_movrels_b64", "s_movreld_b32", "s_movreld_b64", "s_cbranch_join",
    nullptr, "s_abs_i32", "s_mov_fed_b32",
};
constexpr const char* kSopc[] = {
    "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32", "s_cmp_eq_u32",
    "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32", "s_bitcmp0_b32",
    "s_bitcmp1_b32", "s_bitcmp0_b64", "s_bitcmp1_b64", "s_setvskip",
};
constexpr const char* kSopp[] = {
    "s_nop", "s_endpgm", "s_branch", nullptr, "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz",
    "s_cbranch_vccnz", "s_cbranch_execz", "s_cbranch_execnz", "s_barrier", nullptr, "s_waitcnt", "s_sethalt",
    "s_sleep", "s_setprio", "s_sendmsg", "s_sendmsghalt", "s_trap", "s_icache_inv", "s_incperflevel",
    "s_decperflevel", "s_ttracedata", "s_cbranch_cdbgsys", "s_cbranch_cdbguser", "s_cbranch_cdbgsys_or_user",
    "s_cbranch_cdbgsys_and_user",
};
constexpr const char* kSmrd[] = {
    "s_load_dword", "s_load_dwordx2", "s_load_dwordx4", "s_load_dwordx8", "s_load_dwordx16", nullptr, nullptr,
    nullptr, "s_buffer_load_dword", "s_buffer_load_dwordx2", "s_buffer_load_dwordx4", "s_buffer_load_dwordx8",
    "s_buffer_load_dwordx16", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "s_dcache_inv_vol", "s_memtime", "s_dcache_inv",
};
constexpr const char* kVop2[] = {
    "v_cndmask_b32", "v_readlane_b32", "v_writelane_b32", "v_add_f32", "v_sub_f32", "v_subrev_f32",
    "v_mac_legacy_f32", "v_mul_legacy_f32", "v_mul_f32", "v_mul_i32_i24", "v_mul_hi_i32_i24", "v_mul_u32_u24",
    "v_mul_hi_u32_u24", "v_min_legacy_f32", "v_max_legacy_f32", "v_min_f32", "v_max_f32", "v_min_i32", "v_max_i32",
    "v_min_u32", "v_max_u32", "v_lshr_b32", "v_lshrrev_b32", "v_ashr_i32", "v_ashrrev_i32", "v_lshl_b32",
    "v_lshlrev_b32", "v_and_b32", "v_or_b32", "v_xor_b32", "v_bfm_b32", "v_mac_f32", "v_madmk_f32", "v_madak_f32",
    "v_bcnt_u32_b32", "v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32", "v_add_i32", "v_sub_i32", "v_subrev_i32",
    "v_addc_u32", "v_subb_u32", "v_subbrev_u32", "v_ldexp_f32", "v_cvt_pkaccum_u8_f32", "v_cvt_pknorm_i16_f32",
    "v_cvt_pknorm_u16_f32", "v_cvt_pkrtz_f16_f32", "v_cvt_pk_u16_u32", "v_cvt_pk_i16_i32",
};
constexpr const char* kVop1[] = {
    "v_nop", "v_mov_b32", "v_readfirstlane_b32", "v_cvt_i32_f64", "v_cvt_f64_i32", "v_cvt_f32_i32",
    "v_cvt_f32_u32", "v_cvt_u32_f32", "v_cvt_i32_f32", "v_mov_fed_b32", "v_cvt_f16_f32", "v_cvt_f32_f16",
    "v_cvt_rpi_i32_f32", "v_cvt_flr_i32_f32", "v_cvt_off_f32_i4", "v_cvt_f32_f64", "v_cvt_f64_f32",
    "v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2", "v_cvt_f32_ubyte3", "v_cvt_u32_f64",
    "v_cvt_f64_u32", "v_trunc_f64", "v_ceil_f64", "v_rndne_f64", "v_floor_f64", nullptr, nullptr, nullptr,
    nullptr, nullptr, "v_fract_f32", "v_trunc_f32", "v_ceil_f32", "v_rndne_f32", "v_floor_f32", "v_exp_f32",
    "v_log_clamp_f32", "v_log_f32", "v_rcp_clamp_f32", "v_rcp_legacy_f32", "v_rcp_f32", "v_rcp_iflag_f32",
    "v_rsq_clamp_f32", "v_rsq_legacy_f32", "v_rsq_f32", "v_rcp_f64", "v_rcp_clamp_f64", "v_rsq_f64",
    "v_rsq_clamp_f64", "v_sqrt_f32", "v_sqrt_f64", "v_sin_f32", "v_cos_f32", "v_not_b32", "v_bfrev_b32",
    "v_ffbh_u32", "v_ffbl_b32", "v_ffbh_i32", "v_frexp_exp_i32_f64", "v_frexp_mant_f64", "v_fract_f64",
    "v_frexp_exp_i32_f32", "v_frexp_mant_f32", "v_clrexcp", "v_movreld_b32", "v_movrels_b32", "v_movrelsd_b32",
};
constexpr const char* kVop3Only[] = {
    "v_mad_legacy_f32", "v_mad_f32", "v_mad_i32_i24", "v_mad_u32_u24", "v_cubeid_f32", "v_cubesc_f32",
    "v_cubetc_f32", "v_cubema_f32", "v_bfe_u32", "v_bfe_i32", "v_bfi_b32", "v_fma_f32", "v_fma_f64", "v_lerp_u8",
    "v_alignbit_b32", "v_alignbyte_b32", "v_mullit_f32", "v_min3_f32", "v_min3_i32", "v_min3_u32", "v_max3_f32",
    "v_max3_i32", "v_max3_u32", "v_med3_f32", "v_med3_i32", "v_med3_u32", "v_sad_u8", "v_sad_hi_u8", "v_sad_u16",
    "v_sad_u32", "v_cvt_pk_u8_f32", "v_div_fixup_f32", "v_div_fixup_f64", "v_lshl_b64", "v_lshr_b64", "v_ashr_i64",
    "v_add_f64", "v_mul_f64", "v_min_f64", "v_max_f64", "v_ldexp_f64", "v_mul_lo_u32", "v_mul_hi_u32",
    "v_mul_lo_i32", "v_mul_hi_i32", "v_div_scale_f32", "v_div_scale_f64", "v_div_fmas_f32", "v_div_fmas_f64",
    "v_msad_u8", "v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8", "v_trig_preop_f64", "v_mqsad_u32_u8", "v_mad_u64_u32",
    "v_mad_i64_i32",
};
constexpr const char* kCmpF[16] = {"f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"};
constexpr const char* kCmpI[8] = {"f", "lt", "eq", "le", "gt", "ne", "ge", "t"};
constexpr const char* kVintrp[] = {"v_interp_p1_f32", "v_interp_p2_f32", "v_interp_mov_f32"};
constexpr const char* kExpTgt[] = {"mrt0", "mrt1", "mrt2", "mrt3", "mrt4", "mrt5", "mrt6", "mrt7", "mrtz", "null"};

struct Named {
    std::uint16_t op;
    const char* name;
};
constexpr Named kDs[] = {
    {0, "ds_add_u32"}, {1, "ds_sub_u32"}, {2, "ds_rsub_u32"}, {3, "ds_inc_u32"}, {4, "ds_dec_u32"},
    {5, "ds_min_i32"}, {6, "ds_max_i32"}, {7, "ds_min_u32"}, {8, "ds_max_u32"}, {9, "ds_and_b32"},
    {10, "ds_or_b32"}, {11, "ds_xor_b32"}, {12, "ds_mskor_b32"}, {13, "ds_write_b32"}, {14, "ds_write2_b32"},
    {15, "ds_write2st64_b32"}, {16, "ds_cmpst_b32"}, {17, "ds_cmpst_f32"}, {18, "ds_min_f32"}, {19, "ds_max_f32"},
    {20, "ds_nop"}, {24, "ds_gws_sema_release_all"}, {25, "ds_gws_init"}, {26, "ds_gws_sema_v"},
    {27, "ds_gws_sema_br"}, {28, "ds_gws_sema_p"}, {29, "ds_gws_barrier"}, {30, "ds_write_b8"},
    {31, "ds_write_b16"}, {32, "ds_add_rtn_u32"}, {33, "ds_sub_rtn_u32"}, {34, "ds_rsub_rtn_u32"},
    {35, "ds_inc_rtn_u32"}, {36, "ds_dec_rtn_u32"}, {37, "ds_min_rtn_i32"}, {38, "ds_max_rtn_i32"},
    {39, "ds_min_rtn_u32"}, {40, "ds_max_rtn_u32"}, {41, "ds_and_rtn_b32"}, {42, "ds_or_rtn_b32"},
    {43, "ds_xor_rtn_b32"}, {44, "ds_mskor_rtn_b32"}, {45, "ds_wrxchg_rtn_b32"}, {46, "ds_wrxchg2_rtn_b32"},
    {47, "ds_wrxchg2st64_rtn_b32"}, {48, "ds_cmpst_rtn_b32"}, {49, "ds_cmpst_rtn_f32"}, {50, "ds_min_rtn_f32"},
    {51, "ds_max_rtn_f32"}, {52, "ds_wrap_rtn_b32"}, {53, "ds_swizzle_b32"}, {54, "ds_read_b32"},
    {55, "ds_read2_b32"}, {56, "ds_read2st64_b32"}, {57, "ds_read_i8"}, {58, "ds_read_u8"}, {59, "ds_read_i16"},
    {60, "ds_read_u16"}, {61, "ds_consume"}, {62, "ds_append"}, {63, "ds_ordered_count"}, {64, "ds_add_u64"},
    {65, "ds_sub_u64"}, {66, "ds_rsub_u64"}, {67, "ds_inc_u64"}, {68, "ds_dec_u64"}, {69, "ds_min_i64"},
    {70, "ds_max_i64"}, {71, "ds_min_u64"}, {72, "ds_max_u64"}, {73, "ds_and_b64"}, {74, "ds_or_b64"},
    {75, "ds_xor_b64"}, {76, "ds_mskor_b64"}, {77, "ds_write_b64"}, {78, "ds_write2_b64"},
    {79, "ds_write2st64_b64"}, {80, "ds_cmpst_b64"}, {81, "ds_cmpst_f64"}, {82, "ds_min_f64"}, {83, "ds_max_f64"},
    {96, "ds_add_rtn_u64"}, {97, "ds_sub_rtn_u64"}, {98, "ds_rsub_rtn_u64"}, {99, "ds_inc_rtn_u64"},
    {100, "ds_dec_rtn_u64"}, {101, "ds_min_rtn_i64"}, {102, "ds_max_rtn_i64"}, {103, "ds_min_rtn_u64"},
    {104, "ds_max_rtn_u64"}, {105, "ds_and_rtn_b64"}, {106, "ds_or_rtn_b64"}, {107, "ds_xor_rtn_b64"},
    {108, "ds_mskor_rtn_b64"}, {109, "ds_wrxchg_rtn_b64"}, {110, "ds_wrxchg2_rtn_b64"},
    {111, "ds_wrxchg2st64_rtn_b64"}, {112, "ds_cmpst_rtn_b64"}, {113, "ds_cmpst_rtn_f64"}, {114, "ds_min_rtn_f64"},
    {115, "ds_max_rtn_f64"}, {118, "ds_read_b64"}, {119, "ds_read2_b64"}, {120, "ds_read2st64_b64"},
    {126, "ds_condxchg32_rtn_b64"}, {128, "ds_add_src2_u32"}, {129, "ds_sub_src2_u32"}, {130, "ds_rsub_src2_u32"},
    {131, "ds_inc_src2_u32"}, {132, "ds_dec_src2_u32"}, {133, "ds_min_src2_i32"}, {134, "ds_max_src2_i32"},
    {135, "ds_min_src2_u32"}, {136, "ds_max_src2_u32"}, {137, "ds_and_src2_b32"}, {138, "ds_or_src2_b32"},
    {139, "ds_xor_src2_b32"}, {141, "ds_write_src2_b32"}, {146, "ds_min_src2_f32"}, {147, "ds_max_src2_f32"},
    {192, "ds_add_src2_u64"}, {193, "ds_sub_src2_u64"}, {194, "ds_rsub_src2_u64"}, {195, "ds_inc_src2_u64"},
    {196, "ds_dec_src2_u64"}, {197, "ds_min_src2_i64"}, {198, "ds_max_src2_i64"}, {199, "ds_min_src2_u64"},
    {200, "ds_max_src2_u64"}, {201, "ds_and_src2_b64"}, {202, "ds_or_src2_b64"}, {203, "ds_xor_src2_b64"},
    {205, "ds_write_src2_b64"}, {210, "ds_min_src2_f64"}, {211, "ds_max_src2_f64"}, {222, "ds_write_b96"},
    {223, "ds_write_b128"}, {254, "ds_read_b96"}, {255, "ds_read_b128"},
};
constexpr Named kMubuf[] = {
    {0, "buffer_load_format_x"}, {1, "buffer_load_format_xy"}, {2, "buffer_load_format_xyz"},
    {3, "buffer_load_format_xyzw"}, {4, "buffer_store_format_x"}, {5, "buffer_store_format_xy"},
    {6, "buffer_store_format_xyz"}, {7, "buffer_store_format_xyzw"}, {8, "buffer_load_ubyte"},
    {9, "buffer_load_sbyte"}, {10, "buffer_load_ushort"}, {11, "buffer_load_sshort"}, {12, "buffer_load_dword"},
    {13, "buffer_load_dwordx2"}, {14, "buffer_load_dwordx4"}, {15, "buffer_load_dwordx3"},
    {24, "buffer_store_byte"}, {26, "buffer_store_short"}, {28, "buffer_store_dword"},
    {29, "buffer_store_dwordx2"}, {30, "buffer_store_dwordx4"}, {31, "buffer_store_dwordx3"},
    {48, "buffer_atomic_swap"}, {49, "buffer_atomic_cmpswap"}, {50, "buffer_atomic_add"},
    {51, "buffer_atomic_sub"}, {52, "buffer_atomic_rsub"}, {53, "buffer_atomic_smin"}, {54, "buffer_atomic_umin"},
    {55, "buffer_atomic_smax"}, {56, "buffer_atomic_umax"}, {57, "buffer_atomic_and"}, {58, "buffer_atomic_or"},
    {59, "buffer_atomic_xor"}, {60, "buffer_atomic_inc"}, {61, "buffer_atomic_dec"}, {62, "buffer_atomic_fcmpswap"},
    {63, "buffer_atomic_fmin"}, {64, "buffer_atomic_fmax"}, {80, "buffer_atomic_swap_x2"},
    {81, "buffer_atomic_cmpswap_x2"}, {82, "buffer_atomic_add_x2"}, {83, "buffer_atomic_sub_x2"},
    {84, "buffer_atomic_rsub_x2"}, {85, "buffer_atomic_smin_x2"}, {86, "buffer_atomic_umin_x2"},
    {87, "buffer_atomic_smax_x2"}, {88, "buffer_atomic_umax_x2"}, {89, "buffer_atomic_and_x2"},
    {90, "buffer_atomic_or_x2"}, {91, "buffer_atomic_xor_x2"}, {92, "buffer_atomic_inc_x2"},
    {93, "buffer_atomic_dec_x2"}, {94, "buffer_atomic_fcmpswap_x2"}, {95, "buffer_atomic_fmin_x2"},
    {96, "buffer_atomic_fmax_x2"}, {112, "buffer_wbinvl1_vol"}, {113, "buffer_wbinvl1"},
};
constexpr const char* kMtbuf[] = {
    "tbuffer_load_format_x", "tbuffer_load_format_xy", "tbuffer_load_format_xyz", "tbuffer_load_format_xyzw",
    "tbuffer_store_format_x", "tbuffer_store_format_xy", "tbuffer_store_format_xyz", "tbuffer_store_format_xyzw",
};
constexpr Named kMimg[] = {
    {0, "image_load"}, {1, "image_load_mip"}, {2, "image_load_pck"}, {3, "image_load_pck_sgn"},
    {4, "image_load_mip_pck"}, {5, "image_load_mip_pck_sgn"}, {8, "image_store"}, {9, "image_store_mip"},
    {10, "image_store_pck"}, {11, "image_store_mip_pck"}, {14, "image_get_resinfo"}, {15, "image_atomic_swap"},
    {16, "image_atomic_cmpswap"}, {17, "image_atomic_add"}, {18, "image_atomic_sub"}, {19, "image_atomic_rsub"},
    {20, "image_atomic_smin"}, {21, "image_atomic_umin"}, {22, "image_atomic_smax"}, {23, "image_atomic_umax"},
    {24, "image_atomic_and"}, {25, "image_atomic_or"}, {26, "image_atomic_xor"}, {27, "image_atomic_inc"},
    {28, "image_atomic_dec"}, {29, "image_atomic_fcmpswap"}, {30, "image_atomic_fmin"}, {31, "image_atomic_fmax"},
    {32, "image_sample"}, {33, "image_sample_cl"}, {34, "image_sample_d"}, {35, "image_sample_d_cl"},
    {36, "image_sample_l"}, {37, "image_sample_b"}, {38, "image_sample_b_cl"}, {39, "image_sample_lz"},
    {40, "image_sample_c"}, {41, "image_sample_c_cl"}, {42, "image_sample_c_d"}, {43, "image_sample_c_d_cl"},
    {44, "image_sample_c_l"}, {45, "image_sample_c_b"}, {46, "image_sample_c_b_cl"}, {47, "image_sample_c_lz"},
    {48, "image_sample_o"}, {49, "image_sample_cl_o"}, {50, "image_sample_d_o"}, {51, "image_sample_d_cl_o"},
    {52, "image_sample_l_o"}, {53, "image_sample_b_o"}, {54, "image_sample_b_cl_o"}, {55, "image_sample_lz_o"},
    {56, "image_sample_c_o"}, {57, "image_sample_c_cl_o"}, {58, "image_sample_c_d_o"},
    {59, "image_sample_c_d_cl_o"}, {60, "image_sample_c_l_o"}, {61, "image_sample_c_b_o"},
    {62, "image_sample_c_b_cl_o"}, {63, "image_sample_c_lz_o"}, {64, "image_gather4"}, {65, "image_gather4_cl"},
    {68, "image_gather4_l"}, {69, "image_gather4_b"}, {70, "image_gather4_b_cl"}, {71, "image_gather4_lz"},
    {72, "image_gather4_c"}, {73, "image_gather4_c_cl"}, {76, "image_gather4_c_l"}, {77, "image_gather4_c_b"},
    {78, "image_gather4_c_b_cl"}, {79, "image_gather4_c_lz"}, {80, "image_gather4_o"}, {81, "image_gather4_cl_o"},
    {84, "image_gather4_l_o"}, {85, "image_gather4_b_o"}, {86, "image_gather4_b_cl_o"}, {87, "image_gather4_lz_o"},
    {88, "image_gather4_c_o"}, {89, "image_gather4_c_cl_o"}, {92, "image_gather4_c_l_o"},
    {93, "image_gather4_c_b_o"}, {94, "image_gather4_c_b_cl_o"}, {95, "image_gather4_c_lz_o"},
    {96, "image_get_lod"}, {104, "image_sample_cd"}, {105, "image_sample_cd_cl"}, {106, "image_sample_c_cd"},
    {107, "image_sample_c_cd_cl"}, {108, "image_sample_cd_o"}, {109, "image_sample_cd_cl_o"},
    {110, "image_sample_c_cd_o"}, {111, "image_sample_c_cd_cl_o"},
};

template <std::size_t N>
const char* table(const char* const (&t)[N], std::uint32_t op) {
    return op < N ? t[op] : nullptr;
}
template <std::size_t N>
const char* named(const Named (&t)[N], std::uint32_t op) {
    for (const Named& n : t) {
        if (n.op == op) {
            return n.name;
        }
    }
    return nullptr;
}

// VOPC mnemonics are generated: op bits select cmp/cmpx, s (signalling), type.
const char* vopc_name(std::uint32_t op) {
    // Built once: pipeline workers decode concurrently.
    struct Names {
        char names[256][24];
    };
    static const Names* const table = [] {
        auto* t = new Names{};
        auto& names = t->names;
        for (std::uint32_t o = 0; o < 256; ++o) {
            names[o][0] = 0;
            const bool x = (o & 0x10) != 0;
            const std::uint32_t hi = o >> 5;  // 0 f32, 1 f64, 2 s_f32, 3 s_f64, 4 i32, 5 i64, 6 u32, 7 u64
            const std::uint32_t lo = o & 0xf;
            if (hi < 4) {
                std::snprintf(names[o], sizeof(names[o]), "v_cmp%s%s_%s_f%s", (hi & 2) ? "s" : "", x ? "x" : "",
                              kCmpF[lo], (hi & 1) ? "64" : "32");
            } else if (lo < 8) {
                std::snprintf(names[o], sizeof(names[o]), "v_cmp%s_%s_%c%s", x ? "x" : "", kCmpI[lo],
                              (hi & 2) ? 'u' : 'i', (hi & 1) ? "64" : "32");
            } else if (lo == 8 && hi < 6) {
                std::snprintf(names[o], sizeof(names[o]), "v_cmp%s_class_f%s", x ? "x" : "", (hi & 1) ? "64" : "32");
            }
        }
        return t;
    }();
    return table->names[op & 0xff][0] ? table->names[op & 0xff] : nullptr;
}

std::int32_t sext16(std::uint32_t v) { return static_cast<std::int16_t>(v & 0xffff); }

}  // namespace

bool is_vgpr(std::uint16_t code) { return code >= 256; }
bool is_sgpr(std::uint16_t code) { return code < 104; }

const char* enc_name(Enc e) {
    static const char* names[] = {"SOP2", "SOPK", "SOP1", "SOPC", "SOPP", "SMRD", "VOP2", "VOP1", "VOPC",
                                  "VOP3", "VINTRP", "DS", "MUBUF", "MTBUF", "MIMG", "EXP", "UNKNOWN"};
    return names[static_cast<int>(e)];
}

const char* mnemonic(const Inst& in) {
    switch (in.enc) {
    case Enc::SOP2:
        return table(kSop2, in.op);
    case Enc::SOPK:
        return table(kSopk, in.op);
    case Enc::SOP1:
        return table(kSop1, in.op);
    case Enc::SOPC:
        return table(kSopc, in.op);
    case Enc::SOPP:
        return table(kSopp, in.op);
    case Enc::SMRD:
        return table(kSmrd, in.op);
    case Enc::VOP2:
        return table(kVop2, in.op);
    case Enc::VOP1:
        return table(kVop1, in.op);
    case Enc::VOPC:
        return vopc_name(in.op);
    case Enc::VOP3:
        if (in.op < 0x100) {
            return vopc_name(in.op);
        }
        if (in.op < 0x140) {
            return table(kVop2, in.op - 0x100);
        }
        if (in.op < 0x180) {
            return table(kVop3Only, in.op - 0x140);
        }
        return table(kVop1, in.op - 0x180);
    case Enc::VINTRP:
        return table(kVintrp, in.op);
    case Enc::DS:
        return named(kDs, in.op);
    case Enc::MUBUF:
        return named(kMubuf, in.op);
    case Enc::MTBUF:
        return table(kMtbuf, in.op);
    case Enc::MIMG:
        return named(kMimg, in.op);
    case Enc::EXP:
        return "exp";
    default:
        return nullptr;
    }
}

namespace {

Enc classify(std::uint32_t w) {
    if ((w >> 31) == 0) {
        const std::uint32_t top7 = w >> 25;
        if (top7 == 0b0111110) {
            return Enc::VOPC;
        }
        if (top7 == 0b0111111) {
            return Enc::VOP1;
        }
        return Enc::VOP2;
    }
    if ((w >> 30) == 0b10) {
        const std::uint32_t top9 = w >> 23;
        if (top9 == 0b101111101) {
            return Enc::SOP1;
        }
        if (top9 == 0b101111110) {
            return Enc::SOPC;
        }
        if (top9 == 0b101111111) {
            return Enc::SOPP;
        }
        if ((w >> 28) == 0b1011) {
            return Enc::SOPK;
        }
        return Enc::SOP2;
    }
    if ((w >> 27) == 0b11000) {
        return Enc::SMRD;
    }
    switch (w >> 26) {
    case 0b110100:
        return Enc::VOP3;
    case 0b110010:
        return Enc::VINTRP;
    case 0b110110:
        return Enc::DS;
    case 0b111000:
        return Enc::MUBUF;
    case 0b111010:
        return Enc::MTBUF;
    case 0b111100:
        return Enc::MIMG;
    case 0b111110:
        return Enc::EXP;
    default:
        return Enc::Unknown;
    }
}

}  // namespace

bool decode_one(const std::uint32_t* words, std::size_t n, std::size_t i, Inst& out) {
    out = Inst{};
    const std::uint32_t w = words[i];
    out.words[0] = w;
    out.offset = static_cast<std::uint32_t>(i * 4);
    out.enc = classify(w);
    out.size = 1;
    bool need_literal = false;
    auto lit = [&](std::uint16_t code) {
        if (code == kLiteral) {
            need_literal = true;
        }
    };
    switch (out.enc) {
    case Enc::SOP2:
        out.op = (w >> 23) & 0x7f;
        out.dst = (w >> 16) & 0x7f;
        out.src1 = (w >> 8) & 0xff;
        out.src0 = w & 0xff;
        lit(out.src0);
        lit(out.src1);
        break;
    case Enc::SOPK:
        out.op = (w >> 23) & 0x1f;
        out.dst = (w >> 16) & 0x7f;
        out.imm = sext16(w);
        if (out.op == 21) {  // s_setreg_imm32_b32
            need_literal = true;
        }
        break;
    case Enc::SOP1:
        out.op = (w >> 8) & 0xff;
        out.dst = (w >> 16) & 0x7f;
        out.src0 = w & 0xff;
        lit(out.src0);
        break;
    case Enc::SOPC:
        out.op = (w >> 16) & 0x7f;
        out.src1 = (w >> 8) & 0xff;
        out.src0 = w & 0xff;
        lit(out.src0);
        lit(out.src1);
        break;
    case Enc::SOPP:
        out.op = (w >> 16) & 0x7f;
        out.imm = sext16(w);
        break;
    case Enc::SMRD:
        out.op = (w >> 22) & 0x1f;
        out.sdst = (w >> 15) & 0x7f;
        out.dst = out.sdst;
        out.src0 = static_cast<std::uint16_t>(((w >> 9) & 0x3f) * 2);  // SBASE is an SGPR pair index / 2
        out.imm_flag = (w >> 8) & 1;
        out.imm = w & 0xff;
        if (!out.imm_flag && out.imm == 0xff) {
            need_literal = true;  // CI: literal byte offset
        }
        break;
    case Enc::VOP2:
        out.op = (w >> 25) & 0x3f;
        out.dst = (w >> 17) & 0xff;
        out.src1 = static_cast<std::uint16_t>(((w >> 9) & 0xff) + 256);
        out.src0 = w & 0x1ff;
        lit(out.src0);
        if (out.op == 32 || out.op == 33) {  // v_madmk_f32 / v_madak_f32 carry K
            need_literal = true;
        }
        break;
    case Enc::VOP1:
        out.op = (w >> 9) & 0xff;
        out.dst = (w >> 17) & 0xff;
        out.src0 = w & 0x1ff;
        lit(out.src0);
        break;
    case Enc::VOPC:
        out.op = (w >> 17) & 0xff;
        out.src1 = static_cast<std::uint16_t>(((w >> 9) & 0xff) + 256);
        out.src0 = w & 0x1ff;
        lit(out.src0);
        break;
    case Enc::VOP3: {
        if (i + 1 >= n) {
            return false;
        }
        const std::uint32_t w1 = words[i + 1];
        out.words[1] = w1;
        out.size = 2;
        out.op = (w >> 17) & 0x1ff;
        out.dst = w & 0xff;
        // VOP3b (carry-out / div_scale) uses [14:8] as SDST instead of abs/clamp.
        const bool vop3b = (out.op >= 0x100 + 37 && out.op <= 0x100 + 42) || out.op == 0x16d || out.op == 0x16e ||
                           (out.op < 0x100);
        if (vop3b && out.op >= 0x100) {
            out.sdst = (w >> 8) & 0x7f;
        } else if (out.op < 0x100) {
            out.sdst = static_cast<std::uint16_t>(w & 0xff);  // VOPC in VOP3: the SGPR pair sits in the VDST field [7:0]
            out.abs = (w >> 8) & 7;
            out.clamp = (w >> 11) & 1;
        } else {
            out.abs = (w >> 8) & 7;
            out.clamp = (w >> 11) & 1;
        }
        out.src0 = w1 & 0x1ff;
        out.src1 = (w1 >> 9) & 0x1ff;
        out.src2 = (w1 >> 18) & 0x1ff;
        out.omod = (w1 >> 27) & 3;
        out.neg = (w1 >> 29) & 7;
        break;
    }
    case Enc::VINTRP:
        out.op = (w >> 16) & 3;
        out.dst = (w >> 18) & 0xff;
        out.attr = (w >> 10) & 0x3f;
        out.attr_chan = (w >> 8) & 3;
        out.src0 = static_cast<std::uint16_t>((w & 0xff) + 256);
        break;
    case Enc::DS: {
        if (i + 1 >= n) {
            return false;
        }
        const std::uint32_t w1 = words[i + 1];
        out.words[1] = w1;
        out.size = 2;
        out.op = (w >> 18) & 0xff;
        out.gds = (w >> 17) & 1;
        out.offset0 = w & 0xff;
        out.offset1 = (w >> 8) & 0xff;
        out.vaddr = w1 & 0xff;
        out.src0 = static_cast<std::uint16_t>(((w1 >> 8) & 0xff) + 256);   // data0
        out.src1 = static_cast<std::uint16_t>(((w1 >> 16) & 0xff) + 256);  // data1
        out.dst = (w1 >> 24) & 0xff;
        break;
    }
    case Enc::MUBUF:
    case Enc::MTBUF: {
        if (i + 1 >= n) {
            return false;
        }
        const std::uint32_t w1 = words[i + 1];
        out.words[1] = w1;
        out.size = 2;
        out.offset12 = w & 0xfff;
        out.offen = (w >> 12) & 1;
        out.idxen = (w >> 13) & 1;
        out.glc = (w >> 14) & 1;
        out.addr64 = (w >> 15) & 1;
        if (out.enc == Enc::MUBUF) {
            out.lds = (w >> 16) & 1;
            out.op = (w >> 18) & 0x7f;
        } else {
            out.op = (w >> 16) & 7;
            out.dfmt = (w >> 19) & 0xf;
            out.nfmt = (w >> 23) & 7;
        }
        out.vaddr = w1 & 0xff;
        out.vdata = (w1 >> 8) & 0xff;
        out.srsrc = static_cast<std::uint16_t>(((w1 >> 16) & 0x1f) * 4);
        out.slc = (w1 >> 22) & 1;
        out.tfe = (w1 >> 23) & 1;
        out.soffset = (w1 >> 24) & 0xff;
        break;
    }
    case Enc::MIMG: {
        if (i + 1 >= n) {
            return false;
        }
        const std::uint32_t w1 = words[i + 1];
        out.words[1] = w1;
        out.size = 2;
        out.dmask = (w >> 8) & 0xf;
        out.unorm = (w >> 12) & 1;
        out.glc = (w >> 13) & 1;
        out.da = (w >> 14) & 1;
        out.r128 = (w >> 15) & 1;
        out.tfe = (w >> 16) & 1;
        out.lwe = (w >> 17) & 1;
        out.op = (w >> 18) & 0x7f;
        out.slc = (w >> 25) & 1;
        out.vaddr = w1 & 0xff;
        out.vdata = (w1 >> 8) & 0xff;
        out.srsrc = static_cast<std::uint16_t>(((w1 >> 16) & 0x1f) * 4);
        out.ssamp = static_cast<std::uint16_t>(((w1 >> 21) & 0x1f) * 4);
        break;
    }
    case Enc::EXP: {
        if (i + 1 >= n) {
            return false;
        }
        const std::uint32_t w1 = words[i + 1];
        out.words[1] = w1;
        out.size = 2;
        out.dmask = w & 0xf;
        out.tgt = (w >> 4) & 0x3f;
        out.compr = (w >> 10) & 1;
        out.done = (w >> 11) & 1;
        out.vm = (w >> 12) & 1;
        for (int k = 0; k < 4; ++k) {
            out.vsrc[k] = static_cast<std::uint8_t>((w1 >> (8 * k)) & 0xff);
        }
        break;
    }
    default:
        return false;
    }
    if (need_literal) {
        if (i + out.size >= n) {
            return false;
        }
        out.literal = words[i + out.size];
        out.words[out.size] = out.literal;
        out.has_literal = true;
        out.size++;
    }
    return mnemonic(out) != nullptr;
}

Program decode(const std::uint32_t* words, std::size_t n) {
    Program p;
    std::size_t i = 0;
    while (i < n) {
        Inst in;
        const bool ok = decode_one(words, n, i, in);
        if (!ok) {
            char msg[96];
            std::snprintf(msg, sizeof(msg), "%s op=0x%x word=0x%08x", enc_name(in.enc), in.op, words[i]);
            p.errors.push_back({static_cast<std::uint32_t>(i * 4), msg});
        }
        p.insts.push_back(in);
        i += in.size ? in.size : 1;
    }
    // Trim trailing padding (zero dwords / s_nop 0 / s_code_end style fill).
    while (!p.insts.empty()) {
        const Inst& last = p.insts.back();
        const bool pad = last.words[0] == 0 || (last.enc == Enc::SOPP && last.op == 0 && last.imm == 0) ||
                         last.words[0] == 0xbf9f0000;
        if (!pad) {
            break;
        }
        p.insts.pop_back();
    }
    p.end = p.insts.empty() ? 0 : p.insts.back().offset + p.insts.back().size * 4;
    return p;
}

std::string operand_name(std::uint16_t code, std::uint32_t literal, int dwords) {
    char buf[48];
    auto range = [&](const char* prefix, int base) {
        if (dwords > 1) {
            std::snprintf(buf, sizeof(buf), "%s[%d:%d]", prefix, base, base + dwords - 1);
        } else {
            std::snprintf(buf, sizeof(buf), "%s%d", prefix, base);
        }
        return std::string(buf);
    };
    if (code >= 256) {
        return range("v", code - 256);
    }
    if (code < 104) {
        return range("s", code);
    }
    if (code >= 112 && code < 124) {
        return range("ttmp", code - 112);
    }
    if (code >= 129 && code <= 192) {
        return std::to_string(code - 128);
    }
    if (code >= 193 && code <= 208) {
        return "-" + std::to_string(code - 192);
    }
    switch (code) {
    case 104:
        return "flat_scratch_lo";
    case 105:
        return "flat_scratch_hi";
    case 106:
        return dwords > 1 ? "vcc" : "vcc_lo";
    case 107:
        return "vcc_hi";
    case 108:
        return "tba_lo";
    case 109:
        return "tba_hi";
    case 110:
        return "tma_lo";
    case 111:
        return "tma_hi";
    case 124:
        return "m0";
    case 126:
        return dwords > 1 ? "exec" : "exec_lo";
    case 127:
        return "exec_hi";
    case 128:
        return "0";
    case 240:
        return "0.5";
    case 241:
        return "-0.5";
    case 242:
        return "1.0";
    case 243:
        return "-1.0";
    case 244:
        return "2.0";
    case 245:
        return "-2.0";
    case 246:
        return "4.0";
    case 247:
        return "-4.0";
    case 251:
        return "vccz";
    case 252:
        return "execz";
    case 253:
        return "scc";
    case 254:
        return "lds_direct";
    case 255:
        std::snprintf(buf, sizeof(buf), "0x%x", literal);
        return buf;
    default:
        std::snprintf(buf, sizeof(buf), "?%u", code);
        return buf;
    }
}

namespace {

// Operand width in dwords implied by a mnemonic suffix (b64/i64/u64/f64,
// dwordxN, x2). Good enough for disassembly text.
int width_of(const char* name, bool dst) {
    (void)dst;
    const std::size_t len = std::strlen(name);
    if (len >= 3 && std::strcmp(name + len - 3, "x16") == 0) return 16;
    if (len >= 2 && std::strcmp(name + len - 2, "x8") == 0) return 8;
    if (len >= 2 && std::strcmp(name + len - 2, "x4") == 0) return 4;
    if (len >= 2 && std::strcmp(name + len - 2, "x3") == 0) return 3;
    if (len >= 2 && std::strcmp(name + len - 2, "x2") == 0) return 2;
    if (len >= 2 && std::strcmp(name + len - 2, "64") == 0) return 2;
    return 1;
}

std::string vop3_src(const Inst& in, int k, std::uint16_t code, int dwords) {
    std::string s = operand_name(code, in.literal, dwords);
    if (in.abs & (1 << k)) {
        s = "|" + s + "|";
    }
    if (in.neg & (1 << k)) {
        s = "-" + s;
    }
    return s;
}

}  // namespace

std::string format(const Inst& in) {
    const char* name = mnemonic(in);
    char buf[160];
    if (!name) {
        std::snprintf(buf, sizeof(buf), "<unknown %s op=0x%x 0x%08x>", enc_name(in.enc), in.op, in.words[0]);
        return buf;
    }
    std::string s = name;
    const int w = width_of(name, true);
    auto sreg = [&](std::uint16_t code, int dw) { return operand_name(code, in.literal, dw); };
    auto vreg = [&](std::uint16_t idx, int dw) { return operand_name(static_cast<std::uint16_t>(idx + 256), 0, dw); };
    switch (in.enc) {
    case Enc::SOP2:
        s += " " + sreg(in.dst, w) + ", " + sreg(in.src0, w) + ", " + sreg(in.src1, w);
        break;
    case Enc::SOPK:
        std::snprintf(buf, sizeof(buf), " %s, 0x%x", sreg(in.dst, 1).c_str(), in.imm & 0xffff);
        s += buf;
        break;
    case Enc::SOP1:
        s += " " + sreg(in.dst, w) + ", " + sreg(in.src0, w);
        break;
    case Enc::SOPC:
        s += " " + sreg(in.src0, w) + ", " + sreg(in.src1, w);
        break;
    case Enc::SOPP:
        if (in.op == 1 || in.op == 10 || in.op == 19) {
            break;
        }
        if (in.op == 2 || (in.op >= 4 && in.op <= 9) || (in.op >= 23 && in.op <= 26)) {
            std::snprintf(buf, sizeof(buf), " %+d (-> 0x%x)", in.imm, in.offset + 4 + in.imm * 4);
        } else if (in.op == 12) {
            const unsigned vm = in.imm & 0xf, exp = (in.imm >> 4) & 7, lgk = (in.imm >> 8) & 0x1f;
            std::string parts;
            if (vm != 0xf) parts += " vmcnt(" + std::to_string(vm) + ")";
            if (exp != 7) parts += " expcnt(" + std::to_string(exp) + ")";
            if (lgk != 0x1f) parts += " lgkmcnt(" + std::to_string(lgk) + ")";
            std::snprintf(buf, sizeof(buf), "%s", parts.empty() ? " 0" : parts.c_str());
        } else {
            std::snprintf(buf, sizeof(buf), " 0x%x", in.imm & 0xffff);
        }
        s += buf;
        break;
    case Enc::SMRD: {
        const int dw = width_of(name, true);
        s += " " + sreg(in.sdst, dw) + ", " + sreg(in.src0, in.op >= 8 ? 4 : 2) + ", ";
        if (in.imm_flag) {
            std::snprintf(buf, sizeof(buf), "0x%x", in.imm);
        } else if (in.has_literal) {
            std::snprintf(buf, sizeof(buf), "0x%x", in.literal);
        } else {
            std::snprintf(buf, sizeof(buf), "%s", sreg(static_cast<std::uint16_t>(in.imm), 1).c_str());
        }
        s += buf;
        break;
    }
    case Enc::VOP2:
        if (in.op == 1 || in.op == 2) {
            // v_readlane_b32 sD, vN, lane / v_writelane_b32 vN, s, lane: the
            // lane is a scalar operand in the VGPR field, and a read's
            // destination an SGPR.
            s += " " + (in.op == 1 ? sreg(in.dst, 1) : vreg(in.dst, 1)) + ", " + sreg(in.src0, 1) + ", " +
                 sreg(static_cast<std::uint16_t>(in.src1 - 256), 1);
            break;
        }
        s += " " + vreg(in.dst, 1) + ", " + sreg(in.src0, 1) + ", " + sreg(in.src1, 1);
        if (in.op == 32) {
            std::snprintf(buf, sizeof(buf), ", 0x%x", in.literal);
            s += buf;
        } else if (in.op == 33) {
            std::snprintf(buf, sizeof(buf), ", 0x%x", in.literal);
            s += buf;
        }
        break;
    case Enc::VOP1:
        s += " " + vreg(in.dst, w) + ", " + sreg(in.src0, w);
        break;
    case Enc::VOPC:
        s += " vcc, " + sreg(in.src0, w) + ", " + sreg(in.src1, w);
        break;
    case Enc::VOP3: {
        // The 64-bit shifts, f64 arithmetic and the multiplies at 0x161-0x16c
        // take two sources.
        const int nsrc = in.op < 0x100 ? 2 : in.op < 0x140 ? 2 : in.op < 0x180 ? (in.op >= 0x161 && in.op <= 0x16c ? 2 : 3) : 1;
        if (in.op < 0x100) {
            s += " " + sreg(in.sdst, 2);
        } else if (in.op == 0x101) {
            s += " " + sreg(in.dst, 1);  // v_readlane_b32: an SGPR
        } else {
            s += " " + vreg(in.dst, w);
            if (in.sdst || (in.op >= 0x125 && in.op <= 0x12a) || in.op == 0x16d || in.op == 0x16e) {
                s += ", " + sreg(in.sdst, 2);
            }
        }
        const std::uint16_t srcs[3] = {in.src0, in.src1, in.src2};
        int count = nsrc;
        // VOP2 ops that in VOP3 form take 3 sources.
        if (in.op == 0x100 || in.op == 0x11f || (in.op >= 0x128 && in.op <= 0x12a)) {
            count = 3;
        }
        for (int k = 0; k < count; ++k) {
            // A 64-bit shift's amount, and a lane operand, are one dword.
            const bool one = (k == 1 && in.op >= 0x161 && in.op <= 0x163) || in.op == 0x101 || in.op == 0x102;
            s += ", " + vop3_src(in, k, srcs[k], one ? 1 : w);
        }
        if (in.clamp) s += " clamp";
        if (in.omod == 1) s += " mul:2";
        if (in.omod == 2) s += " mul:4";
        if (in.omod == 3) s += " div:2";
        break;
    }
    case Enc::VINTRP:
        std::snprintf(buf, sizeof(buf), " %s, %s, attr%d.%c", vreg(in.dst, 1).c_str(), sreg(in.src0, 1).c_str(),
                      in.attr, "xyzw"[in.attr_chan]);
        s += buf;
        break;
    case Enc::DS: {
        const bool rtn = std::strstr(name, "_rtn") || std::strstr(name, "read") || in.op == 53 || in.op == 61 ||
                         in.op == 62 || in.op == 63;
        const bool two = std::strstr(name, "write2") || std::strstr(name, "read2") || std::strstr(name, "wrxchg2");
        int dw = width_of(name, true);
        if (two) dw *= 2;
        if (rtn) {
            s += " " + vreg(in.dst, dw) + ",";
        }
        s += " " + vreg(in.vaddr, 1);
        const bool has_data0 = !std::strstr(name, "read") && in.op != 53 && in.op != 61 && in.op != 62 && in.op != 63;
        if (has_data0) {
            s += ", " + operand_name(in.src0, 0, width_of(name, false));
            if (two || std::strstr(name, "cmpst") || std::strstr(name, "mskor")) {
                s += ", " + operand_name(in.src1, 0, width_of(name, false));
            }
        }
        if (two) {
            std::snprintf(buf, sizeof(buf), " offset0:%u offset1:%u", in.offset0, in.offset1);
        } else {
            std::snprintf(buf, sizeof(buf), " offset:%u", in.offset0 | (in.offset1 << 8));
        }
        s += buf;
        if (in.gds) s += " gds";
        break;
    }
    case Enc::MUBUF:
    case Enc::MTBUF: {
        int dw = width_of(name, true);
        if (std::strstr(name, "format_xyzw")) dw = 4;
        else if (std::strstr(name, "format_xyz")) dw = 3;
        else if (std::strstr(name, "format_xy")) dw = 2;
        s += " " + vreg(in.vdata, dw) + ", ";
        if (in.offen || in.idxen || in.addr64) {
            s += vreg(in.vaddr, (in.offen && in.idxen) || in.addr64 ? 2 : 1) + ", ";
        } else {
            s += "off, ";
        }
        s += sreg(in.srsrc, 4) + ", " + sreg(in.soffset, 1);
        if (in.idxen) s += " idxen";
        if (in.offen) s += " offen";
        if (in.addr64) s += " addr64";
        if (in.offset12) {
            std::snprintf(buf, sizeof(buf), " offset:%u", in.offset12);
            s += buf;
        }
        if (in.enc == Enc::MTBUF) {
            std::snprintf(buf, sizeof(buf), " dfmt:%u nfmt:%u", in.dfmt, in.nfmt);
            s += buf;
        }
        if (in.glc) s += " glc";
        if (in.slc) s += " slc";
        if (in.tfe) s += " tfe";
        if (in.lds) s += " lds";
        break;
    }
    case Enc::MIMG: {
        int dw = 0;
        for (int k = 0; k < 4; ++k) dw += (in.dmask >> k) & 1;
        if (dw == 0) dw = 1;
        s += " " + vreg(in.vdata, dw) + ", " + vreg(in.vaddr, 1) + ", " + sreg(in.srsrc, in.r128 ? 4 : 8);
        if (in.op >= 32) {
            s += ", " + sreg(in.ssamp, 4);
        }
        std::snprintf(buf, sizeof(buf), " dmask:0x%x", in.dmask);
        s += buf;
        if (in.unorm) s += " unorm";
        if (in.glc) s += " glc";
        if (in.da) s += " da";
        if (in.r128) s += " r128";
        if (in.tfe) s += " tfe";
        if (in.lwe) s += " lwe";
        if (in.slc) s += " slc";
        break;
    }
    case Enc::EXP: {
        const char* tgt = in.tgt < 10 ? kExpTgt[in.tgt] : nullptr;
        if (tgt) {
            std::snprintf(buf, sizeof(buf), " %s", tgt);
        } else if (in.tgt >= 12 && in.tgt < 16) {
            std::snprintf(buf, sizeof(buf), " pos%u", in.tgt - 12);
        } else if (in.tgt >= 32 && in.tgt < 64) {
            std::snprintf(buf, sizeof(buf), " param%u", in.tgt - 32);
        } else {
            std::snprintf(buf, sizeof(buf), " tgt%u", in.tgt);
        }
        s += buf;
        for (int k = 0; k < 4; ++k) {
            s += ", " + ((in.dmask >> k) & 1 ? vreg(in.vsrc[k], 1) : std::string("off"));
        }
        if (in.done) s += " done";
        if (in.compr) s += " compr";
        if (in.vm) s += " vm";
        break;
    }
    default:
        break;
    }
    return s;
}

}  // namespace gcn
