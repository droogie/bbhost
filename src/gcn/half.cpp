#include "gcn/half.h"

#include "gcn/isa.h"

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace gcn {

spv::Id emit_unpack_half16(spv::Module& m, const SpvScalarTypes& t, spv::Id h) {
    using spv::Id;
    const auto u = [&](std::uint32_t v) { return m.const_u32(v); };
    const auto ubin = [&](spv::Op op, Id a, Id b) { return m.emit(op, t.t_u32, {a, b}); };
    const auto is = [&](Id a, std::uint32_t v) { return m.emit(spv::OpIEqual, t.t_bool, {a, u(v)}); };
    const Id e = ubin(spv::OpBitwiseAnd, ubin(spv::OpShiftRightLogical, h, u(10)), u(0x1f));
    const Id man = ubin(spv::OpBitwiseAnd, h, u(0x3ff));
    const Id subnormal = is(e, 0);
    // (1024 + m) * 2^(e - 25), or m * 2^-24 for a subnormal half
    const Id significand = m.emit(spv::OpSelect, t.t_u32, {subnormal, man, ubin(spv::OpBitwiseOr, man, u(0x400))});
    const Id biased = m.emit(spv::OpISub, t.t_i32, {m.emit(spv::OpBitcast, t.t_i32, {e}), m.const_i32(25)});
    const Id scale = m.emit(spv::OpSelect, t.t_i32, {subnormal, m.const_i32(-24), biased});
    Id mag = m.ext_inst(t.t_f32, spv::GlslLdexp, {m.emit(spv::OpConvertUToF, t.t_f32, {significand}), scale});
    const Id special = m.emit(spv::OpSelect, t.t_f32,
                              {is(man, 0), m.const_f32(std::bit_cast<float>(0x7f800000u)), m.const_f32(std::bit_cast<float>(0x7fc00000u))});
    mag = m.emit(spv::OpSelect, t.t_f32, {is(e, 31), special, mag});
    const Id negative = is(ubin(spv::OpShiftRightLogical, h, u(15)), 1);
    return m.emit(spv::OpSelect, t.t_f32, {negative, m.emit(spv::OpFNegate, t.t_f32, {mag}), mag});
}

bool export_rtz_on() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_EXPORT_RTZ");
        return !(e && e[0] == '0');
    }();
    return on;
}

namespace {
std::atomic<bool> g_native_half_rtz{false};
}
void set_native_half_rtz(bool on) { g_native_half_rtz.store(on, std::memory_order_relaxed); }
bool native_half_rtz() { return g_native_half_rtz.load(std::memory_order_relaxed) && export_rtz_on(); }

bool program_allows_native_half_rtz(const Program& prog) {
    for (const Inst& in : prog.insts) {
        const char* n = mnemonic(in);
        if (!n) continue;
        if (!std::strcmp(n, "v_cvt_f16_f32") || !std::strncmp(n, "tbuffer_store_format", 20) || !std::strncmp(n, "buffer_store_format", 19) ||
            !std::strncmp(n, "image_store", 11)) {
            return false;
        }
    }
    return true;
}

// BBHOST_LEGACY_MUL=plain (a measurement, not a mode to play with): an
// ordinary multiply, which gives NaN for 0 x inf where GCN gives 0 - what the
// zero rule's two or three extra instructions cost, as an upper bound.
bool legacy_mul_plain() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_LEGACY_MUL");
        return e && std::strcmp(e, "plain") == 0;
    }();
    return on;
}

namespace {
std::atomic<bool> g_legacy_mul_select{false};
}
void set_legacy_mul_select(bool on) { g_legacy_mul_select.store(on, std::memory_order_relaxed); }
bool legacy_mul_min_form() {
    // BBHOST_LEGACY_MUL=min or =select decides; otherwise the device's form.
    static const int forced = [] {
        const char* e = std::getenv("BBHOST_LEGACY_MUL");
        return e && std::strcmp(e, "select") == 0 ? 0 : e && std::strcmp(e, "min") == 0 ? 1 : -1;
    }();
    return forced >= 0 ? forced == 1 : !g_legacy_mul_select.load(std::memory_order_relaxed);
}

spv::Id emit_rtz_half(spv::Module& m, const SpvScalarTypes& t, spv::Id x) {
    using spv::Id;
    const auto u = [&](std::uint32_t v) { return m.const_u32(v); };
    const Id bits = m.emit(spv::OpBitcast, t.t_u32, {x});
    const Id mag = m.emit(spv::OpBitwiseAnd, t.t_u32, {bits, u(0x7fffffffu)});
    const Id special = m.emit(spv::OpUGreaterThanEqual, t.t_bool, {mag, u(0x7f800000u)});  // an infinity or a NaN: as it is
    const Id finite = m.emit(spv::OpBitwiseAnd, t.t_u32, {m.ext_inst(t.t_u32, spv::GlslUMin, {mag, u(0x477fe000u)}), u(0xffffe000u)});
    const Id out = m.emit(spv::OpBitwiseOr, t.t_u32,
                          {m.emit(spv::OpBitwiseAnd, t.t_u32, {bits, u(0x80000000u)}), m.emit(spv::OpSelect, t.t_u32, {special, mag, finite})});
    return m.emit(spv::OpBitcast, t.t_f32, {out});
}

}  // namespace gcn
