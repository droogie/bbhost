// GX's render-state objects as host objects; see the header.
#include "engine/gx_state.h"

#include "core/elf.h"
#include "engine/addr.h"
#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

enum Kind : std::uint32_t { kBlend = 1, kDepthStencil = 2, kRaster = 3 };
// GX's own states (D3D11's defaults), which a null pointer in the pending
// state stands for; GX builds them in place at start-up (0x2ad3280,
// 0x2ad3a00) like any other state object.
constexpr std::uint64_t kDefaults[4] = {0, 0x586c8f4, 0x586ca1c, 0x586ca58};
std::uint64_t g_default[4] = {};  // the same, slid
// The object bytes an id is checked against at every lookup: the registers
// the builder encoded (and for blend its alpha-to-coverage and independent
// bytes, the description's first two). A state rebuilt in place or a new
// object at a reused address changes them.
constexpr std::size_t kObjectBytes[4] = {0, 0x128, 0x38, 0x24};
constexpr std::size_t kFingerprintBytes[4] = {0, 0x26, 0x10, 0x24};

const int g_mode = [] {
    const char* e = std::getenv("BBHOST_GX_STATE");
    return e ? std::atoi(e) : 1;
}();

// Ids: the kind in the top two bits, an index into the kind's descriptions
// below it. Descriptions are never freed, so an id's pointer stays good.
constexpr std::uint32_t kIndexBits = 30, kIndexMask = (1u << kIndexBits) - 1;
constexpr std::size_t kMaxDescs = 1 << 14;
std::atomic<const void*> g_descs[4][kMaxDescs];
std::mutex g_intern_mu;
std::unordered_map<std::string, std::uint32_t> g_interned[4];  // description bytes -> id, under g_intern_mu
std::atomic<std::uint64_t> g_parsed[4] = {}, g_distinct[4] = {}, g_refused[4] = {}, g_table_full{0};
std::atomic<std::uint64_t> g_biased{0};  // rasterizer descriptions with a depth bias

// Object address -> {fingerprint, id}, open-addressed and never shrunk,
// read and written by the GX recording threads without a lock: a key is
// claimed with a CAS, its value is one 64-bit store. A reader that sees the
// key before the value reads a fingerprint of 0 and parses the object itself.
constexpr std::size_t kSlots = 1 << 16;
struct Slot {
    std::atomic<std::uint64_t> key{0};
    std::atomic<std::uint64_t> value{0};  // fingerprint << 32 | id
};
Slot g_slots[kSlots];

std::size_t slot_of(std::uint64_t object) {
    return static_cast<std::size_t>((object >> 2) * 0x9e3779b97f4a7c15ull >> 48) & (kSlots - 1);
}

Slot* find_slot(std::uint64_t object, bool claim) {
    std::size_t i = slot_of(object);
    for (std::size_t n = 0; n < kSlots; ++n, i = (i + 1) & (kSlots - 1)) {
        std::uint64_t k = g_slots[i].key.load(std::memory_order_acquire);
        if (k == object) return &g_slots[i];
        if (k == 0) {
            if (!claim) return nullptr;
            if (g_slots[i].key.compare_exchange_strong(k, object, std::memory_order_acq_rel) || k == object) return &g_slots[i];
        }
    }
    if (claim) g_table_full.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

// Eight bytes at a time: it runs three times a draw on the GX threads, which
// the game's job barrier waits for.
std::uint32_t fingerprint(const std::uint8_t* p, std::size_t n) {
    std::uint64_t h = 1469598103934665603ull;
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        std::uint64_t v;
        std::memcpy(&v, p + i, 8);
        h = (h ^ v) * 1099511628211ull;
        h ^= h >> 29;
    }
    for (; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return static_cast<std::uint32_t>(h ^ (h >> 32)) | 1;  // never 0, which marks a value not yet stored
}

// The last object of each kind this thread looked up, with its fingerprint
// and id: draws come in runs that share their states, and the slot table is
// a megabyte - a probe of it is a miss. The fingerprint is still taken, since
// a state can be rebuilt in place.
struct StateMemo {
    std::uint64_t object = 0;
    std::uint64_t value = 0;  // fingerprint << 32 | id
};
thread_local StateMemo t_state_memo[4];

std::uint32_t u32_at(const std::uint8_t* p, std::size_t at) {
    std::uint32_t v;
    std::memcpy(&v, p + at, 4);
    return v;
}

// The descriptions, from the object's own copy of them (each builder keeps
// the description beside the registers it encodes) and parsed into explicit
// fields, so padding never splits one state into two ids. Values outside
// GX's enums come back false: that object stays unknown.
//
//   blend (0x2578050): the description itself from +0x24 - +0 alpha to
//     coverage and +1 independent blend (bytes), then eight 0x20-byte
//     targets from +4: enable (byte), source, destination, op, alpha source,
//     alpha destination, alpha op, write mask
//   depth-stencil (0x25782b0): bytes +9 read mask, +0xa write mask, +0xc
//     depth enable, +0xd stencil enable, +0xe depth bounds; +0x10 depth
//     write mask (0x2ab7bf0: ZERO 0, ALL 1), +0x14 depth func, the front
//     face from +0x18 and the back from +0x28 (fail, depth fail, pass, func)
//   rasterizer (0x2578550): +8 depth bias (the description's integer as a
//     float), +0xc clamp, +0x10 slope, bytes +0x14 multisample, +0x15
//     scissor; +0x18 fill (0x2ab7c40: 0 solid), +0x1c cull, bytes +0x20
//     front counter-clockwise, +0x21 depth clip
bool parse_blend(const std::uint8_t* o, GxBlendDesc& d) {
    const std::uint8_t* raw = o + 0x24;
    d.alpha_to_coverage = raw[0] != 0;
    d.independent = raw[1] != 0;
    for (int i = 0; i < 8; ++i) {
        // Without independent blend GX builds target 0 and copies it to all.
        const std::uint8_t* t = raw + 4 + 0x20 * (d.independent ? i : 0);
        GxBlendTarget& b = d.rt[i];
        b.enable = t[0] != 0;
        const std::uint32_t f[6] = {u32_at(t, 4), u32_at(t, 8), u32_at(t, 0xc), u32_at(t, 0x10), u32_at(t, 0x14), u32_at(t, 0x18)};
        if (f[0] > 16 || f[1] > 16 || f[2] > 4 || f[3] > 16 || f[4] > 16 || f[5] > 4) return false;
        b.src = static_cast<std::uint8_t>(f[0]);
        b.dst = static_cast<std::uint8_t>(f[1]);
        b.op = static_cast<std::uint8_t>(f[2]);
        b.src_alpha = static_cast<std::uint8_t>(f[3]);
        b.dst_alpha = static_cast<std::uint8_t>(f[4]);
        b.op_alpha = static_cast<std::uint8_t>(f[5]);
        b.mask = t[0x1c] & 0xf;
    }
    return true;
}

bool parse_face(const std::uint8_t* f, GxStencilFace& out) {
    const std::uint32_t v[4] = {u32_at(f, 0), u32_at(f, 4), u32_at(f, 8), u32_at(f, 0xc)};
    if (v[0] > 7 || v[1] > 7 || v[2] > 7 || v[3] > 7) return false;
    out.fail = static_cast<std::uint8_t>(v[0]);
    out.depth_fail = static_cast<std::uint8_t>(v[1]);
    out.pass = static_cast<std::uint8_t>(v[2]);
    out.func = static_cast<std::uint8_t>(v[3]);
    return true;
}

bool parse_depth_stencil(const std::uint8_t* o, GxDepthStencilDesc& d) {
    const std::uint32_t func = u32_at(o, 0x14);
    if (func > 7) return false;
    d.read_mask = o[9];
    d.write_mask = o[0xa];
    d.depth_enable = o[0xc] != 0;
    d.stencil_enable = o[0xd] != 0;
    d.bounds = o[0xe] != 0;
    d.depth_write = u32_at(o, 0x10) == 1;
    d.depth_func = static_cast<std::uint8_t>(func);
    return parse_face(o + 0x18, d.front) && parse_face(o + 0x28, d.back);
}

bool parse_raster(const std::uint8_t* o, GxRasterDesc& d) {
    const std::uint32_t cull = u32_at(o, 0x1c);
    if (cull > 2) return false;
    float bias;
    std::memcpy(&bias, o + 8, 4);
    d.depth_bias = static_cast<std::int32_t>(bias);
    std::memcpy(&d.bias_clamp, o + 0xc, 4);
    std::memcpy(&d.slope_bias, o + 0x10, 4);
    d.multisample = o[0x14] != 0;
    d.scissor = o[0x15] != 0;
    d.fill = u32_at(o, 0x18) != 0;
    d.cull = static_cast<std::uint8_t>(cull);
    d.front_ccw = o[0x20] != 0;
    d.depth_clip = o[0x21] != 0;
    return true;
}

template <typename D>
std::uint32_t intern(Kind kind, const D& d, bool* made) {
    const std::string key(reinterpret_cast<const char*>(&d), sizeof(d));
    std::lock_guard<std::mutex> lk(g_intern_mu);
    auto [it, fresh] = g_interned[kind].emplace(key, 0u);
    *made = fresh;
    if (fresh) {
        const std::uint64_t index = g_distinct[kind].load(std::memory_order_relaxed) + 1;
        if (index >= kMaxDescs) {
            g_interned[kind].erase(it);
            return 0;
        }
        g_descs[kind][index].store(new D(d), std::memory_order_release);
        g_distinct[kind].store(index, std::memory_order_relaxed);
        it->second = (static_cast<std::uint32_t>(kind) << kIndexBits) | static_cast<std::uint32_t>(index);
    }
    return it->second;
}

std::uint32_t parse_object(Kind kind, const std::uint8_t* o) {
    bool made = false;
    std::uint32_t id = 0;
    if (kind == kBlend) {
        GxBlendDesc d;
        if (parse_blend(o, d)) id = intern(kind, d, &made);
    } else if (kind == kDepthStencil) {
        GxDepthStencilDesc d;
        if (parse_depth_stencil(o, d)) id = intern(kind, d, &made);
    } else {
        GxRasterDesc d;
        if (parse_raster(o, d)) {
            id = intern(kind, d, &made);
            if (made && (d.depth_bias || d.slope_bias != 0.0f)) g_biased.fetch_add(1, std::memory_order_relaxed);
        }
    }
    g_parsed[kind].fetch_add(1, std::memory_order_relaxed);
    if (!id) g_refused[kind].fetch_add(1, std::memory_order_relaxed);
    return id;
}

// At the draw call: the object is alive and bound, so it is read directly,
// as the draw's own inputs are (gx_trace.cpp).
std::uint32_t lookup(Kind kind, std::uint64_t object) {
    if (!g_mode) return 0;
    if (!object) object = g_default[kind];
    if (!object) return 0;
    const auto* o = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(object));
    const std::uint64_t fp = fingerprint(o, kFingerprintBytes[kind]);
    StateMemo& memo = t_state_memo[kind];
    if (memo.object == object && memo.value >> 32 == fp) return static_cast<std::uint32_t>(memo.value);
    if (Slot* s = find_slot(object, false)) {
        const std::uint64_t v = s->value.load(std::memory_order_acquire);
        if (v >> 32 == fp) {
            memo = {object, v};
            return static_cast<std::uint32_t>(v);
        }
    }
    std::uint8_t copy[0x128];
    std::memcpy(copy, o, kObjectBytes[kind]);
    const std::uint32_t id = parse_object(kind, copy);
    if (Slot* s = find_slot(object, true)) s->value.store(fp << 32 | id, std::memory_order_release);
    if (id) memo = {object, fp << 32 | id};
    return id;
}

const void* desc_of(Kind kind, std::uint32_t id) {
    if (id >> kIndexBits != kind) return nullptr;
    const std::uint32_t index = id & kIndexMask;
    return index < kMaxDescs ? g_descs[kind][index].load(std::memory_order_acquire) : nullptr;
}

}  // namespace

void gx_state_install(ElfImage* image) {
    if (!g_mode) {
        host_log("gx-state: off (BBHOST_GX_STATE=0)");
        return;
    }
    for (int k = kBlend; k <= kRaster; ++k) g_default[k] = image->mem.slide + (kDefaults[k] - kPreferredGuestSlide);
    host_log("gx-state: on%s", g_mode == 2 ? "; compare mode (render.cpp checks every draw)" : "");
}

int gx_state_mode() { return g_mode; }

namespace {
// A plugin's own draws restyled (gx_draw_style): each state's copy, made once
// and remembered per thread for the last id and style asked.
thread_local int t_style = 0;
struct Restyled {
    std::uint32_t from = 0, to = 0;
    int style = 0;
};
thread_local Restyled t_restyled[4];

// BBHOST_WIREFRAME_BIAS / BBHOST_WIREFRAME_SLOPE: the lines' constant and
// slope-scaled depth bias (D3D11 units; negative is toward the viewer with the
// game's depth, less-equal from near 0).
const std::int32_t g_wire_bias = [] {
    const char* e = std::getenv("BBHOST_WIREFRAME_BIAS");
    return e ? std::atoi(e) : -512;
}();
const float g_wire_slope = [] {
    const char* e = std::getenv("BBHOST_WIREFRAME_SLOPE");
    return e ? static_cast<float>(std::atof(e)) : -1.0f;
}();

template <typename D, typename F> std::uint32_t restyled(Kind kind, std::uint32_t id, const char* what, F&& change) {
    Restyled& r = t_restyled[kind];
    if (id == r.from && t_style == r.style) return r.to;
    const auto* d = static_cast<const D*>(desc_of(kind, id));
    if (!d) return id;
    D c = *d;
    change(c);
    bool made = false;
    const std::uint32_t to = intern(kind, c, &made);
    if (made) host_log("gx-state: %s %08x restyled (%d): %08x", what, id, t_style, to);
    if (!to) return id;
    r = {id, to, t_style};
    return to;
}

// GX's blend factors are D3D11's 0-based (ZERO 0, ONE 1), its ops Vulkan's (ADD 0, SUBTRACT 1).
void blend_all(GxBlendDesc& b, std::uint8_t src, std::uint8_t dst, std::uint8_t op) {
    for (GxBlendTarget& t : b.rt) {
        t.enable = 1;
        t.src = t.src_alpha = src;
        t.dst = t.dst_alpha = dst;
        t.op = t.op_alpha = op;
    }
}
}  // namespace

// Subtracting is not idempotent: a pixel two faces cover at one depth (the
// hi-hit has coplanar ones) would come out at the second face's colour. So a
// subtracting draw is pulled toward the viewer a little and tested less-than,
// its depth written: the first face at a pixel passes, the next one there not.
std::uint32_t gx_depth_stencil_id(std::uint64_t object) {
    const std::uint32_t id = lookup(kDepthStencil, object);
    if (!(t_style & (kGxDrawLines | kGxDrawSubtract)) || !id) return id;
    return restyled<GxDepthStencilDesc>(kDepthStencil, id, "depth-stencil", [](GxDepthStencilDesc& d) {
        if (t_style & kGxDrawLines) {
            d.depth_write = 0;
        } else {
            d.depth_write = 1;
            d.depth_func = 1;  // LESS (GX's compare functions are Vulkan's)
        }
    });
}
std::uint32_t gx_blend_id(std::uint64_t object) {
    const std::uint32_t id = lookup(kBlend, object);
    if (!(t_style & (kGxDrawBlack | kGxDrawSubtract)) || !id) return id;
    return restyled<GxBlendDesc>(kBlend, id, "blend", [](GxBlendDesc& b) {
        if (t_style & kGxDrawBlack) blend_all(b, 0, 0, 0);  // src * 0 + dst * 0
        else blend_all(b, 1, 1, 1);                         // src - dst
    });
}
std::uint32_t gx_raster_id(std::uint64_t object) {
    const std::uint32_t id = lookup(kRaster, object);
    if (!(t_style & (kGxDrawLines | kGxDrawSubtract)) || !id) return id;
    return restyled<GxRasterDesc>(kRaster, id, "rasterizer", [](GxRasterDesc& r) {
        const bool lines = t_style & kGxDrawLines;
        r.fill = lines ? 1 : r.fill;
        r.depth_bias = lines ? g_wire_bias : -16;
        r.slope_bias = lines ? g_wire_slope : 0.0f;
        r.bias_clamp = 0.0f;
    });
}
void gx_draw_style(int style) { t_style = style; }
const GxBlendDesc* gx_blend_desc(std::uint32_t id) { return static_cast<const GxBlendDesc*>(desc_of(kBlend, id)); }
const GxDepthStencilDesc* gx_depth_stencil_desc(std::uint32_t id) {
    return static_cast<const GxDepthStencilDesc*>(desc_of(kDepthStencil, id));
}
const GxRasterDesc* gx_raster_desc(std::uint32_t id) { return static_cast<const GxRasterDesc*>(desc_of(kRaster, id)); }

std::string gx_state_report() {
    // Each rasterizer description, for the depth bias it asks for.
    std::string rasters;
    for (std::uint64_t i = 1; i <= g_distinct[kRaster].load(std::memory_order_relaxed) && i < kMaxDescs; ++i) {
        const auto* d = static_cast<const GxRasterDesc*>(g_descs[kRaster][i].load(std::memory_order_acquire));
        if (!d) continue;
        char one[160];
        std::snprintf(one, sizeof(one), "; %08x cull %u front-ccw %u fill %u clip %u bias %d clamp %g slope %g",
                      (static_cast<std::uint32_t>(kRaster) << kIndexBits) | static_cast<std::uint32_t>(i), d->cull, d->front_ccw,
                      d->fill, d->depth_clip, d->depth_bias, d->bias_clamp, d->slope_bias);
        rasters += one;
    }
    char buf[400];
    const auto u = [](const std::atomic<std::uint64_t>& a) { return static_cast<unsigned long long>(a.load(std::memory_order_relaxed)); };
    std::snprintf(buf, sizeof(buf),
                  "gx-state: objects parsed at first sight (distinct descriptions, refused): blend %llu (%llu, %llu), depth-stencil "
                  "%llu (%llu, %llu), rasterizer %llu (%llu, %llu; %llu with a depth bias); address table full %llu",
                  u(g_parsed[kBlend]), u(g_distinct[kBlend]), u(g_refused[kBlend]), u(g_parsed[kDepthStencil]),
                  u(g_distinct[kDepthStencil]), u(g_refused[kDepthStencil]), u(g_parsed[kRaster]), u(g_distinct[kRaster]),
                  u(g_refused[kRaster]), u(g_biased), u(g_table_full));
    return std::string(buf) + "\ngx-state: rasterizer descriptions" + rasters;
}
