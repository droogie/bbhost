#include "host/translation_cache.h"

#include "core/portable.h"
#include "log.h"
#include "translator_fingerprint.h"

#include <zlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace gpu {

namespace {

// A field added to any of these must be added to the key (options) or the
// entry (results) below, and the size here updated: a translation that
// depends on something the key does not hold would be handed to the wrong
// program, and a result field the entry does not hold would come back empty.
#if defined(__GLIBCXX__) && defined(__x86_64__)
static_assert(sizeof(gcn::TranslateOptions) == 312, "a new TranslateOptions field: add it to hash_options");  // a bool can land in padding: check the list too
static_assert(sizeof(gcn::TranslateResult) == 424, "a new TranslateResult field: add it to write_result and read_result");
static_assert(sizeof(gcn::ResourcePath) == 40 && sizeof(gcn::ResourceStep) == 8, "a new ResourcePath field: add it to the entry");
static_assert(sizeof(gcn::ImageBinding) == 64 && sizeof(gcn::SamplerBinding) == 56 && sizeof(gcn::BufferBinding) == 56,
              "a new binding field: add it to the entry");
static_assert(sizeof(gcn::VertexElement) == 16, "a new VertexElement field: add it to hash_options");
static_assert(sizeof(gcn::LiftResult) == 104, "a new LiftResult field: add it to the lift's entry (write_lift, read_lift)");
#endif

constexpr std::uint32_t kMagic = 0x43544242;  // "BBTC"
// The file's own layout. The translator's changes are its fingerprint's;
// this changes only when the entries are written differently.
constexpr std::uint32_t kVersion = 2;  // 2: wave64_needs
// What the file may hold (compressed; it is kept in memory while the game
// runs): past it, the entries this run did not use go first. A world run's
// ~9,000 translations and lifts are a few tens of MiB.
constexpr std::size_t kMaxEntries = 65536;
constexpr std::uint64_t kMaxBytes = 256ull << 20;
constexpr std::uint32_t kMaxRaw = 64u << 20;  // one entry uncompressed: the largest translations are ~1 MiB
// A periodic write at most this often; the exit writes regardless.
constexpr auto kSaveInterval = std::chrono::seconds(60);

struct Key {
    std::uint64_t a = 0, b = 0;
    bool operator==(const Key& o) const { return a == o.a && b == o.b; }
};
struct KeyHash {
    std::size_t operator()(const Key& k) const { return static_cast<std::size_t>(k.a ^ (k.b * 0x9e3779b97f4a7c15ull)); }
};

// Two lanes over the same 32-bit words - FNV-1a over their bytes and a
// multiply-xorshift - so a key is 128 bits: ~10^4 entries a file make a
// collision of both something to ignore.
struct KeyHasher {
    std::uint64_t a = 0xcbf29ce484222325ull, b = 0x2545f4914f6cdd1dull;
    void u32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) a = (a ^ ((v >> (8 * i)) & 0xff)) * 0x100000001b3ull;
        b = (b ^ v) * 0x9e3779b97f4a7c15ull;
        b ^= b >> 31;
    }
    void u64(std::uint64_t v) {
        u32(static_cast<std::uint32_t>(v));
        u32(static_cast<std::uint32_t>(v >> 32));
    }
    void i32(int v) { u32(static_cast<std::uint32_t>(v)); }
    void flag(bool v) { u32(v ? 1u : 0u); }
    void str(const std::string& s) {
        u32(static_cast<std::uint32_t>(s.size()));
        for (char c : s) u32(static_cast<std::uint8_t>(c));
    }
    template <typename T>
    void list(const std::vector<T>& v) {
        u32(static_cast<std::uint32_t>(v.size()));
        for (const T& x : v) u32(static_cast<std::uint32_t>(x));
    }
};

// The decoded program as gcn::translate sees it: its instructions' words.
void hash_program(KeyHasher& h, const gcn::Program& p) {
    h.u32(static_cast<std::uint32_t>(p.insts.size()));
    h.u32(p.end);
    for (const gcn::Inst& in : p.insts) {
        h.u32(in.size);
        for (std::uint32_t k = 0; k < in.size && k < 3; ++k) h.u32(in.words[k]);
        if (in.has_literal) h.u32(in.literal);
    }
}

// Every TranslateOptions field, in declaration order (translate.h).
void hash_options(KeyHasher& h, const gcn::TranslateOptions& o) {
    h.u32(static_cast<std::uint32_t>(o.stage));
    h.flag(o.cb_ssbo);
    h.flag(o.bindless);
    h.u32(o.bindless_set);
    h.u32(static_cast<std::uint32_t>(o.cb_ssbo_exclude.size()));
    for (const std::string& s : o.cb_ssbo_exclude) h.str(s);
    h.flag(o.cb_no_fallback);
    h.flag(o.exec_known);
    h.u32(o.rsrc1);
    h.u32(o.rsrc2);
    h.u32(o.ps_input_ena);
    h.u32(o.ps_flat_mask);
    h.u32(o.vs_out_cntl);
    h.u32(o.ps_clip_discard);
    h.list(o.ps_input_map);
    h.flag(o.link_outputs);
    h.list(o.output_links);
    h.i32(o.debug_ps);
    for (std::uint32_t t : o.cs_threads) h.u32(t);
    h.u32(o.max_lds_bytes);
    h.flag(o.fetch != nullptr);  // the program it points at, after the options
    h.u32(static_cast<std::uint32_t>(o.vertex_input.size()));
    for (const gcn::VertexElement& e : o.vertex_input) {
        h.u32(e.location);
        h.u32(e.vdata);
        h.u32(e.count);
        h.u32(e.w3);
    }
    h.flag(o.vertex_formats_from_params);
    h.flag(o.force_dispatcher);
    h.flag(o.invariant_position);
    h.u32(o.descriptor_set);
    h.i32(o.cb_descriptor_set);
    h.u32(o.cb_binding_base);
    h.u32(static_cast<std::uint32_t>(o.image_dims.size()));
    for (const auto& d : o.image_dims) {
        h.u32(d.first);
        h.flag(d.second);
    }
    h.u32(static_cast<std::uint32_t>(o.sampler_force_unnormalized.size()));
    for (bool m : o.sampler_force_unnormalized) h.flag(m);
    h.flag(o.early_fragment_tests);
    h.u32(static_cast<std::uint32_t>(o.tess_role));
    h.u32(o.domain_level);
    h.u32(o.tess_patch_control_points);
    h.flag(o.tess_lds_attributes);
    h.u32(o.tess_attr_vec4s);
    h.u32(o.tess_window);
    h.flag(o.tess_lds_bound);
    h.flag(o.tess_quads);
    h.i32(o.tess_spacing);
    h.flag(o.tess_cw);
    std::uint32_t near_bits;
    std::memcpy(&near_bits, &o.tess_near_cull, 4);
    h.u32(near_bits);
    h.flag(o.tess_patch_cull);
    h.flag(o.tess_patch_cull_all);
}

// ---- an entry: the TranslateResult, every field ----
struct Writer {
    std::vector<std::uint8_t> out;
    void u8(std::uint8_t v) { out.push_back(v); }
    void u32(std::uint32_t v) { out.insert(out.end(), reinterpret_cast<const std::uint8_t*>(&v), reinterpret_cast<const std::uint8_t*>(&v) + 4); }
    void u64(std::uint64_t v) { out.insert(out.end(), reinterpret_cast<const std::uint8_t*>(&v), reinterpret_cast<const std::uint8_t*>(&v) + 8); }
    void i32(int v) { u32(static_cast<std::uint32_t>(v)); }
    void str(const std::string& s) {
        u32(static_cast<std::uint32_t>(s.size()));
        out.insert(out.end(), s.begin(), s.end());
    }
    void words(const std::vector<std::uint32_t>& v) {
        u32(static_cast<std::uint32_t>(v.size()));
        if (!v.empty()) {
            const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
            out.insert(out.end(), p, p + v.size() * 4);
        }
    }
    void path(const gcn::ResourcePath& p) {
        i32(p.user_sgpr);
        u32(static_cast<std::uint32_t>(p.loads.size()));
        for (const gcn::ResourceStep& s : p.loads) {
            u32(s.offset_dw);
            u8(s.vsharp ? 1 : 0);
        }
        i32(p.final_offset_dw);
        u8(static_cast<std::uint8_t>((p.final_vsharp ? 1 : 0) | (p.immediate ? 2 : 0)));
    }
    void at(const std::map<std::uint32_t, std::uint32_t>& m) {
        u32(static_cast<std::uint32_t>(m.size()));
        for (const auto& [k, v] : m) {
            u32(k);
            u32(v);
        }
    }
};
struct Reader {
    const std::uint8_t* p;
    const std::uint8_t* end;
    bool ok = true;
    bool take(void* dst, std::size_t n) {
        if (!ok || static_cast<std::size_t>(end - p) < n) return ok = false;
        std::memcpy(dst, p, n);
        p += n;
        return true;
    }
    std::uint8_t u8() {
        std::uint8_t v = 0;
        take(&v, 1);
        return v;
    }
    std::uint32_t u32() {
        std::uint32_t v = 0;
        take(&v, 4);
        return v;
    }
    std::uint64_t u64() {
        std::uint64_t v = 0;
        take(&v, 8);
        return v;
    }
    int i32() { return static_cast<int>(u32()); }
    std::uint32_t count(std::size_t each) {  // a length that fits what is left
        const std::uint32_t n = u32();
        if (!ok || static_cast<std::size_t>(end - p) < static_cast<std::size_t>(n) * each) return ok = false, 0;
        return n;
    }
    std::string str() {
        const std::uint32_t n = count(1);
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
    void words(std::vector<std::uint32_t>& v) {
        v.resize(count(4));
        if (!v.empty()) take(v.data(), v.size() * 4);
    }
    void path(gcn::ResourcePath& r) {
        r.user_sgpr = i32();
        r.loads.resize(count(5));
        for (gcn::ResourceStep& s : r.loads) {
            s.offset_dw = u32();
            s.vsharp = u8() != 0;
        }
        r.final_offset_dw = i32();
        const std::uint8_t f = u8();
        r.final_vsharp = f & 1;
        r.immediate = f & 2;
    }
    void at(std::map<std::uint32_t, std::uint32_t>& m) {
        const std::uint32_t n = count(8);
        for (std::uint32_t k = 0; k < n; ++k) {
            const std::uint32_t key = u32();
            m[key] = u32();
        }
    }
};

void write_result(Writer& w, const gcn::TranslateResult& r, bool with_spirv) {
    w.words(with_spirv ? r.spirv : std::vector<std::uint32_t>{});
    w.u8(r.bindless ? 1 : 0);
    w.u32(static_cast<std::uint32_t>(r.images.size()));
    for (const gcn::ImageBinding& b : r.images) {
        w.path(b.path);
        w.i32(b.sgpr);
        w.u32(b.binding);
        w.u8(static_cast<std::uint8_t>((b.storage ? 1 : 0) | (b.r128 ? 2 : 0) | (b.da ? 4 : 0) | (b.arrayed ? 8 : 0) | (b.depth ? 16 : 0) |
                                       (b.cube ? 32 : 0)));
        w.u32(b.dim);
        w.u32(b.kind);
    }
    w.u32(static_cast<std::uint32_t>(r.samplers.size()));
    for (const gcn::SamplerBinding& b : r.samplers) {
        w.path(b.path);
        w.i32(b.sgpr);
        w.u32(b.binding);
        w.u8(b.compare ? 1 : 0);
    }
    w.u32(static_cast<std::uint32_t>(r.buffers.size()));
    for (const gcn::BufferBinding& b : r.buffers) {
        w.path(b.path);
        w.u32(b.binding);
        w.u32(b.max_dw);
        w.u8(static_cast<std::uint8_t>((b.pointer ? 1 : 0) | (b.indexed ? 2 : 0)));
    }
    w.at(r.image_at);
    w.at(r.sampler_at);
    w.at(r.buffer_at);
    w.u32(static_cast<std::uint32_t>(r.errors.size()));
    for (const std::string& e : r.errors) w.str(e);
    w.u32(r.lds_bytes);
    w.u32(static_cast<std::uint32_t>(r.store_buffers.size()));
    for (const gcn::ResourcePath& p : r.store_buffers) w.path(p);
    w.u32(r.untraced_stores);
    w.u32(r.unnormalized_samples);
    w.u32(static_cast<std::uint32_t>(r.walks.size()));
    for (const auto& [why, n] : r.walks) {
        w.str(why);
        w.u32(n);
    }
    w.words(r.vs_params);
    w.words(r.ps_inputs);
    w.i32(r.vs_clip_count);
    w.u8(r.vs_point_size ? 1 : 0);
    w.u32(r.wave64_needs);  // what keeps a pixel program at 64 lanes (gcn/wave.h)
}

bool read_result(Reader& r, gcn::TranslateResult& t) {
    r.words(t.spirv);
    t.bindless = r.u8() != 0;
    t.images.resize(r.count(14));
    for (gcn::ImageBinding& b : t.images) {
        r.path(b.path);
        b.sgpr = r.i32();
        b.binding = r.u32();
        const std::uint8_t f = r.u8();
        b.storage = f & 1;
        b.r128 = f & 2;
        b.da = f & 4;
        b.arrayed = f & 8;
        b.depth = f & 16;
        b.cube = f & 32;
        b.dim = r.u32();
        b.kind = r.u32();
    }
    t.samplers.resize(r.count(14));
    for (gcn::SamplerBinding& b : t.samplers) {
        r.path(b.path);
        b.sgpr = r.i32();
        b.binding = r.u32();
        b.compare = r.u8() != 0;
    }
    t.buffers.resize(r.count(14));
    for (gcn::BufferBinding& b : t.buffers) {
        r.path(b.path);
        b.binding = r.u32();
        b.max_dw = r.u32();
        const std::uint8_t f = r.u8();
        b.pointer = f & 1;
        b.indexed = f & 2;
    }
    r.at(t.image_at);
    r.at(t.sampler_at);
    r.at(t.buffer_at);
    t.errors.resize(r.count(4));
    for (std::string& e : t.errors) e = r.str();
    t.lds_bytes = r.u32();
    t.store_buffers.resize(r.count(10));
    for (gcn::ResourcePath& p : t.store_buffers) r.path(p);
    t.untraced_stores = r.u32();
    t.unnormalized_samples = r.u32();
    const std::uint32_t walks = r.count(8);
    for (std::uint32_t k = 0; k < walks && r.ok; ++k) {
        std::string why = r.str();
        t.walks[why] = r.u32();
    }
    r.words(t.vs_params);
    r.words(t.ps_inputs);
    t.vs_clip_count = r.i32();
    t.vs_point_size = r.u8() != 0;
    t.wave64_needs = r.u32();
    return r.ok && r.p == r.end;
}

// ---- the cache ----
struct Slot {
    const std::uint8_t* data = nullptr;  // compressed: the loaded file's bytes, or an entry this run made (`made`)
    std::uint32_t size = 0, raw = 0;
    bool used = false;  // taken or made in this run: kept first when the file is full
    bool bad = false;   // did not read back: neither taken nor written again
};
struct Cache {
    std::mutex mu;
    std::atomic<bool> ready{false};
    bool on = false;
    std::string switches;                                  // gcn::translation_switches() at the load
    std::vector<std::uint8_t> file;                        // what was loaded: slots point into it, never freed
    std::vector<std::unique_ptr<std::uint8_t[]>> made;     // under mu: entries made in this run, never freed
    std::unordered_map<Key, Slot, KeyHash> slots;          // under mu
    std::uint64_t bytes = 0;                               // under mu: compressed, every slot
    bool dirty = false;                                    // under mu
    std::chrono::steady_clock::time_point saved{};         // under mu: the last write
    std::size_t loaded = 0;
    std::atomic<std::uint64_t> hits{0}, hit_us{0}, misses{0}, miss_us{0}, stored{0}, store_us{0}, bad{0}, early{0};
    std::atomic<std::uint64_t> raw_hit_bytes{0};
    std::atomic<std::uint64_t> lift_hits{0}, lift_hit_us{0}, lift_misses{0}, lift_miss_us{0};
} g_tc;

const bool g_tc_enabled = [] {
    const char* e = std::getenv("BBHOST_TRANSLATION_CACHE");
    return !(e && e[0] == '0');
}();
std::atomic<bool> g_tc_loaded{false};  // translation_cache_load ran (tests/translation_cache_test.cpp resets it)

std::uint64_t us_since(std::chrono::steady_clock::time_point t0) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}

Key key_for(const gcn::Program& prog, const gcn::TranslateOptions& o, TranslationUse use) {
    KeyHasher h;
    h.str(g_tc.switches);
    h.u32(static_cast<std::uint32_t>(use));
    hash_program(h, prog);
    hash_options(h, o);
    if (o.fetch) hash_program(h, *o.fetch);
    return {h.a, h.b};
}

// A lift's entry: its SPIR-V and its rejections.
void write_lift(Writer& w, const gcn::LiftResult& l) {
    w.words(l.spirv);
    w.u32(static_cast<std::uint32_t>(l.rejections.size()));
    for (const std::string& e : l.rejections) w.str(e);
}
bool read_lift(Reader& r, gcn::LiftResult& l) {
    r.words(l.spirv);
    l.rejections.resize(r.count(4));
    for (std::string& e : l.rejections) e = r.str();
    return r.ok && r.p == r.end;
}

// The entry under `key`, uncompressed and given to `parse`: false when there
// is none. One that does not read back is marked, translated again by the
// caller, and left out of the next write.
template <typename Parse>
bool lookup(const Key& key, Parse&& parse) {
    Slot found;
    {
        std::lock_guard<std::mutex> lk(g_tc.mu);
        if (const auto it = g_tc.slots.find(key); it != g_tc.slots.end() && !it->second.bad) {
            it->second.used = true;
            found = it->second;
        }
    }
    if (!found.data) return false;
    std::vector<std::uint8_t> raw(found.raw);
    uLongf n = static_cast<uLongf>(found.raw);
    if (uncompress(raw.data(), &n, found.data, static_cast<uLong>(found.size)) == Z_OK && n == found.raw) {
        Reader r{raw.data(), raw.data() + raw.size()};
        if (parse(r)) {
            g_tc.raw_hit_bytes.fetch_add(found.raw, std::memory_order_relaxed);
            return true;
        }
    }
    std::lock_guard<std::mutex> lk(g_tc.mu);
    if (const auto it = g_tc.slots.find(key); it != g_tc.slots.end()) it->second.bad = true;
    g_tc.dirty = true;  // the next write leaves it out
    if (g_tc.bad.fetch_add(1) < 4) host_log("render: translation cache: an entry did not read back; translating it again");
    return false;
}

void store(const Key& key, const Writer& w) {
    const auto t0 = std::chrono::steady_clock::now();
    uLongf n = compressBound(static_cast<uLong>(w.out.size()));
    std::unique_ptr<std::uint8_t[]> data(new std::uint8_t[n]);
    if (compress2(data.get(), &n, w.out.data(), static_cast<uLong>(w.out.size()), Z_BEST_SPEED) != Z_OK) return;
    {
        std::lock_guard<std::mutex> lk(g_tc.mu);
        Slot s;
        s.data = data.get();
        s.size = static_cast<std::uint32_t>(n);
        s.raw = static_cast<std::uint32_t>(w.out.size());
        s.used = true;
        auto [it, fresh] = g_tc.slots.try_emplace(key, s);
        if (!fresh && it->second.bad) {  // translated again because the entry did not read back: this one replaces it
            g_tc.bytes -= it->second.size;
            it->second = s;
            fresh = true;
        }
        if (fresh) {  // else another thread translated the same and kept its own
            g_tc.made.push_back(std::move(data));
            g_tc.bytes += n;
            g_tc.dirty = true;
            g_tc.stored.fetch_add(1, std::memory_order_relaxed);
        }
    }
    g_tc.store_us.fetch_add(us_since(t0), std::memory_order_relaxed);
}

std::string path_tmp(const std::string& path) { return path + ".tmp" + std::to_string(host_thread_id()); }

std::atomic<std::uint64_t> g_window_hits{0}, g_window_hit_us{0}, g_window_misses{0}, g_window_miss_us{0};
std::atomic<std::uint64_t> g_window_lift_hits{0}, g_window_lift_misses{0};

}  // namespace

gcn::TranslateResult translate_cached(const gcn::Program& prog, const gcn::TranslateOptions& o, TranslationUse use) {
    const bool paths = use == TranslationUse::kPaths;
    if (!g_tc.ready.load(std::memory_order_acquire) || !g_tc.on) {
        if (g_tc_enabled && !g_tc.ready.load(std::memory_order_acquire)) g_tc.early.fetch_add(1, std::memory_order_relaxed);
        gcn::TranslateResult t = gcn::translate(prog, o);
        if (paths) t.spirv = {};
        return t;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const Key key = key_for(prog, o, use);
    gcn::TranslateResult t;
    if (lookup(key, [&](Reader& r) { return read_result(r, t); })) {
        const std::uint64_t us = us_since(t0);
        g_tc.hits.fetch_add(1, std::memory_order_relaxed);
        g_tc.hit_us.fetch_add(us, std::memory_order_relaxed);
        g_window_hits.fetch_add(1, std::memory_order_relaxed);
        g_window_hit_us.fetch_add(us, std::memory_order_relaxed);
        return t;
    }
    const auto t1 = std::chrono::steady_clock::now();
    t = gcn::translate(prog, o);
    if (paths) t.spirv = {};  // as a hit has it: the bindings are what is read
    const std::uint64_t us = us_since(t1);
    g_tc.misses.fetch_add(1, std::memory_order_relaxed);
    g_tc.miss_us.fetch_add(us, std::memory_order_relaxed);
    g_window_misses.fetch_add(1, std::memory_order_relaxed);
    g_window_miss_us.fetch_add(us, std::memory_order_relaxed);
    Writer w;
    write_result(w, t, !paths);
    store(key, w);
    return t;
}

gcn::LiftResult lift_cached(bool vertex, const gcn::Program& prog, const gcn::TranslateOptions& o, const gcn::TranslateResult& reference) {
    const auto lift = [&] {
        return vertex ? gcn::lift_vertex_shader(prog, o, reference) : gcn::lift_pixel_shader(prog, o, reference);
    };
    if (!g_tc.ready.load(std::memory_order_acquire) || !g_tc.on) return lift();
    const auto t0 = std::chrono::steady_clock::now();
    const Key key = key_for(prog, o, vertex ? TranslationUse::kLiftVertex : TranslationUse::kLiftPixel);
    gcn::LiftResult l;
    if (lookup(key, [&](Reader& r) { return read_lift(r, l); })) {
        const std::uint64_t us = us_since(t0);
        g_tc.lift_hits.fetch_add(1, std::memory_order_relaxed);
        g_tc.lift_hit_us.fetch_add(us, std::memory_order_relaxed);
        g_window_lift_hits.fetch_add(1, std::memory_order_relaxed);
        g_window_hit_us.fetch_add(us, std::memory_order_relaxed);
        return l;
    }
    const auto t1 = std::chrono::steady_clock::now();
    l = lift();
    const std::uint64_t us = us_since(t1);
    g_tc.lift_misses.fetch_add(1, std::memory_order_relaxed);
    g_tc.lift_miss_us.fetch_add(us, std::memory_order_relaxed);
    g_window_lift_misses.fetch_add(1, std::memory_order_relaxed);
    g_window_miss_us.fetch_add(us, std::memory_order_relaxed);
    Writer w;
    write_lift(w, l);
    store(key, w);
    return l;
}

void translation_cache_load(const std::string& path) {
    // Once: a second device start keeps the first's slots, which lookups read
    // without a copy.
    if (g_tc_loaded.exchange(true)) return;
    g_tc.switches = gcn::translation_switches();
    g_tc.on = g_tc_enabled && !path.empty();
    if (!g_tc.on) {
        host_log("render: translation cache off (%s)", g_tc_enabled ? "no data root, or BBHOST_PIPELINE_CACHE=0" : "BBHOST_TRANSLATION_CACHE=0");
        g_tc.ready.store(true, std::memory_order_release);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    {
        // Read whole in one call (an istreambuf_iterator read was ~200 MB/s).
        // A file past the cap and its slack is not one this build wrote.
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        const std::streamoff size = in ? static_cast<std::streamoff>(in.tellg()) : 0;
        if (size > 0 && static_cast<std::uint64_t>(size) <= kMaxBytes + (64ull << 20)) {
            g_tc.file.resize(static_cast<std::size_t>(size));
            in.seekg(0);
            if (!in.read(reinterpret_cast<char*>(g_tc.file.data()), size)) g_tc.file.assign(1, 0);  // unreadable: start again
        } else if (size > 0) {
            g_tc.file.assign(1, 0);
        }
    }
    const std::string fingerprint = BBHOST_TRANSLATOR_FINGERPRINT;
    Reader r{g_tc.file.data(), g_tc.file.data() + g_tc.file.size()};
    bool header = g_tc.file.size() >= 16 && r.u32() == kMagic && r.u32() == kVersion;
    const std::string theirs = header ? r.str() : std::string();
    const std::uint32_t n = header ? r.u32() : 0;
    header = header && r.ok;
    if (g_tc.file.empty()) {
        host_log("render: translation cache: none yet (%s); this run's translations are kept for the next start", path.c_str());
    } else if (!header || theirs != fingerprint) {
        host_log("render: translation cache: %s (%s), starting again with this build's translator (%s)",
                 header ? "made by another build of the translator" : "not a file of this format", header ? theirs.c_str() : path.c_str(),
                 fingerprint.c_str());
        g_tc.file.clear();
        g_tc.file.shrink_to_fit();
        g_tc.dirty = true;  // written over at the first save, even if nothing new comes
    } else {
        std::size_t dropped = 0;
        for (std::uint32_t k = 0; k < n && r.ok; ++k) {
            Key key;
            key.a = r.u64();
            key.b = r.u64();
            Slot s;
            s.raw = r.u32();
            s.size = r.count(1);
            if (!r.ok) break;
            s.bad = s.raw > kMaxRaw;  // a damaged length: never allocated for
            s.data = r.p;
            r.p += s.size;
            if (g_tc.slots.emplace(key, s).second) g_tc.bytes += s.size;
            else ++dropped;
        }
        if (!r.ok) {
            host_log("render: translation cache: the file ends early (%zu of %u entries read); the rest are translated again", g_tc.slots.size(), n);
            g_tc.dirty = true;
        }
        g_tc.loaded = g_tc.slots.size();
        host_log("render: translation cache: %zu translations and lifts from earlier runs (%.1f MiB, read in %llu ms); translator %s%s",
                 g_tc.loaded, g_tc.file.size() / 1048576.0, static_cast<unsigned long long>(us_since(t0) / 1000), fingerprint.c_str(),
                 dropped ? ", some keys twice" : "");
    }
    g_tc.saved = std::chrono::steady_clock::now();
    g_tc.ready.store(true, std::memory_order_release);
}

void translation_cache_save(const std::string& path, bool force) {
    if (!g_tc.on || path.empty()) return;
    static std::mutex writing;  // the report's thread and the exit's
    std::lock_guard<std::mutex> wl(writing);
    struct Out {
        Key key;
        Slot slot;
    };
    std::vector<Out> out;
    {
        std::lock_guard<std::mutex> lk(g_tc.mu);
        if (!g_tc.dirty) return;
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - g_tc.saved < kSaveInterval) return;
        g_tc.dirty = false;
        g_tc.saved = now;
        out.reserve(g_tc.slots.size());
        for (const auto& [key, s] : g_tc.slots) {
            if (!s.bad) out.push_back({key, s});
        }
    }
    // Over the limits, what this run used goes first.
    std::stable_sort(out.begin(), out.end(), [](const Out& x, const Out& y) { return x.slot.used && !y.slot.used; });
    std::size_t keep = 0;
    std::uint64_t bytes = 0;
    while (keep < out.size() && keep < kMaxEntries && bytes + out[keep].slot.size <= kMaxBytes) bytes += out[keep++].slot.size;
    const auto t0 = std::chrono::steady_clock::now();
    Writer head;
    head.u32(kMagic);
    head.u32(kVersion);
    head.str(BBHOST_TRANSLATOR_FINGERPRINT);
    head.u32(static_cast<std::uint32_t>(keep));
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    const std::string tmp = path_tmp(path);
    bool ok = false;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(head.out.data()), static_cast<std::streamsize>(head.out.size()));
        for (std::size_t k = 0; k < keep && f; ++k) {
            Writer e;
            e.u64(out[k].key.a);
            e.u64(out[k].key.b);
            e.u32(out[k].slot.raw);
            e.u32(out[k].slot.size);
            f.write(reinterpret_cast<const char*>(e.out.data()), static_cast<std::streamsize>(e.out.size()));
            f.write(reinterpret_cast<const char*>(out[k].slot.data), static_cast<std::streamsize>(out[k].slot.size));
        }
        ok = static_cast<bool>(f);
    }
    if (ok) std::filesystem::rename(tmp, path, ec);
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        host_log("render: saving the translation cache to %s failed", path.c_str());
        return;
    }
    host_log("render: translation cache saved: %zu translations and lifts, %.1f MiB (written in %llu ms)%s", keep, bytes / 1048576.0,
             static_cast<unsigned long long>(us_since(t0) / 1000), keep < out.size() ? ", the oldest left out" : "");
}

void translation_cache_save_async(const std::string& path) {
    static std::atomic<bool> saving{false};
    if (!g_tc.on || path.empty()) return;
    {
        std::lock_guard<std::mutex> lk(g_tc.mu);
        if (!g_tc.dirty || std::chrono::steady_clock::now() - g_tc.saved < kSaveInterval) return;
    }
    if (saving.exchange(true)) return;
    std::thread([path] {
        translation_cache_save(path, false);
        saving.store(false);
    }).detach();
}

void translation_cache_report() {
    if (!g_tc_enabled) return;
    std::size_t entries = 0;
    std::uint64_t bytes = 0;
    {
        std::lock_guard<std::mutex> lk(g_tc.mu);
        entries = g_tc.slots.size();
        bytes = g_tc.bytes;
    }
    host_log("render: translation cache: %llu translations taken from it (%.2f s), %llu translated (%.1f s); %llu lifts taken from it "
             "(%.2f s), %llu lifted (%.1f s); %.1f MiB read back; %llu kept for the next start (%.2f s compressing); %zu entries loaded at "
             "the start, %zu now (%.1f MiB); %llu did not read back, %llu translated before it was loaded",
             static_cast<unsigned long long>(g_tc.hits.load()), g_tc.hit_us.load() / 1e6, static_cast<unsigned long long>(g_tc.misses.load()),
             g_tc.miss_us.load() / 1e6, static_cast<unsigned long long>(g_tc.lift_hits.load()), g_tc.lift_hit_us.load() / 1e6,
             static_cast<unsigned long long>(g_tc.lift_misses.load()), g_tc.lift_miss_us.load() / 1e6, g_tc.raw_hit_bytes.load() / 1048576.0,
             static_cast<unsigned long long>(g_tc.stored.load()), g_tc.store_us.load() / 1e6, g_tc.loaded, entries, bytes / 1048576.0,
             static_cast<unsigned long long>(g_tc.bad.load()), static_cast<unsigned long long>(g_tc.early.load()));
}

std::string translation_cache_window() {
    const std::uint64_t hits = g_window_hits.exchange(0), hit_us = g_window_hit_us.exchange(0);
    const std::uint64_t misses = g_window_misses.exchange(0), miss_us = g_window_miss_us.exchange(0);
    const std::uint64_t lift_hits = g_window_lift_hits.exchange(0), lift_misses = g_window_lift_misses.exchange(0);
    if (!hits && !misses && !lift_hits && !lift_misses) return {};
    char buf[200];
    std::snprintf(buf, sizeof(buf), "translations %llu cached, %llu translated; lifts %llu cached, %llu lifted (%llu ms reading, %llu ms making)",
                  static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses), static_cast<unsigned long long>(lift_hits),
                  static_cast<unsigned long long>(lift_misses), static_cast<unsigned long long>(hit_us / 1000),
                  static_cast<unsigned long long>(miss_us / 1000));
    return buf;
}

const char* translator_fingerprint() { return BBHOST_TRANSLATOR_FINGERPRINT; }

bool translation_cache_cold() { return g_tc.ready.load(std::memory_order_acquire) && g_tc.on && g_tc.loaded == 0; }

}  // namespace gpu
