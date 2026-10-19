#include "host/translation_cache.h"

#include "core/paced_file.h"
#include "core/portable.h"
#include "log.h"
#include "translator_fingerprint.h"
#include "gcn/hlsl.h"
#include "host/dxc.h"

#ifdef BBHOST_HAVE_SPIRV_TOOLS
#include <spirv-tools/libspirv.hpp>
#endif

#include <zlib.h>

#if defined(__GLIBC__) && !defined(_WIN32)
#include <malloc.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
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
static_assert(sizeof(gcn::TranslateOptions) == 320, "a new TranslateOptions field: add it to hash_options");  // a bool can land in padding: check the list too
static_assert(sizeof(gcn::TranslateResult) == 456, "a new TranslateResult field: add it to write_result and read_result");
static_assert(sizeof(gcn::ResourcePath) == 40 && sizeof(gcn::ResourceStep) == 8, "a new ResourcePath field: add it to the entry");
static_assert(sizeof(gcn::ImageBinding) == 64 && sizeof(gcn::SamplerBinding) == 56 && sizeof(gcn::BufferBinding) == 56,
              "a new binding field: add it to the entry");
static_assert(sizeof(gcn::VertexElement) == 20, "a new VertexElement field: add it to hash_options");
// (A lift's proof, listing, wave-operation counts and engine names are not
// kept: the names are put on after the cache (render.cpp name_stage), the rest
// is diagnostics.)
static_assert(sizeof(gcn::LiftResult) == 296, "a new LiftResult field: add it to the lift's entry (write_lift, read_lift)");
#endif

constexpr std::uint32_t kMagic = 0x43544242;  // "BBTC"
// The file's own layout. The translator's changes are its fingerprint's;
// this changes only when the entries are written differently.
constexpr std::uint32_t kVersion = 3;  // 2: wave64_needs; 3: gds_counters
// What the file may hold (compressed; it is kept in memory while the game
// runs): past it, the entries this run did not use go first. A world run's
// ~9,000 translations and lifts are a few tens of MiB.
constexpr std::size_t kMaxEntries = 65536;
constexpr std::uint64_t kMaxBytes = 256ull << 20;
constexpr std::uint32_t kMaxRaw = 64u << 20;  // one entry uncompressed: the largest translations are ~1 MiB
// A periodic write at most this often, and only once this much is new
// (compressed): 4 MiB or 2% of what the file holds, the pipeline cache's rule
// (gpu.cpp). The file is rewritten whole, 170 MiB in a world run, and a minute
// of play adds a dozen entries. The exit writes whatever is new.
constexpr auto kSaveInterval = std::chrono::seconds(60);
constexpr std::uint64_t kSaveMinNew = 4ull << 20;

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
        h.u32(e.packed);
    }
    h.flag(o.vertex_formats_from_params);
    h.flag(o.force_dispatcher);
    h.flag(o.invariant_position);
    h.u32(o.descriptor_set);
    h.i32(o.cb_descriptor_set);
    h.u32(o.cb_binding_base);
    h.flag(o.cb_bias_zero);
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
    h.flag(o.gds_counters);
    if (o.gs_role != gcn::TranslateOptions::GsRole::None) h.u32(0x6510u | static_cast<std::uint32_t>(o.gs_role));  // other stages' keys as they were
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
        w.u8(static_cast<std::uint8_t>((b.pointer ? 1 : 0) | (b.indexed ? 2 : 0) | (b.ranged ? 4 : 0) | (b.written ? 8 : 0) | (b.bytes ? 16 : 0)));
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
    w.u32(static_cast<std::uint32_t>(r.read_buffers.size()));
    for (const gcn::TranslateResult::BufferRead& b : r.read_buffers) {
        w.path(b.path);
        w.u8(b.pointer ? 1 : 0);
        w.u32(b.max_dw);
    }
    w.u32(r.untraced_reads);
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
    w.u32(r.gds_counters);
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
        b.ranged = f & 4;
        b.written = f & 8;
        b.bytes = f & 16;
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
    t.read_buffers.resize(r.count(10));
    for (gcn::TranslateResult::BufferRead& b : t.read_buffers) {
        r.path(b.path);
        b.pointer = r.u8() != 0;
        b.max_dw = r.u32();
    }
    t.untraced_reads = r.u32();
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
    t.gds_counters = r.u32();
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
    std::uint64_t saved_bytes = 0;                         // under mu: `bytes` at the last write (or the load)
    bool dirty = false;                                    // under mu
    std::chrono::steady_clock::time_point saved{};         // under mu: the last write
    std::size_t loaded = 0;
    std::atomic<std::uint64_t> hits{0}, hit_us{0}, misses{0}, miss_us{0}, stored{0}, store_us{0}, bad{0}, early{0};
    std::atomic<std::uint64_t> bounded_left_out{0};  // translation_cache_bounded's, past the byte limit
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

thread_local bool t_bounded = false;  // translation_cache_bounded

void store(const Key& key, const Writer& w) {
    if (t_bounded) {
        std::lock_guard<std::mutex> lk(g_tc.mu);
        if (g_tc.bytes + w.out.size() / 4 > kMaxBytes) {  // a typical entry compresses to a third or less
            g_tc.bounded_left_out.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
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

// Under g_tc.mu: a write is due when entries came in (or one is to be left
// out) and, unless forced, a minute has passed and enough of them are new.
// DXC modules bb-hlsl kept since the last write, and whether its queue has
// emptied since: a few hundred modules are less than the byte threshold, and a
// start ended from outside (SIGTERM, a Steam stop) writes nothing at its exit,
// so they are written on their own count - every 512, and when the queue empties.
std::atomic<std::uint64_t> g_hlsl_unsaved{0};
std::atomic<bool> g_hlsl_drained{false};
bool save_due_locked(bool force) {
    if (!g_tc.dirty) return false;
    if (force) return true;
    if (std::chrono::steady_clock::now() - g_tc.saved < kSaveInterval) return false;
    if (const std::uint64_t n = g_hlsl_unsaved.load(); n >= 512 || (n > 0 && g_hlsl_drained.load())) return true;
    const std::uint64_t fresh = g_tc.bytes > g_tc.saved_bytes ? g_tc.bytes - g_tc.saved_bytes : 0;
    return fresh >= std::max<std::uint64_t>(kSaveMinNew, g_tc.saved_bytes / 50);
}

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

namespace {
template <typename Lift>
gcn::LiftResult lift_through_cache(const Key& key, const Lift& lift);

// ---- BBHOST_HLSL: the lift through HLSL and DXC ----
int hlsl_level() {
    static const int level = [] {
        const char* e = std::getenv("BBHOST_HLSL_O");
        return e && *e ? std::clamp(std::atoi(e), 0, 3) : 3;
    }();
    return level;
}
Dxc& hlsl_dxc() {
    static Dxc dxc;
    static std::once_flag once;
    std::call_once(once, [] {
        if (dxc.load())
            host_log("render: HLSL path on (%s): DXC %s from %s, -O%d",
                     hlsl_mode() == 2 ? "DXC's modules from the cache, the rest compiled behind the game for the next start" : "compiled at use",
                     dxc.version().c_str(), dxc.path().c_str(), hlsl_level());
        else host_log("render: HLSL path on, but %s: every lift keeps the lifter's SPIR-V", dxc.error().c_str());
    });
    return dxc;
}
struct HlslStats {
    std::atomic<std::uint64_t> taken{0}, compiled{0}, compile_us{0}, cached{0}, kept{0};
    std::mutex mu;
    std::map<std::string, std::uint64_t> reasons;  // under mu: why a lift kept its SPIR-V
} g_hlsl;
// A reason without the numbers in it, so like ones count together.
std::string hlsl_reason_key(const std::string& r) {
    std::string k;
    bool digits = false;
    for (char c : r.substr(0, 160)) {
        if (c == '\n') break;
        if (c >= '0' && c <= '9') {
            if (!digits) k += '#';
            digits = true;
            continue;
        }
        digits = false;
        k += c;
    }
    return k;
}
constexpr bool kHlslValidator =
#ifdef BBHOST_HAVE_SPIRV_TOOLS
    true;
#else
    false;
#endif
std::string hlsl_validate(const std::vector<std::uint32_t>& spirv) {
#ifdef BBHOST_HAVE_SPIRV_TOOLS
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
        if (msg.empty()) msg = m;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);
    return tools.Validate(spirv.data(), spirv.size(), vo) ? std::string() : (msg.empty() ? "invalid" : msg);
#else
    (void)spirv;
    return {};
#endif
}
// BBHOST_HLSL_DUMP: the HLSL (and the lift it was printed from, to compile it
// again with gcn2hlsl --spv) under the lift's hash; empty without the variable.
// BBHOST_HLSL_DUMP=1: <data>/bbhost/hlsl-dump (hlsl_set_folders).
std::mutex g_hlsl_dump_mu;
std::string g_hlsl_dump_default;  // under g_hlsl_dump_mu
const char* hlsl_dump_dir() {
    static const char* const env = std::getenv("BBHOST_HLSL_DUMP");
    if (!env || !*env) return nullptr;
    if (std::strcmp(env, "1") != 0) return env;
    std::lock_guard<std::mutex> lk(g_hlsl_dump_mu);
    return g_hlsl_dump_default.empty() ? nullptr : g_hlsl_dump_default.c_str();  // set once at device start, never changed after
}
std::atomic<std::string (*)(const std::string&)> g_hlsl_name_lookup{nullptr};
// The program's name for a header: its bundle entry's beside the GX name.
std::string hlsl_title(const std::string& name) {
    if (name.empty()) return name;
    if (auto lookup = g_hlsl_name_lookup.load()) {
        if (const std::string file = lookup(name); !file.empty()) return file + " (" + name + ")";
    }
    return name;
}
// The shader's stable identity, what its dumped and edited files are named
// by: the bundle entry and GX's name ("GXGuiShader_Tex.ppo@6d921b0b"), or GX's
// name alone for the eboot's own programs. Both come from the game's files, so
// a translator update keeps them; the lift's hash does not survive one. Empty
// when the lift's program is not known.
std::string hlsl_stable_name(const std::string& gx) {
    if (gx.empty()) return {};
    std::string file;
    if (auto lookup = g_hlsl_name_lookup.load()) file = lookup(gx);
    std::string name = file.empty() ? gx : file + "@" + gx;
    for (char& c : name) {
        if (std::strchr("<>:\"/\\|?*", c) || static_cast<unsigned char>(c) < 32) c = '_';  // a file name on every system
    }
    return name;
}
// "ps", "vs", "hs", "ds", "cs": the lift's stage, from its entry point's
// execution model.
std::string lift_stage_tag(const std::vector<std::uint32_t>& spirv) {
    for (std::size_t i = 5; i < spirv.size();) {
        const std::uint32_t len = spirv[i] >> 16, op = spirv[i] & 0xffff;
        if (!len) break;
        if (op == 15 && i + 1 < spirv.size()) {  // OpEntryPoint
            switch (spirv[i + 1]) {
                case 0: return "vs";
                case 1: return "hs";
                case 2: return "ds";
                case 4: return "ps";
                case 5: return "cs";
                default: return "xx";
            }
        }
        i += len;
    }
    return "xx";
}
std::string lift_hash_hex(const std::vector<std::uint32_t>& lifted) {
    KeyHasher n;
    for (const std::uint32_t w : lifted) n.u32(w);
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(n.a));
    return buf;
}
// Whether the dumped file at `p` was written for the lift `hash16` (its second
// line says "// lift <hash>").
bool dump_is_lift(const std::filesystem::path& p, const std::string& hash16) {
    std::ifstream in(p, std::ios::binary);
    std::string line;
    for (int k = 0; k < 3 && std::getline(in, line); ++k) {
        if (line.find("// lift " + hash16) == 0) return true;
    }
    return false;
}
// BBHOST_HLSL_DUMP: the HLSL, and the lift it was printed from (to compile it
// again with gcn2hlsl --spv). Named by the shader's stable identity and stage
// - <stable>.<stage>.hlsl for the first lift of that program and stage met,
// <stable>.<stage>~<lift hash>.hlsl for each other variant - or, for a lift
// whose program is not known, by the lift's hash (<hash>.<stage>.hlsl).
std::string hlsl_dump(const std::vector<std::uint32_t>& lifted, const gcn::HlslResult& h, const std::string& stable) {
    const char* dir = hlsl_dump_dir();
    if (!dir) return {};
    const std::string hash16 = lift_hash_hex(lifted);
    const std::string stage = h.profile.size() >= 2 ? h.profile.substr(0, 2) : lift_stage_tag(lifted);
    std::lock_guard<std::mutex> lk(g_hlsl_dump_mu);  // the name chosen and the file written as one
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string name;
    if (stable.empty()) {
        name = hash16 + "." + stage;
    } else {
        const std::filesystem::path first = std::filesystem::path(dir) / (stable + "." + stage + ".hlsl");
        name = !std::filesystem::exists(first, ec) || dump_is_lift(first, hash16) ? stable + "." + stage : stable + "." + stage + "~" + hash16;
    }
    const std::filesystem::path base = std::filesystem::path(dir) / name;
    if (std::filesystem::exists(base.string() + ".hlsl", ec)) return base.string();  // written already
    if (!h.source.empty()) {
        // The lift's hash on the second line: the dump's own check, and an
        // edited file's fallback match.
        const std::size_t nl = h.source.find('\n');
        std::ofstream out(base.string() + ".hlsl", std::ios::binary);
        if (nl == std::string::npos) {
            out << h.source << "\n// lift " << hash16 << "\n";
        } else {
            out << h.source.substr(0, nl + 1) << "// lift " << hash16 << " (this build's lift; the name above is what a translator update keeps)\n"
                << h.source.substr(nl + 1);
        }
    }
    std::ofstream(base.string() + ".lift.spv", std::ios::binary)
        .write(reinterpret_cast<const char*>(lifted.data()), static_cast<std::streamsize>(lifted.size() * 4));
    return base.string();
}
// Prints, compiles and checks one lifted module: DXC's SPIR-V, or empty with `why`.
std::vector<std::uint32_t> hlsl_compile(const std::vector<std::uint32_t>& lifted, const gcn::HlslOptions& hopt, const Dxc& dxc, std::string& why) {
    gcn::HlslOptions titled = hopt;  // a dump's header names the program by its bundle entry too
    const bool dumping = hlsl_dump_dir() != nullptr;
    if (dumping) titled.title = hlsl_title(hopt.title);
    const gcn::HlslResult h = gcn::emit_hlsl(lifted, titled);
    const std::string dumped = dumping ? hlsl_dump(lifted, h, hlsl_stable_name(hopt.title)) : std::string();
    // The first few failures in full, with where they were dumped.
    const auto log_failure = [&](const std::string& what) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 8)
            host_log("render: HLSL: a lift kept its SPIR-V%s%s: %s", dumped.empty() ? "" : " (", dumped.empty() ? "" : (dumped + ")").c_str(),
                     what.substr(0, 600).c_str());
    };
    if (!h.ok()) {
        why = "the printer does not take it: " + (h.errors.empty() ? std::string("no source") : h.errors[0]);
        log_failure(why);
        return {};
    }
    if (h.validate_offsets && !kHlslValidator) {
        why = "a run-time sample offset, which DXC's validator refuses, and no SPIR-V validator in this build";
        log_failure(why);
        return {};
    }
    DxcOptions o;
    o.optimize = hlsl_level();
    std::vector<std::uint32_t> spirv;
    std::string error;
    if (!dxc.compile(h, o, spirv, error)) {
        why = "DXC: " + error.substr(0, error.find('\n'));
        log_failure("DXC: " + error);
        return {};
    }
    if (std::string d = gcn::hlsl_interface_diff(lifted, spirv); !d.empty()) {
        why = "another interface: " + d;
        log_failure(why);
        return {};
    }
    if (std::string d = gcn::hlsl_operation_diff(lifted, spirv); !d.empty()) {
        why = "other operations: " + d;
        log_failure(why);
        return {};
    }
    if (std::string v = hlsl_validate(spirv); !v.empty()) {
        why = "invalid SPIR-V: " + v;
        log_failure(why);
        return {};
    }
    return spirv;
}
void hlsl_keep(const std::string& why) {
    g_hlsl.kept.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_hlsl.mu);
    g_hlsl.reasons[hlsl_reason_key(why)]++;
}
// BBHOST_HLSL: `l`'s module through HLSL and DXC, cached; unchanged where that
// fails. `lift_key`: the lift's own entry in this cache (null when it was made
// without the cache), which hlsl_lift_of finds it by. `o`: a tessellation
// stage's options (a hull's domain, partitioning and winding are HLSL's to
// declare).
void hlsl_stage(gcn::LiftResult& l, const Key* lift_key, const gcn::TranslateOptions* o = nullptr);

// ---- BBHOST_HLSL: the lift behind each of DXC's modules ----
// A driver may refuse a pipeline of DXC's modules that it builds from the
// lifts' (NVIDIA's compiler did, for eight compute modules, until df928d5);
// the stage is then built from the lift's own module (hlsl_lift_of), never
// skipped. A module is known by its words without the debug instructions -
// OpSource*, OpName, OpMemberName, OpString, OpLine, OpNoLine,
// OpModuleProcessed - since the engine's names go on the module after this
// cache gives it out (render.cpp name_stage), DXC's included.
Key module_identity(const std::vector<std::uint32_t>& spirv) {
    KeyHasher h;
    const std::size_t n = spirv.size();
    h.u32(static_cast<std::uint32_t>(n >= 5 ? 5 : n));
    for (std::size_t i = 0; i < n && i < 5; ++i) h.u32(spirv[i]);
    for (std::size_t i = 5; i < n;) {
        const std::uint32_t len = spirv[i] >> 16, op = spirv[i] & 0xffff;
        if (!len || i + len > n) {  // malformed: every word counts
            for (; i < n; ++i) h.u32(spirv[i]);
            break;
        }
        const bool debug = (op >= 2 && op <= 8) || op == 317 || op == 330;
        if (!debug) {
            for (std::uint32_t k = 0; k < len; ++k) h.u32(spirv[i + k]);
        }
        i += len;
    }
    return {h.a, h.b};
}
struct HlslLiftOf {
    struct From {
        Key lift_key{};  // the lift's entry in this cache
        bool cached = false;
        std::vector<std::uint32_t> words;  // the lift itself, when it was made without the cache
    };
    std::mutex mu;
    std::unordered_map<Key, From, KeyHash> by_module;  // under mu: DXC's module (module_identity) -> its lift
    std::map<std::string, std::uint64_t> refused;      // under mu: pipelines the driver refused with DXC's modules, by what was built
    std::atomic<std::uint64_t> refusals{0}, stages{0}, rebuilt{0}, refused_again{0}, lift_again{0};
} g_hlsl_lift_of;
void hlsl_remember_lift(const std::vector<std::uint32_t>& dxc_module, const Key* lift_key, const std::vector<std::uint32_t>& lift) {
    HlslLiftOf::From from;
    if (lift_key) {
        from.lift_key = *lift_key;
        from.cached = true;
    } else {
        from.words = lift;
    }
    const Key id = module_identity(dxc_module);
    std::lock_guard<std::mutex> lk(g_hlsl_lift_of.mu);
    g_hlsl_lift_of.by_module.insert_or_assign(id, std::move(from));
}
}  // namespace

bool hlsl_lift_of(const std::vector<std::uint32_t>& module, std::vector<std::uint32_t>& lift) {
    if (!hlsl_on() || module.empty()) return false;
    HlslLiftOf::From from;
    {
        const Key id = module_identity(module);
        std::lock_guard<std::mutex> lk(g_hlsl_lift_of.mu);
        const auto it = g_hlsl_lift_of.by_module.find(id);
        if (it == g_hlsl_lift_of.by_module.end()) return false;
        from = it->second;
    }
    if (!from.cached) {
        lift = std::move(from.words);
        return !lift.empty();
    }
    gcn::LiftResult l;
    if (!lookup(from.lift_key, [&](Reader& r) { return read_lift(r, l); }) || !l.ok()) return false;
    lift = std::move(l.spirv);
    return true;
}

void hlsl_note_refused(const char* what, int stages, bool rebuilt) {
    g_hlsl_lift_of.refusals.fetch_add(1, std::memory_order_relaxed);
    g_hlsl_lift_of.stages.fetch_add(static_cast<std::uint64_t>(stages), std::memory_order_relaxed);
    (rebuilt ? g_hlsl_lift_of.rebuilt : g_hlsl_lift_of.refused_again).fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(g_hlsl_lift_of.mu);
        g_hlsl_lift_of.refused[what]++;
    }
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 8) {
        host_log("render: HLSL: the driver refused %s holding DXC's modules; %s from the lifts' own (%d stage%s)", what,
                 rebuilt ? "built" : "refused again when built", stages, stages == 1 ? "" : "s");
    }
}

void hlsl_note_lift_again(int stages) { g_hlsl_lift_of.lift_again.fetch_add(static_cast<std::uint64_t>(stages), std::memory_order_relaxed); }

namespace {
// The Shaders setting (hlsl_set_setting): -1 not given, else the mode it asks for.
std::atomic<int> g_hlsl_setting{-1};
}  // namespace

void hlsl_set_setting(bool hlsl) { g_hlsl_setting.store(hlsl ? 2 : 0); }
namespace {
std::atomic<bool> g_hlsl_inline{false};
}  // namespace
void hlsl_set_inline(bool on) { g_hlsl_inline.store(on); }

int hlsl_mode() {
    // Decided once, at the first lift: BBHOST_HLSL wins (0 lift only, 1 compile
    // at use, 2 behind the game), then the Shaders setting, then the default -
    // DXC's modules from the cache, the rest compiled behind the game.
    static const int mode = [] {
        const char* e = std::getenv("BBHOST_HLSL");
        if (e && (e[0] >= '0' && e[0] <= '2') && !e[1]) return e[0] - '0';
        const int s = g_hlsl_setting.load();
        return s >= 0 ? s : 2;
    }();
    return mode;
}
bool hlsl_on() { return hlsl_mode() != 0; }

namespace {
// ---- The compiles behind the game (BBHOST_HLSL=2, the default) ----
// A stage whose DXC module is not in the cache keeps the lift's SPIR-V for
// this start; bb-hlsl compiles the module at the lowest priority and keeps it
// in the cache, and the next start takes it. Nothing waits on DXC.
//
// The order is the pipeline manifest's: what this start's draws met first
// (kHlslLaneDrawn), then what the last starts drew (the pipeline and stage
// manifests' replays, kHlslLaneManifest, in the manifest's most-recent-first
// order), then what GX created but nothing has drawn yet (kHlslLaneCreated),
// then predictions and the opt-in sweep (kHlslLaneLater). A stage met again
// from a lane before its own moves up to it.
thread_local int t_hlsl_lane = kHlslLaneDrawn;
thread_local std::string t_hlsl_name;  // hlsl_name_next
constexpr int kLanes = 4;
struct HlslBehind {
    struct Job {
        Key key;
        std::vector<std::uint32_t> lifted;
        gcn::HlslOptions hopt;
    };
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::shared_ptr<const Job>> jobs[kLanes];             // under mu
    // Under mu: the stage (render.cpp's stage-cache key) each waiting job was
    // lifted for, so a draw that takes the stage moves its job up (hlsl_note_drawn).
    std::unordered_map<std::uint64_t, std::weak_ptr<const Job>> by_stage;  // the queues hold the jobs
    // Under mu: queued in this run - the lane it waits in, or kRunning, kDone.
    std::unordered_map<Key, int, KeyHash> seen;
    static constexpr int kRunning = 100, kDone = 101;
    std::size_t waiting = 0;                                         // under mu: distinct stages queued, not yet taken
    bool started = false;                                            // under mu
    bool drained_logged = false;                                     // under mu
    int threads = 0;
    std::chrono::steady_clock::time_point first{};                   // under mu: the first job's queueing
    std::atomic<std::uint64_t> deferred{0}, done{0}, done_us{0}, failed{0}, promoted{0}, gated_ms{0};
    std::atomic<std::uint64_t> lane_done[kLanes] = {};
};
// Never destroyed: its threads wait on it until the process ends.
HlslBehind& g_hlsl_behind = *new HlslBehind;
// render.cpp's: whether the start's own compiles (the precompile workers) are
// still busy; bb-hlsl waits for them.
std::atomic<bool (*)()> g_hlsl_busy{nullptr};

void hlsl_behind_thread() {
    host_thread_set_name("bb-hlsl");
    host_thread_background();
    const Dxc& dxc = hlsl_dxc();
    for (;;) {
        std::shared_ptr<const HlslBehind::Job> job;
        int lane = 0;
        {
            std::unique_lock<std::mutex> lk(g_hlsl_behind.mu);
            for (;;) {
                g_hlsl_behind.cv.wait(lk, [] { return g_hlsl_behind.waiting > 0; });
                for (lane = 0; lane < kLanes && g_hlsl_behind.jobs[lane].empty(); ++lane) {
                }
                if (lane == kLanes) {  // only stale copies were counted: none
                    g_hlsl_behind.waiting = 0;
                    continue;
                }
                job = std::move(g_hlsl_behind.jobs[lane].front());
                g_hlsl_behind.jobs[lane].pop_front();
                int& state = g_hlsl_behind.seen[job->key];
                if (state != lane) continue;  // moved up to another lane, or taken from it already
                state = HlslBehind::kRunning;
                --g_hlsl_behind.waiting;
                break;
            }
        }
        // Behind the start's own compiles: while the precompile workers have
        // work, the cores are theirs (and on an integrated GPU, while the world
        // is drawn: render.cpp's probe).
        if (bool (*busy)() = g_hlsl_busy.load()) {
            const auto g0 = std::chrono::steady_clock::now();
            while (busy()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            g_hlsl_behind.gated_ms.fetch_add(static_cast<std::uint64_t>(us_since(g0) / 1000), std::memory_order_relaxed);
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::string why;
        const std::vector<std::uint32_t> spirv = hlsl_compile(job->lifted, job->hopt, dxc, why);
        Writer w;
        w.u8(spirv.empty() ? 0 : 1);
        w.words(spirv);
        w.str(why);
        store(job->key, w);
        g_hlsl_unsaved.fetch_add(1, std::memory_order_relaxed);
        g_hlsl_behind.done.fetch_add(1, std::memory_order_relaxed);
        g_hlsl_behind.lane_done[lane].fetch_add(1, std::memory_order_relaxed);
        g_hlsl_behind.done_us.fetch_add(us_since(t0), std::memory_order_relaxed);
        if (spirv.empty()) g_hlsl_behind.failed.fetch_add(1, std::memory_order_relaxed);
        bool drained = false;
        {
            std::lock_guard<std::mutex> lk(g_hlsl_behind.mu);
            g_hlsl_behind.seen[job->key] = HlslBehind::kDone;
            bool running = false;
            for (const auto& [k, st] : g_hlsl_behind.seen) {
                if (st == HlslBehind::kRunning) {
                    running = true;
                    break;
                }
            }
            if (g_hlsl_behind.waiting == 0 && !running && !g_hlsl_behind.drained_logged) {
                g_hlsl_behind.drained_logged = true;
                g_hlsl_drained.store(true);  // the next periodic save writes them
                drained = true;
            }
        }
        job.reset();
        if (!drained) continue;
        // DXC's heap on these threads stays with them (glibc keeps a thread's
        // arena): handed back to the system each time the queue empties, out
        // of the queue's lock.
        long trimmed_kib = 0;
#if defined(__GLIBC__) && !defined(_WIN32)
        {
            const auto rss_kib = [] {
                long pages = 0, rss = 0;
                if (FILE* f = std::fopen("/proc/self/statm", "r")) {
                    if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0;
                    std::fclose(f);
                }
                return rss * (sysconf(_SC_PAGESIZE) / 1024);
            };
            const long before = rss_kib();
            malloc_trim(0);
            trimmed_kib = before - rss_kib();
        }
#endif
        // Logged the first few times: later ones are a few stages a new place brought.
        static std::atomic<int> drained_logs{0};
        if (drained_logs.fetch_add(1) < 3) {
            host_log("render: HLSL: every stage met so far has its DXC module for the next start: %llu compiled on %d bb-hlsl thread%s in %.1f s of "
                     "their time, %.1f s after the first was queued (%llu waited %.1f s, held back by the start's own compiles or the world); %llu failed; %.0f MiB of "
                     "DXC's heap handed back to the system",
                     static_cast<unsigned long long>(g_hlsl_behind.done.load()), g_hlsl_behind.threads, g_hlsl_behind.threads == 1 ? "" : "s",
                     g_hlsl_behind.done_us.load() / 1e6,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - g_hlsl_behind.first).count(),
                     static_cast<unsigned long long>(g_hlsl_behind.done.load()), g_hlsl_behind.gated_ms.load() / 1e3,
                     static_cast<unsigned long long>(g_hlsl_behind.failed.load()), trimmed_kib > 0 ? trimmed_kib / 1024.0 : 0.0);
        }
    }
}
thread_local std::shared_ptr<const HlslBehind::Job> t_last_deferred;  // hlsl_tag_stage's
void hlsl_promote_locked(const std::shared_ptr<const HlslBehind::Job>& job, int lane) {
    int& state = g_hlsl_behind.seen[job->key];
    if (state >= HlslBehind::kRunning || state <= lane) return;
    state = lane;
    g_hlsl_behind.jobs[lane].push_back(job);
    g_hlsl_behind.promoted.fetch_add(1, std::memory_order_relaxed);
    g_hlsl_behind.cv.notify_one();
}
void hlsl_defer(const Key& key, const std::vector<std::uint32_t>& lifted, const gcn::HlslOptions& hopt) {
    const int lane = std::clamp(t_hlsl_lane, 0, kLanes - 1);
    std::lock_guard<std::mutex> lk(g_hlsl_behind.mu);
    const auto [it, fresh] = g_hlsl_behind.seen.try_emplace(key, lane);
    if (!fresh) {
        // Met again from an earlier lane while it waits: it moves up. The
        // job queued first is found through the stage it was made for, or
        // made again.
        t_last_deferred = nullptr;
        if (it->second >= HlslBehind::kRunning || it->second <= lane) return;
        auto job = std::make_shared<const HlslBehind::Job>(HlslBehind::Job{key, lifted, hopt});
        t_last_deferred = job;
        hlsl_promote_locked(job, lane);
        return;
    }
    auto job = std::make_shared<const HlslBehind::Job>(HlslBehind::Job{key, lifted, hopt});
    t_last_deferred = job;
    g_hlsl_behind.jobs[lane].push_back(std::move(job));
    ++g_hlsl_behind.waiting;
    g_hlsl_behind.drained_logged = false;
    g_hlsl_behind.deferred.fetch_add(1, std::memory_order_relaxed);
    if (!g_hlsl_behind.started) {
        g_hlsl_behind.started = true;
        g_hlsl_behind.first = std::chrono::steady_clock::now();
        // BBHOST_HLSL_THREADS, else two (one on a machine of fewer than eight
        // hardware threads), at the lowest priority: the module set fills in
        // a few minutes of play without taking the game's cores.
        const char* e = std::getenv("BBHOST_HLSL_THREADS");
        const int hw = static_cast<int>(std::thread::hardware_concurrency());
        const int n = e && *e ? std::clamp(std::atoi(e), 1, 16) : (hw > 0 && hw < 8 ? 1 : 2);
        g_hlsl_behind.threads = n;
        for (int k = 0; k < n; ++k) std::thread(hlsl_behind_thread).detach();
    }
    g_hlsl_behind.cv.notify_one();
}

// ---- Edited HLSL: <data>/bbhost/shaders/<name>.hlsl ----
// A player's own source for a stage, named as BBHOST_HLSL_DUMP names it:
// compiled with DXC at the stage's first use and taken in place of the lift's
// module when it declares the same interface and validates; otherwise the
// stage keeps its own module (the lift's, or DXC's of it). For a lift, the
// files tried in order are <stable>.<stage>~<lift hash> (that one variant),
// <stable>.<stage> (every variant of the program and stage whose interface it
// matches - the name a translator update keeps), then <lift hash>.<stage>
// (the fallback, this build's lift only). Every outcome is logged once.
struct HlslOverrides {
    std::mutex mu;
    std::string dir;                                      // under mu
    std::unordered_map<std::string, std::string> files;   // under mu: name without .hlsl -> file
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> compiled;  // under mu: this run's results by lift (empty: refused)
    std::atomic<std::uint64_t> taken{0}, refused{0};
    std::atomic<bool> any{false};
} g_hlsl_overrides;

std::uint64_t lift_hash(const std::vector<std::uint32_t>& lifted) {
    KeyHasher n;
    for (const std::uint32_t w : lifted) n.u32(w);
    return n.a;
}

// The override's module for `l`, compiled once a run; false when there is no
// override for it (or it was refused: logged, and the caller goes on as without it).
bool hlsl_override(const std::vector<std::uint32_t>& lifted, const gcn::HlslOptions& hopt, const Dxc& dxc, std::vector<std::uint32_t>& out) {
    if (!g_hlsl_overrides.any.load(std::memory_order_acquire)) return false;
    const std::uint64_t hash = lift_hash(lifted);
    {
        std::lock_guard<std::mutex> lk(g_hlsl_overrides.mu);
        if (const auto it = g_hlsl_overrides.compiled.find(hash); it != g_hlsl_overrides.compiled.end()) {
            out = it->second;
            return !out.empty();
        }
    }
    const std::string hash16 = lift_hash_hex(lifted), stage = lift_stage_tag(lifted), stable = hlsl_stable_name(hopt.title);
    std::vector<std::string> names;
    if (!stable.empty()) {
        names.push_back(stable + "." + stage + "~" + hash16);
        names.push_back(stable + "." + stage);
    }
    names.push_back(hash16 + "." + stage);
    std::string file;
    {
        std::lock_guard<std::mutex> lk(g_hlsl_overrides.mu);
        for (const std::string& n : names) {
            if (const auto it = g_hlsl_overrides.files.find(n); it != g_hlsl_overrides.files.end()) {
                file = it->second;
                break;
            }
        }
    }
    if (file.empty()) return false;
    std::vector<std::uint32_t> spirv;
    std::string why;
    gcn::HlslResult h = gcn::emit_hlsl(lifted, hopt);  // the profile, entry point and arguments the lift needs
    std::string source;
    {
        std::ifstream in(file, std::ios::binary);
        source.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    if (h.profile.empty()) {
        why = "the printer does not take this lift, so its profile and arguments are unknown" +
              (h.errors.empty() ? std::string() : " (" + h.errors[0] + ")");
    } else if (source.empty()) {
        why = "the file is empty or unreadable";
    } else if (!dxc.loaded()) {
        why = "DXC not found";
    } else if (h.validate_offsets && !kHlslValidator) {
        why = "the lift takes a run-time sample offset, and this build has no SPIR-V validator to check it";
    } else {
        h.source = source;
        h.errors.clear();
        DxcOptions o;
        o.optimize = hlsl_level();
        std::string error;
        if (!dxc.compile(h, o, spirv, error)) {
            why = "DXC: " + error.substr(0, 400);
        } else if (std::string d = gcn::hlsl_interface_diff(lifted, spirv); !d.empty()) {
            why = "another interface than the lift's: " + d;
        } else if (std::string v = hlsl_validate(spirv); !v.empty()) {
            why = "invalid SPIR-V: " + v;
        }
        if (!why.empty()) spirv.clear();
    }
    {
        // Two threads can meet the same stage at once: the first result is
        // kept, counted and logged; the other takes it.
        std::lock_guard<std::mutex> lk(g_hlsl_overrides.mu);
        const auto [it, fresh] = g_hlsl_overrides.compiled.try_emplace(hash, spirv);
        if (!fresh) {
            out = it->second;
            return !out.empty();
        }
    }
    if (spirv.empty()) {
        g_hlsl_overrides.refused.fetch_add(1, std::memory_order_relaxed);
        host_log("render: HLSL override %s refused for %s (lift %s), the stage keeps its own module: %s", file.c_str(),
                 stable.empty() ? "a lift of no known program" : stable.c_str(), hash16.c_str(), why.c_str());
    } else {
        g_hlsl_overrides.taken.fetch_add(1, std::memory_order_relaxed);
        host_log("render: HLSL override %s compiled and taken for %s (lift %s, %zu words)", file.c_str(),
                 stable.empty() ? "a lift of no known program" : stable.c_str(), hash16.c_str(), spirv.size());
    }
    out = std::move(spirv);
    return !out.empty();
}

void hlsl_stage(gcn::LiftResult& l, const Key* lift_key, const gcn::TranslateOptions* to) {
    t_last_deferred = nullptr;
    std::string name = std::move(t_hlsl_name);  // taken by this lift, whatever happens to it
    t_hlsl_name.clear();
    if (!hlsl_on() || !l.ok()) return;
    gcn::HlslOptions hopt;
    hopt.title = std::move(name);
    if (to) {
        hopt.tess_quads = to->tess_quads;
        hopt.tess_spacing = to->tess_spacing;
        hopt.tess_cw = to->tess_cw;
    }
    const Dxc& dxc = hlsl_dxc();
    if (!dxc.loaded()) {
        hlsl_keep("DXC not found");
        return;
    }
    if (std::vector<std::uint32_t> own; hlsl_override(l.spirv, hopt, dxc, own)) {
        g_hlsl.taken.fetch_add(1, std::memory_order_relaxed);
        hlsl_remember_lift(own, lift_key, l.spirv);
        l.spirv = std::move(own);
        return;
    }
    const bool cache = g_tc.ready.load(std::memory_order_acquire) && g_tc.on;
    KeyHasher h;
    h.str(g_tc.switches);
    h.u32(static_cast<std::uint32_t>(TranslationUse::kHlsl));
    h.str(dxc.version());
    DxcOptions o;
    o.optimize = hlsl_level();
    for (const std::string& a : dxc_arguments(o)) h.str(a);
    h.flag(hopt.tess_quads);
    h.i32(hopt.tess_spacing);
    h.flag(hopt.tess_cw);
    h.u32(static_cast<std::uint32_t>(l.spirv.size()));
    for (const std::uint32_t w : l.spirv) h.u32(w);
    const Key key{h.a, h.b};
    std::vector<std::uint32_t> spirv;
    std::string why;
    bool found = false;
    if (cache) {
        found = lookup(key, [&](Reader& r) {
            const bool ok = r.u8() != 0;
            r.words(spirv);
            why = r.str();
            if (ok == spirv.empty()) return false;
            return r.ok && r.p == r.end;
        });
    }
    if (found) {
        g_hlsl.cached.fetch_add(1, std::memory_order_relaxed);
        // BBHOST_HLSL_DUMP: the source of a module the cache had, too.
        if (!spirv.empty() && hlsl_dump_dir()) {
            gcn::HlslOptions titled = hopt;
            titled.title = hlsl_title(hopt.title);
            hlsl_dump(l.spirv, gcn::emit_hlsl(l.spirv, titled), hlsl_stable_name(hopt.title));
        }
    } else if (hlsl_mode() == 2 && !g_hlsl_inline.load(std::memory_order_relaxed)) {
        // Behind the game: never compiled on the thread that needs the stage.
        // Before the cache is loaded (or with it off) there is nowhere to keep
        // the module for the next start, so the lift's is all there is.
        if (cache) hlsl_defer(key, l.spirv, hopt);
        else hlsl_keep("no translation cache to keep DXC's module in");
        return;  // the lift's SPIR-V this start
    } else {
        const auto t0 = std::chrono::steady_clock::now();
        spirv = hlsl_compile(l.spirv, hopt, dxc, why);
        g_hlsl.compiled.fetch_add(1, std::memory_order_relaxed);
        g_hlsl.compile_us.fetch_add(us_since(t0), std::memory_order_relaxed);
        if (cache) {
            Writer w;
            w.u8(spirv.empty() ? 0 : 1);
            w.words(spirv);
            w.str(why);
            store(key, w);
        }
    }
    if (spirv.empty()) {
        hlsl_keep(why);
        return;
    }
    g_hlsl.taken.fetch_add(1, std::memory_order_relaxed);
    hlsl_remember_lift(spirv, lift_key, l.spirv);
    l.spirv = std::move(spirv);
}
}  // namespace

void hlsl_counts(std::uint64_t& taken, std::uint64_t& compiled, std::uint64_t& kept, std::string& reasons) {
    taken = g_hlsl.taken.load();
    compiled = g_hlsl.compiled.load();
    kept = g_hlsl.kept.load();
    std::lock_guard<std::mutex> lk(g_hlsl.mu);
    for (const auto& [why, n] : g_hlsl.reasons) reasons += why + " x" + std::to_string(n) + "\n";
}
void hlsl_lane(int lane) { t_hlsl_lane = lane; }
void hlsl_name_next(std::string name) { t_hlsl_name = std::move(name); }
void hlsl_set_name_lookup(std::string (*lookup)(const std::string& name)) { g_hlsl_name_lookup.store(lookup); }
int hlsl_lane_now() { return t_hlsl_lane; }
void hlsl_set_busy_probe(bool (*busy)()) { g_hlsl_busy.store(busy); }

void hlsl_tag_stage(std::uint64_t stage_key) {
    if (!t_last_deferred) return;
    std::lock_guard<std::mutex> lk(g_hlsl_behind.mu);
    g_hlsl_behind.by_stage[stage_key] = t_last_deferred;
    t_last_deferred = nullptr;
}

void hlsl_note_drawn(std::uint64_t stage_key) {
    if (hlsl_mode() != 2) return;
    std::lock_guard<std::mutex> lk(g_hlsl_behind.mu);
    const auto it = g_hlsl_behind.by_stage.find(stage_key);
    if (it == g_hlsl_behind.by_stage.end()) return;
    if (const std::shared_ptr<const HlslBehind::Job> job = it->second.lock()) hlsl_promote_locked(job, kHlslLaneDrawn);
    g_hlsl_behind.by_stage.erase(it);
}

void hlsl_set_folders(const std::string& bbhost_dir) {
    if (bbhost_dir.empty() || !hlsl_on()) return;
    {
        std::lock_guard<std::mutex> lk(g_hlsl_dump_mu);
        g_hlsl_dump_default = (std::filesystem::path(bbhost_dir) / "hlsl-dump").string();
    }
    const std::string dir = (std::filesystem::path(bbhost_dir) / "shaders").string();
    std::unordered_map<std::string, std::string> files;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) continue;
        const std::string name = e.path().filename().string();
        if (name.size() <= 5 || name.compare(name.size() - 5, 5, ".hlsl") != 0) continue;
        files[name.substr(0, name.size() - 5)] = e.path().string();
    }
    std::lock_guard<std::mutex> lk(g_hlsl_overrides.mu);
    g_hlsl_overrides.dir = dir;
    if (!files.empty()) host_log("render: HLSL overrides: %zu edited shader%s in %s, each compiled at its stage's first use", files.size(),
                                 files.size() == 1 ? "" : "s", dir.c_str());
    g_hlsl_overrides.files = std::move(files);
    g_hlsl_overrides.any.store(!g_hlsl_overrides.files.empty(), std::memory_order_release);
}

gcn::LiftResult lift_cached(bool vertex, const gcn::Program& prog, const gcn::TranslateOptions& o, const gcn::TranslateResult& reference) {
    const auto lift = [&] {
        return vertex ? gcn::lift_vertex_shader(prog, o, reference) : gcn::lift_pixel_shader(prog, o, reference);
    };
    if (!g_tc.ready.load(std::memory_order_acquire) || !g_tc.on) {
        gcn::LiftResult l = lift();
        hlsl_stage(l, nullptr);
        return l;
    }
    const Key key = key_for(prog, o, vertex ? TranslationUse::kLiftVertex : TranslationUse::kLiftPixel);
    gcn::LiftResult l = lift_through_cache(key, lift);
    hlsl_stage(l, &key);
    return l;
}

gcn::LiftResult lift_compute_cached(const gcn::Program& prog, const gcn::TranslateOptions& o, const gcn::TranslateResult& reference) {
    const auto lift = [&] { return gcn::lift_compute_shader(prog, o, reference); };
    if (!g_tc.ready.load(std::memory_order_acquire) || !g_tc.on) {
        gcn::LiftResult l = lift();
        hlsl_stage(l, nullptr);
        return l;
    }
    const Key key = key_for(prog, o, TranslationUse::kLiftCompute);
    gcn::LiftResult l = lift_through_cache(key, lift);
    hlsl_stage(l, &key);
    return l;
}

gcn::LiftResult lift_tess_cached(TranslationUse use, const gcn::Program& prog, const gcn::TranslateOptions& o, const gcn::TranslateResult& reference,
                                 const gcn::TessLift& tess) {
    const auto lift = [&] {
        return use == TranslationUse::kLiftLs     ? gcn::lift_ls_shader(prog, o, reference, tess)
               : use == TranslationUse::kLiftHull ? gcn::lift_hull_shader(prog, o, reference, tess)
                                                  : gcn::lift_domain_shader(prog, o, reference, tess);
    };
    if (!g_tc.ready.load(std::memory_order_acquire) || !g_tc.on) {
        gcn::LiftResult l = lift();
        hlsl_stage(l, nullptr, &o);
        return l;
    }
    KeyHasher h;
    h.str(g_tc.switches);
    h.u32(static_cast<std::uint32_t>(use));
    hash_program(h, prog);
    hash_options(h, o);
    if (o.fetch) hash_program(h, *o.fetch);
    h.u32(tess.attr_vec4s);
    h.flag(tess.window);
    h.flag(tess.own_params);
    for (const std::uint32_t v : {tess.control_points, tess.ls_stride, tess.cp_stride, tess.patch_constants, tess.params_binding}) h.u32(v);
    h.str(tess.constants_path);
    for (const std::uint32_t v : tess.constants) h.u32(v);
    const Key key{h.a, h.b};
    gcn::LiftResult l = lift_through_cache(key, lift);
    hlsl_stage(l, &key, &o);
    return l;
}

gcn::LiftResult lift_gs_cached(TranslationUse use, const gcn::Program& prog, const gcn::TranslateOptions& o, const gcn::TranslateResult& reference,
                               const gcn::GsLift& gs) {
    const auto lift = [&] {
        return use == TranslationUse::kLiftEs ? gcn::lift_es_shader(prog, o, reference, gs) : gcn::lift_gs_shader(prog, o, reference, gs);
    };
    if (!g_tc.ready.load(std::memory_order_acquire) || !g_tc.on) {
        gcn::LiftResult l = lift();
        hlsl_stage(l, nullptr, &o);
        return l;
    }
    KeyHasher h;
    h.str(g_tc.switches);
    h.u32(static_cast<std::uint32_t>(use));
    hash_program(h, prog);
    hash_options(h, o);
    for (const std::uint32_t v : {gs.es_dwords, gs.max_vertices, gs.vertex_dwords, gs.out_prim}) h.u32(v);
    h.u32(static_cast<std::uint32_t>(gs.copy.size()));
    for (const gcn::GsCopyOut& c : gs.copy) {
        h.u32(static_cast<std::uint32_t>(c.target) | static_cast<std::uint32_t>(c.comp) << 8 | (c.constant ? 1u << 16 : 0u));
        h.u32(c.value);
    }
    const Key key{h.a, h.b};
    gcn::LiftResult l = lift_through_cache(key, lift);
    hlsl_stage(l, &key, &o);
    return l;
}

namespace {
template <typename Lift>
gcn::LiftResult lift_through_cache(const Key& key, const Lift& lift) {
    const auto t0 = std::chrono::steady_clock::now();
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
}  // namespace

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
    g_tc.saved_bytes = g_tc.bytes;
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
    std::uint64_t fresh = 0;
    {
        std::lock_guard<std::mutex> lk(g_tc.mu);
        if (!save_due_locked(force)) return;
        g_hlsl_unsaved.store(0);
        g_hlsl_drained.store(false);
        g_tc.dirty = false;
        g_tc.saved = std::chrono::steady_clock::now();
        fresh = g_tc.bytes > g_tc.saved_bytes ? g_tc.bytes - g_tc.saved_bytes : 0;
        g_tc.saved_bytes = g_tc.bytes;
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
    // A chunk at a time: written at once, its write-back held every other file
    // operation on a slow disk for as long - 1.5 s over NFS.
    HostPacedFile f(path, path_tmp(path));
    bool ok = f.write(head.out.data(), head.out.size());
    for (std::size_t k = 0; k < keep && ok; ++k) {
        Writer e;
        e.u64(out[k].key.a);
        e.u64(out[k].key.b);
        e.u32(out[k].slot.raw);
        e.u32(out[k].slot.size);
        ok = f.write(e.out.data(), e.out.size()) && f.write(out[k].slot.data, out[k].slot.size);
    }
    if (!ok || !f.commit()) {
        host_log("render: saving the translation cache to %s failed", path.c_str());
        return;
    }
    host_log("render: translation cache saved: %zu translations and lifts, %.1f MiB, %.1f MiB of it new (written in %llu ms)%s", keep,
             bytes / 1048576.0, fresh / 1048576.0, static_cast<unsigned long long>(us_since(t0) / 1000),
             keep < out.size() ? ", the oldest left out" : "");
}

void translation_cache_save_async(const std::string& path) {
    static std::atomic<bool> saving{false};
    if (!g_tc.on || path.empty()) return;
    {
        std::lock_guard<std::mutex> lk(g_tc.mu);
        if (!save_due_locked(false)) return;
    }
    if (saving.exchange(true)) return;
    std::thread([path] {
        translation_cache_save(path, false);
        saving.store(false);
    }).detach();
}

void translation_cache_report() {
    if (hlsl_on()) {
        std::string reasons;
        {
            std::lock_guard<std::mutex> lk(g_hlsl.mu);
            for (const auto& [why, n] : g_hlsl.reasons) reasons += "; " + why + " x" + std::to_string(n);
        }
        std::string behind;
        if (hlsl_mode() == 2) {
            std::size_t waiting = 0;
            {
                std::lock_guard<std::mutex> lk(g_hlsl_behind.mu);
                waiting = g_hlsl_behind.waiting;
            }
            char b[400];
            std::snprintf(b, sizeof(b), "; %llu kept the lift for this start, their modules compiled on bb-hlsl for the next (%llu done in %.1f s - "
                                        "drawn %llu, manifests %llu, created %llu, later %llu; %llu moved up a lane; %.1f s held back by the start's "
                                        "compiles or the world - %llu of them failed, %zu waiting)",
                          static_cast<unsigned long long>(g_hlsl_behind.deferred.load()), static_cast<unsigned long long>(g_hlsl_behind.done.load()),
                          g_hlsl_behind.done_us.load() / 1e6, static_cast<unsigned long long>(g_hlsl_behind.lane_done[0].load()),
                          static_cast<unsigned long long>(g_hlsl_behind.lane_done[1].load()),
                          static_cast<unsigned long long>(g_hlsl_behind.lane_done[2].load()),
                          static_cast<unsigned long long>(g_hlsl_behind.lane_done[3].load()),
                          static_cast<unsigned long long>(g_hlsl_behind.promoted.load()), g_hlsl_behind.gated_ms.load() / 1e3,
                          static_cast<unsigned long long>(g_hlsl_behind.failed.load()), waiting);
            behind = b;
        }
        if (g_hlsl_overrides.any.load()) {
            char b[160];
            std::snprintf(b, sizeof(b), "; edited HLSL (shaders/): %llu files taken, %llu refused",
                          static_cast<unsigned long long>(g_hlsl_overrides.taken.load()),
                          static_cast<unsigned long long>(g_hlsl_overrides.refused.load()));
            behind += b;
        }
        // The pipelines a driver refused while they held DXC's modules, which
        // were built from the lifts' instead (hlsl_lift_of), by what was built.
        std::string refused;
        {
            std::lock_guard<std::mutex> lk(g_hlsl_lift_of.mu);
            for (const auto& [what, n] : g_hlsl_lift_of.refused) refused += (refused.empty() ? " (" : ", ") + what + " x" + std::to_string(n);
        }
        if (!refused.empty()) refused += ")";
        host_log("render: HLSL: %llu lifted stages took DXC's module (%llu compiled at this start in %.1f s of worker time, %llu from the "
                 "translation cache)%s; %llu kept the lifter's SPIR-V%s; the driver refused %llu pipelines holding DXC's modules%s: %llu built "
                 "from the lifts' own (%llu stages), %llu refused again; %llu stages of later pipelines took their lift's at once",
                 static_cast<unsigned long long>(g_hlsl.taken.load()), static_cast<unsigned long long>(g_hlsl.compiled.load()),
                 g_hlsl.compile_us.load() / 1e6, static_cast<unsigned long long>(g_hlsl.cached.load()), behind.c_str(),
                 static_cast<unsigned long long>(g_hlsl.kept.load()), reasons.c_str(),
                 static_cast<unsigned long long>(g_hlsl_lift_of.refusals.load()), refused.c_str(),
                 static_cast<unsigned long long>(g_hlsl_lift_of.rebuilt.load()), static_cast<unsigned long long>(g_hlsl_lift_of.stages.load()),
                 static_cast<unsigned long long>(g_hlsl_lift_of.refused_again.load()),
                 static_cast<unsigned long long>(g_hlsl_lift_of.lift_again.load()));
    }
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

void translation_cache_bounded(bool on) { t_bounded = on; }
std::uint64_t translation_cache_bounded_left_out() { return g_tc.bounded_left_out.load(std::memory_order_relaxed); }

bool translation_cache_cold() { return g_tc.ready.load(std::memory_order_acquire) && g_tc.on && g_tc.loaded == 0; }

}  // namespace gpu
