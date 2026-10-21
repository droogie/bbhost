// The glitch hunt (BBHOST_GLITCH=1): find the frames that flash or stretch,
// with nobody watching, and name the draws behind them.
//
// Two detectors, each blind where the other sees:
//
// - The frame. Every presented frame is converted to RGBA8 and box-filtered
//   on the GPU - a blit chain, each level the 2x2 average of the one above -
//   down to about 60x34 cells, read back a few flips later. A frame whose
//   cells differ from both of its neighbours while the neighbours agree with
//   each other is a one-frame event: a cut or a fade moves the picture and it
//   stays moved, camera motion makes the neighbours differ *more* from each
//   other than from the frame between them, and a flash or a stretch comes
//   back. Two frames that differ together from the frames either side count
//   the same way.
// - The draws. Every draw is wrapped in an occlusion query, so what each one
//   covered is known once its submission retires. A triangle pulled across
//   the screen covers far more than it did the frame before and the frame
//   after, so a draw covering `ratio` times what the same draw covered in
//   both neighbours is the stretch itself, whatever the picture did. "The
//   same draw" is the same pipeline into the same target from the same index
//   buffer with the same count, the k-th such in its frame; failing that
//   (dynamic geometry moves between ring addresses every frame), the k-th
//   draw of that pipeline into that target.
//
// A catch writes <dir>/<flip>/ - the frame and both neighbours (PPM), and
// report.txt: the scores, the draws whose coverage jumped, then every draw of
// the frame with its coverage and its neighbours'. Each catch is a log line
// (`glitch: flip ...`); exit prints a summary with the pipelines that jumped.
// tools/glitch_hunt.sh drives a run through menus and combat and collects it.
//
//   BBHOST_GLITCH=1           on (the queries and the readbacks cost a few ms
//                             of GPU time a frame; off, none of this runs)
//   BBHOST_GLITCH_DIR=path    where catches go (build/glitch)
//   BBHOST_GLITCH_MAX=n       catches written in full (40); all are logged
//   BBHOST_GLITCH_CELL=x      a cell's change, 0..1, that counts (0.10)
//   BBHOST_GLITCH_CELLS=n     changed cells that make a frame a catch (6)
//   BBHOST_GLITCH_COV_RATIO=x coverage jump that makes a draw a spike (8; the same
//                             draw only)
//   BBHOST_GLITCH_COV_MIN=x   and the least of its viewport it must cover (0.02)
//
// The readbacks are not waited for (the frame watcher in render.cpp waits
// for the GPU at every flip, which keeps it in step with the CPU - and a race
// between the two is one of the things being looked for).
#include "host/gpu.h"
#include "host/gpu_internal.h"
#include "hle/modules.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gpu {

namespace {

using ull = unsigned long long;

struct Config {
    bool on = false;
    std::string dir = "build/glitch";
    unsigned max_catches = 40;
    unsigned max_big = 100;  // BBHOST_GLITCH_MAX_BIG: catches written on a budget of their own
    double cell = 0.10;
    unsigned cells = 6;
    double ratio = 8.0;
    double cov_min = 0.02;
};

const Config& cfg() {
    static const Config c = [] {
        Config k;
        const char* e = std::getenv("BBHOST_GLITCH");
        k.on = e && e[0] == '1';
        if (const char* v = std::getenv("BBHOST_GLITCH_DIR"); v && v[0]) k.dir = v;
        if (const char* v = std::getenv("BBHOST_GLITCH_MAX")) k.max_catches = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
        if (const char* v = std::getenv("BBHOST_GLITCH_MAX_BIG")) k.max_big = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
        if (const char* v = std::getenv("BBHOST_GLITCH_CELL")) k.cell = std::atof(v);
        if (const char* v = std::getenv("BBHOST_GLITCH_CELLS")) k.cells = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
        if (const char* v = std::getenv("BBHOST_GLITCH_COV_RATIO")) k.ratio = std::atof(v);
        if (const char* v = std::getenv("BBHOST_GLITCH_COV_MIN")) k.cov_min = std::atof(v);
        return k;
    }();
    return c;
}

bool g_precise = false;

// ---- coverage ----------------------------------------------------------------

// Queries a submission can hold: more than the draws one carries (a flush
// comes at kMaxQueued draws and dispatches together).
constexpr std::uint32_t kCovQueries = 4096;

// Each draw record's covered samples, parallel to g_draw_recs.
struct Cov {
    std::uint64_t rec = ~0ull;
    std::uint64_t samples = 0;
};
Cov g_cov[kDrawRecs];

std::uint64_t g_measured = 0, g_unmeasured = 0, g_pool_full = 0;

// ---- the race watch (BBHOST_GLITCH_WATCH) ------------------------------------

std::vector<std::string> watch_prefixes() {
    std::vector<std::string> out;
    const char* e = std::getenv("BBHOST_GLITCH_WATCH");
    for (const char* p = e; p && *p;) {
        const char* end = std::strchr(p, ',');
        const std::size_t n = end ? static_cast<std::size_t>(end - p) : std::strlen(p);
        if (n) out.emplace_back(p, n);
        if (!end) break;
        p = end + 1;
    }
    return out;
}
const std::vector<std::string> g_watch_prefixes = watch_prefixes();
const bool g_watch_snapshot = [] {
    const char* e = std::getenv("BBHOST_GLITCH_SNAPSHOT");
    return e && e[0] == '1';
}();
bool g_watch_now = false;  // the draw being resolved is watched (under g.mu)
std::vector<std::pair<std::uint64_t, std::uint64_t>> g_watch_ranges;
std::vector<int> g_watch_kinds;  // parallel: glitch_watch_read_locked's kind
std::uint64_t g_watch_changed_kind[3] = {};  // ranges that changed in flight, by kind
std::string g_watch_text;  // what the draw being resolved was given
struct WatchText {
    std::uint64_t rec = ~0ull;
    std::string text;
};
constexpr std::size_t kWatchTexts = 4096;
WatchText g_watch_texts[kWatchTexts];

std::uint64_t fnv(std::uint64_t va, std::uint64_t bytes) {
    if (!hle_kernel_va_mapped(va, static_cast<std::size_t>(bytes))) return 0;
    const auto* p = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(va));
    std::uint64_t h = 1469598103934665603ull;
    for (std::uint64_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

// A watched draw's ranges and their bytes when it was recorded, until the
// submission it went into has finished.
struct WatchRec {
    std::uint64_t rec = 0, serial = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::vector<int> kinds;
    std::vector<std::uint64_t> hashes;
};
std::vector<WatchRec> g_watch_pending;  // in serial order
struct WatchDone {
    std::uint64_t rec = ~0ull;
    std::uint32_t ranges = 0, changed = 0;
    std::uint64_t first_va = 0, first_bytes = 0;  // the first range that changed
    int first_kind = 0;
};
WatchDone g_watch_done[kDrawRecs];
std::uint64_t g_watch_draws = 0, g_watch_changed = 0, g_watch_spikes = 0, g_watch_spikes_changed = 0;

// The watched draws of submissions up to `serial`, now that it has finished.
void watch_settle_locked(std::uint64_t serial) {
    std::size_t k = 0;
    for (; k < g_watch_pending.size() && g_watch_pending[k].serial <= serial; ++k) {
        const WatchRec& w = g_watch_pending[k];
        WatchDone& d = g_watch_done[w.rec % kDrawRecs];
        d = WatchDone{};
        d.rec = w.rec;
        d.ranges = static_cast<std::uint32_t>(w.ranges.size());
        for (std::size_t i = 0; i < w.ranges.size(); ++i) {
            if (fnv(w.ranges[i].first, w.ranges[i].second) == w.hashes[i]) continue;
            const int kind = i < w.kinds.size() ? w.kinds[i] : 0;
            if (!d.changed) {
                d.first_va = w.ranges[i].first;
                d.first_bytes = w.ranges[i].second;
                d.first_kind = kind;
            }
            ++d.changed;
            ++g_watch_changed_kind[kind & 3 ? (kind == 2 ? 2 : 1) : 0];
        }
        if (d.changed) ++g_watch_changed;
    }
    g_watch_pending.erase(g_watch_pending.begin(), g_watch_pending.begin() + static_cast<std::ptrdiff_t>(k));
}

void collect_locked(Gpu::Slot& sl) {
    sl.cov_reset = false;
    if (!g_watch_pending.empty()) watch_settle_locked(sl.serial);
    if (sl.cov_recs.empty()) return;
    const auto n = static_cast<std::uint32_t>(sl.cov_recs.size());
    std::vector<std::uint64_t> res(static_cast<std::size_t>(n) * 2);
    const VkResult r = vkGetQueryPoolResults(g.device, sl.cov_pool, 0, n, res.size() * 8, res.data(), 16,
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (r == VK_SUCCESS || r == VK_NOT_READY) {
        for (std::uint32_t i = 0; i < n; ++i) {
            if (!res[static_cast<std::size_t>(i) * 2 + 1]) {
                ++g_unmeasured;
                continue;
            }
            const std::uint64_t rec = sl.cov_recs[i];
            Cov& c = g_cov[rec % kDrawRecs];
            c.rec = rec;
            c.samples = res[static_cast<std::size_t>(i) * 2];
            ++g_measured;
        }
    } else {
        g_unmeasured += n;
    }
    sl.cov_recs.clear();
}

bool coverage(std::uint64_t rec, std::uint64_t& out) {
    const Cov& c = g_cov[rec % kDrawRecs];
    if (c.rec != rec) return false;
    out = c.samples;
    return true;
}

// ---- frames ------------------------------------------------------------------

// Frames kept at full size, in flip order around. A frame is compared once
// the one after it has landed, a flip or two after it was recorded, and a
// catch writes it with both neighbours - so a handful is enough. Not many
// more: these are host-visible readbacks, 8 MiB each at 1080p, and past
// about 128 MiB of those the NVIDIA driver faults inside its copy path.
constexpr int kRing = 10;
constexpr std::uint32_t kThumbWidth = 64;  // the chain stops at the first level this narrow

struct Keys;

struct Frame {
    std::uint64_t flip = ~0ull;
    std::uint64_t serial = 0;              // the submission its copies went into
    std::uint64_t rec_lo = 0, rec_hi = 0;  // its draw records
    bool range_ok = false;                 // the flip before it was recorded too, so rec_lo is its first draw
    bool pending = false, landed = false, evaluated = false;
    std::uint32_t w = 0, h = 0, tw = 0, th = 0;
    DevBuffer thumb, full;
    std::vector<std::uint8_t> cells;  // tw * th RGBA, once landed
    double mean[3] = {};
    std::shared_ptr<const Keys> keys;  // its draws by identity, once landed (null: not all in the ring)
};
Frame g_frames[kRing];
int g_next = 0;
std::uint64_t g_last_flip = ~0ull, g_last_rec = 0;

struct Chain {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::uint32_t w = 0, h = 0, levels = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
};
Chain g_chain;

std::uint64_t g_watched = 0, g_compared = 0, g_skipped = 0, g_overrun = 0;
std::uint64_t g_catches = 0, g_image_catches = 0, g_spike_catches = 0, g_written = 0, g_written_big = 0, g_big = 0;
// Every catch, one line each (<dir>/catches.csv), flushed in batches.
std::string g_csv;
bool g_csv_started = false;

void csv_flush(bool all) {
    if (g_csv.empty() || (!all && g_csv.size() < 16384)) return;
    auto text = std::make_shared<std::string>(std::move(g_csv));
    g_csv.clear();
    const std::string path = cfg().dir + "/catches.csv";
    const bool first = !g_csv_started;
    g_csv_started = true;
    auto write = [text, path, first] {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
        if (FILE* f = std::fopen(path.c_str(), first ? "wb" : "ab")) {
            if (first) {
                std::fputs("flip,pair,cells,percent,mass,mean_before,mean,mean_after,box_x0,box_y0,box_x1,box_y1,draws,spikes,"
                           "spike_pipeline,spike_prim,spike_count,spike_samples,spike_viewport_percent,spike_before,spike_after,"
                           "spike_inputs_changed,written\n",
                           f);
            }
            std::fwrite(text->data(), 1, text->size(), f);
            std::fclose(f);
        }
    };
    if (all) write();
    else std::thread(write).detach();
}
std::uint64_t g_draws_seen = 0, g_draws_exact = 0, g_draws_loose = 0;
std::map<std::string, std::uint64_t> g_spikes_by_pipeline;

bool is_srgb(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        return true;
    default:
        return false;
    }
}

bool ensure_chain_locked(std::uint32_t w, std::uint32_t h, VkFormat format) {
    if (g_chain.image && g_chain.w == w && g_chain.h == h && g_chain.format == format) return true;
    if (g_chain.image) {
        {
            QueueGuard queue;
            vkDeviceWaitIdle(g.device);  // a size change: rare (a resolution option)
        }
        vkDestroyImage(g.device, g_chain.image, nullptr);
        vkFreeMemory(g.device, g_chain.memory, nullptr);
        g_chain = Chain{};
    }
    std::uint32_t levels = 1;
    while ((w >> (levels - 1)) > kThumbWidth && (h >> levels) > 0) ++levels;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {w, h, 1};
    ici.mipLevels = levels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g.device, &ici, nullptr, &g_chain.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(g.device, g_chain.image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, 0);
    MemOwner own_mem("glitch hunt");
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g.device, &mai, nullptr, &g_chain.memory) != VK_SUCCESS) {
        vkDestroyImage(g.device, g_chain.image, nullptr);
        g_chain = Chain{};
        return false;
    }
    vkBindImageMemory(g.device, g_chain.image, g_chain.memory, 0);
    g_chain.w = w;
    g_chain.h = h;
    g_chain.levels = levels;
    g_chain.format = format;
    return true;
}

bool ensure_buffer_locked(DevBuffer& b, std::uint64_t bytes) {
    if (b.buffer && b.size == bytes) return true;
    if (b.buffer) {
        vkDestroyBuffer(g.device, b.buffer, nullptr);
        vkFreeMemory(g.device, b.memory, nullptr);
        b = DevBuffer{};
    }
    MemOwner own_mem("glitch hunt");
    return create_dev_buffer(b, bytes, true, true);
}

// A submission serial has finished on the GPU: retired, or its fence signalled.
bool serial_done_locked(std::uint64_t serial) {
    if (g.completed_submits > serial) return true;
    for (const Gpu::Slot& sl : g.slots) {
        if (sl.in_flight && sl.serial == serial) {
            return serial < stream_submitted_serial() && vkGetFenceStatus(g.device, sl.fence) == VK_SUCCESS;  // submitted (bb-submit) first
        }
    }
    return false;  // not submitted yet
}

Frame* find(std::uint64_t flip, bool landed_only) {
    for (Frame& f : g_frames) {
        if (f.flip == flip && (f.landed || (!landed_only && f.pending))) return &f;
    }
    return nullptr;
}

// ---- comparing ---------------------------------------------------------------

double cell_diff(const std::uint8_t* a, const std::uint8_t* b) {
    return (std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]) + std::abs(a[2] - b[2])) / (3.0 * 255.0);
}

struct ImageScore {
    unsigned cells = 0;
    double mass = 0;
    std::uint32_t x0 = ~0u, y0 = ~0u, x1 = 0, y1 = 0;  // cell box
};

// `mid` (one or two frames) against the frames either side of it.
ImageScore score_image(const Frame& before, const std::vector<const Frame*>& mid, const Frame& after, double cell) {
    ImageScore s;
    const std::uint32_t tw = before.tw, th = before.th;
    for (const Frame* m : mid) {
        if (m->tw != tw || m->th != th) return s;
    }
    if (after.tw != tw || after.th != th) return s;
    for (std::uint32_t y = 0; y < th; ++y) {
        for (std::uint32_t x = 0; x < tw; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * tw + x) * 4;
            const double c = cell_diff(&before.cells[i], &after.cells[i]);
            double m = 1e9;
            for (const Frame* f : mid) {
                m = std::min(m, cell_diff(&f->cells[i], &before.cells[i]));
                m = std::min(m, cell_diff(&f->cells[i], &after.cells[i]));
            }
            if (m > cell && c < 0.5 * m) {
                ++s.cells;
                s.mass += m - c;
                s.x0 = std::min(s.x0, x);
                s.y0 = std::min(s.y0, y);
                s.x1 = std::max(s.x1, x);
                s.y1 = std::max(s.y1, y);
            }
        }
    }
    return s;
}

std::uint64_t mix(std::uint64_t h, std::uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

std::uint64_t name_hash(const char* s) {
    std::uint64_t h = 1469598103934665603ull;
    for (; *s; ++s) h = (h ^ static_cast<std::uint8_t>(*s)) * 1099511628211ull;
    return h;
}

// A frame's draws are all still in the ring.
bool draws_available(const Frame& f) {
    return f.range_ok && f.rec_hi >= f.rec_lo && g_draw_rec_next - f.rec_lo <= kDrawRecs;
}

// The frame's draws by identity: exact (pipeline, target, index buffer, vertex
// binding, count, instances, k-th) and loose (pipeline, target, k-th).
struct Keys {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> exact, loose;  // per draw, in order
    std::unordered_map<std::uint64_t, std::uint64_t> cov_exact, cov_loose;
};

Keys keys_of(const Frame& f) {
    Keys k;
    std::unordered_map<std::uint64_t, std::uint32_t> occ_exact, occ_loose;
    const std::size_t n = static_cast<std::size_t>(f.rec_hi - f.rec_lo);
    k.exact.reserve(n);
    k.loose.reserve(n);
    for (std::uint64_t r = f.rec_lo; r < f.rec_hi; ++r) {
        const DrawRec& d = g_draw_recs[r % kDrawRecs];
        const std::uint64_t base = mix(name_hash(d.name), d.rt0);
        std::uint64_t ex = mix(mix(mix(mix(base, d.index_va), d.vb[0]), d.count), d.inst);
        ex = mix(ex, occ_exact[ex]++);
        const std::uint64_t lo = mix(base, occ_loose[base]++);
        k.exact.emplace_back(ex, r);
        k.loose.emplace_back(lo, r);
        std::uint64_t c = 0;
        if (coverage(r, c)) {
            k.cov_exact[ex] = c;
            k.cov_loose[lo] = c;
        }
    }
    return k;
}

struct Spike {
    std::uint64_t rec = 0, cov = 0, before = 0, after = 0;
    bool exact = false;
    double frac = 0;
};

std::string draw_line(std::uint64_t r) {
    const DrawRec& d = g_draw_recs[r % kDrawRecs];
    char line[640];
    std::snprintf(line, sizeof(line),
                  "draw[%llu] %s prim=%u n=%u inst=%u%s vp=(%.0f,%.0f %.0fx%.0f) rt0=0x%llx depth=0x%llx dctl=%08x blend=%08x "
                  "idx=0x%llx/%s base=%d vb=0x%llx+%u,0x%llx+%u indirect=0x%llx tex=0x%llx,0x%llx%s",
                  static_cast<ull>(r), d.name, d.prim, d.count, d.inst, d.indexed ? " indexed" : "", d.vp[0], d.vp[1], d.vp[2],
                  d.vp[3], static_cast<ull>(d.rt0), static_cast<ull>(d.depth), d.depth_ctl, d.blend0, static_cast<ull>(d.index_va),
                  d.index_type == 1 ? "32" : "16", d.base_vertex, static_cast<ull>(d.vb[0]), d.vb_stride[0], static_cast<ull>(d.vb[1]),
                  d.vb_stride[1], static_cast<ull>(d.indirect_va), static_cast<ull>(d.tex[0]), static_cast<ull>(d.tex[1]),
                  d.dummies ? (" DUMMY IMAGES " + std::to_string(d.dummies)).c_str() : "");
    return line;
}

struct Pixels {
    std::string path;
    std::uint32_t w = 0, h = 0;
    std::vector<std::uint8_t> rgb;
};

Pixels take_pixels(const Frame& f, const std::string& path) {
    Pixels p;
    p.path = path;
    p.w = f.w;
    p.h = f.h;
    if (!f.full.map) return p;
    const auto* src = static_cast<const std::uint8_t*>(f.full.map);
    const std::size_t n = static_cast<std::size_t>(f.w) * f.h;
    p.rgb.resize(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        p.rgb[i * 3] = src[i * 4];
        p.rgb[i * 3 + 1] = src[i * 4 + 1];
        p.rgb[i * 3 + 2] = src[i * 4 + 2];
    }
    return p;
}

void write_ppm(const Pixels& p) {
    if (p.rgb.empty()) return;
    if (FILE* f = std::fopen(p.path.c_str(), "wb")) {
        std::fprintf(f, "P6\n%u %u\n255\n", p.w, p.h);
        std::fwrite(p.rgb.data(), 1, p.rgb.size(), f);
        std::fclose(f);
    }
}

// Compares frame `x` with its neighbours; `x2`, when given, is the frame after
// it and the pair is compared as one event.
void evaluate_locked(const Frame& before, const Frame& x, const Frame* x2, const Frame& after) {
    const Config& c = cfg();
    ++g_compared;
    std::vector<const Frame*> mid{&x};
    if (x2) mid.push_back(x2);
    const ImageScore img = score_image(before, mid, after, c.cell);
    const bool image_catch = img.cells >= c.cells;

    // The draws, for the single-frame comparison (a pair has no draw
    // neighbours to speak of, the image says enough).
    std::vector<Spike> spikes;
    std::size_t matched_exact = 0, matched_loose = 0, draws = 0;
    const bool with_draws = !x2 && before.keys && x.keys && after.keys;
    static const Keys kNone;
    const Keys& kb = with_draws ? *before.keys : kNone;
    const Keys& ka = with_draws ? *after.keys : kNone;
    const Keys& kx = with_draws ? *x.keys : kNone;
    if (with_draws) {
        draws = kx.exact.size();
        for (std::size_t i = 0; i < kx.exact.size(); ++i) {
            const std::uint64_t r = kx.exact[i].second;
            std::uint64_t cov = 0;
            if (!coverage(r, cov)) continue;
            std::uint64_t b = 0, a = 0;
            bool exact = false;
            auto eb = kb.cov_exact.find(kx.exact[i].first), ea = ka.cov_exact.find(kx.exact[i].first);
            if (eb != kb.cov_exact.end() && ea != ka.cov_exact.end()) {
                b = eb->second;
                a = ea->second;
                exact = true;
                ++matched_exact;
            } else {
                auto lb = kb.cov_loose.find(kx.loose[i].first), la = ka.cov_loose.find(kx.loose[i].first);
                if (lb == kb.cov_loose.end() || la == ka.cov_loose.end()) continue;
                b = lb->second;
                a = la->second;
                ++matched_loose;
            }
            const DrawRec& d = g_draw_recs[r % kDrawRecs];
            const double area = std::max(1.0, static_cast<double>(std::fabs(d.vp[2]) * std::fabs(d.vp[3])));
            const double frac = cov / area;
            // Only the same draw: the k-th draw of a pipeline into a target is
            // a different object whenever culling changed what came before
            // it, and those matches jumped all the time.
            if (exact && frac >= c.cov_min && static_cast<double>(cov) > c.ratio * static_cast<double>(std::max(a, b)) + 64.0) {
                spikes.push_back({r, cov, b, a, exact, frac});
                if (g_watch_done[r % kDrawRecs].rec == r) {
                    ++g_watch_spikes;
                    if (g_watch_done[r % kDrawRecs].changed) ++g_watch_spikes_changed;
                }
            }
        }
        g_draws_seen += draws;
        g_draws_exact += matched_exact;
        g_draws_loose += matched_loose;
    }
    std::sort(spikes.begin(), spikes.end(), [](const Spike& p, const Spike& q) {
        return p.cov - std::max(p.before, p.after) > q.cov - std::max(q.before, q.after);
    });
    if (!image_catch && spikes.empty()) return;

    ++g_catches;
    if (image_catch) ++g_image_catches;
    if (!spikes.empty()) ++g_spike_catches;
    for (const Spike& s : spikes) ++g_spikes_by_pipeline[g_draw_recs[s.rec % kDrawRecs].name];

    const double cw = static_cast<double>(x.w) / x.tw, ch = static_cast<double>(x.h) / x.th;
    auto mean = [](const Frame& f) { return (f.mean[0] + f.mean[1] + f.mean[2]) / 3; };
    char head[512];
    std::snprintf(head, sizeof(head),
                  "flip %llu%s: %u cells unlike both neighbours (%.1f%% of the frame, mass %.2f); mean %.3f -> %.3f%s -> %.3f; "
                  "%zu draw spikes of %zu draws (%zu matched exactly, %zu by pipeline and target)%s",
                  static_cast<ull>(x.flip), x2 ? " and the next" : "", img.cells, 100.0 * img.cells / (x.tw * x.th), img.mass,
                  mean(before), mean(x), x2 ? (" " + std::to_string(mean(*x2)).substr(0, 5)).c_str() : "", mean(after),
                  spikes.size(), draws, matched_exact, matched_loose, with_draws ? "" : " (draws not compared)");
    std::string box;
    if (img.cells) {
        char b[128];
        std::snprintf(b, sizeof(b), "; changed box %.0f,%.0f-%.0f,%.0f px", img.x0 * cw, img.y0 * ch, (img.x1 + 1) * cw,
                      (img.y1 + 1) * ch);
        box = b;
    }
    // Big events get a budget of their own, so a run's first few dozen small
    // ones (sparks crossing the screen are one-frame events too) cannot use up
    // the writing before they come.
    std::uint64_t top_spike = 0;
    for (const Spike& sp : spikes) top_spike = std::max<std::uint64_t>(top_spike, static_cast<std::uint64_t>(sp.frac * 1000));
    const bool big = img.cells * 100 >= 3 * x.tw * x.th || top_spike >= 250;
    if (big) ++g_big;
    const bool write = big ? g_written_big < c.max_big : g_written < c.max_catches;
    const std::string dir = c.dir + "/" + std::to_string(x.flip);
    static std::atomic<int> lines{0};
    if (lines.fetch_add(1) < 400) {
        host_log("glitch: %s%s%s", head, box.c_str(), write ? (" -> " + dir).c_str() : "");
        for (std::size_t i = 0; i < spikes.size() && i < 3; ++i) {
            const Spike& s = spikes[i];
            const DrawRec& d = g_draw_recs[s.rec % kDrawRecs];
            const WatchDone& wd = g_watch_done[s.rec % kDrawRecs];
            char watched[160] = "";
            if (wd.rec == s.rec) {
                static const char* const kKinds[3] = {"a buffer binding", "the index buffer", "the LS fetch"};
                std::snprintf(watched, sizeof(watched), "; watched: %u of its %u input ranges changed while in flight%s%s%s", wd.changed,
                              wd.ranges, wd.changed ? " (first " : " (none)", wd.changed ? kKinds[wd.first_kind % 3] : "",
                              wd.changed ? ")" : "");
            }
            host_log("glitch:   spike draw[%llu] %s rt0=0x%llx prim=%u n=%u inst=%u: %llu samples (%.1f%% of its viewport) "
                     "against %llu before and %llu after (%s)%s",
                     static_cast<ull>(s.rec), d.name, static_cast<ull>(d.rt0), d.prim, d.count, d.inst, static_cast<ull>(s.cov),
                     100.0 * s.frac, static_cast<ull>(s.before), static_cast<ull>(s.after), s.exact ? "same draw" : "same pipeline and target",
                     watched);
        }
    }
    {
        auto mean_of = [](const Frame& f) { return (f.mean[0] + f.mean[1] + f.mean[2]) / 3; };
        char row[640];
        const Spike* sp = spikes.empty() ? nullptr : &spikes[0];
        const DrawRec* sd = sp ? &g_draw_recs[sp->rec % kDrawRecs] : nullptr;
        const WatchDone* wd = sp && g_watch_done[sp->rec % kDrawRecs].rec == sp->rec ? &g_watch_done[sp->rec % kDrawRecs] : nullptr;
        std::snprintf(row, sizeof(row), "%llu,%d,%u,%.2f,%.2f,%.4f,%.4f,%.4f,%.0f,%.0f,%.0f,%.0f,%zu,%zu,%s,%u,%u,%llu,%.1f,%llu,%llu,%d,%d\n",
                      static_cast<ull>(x.flip), x2 ? 1 : 0, img.cells, 100.0 * img.cells / (x.tw * x.th), img.mass, mean_of(before),
                      mean_of(x), mean_of(after), img.cells ? img.x0 * cw : 0.0, img.cells ? img.y0 * ch : 0.0,
                      img.cells ? (img.x1 + 1) * cw : 0.0, img.cells ? (img.y1 + 1) * ch : 0.0, draws, spikes.size(), sd ? sd->name : "",
                      sd ? sd->prim : 0, sd ? sd->count : 0, static_cast<ull>(sp ? sp->cov : 0), sp ? 100.0 * sp->frac : 0.0,
                      static_cast<ull>(sp ? sp->before : 0), static_cast<ull>(sp ? sp->after : 0), wd ? static_cast<int>(wd->changed) : -1,
                      write ? 1 : 0);
        g_csv += row;
        csv_flush(false);
    }
    if (!write) return;
    if (big) ++g_written_big;
    else ++g_written;

    // The report, while the ring still holds the draws; the files on a
    // thread of their own.
    auto text = std::make_shared<std::string>();
    *text += std::string("glitch at ") + head + box + "\n\n";
    *text += "spikes (coverage in samples: this frame, the frame before, the frame after):\n";
    for (const Spike& s : spikes) {
        char l[160];
        std::snprintf(l, sizeof(l), "  %llu (%.1f%% of the viewport) / %llu / %llu %s  ", static_cast<ull>(s.cov), 100.0 * s.frac,
                      static_cast<ull>(s.before), static_cast<ull>(s.after), s.exact ? "exact" : "loose");
        *text += l + draw_line(s.rec) + "\n";
    }
    // A watched draw's inputs, against the same draw's in the frames either
    // side (BBHOST_GLITCH_WATCH): every spike, and every watched draw whose
    // coverage is less than half of what it was both before and after (a
    // quad collapsed rather than stretched).
    if (with_draws) {
        std::vector<Spike> shown = spikes;
        for (std::size_t i = 0; i < kx.exact.size() && shown.size() < 24; ++i) {
            const std::uint64_t r = kx.exact[i].second;
            if (g_watch_texts[r % kWatchTexts].rec != r) continue;
            std::uint64_t cov = 0;
            if (!coverage(r, cov)) continue;
            const auto b = kb.cov_exact.find(kx.exact[i].first), a = ka.cov_exact.find(kx.exact[i].first);
            if (b == kb.cov_exact.end() || a == ka.cov_exact.end()) continue;
            if (cov * 2 < std::min(b->second, a->second)) shown.push_back({r, cov, b->second, a->second, true, 0});
        }
        for (const Spike& sp : shown) {
            const WatchText& mine = g_watch_texts[sp.rec % kWatchTexts];
            if (mine.rec != sp.rec) continue;
            std::uint64_t key = 0;
            for (const auto& [k, r] : kx.exact) {
                if (r == sp.rec) key = k;
            }
            auto same_in = [&](const Keys& n) -> const WatchText* {
                for (const auto& [k, r] : n.exact) {
                    if (k == key && g_watch_texts[r % kWatchTexts].rec == r) return &g_watch_texts[r % kWatchTexts];
                }
                return nullptr;
            };
            const WatchText* b = same_in(kb);
            const WatchText* a = same_in(ka);
            *text += "\ninputs of draw[" + std::to_string(sp.rec) + "] (" + std::to_string(sp.cov) + " samples against " +
                     std::to_string(sp.before) + " and " + std::to_string(sp.after) + "):\n" + mine.text;
            *text += "the same draw the frame before" + (b ? " (draw[" + std::to_string(b->rec) + "]):\n" + b->text : std::string(": not kept\n"));
            *text += "the same draw the frame after" + (a ? " (draw[" + std::to_string(a->rec) + "]):\n" + a->text : std::string(": not kept\n"));
        }
    }
    if (with_draws) {
        // What the frame lacks that both neighbours drew, and what only it
        // drew: a pass that skipped a draw for a frame, or one that came in.
        std::unordered_map<std::uint64_t, std::uint64_t> in_x, in_before, in_after;
        for (const auto& [key, rec] : kx.exact) in_x.emplace(key, rec);
        for (const auto& [key, rec] : kb.exact) in_before.emplace(key, rec);
        for (const auto& [key, rec] : ka.exact) in_after.emplace(key, rec);
        std::string missing, only;
        std::size_t n_missing = 0, n_only = 0;
        for (const auto& [key, rec] : kb.exact) {
            if (in_x.count(key) || !in_after.count(key)) continue;
            if (++n_missing <= 200) missing += "  " + draw_line(rec) + "\n";
        }
        for (const auto& [key, rec] : kx.exact) {
            if (in_before.count(key) || in_after.count(key)) continue;
            if (++n_only <= 200) only += "  " + draw_line(rec) + "\n";
        }
        std::uint32_t dummies = 0;
        for (const auto& [key, rec] : kx.exact) dummies += g_draw_recs[rec % kDrawRecs].dummies;
        *text += "\n" + std::to_string(n_missing) + " draws both neighbours made that this frame did not (as drawn the frame before):\n" +
                 missing;
        *text += "\n" + std::to_string(n_only) + " draws only this frame made:\n" + only;
        *text += "\n" + std::to_string(dummies) + " images this frame's draws read resolved to nothing (a dummy bound instead)\n";
        *text += "\nevery draw of the frame (coverage / before / after; ~ matched by pipeline and target only, - unmatched "
                 "or unmeasured):\n";
        for (std::size_t i = 0; i < kx.exact.size(); ++i) {
            const std::uint64_t r = kx.exact[i].second;
            std::uint64_t cov = 0;
            const bool have = coverage(r, cov);
            auto num = [](const std::unordered_map<std::uint64_t, std::uint64_t>& m, std::uint64_t k, std::uint64_t k2,
                          const std::unordered_map<std::uint64_t, std::uint64_t>& m2) {
                if (auto it = m.find(k); it != m.end()) return std::to_string(it->second);
                if (auto it = m2.find(k2); it != m2.end()) return std::to_string(it->second) + "~";
                return std::string("-");
            };
            const std::string b = num(kb.cov_exact, kx.exact[i].first, kx.loose[i].first, kb.cov_loose);
            const std::string a = num(ka.cov_exact, kx.exact[i].first, kx.loose[i].first, ka.cov_loose);
            char l[128];
            std::snprintf(l, sizeof(l), "  %10s %10s %10s  ", have ? std::to_string(cov).c_str() : "-", b.c_str(), a.c_str());
            *text += l + draw_line(r) + "\n";
        }
    }
    auto frames = std::make_shared<std::vector<Pixels>>();
    frames->push_back(take_pixels(before, dir + "/" + std::to_string(before.flip) + "-before.ppm"));
    frames->push_back(take_pixels(x, dir + "/" + std::to_string(x.flip) + "-caught.ppm"));
    if (x2) frames->push_back(take_pixels(*x2, dir + "/" + std::to_string(x2->flip) + "-caught2.ppm"));
    frames->push_back(take_pixels(after, dir + "/" + std::to_string(after.flip) + "-after.ppm"));
    const std::string report = dir + "/report.txt";
    std::thread([dir, report, text, frames] {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (FILE* f = std::fopen(report.c_str(), "wb")) {
            std::fwrite(text->data(), 1, text->size(), f);
            std::fclose(f);
        }
        for (const Pixels& p : *frames) write_ppm(p);
    }).detach();
}

// The frames that have landed, oldest first, each compared once its next has.
void evaluate_ready_locked() {
    for (;;) {
        Frame* x = nullptr;
        for (Frame& f : g_frames) {
            if (f.landed && !f.evaluated && (!x || f.flip < x->flip)) x = &f;
        }
        if (!x) return;
        Frame* next = find(x->flip + 1, true);
        if (!next) {
            if (find(x->flip + 1, false)) return;  // recorded, still on the GPU
            // Never recorded (no free slot that flip): nothing comes after
            // this one to compare it with, once anything later exists.
            bool later = false;
            for (const Frame& f : g_frames) later |= (f.landed || f.pending) && f.flip > x->flip + 1 && f.flip != ~0ull;
            if (!later) return;
            x->evaluated = true;
            ++g_skipped;
            continue;
        }
        x->evaluated = true;
        if (Frame* before = find(x->flip - 1, true)) {
            evaluate_locked(*before, *x, nullptr, *next);
            // The pair (before, x) against the frame before it and `next`.
            if (Frame* before2 = find(x->flip - 2, true)) {
                const ImageScore pair = score_image(*before2, {before, x}, *next, cfg().cell);
                const ImageScore one_a = score_image(*before2, {before}, *x, cfg().cell);
                const ImageScore one_b = score_image(*before, {x}, *next, cfg().cell);
                // Only where neither frame alone explains it.
                if (pair.cells >= cfg().cells && one_a.cells < cfg().cells && one_b.cells < cfg().cells) {
                    evaluate_locked(*before2, *before, x, *next);
                }
            }
        } else {
            ++g_skipped;
        }
    }
}

void land_locked(Frame& f) {
    f.pending = false;
    f.landed = true;
    f.evaluated = false;
    f.keys.reset();
    const std::size_t n = static_cast<std::size_t>(f.tw) * f.th * 4;
    f.cells.assign(static_cast<const std::uint8_t*>(f.thumb.map), static_cast<const std::uint8_t*>(f.thumb.map) + n);
    double sum[3] = {};
    for (std::size_t i = 0; i < n; i += 4) {
        for (int k = 0; k < 3; ++k) sum[k] += f.cells[i + static_cast<std::size_t>(k)];
    }
    for (int k = 0; k < 3; ++k) f.mean[k] = sum[k] / (255.0 * (static_cast<double>(n) / 4));
}

void image_barrier(VkCommandBuffer cmd, VkImage image, std::uint32_t level, std::uint32_t count, VkImageLayout from, VkImageLayout to,
                   VkAccessFlags src_access, VkAccessFlags dst_access, VkPipelineStageFlags src_stage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, count, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// The chain over `src`: level 0 its pixels as RGBA8 (a blit converts the
// format), each level below the 2x2 average of the one above, the last
// copied to `thumb` and level 0 to `full` (when given).
void run_chain_locked(VkCommandBuffer cmd, VkImage src, std::uint32_t w, std::uint32_t h, const DevBuffer& thumb,
                      const DevBuffer* full) {
    const std::uint32_t last = g_chain.levels - 1;
    // Everything written to the source so far, and the chain's last reads,
    // before this.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = g_chain.image;
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, g_chain.levels, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 1, &ib);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {static_cast<std::int32_t>(w), static_cast<std::int32_t>(h), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = blit.srcOffsets[1];
    vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, g_chain.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    for (std::uint32_t lv = 1; lv < g_chain.levels; ++lv) {
        image_barrier(cmd, g_chain.image, lv - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkImageBlit down{};
        down.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv - 1, 0, 1};
        down.srcOffsets[1] = {static_cast<std::int32_t>(std::max(1u, w >> (lv - 1))), static_cast<std::int32_t>(std::max(1u, h >> (lv - 1))),
                              1};
        down.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv, 0, 1};
        down.dstOffsets[1] = {static_cast<std::int32_t>(std::max(1u, w >> lv)), static_cast<std::int32_t>(std::max(1u, h >> lv)), 1};
        vkCmdBlitImage(cmd, g_chain.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_chain.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                       &down, VK_FILTER_LINEAR);
    }
    image_barrier(cmd, g_chain.image, last, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    if (full) {
        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(cmd, g_chain.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, full->buffer, 1, &c);
    }
    VkBufferImageCopy small{};
    small.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, last, 0, 1};
    small.imageExtent = {std::max(1u, w >> last), std::max(1u, h >> last), 1};
    vkCmdCopyImageToBuffer(cmd, g_chain.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, thumb.buffer, 1, &small);
    // Later writes to the source (the game's next frame) wait for these reads.
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
}

// This flip's copies of the display buffer.
void record_locked(std::uint64_t display_va, std::uint64_t flip, std::uint64_t mark) {
    RtImage* rt = find_render_target(display_va);
    if (!rt || !rt->initialised || rt->depth || !rt->image || !rt->width || !rt->height) return;
    const VkImage image = rt->image;
    const VkFormat format = rt->format;
    const std::uint32_t w = rt->width, h = rt->height;
    static std::vector<std::uint64_t> seen;
    if (std::find(seen.begin(), seen.end(), display_va) == seen.end() && seen.size() < 8) {
        seen.push_back(display_va);
        host_log("glitch: display buffer 0x%llx, %ux%u format %d", static_cast<ull>(display_va), w, h, static_cast<int>(format));
    }
    Frame& f = g_frames[g_next];
    if (f.pending) {  // the GPU is further behind than the ring is deep
        ++g_overrun;
        return;
    }
    if (f.landed && !f.evaluated) ++g_overrun;
    if (!ensure_chain_locked(w, h, is_srgb(format) ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM)) return;
    const std::uint32_t last = g_chain.levels - 1;
    const std::uint32_t tw = std::max(1u, w >> last), th = std::max(1u, h >> last);
    const std::uint64_t thumb_bytes = static_cast<std::uint64_t>(tw) * th * 4;
    if (!ensure_buffer_locked(f.thumb, thumb_bytes) || !ensure_buffer_locked(f.full, static_cast<std::uint64_t>(w) * h * 4)) return;
    g_next = (g_next + 1) % kRing;
    begin_recording_locked();
    transfer_flush_locked();
    render_end_pass_locked();
    view_clears_flush_locked(kVcwDump);  // the display it reads holds every clear made so far
    run_chain_locked(g_cmd(), image, w, h, f.thumb, &f.full);
    f.flip = flip;
    f.serial = g.flushes;
    // The frame's draws: those recorded between the flip before it and this
    // one reaching the command processor. Without a mark (no flip went
    // through it), up to now.
    const std::uint64_t hi = mark ? mark : g_draw_rec_next;
    f.range_ok = g_last_flip != ~0ull && g_last_flip + 1 == flip && hi >= g_last_rec;
    f.rec_lo = g_last_rec;
    f.rec_hi = hi;
    f.w = w;
    f.h = h;
    f.tw = tw;
    f.th = th;
    f.pending = true;
    f.landed = false;
    f.evaluated = false;
    g_last_flip = flip;
    g_last_rec = hi;
    ++g_watched;
}

}  // namespace

bool glitch_on() { return cfg().on; }

VkQueryControlFlags glitch_query_flags() { return g_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0; }

void glitch_device_features(VkPhysicalDeviceFeatures& enable) {
    if (!glitch_on()) return;
    VkPhysicalDeviceFeatures sup{};
    vkGetPhysicalDeviceFeatures(g.phys, &sup);
    g_precise = sup.occlusionQueryPrecise == VK_TRUE;
    enable.occlusionQueryPrecise = sup.occlusionQueryPrecise;
    host_log("glitch: on (BBHOST_GLITCH=1): every frame compared with its neighbours, every draw's coverage counted%s; catches "
             "go to %s",
             g_precise ? "" : " (not precisely: the device has no occlusionQueryPrecise)", cfg().dir.c_str());
}

void glitch_slot_init_locked(Gpu::Slot& sl) {
    if (!glitch_on()) return;
    VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qci.queryType = VK_QUERY_TYPE_OCCLUSION;
    qci.queryCount = kCovQueries;
    if (vkCreateQueryPool(g.device, &qci, nullptr, &sl.cov_pool) != VK_SUCCESS) sl.cov_pool = VK_NULL_HANDLE;
}

void glitch_begin_recording_locked() {
    if (!glitch_on()) return;
    Gpu::Slot& sl = g.slots[g.slot];
    if (!sl.cov_pool) return;
    rec().reset_query_pool(sl.cov_pool, 0, kCovQueries);
    sl.cov_recs.clear();
    sl.cov_reset = true;
}

void glitch_draw_locked(DrawCall& call) {
    if (!glitch_on()) return;
    if (g_watch_now) {
        g_watch_now = false;
        WatchRec w;
        w.rec = g_draw_rec_next;
        w.serial = g.flushes;  // the recording under way becomes this submission
        w.ranges = std::move(g_watch_ranges);
        w.kinds = std::move(g_watch_kinds);
        g_watch_ranges.clear();
        g_watch_kinds.clear();
        w.hashes.reserve(w.ranges.size());
        for (const auto& [va, bytes] : w.ranges) w.hashes.push_back(fnv(va, bytes));
        g_watch_pending.push_back(std::move(w));
        ++g_watch_draws;
        WatchText& t = g_watch_texts[g_draw_rec_next % kWatchTexts];
        t.rec = g_draw_rec_next;
        t.text = std::move(g_watch_text);
        g_watch_text.clear();
    }
    Gpu::Slot& sl = g.slots[g.slot];
    if (!sl.cov_pool || !sl.cov_reset) return;
    if (sl.cov_recs.size() >= kCovQueries) {
        ++g_pool_full;
        return;
    }
    call.query_pool = sl.cov_pool;
    call.query = static_cast<std::uint32_t>(sl.cov_recs.size());
    call.query_flags = glitch_query_flags();
    sl.cov_recs.push_back(g_draw_rec_next);
}

void glitch_slot_done_locked(Gpu::Slot& sl) {
    if (!glitch_on()) return;
    collect_locked(sl);
}

void glitch_watch_draw_locked(const std::string& pipeline) {
    g_watch_now = false;
    g_watch_ranges.clear();
    g_watch_kinds.clear();
    g_watch_text.clear();
    if (!glitch_on() || g_watch_prefixes.empty()) return;
    for (const std::string& p : g_watch_prefixes) {
        if (pipeline.compare(0, p.size(), p) == 0) {
            g_watch_now = true;
            return;
        }
    }
}

bool glitch_watching_draw() { return g_watch_now; }
bool glitch_snapshot_draw() { return g_watch_now && g_watch_snapshot; }

void glitch_watch_read_locked(std::uint64_t base, std::uint64_t bytes, int kind) {
    if (!g_watch_now || !base || !bytes || bytes > (64ull << 20) || g_watch_ranges.size() >= 64) return;
    g_watch_ranges.emplace_back(base, bytes);
    g_watch_kinds.push_back(kind);
}

void glitch_watch_note_locked(const char* fmt, ...) {
    if (!g_watch_now || g_watch_text.size() > 16384) return;
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    g_watch_text += line;
}

void glitch_watch_params_locked(const gcn::StageParams& params, int stage) {
    if (!g_watch_now) return;
    std::string& t = g_watch_text;
    char line[512];
    std::snprintf(line, sizeof(line), "  %s params: cb_valid %08x, user", stage ? "PS" : "VS", params.cb_valid);
    t += line;
    for (int k = 0; k < 16; ++k) {
        std::snprintf(line, sizeof(line), " %08x", params.user_sgpr[k]);
        t += line;
    }
    t += "\n    bias/stride/w3:";
    for (int k = 0; k < 8; ++k) {
        if (!((params.cb_valid >> k) & 1)) continue;
        std::snprintf(line, sizeof(line), " [%d] %u/%u/%08x", k, params.cb_bias_dw[k], params.cb_stride[k], params.cb_w3[k]);
        t += line;
    }
    t += "\n";
    // Every user-data pair that points at mapped memory: its first 16 dwords
    // (the fetch shader's V# table is one), and for each V# there, the start
    // of the buffer it describes.
    for (int k = 0; k + 1 < 16; k += 2) {
        const std::uint64_t ptr = static_cast<std::uint64_t>(params.user_sgpr[k]) | (static_cast<std::uint64_t>(params.user_sgpr[k + 1] & 0xffff) << 32);
        if (!ptr || !hle_kernel_va_mapped(ptr, 64)) continue;
        const auto* u = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(ptr));
        std::snprintf(line, sizeof(line), "    s[%d:%d] -> 0x%llx:", k, k + 1, static_cast<unsigned long long>(ptr));
        t += line;
        for (int q = 0; q < 16; ++q) {
            std::snprintf(line, sizeof(line), " %08x", u[q]);
            t += line;
        }
        t += "\n";
        for (int v = 0; v < 4; ++v) {
            const std::uint32_t* w = u + v * 4;
            const std::uint64_t base = static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xfff) << 32);
            const std::uint32_t stride = (w[1] >> 16) & 0x3fff;
            if (!base || !stride || !w[2] || !hle_kernel_va_mapped(base, 64)) continue;
            const auto* f = reinterpret_cast<const float*>(static_cast<std::uintptr_t>(base));
            std::snprintf(line, sizeof(line), "      as V# %d: base 0x%llx (%llu mod 16) stride %u records %u: %g %g %g %g | %g %g %g %g\n", v,
                          static_cast<unsigned long long>(base), static_cast<unsigned long long>(base & 15), stride, w[2], f[0], f[1],
                          f[2], f[3], f[4], f[5], f[6], f[7]);
            t += line;
        }
    }
}

std::string glitch_report() {
    csv_flush(true);
    std::string pipes;
    std::vector<std::pair<std::uint64_t, std::string>> by;
    for (const auto& [name, n] : g_spikes_by_pipeline) by.emplace_back(n, name);
    std::sort(by.rbegin(), by.rend());
    for (std::size_t i = 0; i < by.size() && i < 12; ++i) {
        pipes += (i ? ", " : "") + by[i].second + " x" + std::to_string(by[i].first);
    }
    char line[1024];
    std::snprintf(line, sizeof(line),
                  "glitch: %llu frames watched, %llu compared (%llu without both neighbours, %llu readbacks the GPU was too far "
                  "behind for); %llu caught - %llu one-off in the picture, %llu with a draw whose coverage jumped - %llu written (%llu of the %llu big ones); "
                  "%llu draws measured (%llu without a result, %llu past a pool), of the compared frames' %llu draws %llu matched "
                  "exactly and %llu by pipeline and target; watched draws %llu, inputs changed in flight in %llu (ranges: %llu buffer "
                  "bindings, %llu index buffers, %llu LS fetches), of their spikes %llu with changed inputs %llu; spikes by pipeline: %s",
                  static_cast<ull>(g_watched), static_cast<ull>(g_compared), static_cast<ull>(g_skipped), static_cast<ull>(g_overrun),
                  static_cast<ull>(g_catches), static_cast<ull>(g_image_catches), static_cast<ull>(g_spike_catches),
                  static_cast<ull>(g_written + g_written_big), static_cast<ull>(g_written_big), static_cast<ull>(g_big),
                  static_cast<ull>(g_measured), static_cast<ull>(g_unmeasured),
                  static_cast<ull>(g_pool_full), static_cast<ull>(g_draws_seen), static_cast<ull>(g_draws_exact),
                  static_cast<ull>(g_draws_loose), static_cast<ull>(g_watch_draws), static_cast<ull>(g_watch_changed),
                  static_cast<ull>(g_watch_changed_kind[0]), static_cast<ull>(g_watch_changed_kind[1]),
                  static_cast<ull>(g_watch_changed_kind[2]), static_cast<ull>(g_watch_spikes),
                  static_cast<ull>(g_watch_spikes_changed), pipes.empty() ? "none" : pipes.c_str());
    return line;
}

}  // namespace gpu

using namespace gpu;

std::uint64_t host_gpu_draw_mark() { return g_draw_rec_next; }

void host_gpu_glitch_watch(std::uint64_t display_va, std::uint64_t mark) {
    if (!glitch_on() || !display_va) return;
    std::lock_guard<GpuMutex> lock(g.mu);
    if (!g.ok) return;
    // What has landed first, then the coverage of every submission that has
    // finished - which includes all those the landed frames' draws went into.
    Frame* landed[kRing];
    int n = 0;
    for (Frame& f : g_frames) {
        if (f.pending && f.thumb.map && serial_done_locked(f.serial)) {
            land_locked(f);
            landed[n++] = &f;
        }
    }
    for (Gpu::Slot& sl : g.slots) {
        if (sl.in_flight && (!sl.cov_recs.empty() || !g_watch_pending.empty()) && sl.serial < stream_submitted_serial() &&
            vkGetFenceStatus(g.device, sl.fence) == VK_SUCCESS) {
            collect_locked(sl);
        }
    }
    for (int i = 0; i < n; ++i) {
        if (draws_available(*landed[i])) landed[i]->keys = std::make_shared<const Keys>(keys_of(*landed[i]));
    }
    evaluate_ready_locked();
    record_locked(display_va, hle_video_flip_count(), mark);
}
