// BBHOST_COPY_CENSUS=1: the frame's image copies and what happens to their
// sources and destinations (host/copy_census.h).
#include "host/copy_census.h"

#include "host/gpu_internal.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace gpu {
namespace copy_census {

const bool g_on = [] {
    const char* e = std::getenv("BBHOST_COPY_CENSUS");
    return e && e[0] == '1';
}();

namespace {
thread_local int t_quiet = 0;
thread_local int t_as = kTouchOther;

constexpr int kNextCopy = kTouches;  // a close reason past the touches: the next copy into the destination
constexpr int kReasons = kTouches + 1;
const char* const kKindName[kKinds] = {"target-colour", "target-depth-compute", "target-depth-buffer", "region-bind",
                                       "image-region",  "resource",             "point-copy"};
const char* const kReasonName[kReasons] = {"draw", "draw-full", "draw-proven", "bound", "clear", "clear-part", "fill", "copy-into", "dispatch",
                                           "retire", "other", "next-copy"};
// A closed copy's cell: what changed its source first (kTouches: nothing did), what closed it, and whether its
// destination was read before that change and after it.
constexpr int kCells = (kTouches + 1) * kReasons * 4;
int cell(int src_first, int reason, bool before, bool after) {
    return ((src_first < 0 ? kTouches : src_first) * kReasons + reason) * 4 + (before ? 2 : 0) + (after ? 1 : 0);
}

struct Counts {
    std::uint64_t copies = 0, frames = 0;
    std::uint64_t closed = 0, unread = 0, unchanged = 0, changed = 0;  // closed: no read; every read before the source changed; one after
    std::uint64_t reads = 0, reads_after = 0;                           // of the closed records
    std::uint64_t src_first[kTouches] = {};                             // what changed the source first (the changed ones)
    std::uint64_t closed_by[kReasons] = {};
    std::uint64_t cells[kCells] = {};
    void add(const Counts& o) {
        copies += o.copies;
        frames += o.frames;
        closed += o.closed;
        unread += o.unread;
        unchanged += o.unchanged;
        changed += o.changed;
        reads += o.reads;
        reads_after += o.reads_after;
        for (int k = 0; k < kTouches; ++k) src_first[k] += o.src_first[k];
        for (int k = 0; k < kReasons; ++k) closed_by[k] += o.closed_by[k];
        for (int k = 0; k < kCells; ++k) cells[k] += o.cells[k];
    }
};
struct Names {
    const char* n[3] = {};
    void add(const char* s) {
        if (!s) return;
        for (const char*& p : n) {
            if (p == s) return;
            if (!p) {
                p = s;
                return;
            }
        }
    }
    std::string str() const {
        std::string o;
        for (const char* p : n) {
            if (p) o += (o.empty() ? "" : ",") + std::string(p);
        }
        return o.empty() ? "-" : o;
    }
};
struct Site {
    int kind = 0;
    std::uint64_t src = 0, dst = 0;
    int sf = 0, df = 0;
    std::uint32_t sw = 0, sh = 0, dw = 0, dh = 0, flags = 0;
    Counts win, total;
    Names readers, writers, closers, via;
    std::uint64_t last_flip = ~0ull;
};
using Key = std::tuple<int, std::uint64_t, std::uint64_t, int, int, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>;
std::map<Key, Site> g_sites;  // under g.mu

struct Live {
    Site* site;
    std::uint64_t s_lo, s_hi, d_lo, d_hi;
    std::uint32_t reads = 0, reads_after = 0;
    int src_touch = -1;
};
const char* const kCellFirst[kTouches + 1] = {"draw", "draw-full", "draw-proven", "bound", "clear", "clear-part", "fill", "copy-into", "dispatch",
                                              "retire", "other", "none"};
std::vector<Live> g_live;  // under g.mu: the copies whose destination still holds what they wrote
std::uint64_t g_lo = ~0ull, g_hi = 0;
std::uint64_t g_parts_skipped = 0, g_dropped = 0, g_last_report_flip = 0;

// Region binds: the source's write stamp and image at the last copy of each region.
struct RegionLast {
    std::uint64_t last_write = 0;
    VkImage image = VK_NULL_HANDLE;
};
std::map<std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, int>, RegionLast> g_regions;

const char* intern(const char* s) {
    if (!s) return nullptr;
    static std::set<std::string> names;  // under g.mu
    if (names.size() > 4096) return "(many)";
    return names.insert(s).first->c_str();
}

void bounds() {
    g_lo = ~0ull;
    g_hi = 0;
    for (const Live& l : g_live) {
        g_lo = std::min({g_lo, l.s_lo, l.d_lo});
        g_hi = std::max({g_hi, l.s_hi, l.d_hi});
    }
}

void close(const Live& l, int reason) {
    Counts& c = l.site->win;
    ++c.closed;
    ++c.closed_by[reason];
    ++c.cells[cell(l.src_touch, reason, l.reads > l.reads_after, l.reads_after > 0)];
    c.reads += l.reads;
    c.reads_after += l.reads_after;
    if (!l.reads) {
        ++c.unread;
    } else if (!l.reads_after) {
        ++c.unchanged;
    } else {
        ++c.changed;
        if (l.src_touch >= 0) ++c.src_first[l.src_touch];
    }
}

bool touch_impl(std::uint64_t base, int kind, const char* who) {
    if (base < g_lo || base >= g_hi) return false;
    bool erased = false, reached = false;
    for (std::size_t i = g_live.size(); i-- > 0;) {
        Live& l = g_live[i];
        if (base >= l.d_lo && base < l.d_hi) {
            close(l, kind);
            l.site->closers.add(intern(who));
            g_live.erase(g_live.begin() + static_cast<std::ptrdiff_t>(i));
            erased = reached = true;
            continue;
        }
        if (base >= l.s_lo && base < l.s_hi && l.src_touch < 0) {
            l.src_touch = kind;
            l.site->writers.add(intern(who));
            reached = true;
        }
    }
    if (erased) bounds();
    return reached;
}
}  // namespace

Quiet::Quiet() { ++t_quiet; }
Quiet::~Quiet() { --t_quiet; }
As::As(int touch) : prev(t_as) { t_as = touch; }
As::~As() { t_as = prev; }

void part_skipped() { ++g_parts_skipped; }

// Phase 2's census: each writer's draws by why they are not proven to write all of their target (0: proven).
std::map<std::pair<const char*, int>, std::uint64_t> g_proofs;  // under g.mu, by interned writer
void proof(const char* writer, int fail) {
    if (!g_on) return;
    if (g_proofs.size() < 4096) ++g_proofs[{intern(writer), fail}];
}

void read(std::uint64_t base, const char* who) {
    if (!g_on || base < g_lo || base >= g_hi) return;
    for (Live& l : g_live) {
        if (base < l.d_lo || base >= l.d_hi) continue;
        ++l.reads;
        if (l.src_touch >= 0) ++l.reads_after;
        l.site->readers.add(intern(who));
    }
}

bool touch(std::uint64_t base, int kind, const char* who) {
    if (!g_on) return false;
    return touch_impl(base, kind, who);
}

void rt_written(const RtImage& r) {
    if (!g_on || t_quiet) return;
    touch_impl(r.base, t_as, nullptr);
}

void retire(std::uint64_t base) {
    if (!g_on) return;
    touch_impl(base, kTouchRetire, nullptr);
}

void copy(int kind, const Side& src, const Side& dst, const char* who, std::uint32_t flags) {
    if (!g_on) return;
    read(src.base, who);  // the copy reads its source: a read of a copy that wrote it
    // The destination's last copy is done with; a copy whose source it was sees its source written.
    if (dst.base >= g_lo && dst.base < g_hi) {
        for (std::size_t i = g_live.size(); i-- > 0;) {
            Live& l = g_live[i];
            if (dst.base >= l.d_lo && dst.base < l.d_hi) {
                close(l, kNextCopy);
                g_live.erase(g_live.begin() + static_cast<std::ptrdiff_t>(i));
            } else if (dst.base >= l.s_lo && dst.base < l.s_hi && l.src_touch < 0) {
                l.src_touch = kTouchCopyInto;
                l.site->writers.add(intern(who));
            }
        }
    }
    const Key key{kind, src.base, dst.base, src.format, dst.format, src.w, src.h, dst.w, dst.h, flags};
    Site& s = g_sites[key];
    if (!s.win.copies && !s.total.copies) {
        s.kind = kind;
        s.src = src.base;
        s.dst = dst.base;
        s.sf = src.format;
        s.df = dst.format;
        s.sw = src.w;
        s.sh = src.h;
        s.dw = dst.w;
        s.dh = dst.h;
        s.flags = flags;
    }
    s.via.add(intern(who));
    ++s.win.copies;
    const std::uint64_t flip = hle_video_flip_count();
    if (s.last_flip != flip) {
        s.last_flip = flip;
        ++s.win.frames;
    }
    if (g_live.size() >= 256) {  // a destination nothing ever writes again: counted as dropped
        close(g_live.front(), kTouchOther);
        g_live.erase(g_live.begin());
        ++g_dropped;
    }
    Live l{&s, src.base, src.base + std::max<std::uint64_t>(src.bytes, 1), dst.base, dst.base + std::max<std::uint64_t>(dst.bytes, 1)};
    g_live.push_back(l);
    bounds();
}

void region_bind(const RtImage& src, std::uint32_t w, std::uint32_t h, bool fresh) {
    if (!g_on) return;
    RegionLast& last = g_regions[{src.base, w, h, static_cast<int>(src.format)}];
    const bool same = !fresh && last.image == src.image && last.last_write == src.last_write;
    last.image = src.image;
    last.last_write = src.last_write;
    const Key key{kRegionBind, src.base, 0, static_cast<int>(src.format), static_cast<int>(src.format), src.width, src.height, w, h, 0};
    Site& s = g_sites[key];
    if (!s.win.copies && !s.total.copies) {
        s.kind = kRegionBind;
        s.src = src.base;
        s.sf = s.df = static_cast<int>(src.format);
        s.sw = src.width;
        s.sh = src.height;
        s.dw = w;
        s.dh = h;
    }
    ++s.win.copies;
    const std::uint64_t flip = hle_video_flip_count();
    if (s.last_flip != flip) {
        s.last_flip = flip;
        ++s.win.frames;
    }
    // A region copy is read by the bind that made it: closed at once, "unchanged"
    // when the source had not been written since the region's last copy (the
    // copy repeated what the region image held), "changed" otherwise.
    ++s.win.closed;
    ++s.win.reads;
    if (same) {
        ++s.win.unchanged;
    } else {
        ++s.win.changed;
        ++s.win.src_first[kTouchOther];
    }
}

void report_locked() {
    if (!g_on) return;
    const std::uint64_t flip = hle_video_flip_count();
    const std::uint64_t flips = flip > g_last_report_flip ? flip - g_last_report_flip : 0;
    g_last_report_flip = flip;
    std::uint64_t by_kind[kKinds] = {};
    for (auto& [k, s] : g_sites) {
        Counts& c = s.win;
        if (!c.copies && !c.closed) continue;
        by_kind[s.kind] += c.copies;
        std::string first, by;
        for (int t = 0; t < kTouches; ++t) {
            if (c.src_first[t]) first += (first.empty() ? "" : " ") + std::string(kReasonName[t]) + " " + std::to_string(c.src_first[t]);
        }
        for (int t = 0; t < kReasons; ++t) {
            if (c.closed_by[t]) by += (by.empty() ? "" : " ") + std::string(kReasonName[t]) + " " + std::to_string(c.closed_by[t]);
        }
        std::string cells;
        for (int f = 0; f <= kTouches; ++f) {
            for (int r = 0; r < kReasons; ++r) {
                for (int ba = 0; ba < 4; ++ba) {
                    const std::uint64_t n = c.cells[(f * kReasons + r) * 4 + ba];
                    if (!n) continue;
                    cells += (cells.empty() ? "" : " ") + std::string(kCellFirst[f]) + "/" + kReasonName[r] + "/" + ((ba & 2) ? "b" : "-") +
                             ((ba & 1) ? "a" : "-") + "=" + std::to_string(n);
                }
            }
        }
        host_log("copy census: %s 0x%llx %ux%u f%d -> 0x%llx %ux%u f%d flags %02x: copies %llu in %llu flips (of %llu); closed %llu: unread "
                 "%llu, src unchanged %llu, src changed %llu [%s]; reads %llu (%llu after the source changed); closed by [%s]; cells [%s]; "
                 "via %s; readers %s; src writers %s; closers %s",
                 kKindName[s.kind], static_cast<unsigned long long>(s.src), s.sw, s.sh, s.sf, static_cast<unsigned long long>(s.dst), s.dw,
                 s.dh, s.df, s.flags, static_cast<unsigned long long>(c.copies), static_cast<unsigned long long>(c.frames),
                 static_cast<unsigned long long>(flips), static_cast<unsigned long long>(c.closed),
                 static_cast<unsigned long long>(c.unread), static_cast<unsigned long long>(c.unchanged),
                 static_cast<unsigned long long>(c.changed), first.c_str(), static_cast<unsigned long long>(c.reads),
                 static_cast<unsigned long long>(c.reads_after), by.c_str(), cells.c_str(), s.via.str().c_str(), s.readers.str().c_str(),
                 s.writers.str().c_str(), s.closers.str().c_str());
        s.total.add(c);
        c = Counts{};
    }
    if (!g_proofs.empty()) {
        // The writers whose draws ended or changed a copy here, by why each is (not) proven to write all of its target.
        std::map<const char*, std::string> by;
        for (const auto& [k, n] : g_proofs) {
            by[k.first] += (by[k.first].empty() ? "" : ", ") + std::to_string(k.second) + "=" + std::to_string(n);
        }
        std::string line;
        for (const auto& [w, t] : by) {
            line += (line.empty() ? "" : "; ") + std::string(w ? w : "?") + " {" + t + "}";
            if (line.size() > 800) {
                host_log("copy census: proofs (full_write::Fail=draws): %s", line.c_str());
                line.clear();
            }
        }
        if (!line.empty()) host_log("copy census: proofs (full_write::Fail=draws): %s", line.c_str());
        g_proofs.clear();
    }
    std::string kinds;
    for (int k = 0; k < kKinds; ++k) {
        if (by_kind[k]) kinds += " " + std::string(kKindName[k]) + "=" + std::to_string(by_kind[k]);
    }
    host_log("copy census: %llu flips, copies by kind:%s; live %zu, dropped %llu, run parts left out %llu",
             static_cast<unsigned long long>(flips), kinds.empty() ? " none" : kinds.c_str(), g_live.size(),
             static_cast<unsigned long long>(g_dropped), static_cast<unsigned long long>(g_parts_skipped));
}

}  // namespace copy_census
}  // namespace gpu
