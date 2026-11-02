// Stage 4's census (host/gxvk_lists.h): the command processor's half, the
// fold thread that drains the leaves' rings, the records file and the lines.
#include "host/gxvk_lists.h"
#include "host/gxvk_lists_internal.h"

#include "hle/modules.h"
#include "host/gpu.h"
#include "host/list_resolve.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <x86intrin.h>

namespace gxvk_lists {

thread_local constinit std::uint32_t t_effects = 0;
thread_local constinit std::uint32_t t_order = 0;

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using ull = unsigned long long;
using namespace detail;

// list_split_of's kinds (list_resolve::Split), as the lines name them.
constexpr int kKinds = list_resolve::kSplits;
const char* const kKindName[kKinds] = {"targets", "waits",  "labels", "copies/fills/uploads/clears", "dispatches", "queries",
                                       "YEBIS/Scaleform draws", "CP writes", "predication", "constant engine", "calls",
                                       "draw packets", "submission ends"};
// render.cpp RunStop, in its order.
constexpr int kEffects = 15;
const char* const kEffectName[kEffects] = {"params", "targets", "tables", "program", "pipeline", "images", "buffers", "refresh",
                                           "store", "ring", "sets", "order", "index", "mirror", "rare"};
constexpr int kOrderKinds = static_cast<int>(Order::kOrders);
// The needs a stitcher can meet before a segment without changing what its draws bind: a refresh gate (the image is
// the same), the in-order work (a barrier, owed clears, a pass, a depth layout), a copy ring's wait, the params
// reservation; and, apart, the buffer shadow's mirror work (a page's check or copy - though a mirror to make is not).
constexpr std::uint32_t kStitchable = 1u << 7 | 1u << 11 | 1u << 9 | 1u << 0;
constexpr std::uint32_t kMirrorNeed = 1u << 13;
// Draws a list or a segment holds, by powers of two: 1, 2-3, ..., 128 and more.
constexpr int kSizes = 8;
const char* const kSizeName[kSizes] = {"1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128+"};
int size_of(u32 n) {
    int k = 0;
    while (k + 1 < kSizes && n >= (2u << k)) ++k;
    return k;
}

// A list as the command processor's walk met it (the records file's 'L').
struct CpList {
    u64 ctx = 0;
    u32 seq = 0;
    u32 draws = 0, runs = 0, segments = 0;
    u32 tess = 0, gs = 0, indirect = 0;
    u32 vtx_slots = 0, vtx_nohandle = 0, draws_nohandle = 0, ib_nohandle = 0;
    u32 effects_or = 0, clean = 0;
    u32 effect_draws[kEffects] = {};
    u32 order_or = 0, order_clean = 0;  // ... the order's answers it read (Order), and its draws that read none
    u32 order_draws[kOrderKinds] = {};
    u32 inside[kKinds] = {};      // in-order work between two of its draws
    u32 seg_sizes[kSizes] = {};   // its segments (would-be secondaries) by draws
    u32 seg_draws[kSizes] = {};   // ... and their draws
    u32 both_clean = 0;           // draws that needed and read nothing of the order
    u32 stitch = 0;               // draws whose needs are all work a stitcher can do before their segment, and read no answer
    u32 stitch_mirror = 0;        // ... the buffer shadow's mirror work counted as such too
    u32 cont = 0;                 // the rest of a list a submission's end cut (its first part went before)
    u64 call_flip = 0, cp_flip = 0;
    u64 tsc_first = 0, tsc_end = 0, tsc_draws = 0;
};
static_assert(sizeof(CpList) % 8 == 0, "a record of whole qwords");

// The command processor's list under way (one thread draws GX-native draws).
struct CpState {
    CpList cur;
    bool open = false;
    u64 targets = 0;  // its run's target key
    u32 seg = 0;      // draws in its segment under way
    u32 pending[kKinds] = {};  // in-order work since its last draw: inside if the list goes on, between if not
    u64 draw_t0 = 0;
    bool in_draw = false;  // cp_draw_begin opened the draw cp_draw_end closes
    u64 next_report = 0;
    u64 cut_ctx = 0;  // the list a submission's end closed: its draws after it are its rest
    u32 cut_seq = 0;
};
thread_local CpState t_cp;

// Shared between the command processor, which hands finished lists over,
// and the fold thread.
std::mutex g_mu;
std::vector<CpList> g_done;
u64 g_between[kKinds] = {};  // in-order work between lists (the walk's, under g_mu at a list's close)

// The fold thread's aggregate: a window and the whole run.
struct Agg {
    u64 flips_lo = 0, flips_hi = 0;
    u64 lists = 0, draws = 0, runs = 0, segments = 0, tess = 0, gs = 0, indirect = 0;
    u64 vtx_slots = 0, vtx_nohandle = 0, draws_nohandle = 0, ib_nohandle = 0, clean = 0, lists_clean = 0, lists_clean_draws = 0;
    u64 effect_draws[kEffects] = {}, first_effect_lists[kEffects] = {}, first_effect_draws[kEffects] = {};
    u64 order_draws[kOrderKinds] = {}, order_clean = 0, both_clean = 0, lists_both_clean = 0, lists_both_clean_draws = 0;
    u64 stitch = 0, stitch_mirror = 0, lists_stitch = 0, lists_stitch_draws = 0, lists_stitch_mirror = 0, lists_stitch_mirror_draws = 0;
    u64 inside[kKinds] = {}, between[kKinds] = {};
    u64 list_sizes[kSizes] = {}, list_size_draws[kSizes] = {}, seg_sizes[kSizes] = {}, seg_draws[kSizes] = {};
    u64 tsc_span = 0, tsc_draws = 0;
    // GX's side, joined: finish -> execute -> the command processor's first draw.
    u64 joined = 0, finish_exec_tsc = 0, exec_cp_tsc = 0, lag[4] = {};
    u64 finishes = 0, finish_tokens = 0, execs = 0, exec_unknown = 0, exec_again = 0, not_found = 0, cont = 0;
    std::map<u32, u64> exec_batches;  // ExecuteCommandList(s) calls by lists given
};

class Fold {
public:
    void start();
    void report(bool at_exit);
    void step(bool all);

private:
    void run();
    void add(Agg& a, const CpList& l, u64 fin, u64 ex);
    void line(const Agg& a, const char* what);
    double ms(u64 ticks) const { return tick_per_ms_ > 0 ? static_cast<double>(ticks) / tick_per_ms_ : 0; }

    std::mutex mu_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    u64 fin_read_ = 0, exec_read_ = 0;
    std::map<std::pair<u64, u32>, std::pair<u64, u64>> gx_;  // (ctx, seq) -> (finish tsc, execute tsc)
    Agg win_, all_;
    FILE* out_ = nullptr;
    double tick_per_ms_ = 0;
    u64 t0_ = 0;
    std::chrono::steady_clock::time_point c0_;
};
Fold g_fold;

void Fold::start() {
    static std::once_flag once;
    std::call_once(once, [this] {
        if (const char* p = std::getenv("BBHOST_GXVK_LISTS_OUT"); p && *p) {
            out_ = std::fopen(p, "wb");
            if (!out_) host_log("gxvk lists: cannot write %s; lines only", p);
        }
        if (out_) {
            const char magic[8] = {'G', 'X', 'L', 'I', 'S', 'T', '1', '\n'};
            std::fwrite(magic, 1, sizeof magic, out_);
            const u32 sizes[3] = {static_cast<u32>(sizeof(CpList)), static_cast<u32>(sizeof(FinishRec)), static_cast<u32>(sizeof(ExecRec))};
            std::fwrite(sizes, sizeof sizes, 1, out_);
        }
        t0_ = __rdtsc();
        c0_ = std::chrono::steady_clock::now();
        thread_ = std::thread([this] { run(); });
        host_log("gxvk lists: census on (GX's command lists: finished, executed, and as the command processor walks them)%s",
                 out_ ? "; records to BBHOST_GXVK_LISTS_OUT" : "");
    });
}

void Fold::run() {
    while (!stop_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        step(false);
    }
}

template <class R>
bool take(R* ring, u64& read, std::atomic<u64>& pos, R& got, bool all) {
    const u64 end = pos.load(std::memory_order_acquire);
    if (read >= end) return false;
    if (end - read > kRing) read = end - kRing;  // fallen behind: the oldest were written over
    R& r = ring[read & (kRing - 1)];
    if (r.done.load(std::memory_order_acquire) != read + 1 && !all) return false;
    std::memcpy(static_cast<void*>(&got), static_cast<const void*>(&r), sizeof(R));
    ++read;
    return true;
}

void Fold::step(bool all) {
    std::lock_guard<std::mutex> lk(mu_);
    const double since = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0_).count();
    if (since > 50) tick_per_ms_ = static_cast<double>(__rdtsc() - t0_) / since;
    FinishRec f;
    while (take(g_finish, fin_read_, g_finish_pos, f, all)) {
        if (out_) {
            std::fputc('F', out_);
            std::fwrite(&f, sizeof f, 1, out_);
        }
        gx_[{f.ctx, f.seq}] = {f.tsc, 0};
        ++win_.finishes;
        ++all_.finishes;
        win_.finish_tokens += f.tokens;
        all_.finish_tokens += f.tokens;
    }
    ExecRec e;
    while (take(g_exec, exec_read_, g_exec_pos, e, all)) {
        if (out_) {
            std::fputc('E', out_);
            std::fwrite(&e, sizeof e, 1, out_);
        }
        ++win_.execs;
        ++all_.execs;
        if (e.index == 0) {
            ++win_.exec_batches[e.count];
            ++all_.exec_batches[e.count];
        }
        if (!e.seq) {
            ++win_.exec_unknown;
            ++all_.exec_unknown;
            continue;
        }
        auto it = gx_.find({e.ctx, e.seq});
        if (it != gx_.end()) {
            if (it->second.second) {  // GX empties a list it executes: a second execution would be news
                ++win_.exec_again;
                ++all_.exec_again;
            }
            it->second.second = e.tsc;
        }
    }
    std::vector<CpList> done;
    u64 between[kKinds];
    {
        std::lock_guard<std::mutex> lk2(g_mu);
        done.swap(g_done);
        std::memcpy(between, g_between, sizeof between);
        std::memset(g_between, 0, sizeof g_between);
    }
    for (int k = 0; k < kKinds; ++k) {
        win_.between[k] += between[k];
        all_.between[k] += between[k];
    }
    for (const CpList& l : done) {
        if (out_) {
            std::fputc('L', out_);
            std::fwrite(&l, sizeof l, 1, out_);
        }
        u64 fin = 0, ex = 0;
        if (auto it = gx_.find({l.ctx, l.seq}); it != gx_.end() && !l.cont) {
            fin = it->second.first;
            ex = it->second.second;
        }
        add(win_, l, fin, ex);
        add(all_, l, fin, ex);
    }
    // Lists finished over two seconds ago that the walk never met (no
    // GX-native draw in them), or met in parts.
    if (gx_.size() > 20000 && tick_per_ms_ > 0) {
        const u64 old = __rdtsc() - static_cast<u64>(2000 * tick_per_ms_);
        for (auto it = gx_.begin(); it != gx_.end();) it = it->second.first < old ? gx_.erase(it) : std::next(it);
    }
    if (out_) std::fflush(out_);
}

void Fold::add(Agg& a, const CpList& l, u64 fin, u64 ex) {
    if (!a.flips_lo || l.cp_flip < a.flips_lo) a.flips_lo = l.cp_flip;
    a.flips_hi = std::max(a.flips_hi, l.cp_flip);
    if (l.cont) ++a.cont;
    ++a.lists;
    a.draws += l.draws;
    a.runs += l.runs;
    a.segments += l.segments;
    a.tess += l.tess;
    a.gs += l.gs;
    a.indirect += l.indirect;
    a.vtx_slots += l.vtx_slots;
    a.vtx_nohandle += l.vtx_nohandle;
    a.draws_nohandle += l.draws_nohandle;
    a.ib_nohandle += l.ib_nohandle;
    a.clean += l.clean;
    if (!l.effects_or) {
        ++a.lists_clean;
        a.lists_clean_draws += l.draws;
    } else {
        const int first = __builtin_ctz(l.effects_or);  // the lowest bit: a stable name, not the first in time
        if (first < kEffects) {
            ++a.first_effect_lists[first];
            a.first_effect_draws[first] += l.draws;
        }
    }
    for (int k = 0; k < kEffects; ++k) a.effect_draws[k] += l.effect_draws[k];
    for (int k = 0; k < kOrderKinds; ++k) a.order_draws[k] += l.order_draws[k];
    a.order_clean += l.order_clean;
    a.both_clean += l.both_clean;
    a.stitch += l.stitch;
    a.stitch_mirror += l.stitch_mirror;
    if (l.stitch == l.draws) {
        ++a.lists_stitch;
        a.lists_stitch_draws += l.draws;
    }
    if (l.stitch_mirror == l.draws) {
        ++a.lists_stitch_mirror;
        a.lists_stitch_mirror_draws += l.draws;
    }
    if (!l.effects_or && !l.order_or) {
        ++a.lists_both_clean;
        a.lists_both_clean_draws += l.draws;
    }
    for (int k = 0; k < kKinds; ++k) a.inside[k] += l.inside[k];
    const int s = size_of(l.draws);
    ++a.list_sizes[s];
    a.list_size_draws[s] += l.draws;
    for (int k = 0; k < kSizes; ++k) {
        a.seg_sizes[k] += l.seg_sizes[k];
        a.seg_draws[k] += l.seg_draws[k];
    }
    a.tsc_span += l.tsc_end - l.tsc_first;
    a.tsc_draws += l.tsc_draws;
    if (fin && ex && ex >= fin && l.tsc_first >= ex) {
        ++a.joined;
        a.finish_exec_tsc += ex - fin;
        a.exec_cp_tsc += l.tsc_first - ex;
    } else if (!fin) {
        ++a.not_found;
    }
    const u64 lag = l.cp_flip > l.call_flip ? l.cp_flip - l.call_flip : 0;
    ++a.lag[lag < 4 ? lag : 3];
}

double pct(u64 a, u64 b) { return b ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0; }
double per(u64 a, u64 b) { return b ? static_cast<double>(a) / static_cast<double>(b) : 0.0; }

void Fold::line(const Agg& a, const char* what) {
    const u64 flips = a.flips_hi > a.flips_lo ? a.flips_hi - a.flips_lo : 1;
    std::string sizes, segs, eff, first, inside, between, batches;
    for (int k = 0; k < kSizes; ++k) {
        char b[80];
        std::snprintf(b, sizeof b, "%s%s %.1f%% (%.1f%%)", k ? ", " : "", kSizeName[k], pct(a.list_sizes[k], a.lists), pct(a.list_size_draws[k], a.draws));
        sizes += b;
        std::snprintf(b, sizeof b, "%s%s %.1f%% (%.1f%%)", k ? ", " : "", kSizeName[k], pct(a.seg_sizes[k], a.segments), pct(a.seg_draws[k], a.draws));
        segs += b;
    }
    for (int k = 0; k < kEffects; ++k) {
        if (!a.effect_draws[k] && !a.first_effect_lists[k]) continue;
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %.2f%%", eff.empty() ? "" : ", ", kEffectName[k], pct(a.effect_draws[k], a.draws));
        eff += b;
        if (a.first_effect_lists[k]) {
            std::snprintf(b, sizeof b, "%s%s %.1f%% (%.1f%% of draws)", first.empty() ? "" : ", ", kEffectName[k], pct(a.first_effect_lists[k], a.lists),
                          pct(a.first_effect_draws[k], a.draws));
            first += b;
        }
    }
    for (int k = 0; k < kKinds; ++k) {
        char b[96];
        if (a.inside[k]) {
            std::snprintf(b, sizeof b, "%s%s %.2f", inside.empty() ? "" : ", ", kKindName[k], per(a.inside[k], flips));
            inside += b;
        }
        if (a.between[k]) {
            std::snprintf(b, sizeof b, "%s%s %.2f", between.empty() ? "" : ", ", kKindName[k], per(a.between[k], flips));
            between += b;
        }
    }
    for (const auto& [n, c] : a.exec_batches) {
        char b[48];
        std::snprintf(b, sizeof b, "%s%u lists %llu", batches.empty() ? "" : ", ", n, static_cast<ull>(c));
        batches += b;
        if (batches.size() > 300) break;
    }
    const u64 lags = a.lag[0] + a.lag[1] + a.lag[2] + a.lag[3];
    host_log("gxvk lists (%s, flips %llu-%llu): %.1f lists a flip with %.1f GX-native draws (%.1f a list), %.1f render passes a list "
             "(%.1f draws each), %.1f segments a list - would-be secondaries, split at a pass and at in-order work inside - (%.1f draws "
             "each); lists by draws: %s; segments by draws: %s",
             what, static_cast<ull>(a.flips_lo), static_cast<ull>(a.flips_hi), per(a.lists, flips), per(a.draws, flips), per(a.draws, a.lists),
             per(a.runs, a.lists), per(a.draws, a.runs), per(a.segments, a.lists), per(a.draws, a.segments), sizes.c_str(), segs.c_str());
    host_log("gxvk lists (%s): in-order work a flip inside lists: %s; between lists: %s", what, inside.empty() ? "none" : inside.c_str(),
             between.empty() ? "none" : between.c_str());
    host_log("gxvk lists (%s): draws that needed nothing in the command processor's order %.1f%%, lists all of whose draws did %.1f%% "
             "(%.1f%% of draws); draws by what a run part would stop at: %s; lists by their lowest such bit: %s",
             what, pct(a.clean, a.draws), pct(a.lists_clean, a.lists), pct(a.lists_clean_draws, a.draws), eff.empty() ? "none" : eff.c_str(),
             first.empty() ? "none" : first.c_str());
    host_log("gxvk lists (%s): draws that read an answer of the order %.2f%% - a constant buffer's copy version %.2f%%, a constant buffer "
             "with a GPU write to land %.2f%%, a vertex record's copy version %.2f%%, the indices' or indirect arguments' %.2f%%, a twin "
             "%.2f%%; draws that needed and read nothing of it %.1f%%, lists all of whose draws did %.1f%% (%.1f%% of draws)",
             what, 100.0 - pct(a.order_clean, a.draws), pct(a.order_draws[0], a.draws), pct(a.order_draws[1], a.draws),
             pct(a.order_draws[2], a.draws), pct(a.order_draws[3], a.draws), pct(a.order_draws[4], a.draws), pct(a.both_clean, a.draws),
             pct(a.lists_both_clean, a.lists), pct(a.lists_both_clean_draws, a.draws));
    host_log("gxvk lists (%s): draws whose needs a stitcher meets before their segment (refresh, barrier/clears/pass, ring, params) "
             "and that read no answer %.1f%%, lists all such %.1f%% (%.1f%% of draws); with the shadow's mirror work too %.1f%%, lists "
             "%.1f%% (%.1f%% of draws)",
             what, pct(a.stitch, a.draws), pct(a.lists_stitch, a.lists), pct(a.lists_stitch_draws, a.draws), pct(a.stitch_mirror, a.draws),
             pct(a.lists_stitch_mirror, a.lists), pct(a.lists_stitch_mirror_draws, a.draws));
    host_log("gxvk lists (%s): vertex slots without a GX handle %.1f%% (draws with one %.1f%%), indexed draws whose index buffer has none "
             "%.2f%% of draws; tessellated %.2f%%, geometry-shader %.2f%%, indirect %.2f%% of draws",
             what, pct(a.vtx_nohandle, a.vtx_slots), pct(a.draws_nohandle, a.draws), pct(a.ib_nohandle, a.draws), pct(a.tess, a.draws),
             pct(a.gs, a.draws), pct(a.indirect, a.draws));
    host_log("gxvk lists (%s): the command processor's time a flip from a list's first draw to its last %.3f ms, in its draws %.3f ms; "
             "GX: %.1f lists finished a flip (%.1f GX-native draw tokens each), %.1f executed (%llu unknown, %llu executed again), executed in calls of: %s; "
             "joined %llu lists: finish to execute %.3f ms, execute to the command processor's first draw %.3f ms; flips from the "
             "first draw's GX call to the command processor 0 %.1f%%, 1 %.1f%%, 2 %.1f%%, 3+ %.1f%%; not finished in the census %llu; "
             "lists a submission's end cut %llu; contexts past the table %llu",
             what, ms(a.tsc_span) / static_cast<double>(flips), ms(a.tsc_draws) / static_cast<double>(flips), per(a.finishes, flips),
             per(a.finish_tokens, a.finishes), per(a.execs, flips), static_cast<ull>(a.exec_unknown), static_cast<ull>(a.exec_again), batches.empty() ? "none" : batches.c_str(),
             static_cast<ull>(a.joined), a.joined ? ms(a.finish_exec_tsc) / static_cast<double>(a.joined) : 0.0,
             a.joined ? ms(a.exec_cp_tsc) / static_cast<double>(a.joined) : 0.0, pct(a.lag[0], lags), pct(a.lag[1], lags),
             pct(a.lag[2], lags), pct(a.lag[3], lags), static_cast<ull>(a.not_found), static_cast<ull>(a.cont),
             static_cast<ull>(g_ctx_full.load()));
}

void Fold::report(bool at_exit) {
    if (at_exit) {
        // The thread ends first: an exit through std::exit destroys this object after the report.
        if (thread_.joinable()) {
            stop_.store(true);
            thread_.join();
        }
        step(true);
        std::lock_guard<std::mutex> lk(mu_);
        if (all_.lists) line(all_, "the whole run");
        host_log("gxvk lists: the TSC ran %.0f ticks a ms over the run (the records' clock)", tick_per_ms_);
        if (out_) std::fflush(out_);
        return;
    }
    step(false);
    std::lock_guard<std::mutex> lk(mu_);
    if (win_.lists) line(win_, "the last 1,000 flips");
    win_ = Agg{};
}

// A list's end: handed to the fold thread, with the in-order work met since
// its last draw counted between lists.
void close_list(CpState& s) {
    if (!s.open) return;
    CpList& l = s.cur;
    if (s.seg) {
        const int k = size_of(s.seg);
        ++l.seg_sizes[k];
        l.seg_draws[k] += s.seg;
        ++l.segments;
        s.seg = 0;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    for (int k = 0; k < kKinds; ++k) {
        g_between[k] += s.pending[k];
        s.pending[k] = 0;
    }
    g_done.push_back(l);
    s.open = false;
}

}  // namespace

void cp_draw_begin(const GpuDrawInputs& in, const GxDrawObjects& o) {
    CpState& s = t_cp;
    const u64 now = __rdtsc();
    if (!s.next_report) g_fold.start();
    if (s.open && (o.ctx != s.cur.ctx || o.list_seq != s.cur.seq)) close_list(s);
    const u64 key = list_resolve::targets_of(in);
    bool pending = false;
    for (u32 v : s.pending) pending |= v != 0;
    if (!s.open) {
        s.cur = CpList{};
        s.cur.cont = o.ctx == s.cut_ctx && o.list_seq == s.cut_seq;
        s.cut_ctx = 0;
        s.cut_seq = 0;
        s.cur.ctx = o.ctx;
        s.cur.seq = o.list_seq;
        s.cur.call_flip = o.call_flip;
        s.cur.cp_flip = hle_video_flip_count();
        s.cur.tsc_first = now;
        s.open = true;
        s.targets = key;
        s.seg = 0;
        ++s.cur.runs;
    } else {
        // In-order work since its last draw, inside the list: a segment ends there.
        if (pending) {
            for (int k = 0; k < kKinds; ++k) {
                s.cur.inside[k] += s.pending[k];
                s.pending[k] = 0;
            }
        }
        const bool new_run = key != s.targets;
        if (new_run) ++s.cur.runs;
        if ((new_run || pending) && s.seg) {
            const int k = size_of(s.seg);
            ++s.cur.seg_sizes[k];
            s.cur.seg_draws[k] += s.seg;
            ++s.cur.segments;
            s.seg = 0;
        }
        s.targets = key;
    }
    CpList& l = s.cur;
    ++l.draws;
    ++s.seg;
    if (in.stages & 0x7) ++l.tess;
    if (in.gs && o.shader[3]) ++l.gs;
    if (o.indirect_va) ++l.indirect;
    bool nohandle = false;
    for (int k = 0; k < 16; ++k) {
        if (!((o.vtx_valid >> k) & 1)) continue;
        ++l.vtx_slots;
        if (!o.vtx_handle[k]) {
            ++l.vtx_nohandle;
            nohandle = true;
        }
    }
    if (nohandle) ++l.draws_nohandle;
    if (o.indexed && o.index_known && !o.ib_handle) ++l.ib_nohandle;
    t_effects = 0;
    t_order = 0;
    s.draw_t0 = now;
    s.in_draw = true;
    // Every 1,000 flips with tooling, from the command processor.
    const u64 flip = hle_video_flip_count();
    if (flip >= s.next_report) {
        if (s.next_report > 1 && host_tooling()) g_fold.report(false);
        s.next_report = flip + 1000;
    }
}

void cp_draw_end() {
    CpState& s = t_cp;
    if (!s.open || !s.in_draw) return;
    s.in_draw = false;
    const u64 now = __rdtsc();
    CpList& l = s.cur;
    l.tsc_draws += now - s.draw_t0;
    l.tsc_end = now;
    const u32 e = t_effects;
    if (!e) {
        ++l.clean;
    } else {
        l.effects_or |= e;
        for (int k = 0; k < kEffects; ++k) l.effect_draws[k] += (e >> k) & 1;
    }
    const u32 o = t_order;
    if (!o) {
        ++l.order_clean;
    } else {
        l.order_or |= o;
        for (int k = 0; k < kOrderKinds; ++k) l.order_draws[k] += (o >> k) & 1;
    }
    if (!e && !o) ++l.both_clean;
    if (!o && (e & ~kStitchable) == 0) ++l.stitch;
    if (!o && (e & ~(kStitchable | kMirrorNeed)) == 0) ++l.stitch_mirror;
    t_effects = 0;
    t_order = 0;
}

void cp_packet(int split) {
    if (split < 0 || split >= kKinds) return;
    ++t_cp.pending[split];
}

void cp_submission_end() {
    CpState& s = t_cp;
    if (s.open) {
        ++s.pending[list_resolve::kSplitSubmission];
        s.cut_ctx = s.cur.ctx;
        s.cut_seq = s.cur.seq;
        close_list(s);
    } else {
        std::lock_guard<std::mutex> lk(g_mu);
        for (int k = 0; k < kKinds; ++k) {
            g_between[k] += s.pending[k];
            s.pending[k] = 0;
        }
        ++g_between[list_resolve::kSplitSubmission];
    }
}

void report(bool at_exit) {
    if (!g_census) return;
    g_fold.report(at_exit);
}

}  // namespace gxvk_lists
