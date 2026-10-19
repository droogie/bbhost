// YEBIS's Gnm draw and dispatch wrappers (decomp/yebis/gnm.cpp) beside the
// game's own (tests/decomp/eboot_kit.h). Every callee is a recorder; on
// generated YEBIS contexts and arguments - garbage above each 32-bit
// argument's bits - ours must make the game's calls with the game's
// arguments in the game's order and return what the game's returns; with the
// GX trace's entry set, the indexed draw and (when the trace hooks it) the
// dispatch must hand the trace the frame its prologue hooks handed it and run
// nothing more when it takes the call. The draw and dispatch writers are
// called from the game's wrappers at the return addresses ours are told with,
// and from ours at addresses decomp_gx::game_return takes for those. Three
// deliberate mistakes must be caught, and the entries hold the instructions
// written against. Skips without the 1.09 eboot.
#define DECOMP_YEBIS_GNM_MUTANTS
int g_yebis_mutant = 0;
#include "decomp/yebis/gnm.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <utility>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x9eb1);

struct Call {
    int who;
    u64 a[4];
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec;
struct Callee {
    u64 bn;
    int args;
    u32 narrow;  // a bit per argument that is 32-bit
    const char* name;
};
const Callee kCallees[] = {
    {kPrepDraw, 4, 0xc, "prep_draw"},       {kCommitDraw, 3, 0, "commit_draw"},         {kPostDraw, 3, 0, "post_draw"},
    {kCommitDispatch, 3, 0, "commit_dispatch"}, {kPostDispatch, 3, 0, "post_dispatch"}, {kDrawIndex, 3, 0x2, "draw_index"},
    {kDrawIndexAuto, 2, 0x2, "draw_index_auto"}, {kDispatchDirect, 4, 0xe, "dispatch_direct"},
};
constexpr int kNumCallees = static_cast<int>(std::size(kCallees));
constexpr int kPostDrawId = 2, kPostDispatchId = 4, kDrawIndexId = 5, kDrawAutoId = 6, kDispatchId = 7;
u64 g_post_ret = 0, g_emit_ret = 0;

template <int W>
GUEST_ABI u64 recorder(u64 a0, u64 a1, u64 a2, u64 a3) {
    Call c{W, {a0, a1, a2, a3}};
    for (int k = 0; k < 4; ++k) {
        if (k >= kCallees[W].args) c.a[k] = 0;
        else if ((kCallees[W].narrow >> k) & 1) c.a[k] &= 0xffffffffull;
    }
    g_rec.push_back(c);
    if (W == kDrawIndexId || W == kDrawAutoId || W == kDispatchId) g_emit_ret = reinterpret_cast<u64>(__builtin_return_address(0));
    return W == kPostDrawId || W == kPostDispatchId ? g_post_ret : 0x5eed;
}
template <int... W>
void stub_all(std::integer_sequence<int, W...>) {
    (eboot_kit::stub(kCallees[W].bn, reinterpret_cast<const void*>(&recorder<W>)), ...);
}

int g_failed = 0;
bool g_quiet = false;
void fail(const char* what, int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) std::printf("FAIL: %s, case %d: %s\n", what, trial, why);
}

alignas(64) u8 g_ctx[0x40];
u64 garbage32() { return (static_cast<u64>(g_rng.u32()) << 32) | g_rng.u32(); }

u64 g_hook_answer = 0;
std::vector<u64> g_hook_seen;
GUEST_ABI u64 test_hook(u64 kind, const u64* f) {
    g_hook_seen.assign({kind, f[-1], f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]});
    return g_hook_answer;
}

enum Wrapper { kWDrawIndex, kWDrawAuto, kWDispatch };
const char* const kNames[3] = {"yebis_gnm_draw_index", "yebis_gnm_draw_auto", "yebis_gnm_dispatch"};
const u64 kBn[3] = {0x15fbc40, 0x15fbb50, 0x15fbd10};
const u64 kReturns[3] = {kDrawIndexReturn, kDrawAutoReturn, kDispatchReturn};

u64 call(int w, bool game, const u64* a) {
    const u64 y = reinterpret_cast<u64>(g_ctx);
    using F2 = GUEST_ABI u64 (*)(u64, u64);
    using F4 = GUEST_ABI u64 (*)(u64, u64, u64, u64);
    using F5 = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64);
    switch (w) {
        case kWDrawIndex: return game ? eboot_kit::fn<F5>(kBn[w])(y, a[0], a[1], a[2], a[3]) : yebis_gnm_draw_index(y, a[0], a[1], a[2], a[3]);
        case kWDrawAuto: return game ? eboot_kit::fn<F2>(kBn[w])(y, a[0]) : yebis_gnm_draw_auto(y, a[0]);
        default: return game ? eboot_kit::fn<F4>(kBn[w])(y, a[0], a[1], a[2]) : yebis_gnm_dispatch(y, a[0], a[1], a[2]);
    }
}

long run(int cases) {
    long n = 0;
    for (int w = 0; w < 3; ++w) {
        for (int t = 0; t < cases; ++t) {
            for (u8& b : g_ctx) b = static_cast<u8>(g_rng.u32());
            u64 a[4];
            for (u64& v : a) v = garbage32();
            g_post_ret = garbage32();
            decomp_gx::g_gx_hook_entry = nullptr;
            g_rec.clear();
            g_emit_ret = 0;
            const u64 game_ret = call(w, true, a);
            const std::vector<Call> game = g_rec;
            if (g_emit_ret != game_address(kReturns[w])) fail(kNames[w], t, "the game's writer return address is not the one ours is told with");
            g_rec.clear();
            g_emit_ret = 0;
            const u64 our_ret = call(w, false, a);
            if (our_ret != game_ret) fail(kNames[w], t, "the return value");
            if (g_rec != game) fail(kNames[w], t, "the calls");
            if (!g_emit_ret || decomp_gx::game_return(g_emit_ret) != game_address(kReturns[w])) fail(kNames[w], t, "ours' writer return address");
            // The trace's entry: the indexed draw always, the dispatch when hooked, the auto draw never.
            decomp_gx::g_gx_hook_entry = reinterpret_cast<void*>(&test_hook);
            for (const bool hooked : {true, false}) {
                decomp_gx::g_yebis_dispatch_hooked = hooked;
                for (const u64 answer : {1ull, 0ull}) {
                    g_hook_answer = answer;
                    g_rec.clear();
                    g_hook_seen.clear();
                    const u64 r = call(w, false, a);
                    const bool handed = w == kWDrawIndex || (w == kWDispatch && hooked);
                    if (!handed) {
                        if (!g_hook_seen.empty()) fail(kNames[w], t, "handed to the trace, which does not hook it");
                        if (g_rec != game) fail(kNames[w], t, "the calls with the trace armed");
                        continue;
                    }
                    const u64 kind = w == kWDrawIndex ? decomp_gx::g_yebis_draw_kind : decomp_gx::g_yebis_dispatch_kind;
                    if (g_hook_seen.size() != 10 || g_hook_seen[0] != kind) {
                        fail(kNames[w], t, "the trace's entry not called with its kind");
                        continue;
                    }
                    const u64 y = reinterpret_cast<u64>(g_ctx);
                    // kind, slot, r9 r8 rcx rdx rsi rdi, ret, stack arg
                    const u64 want[6] = {0, w == kWDrawIndex ? a[3] : 0, a[2], a[1], a[0], y};
                    bool ok = g_hook_seen[1] == 0 && g_hook_seen[8] != 0 && g_hook_seen[9] == 0;
                    for (int k = 0; k < 6; ++k) ok = ok && g_hook_seen[2 + k] == want[k];
                    if (!ok) fail(kNames[w], t, "the frame handed to the trace");
                    if (answer == 1 && (!g_rec.empty() || r != 0)) fail(kNames[w], t, "work done after the trace took it");
                    if (answer == 0 && g_rec != game) fail(kNames[w], t, "the calls after the trace left it");
                }
            }
            decomp_gx::g_gx_hook_entry = nullptr;
            ++n;
        }
    }
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_yebis_mutant = mutant;
    g_quiet = true;
    run(200);
    g_quiet = false;
    g_yebis_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("yebis_gnm_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 4000;
    stub_all(std::make_integer_sequence<int, kNumCallees>{});
    decomp_yebis_gnm_add();  // the writer callers told, as the decomp list's add does
    decomp_gx::g_yebis_draw_kind = 21;
    decomp_gx::g_yebis_dispatch_kind = 77;
    struct Entry {
        u64 bn;
        const u8* bytes;
        std::size_t n;
        const char* name;
    };
    const Entry kEntries[] = {
        {0x15fbc40, kEntryDrawIndex, sizeof kEntryDrawIndex, "yebis_gnm_draw_index"},
        {0x15fbb50, kEntryDrawAuto, sizeof kEntryDrawAuto, "yebis_gnm_draw_auto"},
        {0x15fbd10, kEntryDispatch, sizeof kEntryDispatch, "yebis_gnm_dispatch"},
    };
    for (const Entry& e : kEntries) {
        const u8* at = eboot_kit::file_at("yebis_gnm_test", e.bn);
        if (!at || std::memcmp(at, e.bytes, e.n) != 0) fail(e.name, -1, "the entry's instructions");
    }
    const long total = run(cases);
    for (const int mutant : {1, 2, 3}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("yebis_gnm_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("yebis_gnm_test: ok - %ld cases (each also through the trace's entry), 3 mutants caught\n", total);
    return 0;
}
