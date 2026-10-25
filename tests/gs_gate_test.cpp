// Whether a draw that binds a geometry shader is drawn with it (host/gs_gate.h),
// over the devices the game meets: off by default only on GCN under AMD's own
// driver (an RX 580), on everywhere else (a Radeon 680M and an 8060S under
// AMD's driver, the Steam Deck and a GCN card under RADV, NVIDIA, lavapipe),
// and BBHOST_GX_GS=1 / =0 over any of them. Then three mutants of the rule,
// each of which a check here must catch.
#include "host/gs_gate.h"

#include <cstdio>
#include <functional>

namespace {

int g_failures = 0;
bool g_quiet = false;  // a mutant's run: its failures are expected
#define CHECK(cond)                                                                                     \
    do {                                                                                                \
        if (!(cond)) {                                                                                  \
            if (!g_quiet) std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                               \
        }                                                                                               \
    } while (0)

using gpu::GsDevice;
using gpu::GsPath;

// The property sets: vendor, driver, subgroup sizes (min, max) as each reports them.
const GsDevice kRx580{0x1002, VK_DRIVER_ID_AMD_PROPRIETARY, 64, 64};         // GCN 4, AMD's Windows driver
const GsDevice kRx580Amdvlk{0x1002, VK_DRIVER_ID_AMD_OPEN_SOURCE, 64, 64};   // GCN under AMDVLK
const GsDevice kRx580Radv{0x1002, VK_DRIVER_ID_MESA_RADV, 64, 64};           // GCN under RADV (ACO)
const GsDevice k680m{0x1002, VK_DRIVER_ID_AMD_PROPRIETARY, 32, 64};          // RDNA 2, AMD's Windows driver
const GsDevice k8060s{0x1002, VK_DRIVER_ID_AMD_PROPRIETARY, 32, 64};         // RDNA 3.5, AMD's Windows driver
const GsDevice kDeck{0x1002, VK_DRIVER_ID_MESA_RADV, 32, 64};                // the Steam Deck's RDNA 2 under RADV
const GsDevice kNvidia{0x10de, VK_DRIVER_ID_NVIDIA_PROPRIETARY, 32, 32};     // an RTX 4070
const GsDevice kLavapipe{0x10005, VK_DRIVER_ID_MESA_LLVMPIPE, 8, 8};         // llvmpipe
const GsDevice kNoSizes{0x1002, VK_DRIVER_ID_AMD_PROPRIETARY, 0, 0};         // a driver that reports no subgroup sizes

using Rule = std::function<GsPath(const GsDevice&, const char*)>;

// Every check of the rule; returns how many failed.
int check_rule(const Rule& rule) {
    const int before = g_failures;
    // Unset: off on GCN under AMD's own drivers, on everywhere else.
    CHECK(rule(kRx580, nullptr) == GsPath::kOffGcnAmd);
    CHECK(rule(kRx580Amdvlk, nullptr) == GsPath::kOffGcnAmd);
    CHECK(rule(kRx580Radv, nullptr) == GsPath::kOn);
    CHECK(rule(k680m, nullptr) == GsPath::kOn);
    CHECK(rule(k8060s, nullptr) == GsPath::kOn);
    CHECK(rule(kDeck, nullptr) == GsPath::kOn);
    CHECK(rule(kNvidia, nullptr) == GsPath::kOn);
    CHECK(rule(kLavapipe, nullptr) == GsPath::kOn);
    CHECK(rule(kNoSizes, nullptr) == GsPath::kOn);
    // A value that is neither 0 nor 1 changes nothing.
    CHECK(rule(kRx580, "") == GsPath::kOffGcnAmd);
    CHECK(rule(kNvidia, "yes") == GsPath::kOn);
    // =1 forces it on, =0 off, on every device.
    for (const GsDevice* d : {&kRx580, &kRx580Amdvlk, &kRx580Radv, &k680m, &k8060s, &kDeck, &kNvidia, &kLavapipe, &kNoSizes}) {
        CHECK(rule(*d, "1") == GsPath::kForcedOn);
        CHECK(rule(*d, "0") == GsPath::kForcedOff);
    }
    // What the draws get.
    CHECK(gpu::gs_path_on(GsPath::kOn));
    CHECK(gpu::gs_path_on(GsPath::kForcedOn));
    CHECK(!gpu::gs_path_on(GsPath::kOffGcnAmd));
    CHECK(!gpu::gs_path_on(GsPath::kForcedOff));
    return g_failures - before;
}

}  // namespace

int main() {
    check_rule([](const GsDevice& d, const char* env) { return gpu::gs_path_for(d, env); });
    const int real = g_failures;

    // The mutants: each must fail at least one check (the count is put back after).
    struct Mutant {
        const char* name;
        Rule rule;
    };
    const Mutant mutants[] = {
        // Any AMD device off (RDNA's 680M and the Deck lose their particles).
        {"vendor alone", [](const GsDevice& d, const char* env) {
             if (env && env[0] == '0') return GsPath::kForcedOff;
             if (env && env[0] == '1') return GsPath::kForcedOn;
             return d.vendor_id == 0x1002 ? GsPath::kOffGcnAmd : GsPath::kOn;
         }},
        // Wave64 alone, whatever the driver (GCN under RADV loses them too).
        {"subgroup size alone", [](const GsDevice& d, const char* env) {
             if (env && env[0] == '0') return GsPath::kForcedOff;
             if (env && env[0] == '1') return GsPath::kForcedOn;
             return d.subgroup_min == 64 && d.subgroup_max == 64 ? GsPath::kOffGcnAmd : GsPath::kOn;
         }},
        // The switch ignored on the gated device (=1 cannot bring the path back on an RX 580).
        {"=1 ignored on GCN", [](const GsDevice& d, const char* env) {
             if (env && env[0] == '0') return GsPath::kForcedOff;
             const GsPath p = gpu::gs_path_for(d, nullptr);
             if (p == GsPath::kOffGcnAmd) return p;
             return env && env[0] == '1' ? GsPath::kForcedOn : p;
         }},
    };
    for (const Mutant& m : mutants) {
        g_quiet = true;
        const int failed = check_rule(m.rule);
        g_quiet = false;
        g_failures -= failed;
        if (!failed) {
            std::fprintf(stderr, "the mutant '%s' passed every check\n", m.name);
            ++g_failures;
        }
    }
    if (g_failures) {
        std::fprintf(stderr, "gs_gate_test: %d failure(s) (%d from the rule itself)\n", g_failures, real);
        return 1;
    }
    std::printf("gs_gate_test: ok (%zu mutants caught)\n", sizeof(mutants) / sizeof(mutants[0]));
    return 0;
}
