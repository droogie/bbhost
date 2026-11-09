// The draw pool's threads by default (host/gxvk_lists.h pool_threads_for),
// without the game: 2 on 16 hardware threads or more, 1 on 8-15, none below -
// and the count it takes them from (core/portable.h host_cpu_threads) follows
// the process's affinity, not the machine: this process confined to fewer
// CPUs reads that many, where std::thread::hardware_concurrency reads the
// machine's whatever the affinity.
#include "core/portable.h"
#include "host/gxvk_lists.h"

#include <cstdio>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sched.h>
#endif

namespace {

int g_failures = 0;
void check(bool ok, const char* what, unsigned a, unsigned b) {
    if (ok) return;
    if (++g_failures <= 16) std::printf("FAIL: %s (%u, %u)\n", what, a, b);
}

}  // namespace

int main() {
    using gxvk_lists::pool_threads_for;
    for (unsigned t : {0u, 1u, 4u, 6u, 7u}) check(pool_threads_for(t) == 0, "no pool under 8 hardware threads", t, 0);
    for (unsigned t : {8u, 12u, 15u}) check(pool_threads_for(t) == 1, "one thread on 8-15", t, 1);
    for (unsigned t : {16u, 20u, 32u, 80u, 256u}) check(pool_threads_for(t) == 2, "two threads on 16 or more", t, 2);

    // The affinity: confined to its first `want` CPUs, the process counts that many.
    const unsigned all = host_cpu_threads();
    const unsigned machine = std::thread::hardware_concurrency();
    check(all >= 1 && (machine == 0 || all <= machine), "the affinity's count within the machine's", all, machine);
    const unsigned want = all >= 4 ? 4 : all;
#if defined(_WIN32)
    DWORD_PTR process = 0, system = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) && process) {
        DWORD_PTR some = 0;
        unsigned n = 0;
        for (unsigned bit = 0; bit < 8 * sizeof(DWORD_PTR) && n < want; ++bit) {
            if (process & (static_cast<DWORD_PTR>(1) << bit)) {
                some |= static_cast<DWORD_PTR>(1) << bit;
                ++n;
            }
        }
        if (SetProcessAffinityMask(GetCurrentProcess(), some)) {
            check(host_cpu_threads() == n, "the count follows the affinity", host_cpu_threads(), n);
            SetProcessAffinityMask(GetCurrentProcess(), process);
        }
    }
#else
    cpu_set_t had, some;
    CPU_ZERO(&had);
    CPU_ZERO(&some);
    if (sched_getaffinity(0, sizeof(had), &had) == 0) {
        unsigned n = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE && n < want; ++cpu) {
            if (CPU_ISSET(cpu, &had)) {
                CPU_SET(cpu, &some);
                ++n;
            }
        }
        if (sched_setaffinity(0, sizeof(some), &some) == 0) {
            check(host_cpu_threads() == n, "the count follows the affinity", host_cpu_threads(), n);
            check(n >= 8 || pool_threads_for(host_cpu_threads()) == 0, "a process confined under 8 CPUs gets no pool", n, 0);
            sched_setaffinity(0, sizeof(had), &had);
        }
        check(host_cpu_threads() == all, "the count back with the affinity", host_cpu_threads(), all);
    }
#endif
    if (g_failures) {
        std::printf("pool_threads_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("pool_threads_test: ok (%u of the machine's %u hardware threads in the affinity)\n", all, machine);
    return 0;
}
