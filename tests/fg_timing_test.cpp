#include "host/fg_timing.h"
#include <cassert>
#include <limits>

int main() {
    host::FgHistory history;
    assert(history.needs_reset(1));
    history.submitted(1);
    assert(!history.needs_reset(2));
    assert(history.needs_reset(3)); // dropped render: motion vectors no longer match NGX history
    history.submitted(3);
    assert(!history.needs_reset(4)); // one reset recovers the next consecutive pair
    assert(history.needs_reset(3)); // repeated anchor
    assert(history.needs_reset(0));
    history.clear();
    assert(history.needs_reset(4)); // resolution change/failure

    constexpr std::int64_t start = 1000000000;
    assert(host::fg_real_deadline_ns(start, 16.0f, 60) == start + 8333333);
    assert(host::fg_real_deadline_ns(start, 25.0f, 60) == start + 12500000);
    assert(host::fg_real_deadline_ns(start, 40.0f, 60) == start + 20000000);
    // A slow GPU evaluation must exhaust the spacing budget, not add a sleep.
    const auto ready = start + 30000000;
    assert(host::fg_real_deadline_ns(start, 25.0f, 60) < ready);
    assert(host::fg_real_deadline_ns(start, std::numeric_limits<float>::quiet_NaN(), 60) == start + 8333333);
    assert(host::fg_real_deadline_ns(start, 500.0f, 60) == start + 50000000);
    assert(host::fg_real_deadline_ns(start, 20.0f, 0) == start + 10000000);
    assert(host::fg_real_deadline_ns(start, 20.0f, 60, 4, 1) == start + 5000000);
    assert(host::fg_real_deadline_ns(start, 20.0f, 60, 4, 2) == start + 10000000);
    assert(host::fg_real_deadline_ns(start, 20.0f, 60, 4) == start + 15000000);

    host::FgSchedule schedule;
    // GPU source is 40 FPS while CPU recording remains at a nominal 60 FPS.
    schedule.observe(100000000, 104000000, 1, start, 16.7f, 60);
    schedule.observe(125000000, 129000000, 2, start + 25000000, 16.7f, 60);
    assert(schedule.period() == 25000000);
    assert(schedule.deadline(4, 4) - schedule.deadline(3, 4) == 6250000);
    // A driver wait past two subframe deadlines skips them instead of bursting.
    assert(schedule.next_position(1, 4, schedule.deadline(3, 4)) == 3);
    assert(schedule.next_position(1, 4, schedule.deadline(4, 4) + 10000000) == 4);
    assert(schedule.next_position(1, 4, schedule.deadline(1, 4) - 1) == 1);
    // Presentation delay must not become the next source frame's GPU period.
    schedule.observe(150000000, 154000000, 3, start + 80000000, 16.7f, 60);
    assert(schedule.period() == 25000000);
    assert(schedule.deadline(4, 4) == start + 75000000);
    // Lost presentation frames retain per-render cadence rather than expanding it.
    schedule.observe(200000000, 204000000, 5, start + 100000000, 16.7f, 60);
    assert(schedule.period() == 25000000);

    host::FgSchedule capped;
    capped.observe(100000000, 102000000, 1, start, 16.0f, 60);
    // A 116 Hz driver cap takes ~8.6 ms per present. At a 60 Hz source,
    // predicting that wait selects the midpoint then the real endpoint,
    // instead of making every real frame wait for all three generated ones.
    constexpr std::int64_t driver_wait = 8620689;
    const auto evaluation_ready = capped.deadline(0, 4);
    assert(capped.next_position(1, 4, evaluation_ready + driver_wait) == 2);
    assert(capped.next_position(3, 4, evaluation_ready + 2 * driver_wait) == 4);

    // Alternating render/evaluation costs must not alternate output phase.
    host::FgSchedule smooth, raw(false);
    std::int64_t previous_smooth = 0, previous_raw = 0;
    std::int64_t smooth_max_error = 0, raw_max_error = 0;
    constexpr std::int64_t source_period = 33333334;
    for (unsigned i = 1; i < 100; ++i) {
        const auto source = 1000000000ull + i * source_period + (i % 2 ? 2000000 : 0);
        const auto ready = source + (i % 2 ? 6000000 : 2000000);
        smooth.observe(source, ready, i, ready + 100000000, 33.33f, 60);
        raw.observe(source, ready, i, ready + 100000000, 33.33f, 60);
        const auto a = smooth.deadline(2, 2), b = raw.deadline(2, 2);
        if (i > 50) {
            smooth_max_error = std::max(smooth_max_error, std::abs(a - previous_smooth - source_period));
            raw_max_error = std::max(raw_max_error, std::abs(b - previous_raw - source_period));
            assert(a > previous_smooth);
        }
        previous_smooth = a; previous_raw = b;
    }
    assert(smooth_max_error < 1500000);
    assert(raw_max_error > 9000000);
    // A loading gap must reset the timeline instead of pacing stale deadlines.
    smooth.observe(6000000000, 6004000000, 200, 6104000000, 33.33f, 60);
    assert(smooth.deadline(1, 2) > 6104000000);
    // Sustained GPU load converges to 30 FPS instead of keeping a 60 Hz grid.
    host::FgSchedule load;
    std::uint64_t source = 1000000000;
    for (unsigned i = 1; i <= 100; ++i) {
        source += i < 20 ? 16666667 : source_period;
        load.observe(source, source + 4000000, i, source + 104000000, 16.67f, 60);
    }
    assert(std::abs(load.period() - source_period) < 10000);

    // GPU completion jitter can straddle the simulation cap. Clamping each
    // sample before filtering biases the cadence upward and sends the phase
    // further into the future as the presenter drops completed output sets.
    host::FgSchedule jitter;
    for (unsigned i = 1; i <= 2000; ++i) {
        const auto stamp = 1000000000ull + i * 16666667ull + (i % 2 ? 8000000 : 0);
        const auto ready = stamp + 2000000;
        if (i > 500 && i % 4) continue; // presentation backpressure
        jitter.observe(stamp, ready, i, ready + 100000000, 16.67f, 60);
        assert(jitter.deadline(0, 2) <= static_cast<std::int64_t>(ready + 120000000));
    }
}
