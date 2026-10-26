// The bottleneck meter's verdict (host/bottleneck.h), on the windows the
// places showed: each case is one second's counters as host/perf_meter.cpp
// gathers them, in ms of the second.
#include "host/bottleneck.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

using bottleneck::classify;
using bottleneck::Stage;
using bottleneck::Window;

bool near(double a, double b) { return std::fabs(a - b) < 1e-6; }

// A second at `fps` with steady frames and everything measured but the
// stages' loads, which each case sets.
Window second(double fps, double cap) {
    Window w;
    w.secs = 1.0;
    w.flips = fps;
    w.displayed = fps;
    w.cap_fps = cap;
    w.median_ms = 1000.0 / fps;
    w.limiter_ms = 0;
    w.main_cpu_ms = 0;
    w.gx_spin_ms = 0;
    w.render_cpu_ms = 0;
    w.cp_cpu_ms = 0;
    w.rec_busy_ms = 0;
    w.gpu_busy_ms = 0;
    w.present_cpu_ms = 0;
    return w;
}

void test_main_loop() {
    // Hunter's Dream uncapped on the RTX 4070 (a release check): 135 fps, the main
    // loop never in the limiter, the main thread 82% on a core, the command
    // processor 79%, the GPU 64%: the main loop.
    Window w = second(135, 240);
    w.main_cpu_ms = 820;
    w.gx_spin_ms = 120;
    w.main_wait_ms = 185;
    w.render_cpu_ms = 500;
    w.cp_cpu_ms = 790;
    w.rec_busy_ms = 610;
    w.gpu_busy_ms = 636;
    w.present_cpu_ms = 10;
    auto v = classify(w);
    CHECK(v.limit == Stage::kMain);
    CHECK(near(v.limit_load, 1.0));  // the main loop's busy share
    CHECK(near(v.load[static_cast<int>(Stage::kMain)], 0.70));  // its own work: 820 - 120
    CHECK(v.next == Stage::kCp);  // not gx: both are the main loop's
    CHECK(near(v.next_load, 0.79));
    CHECK(near(v.fps, 135));

    // The Cainhurst courtyard uncapped (the meter's run on an RTX 4070, and a
    // sampled run before it): 11.4 ms frames, the barrier spinning 4.7 ms of
    // them with one GX worker on the render queue's tail chunk, the main
    // thread's own work 5.0: the GX workers - a third of the main loop's time.
    // A CPU counter reads the main thread at 85% and would say main.
    w = second(87.7, 240);
    w.main_cpu_ms = 850;
    w.gx_spin_ms = 410;
    w.main_wait_ms = 154;
    w.cp_cpu_ms = 820;
    w.rec_busy_ms = 600;
    w.gpu_busy_ms = 600;
    v = classify(w);
    CHECK(v.limit == Stage::kGx);
    CHECK(v.next == Stage::kCp);
    // Central Yharnam: the spin 2.3 of a 10.3 ms frame, the own work 6.2: main.
    Window cy = second(97.6, 240);
    cy.main_cpu_ms = 830;
    cy.gx_spin_ms = 224;
    cy.cp_cpu_ms = 780;
    CHECK(classify(cy).limit == Stage::kMain);
    // The same second without the spin measured (perf_events refused, an
    // eboot that is not 1.09): the main loop, and the line says so.
    w.gx_spin_ms = -1;
    v = classify(w);
    CHECK(v.limit == Stage::kMain);
    CHECK(std::strstr(v.why, "not measured") != nullptr);
    CHECK(near(v.load[static_cast<int>(Stage::kMain)], 0.85));
    CHECK(v.load[static_cast<int>(Stage::kGx)] < 0);

    // The limiter not measured (an old log): the main loop all the same.
    w.limiter_ms = -1;
    w.gx_spin_ms = 410;
    v = classify(w);
    CHECK(v.limit == Stage::kGx);
    CHECK(v.main_loop < 0);
}

void test_command_processor() {
    // Old Yharnam uncapped: the render thread waits for the command
    // processor's backlog 8% of the second, bb-cp0 95% on a core: cp, even
    // with the main loop busy all the second (its frame-end wait is the same
    // back-pressure).
    Window w = second(98, 240);
    w.main_cpu_ms = 700;
    w.gx_spin_ms = 220;
    w.main_wait_ms = 260;
    w.backlog_ms = 80;
    w.render_cpu_ms = 560;
    w.cp_cpu_ms = 950;
    w.rec_busy_ms = 700;
    w.gpu_busy_ms = 700;
    auto v = classify(w);
    CHECK(v.limit == Stage::kCp);
    CHECK(near(v.limit_load, 0.95));
    CHECK(near(v.backlog, 0.08));
    // Under 5% of back-pressure is the main loop's: the command processor
    // only caught up now and then.
    w.backlog_ms = 40;
    CHECK(classify(w).limit == Stage::kMain);

    // The recorder: the command processor held back, bb-record replaying
    // 95% of the second and busier than bb-cp0.
    w.backlog_ms = 60;
    w.rec_busy_ms = 950;
    w.cp_cpu_ms = 800;
    v = classify(w);
    CHECK(v.limit == Stage::kRecord);
    CHECK(v.next == Stage::kCp);

    // The GPU behind it: the command processor waits for fences 20% of the
    // second and the GPU, 85% busy, is the busiest stage.
    w.rec_busy_ms = 500;
    w.cp_cpu_ms = 600;
    w.gpu_busy_ms = 850;
    w.cp_gpu_wait_ms = 200;
    CHECK(classify(w).limit == Stage::kGpu);
    w.cp_gpu_wait_ms = 50;  // not waiting for it: still the command processor
    CHECK(classify(w).limit == Stage::kCp);
}

void test_gpu() {
    // The Steam Deck at Mensis at 60 (an earlier series): 58.2 fps, the GPU 98% busy
    // at the APU's power limit, the limiter waiting 6% after the frames that
    // came in short. Under the cap, so not capped: the GPU.
    Window w = second(58.23, 60);
    w.limiter_ms = 64;
    w.main_cpu_ms = 600;
    w.gx_spin_ms = 40;
    w.cp_cpu_ms = 950;
    w.rec_busy_ms = 800;
    w.gpu_busy_ms = 980;
    auto v = classify(w);
    CHECK(v.limit == Stage::kGpu);
    CHECK(v.next == Stage::kCp);
    // The Radeon 680M at 1080p: 27 fps, the GPU full.
    w = second(27, 60);
    w.gpu_busy_ms = 995;
    w.cp_cpu_ms = 400;
    w.main_cpu_ms = 300;
    CHECK(classify(w).limit == Stage::kGpu);
    // 90% is the line.
    w.gpu_busy_ms = 900;
    CHECK(classify(w).limit == Stage::kGpu);
    w.gpu_busy_ms = 890;
    CHECK(classify(w).limit != Stage::kGpu);
}

void test_capped() {
    // The Deck at Central Yharnam at 60: 59.9 fps, the main loop's work 9.5
    // ms of the 16.7, the GPU 88%: capped, the GPU next.
    Window w = second(59.89, 60);
    w.limiter_ms = 430;
    w.main_cpu_ms = 500;
    w.gpu_busy_ms = 880;
    w.cp_cpu_ms = 700;
    auto v = classify(w);
    CHECK(v.limit == Stage::kCapped);
    CHECK(v.limit_load < 0);
    CHECK(v.next == Stage::kGpu);
    CHECK(near(v.next_load, 0.88));
    // The RTX at 60 at the Hunter's Dream: the limiter 4.6 ms of the 16.7,
    // the command processor 63%: the main loop is next, by its busy share (72%),
    // not by its own work's 52%.
    Window hd = second(60, 60);
    hd.limiter_ms = 276;
    hd.main_cpu_ms = 610;
    hd.gx_spin_ms = 88;
    hd.main_wait_ms = 141;
    hd.cp_cpu_ms = 630;
    hd.gpu_busy_ms = 390;
    v = classify(hd);
    CHECK(v.limit == Stage::kCapped);
    CHECK(v.next == Stage::kMain);
    CHECK(near(v.next_load, 0.724));
    // At the cap the GPU's 92% is headroom, not the limit.
    w.gpu_busy_ms = 920;
    CHECK(classify(w).limit == Stage::kCapped);
    // The limiter barely waiting (under 2%) at the cap: whatever is busiest.
    w.limiter_ms = 10;
    w.gpu_busy_ms = 500;
    w.main_cpu_ms = 900;
    CHECK(classify(w).limit == Stage::kMain);
    // Uncapped at V-Sync's 144 on the RTX: the cap is the refresh rate.
    w = second(143.9, 144);
    w.limiter_ms = 150;
    w.main_cpu_ms = 600;
    CHECK(classify(w).limit == Stage::kCapped);
    // Under the cap with the limiter waiting a fifth of the time: uneven
    // frames, the short ones waited out.
    w = second(50, 60);
    w.limiter_ms = 200;
    w.main_cpu_ms = 500;
    v = classify(w);
    CHECK(v.limit == Stage::kCapped);
    CHECK(std::strstr(v.why, "uneven") != nullptr);
}

void test_stalls() {
    // A new area: 20 flips in the second, one of 600 ms, the main loop in
    // file I/O 500 ms of it.
    Window w = second(20, 60);
    w.median_ms = 17;
    w.long_ms = 583;
    w.main_file_ms = 500;
    auto v = classify(w);
    CHECK(v.limit == Stage::kStall);
    CHECK(std::strstr(v.why, "file I/O") != nullptr);
    // Pipeline builds on the command processor's thread.
    w.main_file_ms = 0;
    w.pipeline_ms = 400;
    CHECK(std::strstr(classify(w).why, "pipeline builds") != nullptr);
    // Imports.
    w.pipeline_ms = 0;
    w.import_ms = 300;
    CHECK(std::strstr(classify(w).why, "imports") != nullptr);
    // A cause too small to name it.
    w.import_ms = 50;
    CHECK(std::strcmp(classify(w).why, "long frames") == 0);
    // Under a fifth of the second in long frames: not a stall.
    w.long_ms = 150;
    w.main_cpu_ms = 900;
    CHECK(classify(w).limit != Stage::kStall);
    // A loading screen half the second.
    w = second(120, 0);
    w.loading_ms = 600;
    CHECK(classify(w).limit == Stage::kStall);
    CHECK(std::strstr(classify(w).why, "loading") != nullptr);
    // Next to no frames.
    w = second(1, 60);
    CHECK(classify(w).limit == Stage::kStall);
    CHECK(std::strstr(classify(w).why, "no frames") != nullptr);
}

void test_render_and_present() {
    // The render thread on a core 92% of the second, nothing waiting for the
    // command processor.
    Window w = second(80, 240);
    w.main_cpu_ms = 600;
    w.render_cpu_ms = 920;
    w.cp_cpu_ms = 700;
    CHECK(classify(w).limit == Stage::kRender);
    // The presenter: the display showing 60 of 135 flips, the presenter 85%
    // busy, the main loop idle 15% (in the limiter).
    w = second(135, 240);
    w.displayed = 60;
    w.limiter_ms = 150;
    w.main_cpu_ms = 600;
    w.present_cpu_ms = 850;
    CHECK(classify(w).limit == Stage::kPresent);
    w.displayed = 130;  // showing them all: the limiter waits, uneven frames
    CHECK(classify(w).limit == Stage::kCapped);
}

void test_empty() {
    Window w;
    const auto v = classify(w);
    CHECK(v.limit == Stage::kNone);
    CHECK(std::strcmp(bottleneck::stage_name(Stage::kGx), "gx") == 0);
    CHECK(std::strcmp(bottleneck::stage_name(Stage::kRecord), "record") == 0);
}

}  // namespace

int main() {
    test_main_loop();
    test_command_processor();
    test_gpu();
    test_capped();
    test_stalls();
    test_render_and_present();
    test_empty();
    if (g_fail) {
        std::printf("bottleneck_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("bottleneck_test: ok\n");
    return 0;
}
