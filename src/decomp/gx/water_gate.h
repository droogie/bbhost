#pragma once

// The water interaction's wave steps at 30 a second of game time
// (decomp/gx/water_gate.cpp).

#include <cmath>

// What a frame of the water interaction does. Decided once a frame at the
// manager's update (water_gate_time): kWaterStart on the first frame and
// after the game time went back or leapt (a load), kWaterTime on a frame that
// closes a whole 1/30 s of game time since the last step, else kWaterSkip.
// Settled at its task (water_gate_camera): in screen-space mode, whose grid
// follows the camera in whole units, a frame decided kWaterSkip on which the
// camera's truncated x or z left the last step's steps after all
// (kWaterCamera), and takes its 1/30 s from the next steps (at most one step
// ahead). Pure, for the unit test.
enum WaterStep { kWaterSkip = 0, kWaterStart, kWaterTime, kWaterCamera };
struct WaterGate {
    bool started = false;
    bool have_camera = false;
    float last_time = 0.0f;  // the game time at the last update
    float banked = 0.0f;     // game time not yet stepped: since the last step, less what steps took ahead (-1/30 .. 1/30)
    float camera_x = 0.0f, camera_z = 0.0f;  // trunc(camera) at the last step
};

constexpr float kWaterStep = 1.0f / 30.0f;
// A frame steps once the time banked reaches three quarters of a step, and
// every step takes a whole 1/30 s from the bank: the steps are 30 a game
// second at any rate - each frame at 30 - and the bank stays within a step.
// The quarter is for the clock: the game time is a float sum, so after a few
// minutes a frame's 1/30 s comes out short of 1/30 by its rounding (2.4e-4 s
// at an hour), and a frame at 30 must still step.
constexpr float kWaterThreshold = 0.75f * kWaterStep;

#ifdef DECOMP_WATER_GATE_MUTANTS
extern int g_water_gate_mutant;
#define WATER_MUTANT(n) (g_water_gate_mutant == (n))
#else
#define WATER_MUTANT(n) false
#endif

// x truncated toward zero, as the game's vroundps (imm 3) does - by the integer
// conversion, so a leaf calls nothing of the host's; a coordinate past 1e7 (or
// not a number) is left whole, and the comparison then sees it as it is.
inline float water_trunc(float x) {
    if (WATER_MUTANT(4)) return std::floor(x);
    return std::fabs(x) < 1e7f ? static_cast<float>(static_cast<long long>(x)) : x;
}

inline WaterStep water_gate_time(WaterGate& g, float time) {
    if (!g.started || (!(time >= g.last_time) && !WATER_MUTANT(6)) || time - g.last_time > 1.0f) {
        g.started = true;
        g.last_time = time;
        g.banked = 0.0f;
        g.have_camera = false;
        return kWaterStart;
    }
    g.banked += time - g.last_time;
    g.last_time = time;
    if (g.banked >= (WATER_MUTANT(7) ? kWaterStep : kWaterThreshold)) {
        g.banked -= kWaterStep;
        if (g.banked > kWaterStep) g.banked = kWaterStep;  // after a long stretch, one step to catch up, not a burst
        return kWaterTime;
    }
    return kWaterSkip;
}

inline WaterStep water_gate_camera(WaterGate& g, WaterStep why, bool scroll, float camera_x, float camera_z) {
    const float cx = water_trunc(camera_x), cz = water_trunc(camera_z);
    if (why == kWaterSkip) {
        if (!scroll || (g.have_camera && cx == g.camera_x && cz == g.camera_z)) return kWaterSkip;
        why = kWaterCamera;
        if (!WATER_MUTANT(5)) g.banked -= kWaterStep;
        if (g.banked < -kWaterStep) g.banked = -kWaterStep;
    }
    if (scroll) {
        g.have_camera = true;
        g.camera_x = cx;
        g.camera_z = cz;
    }
    return why;
}

// Both at once, for a frame whose update and task see the same time and
// camera.
inline WaterStep water_gate_step(WaterGate& g, float time, bool scroll, float camera_x, float camera_z) {
    return water_gate_camera(g, water_gate_time(g, time), scroll, camera_x, camera_z);
}

void decomp_water_gate_add();
