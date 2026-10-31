#pragma once

#include <string>

// The game's own frame pace: 30, 60, 90, 120, 150, 180, 240, 360, 480, 960 or
// Uncapped (the display's refresh rate; the options call it Display-Limited).
//
// FD4's frame-time manager (0x2434770, called once a frame from the main
// loop) picks an interval by mode - the int at +8, or +0xc while the one-shot
// flag +0x276 is set - through a switch at 0x24347f5. Modes 0, 1, 3 and 4 wait
// out 1/30 s; mode 2 waits 1/60 with a vblank interval of 1. The engine has a
// 60 fps mode; the game never selects it. The wait is a usleep to within 5 ms
// of the deadline and then a spin on gettimeofday - which is why the main
// thread reads ~90% busy at 30 fps while half of that is waiting. Ours
// replaces it (SprjFlipper::Update, BBHOST_FRAME_SOURCE).
//
// The Frame rate option (or BBHOST_GAME_FPS=30/60/.../960/0) when the game starts:
//  - 30: the game's own.
//  - 60 to 960: mode 2 with a 1/fps target, the game's fixed steps converted
//    (frame_rate.cpp's tables, from Kyo's FPS++ lists and yogesh1239's high-FPS
//    fixes, each checked against the eboot): the task step and every step that
//    used a fixed 1/30 read the measured frame time clamped to [target, 1/30]
//    (the FPS++ cave's slot, engine/sixty_fps_sites.inc and kFrameTimeSites);
//    per-frame amounts, weights and rates in .rodata are written from that
//    frame time every frame; code immediates and frame counts take the rate's
//    value. Above 60 the cloth's six constraint caves (ninety_fps_sites.inc)
//    scale its stiffness by (60 dt)^2. The play time adds each frame's
//    milliseconds with the remainder carried (the game truncates them), and
//    the effects - the lamps' flicker among them - and the wind on grass and
//    trees step in whole 1/30 s the same way, 30 times a second as at 30, the
//    frames between holding what the effects count in updates
//    (decomp/sfx/step_gate.cpp, BBHOST_SFX_GATE; [video] effect_steps or
//    BBHOST_SFX_STEP: smooth, the default, for 30 with what the effects move
//    by time moved by the frame's own time in the effects converted for it,
//    30 for whole steps alone, 60 for the effects in whole 1/60 s and the wind
//    by the frame's time, frame for both by the frame's time).
//    The follow camera keeps Kyo's 60 fps retune (Camera tuning FPS++, the
//    default) or runs the game's own values (Original), either converted to
//    each frame's length.
//  - Uncapped: the display's refresh rate when the game starts, to the nearest
//    whole frame a second (BBHOST_UNCAPPED_FPS=N sets it; 60 with no display),
//    run as that fixed rate - its target, steps, code immediates and counts,
//    1/180 s each on a 180 Hz display, V-Sync on or off (engine/uncapped_limit.h).
// Above 60 flips complete as soon as they are recorded (the game paces itself;
// hle/video.cpp). The one-shot overrides (+0x2bc, +0x2c0, +0x271, +0x272,
// +0x2c4) are cleared so nothing drops the game back to 30.
//
// The choice is made when the game starts: the game's fixed 1/30 s steps are
// constants in its code, and a rate needs them all changed together.

// The frame time and the camera's flag code on the game's side reads:
// engine/frame_rate_step.h.

struct ElfImage;

void frame_rate_install(ElfImage* image);

// The pace the game was started at: 30, or the rate it was converted to (one
// of the option's, or Uncapped's refresh rate).
int frame_rate_game_fps();

// The frames a second the pace holds the world to: the rate the game was
// started at. Any thread. The render queue's cut follows it
// (decomp/gx/render_queue.cpp).
float frame_rate_limit_fps();

// AtkParam_Pc new or reloaded (engine/params.cpp): at a converted rate the
// Bloodletter's L2 self-hit gets the reach its frenzy aura needs above 30
// (frame_rate.cpp, kBloodletterSelfHit).
void frame_rate_atk_params_loaded();

// Game time against real time (plugins' set_time_scale): 0.5 is half speed,
// 2 double, clamped to 0.1-4. At 30 the game steps a fixed amount a frame, so
// this is the frame pacing's interval (as far as the machine keeps up). At a
// converted rate every step follows the frame's time, so it scales the task
// step instead (at most 1/30 s a frame) and the frames keep their pace;
// what still counts frames (frame_rate.cpp's kCounts and the rate's code
// immediates) keeps real time.
void frame_rate_set_time_scale(float scale);

// The limiter since the last call, for the 300-flip report's timing line:
// its margin (learnt from its own sleeps' lateness, or
// BBHOST_LIMITER_MARGIN_US), how late its sleeps woke, and how long it spun
// a frame.
std::string frame_rate_limiter_window();
