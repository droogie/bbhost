// The host's settings as one object: what the options
// screen (F10), the port's rows in the game's System menu and
// bbhost-options.toml edit by name (host/options.h), read back by every
// consumer as typed, already-scaled fields. The table in options.cpp is the
// single source; it publishes a new snapshot on every change
// (host/settings_store.h) and tells the consumers that registered a listener
// (host_settings_listen), which keep what they act on - so no consumer asks
// for the settings every frame or view. A reader that wants one now takes the
// newest snapshot without a lock (host_settings_ref, host_settings).
//
// The graphics half is shaped after what DS3 keeps in its CSGraphicsConfig
// (the released PC binary's RTTI names the class): per-effect switches and
// strengths the renderer applies each frame, held in one place instead of
// a copy in each consumer.
#pragma once

#include <cstdint>

struct HostSettings {
    std::uint64_t serial = 0;  // host_opt_serial() when this was built

    // Display
    bool fullscreen = false;
    bool vsync = true;
    int frame_cap = 30;   // the game's frame rate (engine/frame_rate.h): 30 to 960; 0 is Display-Limited (the display's rate)
    bool fps_counter = false;
    bool memory_counter = false;  // video memory and shared GPU memory, under the FPS counter (host/window.cpp)
    bool perf_overlay = false;    // the bottleneck meter's verdict under them (host/window.cpp, host/perf_meter.h)
    int res_width = 1920, res_height = 1080;  // the render resolution (BBHOST_RES wins)

    // Camera: multipliers on LockCamParam's values (engine/camera.cpp)
    float fov_scale = 1.0f;              // 1.0 the game's own; +5% a step
    float camera_distance_scale = 1.0f;  // 1.0 the game's own; 10% a step either way
    float camera_height_scale = 1.0f;
    // The follow camera's tuning above 30 fps, read when the game starts
    // (engine/frame_rate.cpp): false FPS++, Kyo's 60 fps retune; true
    // Original, the game's own values. At 30 the camera is the game's either way.
    bool camera_original = false;

    // Controls
    bool mouse_menu = true;     // the pointer drives the menus through the pad
    bool mouse_camera = true;   // the mouse turns the camera outside menus
    int mouse_sens = 5;         // 0..10, DS3's: 0.2 * (s * 0.15 + 0.5) degrees a count (engine/mouse_camera.h)
    bool mouse_invert_x = false, mouse_invert_y = false;
    // While the mouse turns the camera (the keyboard and mouse used last): off
    // holds the camera's own turns as the character moves (engine/mouse_camera.h).
    bool mouse_auto_rotation = false;
    bool draw_cursor = true;    // the host draws its own pointer
    // What the game's own button prompts show: 0 follows whether a pad is
    // plugged in, 1 always the bound key, 2 always the pad's glyph
    // (engine/key_prompts.h).
    int button_prompts = 0;

    // Graphics (the CSGraphicsConfig-shaped part; engine/graphics_patch.cpp,
    // host/shader_patch.cpp apply them)
    bool ssao = true;
    bool motion_blur = true;
    bool anti_alias = true;
    bool depth_of_field = true;
    bool chromatic_aberration = true;
    bool vignette = true;
    float bloom = 1.0f;         // 1.0 the area's own glare, 0 none
    float saturation = 1.0f;    // 1.0 the area's own, 0 black and white, 2 twice
    float fog = 1.0f;           // 1.0 the area's own opacities, 0 clear
    float ao_strength = 1.0f;   // 1.0 the area's own, 0 none, 2 twice
    int lod_bias = 0;           // the game's model detail bias: -2 finest always, 0 its own, 2 coarsest
    float shadow_scale = 1.0f;  // the cascades stretched: 1.0 to 3.0

    // Startup (read once by the loader)
    bool skip_logos = false;
    bool performance_mode = false;  // the post-effects and YEBIS passes off (engine/patch_manifest.cpp)
    bool debug_menu = false;
    bool debug_camera = false;  // patches/debug-camera.toml (Lance McDonald's free camera)
    // The PC enhancements (also read once, through config_set_enhancements)
    bool change_appearance = true;
    bool rebirth = true;
    bool five_players = true;
    bool skill_issue = false;
};

// A copy of the current settings (no lock: the snapshot published last).
HostSettings host_settings();
// The same without the copy, for a path that reads a field or two: a
// published snapshot is never written again and is kept for the run, so the
// reference stays good; the next change publishes another.
const HostSettings& host_settings_ref();
// Tells `fn` the current settings at once and then the new ones after every
// change, on the thread that made it (host/settings_store.h has the rules: be
// quick, keep what is needed, change nothing). Up to 32 listeners.
using HostSettingsListener = void (*)(const HostSettings&);
void host_settings_listen(HostSettingsListener fn);
