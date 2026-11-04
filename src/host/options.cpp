#include "host/options.h"
#include "host/updater.h"
#include "host/settings.h"
#include "host/settings_store.h"

#include "host/bindings.h"
#include "host/overlay.h"
#include "host/window.h"
#include "core/config.h"
#include "engine/live_resolution.h"
#include "hle/modules.h"
#include "hle/np.h"
#include "net/account.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <string_view>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#if defined(BBHOST_HAVE_SDL3)
#include "host/launcher_ui.h"

#include "imgui.h"

#include <SDL3/SDL.h>
#endif

namespace {

struct Setting;
bool reads_on(const Setting& s, int index);  // below
int fps_from_label(const char* label);

// "-180°" to "+180°" a step at a time (`unit` after each number), for a
// slider of fine steps; the strings live as long as the program.
std::vector<const char*> signed_steps(int from, int to, const char* unit, bool sign = true) {
    // A deque: its strings never move as more are added, so the pointers
    // handed out stay good (a vector's growth moved short strings' bytes).
    static std::deque<std::string>* store = new std::deque<std::string>();
    std::vector<const char*> out;
    for (int v = from; v <= to; ++v) {
        store->push_back((sign && v > 0 ? "+" : "") + std::to_string(v) + unit);
        out.push_back(store->back().c_str());
    }
    return out;
}

// The steps of a 0-100 slider ("0" to "100").
std::vector<const char*> hundred() { return signed_steps(0, 100, "", false); }

// One setting. Everything here is a choice between a few named values: a
// toggle is a choice of two, which keeps drawing and input to one case.
struct Setting {
    const char* key;                 // what it is saved as
    const char* label;
    std::vector<const char*> values;
    int index = 0;
    const char* note = nullptr;      // shown under the row when it is selected
    bool restart = false;            // takes effect on the next run
};

enum SettingId {
    kWindowMode,
    kVsync,
    kFrameCap,
    kFrameClock,
    kEffectTiming,
    kFpsCounter,
    kMemoryCounter,
    kPerfOverlay,
    kFov,
    kCameraDistance,
    kCameraHeight,
    kCameraTuning,
    kMouseMenu,
    kMouseCamera,
    kMouseSens,
    kMouseInvertX,
    kMouseInvertY,
    kMouseAutoRotation,
    kDrawCursor,
    kButtonPrompts,
    kSsao,
    kMotionBlur,
    kAntiAlias,
    kDepthOfField,
    kChromaticAberration,
    kBloom,
    kVignette,
    kShadowDistance,
    kAoStrength,
    kSaturation,
    kFog,
    kModelDetail,
    kSsr,
    kHdMotionBlur,
    kHdMotionBlurRes,
    kUpscaler,
    kUpscaleQuality,
    kDlssPreset,
    kSharpness,
    kResolution,
    kShaders,
    kSkipLogos,
    kPerformanceMode,
    kDebugMenu,
    kDebugCamera,
    kChangeAppearance,
    kRebirth,
    kFivePlayers,
    kSkillIssue,
    kMenuTextSize,
    kMenuHue,
    kMenuVibrance,
    kSettingCount,
};

// Effect timing's choices as the frame rate reads them (core/config.h's
// config_effect_timing), in the setting's order: 30 fps, 60 fps (auto: 60
// above 30) and Frame rate-dependent. Smooth, a fourth until 2026-11-04, is
// still read from bbhost.toml or BBHOST_SFX_STEP (the row's Custom entry).
constexpr int kEffectTimingCount = 3;
constexpr const char* kEffectTimingValues[kEffectTimingCount] = {"30", "auto", "fps"};
constexpr int effect_timing_index(std::string_view v) {
    for (int i = 0; i < kEffectTimingCount; ++i) {
        if (v == kEffectTimingValues[i]) return i;
    }
    return -1;
}
static_assert(effect_timing_index(kEffectTimingDefault) >= 0, "the default effect timing is one of the setting's choices");

// A row on screen: a section header, a setting, or the close button.
struct Row {
    // Text: a field the player types (field 0 the account name, 1 a recovery
    // code). Action: a button (field 0 link with Discord, 1 create an account
    // on this PC, 2 recover, 3 website code, 4 sign out). Info: a line the
    // host writes (field 0 who is signed in, 1 the code in play, if any).
    // Fields 10 and up are the UPDATES rows (host/updater.h): Info 10 the
    // version and what the check found, 11 why it failed or the newer
    // release's page; Action 10 check, 11 open that page in the browser.
    enum Kind { Header, Option, Close, Text, Action, Info } kind;
    const char* text;
    int setting = -1;
    int field = -1;
};

Setting g_set[kSettingCount] = {
    // Borderless was this setting's Fullscreen until 2026-11-04 (an older
    // file's Fullscreen reads as Borderless: host_options_load).
    {"window_mode", "Window mode", {"Windowed", "Borderless", "Fullscreen"}, 0,
     "Borderless fills the screen at the desktop's resolution and switches away at once. Fullscreen takes the display "
     "over at the mode nearest the Resolution, which can change the screen's resolution.",
     false},
    {"vsync", "V-Sync", {"On", "Off"}, 0,
     "Off presents as soon as a frame is ready, and can tear.", false},
    // The game's own pace (engine/frame_rate.h), the same eleven the game's
    // Frame Rate pick list offers (engine/option_menu.cpp). The key is the old
    // Frame cap's, so a file keeps its 30 or 60; a file older than version 3
    // holds a cap on presentation, whose 90, 120, 144 and Off ran the game at
    // 60, and reads as 60 (host_options_load).
    {"frame_cap", "Frame rate", {"30", "60", "90", "120", "150", "180", "240", "360", "480", "960", "Display-Limited"}, 0,
     "30 is the game's own pace; the others run its logic at that rate. Display-Limited caps it at your monitor's refresh "
     "rate when the game starts (180 on a 180 Hz monitor) and runs it the same way. Applies on the next run.",
     false},
    // The frame clock (engine/frame_clock.h), experimental and off by default:
    // what the frame rate otherwise sets once for the rate chosen at start
    // follows each frame's own length. Read once as the game starts
    // (engine/frame_rate.cpp, config_frame_clock): bbhost.toml's [video]
    // frame_clock is its default, BBHOST_FRAME_CLOCK=0|1 wins over both.
    {"frame_clock", "Native frame clock (experimental)", {"Off", "On"}, 0,
     "Requires a restart. Timing follows each frame's real length, for PCs that do not hold their chosen frame rate.",
     true},
    // How the effects keep time above 30 fps (engine/frame_rate.cpp), read
    // once as the game starts: BBHOST_SFX_STEP, else this once chosen, else
    // bbhost.toml's [video] effect_steps, else kEffectTimingDefault, 60 fps
    // (config_effect_timing). Written to the file only once chosen (save).
    {"effect_timing", "Effect timing", {kHostEffectTimingLabels[0], kHostEffectTimingLabels[1], kHostEffectTimingLabels[2]},
     effect_timing_index(kEffectTimingDefault), kHostEffectTimingNote, true},
    // The game's frames a second, counted as they start (host/window.cpp,
    // host_fps_frame_start), with how busy the GPU is beside it.
    {"fps_counter", "FPS counter", {"Off", "On"}, 0,
     "The game's frames per second, in the top right corner.", false},
    // What bbhost holds of video memory and of the memory Windows counts as
    // shared GPU memory (host/window.cpp, host_gpu_memory_glance).
    {"memory_counter", "Memory counter", {"Off", "On"}, 0,
     "Video memory in use (how much is cache, given back when it runs short) and the game's memory the GPU reads, "
     "which Windows counts as shared GPU memory.", false},
    // The bottleneck meter's last second (host/perf_meter.h): what limits the
    // frame rate and what is next, and how busy the main loop, the command
    // processor and the GPU are. A diagnostic, off by default; it starts the
    // meter when Tooling and logging is off.
    {"perf_overlay", "Performance meter", {"Off", "On"}, 0,
     "What limits the frame rate - the GPU, the main thread, the GX workers, the command processor, the frame cap - "
     "and how busy each part is, under the FPS counter (docs/performance.md).", false},
    // The follow camera's vertical field of view, LockCamParam's camFovY
    // (43 degrees in most areas) widened 5% a step (engine/camera.h).
    {"fov", "Field of view", signed_steps(0, 100, "%"), 0,
     "How wide the camera sees. +0% is the game's own; +50% is about 97 degrees across, +100% about 118.", false},
    // The same row's camDistTarget and chrOrgOffset_Y: 5 the game's own, 10%
    // a step either way.
    {"camera_distance", "Camera distance", hundred(), 50,
     "How far behind the character the camera sits. 50 is the game's own.", false},
    {"camera_height", "Camera height", hundred(), 50,
     "How high above the character the camera looks. 50 is the game's own.", false},
    // The follow camera above 30 fps (engine/frame_rate.cpp camera_original):
    // the game's own values, the same camera at every frame rate, whatever is
    // saved here - the menus no longer offer it, and only
    // BBHOST_CAMERA_TUNING=fps++ runs the retune from Kyo's 60 FPS++ list. So
    // the default is Original, and a save records the camera that runs.
    {"camera_tuning", "Camera tuning", {"FPS++ (Kyo)", "Original"}, 1,
     "Requires a restart. Above 30 fps: FPS++ is Kyo's retuned camera; Original is the game's own, as at 30.",
     true},
    // On, as in DS3: a click only reaches the menus while one is open
    // (engine/menu_pointer.h), so it cannot swing a weapon in the world.
    {"mouse_menu", "Mouse in menus", {"Off", "On"}, 1,
     "Click confirms, right-click goes back, the wheel moves the cursor.", false},
    // On by default, and it only *disables* the automatic behaviour: the
    // mouse turns the camera whenever a menu is not open, which is what DS3
    // does and what a PC player expects. Off hands the camera back to the pad.
    {"mouse_camera", "Mouse camera", {"On", "Off"}, 0,
     "The mouse turns the camera outside menus. Off leaves it to the pad.", false},
    // 0..10, the range of the game's own Camera Sensitivity slider, which is
    // the widget the PC Controls screen draws it with, and DS3's: the same
    // angle a count at each step (engine/mouse_camera.h), 5 its default.
    {"mouse_sens", "Mouse sensitivity", hundred(), 50,
     "How far a given mouse movement turns the camera.", false},
    {"mouse_invert_x", "Reverse mouse X-axis", {"Off", "On"}, 0,
     "Reverses the mouse's horizontal camera movement.", false},
    {"mouse_invert_y", "Reverse mouse Y-axis", {"Off", "On"}, 0,
     "Reverses the mouse's vertical camera movement.", false},
    // Off by default: with the mouse the camera stays where it is put, as
    // the community's "Disable Camera Auto Rotation" patch has it - but only
    // while the keyboard and mouse were used last; a controller gets the
    // game's own camera back (engine/mouse_camera.h).
    {"mouse_auto_rotation", "Mouse auto-rotation", {"Off", "On"}, 0,
     "Off: while the mouse turns the camera, it stays where you put it instead of swinging behind you as you move. A controller always has the game's own camera.", false},
    {"draw_cursor", "Draw the pointer", {"On", "Off"}, 0,
     "The host draws it; the game has no cursor of its own.", false},
    // The game's key guide and its tutorial notes name a pad button as an
    // image tag; the port rewrites the tag into the key bound to it
    // (engine/key_prompts.h). Auto follows the device last used (host_input_device).
    {"button_prompts", "Button prompts", {"Auto", "Keyboard", "Controller"}, 0,
     "What the game's own prompts show. Auto follows the device you last used, keys or controller.", false},
    // The post-processing switches. They follow their settings live
    // (engine/graphics_patch.cpp, host/shader_patch.h).
    {"ssao", "Ambient occlusion", {"On", "Off"}, 0,
     "Screen-space ambient occlusion. Off is faster.", false},
    // Not a byte patch: the blur shader passes every pixel through
    // (host/shader_patch.h).
    {"motion_blur", "Motion blur", {"On", "Off"}, 0,
     "The blur on fast camera and character movement.", false},
    {"anti_alias", "Anti-aliasing", {"On", "Off"}, 0,
     "The game's own AA pass.", false},
    {"depth_of_field", "Depth of field", {"On", "Off"}, 0,
     "Blurs distant scenery; indoors there is rarely anything far enough to blur.", false},
    {"chromatic_aberration", "Chromatic aberration", {"On", "Off"}, 0,
     "The colour fringing at the edges of the frame.", false},
    // Two of the area's draw parameters (param/drawparam), scaled or cleared
    // as the game hands them to YEBIS each frame - engine/graphics_patch.cpp.
    // 10 is the area's own glare; 0 is none.
    {"bloom", "Bloom", hundred(), 100,
     "The glow around bright light. 100 is the game's own; lower dims it.", false},
    {"vignette", "Vignette", {"On", "Off"}, 0,
     "The darkened frame edges some areas use.", false},
    // The area's shadow cascades, stretched past the first split: 1 + 0.2 a
    // step, so the game's slider row (0..10) is the same setting.
    {"shadow_distance", "Shadow distance",
     {"x1.0", "x1.2", "x1.4", "x1.6", "x1.8", "x2.0", "x2.2", "x2.4", "x2.6", "x2.8", "x3.0"}, 0,
     "How far the sun and moon cast shadows. Farther spreads each shadow map wider.", false},
    // Two more of the area's numbers, scaled: 5 is the area's own, 0 none, 10
    // twice it.
    {"ao_strength", "AO strength", hundred(), 50,
     "How dark ambient occlusion makes corners and contact. 50 is the game's own.", false},
    {"saturation", "Saturation", hundred(), 50,
     "Colour. 50 is the game's own; 0 is black and white.", false},
    // The fog table's opacities (engine/graphics_patch.cpp): 10 the area's own.
    {"fog", "Fog", hundred(), 100,
     "How thick distance and height fog are. 100 is the game's own; 0 clears it.", false},
    // The game's level-of-detail bias (engine/graphics_patch.cpp): Full keeps
    // every model at its finest mesh, Low and Lowest move each to a coarser
    // one sooner. Lowest is the community "model LOD 2" (-8% GPU, models
    // visibly simpler up close).
    {"model_detail", "Model detail", {"Full", "Normal", "Low", "Lowest"}, 1,
     "Normal is the game's own. Low and Lowest swap in simpler models sooner; Full never does.", false},
    // The game's screen space reflections, off as it shipped
    // (engine/patch_manifest.cpp ssr_manifest): its reflection pass at the
    // game's own half size. Its targets are made as the game starts. The
    // first time, a bbhost.toml that turned the old patch file on
    // ([patches] screen-space-reflections) seeds it (host_options_load).
    {"ssr", "Screen space reflections", {"Off", "On"}, 0,
     "Requires a restart. Wet streets, water and polished floors reflect what is on screen, at the game's own half "
     "size, without fading out where they face the camera. Costs GPU time.",
     true},
    // Kyo's HD Motion Blur patches as a setting (engine/patch_manifest.cpp
    // hd_motion_blur_manifest): their buffers at a share of the render size
    // rather than at one fixed size each. The targets are made as the game
    // starts. A bbhost.toml that turned one of the old files on seeds it
    // (host_options_load).
    {"hd_motion_blur", "HD motion blur", {"Off", "On"}, 0,
     "Requires a restart. Motion blur's velocity map and the depth it blurs with at the HD motion blur resolution "
     "below instead of a fixed 640x360, with the camera's own blur on: cleaner, steadier blur. Costs GPU time and "
     "video memory, more the larger. Shows only with Motion blur on.",
     true},
    {"hd_motion_blur_res", "HD motion blur resolution", {"Quarter", "Half", "Full"}, 2,
     "Requires a restart. HD motion blur's velocity map against the render resolution: Quarter, Half (the game's own "
     "size) or Full. It follows the render resolution, Resolution and upscaling both.",
     true},
    // The temporal upscaler (host/upscaler.h, Numity1504's bridge): its
    // picture in place of the game's after the 3D, before the post effects.
    // Off by default; the order of the values is the settings object's
    // (0 off, then DLSS, FSR) turned round for the menu (Off, FSR, DLSS).
    {"upscaler", "Upscaling", {"Off", "FSR 4", "FSR 3", "DLSS"}, 0,
     "FSR 4 (AMD's newest, where the GPU and driver offer it), FSR 3 (any GPU) or DLSS (NVIDIA RTX) in place of the game's "
     "own anti-aliasing. Needs upscaler_bridge.dll and AMD's FidelityFX files beside bbhost.exe, and for DLSS nvngx_dlss.dll; "
     "without them it stays off. FSR 4 runs as FSR 3 where it cannot run. Changes at once; with Upscaling quality below "
     "Native AA, the smaller render size it draws the 3D at comes and goes at the next start.",
     false},
    // The upscaler's render size against the output (Resolution): Native AA
    // the same (DLAA, FSR native AA), Quality 1/1.5, Balanced about 1/1.7,
    // Performance 1/2 each way (engine/render_scale.cpp). The scene contexts
    // are made at it as the game starts, so it applies at the next start - as
    // does Upscaling turned on or off with a quality above Native AA, which
    // decides whether the 3D is drawn smaller at all.
    {"upscale_quality", "Upscaling quality", {"Native AA", "Quality", "Balanced", "Performance"}, 1,
     "Requires a restart. How much smaller than the Resolution the 3D is drawn before the upscaler brings it back: Native AA "
     "draws it full size (anti-aliasing only), Quality at two thirds, Balanced at about 59%, Performance at half each way - "
     "faster, softer. The menus, the HUD and the post effects stay at the Resolution. Below Native AA, a different Resolution "
     "also applies at the next start.",
     true},
    // DLSS's render preset (NGX's DLSS_Hint_Render_Preset for every quality;
    // needs upscaler_bridge 1.1.0 and nvngx_dlss 310 or newer): K the
    // transformer model and NVIDIA's own for DLAA, Quality and Balanced, J
    // like it with less ghosting and more flicker, L made for Ultra
    // Performance, M for Performance.
    {"dlss_preset", "DLSS preset", {"K", "J", "L", "M"}, 0,
     "Which of DLSS's models runs. K (recommended): the transformer model, the sharpest. J: like K, a little less "
     "ghosting, a little more flicker. L: made for the lowest render sizes. M: made for Performance. DLSS only.",
     false},
    // RCAS over the upscaler's picture (host/upscale_pass.cpp): FSR 3 and 4
    // through the bridge and DLSS do no sharpening of their own. 0% none, the
    // default (the picture as the upscaler leaves it); 100% AMD's most.
    {"sharpness", "Sharpness", signed_steps(0, 100, "%", false), 0,
     "Sharpens the upscaled 3D (FSR or DLSS) before the post effects and the HUD, which it never touches. 0% leaves the "
     "upscaler's picture as it is; higher brings out more detail, at 100% with some ringing on hard edges. Needs Upscaling on.",
     false},
    // The render resolution. res_width and res_height are two int32s the eboot
    // never writes, so this is a data patch and every pass scales together -
    // see engine/graphics_patch.cpp. The game's pick list shows six entries
    // and scrolls through more. In order of size within each shape - 16:9,
    // then 16:10, 4:3, 21:9 and 32:9 - and saved by name, not by place, so a
    // saved entry survives the order changing (engine/option_menu.cpp's
    // kResolutions and host/launcher.cpp's labels follow this list). The
    // other shapes are engine/live_resolution.cpp's. 960x600 and 1024x640 are
    // the Deck's screen at 75% and 80%: it holds 60 fps there (pixel-bound at
    // 1280x800, ~45), and the presenter upscales by FSR 1.
    {"resolution", "Resolution", {"640x360", "960x540", "1280x720", "1440x810", "1600x900",
                                  "1920x1080", "2560x1440", "2880x1620", "3200x1800", "3840x2160",
                                  "960x600", "1024x640", "1280x800", "1920x1200", "2560x1600",
                                  "1280x960", "1440x1080", "2560x1080", "3440x1440", "5120x2160",
                                  "3840x1080", "5120x1440"}, 5,
     "What the game renders at. 1920x1080 is what it shipped with; wide and 16:10 screens are filled, the HUD kept 16:9. "
     "The largest sizes can need a restart: the memory for them is set aside when the game starts.",
     true},
    // The shaders' form on the GPU (host/translation_cache.h): HLSL is the
    // lifted shaders printed as HLSL and compiled by DXC on this machine,
    // taken from the cache while the rest compile behind the game; Lifted is
    // the lifter's own SPIR-V alone. The same pictures either way.
    {"shaders", "Shaders", {"HLSL", "Lifted"}, 0,
     "Requires a restart. HLSL (recommended) also builds an optimized version of each of the game's shaders in the background "
     "while you play, used from the next start. Lifted uses bbhost's direct translation only. The picture is the same either way.",
     true},
    {"skip_logos", "Skip company logos", {"Off", "On"}, 1,
     "The three logos before the title screen, and the warning that the last session did not end with Exit Game.", true},
    // Kyo's Giga Performance Patch's post-effects (engine/patch_manifest.cpp's
    // performance-mode), and chromatic aberration through its own setting's mask.
    {"performance_mode", "Performance mode", {"Off", "On"}, 0,
     "Requires a restart. Every post-effect and YEBIS pass off for frame rate: SSAO, motion blur, anti-aliasing, depth of field, chromatic aberration, bloom, and the dynamic lights' shadows (lamps, torches, the lantern; the sun's and the moon's stay). The map's fog stays.", true},
    // The old switch for the developers' debug menu, which the Debug Menu
    // plugin replaced (engine/debug_menu.h): no row shows it, and an "On" an
    // older bbhost saved turns the plugin on once (main.cpp).
    {"debug_menu", "Debug menu", {"Off", "On"}, 0, "Replaced by the Debug Menu plugin.", true},
    // Lance McDonald's "Restore Debug Camera" (patches/debug-camera.toml). Its
    // code takes the place of a debug-only load-test step, so the debug menu's
    // LOAD TEST and DUNGEON MOVEMAP TEST crash while it is on.
    {"debug_camera", "Debug camera", {"Off", "On"}, 0,
     "Requires a restart. Hold Interact (E) and press L3 (Left Ctrl) to cycle the free camera's modes.", true},
    // The PC enhancements: additions to the game itself, each read once at
    // start (main.cpp hands them to config_set_enhancements), all on unless
    // bbhost.toml's [world] says otherwise. Off, all three, is the game as it
    // shipped. (The Old Hunters, a fourth in v0.2.9, is the patch old-hunters.)
    {"change_appearance", "Hunter's Dream mirror", {"On", "Off"}, 0,
     "Requires a restart. The Dream's mirror opens the appearance editor (the unused \"Put on Disguise\").", true},
    {"rebirth", "Rebirth at the Altar", {"On", "Off"}, 0,
     "Requires a restart. With the Yharnam Stone, the Altar of Despair resets level and attributes.", true},
    {"five_players", "Five players", {"On", "Off"}, 0,
     "Requires a restart. Two invaders in Mensis, Mergo's Loft, the Frontier and the DLC; all players need it.", true},
    // Off unless bbhost.toml's [world] turns it on: a joke, not an addition.
    {"skill_issue", "SKILL ISSUE", {"On", "Off"}, 0, "Requires a restart. The death message says SKILL ISSUE instead of YOU DIED.",
     true},
    // bbhost's own menus - the setup window, F9 and F10 - and everything in
    // them grow with it (launcher_ui set_scale). 100% is their usual size: the
    // in-game menus' follows the game's resolution (height / 900), the setup
    // window's is 1120x720 at Windows' scaling, and grows (host/launcher.cpp).
    {"menu_text_size", "Menu text size", {"100%", "125%", "150%", "175%", "200%", "250%", "300%", "350%", "400%"}, 0,
     "The text in bbhost's own menus (this one, F9 and the setup window), and the menus with it. 100% is the usual size; "
     "the setup window grows with it too. A menu larger than the screen scrolls.",
     false},
    // The colours of the same menus (launcher_ui recolour): 0 degrees and 0 percent
    // are their own blue and violet. Hue turns every colour in them round the
    // colour wheel but the green, amber and red that mean ok, a warning and a
    // problem; vibrance takes colour out (-100%: black and white) or adds it.
    {"menu_hue", "Menu hue", signed_steps(-180, 180, "°"), 180, "Turns the colours of bbhost's own menus (this one, F9 and the setup window) round the colour wheel. 0 degrees "
     "is their own blue and violet.",
     false},
    {"menu_vibrance", "Menu vibrance", signed_steps(-100, 100, "%"), 100, "How much colour bbhost's own menus have: -100% is black and white, 0% their own, +100% the most.", false},
};

const Row g_rows[] = {
    {Row::Header, "DISPLAY"},
    {Row::Option, nullptr, kWindowMode},
    {Row::Option, nullptr, kVsync},
    {Row::Option, nullptr, kFrameCap},
    {Row::Option, nullptr, kFrameClock},
    {Row::Option, nullptr, kEffectTiming},
    {Row::Option, nullptr, kFpsCounter},
    {Row::Option, nullptr, kMemoryCounter},
    {Row::Option, nullptr, kPerfOverlay},
    {Row::Option, nullptr, kMenuTextSize},
    {Row::Option, nullptr, kMenuHue},
    {Row::Option, nullptr, kMenuVibrance},
    {Row::Option, nullptr, kFov},
    {Row::Option, nullptr, kCameraDistance},
    {Row::Option, nullptr, kCameraHeight},
    {Row::Header, "INPUT"},
    {Row::Option, nullptr, kMouseMenu},
    {Row::Option, nullptr, kMouseCamera},
    {Row::Option, nullptr, kMouseSens},
    {Row::Option, nullptr, kMouseInvertX},
    {Row::Option, nullptr, kMouseInvertY},
    {Row::Option, nullptr, kMouseAutoRotation},
    {Row::Option, nullptr, kDrawCursor},
    {Row::Option, nullptr, kButtonPrompts},
    {Row::Header, "GRAPHICS"},
    {Row::Option, nullptr, kSsao},
    {Row::Option, nullptr, kMotionBlur},
    {Row::Option, nullptr, kHdMotionBlur},
    {Row::Option, nullptr, kHdMotionBlurRes},
    {Row::Option, nullptr, kAntiAlias},
    {Row::Option, nullptr, kDepthOfField},
    {Row::Option, nullptr, kChromaticAberration},
    {Row::Option, nullptr, kSsr},
    {Row::Option, nullptr, kResolution},
    {Row::Option, nullptr, kUpscaler},
    // The preset row (preset_row): for FSR its preset - the Upscaling quality
    // setting, which has no row of its own - for DLSS that quality and DLSS's
    // preset, greyed with Upscaling off. The Resolution the upscaler brings the
    // 3D back to above them, Sharpness over the upscaler's picture below.
    {Row::Option, nullptr, kDlssPreset},
    {Row::Option, nullptr, kSharpness},
    {Row::Option, nullptr, kBloom},
    {Row::Option, nullptr, kVignette},
    {Row::Option, nullptr, kShadowDistance},
    {Row::Option, nullptr, kAoStrength},
    {Row::Option, nullptr, kSaturation},
    {Row::Option, nullptr, kFog},
    {Row::Option, nullptr, kModelDetail},
    {Row::Option, nullptr, kShaders},
    {Row::Header, "PC ENHANCEMENTS"},
    {Row::Option, nullptr, kChangeAppearance},
    {Row::Option, nullptr, kRebirth},
    {Row::Option, nullptr, kFivePlayers},
    {Row::Option, nullptr, kSkillIssue},
    {Row::Header, "STARTUP"},
    {Row::Option, nullptr, kSkipLogos},
    {Row::Option, nullptr, kPerformanceMode},
    // The private server's account. No passwords: link this PC to
    // the account signed into on the website with Discord, or make one held
    // by this PC with a recovery code. The token is kept in the options file.
    {Row::Header, "ACCOUNT"},
    {Row::Info, nullptr, -1, 0},
    {Row::Info, nullptr, -1, 1},
    {Row::Action, "Link with Discord", -1, 0},
    {Row::Text, "Name", -1, 0},
    {Row::Action, "Create account on this PC", -1, 1},
    {Row::Text, "Recovery code", -1, 1},
    {Row::Action, "Recover account", -1, 2},
    {Row::Action, "Website sign-in code", -1, 3},
    {Row::Action, "Sign out", -1, 4},
    {Row::Header, "UPDATES"},
    {Row::Info, nullptr, -1, 10},
    {Row::Info, nullptr, -1, 11},
    {Row::Action, "Check for updates", -1, 10},
    {Row::Action, "Open the release page", -1, 11},
    {Row::Close, "Close"},
};
constexpr int kRowCount = static_cast<int>(sizeof(g_rows) / sizeof(g_rows[0]));

std::atomic<bool> g_open{false};
// The in-game menu's changes (host_options_draw_page, on the presenter's
// thread), run by the window's thread at its next pump (host_options_frame):
// a setting can touch the window (fullscreen) and the file.
std::mutex g_post_mu;
std::vector<std::function<void()>> g_posted;  // under g_post_mu
// What the menu set and the pump has not applied yet, by setting (-1 none):
// the menu shows it at once, not the value a pump later.
std::array<int, kSettingCount> g_pending = [] {
    std::array<int, kSettingCount> a{};
    a.fill(-1);
    return a;
}();
bool g_dirty = false;  // something changed since the last save
// Which settings are the player's own: the options file held a value for it
// when it was read, or it was set since (set_index); the rest keep their
// defaults, the table's or bbhost.toml's. For host_opt_saved, and save's
// Effect timing.
std::array<bool, kSettingCount> g_saved{};

// What every setting starts with: the table's own indices, taken before
// anything is loaded over them, then whatever bbhost.toml sets for it - the
// frame cap, the logos, and a package's resolution and window mode (the
// Steam Deck kit's) - so the menus' Defaults go back to those.
std::array<int, kSettingCount> g_default_index = [] {
    std::array<int, kSettingCount> a{};
    for (int id = 0; id < kSettingCount; ++id) a[static_cast<std::size_t>(id)] = g_set[id].index;
    return a;
}();

// A resolution change made while the game runs (engine/live_resolution.h)
// asks to be kept, the way Dark Souls III's ApplyScreenSettingJob does: the
// new size at once, then the game's own yes/no dialog, ten seconds, or it
// goes back. While it waits, g_res_prev is the entry it goes back to and the
// file keeps.
int g_res_prev = -1;
bool g_res_reverting = false;  // the change back itself asks nothing

// Whether the resolution entry `index` changes while the game runs. One past
// what this run's render-target heap was grown for at its start (5120x2160
// after a smaller start) applies on the next run like any restart setting.
bool res_live(int index) {
    unsigned w = 0, h = 0;
    return live_resolution_available() && std::sscanf(g_set[kResolution].values[static_cast<std::size_t>(index)], "%ux%u", &w, &h) == 2 &&
           live_resolution_can(w, h);
}

// The account rows. The calls run on their own thread so the menu keeps
// drawing, and their outcome is the two status lines.
std::string g_field[2];       // name, recovery code (never saved)
std::mutex g_acct_mu;
std::string g_acct_status;    // the last outcome, for the first status line
std::string g_acct_detail;    // a code the player needs (link, recovery, website)
std::atomic<bool> g_acct_busy{false};
std::atomic<bool> g_acct_save{false};    // signed in or out: the token wants saving
std::atomic<bool> g_link_cancel{false};  // stops a link that is waiting for approval

void set_account_lines(const std::string& status, const std::string& detail) {
    std::lock_guard<std::mutex> lk(g_acct_mu);
    g_acct_status = status;
    g_acct_detail = detail;
}

std::string signed_in_line() {
    // The game read its NpId when it went online, so a sign-in after that
    // names the account to the server but not to the game until the next run.
    if (net::account_refused())
        return "Signed in as " + net::account_name() + " - the server refused it (expired or replaced): link this PC again";
    return "Signed in as " + net::account_name() +
           (hle_np_context_started() ? " (in full on the next launch)" : "");
}

// Link with Discord: a code the player approves on the website's /link page,
// signed in there with Discord; this thread polls until it is approved,
// denied or expired.
void link_thread() {
    net::DeviceLink link;
    std::string err;
    if (!net::account_link_start(link, err)) {
        set_account_lines("Link failed: " + err, "");
        g_acct_busy.store(false);
        return;
    }
    const std::string uri = link.verification_uri + "?code=" + link.user_code;
    set_account_lines("Waiting for approval on the website...",
                      "Code " + link.user_code + "  -  approve it at " + link.verification_uri);
    SDL_OpenURL(uri.c_str());
    int interval = link.interval;
    for (int waited = 0; waited < link.expires_in && !g_link_cancel.load();) {
        SDL_Delay(static_cast<Uint32>(interval) * 1000u);
        waited += interval;
        const std::string status = net::account_link_poll(link, err);
        if (status == "pending") continue;
        if (status == "slow_down") {
            ++interval;
            continue;
        }
        if (status == "approved") {
            set_account_lines(signed_in_line(), "");
            g_acct_save.store(true);
        } else {
            set_account_lines(status == "denied" ? "The link was denied on the website"
                              : status == "expired" ? "The code expired; link again"
                                                    : "Link failed: " + err, "");
        }
        g_acct_busy.store(false);
        return;
    }
    set_account_lines(g_link_cancel.load() ? "Link cancelled" : "The code expired; link again", "");
    g_acct_busy.store(false);
}

// The UPDATES buttons, and whether each can do anything right now.
void update_action(int which) {
    if (which == 10) updater::check_now();
    if (which == 11) updater::open_release_page();
}

bool action_idle(int which) {
    if (which < 10) return !g_acct_busy.load();
    if (updater::busy()) return false;
    if (which == 11) return updater::update_available();
    return true;
}

void account_action(int which, const std::string& name, const std::string& code) {
    if (which >= 10) {
        update_action(which);
        return;
    }
    if (which == 4) {  // sign out, also cancelling a link in progress
        g_link_cancel.store(true);
        std::thread([] {
            net::account_logout();
            set_account_lines("Signed out", "");
            g_acct_save.store(true);
        }).detach();
        return;
    }
    if (g_acct_busy.load()) return;
    if ((which == 1 || which == 2) && name.empty()) {
        set_account_lines("Type a name first", "");
        return;
    }
    if (which == 2 && code.empty()) {
        set_account_lines("Type the recovery code first", "");
        return;
    }
    if (which == 3 && !net::account_logged_in()) {
        set_account_lines("Sign in first", "");
        return;
    }
    g_acct_busy.store(true);
    g_link_cancel.store(false);
    if (which == 0) {
        set_account_lines("Asking the server for a code...", "");
        std::thread(link_thread).detach();
        return;
    }
    set_account_lines(which == 1 ? "Creating the account..." : which == 2 ? "Recovering..." : "Asking for a code...", "");
    std::thread([which, name, code] {
        std::string err, out;
        bool ok = false;
        if (which == 1) {
            ok = net::account_create(name, out, err);
        } else if (which == 2) {
            ok = net::account_recover(name, code, out, err);
        } else {
            ok = net::account_web_code(out, err);
        }
        if (!ok) {
            // A server without accounts says so itself; anything else is the
            // server's refusal (a name taken, a daily limit) in its words.
            set_account_lines(err.find("no account service") != std::string::npos ? err : "Refused: " + err, "");
        } else if (which == 3) {
            set_account_lines(signed_in_line(), "Website code " + out + "  -  use it on the account page within 10 minutes");
        } else {
            // The recovery code is shown here once and never written anywhere.
            set_account_lines(signed_in_line(), "Recovery code " + out + "  -  write it down now; it is shown once");
            g_field[1].clear();
            g_acct_save.store(true);
        }
        g_acct_busy.store(false);
    }).detach();
}

std::string account_status() {
    std::lock_guard<std::mutex> lk(g_acct_mu);
    if (!g_acct_status.empty()) return g_acct_status;
    return net::account_logged_in() ? "Signed in as " + net::account_name() : "Not signed in";
}

std::string account_detail() {
    std::lock_guard<std::mutex> lk(g_acct_mu);
    return g_acct_detail;
}

// The frame-rate values are the labels themselves: "Display-Limited" is 0,
// anything else is the number.
int fps_from_label(const char* v) { return std::atoi(v); }

int fps_index(int fps) {
    for (std::size_t v = 0; v < g_set[kFrameCap].values.size(); ++v) {
        if (fps_from_label(g_set[kFrameCap].values[v]) == fps) {
            return static_cast<int>(v);
        }
    }
    return -1;
}

std::string options_path() {
    // BBHOST_OPTIONS_PATH: a file of its own, so a scripted test that changes
    // settings (tools/menu_drive.sh) never writes the player's.
    if (const char* e = std::getenv("BBHOST_OPTIONS_PATH"); e && e[0]) {
        return e;
    }
    // The per-user folder (config_user_dir), beside the per-user config: the
    // F10 settings and the account outlive any copy of bbhost. A --config is a
    // profile with a file of its own (bbhost-options-<name>.toml), so a dev
    // server's account never mixes with the live one's. The file from before
    // (beside that config, or the config a package or tree used, or in the
    // working directory) is copied over the first time: nobody is signed out.
    static const std::string path = [] {
        const HostConfig& c = config();
        const std::string file = c.profile.empty() ? std::string("bbhost-options.toml") : "bbhost-options-" + c.profile + ".toml";
        // One beside the exe wins, as a bbhost.toml there does
        // (config_user_file): a copy of bbhost kept with settings of its own.
        // Not in a test's folder of its own (BBHOST_CONFIG_DIR).
        const char* sandbox = std::getenv("BBHOST_CONFIG_DIR");
        if (!(sandbox && sandbox[0])) {
            const std::string local = config_exe_dir() + "/" + file;
            std::ifstream here(local);
            if (here) return local;
        }
        const std::string mine = config_user_dir() + "/" + file;
        std::ifstream have(mine);
        if (have) return mine;
        const auto beside = [](const std::string& f) {
            const std::size_t slash = f.find_last_of("/\\");
            return (slash == std::string::npos ? std::string() : f.substr(0, slash + 1)) + "bbhost-options.toml";
        };
        std::vector<std::string> old;
        if (!c.profile.empty()) {
            old.push_back(beside(c.profile_config));
        } else {
            for (const char* f : {"bbhost.toml"}) old.push_back(beside(f));
            old.push_back(beside(config_exe_dir() + "/bbhost.toml"));
        }
        // A folder of its own (BBHOST_CONFIG_DIR: every test harness) takes the
        // settings but not the [account]: one old file beside several test
        // configs signed each of their profiles in as that player, with a
        // token long replaced, and the server refused them all day.
        const char* own_dir = std::getenv("BBHOST_CONFIG_DIR");
        const bool keep_account = !(own_dir && own_dir[0]);
        for (const std::string& f : old) {
            if (f == mine) continue;
            std::ifstream in(f, std::ios::binary);
            if (!in) continue;
            std::ofstream out(mine, std::ios::binary);
            bool in_account = false, dropped = false;
            for (std::string line; std::getline(in, line);) {
                if (!line.empty() && line[0] == '[') in_account = line.rfind("[account]", 0) == 0;
                if (in_account && !keep_account) {
                    dropped = true;
                    continue;
                }
                out << line << '\n';
            }
            if (out) host_log("options: copied %s to %s (the per-user settings every copy of bbhost reads)%s", f.c_str(),
                              mine.c_str(), dropped ? ", not its account (a folder of its own)" : "");
            break;
        }
        return mine;
    }();
    return path;
}

// The resolution as width and height: BBHOST_RES's when it is set (it wins
// here too, or applying the settings at the start undid it), else the
// setting's.
bool resolution_of(unsigned* w, unsigned* h) {
    const char* forced = std::getenv("BBHOST_RES");
    return (forced && forced[0] && std::sscanf(forced, "%ux%u", w, h) == 2) ||
           std::sscanf(g_set[kResolution].values[static_cast<std::size_t>(g_set[kResolution].index)], "%ux%u", w, h) == 2;
}

// Applied every time one changes, and once at load, so there is one path to
// the live state instead of a copy in each caller.
void apply(int id) {
    const Setting& s = g_set[id];
    switch (id) {
        case kWindowMode: {
            unsigned w = 0, h = 0;
            resolution_of(&w, &h);
            host_window_set_mode(s.index, static_cast<int>(w), static_cast<int>(h));
            break;
        }
        case kVsync: host_window_set_vsync(s.index == 0); break;
        case kFrameCap:
            hle_video_set_fps_cap(fps_from_label(s.values[static_cast<std::size_t>(s.index)]));
            break;
        case kMouseMenu:
        case kMouseCamera:
        case kMouseSens:
        case kMouseInvertX:
        case kMouseInvertY:
        case kMouseAutoRotation:
        case kDrawCursor:
        case kButtonPrompts:
            break;  // the settings object carries them (host/settings.h)
        case kSsao:
        case kMotionBlur:
        case kAntiAlias:
        case kDepthOfField:
        case kChromaticAberration:
        case kResolution: {
            unsigned w = 0, h = 0;
            // A change the player made asks to be kept (g_res_prev set by
            // set_index); going back, and the start, ask nothing.
            if (resolution_of(&w, &h)) {
                live_resolution_request(w, h, g_res_prev >= 0 && !g_res_reverting);
                // Fullscreen's display mode follows the resolution.
                if (g_set[kWindowMode].index == 2) host_window_set_mode(2, static_cast<int>(w), static_cast<int>(h));
            }
            break;
        }
        case kSkipLogos:
        case kPerformanceMode:
        case kDebugMenu:
        case kDebugCamera:
        case kChangeAppearance:
        case kRebirth:
        case kFivePlayers:
        case kSkillIssue:
            break;  // read by the loader before any of this exists
        case kMenuTextSize:
        case kMenuHue:
        case kMenuVibrance:
            break;  // the menus read them every frame (host_opt_menu_text_scale, host_opt_menu_colour)
        default: break;
    }
}

// The file's format. 2: Mouse in menus defaults to On. A file without a
// version was written by a build where it defaulted to Off, and every save
// writes every setting - so its "Off" is that old default far more often
// than a choice, and reads as On (once: the next save writes version 2).
// 3: frame_cap is the game's frame rate (30, 60, 90, Uncapped; 120 since
// 2026-10-10 and 150 to 960 since 2026-10-28, new values, so the version
// stays - and Uncapped, the display's refresh rate since then, is called
// Display-Limited: a saved "Uncapped" reads as it); an older file's 90, 120,
// 144 and Off were caps on presentation that ran the game at 60, and read as
// 60. camera_tuning (2026-10-10) is a new key whose default, FPS++, is what
// every older build ran, so a file without it reads as before and the
// version stays. 4: the camera zoom actions (zoom_in, zoom_out, zoom_reset on
// the wheel and the middle button); an older file's Transform on the middle
// button, its default until then, gives it to Camera Zoom Reset. 7: Window
// mode's Borderless and an exclusive Fullscreen; an older file's Fullscreen
// was the borderless one.
constexpr int kOptionsVersion = 7;

void save() {
    if (!g_dirty) {
        return;
    }
    const std::string path = options_path();
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        host_log("options: cannot write %s", path.c_str());
        return;
    }
    std::fprintf(f, "# bbhost PC port options. Written by the options screen (F10).\n");
    std::fprintf(f, "[options]\n");
    std::fprintf(f, "version = \"%d\"\n", kOptionsVersion);
    for (const Setting& s : g_set) {
        // Effect timing only once it is the player's own (g_saved): written at
        // every save like the rest, it would hold every player at the default
        // of the day they first saved - a new kEffectTimingDefault would reach
        // nobody - and take over from a [video] effect_steps the four choices
        // cannot show (a number of steps, frame) at the first save.
        if (&s == &g_set[kEffectTiming] && !g_saved[kEffectTiming]) continue;
        // A resolution still waiting for its keep-or-revert answer is not
        // saved: the file keeps the one it would go back to.
        const int index = &s == &g_set[kResolution] && g_res_prev >= 0 ? g_res_prev : s.index;
        std::fprintf(f, "%s = \"%s\"\n", s.key, s.values[static_cast<std::size_t>(index)]);
    }
    host_bindings_save(f);
    // The account: its name and token.
    if (net::account_logged_in()) {
        std::fprintf(f, "[account]\nname = \"%s\"\ntoken = \"%s\"\n", net::account_name().c_str(),
                     net::account_token().c_str());
    }
    std::fclose(f);
    g_dirty = false;
    host_log("options: saved %s", path.c_str());
}

// The settings object (host/settings.h): every field from the table, scaled
// the way its consumer wants it, published again whenever a setting changes
// (its serial is host_opt_serial) and told to the listeners. Made on first
// use, so a listener may register from anywhere at any time, and never
// destroyed: threads still reading it while the process exits read memory
// that is still there.
HostSettingsStore& settings_store() {
    static HostSettingsStore& store = *new HostSettingsStore;
    return store;
}

bool on_of(int id) { return reads_on(g_set[id], g_set[id].index); }
int index_of(int id) { return g_set[id].index; }

void rebuild_settings() {
    HostSettings h;
    h.fullscreen = index_of(kWindowMode) != 0;  // Borderless or Fullscreen
    h.vsync = index_of(kVsync) == 0;
    h.frame_cap = fps_from_label(g_set[kFrameCap].values[static_cast<std::size_t>(index_of(kFrameCap))]);
    h.frame_clock = on_of(kFrameClock);
    h.effect_timing = kEffectTimingValues[std::clamp(index_of(kEffectTiming), 0, kEffectTimingCount - 1)];
    h.fps_counter = on_of(kFpsCounter);
    h.memory_counter = on_of(kMemoryCounter);
    h.perf_overlay = on_of(kPerfOverlay);
    {
        // BBHOST_RES=WxH still wins, so an A/B run started with one behaves
        // the way its command line says.
        unsigned w = 0, hh = 0;
        const char* e = std::getenv("BBHOST_RES");
        if (!(e && e[0] && std::sscanf(e, "%ux%u", &w, &hh) == 2)) {
            const Setting& r = g_set[kResolution];
            if (std::sscanf(r.values[static_cast<std::size_t>(r.index)], "%ux%u", &w, &hh) != 2) w = hh = 0;
        }
        if (w && hh) {
            h.res_width = static_cast<int>(w);
            h.res_height = static_cast<int>(hh);
        }
    }
    // Distance and height: 50 is the game's own, 1% a step either way. Field
    // of view: 0 is the game's own, 1% wider a step (BBHOST_FOV overrides).
    const auto around = [](int id) { return 0.5f + 0.01f * static_cast<float>(index_of(id)); };
    h.camera_distance_scale = around(kCameraDistance);
    h.camera_height_scale = around(kCameraHeight);
    h.camera_original = index_of(kCameraTuning) == 1;
    {
        const char* e = std::getenv("BBHOST_FOV");
        const float env = e && *e ? std::strtof(e, nullptr) : 0.0f;
        const int fov = index_of(kFov);
        h.fov_scale = env > 0.0f ? env : fov <= 0 ? 1.0f : 1.0f + 0.01f * static_cast<float>(fov);
    }
    h.mouse_menu = on_of(kMouseMenu);
    h.mouse_camera = on_of(kMouseCamera);
    // DS3's mouse sensitivity (engine/mouse_camera.h): its 0..10 and its
    // default, a tenth a step (0-100).
    h.mouse_sens = index_of(kMouseSens) < 0 ? 5.0f : static_cast<float>(index_of(kMouseSens)) / 10.0f;
    h.mouse_invert_x = on_of(kMouseInvertX);
    h.mouse_invert_y = on_of(kMouseInvertY);
    h.mouse_auto_rotation = on_of(kMouseAutoRotation);
    h.draw_cursor = on_of(kDrawCursor);
    h.button_prompts = index_of(kButtonPrompts) < 0 ? 0 : index_of(kButtonPrompts);
    h.ssao = on_of(kSsao);
    h.motion_blur = on_of(kMotionBlur);
    h.anti_alias = on_of(kAntiAlias);
    h.depth_of_field = on_of(kDepthOfField);
    h.chromatic_aberration = on_of(kChromaticAberration);
    h.vignette = on_of(kVignette);
    h.bloom = static_cast<float>(index_of(kBloom)) / 100.0f;
    h.saturation = static_cast<float>(index_of(kSaturation)) / 50.0f;  // 50 the area's own
    h.fog = static_cast<float>(index_of(kFog)) / 100.0f;
    h.ao_strength = static_cast<float>(index_of(kAoStrength)) / 50.0f;
    h.shadow_scale = 1.0f + 0.2f * static_cast<float>(index_of(kShadowDistance));
    {
        static constexpr int kBias[] = {-2, 0, 1, 2};
        const int i = index_of(kModelDetail);
        h.lod_bias = i >= 0 && i < 4 ? kBias[i] : 0;
    }
    h.ssr = on_of(kSsr);
    h.hd_motion_blur = on_of(kHdMotionBlur);
    h.hd_motion_blur_res = std::clamp(index_of(kHdMotionBlurRes), 0, 2);
    {
        const int u = index_of(kUpscaler);
        h.upscaler = u == 1 ? 2 : u == 2 ? 3 : u == 3 ? 1 : 0;  // the menu's FSR 4, FSR 3, DLSS
        h.upscale_quality = std::clamp(index_of(kUpscaleQuality), 0, 3);
        static constexpr int kPresets[] = {11, 10, 12, 13};  // the menu's K, J, L, M
        h.dlss_preset = kPresets[std::clamp(index_of(kDlssPreset), 0, 3)];
        h.sharpness = static_cast<float>(std::clamp(index_of(kSharpness), 0, 100)) / 100.0f;
    }
    h.skip_logos = on_of(kSkipLogos);
    h.performance_mode = on_of(kPerformanceMode);
    h.debug_menu = on_of(kDebugMenu);
    h.debug_camera = on_of(kDebugCamera);
    h.change_appearance = on_of(kChangeAppearance);
    h.rebirth = on_of(kRebirth);
    h.five_players = on_of(kFivePlayers);
    h.skill_issue = on_of(kSkillIssue);
    settings_store().publish(h);
}

void set_index(int id, int index, bool from_file) {
    Setting& s = g_set[id];
    const int n = static_cast<int>(s.values.size());
    index = ((index % n) + n) % n;
    if (index == s.index && from_file) {
        return;
    }
    if (id == kResolution && !from_file && !g_res_reverting && index != s.index && res_live(index)) {
        if (g_res_prev < 0) g_res_prev = s.index;  // a second change while one waits goes back to the first's start
        if (index == g_res_prev) g_res_prev = -1;  // back where it started: nothing to ask
    }
    s.index = index;
    rebuild_settings();
    if (!from_file) {
        g_dirty = true;
        g_saved[static_cast<std::size_t>(id)] = true;
    }
    apply(id);
}

void res_keep() {
    if (g_res_prev < 0) return;
    host_log("options: resolution %s kept", g_set[kResolution].values[static_cast<std::size_t>(g_set[kResolution].index)]);
    g_res_prev = -1;
    g_dirty = true;
    save();
}

void res_revert() {
    if (g_res_prev < 0) return;
    const int prev = g_res_prev;
    g_res_prev = -1;
    host_log("options: resolution back to %s", g_set[kResolution].values[static_cast<std::size_t>(prev)]);
    g_res_reverting = true;
    set_index(kResolution, prev, false);
    g_res_reverting = false;
    // An entry waiting for the next start (res_live) cannot be gone back to
    // now: the picture goes back to the size it had before the change.
    if (!res_live(prev)) live_resolution_go_back();
    save();
}

}  // namespace



void host_options_apply() {
    for (int id = 0; id < kSettingCount; ++id) {
        apply(id);
    }
}

void host_options_load() {
    // Once: the setup window (host/launcher.h) loads them before it shows them.
    static bool loaded = false;
    if (loaded) return;
    loaded = true;
    // bbhost.toml still sets the defaults; the options file is what the player
    // changed, and the environment switches still win over both.
    host_bindings_load_defaults();
    g_set[kSkipLogos].index = config().skip_intro ? 1 : 0;
    // Screen space reflections were a patch file until they became a setting:
    // a bbhost.toml that turned it on keeps them on until the player chooses.
    if (const std::string v = config_value("patches.screen-space-reflections"); !v.empty())
        g_set[kSsr].index = v == "false" || v == "0" ? 0 : 1;
    // So were the HD Motion Blur patches, each at one size: the one a
    // bbhost.toml turned on becomes HD motion blur at the share nearest it
    // (360p a quarter of 1440p, 1080p and up the full size).
    for (const auto& [file, res] : {std::pair{"360p", 0}, std::pair{"1080p", 2}, std::pair{"1440p", 2}, std::pair{"4k", 2}}) {
        const std::string v = config_value(std::string("patches.hd-motion-blur-") + file);
        if (v.empty() || v == "false" || v == "0") continue;
        g_set[kHdMotionBlur].index = 1;
        g_set[kHdMotionBlurRes].index = res;
    }
    // The PC enhancements seed from bbhost.toml's [world] (all on unless it
    // says otherwise); the options file then holds the player's choice, and a
    // BBHOST_* switch still wins over both (config_set_enhancements).
    g_set[kChangeAppearance].index = config().change_appearance ? 0 : 1;
    g_set[kRebirth].index = config().rebirth ? 0 : 1;
    g_set[kFivePlayers].index = config().five_players ? 0 : 1;
    g_set[kSkillIssue].index = config().skill_issue ? 0 : 1;
    // video.fps_cap is the frame rate's default: one of the setting's rates or
    // 0 (Display-Limited); another number takes the nearest rate below it.
    {
        const int cap = config().fps_cap;
        int mode = cap <= 0 ? 0 : 30;
        for (const char* v : g_set[kFrameCap].values) {
            const int r = fps_from_label(v);
            if (cap > 0 && r > mode && r <= cap) mode = r;
        }
        if (mode != cap) host_log("options: video.fps_cap = %d is not a frame rate; %d", cap, mode);
        g_set[kFrameCap].index = std::max(0, fps_index(mode));
    }
    // video.frame_clock is the Native frame clock's default (off).
    g_set[kFrameClock].index = config().frame_clock ? 1 : 0;
    // video.effect_steps is Effect timing's when it is one of its choices ("60"
    // reads as 60 a second, auto's 60 above 30); another value (a number of
    // steps, frame) leaves the row at the default and still runs until a
    // choice is made (config_effect_timing).
    {
        const std::string v = config_value("video.effect_steps");
        if (const int i = effect_timing_index(v == "60" ? "auto" : v); i >= 0) g_set[kEffectTiming].index = i;
    }
    // A package's own (video.resolution, video.window_mode: the Steam Deck
    // kit's 1280x800, fullscreen; video.model_detail), by the entry's name.
    const auto seed = [](int id, const std::string& value) {
        for (std::size_t k = 0; !value.empty() && k < g_set[id].values.size(); ++k)
            if (value == g_set[id].values[k]) g_set[id].index = static_cast<int>(k);
    };
    seed(kResolution, config().resolution);
    seed(kWindowMode, config().window_mode);
    seed(kModelDetail, config().model_detail);
    seed(kShaders, config().shaders);
    if (config().setup_text_size > 0) seed(kMenuTextSize, std::to_string(config().setup_text_size) + "%");
    for (const int id : {static_cast<int>(kSkipLogos), static_cast<int>(kFrameCap), static_cast<int>(kFrameClock),
                         static_cast<int>(kEffectTiming), static_cast<int>(kResolution), static_cast<int>(kWindowMode),
                         static_cast<int>(kModelDetail), static_cast<int>(kShaders), static_cast<int>(kChangeAppearance),
                         static_cast<int>(kRebirth), static_cast<int>(kFivePlayers), static_cast<int>(kSkillIssue)})
        g_default_index[static_cast<std::size_t>(id)] = g_set[id].index;
    const std::string path = options_path();
    FILE* f = std::fopen(path.c_str(), "r");
    if (f) {
        char line[512];
        bool keys = false;  // in [keys]: bindings, not settings
        bool pad = false;   // in [pad]: the controller's buttons
        bool account = false;  // in [account]: the name and token
        std::string acct_name, acct_token;
        int file_version = 1;
        bool old_hunters_off = false;  // v0.2.9's PC enhancement, turned off
        std::string frame_cap_val;
        std::string window_mode_val;  // its label as saved (Fullscreen was Borderless before version 7)
        while (std::fgets(line, sizeof(line), f)) {
            if (line[0] == '[') {
                keys = std::strncmp(line, "[keys]", 6) == 0;
                pad = std::strncmp(line, "[pad]", 5) == 0;
                account = std::strncmp(line, "[account]", 9) == 0;
                continue;
            }
            char* eq = std::strchr(line, '=');
            if (!eq || line[0] == '#') {
                continue;
            }
            *eq = 0;
            std::string key(line), val(eq + 1);
            auto trim = [](std::string& v) {
                while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' ' ||
                                      v.back() == '\t' || v.back() == '"')) {
                    v.pop_back();
                }
                std::size_t i = 0;
                while (i < v.size() && (v[i] == ' ' || v[i] == '\t' || v[i] == '"')) {
                    ++i;
                }
                v.erase(0, i);
            };
            trim(key);
            trim(val);
            if (keys) {
                host_bindings_parse(key.c_str(), val.c_str());
                continue;
            }
            if (pad) {
                host_pad_parse(key.c_str(), val.c_str());
                continue;
            }
            if (account) {
                if (key == "name") acct_name = val;
                if (key == "token") acct_token = val;
                continue;
            }
            if (key == "version") {
                file_version = std::atoi(val.c_str());
                continue;
            }
            if (key == "old_hunters") {
                old_hunters_off = val == "Off";
                continue;
            }
            // Looked at again once the file's version is known (below).
            if (key == "frame_cap") frame_cap_val = val;
            if (key == "window_mode") window_mode_val = val;
            // Display-Limited was called Uncapped until 2026-10-28.
            if (key == "frame_cap" && val == "Uncapped") val = "Display-Limited";
            // Effect timing's names before it had three choices; a saved Smooth,
            // no longer one, reads as not chosen (the default, 60 fps).
            if (key == "effect_timing") {
                if (val == "Original (30)") val = kHostEffectTimingLabels[0];
                else if (val == "60 a second") val = kHostEffectTimingLabels[1];
                else if (val == "Your frame rate") val = kHostEffectTimingLabels[2];
            }
            // The four named steps mouse_sens used to have, on the new scale.
            if (key == "mouse_sens") {
                const char* old[] = {"Low", "Medium", "High", "Very high"};
                const int to[] = {2, 5, 7, 9};
                for (int k = 0; k < 4; ++k) {
                    if (val == old[k]) val = std::to_string(to[k]);
                }
            }
            for (int id = 0; id < kSettingCount; ++id) {
                if (key != g_set[id].key) {
                    continue;
                }
                for (std::size_t v = 0; v < g_set[id].values.size(); ++v) {
                    if (val == g_set[id].values[v]) {
                        g_set[id].index = static_cast<int>(v);
                        g_saved[static_cast<std::size_t>(id)] = true;
                    }
                }
            }
        }
        std::fclose(f);
        // Before version 3 the frame_cap was a cap on presentation, and its 90,
        // 120, 144 and Off ran the game at 60: the frame rate that ran is 60
        // (before any save below writes version 3).
        // Before version 4 there were no camera zoom actions: the middle
        // button they take was Transform's (host/bindings.h).
        // Before version 6 the 0-10 sliders were eleven steps: a saved value is
        // the same place on today's 0-100 (Field of view's percents read as
        // they are).
        if (file_version < 6) {
            for (const int id : {kBloom, kSaturation, kFog, kAoStrength, kCameraDistance, kCameraHeight, kMouseSens}) {
                if (g_saved[static_cast<std::size_t>(id)]) g_set[id].index = std::min(g_set[id].index, 10) * 10;
            }
        }
        // Before version 7 Window mode's Fullscreen was borderless: it is
        // Borderless now, and Fullscreen the exclusive mode.
        if (file_version < 7 && window_mode_val == "Fullscreen") {
            g_set[kWindowMode].index = 1;
            g_dirty = true;
        }
        if (file_version < 4) {
            host_bindings_migrate_zoom();
            g_dirty = true;
        }
        // Before version 5 there were no menu pages and options (host/bindings.h).
        if (file_version < 5) {
            host_bindings_migrate_menu_keys();
            g_dirty = true;
        }
        if (file_version < 3 &&
            (frame_cap_val == "90" || frame_cap_val == "120" || frame_cap_val == "144" || frame_cap_val == "Off")) {
            g_set[kFrameCap].index = std::max(0, fps_index(60));
            g_dirty = true;
            host_log("options: frame_cap \"%s\" (a cap that ran the game at 60) reads as the frame rate 60",
                     frame_cap_val.c_str());
        }
        if (!acct_name.empty() && !acct_token.empty()) {
            net::account_set(acct_name, acct_token);
            g_field[0] = acct_name;
            host_log("options: account %s from %s", acct_name.c_str(), path.c_str());
        }
        // After the account: save() writes everything it holds, and saving
        // before the account was set dropped the [account] section - a
        // dev-server tester was signed out (2026-10-02).
        if (file_version < 2 && g_set[kMouseMenu].index == 0) {
            g_set[kMouseMenu].index = 1;
            g_dirty = true;
            host_log("options: Mouse in menus read as On - this file predates it defaulting to On");
            save();  // as version 2, so a later Off is the player's and stays
        }
        // The Old Hunters was a PC enhancement in v0.2.9 and is the patch
        // old-hunters since. A player who turned it off keeps it off: the
        // per-user bbhost.toml's [patches] says so from now on, and the next
        // save drops the old key (unless [patches] already says something).
        if (old_hunters_off) {
            bool carried = !config_value("patches.old-hunters").empty();
            if (!carried) {
                const std::string user = config_user_file();
                carried = config_set_values(user, {{"patches", "old-hunters", "false"}});
                if (carried) {
                    config_value_set("patches.old-hunters", "false");
                    host_log("options: The Old Hunters was off here; it is the patch old-hunters now, off under [patches] in %s",
                             user.c_str());
                }
            }
            if (carried) {
                g_dirty = true;
                save();
            }
        }
        host_log("options: read %s", path.c_str());
    }
    // BBHOST_ACCOUNT=name:token, for a scripted run that must not touch the file.
    if (const char* e = std::getenv("BBHOST_ACCOUNT"); e && e[0]) {
        const std::string v = e;
        if (const std::size_t c = v.find(':'); c != std::string::npos) {
            net::account_set(v.substr(0, c), v.substr(c + 1));
            g_field[0] = v.substr(0, c);
        }
    }
    // The environment switches the track was brought up with still win, so a
    // run started with one behaves the way it says on the command line.
    if (const char* e = std::getenv("BBHOST_MOUSE_MENU"); e && e[0]) {
        g_set[kMouseMenu].index = e[0] == '0' ? 0 : 1;
    }
    // The post-processing switches, for A/B runs: BBHOST_SSAO=0 and friends.
    for (const auto& [env, id] : {std::pair<const char*, int>{"BBHOST_SSAO", kSsao},
                                  {"BBHOST_MOTION_BLUR", kMotionBlur},
                                  {"BBHOST_ANTI_ALIAS", kAntiAlias},
                                  {"BBHOST_DEPTH_OF_FIELD", kDepthOfField},
                                  {"BBHOST_CHROMATIC_ABERRATION", kChromaticAberration}}) {
        if (const char* e = std::getenv(env); e && e[0]) {
            g_set[id].index = e[0] == '0' ? 1 : 0;  // these are {"On","Off"}
        }
    }
    if (const char* e = std::getenv("BBHOST_MOUSE_CAMERA"); e && e[0]) {
        g_set[kMouseCamera].index = e[0] == '0' ? 0 : 1;
    }
    if (const char* e = std::getenv("BBHOST_MOUSE_SENS"); e && e[0]) {
        const int v = std::atoi(e);
        g_set[kMouseSens].index = (v < 0 ? 0 : v > 10 ? 10 : v) * 10;  // BBHOST_MOUSE_SENS stays 0..10
    }
    if (const char* e = std::getenv("BBHOST_MOUSE_CURSOR"); e && e[0]) {
        g_set[kDrawCursor].index = e[0] == '0' ? 1 : 0;
    }
    host_options_apply();
    rebuild_settings();
}

bool host_options_open() { return g_open.load(std::memory_order_relaxed); }

void host_options_set_open(bool open) {
    if (open == g_open.load(std::memory_order_relaxed)) {
        return;
    }
    g_open.store(open, std::memory_order_relaxed);
    if (open) {
        // Nothing has consumed the pointer while the menu was closed, so a
        // click from minutes ago is still latched. Drop it: the game's menus
        // behind must not take it when this one closes.
        host_mouse_state();
    } else {
        save();
    }
    host_log("options: %s", open ? "open" : "closed");
}

void host_options_close() {
    if (!g_open.exchange(false)) return;
    host_options_post([] {
        save();
        host_log("options: closed");
    });
}

void host_options_post(std::function<void()> fn) {
    std::lock_guard<std::mutex> lk(g_post_mu);
    g_posted.push_back(std::move(fn));
}

void host_options_frame(float, float) {
    {
        std::vector<std::function<void()>> todo;
        {
            std::lock_guard<std::mutex> lk(g_post_mu);
            todo.swap(g_posted);
        }
        for (const std::function<void()>& fn : todo) fn();
    }
    if (g_acct_save.exchange(false)) {
        g_dirty = true;
        save();
    }
    // BBHOST_RES_PROMPT_TEST=<flip>:<entry>: the Resolution setting changed
    // to that entry at that flip, the way the menus change it (tests).
    static long test_flip = [] {
        const char* e = std::getenv("BBHOST_RES_PROMPT_TEST");
        return e && *e ? std::atol(e) : -1L;
    }();
    if (test_flip >= 0 && static_cast<long>(hle_video_flip_count()) >= test_flip) {
        test_flip = -1;
        const char* e = std::strchr(std::getenv("BBHOST_RES_PROMPT_TEST"), ':');
        if (e) host_opt_set_resolution_index(std::atoi(e + 1));
    }
    // The game's keep-or-revert dialog answered (engine/live_resolution.h).
    if (const int a = live_resolution_take_answer()) {
        if (a == 1) res_keep(); else res_revert();
    }
}

#if defined(BBHOST_HAVE_SDL3)
namespace {

namespace ui = launcher_ui;

// The menu's pages: the table's sections (g_rows' headers), in its order.
struct OptionsPage {
    const char* header;  // the section's header in g_rows
    const char* name;
    const char* blurb;
    int first = 0, end = 0;  // its rows in g_rows, after the header
};

const std::vector<OptionsPage>& pages() {
    static const std::vector<OptionsPage> list = [] {
        static const struct {
            const char* header;
            const char* name;
            const char* blurb;
        } kNames[] = {
            {"DISPLAY", "Display", "The window, the frame rate, and what is shown over the game."},
            {"INPUT", "Input", "The mouse, the pointer and the button prompts."},
            {"GRAPHICS", "Graphics", "The game's effects and detail. Most change at once."},
            {"PC ENHANCEMENTS", "PC Enhancements", "Additions to the game itself, read as it starts."},
            {"STARTUP", "Startup", "What happens as the game starts."},
            {"ACCOUNT", "Account", "The private server's account. No passwords: this PC keeps a token."},
            {"UPDATES", "Updates", "Newer releases of bbhost."},
        };
        std::vector<OptionsPage> out;
        for (int r = 0; r < kRowCount; ++r) {
            if (g_rows[r].kind != Row::Header) continue;
            OptionsPage pg{g_rows[r].text, g_rows[r].text, "", r + 1, kRowCount};
            for (const auto& k : kNames) {
                if (std::strcmp(k.header, g_rows[r].text) == 0) {
                    pg.name = k.name;
                    pg.blurb = k.blurb;
                }
            }
            if (!out.empty()) out.back().end = r;
            out.push_back(pg);
        }
        return out;
    }();
    return list;
}

// Drawing for the setup window (host_options_draw_setup_page), not the game.
bool g_setup_page = false;

// A setting as the menu shows it: what it set and the pump has not applied
// yet, else the setting. The presenter's own (g_pending is never the pump's).
int shown_index(int id) {
    const int p = g_pending[static_cast<std::size_t>(id)];
    if (p >= 0 && p != g_set[id].index) return p;
    g_pending[static_cast<std::size_t>(id)] = -1;
    return g_set[id].index;
}

void change(int id, int index) {
    g_pending[static_cast<std::size_t>(id)] = index;
    host_options_post([id, index] { set_index(id, index, false); });
}

// One setting's row: a switch for On and Off, else its label at the left and
// at the right a choice of a few side by side, a list, or a slider through a
// scale's steps; the note in the bubble. A setting read only as the game
// starts says so under its label (the resolution only when it cannot change
// now).
// `label` and `note`, when given, stand for the setting's own: one row can
// show another face (the upscaler's preset row, preset_row).
void setting_row(int id, const char* label = nullptr, const char* note = nullptr) {
    const Setting& s = g_set[id];
    if (!label) label = s.label;
    if (!note) note = s.note;
    const float sc = ui::scale();
    const int n = static_cast<int>(s.values.size());
    const int cur = std::clamp(shown_index(id), 0, n - 1);
    const bool restart = s.restart && !g_setup_page && !(id == kResolution && res_live(cur));
    ImGui::PushID(id);
    int on_i = -1, off_i = -1;
    for (int v = 0; v < n; ++v) {
        if (std::strcmp(s.values[static_cast<std::size_t>(v)], "On") == 0) on_i = v;
        if (std::strcmp(s.values[static_cast<std::size_t>(v)], "Off") == 0) off_i = v;
    }
    if (n == 2 && on_i >= 0 && off_i >= 0) {
        bool v = cur == on_i;
        if (ui::toggle_row(label, restart ? "Applies at the next start" : nullptr, &v, note)) change(id, v ? on_i : off_i);
        ImGui::PopID();
        return;
    }
    const float w = ImGui::GetContentRegionAvail().x;
    const float cw = std::min(std::round(330 * sc), w * 0.55f);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::BeginGroup();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    if (restart) {
        ImGui::SameLine();
        ui::pill("next start", ui::kWarn, ui::alpha(ui::kWarn, 38));
    }
    ImGui::EndGroup();
    if (note) ui::tooltip_for_item(label, note);
    ImGui::SameLine();
    ImGui::SetCursorScreenPos(ImVec2(p.x + w - cw, p.y));
    int want = cur;
    bool slider = false;
    if (id == kEffectTiming) {
        // What will run (host_effect_timing_row): one of the three, or a
        // fourth entry for what bbhost.toml or BBHOST_SFX_STEP asks that none
        // of them is. A pick is one of the three, the player's own even when
        // the setting held it already (set_index).
        const HostEffectTimingRow et = host_effect_timing_row();
        const char* labels[4] = {s.values[0], s.values[1], s.values[2], et.custom.c_str()};
        const char* tips[4] = {kHostEffectTimingTips[0], kHostEffectTimingTips[1], kHostEffectTimingTips[2], et.tip.c_str()};
        const int shown = g_pending[static_cast<std::size_t>(id)] >= 0 ? cur : et.index;
        int pick = shown;
        ImGui::SetNextItemWidth(cw);
        ui::combo("##v", &pick, labels, et.index == kEffectTimingCount ? 4 : 3, tips);
        if (pick != shown && pick < kEffectTimingCount) change(id, pick);
    } else if (id == kFrameCap || id == kResolution) {
        ImGui::SetNextItemWidth(cw);
        ui::combo("##v", &want, s.values.data(), n);
    } else if (n <= 4) {
        ui::segmented("##v", &want, s.values.data(), n, cw, id == kShaders ? kHostShaderModeTips : nullptr,
                      id == kShaders ? "Applies at the next start." : nullptr);
    } else {
        // A scale: the slider shows the step's own name (+10%, x1.4, 5).
        std::string shown;
        for (const char* c = s.values[static_cast<std::size_t>(cur)]; *c; ++c) {
            shown += *c;
            if (*c == '%') shown += '%';
        }
        // Longer than the other controls, with its reset beside it.
        slider = true;
        const float sw = std::min(std::round(520 * sc), w * 0.62f);
        ImGui::SetCursorScreenPos(ImVec2(p.x + w - sw, p.y));
        ImGui::SetNextItemWidth(sw - ui::reset_width());
        ImGui::SliderInt("##v", &want, 0, n - 1, shown.c_str(), ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
        if (note) ui::tooltip_for_item(label, note);
        const int def = g_default_index[static_cast<std::size_t>(id)];
        if (ui::reset_button("##reset", cur == def)) want = def;
    }
    if (note && id != kShaders && !slider) ui::tooltip_for_item(label, note);  // a slider's is on the slider, not its reset
    if (want != cur && want >= 0 && want < n) change(id, want);
    ImGui::PopID();
}

// A thin line between two rows of a card.
void row_rule() {
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float y = at.y - ImGui::GetStyle().ItemSpacing.y * 0.5f + 1;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(at.x, y), ImVec2(at.x + ImGui::GetContentRegionAvail().x, y), ui::alpha(ui::kViolet, 40));
}

// The ACCOUNT actions' bubbles (account_action's `which`).
const char* const kAccountNotes[] = {
    "Shows a code and opens the website: sign in there with Discord and approve it.",
    "Makes an account held by this PC under the name beside it; write down its recovery code.",
    "The name and recovery code give this PC the account back (other PCs are signed out).",
    "A one-time code to sign in on the website's account page with this PC's account.",
    "Forgets the token; the game is signed out if the server requires an account.",
};

// An ACCOUNT or UPDATES button, run on the window's thread with the fields as
// typed.
void action_button(const char* label, int which, bool primary, const char* name = "", const char* code = "") {
    const bool idle = action_idle(which);
    bool pressed = false;
    if (primary) {
        pressed = ui::primary_button(label, ImVec2(0, 0), idle);
    } else {
        ImGui::BeginDisabled(!idle);
        pressed = ui::ghost_button(label);
        ImGui::EndDisabled();
    }
    const char* note = which < 5 ? kAccountNotes[which] : which == 10 ? "Asks GitHub for the newest release."
                                                                      : "Opens the newer release's page in your browser.";
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) ui::tooltip(nullptr, note);
    if (pressed && idle) {
        host_options_post([which, n = std::string(name), c = std::string(code)] {
            if (which == 1 || which == 2) {
                g_field[0] = n;
                g_field[1] = c;
            }
            account_action(which, n, c);
        });
    }
}

char g_name_buf[64] = {};  // the account fields as typed (the presenter's)
char g_code_buf[64] = {};
bool g_bufs_seeded = false;

void account_page() {
    const float sc = ui::scale(), w = ImGui::GetContentRegionAvail().x;
    if (!g_bufs_seeded) {
        g_bufs_seeded = true;
        std::snprintf(g_name_buf, sizeof(g_name_buf), "%s", g_field[0].c_str());
    }
    ui::card_begin("##account", ImVec2(w, 0), "SIGNED IN");
    {
        const bool in = net::account_logged_in();
        ui::avatar(in ? net::account_name() : std::string(), std::round(46 * sc));
        ImGui::SameLine(0, 14 * sc);
        ImGui::BeginGroup();
        ImGui::PushFont(ui::font_strong(), 0.0f);
        ImGui::TextUnformatted(account_status().c_str());
        ImGui::PopFont();
        ImGui::PushStyleColor(ImGuiCol_Text, ui::kTextDim);
        ImGui::TextUnformatted(hle_np_online_status().c_str());
        ImGui::PopStyleColor();
        ImGui::EndGroup();
    }
    if (const std::string d = account_detail(); !d.empty()) {
        ImGui::Spacing();
        ui::banner(ui::Icon::Info, ui::kIce, d);
    }
    ImGui::Spacing();
    action_button("Link with Discord", 0, true);
    ImGui::SameLine();
    action_button("Website sign-in code", 3, false);
    ImGui::SameLine();
    action_button("Sign out", 4, false);
    ui::card_end();
    ImGui::Dummy(ImVec2(0, 6 * sc));
    ui::card_begin("##own", ImVec2(w, 0), "AN ACCOUNT THIS PC HOLDS");
    const float bw = std::round(250 * sc);
    const float fw = ImGui::GetContentRegionAvail().x - bw - ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetNextItemWidth(fw);
    ImGui::InputTextWithHint("##name", "Account name", g_name_buf, sizeof(g_name_buf));
    ImGui::SameLine();
    action_button("Create account on this PC", 1, false, g_name_buf, g_code_buf);
    ImGui::SetNextItemWidth(fw);
    ImGui::InputTextWithHint("##code", "Recovery code", g_code_buf, sizeof(g_code_buf));
    ImGui::SameLine();
    action_button("Recover account", 2, false, g_name_buf, g_code_buf);
    ui::card_end();
}

void updates_page() {
    const float w = ImGui::GetContentRegionAvail().x;
    ui::card_begin("##updates", ImVec2(w, 0), "THIS BBHOST");
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(updater::status().c_str());
    if (const std::string d = updater::detail(); !d.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ui::kTextDim);
        ImGui::TextUnformatted(d.c_str());
        ImGui::PopStyleColor();
    }
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    action_button("Check for updates", 10, true);
    ImGui::SameLine();
    action_button("Open the release page", 11, false);
    ui::card_end();
}

}  // namespace

int host_options_page_count() { return static_cast<int>(pages().size()); }
const char* host_options_page_name(int page) {
    return page >= 0 && page < host_options_page_count() ? pages()[static_cast<std::size_t>(page)].name : "";
}
const char* host_options_page_blurb(int page) {
    return page >= 0 && page < host_options_page_count() ? pages()[static_cast<std::size_t>(page)].blurb : "";
}

// The upscaler's preset row, which shows what the chosen upscaler takes:
// for FSR 4 and FSR 3 the render size against the Resolution (Native AA,
// Quality, Balanced, Performance - Upscaling quality, upscale_quality), for
// DLSS that render size and then DLSS's model (K, J, L, M); with Upscaling
// off, DLSS's model, greyed. The render size is drawn at as the game starts
// (engine/render_scale.cpp), so it applies at the next start.
void preset_row() {
    const int u = shown_index(kUpscaler);  // the menu's Off, FSR 4, FSR 3, DLSS
    const auto quality_note = [](const char* who) {
        static std::string notes[2];
        std::string& n = notes[who[0] == 'D' ? 1 : 0];
        if (n.empty())
            n = std::string("Requires a restart. How much smaller than the Resolution the 3D is drawn before ") + who +
                " brings it back: Native AA draws it full size (anti-aliasing only), Quality at two thirds, Balanced at about "
                "59%, Performance at half each way - faster, softer. The menus, the HUD and the post effects stay at the "
                "Resolution. Below Native AA, a different Resolution also applies at the next start.";
        return n.c_str();
    };
    if (u == 1 || u == 2) {
        setting_row(kUpscaleQuality, "FSR preset", quality_note("FSR"));
        return;
    }
    if (u == 3) {
        setting_row(kUpscaleQuality, "DLSS quality", quality_note("DLSS"));
        row_rule();
    }
    ImGui::BeginDisabled(u == 0);
    setting_row(kDlssPreset);
    ImGui::EndDisabled();
    if (u == 0) ui::tooltip_for_item(g_set[kDlssPreset].label, "Choose an upscaler under Upscaling first.");
}

void host_options_draw_page(int page) {
    if (page < 0 || page >= host_options_page_count()) return;
    const OptionsPage& pg = pages()[static_cast<std::size_t>(page)];
    if (std::strcmp(pg.header, "ACCOUNT") == 0) return account_page();
    if (std::strcmp(pg.header, "UPDATES") == 0) return updates_page();
    ui::card_begin("##rows", ImVec2(ImGui::GetContentRegionAvail().x, 0));
    bool first = true;
    for (int r = pg.first; r < pg.end; ++r) {
        if (g_rows[r].kind != Row::Option) continue;
        if (g_setup_page && g_rows[r].setting == kShaders) continue;
        if (!first) row_rule();
        first = false;
        if (g_rows[r].setting == kDlssPreset) preset_row();
        else setting_row(g_rows[r].setting);
        ImGui::Dummy(ImVec2(0, 1 * ui::scale()));
    }
    ui::card_end();
}

void host_options_draw_setup_page(int page) {
    // Resolution is the Graphics tab's here too (the upscaler's output size
    // and the render size follow it); only the shader form stays the Shaders
    // page's.
    g_setup_page = true;
    host_options_draw_page(page);
    g_setup_page = false;
}

#else  // a build without a window: no menu

void host_options_draw_setup_page(int) {}
int host_options_page_count() { return 0; }
const char* host_options_page_name(int) { return ""; }
const char* host_options_page_blurb(int) { return ""; }
void host_options_draw_page(int) {}

#endif

const char* const kHostShaderModeTips[2] = {
    "Recommended. bbhost also turns each of the game's shaders - the small programs your graphics card draws every "
    "surface and effect with - into an optimized version, using Microsoft's shader compiler. It does this in the "
    "background while you play, and the game uses them from the next start. The picture is the same either way.",
    "Only bbhost's direct translation of the shaders: nothing is compiled in the background while you play. Try it if "
    "HLSL gives you trouble. The picture is the same either way.",
};

const char* const kHostEffectTimingLabels[3] = {"30 fps", "60 fps", "Frame rate-dependent"};
const char* const kHostEffectTimingNote =
    "Requires a restart. How fire, blood, magic and the flicker of lamps and candles keep time above 30 fps.";
const char* const kHostEffectTimingTips[3] = {
    "Effects step 30 times a second, as the game at 30 fps, whatever your frame rate: above 30 fps their motion is less "
    "smooth.",
    "The default. Effects step 60 times a second at any frame rate above 30. The flicker and the other counted parts "
    "speed up with them, to twice the game's own pace.",
    "Effects step once a frame. The flicker and the other counted parts speed up with them: twice the game's own pace "
    "at 60 fps, four times at 120.",
};

HostSettings host_settings() { return settings_store().now(); }
const HostSettings& host_settings_ref() { return settings_store().now(); }
void host_settings_listen(HostSettingsListener fn) {
    if (!settings_store().listen(fn)) host_log("options: no room for another settings listener; one consumer will not follow changes");
}

namespace {
// Which way round a two-value setting reads comes from the values themselves,
// not a list of ids kept in step by hand: a new {"On","Off"} setting was
// silently inverted by that list, which is the sort of bug that only shows up
// as "the option does the opposite".
bool reads_on(const Setting& s, int index) {
    const std::size_t i = static_cast<std::size_t>(index);
    if (i < s.values.size()) {
        if (std::strcmp(s.values[i], "On") == 0) return true;
        if (std::strcmp(s.values[i], "Off") == 0) return false;
    }
    return index != 0;  // anything else: the first value is the "no"
}

int setting_id(const char* key) {
    for (int id = 0; key && id < kSettingCount; ++id) {
        if (std::strcmp(key, g_set[id].key) == 0) return id;
    }
    return -1;
}

}  // namespace

std::uint64_t host_opt_serial() { return settings_store().now().serial; }

bool host_opt_default_get(const char* key) {
    const int id = setting_id(key);
    return id >= 0 && reads_on(g_set[id], g_default_index[static_cast<std::size_t>(id)]);
}
int host_opt_default_index(const char* key) {
    const int id = setting_id(key);
    return id < 0 ? -1 : g_default_index[static_cast<std::size_t>(id)];
}
int host_opt_default_frame_cap() {
    return fps_from_label(g_set[kFrameCap].values[static_cast<std::size_t>(g_default_index[kFrameCap])]);
}
int host_opt_default_resolution_index() { return g_default_index[kResolution]; }

bool host_opt_get(const char* key) {
    const int id = setting_id(key);
    return id >= 0 && reads_on(g_set[id], g_set[id].index);
}

bool host_opt_saved(const char* key) {
    const int id = setting_id(key);
    return id >= 0 && g_saved[static_cast<std::size_t>(id)];
}

HostEffectTimingRow host_effect_timing_row() {
    HostEffectTimingRow r;
    const int chosen = std::clamp(g_set[kEffectTiming].index, 0, kEffectTimingCount - 1);
    const EffectTiming t = config_effect_timing(60, kEffectTimingValues[chosen], g_saved[kEffectTiming]);
    const std::string v = t.readable ? t.value : std::string(kEffectTimingDefault);  // one it cannot read runs the default
    r.index = effect_timing_index(v == "60" ? "auto" : v);
    if (r.index >= 0) return r;
    const bool env = std::strcmp(t.from, "BBHOST_SFX_STEP") == 0;
    r.index = kEffectTimingCount;
    r.custom = "Custom: " + v + (env ? " (BBHOST_SFX_STEP)" : " (bbhost.toml)");
    r.tip = (v == "frame"    ? std::string("Effects move by each frame's own time")
             : v == "smooth" ? std::string("Effects move at your frame rate while what the game counts keeps 30 a second (the old Smooth)")
                             : "Effects step " + v + " times a second") +
            (env ? ", as BBHOST_SFX_STEP asks: it wins over this setting while it is set."
                 : ", as effect_steps in bbhost.toml asks. Choosing one of the others replaces it.");
    return r;
}

void host_opt_choose_effect_timing(int index) {
    if (index < 0 || index >= kEffectTimingCount) return;
    set_index(kEffectTiming, index, false);  // the player's own (g_saved) and written, even when it held it already
    save();
}

bool host_opt_resolution(int* w, int* h) {
    const HostSettings hs = host_settings();
    if (hs.res_width <= 0 || hs.res_height <= 0) return false;
    *w = hs.res_width;
    *h = hs.res_height;
    return true;
}

int host_opt_resolution_index() { return g_set[kResolution].index; }

void host_opt_set_resolution_index(int i) {
    if (i < 0 || static_cast<std::size_t>(i) >= g_set[kResolution].values.size()) {
        return;
    }
    if (i == g_set[kResolution].index) return;
    set_index(kResolution, i, false);
    save();
}

int host_opt_frame_cap() {
    return fps_from_label(g_set[kFrameCap].values[static_cast<std::size_t>(g_set[kFrameCap].index)]);
}

void host_opt_set_frame_cap(int fps) {
    const int i = fps_index(fps);
    if (i < 0) {
        return;
    }
    if (i == g_set[kFrameCap].index) return;
    set_index(kFrameCap, i, false);
    save();
}

void host_opt_set(const char* key, bool on) {
    if (!key) {
        return;
    }
    for (int id = 0; id < kSettingCount; ++id) {
        if (std::strcmp(key, g_set[id].key) != 0) {
            continue;
        }
        // Which index is "on" comes from the values, as in host_opt_get: a
        // hand-kept list of the {"On","Off"} settings missed Mouse camera and
        // the four graphics switches, so turning one of those on from the
        // game's menu saved it off. Two values that are not On/Off
        // (Windowed/Fullscreen) read the second as on.
        const Setting& s = g_set[id];
        // A setting with no On (Window mode) is on at any value past its
        // first: on keeps Fullscreen as it is, and turns Windowed Borderless.
        const bool has_on = std::any_of(s.values.begin(), s.values.end(), [](const char* v) { return std::strcmp(v, "On") == 0; });
        if (on && !has_on && s.index != 0) return;
        int want = on ? 1 : 0;
        for (std::size_t v = 0; v < s.values.size(); ++v) {
            if (std::strcmp(s.values[v], on ? "On" : "Off") == 0) {
                want = static_cast<int>(v);
            }
        }
        if (want == s.index) return;  // the setup window sets every one; only a change is written
        set_index(id, want, false);
        // The F10 screen saves when it closes; a row in the game's menu has no
        // such moment, so a change made there is written now or not at all.
        save();
        return;
    }
}

float host_opt_menu_text_scale() {
    const Setting& s = g_set[kMenuTextSize];
    const int i = std::clamp(s.index, 0, static_cast<int>(s.values.size()) - 1);
    return static_cast<float>(std::atoi(s.values[static_cast<std::size_t>(i)])) / 100.0f;
}

void host_opt_menu_colour(float* hue_degrees, float* vibrance) {
    const auto value = [](SettingId id) {
        const Setting& s = g_set[id];
        const int i = std::clamp(s.index, 0, static_cast<int>(s.values.size()) - 1);
        return static_cast<float>(std::atoi(s.values[static_cast<std::size_t>(i)]));
    };
    *hue_degrees = value(kMenuHue);
    *vibrance = value(kMenuVibrance) / 100.0f;
}

int host_opt_index(const char* key) {
    for (int id = 0; key && id < kSettingCount; ++id) {
        if (std::strcmp(key, g_set[id].key) == 0) return g_set[id].index;
    }
    return -1;
}

void host_opt_set_index(const char* key, int index) {
    for (int id = 0; key && id < kSettingCount; ++id) {
        if (std::strcmp(key, g_set[id].key) != 0) continue;
        if (index < 0 || static_cast<std::size_t>(index) >= g_set[id].values.size()) return;
        if (index == g_set[id].index) return;
        set_index(id, index, false);
        save();
        return;
    }
}

void host_options_save_now() {
    g_dirty = true;
    save();
}

bool host_steam_deck() {
    static const bool deck = [] {
        if (const char* e = std::getenv("BBHOST_STEAM_DECK"); e && e[0]) return e[0] == '1';
        // The firmware's vendor and product: Valve's Jupiter (the LCD Deck) or
        // Galileo (the OLED one). Not any Valve machine - another has another
        // GPU, and these two configurations are this one's.
        std::string vendor, product;
#if defined(_WIN32)
        const auto bios = [](const char* name) {
            char buf[128] = {};
            DWORD size = sizeof(buf);
            if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\BIOS", name, RRF_RT_REG_SZ, nullptr, buf, &size) !=
                ERROR_SUCCESS)
                return std::string();
            return std::string(buf);
        };
        vendor = bios("SystemManufacturer");
        product = bios("SystemProductName");
#else
        const auto line = [](const char* path) {
            std::ifstream f(path);
            std::string s;
            std::getline(f, s);
            return s;
        };
        vendor = line("/sys/devices/virtual/dmi/id/sys_vendor");
        product = line("/sys/devices/virtual/dmi/id/product_name");
#endif
        return vendor == "Valve" && (product == "Jupiter" || product == "Galileo");
    }();
    return deck;
}

namespace {
// The Deck's configurations, by the frame rate: 60 or more (or uncapped) is 60 fps.
int deck_profile_of(int fps) { return fps == 0 || fps >= 60 ? 1 : 0; }
}  // namespace

int host_deck_profile() { return deck_profile_of(host_opt_frame_cap()); }
int host_deck_default_profile() { return deck_profile_of(host_opt_default_frame_cap()); }

void host_set_deck_profile(int profile) {
    const bool sixty = profile == 1;
    if (const int i = fps_index(sixty ? 60 : 30); i >= 0 && i != g_set[kFrameCap].index) set_index(kFrameCap, i, false);
    if (const int m = sixty ? 2 : 1; m != g_set[kModelDetail].index) set_index(kModelDetail, m, false);  // Low, Normal
    const char* const size = sixty ? "960x600" : "1280x800";
    for (std::size_t v = 0; v < g_set[kResolution].values.size(); ++v) {
        if (std::strcmp(g_set[kResolution].values[v], size) != 0 || static_cast<int>(v) == g_set[kResolution].index) continue;
        // Both sizes are known to fit the Deck: no keep-or-revert question,
        // and one still waiting is settled by this.
        g_res_prev = -1;
        g_res_reverting = true;
        set_index(kResolution, static_cast<int>(v), false);
        g_res_reverting = false;
    }
    host_log("options: Steam Deck at %s", sixty ? "60 fps (960x600 upscaled, Model detail Low)" : "30 fps (1280x800)");
    save();
}

// The ACCOUNT actions for the setup window (host/options.h).
void host_account_action(int which, const std::string& name, const std::string& code) {
    if (which >= 0 && which <= 4) account_action(which, name, code);
}
bool host_account_busy() { return g_acct_busy.load(); }
std::string host_account_signed_in() {
    return net::account_logged_in() ? "Signed in as " + net::account_name() : "Not signed in";
}
std::string host_account_outcome() {
    std::lock_guard<std::mutex> lk(g_acct_mu);
    return g_acct_status;
}
std::string host_account_detail() { return account_detail(); }
bool host_account_take_save() { return g_acct_save.exchange(false); }
