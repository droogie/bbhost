#include "host/window.h"
#include "core/host_clock.h"
#include "core/portable.h"

#include "core/config.h"

#include "engine/graphics_patch.h"
#include "engine/menu_pointer.h"
#include "engine/mouse_camera_step.h"
#include "engine/option_menu.h"
#include "hle/modules.h"
#include "host/bindings.h"
#include "host/options.h"
#include "host/settings.h"
#include "host/overlay.h"
#include "host/present_pass.h"
#include "host/ingame_menu.h"
#include "host/audio.h"
#include "host/foreign_hooks.h"
#include "host/gpu.h"
#include "log.h"
#if !defined(_WIN32)
#include <pthread.h>
#include <signal.h>
#include <cerrno>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(BBHOST_HAVE_SDL3)
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#endif

// The swapchain's extent (width << 32 | height) for host_window_pixels; 0 before one.
static std::atomic<std::uint64_t> g_window_pixels{0};

namespace {

std::atomic<bool> g_active{false};
std::mutex g_pad_mu;
PadState g_pad;

// Where the game's picture (dw x dh) goes in a w x h surface: centred, as big
// as fits with its aspect kept - black bars at the sides of a wider display
// (21:9, 32:9) or above and below a taller one. It was stretched over the
// whole window, which on an ultrawide made everything a third wider. The same
// rectangle serves the swapchain (pixels) and the pointer (window units).
struct FitRect {
    float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
};
FitRect fit_picture(float w, float h, float dw, float dh) {
    if (w <= 0.0f || h <= 0.0f || dw <= 0.0f || dh <= 0.0f) return {0.0f, 0.0f, w, h};
    const float s = std::min(w / dw, h / dh);
    const float fw = dw * s, fh = dh * s;
    return {(w - fw) * 0.5f, (h - fh) * 0.5f, fw, fh};
}

// The pointer. Positions arrive in window coordinates and are kept in the
// game's display-buffer pixels, through the picture's rectangle (fit_picture);
// over the bars it is outside the game. The size comes from whatever was last
// presented; before the first present there is nothing to scale by and the
// pointer reads as outside the window.
std::mutex g_mouse_mu;
MouseState g_mouse;
unsigned g_display_w = 0, g_display_h = 0;
std::atomic<bool> g_mouse_visible{true};
std::atomic<bool> g_mouse_relative{false};
std::atomic<bool> g_mouse_relative_applied{false};
// BBHOST_IME_TEST: opens the text box once at startup and again on F11.
std::atomic<bool> g_ime_test_reopen{true};

// The options screen's two display settings (host/options.h). Both can be set
// before the presenter exists, so they are wishes the swapchain reads, not
// calls into it.
std::atomic<bool> g_want_fullscreen{false};
std::atomic<bool> g_want_vsync{true};
// When the presenting thread last finished a present, and how many it has
// finished: with V-Sync on that instant is a display vblank, which is the
// phase the flip clock steers towards.
std::atomic<std::uint64_t> g_present_last_ns{0}, g_present_serial{0};
std::atomic<bool> g_swap_dirty{false};

// Text entry state (host_text_entry_*). SDL text input is toggled from the
// pump (main thread) when `want` changes.
std::mutex g_text_mu;
// Plugin banners (host_message_show): the text and when it stops showing.
std::mutex g_banner_mu;
std::string g_banner_text;                                  // under g_banner_mu
std::chrono::steady_clock::time_point g_banner_until;       // under g_banner_mu
bool g_text_want = false, g_text_on = false, g_text_hidden = false;
int g_text_result = 0;
// The key that ended the text entry (Return or Escape), kept from the pad
// until it is let go: still held when the entry closed, it read as the menu's
// confirm (or Options) and the game opened the name entry again at once.
std::atomic<int> g_text_end_key{-1};
// The same for a controller's button (an SDL_GamepadButton), when one ended it.
std::atomic<int> g_text_end_button{-1};
TextEntry g_entry;  // the text and what the box takes (host/text_entry.h)
std::string g_text_label = "Enter name";  // what the box over the game asks for


#if defined(BBHOST_HAVE_SDL3)

SDL_Window* g_window = nullptr;

// When the pointer last moved or clicked. DS3's CSMouseMan hides the cursor
// after five seconds without either (its +0x08, 5.0f), so a player who has
// gone back to the pad does not have an arrow parked over the menu.
std::chrono::steady_clock::time_point g_mouse_last_motion = std::chrono::steady_clock::now();

// The swapchain's size (pixels), set when it is made. The picture's rectangle
// is fitted in it, and the window shows it scaled to the window's own size -
// which is what a driver does with a swapchain the window has outgrown before
// it is rebuilt - so the pointer goes window -> swapchain -> picture, and lands
// where the picture is drawn even in the frames before a rebuild.
std::atomic<std::uint32_t> g_swap_w{0}, g_swap_h{0};
float g_mouse_win_x = 0.0f, g_mouse_win_y = 0.0f;  // under g_mouse_mu: the last position in window units
int g_mouse_win_w = 0, g_mouse_win_h = 0;          // and the window's size then

void mouse_moved_locked(float win_x, float win_y) {
    int ww = 0, wh = 0;
    SDL_GetWindowSize(g_window, &ww, &wh);
    g_mouse_win_x = win_x;
    g_mouse_win_y = win_y;
    g_mouse_win_w = ww;
    g_mouse_win_h = wh;
    if (ww <= 0 || wh <= 0 || !g_display_w || !g_display_h) {
        g_mouse.in_window = false;
        return;
    }
    const std::uint32_t sw = g_swap_w.load(std::memory_order_relaxed), sh = g_swap_h.load(std::memory_order_relaxed);
    const float fw = sw ? static_cast<float>(sw) : static_cast<float>(ww), fh = sh ? static_cast<float>(sh) : static_cast<float>(wh);
    const float px = win_x * fw / static_cast<float>(ww), py = win_y * fh / static_cast<float>(wh);
    const FitRect r = fit_picture(fw, fh, static_cast<float>(g_display_w), static_cast<float>(g_display_h));
    g_mouse.x = (px - r.x) * static_cast<float>(g_display_w) / r.w;
    g_mouse.y = (py - r.y) * static_cast<float>(g_display_h) / r.h;
    g_mouse.in_window = g_mouse.x >= 0.0f && g_mouse.y >= 0.0f && g_mouse.x < static_cast<float>(g_display_w) &&
                        g_mouse.y < static_cast<float>(g_display_h);
    g_mouse.moved = true;
    g_mouse_last_motion = std::chrono::steady_clock::now();
}

std::uint32_t mouse_bit(std::uint8_t sdl_button) {
    switch (sdl_button) {
        case SDL_BUTTON_LEFT: return 1u;
        case SDL_BUTTON_RIGHT: return 2u;
        case SDL_BUTTON_MIDDLE: return 4u;
        case SDL_BUTTON_X1: return 8u;
        case SDL_BUTTON_X2: return 16u;
        default: return 0u;
    }
}

// Every controller plugged in, at start or later, read together - as DS3
// reads any pad - so the one in the player's hands works whichever it is and
// unplugging one leaves the rest. Opened and read on the pump thread; the
// mutex is for rumble, which the game's pad thread asks for.
std::vector<SDL_Gamepad*> g_pads;
std::mutex g_pads_mu;
bool pads_ignored() {
    static const bool no_pad = [] {
        const char* e = std::getenv("BBHOST_NO_GAMEPAD");
        return e && e[0] == '1';
    }();
    return no_pad;
}
// BBHOST_GAMEPAD_MATCH=vvvv:pppp (USB vendor:product, hex): only that pad is
// opened - a scripted run's virtual pad, not the controller on the desk.
bool pad_matches(SDL_JoystickID id) {
    static const char* want = std::getenv("BBHOST_GAMEPAD_MATCH");
    if (!want || !*want) return true;
    char have[16];
    std::snprintf(have, sizeof(have), "%04x:%04x", SDL_GetGamepadVendorForID(id), SDL_GetGamepadProductForID(id));
    return std::strcmp(have, want) == 0;
}
// The Deck's own controls, opened raw while bbhost was not started by Steam:
// Steam's desktop layout is then live as well and turns the same buttons into
// Return, Escape and Space (A, B, Y), the arrow keys and mouse clicks, so a
// press does two things - menus open and close under the player (the first
// Deck run's "inputs flood"). Started from Steam, the game gets Steam's own
// layout and SDL is handed Steam's virtual pad alone (Steam sets
// SDL_GAMECONTROLLER_IGNORE_DEVICES), so none of this happens.
bool g_deck_outside_steam = false;
std::chrono::steady_clock::time_point g_last_pad_press;
bool steam_running() {
#if defined(__linux__)
    const char* home = std::getenv("HOME");
    if (!home) return false;
    FILE* f = std::fopen((std::string(home) + "/.steam/steam.pid").c_str(), "r");
    if (!f) return false;
    long pid = 0;
    const bool read = std::fscanf(f, "%ld", &pid) == 1;
    std::fclose(f);
    return read && pid > 0 && (kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM);
#else
    return false;
#endif
}
void note_deck_outside_steam(SDL_Gamepad* g) {
    const char* launched = std::getenv("SteamGameId");
    if (SDL_GetGamepadVendor(g) != 0x28de || SDL_GetGamepadProduct(g) != 0x1205 || (launched && *launched) || !steam_running()) return;
    g_deck_outside_steam = true;
    host_log("input: the Deck's controls outside Steam - Steam's desktop layout also sends keys and clicks for them, so each "
             "press does two things; start bbhost from Steam (Add a Non-Steam Game), in Game Mode or Desktop Mode");
}
// A key or click right behind a pad press on such a Deck is the desktop
// layout's twin of it: say so on screen, at most every half minute.
void warn_deck_double_input() {
    if (!g_deck_outside_steam) return;
    static std::chrono::steady_clock::time_point said;
    const auto now = std::chrono::steady_clock::now();
    if (now - g_last_pad_press > std::chrono::milliseconds(200) || (said.time_since_epoch().count() && now - said < std::chrono::seconds(30))) return;
    said = now;
    host_message_show("Start bbhost from Steam: outside it, Deck buttons also send keys", 8.0f);
}

void pad_open(SDL_JoystickID id) {
    if (pads_ignored() || !pad_matches(id)) return;
    std::lock_guard<std::mutex> lock(g_pads_mu);
    for (SDL_Gamepad* g : g_pads) {
        if (SDL_GetGamepadID(g) == id) return;
    }
    if (SDL_Gamepad* g = SDL_OpenGamepad(id)) {
        g_pads.push_back(g);
        const char* path = SDL_GetGamepadPath(g);
        host_log("gamepad: %s connected (id %u, %s; %zu in all)", SDL_GetGamepadName(g), static_cast<unsigned>(id),
                 path ? path : "no path", g_pads.size());
        note_deck_outside_steam(g);
    }
}
void pad_close(SDL_JoystickID id) {
    std::lock_guard<std::mutex> lock(g_pads_mu);
    for (auto it = g_pads.begin(); it != g_pads.end(); ++it) {
        if (SDL_GetGamepadID(*it) == id) {
            host_log("gamepad: %s disconnected (id %u)", SDL_GetGamepadName(*it), static_cast<unsigned>(id));
            SDL_CloseGamepad(*it);
            g_pads.erase(it);
            return;
        }
    }
}

// Which device the player last used, the way DS3 switches its prompts and
// pointer: the latest real input decides - a key, a click, the wheel or the
// mouse moving for the keyboard and mouse; a button, or a stick or trigger
// pushed past its dead zone, for a controller. A pad that is merely plugged in
// changes nothing.
std::atomic<int> g_input_device{0};  // InputDevice
std::atomic<std::uint64_t> g_input_serial{0};
void note_input(InputDevice d) {
    if (g_input_device.exchange(static_cast<int>(d), std::memory_order_relaxed) != static_cast<int>(d)) {
        g_input_serial.fetch_add(1, std::memory_order_relaxed);
        host_log("input: %s", d == InputDevice::Pad ? "controller" : "keyboard and mouse");
    }
}

// ---- Vulkan presenter ------------------------------------------------------
struct Presenter {
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;          // the renderer's: under host_gpu_lock
    VkQueue present_queue = VK_NULL_HANDLE;  // this thread's own, when the device has one
    std::uint32_t family = 0;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;  // for the overlay's dynamic rendering, and FSR's RCAS when `storage`
    bool storage = false;            // the swapchain images take storage writes (host/fsr.cpp writes them directly)
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore acquire_sem = VK_NULL_HANDLE;
    VkSemaphore render_sem = VK_NULL_HANDLE;
    // One "rendered" semaphore per swapchain image: a present holds its
    // semaphore past the submit that signalled it, until that image is
    // acquired again, so one shared semaphore was signalled while a present
    // might still hold it (the validation layer said so every frame).
    std::vector<VkSemaphore> render_sems;
    VkFence fence = VK_NULL_HANDLE;
    // Signalled by the acquire instead of acquire_sem (BBHOST_PRESENT_ACQUIRE_FENCE).
    VkFence acquire_fence = VK_NULL_HANDLE;
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    bool latest_ready = false;  // the device has FIFO_LATEST_READY enabled (gpu.cpp)
    bool gpu_wait = false;      // sleep on the frame's fence before presenting (present_gpu_wait_default)
    bool via_dxgi = false;      // the driver presents through DXGI: seen at the first swapchain, kept for the later ones
    std::mutex mu;
    bool ok = false;
    std::uint64_t frames = 0;
};
Presenter g_vk;

// ---- How a frame reaches the display, and the switches back ----------------
// BBHOST_PRESENT_LEGACY=1 puts every one of these back as it was before them,
// for an A/B run with one variable - showing frames on arrival too
// (BBHOST_PRESENT_ON_ARRIVAL, hle/video.cpp). A switch set on its own wins.
bool present_legacy() {
    static const bool legacy = [] {
        const char* e = std::getenv("BBHOST_PRESENT_LEGACY");
        return e && e[0] == '1';
    }();
    return legacy;
}
// A switch that is on unless set to 0 (or BBHOST_PRESENT_LEGACY=1).
bool present_env_off(const char* name) {
    const char* e = std::getenv(name);
    return e ? e[0] == '0' : present_legacy();
}
// BBHOST_PRESENT_PASS=0: the frame is blitted (vkCmdBlitImage, after a clear
// of the whole image when it has bars) and anything over it - the FPS counter,
// the pointer, F10's screen, the F9 menu - goes in a second rendering that
// loads the image back. By default it is one rendering: the picture drawn by a
// sampling triangle, the overlay and the menu in the same pass, the bars the
// pass's clear (host/present_pass.h). A magnified picture keeps FSR's passes.
const bool g_present_pass = !present_env_off("BBHOST_PRESENT_PASS");
// BBHOST_PRESENT_ACQUIRE_FENCE=0: the acquire signals a semaphore that the
// blit's submission waits on, on the renderer's queue. A wait on a queue holds
// everything queued behind it - the game's next submissions included - until
// the display gives the image back, which on a FIFO swapchain with frames
// queued is the next vblank. By default the acquire signals a fence the
// presenting thread waits for, so only this thread waits for the display.
const bool g_acquire_fence = !present_env_off("BBHOST_PRESENT_ACQUIRE_FENCE");
// BBHOST_PRESENT_SUBMIT=1: before its blit the presenter submits whatever the
// renderer is recording, as it did on every frame. The frame it shows is
// normally submitted already (the command processor submits each job before
// it queues the flip), and what it cut short was the next frame's recording:
// its render pass closed, and one more submission a frame. By default it
// submits only when the flip's frame is still in the recording.
const bool g_present_submit_always = [] {
    const char* e = std::getenv("BBHOST_PRESENT_SUBMIT");
    return e ? e[0] == '1' : present_legacy();
}();
// BBHOST_PRESENT_GPU_WAIT=1 or 0: whether the presenting thread sleeps on the
// blit's fence before vkQueuePresentKHR. AMD's Windows driver polls inside the
// present until the frame's GPU work is done (Kyo's trace of his fork: the
// present thread at 97% of a core, mostly that poll - on an APU, power the GPU
// could have had), and showing frames as they arrive (video.cpp) makes that
// the whole of the frame's GPU time; asleep first, the driver's wait finds the
// frame done, and the present waits on the same frame either way. So it is on
// for AMD's own driver (vk_start). Elsewhere the present does not poll, and a
// wait there would only queue the next frame's copy later.
bool present_gpu_wait_default(VkDriverId driver) {
    const char* e = std::getenv("BBHOST_PRESENT_GPU_WAIT");
    if (e) return e[0] == '1';
    return !present_legacy() && driver == VK_DRIVER_ID_AMD_PROPRIETARY;
}

const char* present_mode_name(VkPresentModeKHR m) {
    switch (m) {
        case VK_PRESENT_MODE_IMMEDIATE_KHR: return "IMMEDIATE";
        case VK_PRESENT_MODE_MAILBOX_KHR: return "MAILBOX";
        case VK_PRESENT_MODE_FIFO_KHR: return "FIFO";
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "FIFO_RELAXED";
#if defined(VK_KHR_present_mode_fifo_latest_ready)
        case VK_PRESENT_MODE_FIFO_LATEST_READY_KHR: return "FIFO_LATEST_READY";
#endif
        default: return "other";
    }
}

// The present mode for the V-Sync setting, from what the surface offers.
// V-Sync on: MAILBOX (the newest finished frame goes up at the next refresh, no
// tearing, no queue of frames behind the display), else FIFO_LATEST_READY (the
// same thing as a FIFO mode: AMD's Windows driver has it and no MAILBOX), else
// FIFO - where every frame waits its turn, up to two refreshes behind with
// three images. V-Sync off: IMMEDIATE (shown at once, tearing), else MAILBOX.
// FIFO is the only mode a surface must support, so it is what is left.
// V-Sync on where the driver presents through DXGI (NVIDIA's on a hybrid
// laptop's display wired to the integrated GPU): FIFO. There MAILBOX - and
// IMMEDIATE - fall to ~5 presents a second once the game's GPU work passes
// ~60% busy, the game still at 60, and stay there while it does; FIFO holds
// 60 (RTX 4060 Laptop, 610.88: 937 frames dropped in a 75 s run against 6).
// BBHOST_PRESENT_MODE=fifo|mailbox|latest|immediate|relaxed asks for one
// (FIFO when the surface lacks it); =old is the choice before this one (FIFO
// with V-Sync; MAILBOX, else IMMEDIATE, without).
VkPresentModeKHR choose_present_mode(bool vsync, std::string& why) {
    std::uint32_t nmodes = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.phys, g_vk.surface, &nmodes, nullptr);
    std::vector<VkPresentModeKHR> modes(nmodes);
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.phys, g_vk.surface, &nmodes, modes.data());
    const auto offered = [&](VkPresentModeKHR m) {
#if defined(VK_KHR_present_mode_fifo_latest_ready)
        if (m == VK_PRESENT_MODE_FIFO_LATEST_READY_KHR && !g_vk.latest_ready) return false;  // the device's extension is off
#endif
        return std::find(modes.begin(), modes.end(), m) != modes.end();
    };
    why = "the surface offers";
    for (VkPresentModeKHR m : modes) why += std::string(" ") + present_mode_name(m);
    std::vector<VkPresentModeKHR> want;
    const char* forced = std::getenv("BBHOST_PRESENT_MODE");
    const std::string f = forced ? forced : present_legacy() ? "old" : "";
    if (f == "old") {
        if (!vsync) want = {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR};
        why += "; the old choice";
    } else if (!f.empty()) {
        if (f == "mailbox") want = {VK_PRESENT_MODE_MAILBOX_KHR};
        if (f == "immediate") want = {VK_PRESENT_MODE_IMMEDIATE_KHR};
        if (f == "relaxed") want = {VK_PRESENT_MODE_FIFO_RELAXED_KHR};
#if defined(VK_KHR_present_mode_fifo_latest_ready)
        if (f == "latest") want = {VK_PRESENT_MODE_FIFO_LATEST_READY_KHR};
#endif
        why += "; BBHOST_PRESENT_MODE=" + f;
    } else if (vsync && g_vk.via_dxgi) {
        why += "; FIFO: the driver presents through DXGI";
    } else if (vsync) {
        want = {VK_PRESENT_MODE_MAILBOX_KHR};
#if defined(VK_KHR_present_mode_fifo_latest_ready)
        want.push_back(VK_PRESENT_MODE_FIFO_LATEST_READY_KHR);
#endif
    } else {
        want = {VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_MAILBOX_KHR};
    }
    for (VkPresentModeKHR m : want) {
        if (offered(m)) return m;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

// Whether the surface's size differs from the swapchain's: what makes a
// SUBOPTIMAL result worth a rebuild.
bool surface_size_changed() {
    VkSurfaceCapabilitiesKHR caps{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_vk.phys, g_vk.surface, &caps) != VK_SUCCESS) return false;
    VkExtent2D now = caps.currentExtent;
    if (now.width == 0xFFFFFFFFu) {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(g_window, &w, &h);
        now = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)};
    }
    const bool changed = now.width != g_vk.extent.width || now.height != g_vk.extent.height;
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 8) {
        host_log("present: the swapchain is suboptimal (%ux%u, the surface %ux%u)%s", g_vk.extent.width, g_vk.extent.height, now.width,
                 now.height, changed ? "; rebuilt for the new size" : "; kept");
    }
    return changed;
}

bool vk_create_swapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_vk.phys, g_vk.surface, &caps);
    std::uint32_t nfmt = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.phys, g_vk.surface, &nfmt, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(nfmt);
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.phys, g_vk.surface, &nfmt, fmts.data());
    // B8G8R8A8 where the surface lists it. BBHOST_SWAPCHAIN_FORMAT=rgba8:
    // R8G8B8A8 instead, as shadPS4 makes its swapchain - with the other
    // swapchain switches below, an A/B for the path the driver takes to the
    // screen (host/gpu_busy.cpp). The blit converts,
    // and the overlay's and FSR's targets follow the format.
    static const VkFormat wanted = [] {
        const char* e = std::getenv("BBHOST_SWAPCHAIN_FORMAT");
        const std::string v = e ? e : "";
        if (v == "rgba8") return VK_FORMAT_R8G8B8A8_UNORM;
        if (!v.empty() && v != "bgra8") host_log("present: BBHOST_SWAPCHAIN_FORMAT=%s is not rgba8 or bgra8; ignored", v.c_str());
        return VK_FORMAT_B8G8R8A8_UNORM;
    }();
    g_vk.format = fmts.empty() ? VK_FORMAT_B8G8R8A8_UNORM : fmts[0].format;
    for (const auto& f : fmts) {
        if (f.format == wanted && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            g_vk.format = f.format;
        }
    }
    g_vk.extent = caps.currentExtent;
    if (g_vk.extent.width == 0xFFFFFFFFu) {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(g_window, &w, &h);
        g_vk.extent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)};
    }
    if (g_vk.extent.width == 0 || g_vk.extent.height == 0) {
        return false;
    }
    g_window_pixels.store(static_cast<std::uint64_t>(g_vk.extent.width) << 32 | g_vk.extent.height, std::memory_order_relaxed);
    VkSwapchainKHR old = g_vk.swapchain;
    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = g_vk.surface;
    // One more than the least the surface takes: with FIFO that lets one frame
    // wait for the display while the next is drawn, and MAILBOX and
    // FIFO_LATEST_READY need a third - one shown, one waiting, one drawn - to
    // replace the waiting one rather than block. More only adds frames the
    // display can fall behind by. BBHOST_SWAPCHAIN_IMAGES=<n> asks for n.
    std::uint32_t images = caps.minImageCount + 1;
    if (const char* e = std::getenv("BBHOST_SWAPCHAIN_IMAGES"); e && std::atoi(e) > 0) images = static_cast<std::uint32_t>(std::atoi(e));
    images = std::max(images, caps.minImageCount);
    if (caps.maxImageCount) images = std::min(images, caps.maxImageCount);
    sci.minImageCount = images;
    sci.imageFormat = g_vk.format;
    sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sci.imageExtent = g_vk.extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    // Storage too where the surface and the format allow it: FSR's last pass
    // then writes the swapchain image itself instead of a copy blitted into it.
    // BBHOST_FSR_DIRECT=0: always the blit.
    {
        static const bool direct = [] {
            const char* e = std::getenv("BBHOST_FSR_DIRECT");
            return !(e && e[0] == '0');
        }();
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(g_vk.phys, g_vk.format, &fp);
        g_vk.storage = direct && (caps.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) &&
                       (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
        if (g_vk.storage) sci.imageUsage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    std::string why;
    const bool vsync = g_want_vsync.load();
    sci.presentMode = choose_present_mode(vsync, why);
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = old;
    // Windows' graphics modules before and after: a driver that presents
    // through DXGI (a D3D12 queue of its own copying our images into a DXGI
    // swapchain) loads d3d12.dll inside vkCreateSwapchainKHR, as happened in
    // shadPS4 on the Radeon 8060S (ReShade's log there, 2026-10-07).
    [[maybe_unused]] const std::string modules_before = host_graphics_modules();
    if (vkCreateSwapchainKHR(g_vk.device, &sci, nullptr, &g_vk.swapchain) != VK_SUCCESS) {
        host_log("vulkan: swapchain creation failed");
        return false;
    }
    [[maybe_unused]] const std::string modules_after = host_graphics_modules();
    g_vk.mode = sci.presentMode;
    bool remake = false;  // DXGI seen for the first time and it changes the mode: made again below
    if (old) {
        vkDestroySwapchainKHR(g_vk.device, old, nullptr);
        host_log("present: swapchain %ux%u", g_vk.extent.width, g_vk.extent.height);
    }
    {
        // Which way the frames go to the screen, each time it is made.
        const auto mode_name = [](VkPresentModeKHR m) -> const char* {
            switch (m) {
                case VK_PRESENT_MODE_IMMEDIATE_KHR: return "immediate";
                case VK_PRESENT_MODE_MAILBOX_KHR: return "mailbox";
                case VK_PRESENT_MODE_FIFO_KHR: return "fifo";
                case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "fifo-relaxed";
#if defined(VK_KHR_present_mode_fifo_latest_ready)
                case VK_PRESENT_MODE_FIFO_LATEST_READY_KHR: return "fifo-latest-ready";
#endif
                default: return "other";
            }
        };
        std::uint32_t nmodes = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.phys, g_vk.surface, &nmodes, nullptr);
        std::vector<VkPresentModeKHR> modes(nmodes);
        if (nmodes) vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.phys, g_vk.surface, &nmodes, modes.data());
        std::string have;
        for (VkPresentModeKHR m : modes) have += std::string(have.empty() ? "" : " ") + mode_name(m);
        const std::string chosen = why.empty() ? std::string() : " (" + why + ")";
        host_log("present: mode %s%s (the surface has: %s), %s, %s, %u images, usage 0x%x%s", mode_name(sci.presentMode),
                 chosen.c_str(), have.c_str(),
                 g_want_fullscreen.load() ? "fullscreen" : "windowed",
                 sci.imageFormat == VK_FORMAT_R8G8B8A8_UNORM   ? "R8G8B8A8"
                 : sci.imageFormat == VK_FORMAT_B8G8R8A8_UNORM ? "B8G8R8A8"
                                                               : "another format",
                 sci.minImageCount, static_cast<unsigned>(sci.imageUsage),
                 g_vk.storage ? " (storage: FSR writes the images; BBHOST_FSR_DIRECT=0 not)" : "");
#if defined(_WIN32)
        const bool via_dxgi = modules_after.find("d3d12.dll") != std::string::npos && modules_before.find("d3d12.dll") == std::string::npos;
        host_log("present: Windows' graphics modules: %s before the swapchain, %s after%s",
                 modules_before.empty() ? "none" : modules_before.c_str(), modules_after.empty() ? "none" : modules_after.c_str(),
                 via_dxgi ? " - the driver presents through DXGI" : "");
        // Only the first swapchain sees d3d12.dll load, so the finding is kept.
        if (via_dxgi && !g_vk.via_dxgi) {
            g_vk.via_dxgi = true;
            std::string again;
            remake = choose_present_mode(vsync, again) != g_vk.mode;
        }
#endif
    }
    g_swap_w.store(g_vk.extent.width, std::memory_order_relaxed);
    g_swap_h.store(g_vk.extent.height, std::memory_order_relaxed);
    std::uint32_t n = 0;
    vkGetSwapchainImagesKHR(g_vk.device, g_vk.swapchain, &n, nullptr);
    g_vk.images.resize(n);
    vkGetSwapchainImagesKHR(g_vk.device, g_vk.swapchain, &n, g_vk.images.data());
    for (VkImageView v : g_vk.views) {
        if (v) vkDestroyImageView(g_vk.device, v, nullptr);
    }
    g_vk.views.assign(n, VK_NULL_HANDLE);
    for (std::uint32_t i = 0; i < n; ++i) {
        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivci.image = g_vk.images[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = g_vk.format;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(g_vk.device, &ivci, nullptr, &g_vk.views[i]);
    }
    // Fresh ones for the new images; the old swapchain's are left alive (a
    // few semaphores a resize), since a present of its images may hold them.
    g_vk.render_sems.clear();
    for (std::uint32_t i = 0; i < n; ++i) {
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore sem = VK_NULL_HANDLE;
        if (vkCreateSemaphore(g_vk.device, &sci, nullptr, &sem) != VK_SUCCESS) sem = VK_NULL_HANDLE;
        g_vk.render_sems.push_back(sem);
    }
    host_overlay_init(g_vk.device, g_vk.phys, g_vk.format, g_vk.queue, g_vk.family);
    ingame_menu_init(g_vk.instance, g_vk.phys, g_vk.device, g_vk.family, g_vk.queue, g_vk.format, n);
    const bool pass = g_present_pass && present_pass_init(g_vk.device, g_vk.phys, g_vk.format);
    host_log("present: %s, V-Sync %s (%s); %u images (%u asked, the surface's least %u); %s", present_mode_name(g_vk.mode), vsync ? "on" : "off",
             why.c_str(), n, images, caps.minImageCount,
             pass ? "the picture, the overlay and the menu in one pass" : "the picture blitted, the overlay in a pass of its own");
    if (remake) {
        host_log("present: made again for the mode the DXGI path takes");
        return vk_create_swapchain();
    }
    return true;
}

bool vk_start() {
    std::uint32_t next = 0;
    const char* const* ext = SDL_Vulkan_GetInstanceExtensions(&next);
    if (!ext) {
        host_log("vulkan: SDL has no instance extensions (%s)", SDL_GetError());
        return false;
    }
    if (!host_gpu_init(ext, next, true)) {
        host_log("vulkan: shared GPU device unavailable");
        return false;
    }
    const GpuHandles h = host_gpu_handles();
    g_vk.instance = static_cast<VkInstance>(h.instance);
    g_vk.phys = static_cast<VkPhysicalDevice>(h.physical);
    g_vk.device = static_cast<VkDevice>(h.device);
    g_vk.queue = static_cast<VkQueue>(h.queue);
    g_vk.present_queue = static_cast<VkQueue>(h.present_queue);
    g_vk.family = h.family;
    g_vk.latest_ready = h.fifo_latest_ready;
    if (!SDL_Vulkan_CreateSurface(g_window, g_vk.instance, nullptr, &g_vk.surface)) {
        host_log("vulkan: surface creation failed: %s", SDL_GetError());
        return false;
    }
    VkBool32 present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(g_vk.phys, g_vk.family, g_vk.surface, &present);
    if (!present) {
        host_log("vulkan: the GPU queue family cannot present to this surface");
        return false;
    }
    VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &drv;
    vkGetPhysicalDeviceProperties2(g_vk.phys, &props2);
    const VkPhysicalDeviceProperties& props = props2.properties;
    g_vk.gpu_wait = present_gpu_wait_default(drv.driverID);
    if (!vk_create_swapchain()) {
        return false;
    }
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = g_vk.family;
    vkCreateCommandPool(g_vk.device, &pci, nullptr, &g_vk.pool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = g_vk.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(g_vk.device, &cai, &g_vk.cmd);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(g_vk.device, &sci, nullptr, &g_vk.acquire_sem);
    vkCreateSemaphore(g_vk.device, &sci, nullptr, &g_vk.render_sem);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(g_vk.device, &fci, nullptr, &g_vk.fence);
    VkFenceCreateInfo afci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(g_vk.device, &afci, nullptr, &g_vk.acquire_fence) != VK_SUCCESS) g_vk.acquire_fence = VK_NULL_HANDLE;
    g_vk.ok = true;
    host_log("vulkan: %s, swapchain %ux%u, %zu images", props.deviceName, g_vk.extent.width, g_vk.extent.height,
             g_vk.images.size());
    host_log("present: the acquire %s; the renderer's recording submitted %s; the present %s",
             g_acquire_fence && g_vk.acquire_fence ? "waited for on this thread (a fence)" : "waited for on the GPU queue (a semaphore)",
             g_present_submit_always ? "before every blit" : "only when the flip's frame is still in it",
             g_vk.gpu_wait ? "after the frame is done on the GPU (a sleeping wait)" : "at once (the driver waits for the frame)");
    return true;
}

std::atomic<std::uint64_t> g_present_dropped{0};

// The FPS counter (host setting fps_counter): what reaches the screen - every
// present vk_present makes, not the flips it drops - over the last second.
std::mutex g_fps_mu;
std::deque<std::chrono::steady_clock::time_point> g_present_times;  // the last second's presents

void note_present(std::chrono::steady_clock::time_point t) {
    std::lock_guard<std::mutex> lock(g_fps_mu);
    g_present_times.push_back(t);
    while (!g_present_times.empty() && t - g_present_times.front() > std::chrono::seconds(1)) {
        g_present_times.pop_front();
    }
}

int presents_in_last_second(std::chrono::steady_clock::time_point now) {
    std::lock_guard<std::mutex> lock(g_fps_mu);
    int n = 0;
    for (const auto& t : g_present_times) n += now - t <= std::chrono::seconds(1);
    return n;
}

// Which step of a present this thread is on, for the hang watchdog: a present
// that never comes back stops the flips, and everything behind them.
std::atomic<const char*> g_present_step{"idle"};
std::atomic<std::uint64_t> g_present_step_ns{0};
void present_step(const char* what) {
    g_present_step_ns.store(static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()),
                            std::memory_order_relaxed);
    g_present_step.store(what, std::memory_order_relaxed);
}

// What the presenting thread did over the last 300 presents, for the second
// `present:` line. Only that thread touches it.
struct PresentStats {
    std::uint64_t cpu_ns = 0;  // bb-present's own CPU time in vk_present
    std::uint64_t latency_ns = 0, latency_max_ns = 0, latency_n = 0;  // flip queued -> its present returned
    std::uint64_t on_arrival = 0;                                     // frames shown before their vblank (video.cpp)
    std::uint64_t submits_forced = 0, submits_left = 0;               // the renderer's recording, submitted here or not
    std::uint64_t one_pass = 0, blits = 0, clears = 0, overlay_passes = 0;
    std::uint64_t overlay_frames = 0, menu_frames = 0;
    std::uint64_t gpu_wait_us = 0, acquire_wait_us = 0;
};
PresentStats g_pstats;

void vk_present(int buffer_index, std::uint64_t display_va, unsigned display_w, unsigned display_h, const PresentFlip& flip) {
    const auto t_enter = std::chrono::steady_clock::now();
    const std::uint64_t cpu_enter = host_thread_cpu_ns();
    const auto us = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
    };
    present_step("taking the present lock");
    std::lock_guard<std::mutex> lock(g_vk.mu);
    if (!g_vk.ok) {
        return;
    }
    // BBHOST_PRESENT_DELAY_US: pretend the display blocks us for this long, so
    // a run on a virtual display (Xvfb never blocks in acquire) can be checked
    // against what a real vsync does to the game's frame rate.
    static const long fake_wait_us = [] {
        const char* e = std::getenv("BBHOST_PRESENT_DELAY_US");
        return e ? std::strtol(e, nullptr, 10) : 0L;
    }();
    if (fake_wait_us > 0) host_sleep_us(static_cast<std::uint64_t>(fake_wait_us));
    // Wait for the previous present and acquire the next image *without* the
    // renderer lock: on a FIFO swapchain the acquire blocks until the display's
    // next vblank, and holding the GPU lock across that stalls the command
    // processor - the game's own thread - for most of a frame.
    present_step("waiting for the last present");
    vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX);
    host_gpu_busy_present_done();  // the last blit's two timestamps, before this one rewrites them
    // A V-Sync change has no surface event behind it, so the options screen
    // asks for the rebuild the acquire would otherwise never be told to do.
    if (g_swap_dirty.exchange(false)) {
        host_gpu_lock();
        host_gpu_queue_lock();
        vkDeviceWaitIdle(g_vk.device);
        host_gpu_queue_unlock();
        vk_create_swapchain();
        host_gpu_unlock();
    }
    present_step("acquiring an image");
    // The image is ours once the acquire's fence signals (or, the old way, once
    // the GPU has waited on its semaphore: BBHOST_PRESENT_ACQUIRE_FENCE=0).
    const bool by_fence = g_acquire_fence && g_vk.acquire_fence;
    const VkSemaphore acquire_sem = by_fence ? VK_NULL_HANDLE : g_vk.acquire_sem;
    const VkFence acquire_fence = by_fence ? g_vk.acquire_fence : VK_NULL_HANDLE;
    if (by_fence) vkResetFences(g_vk.device, 1, &g_vk.acquire_fence);
    std::uint32_t idx = 0;
    // A bounded wait: a window nobody sees gets no frame callbacks on Wayland
    // (a Steam Deck's blanked screen, a minimized window), and an acquire
    // without a timeout then waited for ever - the presenter stuck, the game
    // waiting on its flips (two runs frozen 20 s into the title, "acquiring an
    // image (43.4 s)"). Past 100 ms this frame is not presented; the game goes
    // on and the next frame tries again.
    VkResult r = vkAcquireNextImageKHR(g_vk.device, g_vk.swapchain, 100000000ull, acquire_sem, acquire_fence, &idx);
    if (r == VK_TIMEOUT || r == VK_NOT_READY) {
        static std::atomic<int> said{0};
        if (said.fetch_add(1) < 4) host_log("present: no swapchain image within 100 ms (the window is not being shown?); frame not presented");
        return;
    }
    // SUBOPTIMAL is a success: the image is ours and can be presented. Windows
    // can say it every frame (borderless fullscreen after an alt-tab), and a
    // rebuild each time - a whole-device wait and a new swapchain a frame - is
    // a storm (GPU 62% -> 98% on an AMD iGPU). So this frame is presented, and
    // the next one rebuilds only if the surface's size really changed.
    if (r == VK_SUBOPTIMAL_KHR) {
        if (surface_size_changed()) g_swap_dirty.store(true);
        r = VK_SUCCESS;
    }
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        host_gpu_lock();
        host_gpu_queue_lock();
        vkDeviceWaitIdle(g_vk.device);
        host_gpu_queue_unlock();
        const bool remade = vk_create_swapchain();
        host_gpu_unlock();
        if (!remade) {
            return;
        }
        if (by_fence) vkResetFences(g_vk.device, 1, &g_vk.acquire_fence);
        r = vkAcquireNextImageKHR(g_vk.device, g_vk.swapchain, UINT64_MAX, acquire_sem, acquire_fence, &idx);
        if (r == VK_SUBOPTIMAL_KHR) r = VK_SUCCESS;  // ours all the same: not presenting it would keep it for ever
    }
    if (r != VK_SUCCESS) {
        return;
    }
    if (by_fence) {
        // Until the display lets go of the image. With FIFO and frames queued
        // that is a vblank away; this thread sleeps through it, and the
        // renderer's queue never waits for the display.
        present_step("waiting for the acquired image");
        const auto t0 = std::chrono::steady_clock::now();
        vkWaitForFences(g_vk.device, 1, &g_vk.acquire_fence, VK_TRUE, UINT64_MAX);
        g_pstats.acquire_wait_us += us(t0, std::chrono::steady_clock::now());
    }
    const auto t_acquired = std::chrono::steady_clock::now();
    // Queued draws land before the blit: submissions on one queue execute in
    // submission order, so submitting them first is enough. Waiting for them
    // to finish (host_gpu_flush) held the renderer lock through a GPU drain on
    // every presented frame - about 13 ms a frame in the world, all of it the
    // command processor blocked. BBHOST_PRESENT_FLUSH=1 (checks) waits again.
    // And only when the flip's frame is not in a submission yet: by then the
    // recording is usually the next frame's (host_gpu_submit_for_flip).
    static const bool present_flush = [] {
        const char* e = std::getenv("BBHOST_PRESENT_FLUSH");
        return e && e[0] == '1';
    }();
    present_step("submitting the renderer's work");
    if (present_flush) {
        host_gpu_flush();
        ++g_pstats.submits_forced;
    } else if (host_gpu_submit_for_flip(g_present_submit_always ? ~0ull : flip.submit_need)) {
        ++g_pstats.submits_forced;
    } else {
        ++g_pstats.submits_left;
    }
    const auto t_flushed = std::chrono::steady_clock::now();
    present_step("taking the renderer lock");
    host_gpu_lock();
    bool locked = true;
    struct Unlock {
        bool& locked;
        ~Unlock() {
            if (locked) host_gpu_unlock();
        }
    } unlock{locked};
    const auto t_locked = std::chrono::steady_clock::now();
    vkResetFences(g_vk.device, 1, &g_vk.fence);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g_vk.cmd, &bi);
    host_gpu_busy_present_begin(g_vk.cmd);  // the busy meter's first timestamp (host/gpu_busy.cpp)
    // The picture's rectangle (fit_picture); what it does not cover is black.
    VkRect2D area{{0, 0}, g_vk.extent};
    if (display_w && display_h) {
        const FitRect f = fit_picture(static_cast<float>(g_vk.extent.width), static_cast<float>(g_vk.extent.height),
                                      static_cast<float>(display_w), static_cast<float>(display_h));
        const auto x0 = static_cast<std::int32_t>(std::lround(f.x)), y0 = static_cast<std::int32_t>(std::lround(f.y));
        const auto x1 = static_cast<std::int32_t>(std::lround(f.x + f.w)), y1 = static_cast<std::int32_t>(std::lround(f.y + f.h));
        if (x1 > x0 && y1 > y0) area = {{x0, y0}, {static_cast<std::uint32_t>(x1 - x0), static_cast<std::uint32_t>(y1 - y0)}};
    }
    // Another shape than 16:9: the game's menus, loading screens and movies
    // are the 1920x1080 stage in the middle of the picture (Scaleform's
    // show-all), and with no scene drawn - the title, a load, a movie -
    // nothing around it but the frame's clear, a flat grey. Then only the
    // stage's part of the picture is shown, black beside it on a wider
    // screen or above and below it on a taller one (the Steam Deck's 16:10).
    std::uint32_t src_x = 0, src_y = 0, src_w = display_w, src_h = display_h;
    VkRect2D shown = area;
    const std::uint64_t w9 = static_cast<std::uint64_t>(display_w) * 9, h16 = static_cast<std::uint64_t>(display_h) * 16;
    if (display_w && display_h && (w9 > h16 + 9 || h16 > w9 + 16) && engine_scene_view_flip() + 3 < hle_video_flip_count()) {
        if (w9 > h16) {
            src_w = (display_h * 16 + 4) / 9;
            src_x = (display_w - src_w) / 2;
        } else {
            src_h = (display_w * 9 + 8) / 16;
            src_y = (display_h - src_h) / 2;
        }
        const double kx = static_cast<double>(area.extent.width) / static_cast<double>(display_w);
        const double ky = static_cast<double>(area.extent.height) / static_cast<double>(display_h);
        shown.offset.x = area.offset.x + static_cast<std::int32_t>(std::lround(src_x * kx));
        shown.offset.y = area.offset.y + static_cast<std::int32_t>(std::lround(src_y * ky));
        shown.extent.width = static_cast<std::uint32_t>(std::lround(src_w * kx));
        shown.extent.height = static_cast<std::uint32_t>(std::lround(src_h * ky));
    }
    const bool bars = shown.extent.width != g_vk.extent.width || shown.extent.height != g_vk.extent.height;
    const float overlay_w = static_cast<float>(display_w ? display_w : g_vk.extent.width);
    const float overlay_h = static_cast<float>(display_h ? display_h : g_vk.extent.height);
    // The swapchain image is written from the stage its acquire is waited at.
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    bool recorded = false;
    if (g_present_pass && present_pass_ready() && idx < g_vk.views.size() && g_vk.views[idx]) {
        // One rendering (host/present_pass.h): the picture, the overlay and the
        // menu, the bars its clear. Not for a picture smaller than its
        // rectangle: FSR's upscale, on the blit's path below.
        PresentSource src;
        void* image = nullptr;
        std::uint32_t format = 0, iw = 0, ih = 0;
        const bool have = display_va && host_gpu_display_image(display_va, &image, &format, &iw, &ih);
        bool magnified = false;
        if (have) {
            src.image = static_cast<VkImage>(image);
            src.format = static_cast<VkFormat>(format);
            src.width = iw;
            src.height = ih;
            // The region render_blit_display_locked takes: the registered
            // size, inside the target (which may carry tiling padding rows).
            src.x = std::min(src_x, iw);
            src.y = std::min(src_y, ih);
            src.w = src_w && src.x + src_w <= iw ? src_w : iw - src.x;
            src.h = src_h && src.y + src_h <= ih ? src_h : ih - src.y;
            magnified = src.w <= shown.extent.width && src.h <= shown.extent.height &&
                        (src.w < shown.extent.width || src.h < shown.extent.height);
        }
        if (!magnified) {
            // Dark red (cycling with the buffer index) when the renderer has
            // not produced the frame yet, as the blit's path clears.
            const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            const float none[4] = {0.06f + 0.04f * static_cast<float>(buffer_index % 3), 0.0f, 0.02f, 1.0f};
            if (present_pass_begin(g_vk.cmd, g_vk.images[idx], g_vk.views[idx], g_vk.extent, shown, have ? &src : nullptr,
                                   have ? black : none)) {
                const std::uint32_t verts = host_overlay_record(g_vk.cmd, area, overlay_w, overlay_h);
                // The plugin menu (F9) over everything else (host/ingame_menu.h).
                const bool menu = ingame_menu_record_in_pass(g_vk.cmd, g_vk.extent);
                present_pass_end(g_vk.cmd, g_vk.images[idx]);
                recorded = true;
                wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                ++g_pstats.one_pass;
                g_pstats.overlay_frames += verts != 0;
                g_pstats.menu_frames += menu;
            }
        }
    }
    if (!recorded) {
        // The renderer's writes to what the blit reads - the frame taken at its
        // flip, or the display buffer - are in earlier submissions on this queue;
        // submission order alone does not make them visible to this read.
        VkMemoryBarrier frame_written{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        frame_written.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        frame_written.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &frame_written, 0,
                             nullptr, 0, nullptr);
        VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = g_vk.images[idx];
        to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        // From the stage the submission waits for the acquire at (wait_stage,
        // below): from the top of the pipe, the layout change was not ordered
        // after the acquire, and could rewrite the image while the presentation
        // engine still read it (the validation layer's WRITE_AFTER_READ).
        vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &to_dst);
        if (bars) {
            const VkClearColorValue black{{0.0f, 0.0f, 0.0f, 1.0f}};
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(g_vk.cmd, g_vk.images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
            VkMemoryBarrier cleared{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            cleared.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            cleared.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &cleared, 0, nullptr, 0,
                                 nullptr);
            ++g_pstats.clears;
        }
        // Blit the game's display buffer; dark red (cycling with the buffer
        // index) when the renderer has not produced it yet.
        void* const storage_view = g_vk.storage && idx < g_vk.views.size() ? static_cast<void*>(g_vk.views[idx]) : nullptr;
        if (!display_va || !host_gpu_blit_display(g_vk.cmd, display_va, g_vk.images[idx], shown.offset.x, shown.offset.y,
                                                  shown.extent.width, shown.extent.height, src_w, src_h, src_x, src_y, storage_view)) {
            VkClearColorValue color{{0.06f + 0.04f * (buffer_index % 3), 0.0f, 0.02f, 1.0f}};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(g_vk.cmd, g_vk.images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
            ++g_pstats.clears;
        } else {
            ++g_pstats.blits;
        }
        // The host overlay goes over the game's frame: the pointer now, the text
        // box and the options screen later. Drawing needs
        // the image as a colour attachment rather than a transfer destination.
        bool as_colour = false;  // the image was moved to the colour-attachment layout, whether or not the overlay then drew
        if (idx < g_vk.views.size() && g_vk.views[idx] && (!host_overlay_empty() || ingame_menu_open())) {
            VkImageMemoryBarrier to_colour = to_dst;
            to_colour.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_colour.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            to_colour.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_colour.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &to_colour);
            as_colour = true;
            const bool overlaid = host_overlay_draw(g_vk.device, g_vk.cmd, g_vk.images[idx], g_vk.views[idx], g_vk.extent, area,
                                                    overlay_w, overlay_h);
            g_pstats.overlay_passes += overlaid;
            g_pstats.overlay_frames += overlaid;
            // The plugin menu (F9) over everything else (host/ingame_menu.h).
            g_pstats.overlay_passes += ingame_menu_open();
            g_pstats.menu_frames += ingame_menu_open();
            ingame_menu_record(g_vk.cmd, g_vk.views[idx], g_vk.extent);
        }
        // From the layout the image is in: colour attachment once the overlay's
        // barrier ran, even when it then drew nothing (it said transfer
        // destination then - the validation layer's layout mismatch, and what a
        // driver that keeps colour compressed would read wrongly).
        VkImageMemoryBarrier to_present = to_dst;
        to_present.oldLayout = as_colour ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        to_present.srcAccessMask = as_colour ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
        to_present.dstAccessMask = 0;
        vkCmdPipelineBarrier(g_vk.cmd,
                             as_colour ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_present);
    }
    host_gpu_busy_present_end(g_vk.cmd);  // counted as on its way from here: it is always submitted below
    vkEndCommandBuffer(g_vk.cmd);
    const auto t_recorded = std::chrono::steady_clock::now();
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = by_fence ? 0 : 1;
    si.pWaitSemaphores = by_fence ? nullptr : &acquire_sem;
    si.pWaitDstStageMask = by_fence ? nullptr : &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_vk.cmd;
    const VkSemaphore rendered = idx < g_vk.render_sems.size() && g_vk.render_sems[idx] ? g_vk.render_sems[idx] : g_vk.render_sem;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &rendered;
    present_step("submitting the blit");
    // Through the submission thread when it runs: queued behind the game's
    // work under the renderer's lock, which then goes at once - the kernel's
    // ~1.3 ms of vkQueueSubmit on a Steam Deck, held under it, cost the
    // command processor ~1.5 ms a frame. Only presenting waits for the
    // submission to have gone in.
    const std::uint64_t ticket = host_gpu_submit_presenter(g_vk.cmd, by_fence ? nullptr : acquire_sem, wait_stage, rendered, g_vk.fence);
    if (ticket) {
        host_gpu_unlock();
        locked = false;
        host_gpu_wait_submitted(ticket);
    } else {
        host_gpu_queue_lock();
        vkQueueSubmit(g_vk.queue, 1, &si, g_vk.fence);
        host_gpu_queue_unlock();
        host_gpu_unlock();
        locked = false;
    }
    const auto t_submitted = std::chrono::steady_clock::now();
    // The frame done on the GPU before the present is asked for, on no lock
    // (BBHOST_PRESENT_GPU_WAIT). Bounded: a lost device signals nothing.
    if (g_vk.gpu_wait) {
        present_step("waiting for the frame on the GPU");
        vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, 1000000000ull);
    }
    const auto t_gpu_done = std::chrono::steady_clock::now();
    // The present itself goes on this thread's own queue, with no lock: it can
    // block - Xvfb copies the image inside it, ~20 ms a frame, and a full FIFO
    // waits for the display - and under the renderer's lock that held the
    // command processor for 44% of the time. A device with one queue presents
    // on the renderer's under its queue lock (the submission thread's blit is
    // in; what the game queued since need not be), and with no submission
    // thread under the renderer's lock too, which is what orders the command
    // processor's own submits there.
    bool queue_held = false;
    if (!g_vk.present_queue) {
        if (!ticket) {
            host_gpu_lock();
            locked = true;
        }
        host_gpu_queue_lock_only();
        queue_held = true;
    }
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &rendered;
    pi.swapchainCount = 1;
    pi.pSwapchains = &g_vk.swapchain;
    pi.pImageIndices = &idx;
    present_step("presenting");
    const VkResult pr = vkQueuePresentKHR(g_vk.present_queue ? g_vk.present_queue : g_vk.queue, &pi);
    if (queue_held) host_gpu_queue_unlock();
    if (locked) {
        host_gpu_unlock();
        locked = false;
    }
    // The next frame rebuilds first when the present says the swapchain no
    // longer fits: out of date always, suboptimal only on a new size.
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || (pr == VK_SUBOPTIMAL_KHR && surface_size_changed())) g_swap_dirty.store(true);
    const auto t_presented = std::chrono::steady_clock::now();
    present_step("idle");
    ++g_vk.frames;
    note_present(t_presented);
    // Where a windowed frame goes: waiting for the display (fence + acquire) or
    // doing work (flush, blit, submit). A choppy windowed run with a healthy
    // headless one is decided by these two numbers. The work splits into
    // submitting the renderer's queued work, waiting for the renderer lock,
    // recording the blit, the queue submit and the present; the blit holds
    // the renderer lock, and so does the present when the device has neither
    // a second queue nor the submission thread. The wait for the frame on the
    // GPU before the present is in the second line, not in the work.
    static std::uint64_t wait_us = 0, work_us = 0, presents = 0;
    static std::uint64_t flush_us = 0, lock_us = 0, blit_us = 0, submit_us = 0, queue_present_us = 0;
    static auto last = std::chrono::steady_clock::now();
    wait_us += us(t_enter, t_acquired);
    work_us += us(t_acquired, t_submitted) + us(t_gpu_done, t_presented);
    flush_us += us(t_acquired, t_flushed);
    lock_us += us(t_flushed, t_locked);
    blit_us += us(t_locked, t_recorded);
    submit_us += us(t_recorded, t_submitted);
    queue_present_us += us(t_gpu_done, t_presented);
    PresentStats& ps = g_pstats;
    ps.gpu_wait_us += us(t_submitted, t_gpu_done);
    if (flip.arrived_ns) {
        const auto now_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_presented.time_since_epoch()).count());
        const std::uint64_t lat = now_ns > flip.arrived_ns ? now_ns - flip.arrived_ns : 0;
        ps.latency_ns += lat;
        ps.latency_max_ns = std::max(ps.latency_max_ns, lat);
        ++ps.latency_n;
    }
    ps.on_arrival += flip.on_arrival;
    ps.cpu_ns += host_thread_cpu_ns() - cpu_enter;
    if (++presents % 300 == 0) {
        const auto now = std::chrono::steady_clock::now();
        host_log("present: 300 frames in %lld ms (display wait %llu ms, work %llu ms: submit queued work %llu, lock wait %llu, blit %llu, "
                 "queue submit %llu, present %llu), %llu dropped so far",
                 static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count()),
                 static_cast<unsigned long long>(wait_us / 1000), static_cast<unsigned long long>(work_us / 1000),
                 static_cast<unsigned long long>(flush_us / 1000), static_cast<unsigned long long>(lock_us / 1000),
                 static_cast<unsigned long long>(blit_us / 1000), static_cast<unsigned long long>(submit_us / 1000),
                 static_cast<unsigned long long>(queue_present_us / 1000), static_cast<unsigned long long>(g_present_dropped.load()));
        // What each of those frames cost and how old it was when it went up:
        // bb-present's own CPU, the time from the command processor queueing
        // the flip to its present returning, how many were shown as they
        // arrived, who submitted the renderer's recording, what was recorded
        // to put it on screen (one pass, or a blit with a clear for bars and a
        // rendering of its own for the overlay) and how long this thread slept
        // on the GPU's frame and on the display's image.
        const double n = 300.0;
        host_log("present: per frame: bb-present CPU %.3f ms; flip to present %.1f ms (max %.1f), %llu of 300 shown on arrival; "
                 "the renderer's recording submitted here %llu, left to the command processor %llu; one pass %llu, blits %llu, clears %llu, "
                 "overlay renderings %llu; overlay drawn %llu, F9 menu %llu; slept %.2f ms on the frame's GPU work, %.2f ms on the "
                 "acquired image; %s, %zu images",
                 static_cast<double>(ps.cpu_ns) / 1e6 / n, ps.latency_n ? static_cast<double>(ps.latency_ns) / 1e6 / static_cast<double>(ps.latency_n) : 0.0,
                 static_cast<double>(ps.latency_max_ns) / 1e6, static_cast<unsigned long long>(ps.on_arrival),
                 static_cast<unsigned long long>(ps.submits_forced), static_cast<unsigned long long>(ps.submits_left),
                 static_cast<unsigned long long>(ps.one_pass), static_cast<unsigned long long>(ps.blits), static_cast<unsigned long long>(ps.clears),
                 static_cast<unsigned long long>(ps.overlay_passes), static_cast<unsigned long long>(ps.overlay_frames),
                 static_cast<unsigned long long>(ps.menu_frames), static_cast<double>(ps.gpu_wait_us) / 1e3 / n,
                 static_cast<double>(ps.acquire_wait_us) / 1e3 / n, present_mode_name(g_vk.mode), g_vk.images.size());
        ps = PresentStats{};
        wait_us = work_us = flush_us = lock_us = blit_us = submit_us = queue_present_us = 0;
        last = now;
    }
}

// ---- input -----------------------------------------------------------------
constexpr std::uint32_t kL3 = 0x2, kR3 = 0x4, kOptions = 0x8, kUp = 0x10, kRight = 0x20, kDown = 0x40, kLeft = 0x80,
                        kL2 = 0x100, kR2 = 0x200, kL1 = 0x400, kR1 = 0x800, kTriangle = 0x1000, kCircle = 0x2000,
                        kCross = 0x4000, kSquare = 0x8000, kTouchPad = 0x100000;

std::atomic<bool> g_key_strong{false};

void text_title_locked() {
#if defined(BBHOST_HAVE_SDL3)
    if (!g_window) return;
    if (g_text_want) {
        std::string shown = g_entry.text;
        if (g_text_hidden) shown.assign(g_entry.length(), '*');
        const std::string t = "Bloodborne (bbhost)  |  " + g_text_label + ": " + shown +
                              "_   (Enter = OK, Esc = cancel, Ctrl+V = paste)";
        SDL_SetWindowTitle(g_window, t.c_str());
    } else {
        SDL_SetWindowTitle(g_window, "Bloodborne (bbhost)");
    }
#endif
}

// A box that takes ASCII alone - each of the game's three - has no use for an
// IME: whatever it composed (kana, hanzi) the box would leave out, and the
// console's basic-Latin keyboard has no such thing either. On Windows the
// window's IME is taken off while such a box is up, so the keys type the
// layout's own letters whichever input method is chosen. SDL puts the IME back
// each time text input starts, a focus gain included, which is when this runs.
// Under g_text_mu.
void ime_off_for_ascii_locked() {
#if defined(_WIN32)
    if (!g_window || !text_charset_ascii(g_entry.charset)) return;
    using AssociateContext = void* (*)(void* hwnd, void* himc);  // imm32's ImmAssociateContext
    static const AssociateContext associate = [] {
        SDL_SharedObject* imm = SDL_LoadObject("imm32.dll");  // SDL's own IME support has it loaded already
        return imm ? reinterpret_cast<AssociateContext>(SDL_LoadFunction(imm, "ImmAssociateContext")) : nullptr;
    }();
    void* hwnd = SDL_GetPointerProperty(SDL_GetWindowProperties(g_window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    if (associate && hwnd) associate(hwnd, nullptr);
#endif
}

std::uint8_t axis_to_u8(std::int16_t v) {
    int a = (static_cast<int>(v) + 32768) / 257;
    return static_cast<std::uint8_t>(a < 0 ? 0 : (a > 255 ? 255 : a));
}

void update_pad() {
    PadState p;
    p.timestamp = SDL_GetTicksNS() / 1000;
    // Once, from 20 s in: no window of ours has the keyboard focus, so the
    // keys and the controller are both ignored until it is given back.
    static bool no_focus_noted = false;
    if (!no_focus_noted && SDL_GetTicks() > 20000 && !SDL_GetKeyboardFocus()) {
        no_focus_noted = true;
        host_log("window: no keyboard focus 20 s in - the keys and the controller are ignored until the window has it");
    }
    // The options screen is modal: while it is open the game sees a connected
    // pad with nothing pressed, so nothing the player does to the screen also
    // happens in the game behind it.
    // So is a key binding being chosen (host/bindings.h): from the moment it
    // starts until the key that ended it is let go, or that key would also do
    // what it is bound to - Escape would close the menu it was pressed in.
    const bool binding = host_bind_capture_blocking();
    if (binding) {
        int n = 0;
        const bool* ks = SDL_GetKeyboardState(&n);
        const SDL_MouseButtonFlags mb = SDL_GetMouseState(nullptr, nullptr);
        int held_sc = -1;
        for (int i = 0; ks && i < n && held_sc < 0; ++i) {
            if (ks[i]) held_sc = i;
        }
        host_bind_capture_note_held(mb != 0 || held_sc >= 0);
        static int logs = 0;
        if (host_bind_capturing() < 0 && (mb != 0 || held_sc >= 0) && logs < 8) {
            ++logs;
            host_log("keys: after the capture, waiting for release (key %d, mouse %x)", held_sc,
                     static_cast<unsigned>(mb));
        }
    }
    // And so is a text box. On the console the keyboard is the system's, and
    // while it is up the game's pad reads nothing pressed (the controller is
    // the system's); here the box takes the keys and a controller's confirm
    // and back, and a Circle or a d-pad press meant for it must not also work
    // the menu behind.
    bool typing;
    {
        std::lock_guard<std::mutex> lock(g_text_mu);
        typing = g_text_want;
    }
    if (host_options_open() || ingame_menu_open() || binding || typing) {
        p.connected = true;
        g_key_strong.store(false, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(g_pad_mu);
        g_pad = p;
        return;
    }
    // The controller button that ended a text entry stays out of the pad
    // until it is let go, as the key does (g_text_end_key).
    int end_button = g_text_end_button.load(std::memory_order_relaxed);
    if (end_button >= 0) {
        bool held = false;
        for (SDL_Gamepad* g : g_pads) held = held || SDL_GetGamepadButton(g, static_cast<SDL_GamepadButton>(end_button));
        if (!held) {
            g_text_end_button.store(-1, std::memory_order_relaxed);
            end_button = -1;
        }
    }
    // Each pad's buttons are or-ed together, and each stick axis and trigger
    // is whichever pad pushes it furthest, so two pads never fight.
    bool touch_done = false, back_held = false;
    for (SDL_Gamepad* g_gamepad : g_pads) {
        p.connected = true;
        auto btn = [&](SDL_GamepadButton b, std::uint32_t bit) {
            if (static_cast<int>(b) != end_button && SDL_GetGamepadButton(g_gamepad, b)) {
                p.buttons |= bit;
            }
        };
        btn(SDL_GAMEPAD_BUTTON_SOUTH, kCross);
        btn(SDL_GAMEPAD_BUTTON_EAST, kCircle);
        btn(SDL_GAMEPAD_BUTTON_WEST, kSquare);
        btn(SDL_GAMEPAD_BUTTON_NORTH, kTriangle);
        btn(SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, kL1);
        btn(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, kR1);
        btn(SDL_GAMEPAD_BUTTON_LEFT_STICK, kL3);
        btn(SDL_GAMEPAD_BUTTON_RIGHT_STICK, kR3);
        btn(SDL_GAMEPAD_BUTTON_START, kOptions);
        back_held |= SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_BACK);  // below, after the touchpad's own fingers
        btn(SDL_GAMEPAD_BUTTON_TOUCHPAD, kTouchPad);
        btn(SDL_GAMEPAD_BUTTON_DPAD_UP, kUp);
        btn(SDL_GAMEPAD_BUTTON_DPAD_DOWN, kDown);
        btn(SDL_GAMEPAD_BUTTON_DPAD_LEFT, kLeft);
        btn(SDL_GAMEPAD_BUTTON_DPAD_RIGHT, kRight);
        const auto further = [](std::uint8_t have, std::uint8_t v) {
            return std::abs(static_cast<int>(v) - kStickRest) > std::abs(static_cast<int>(have) - kStickRest) ? v : have;
        };
        p.lx = further(p.lx, axis_to_u8(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFTX)));
        p.ly = further(p.ly, axis_to_u8(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFTY)));
        p.rx = further(p.rx, axis_to_u8(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHTX)));
        p.ry = further(p.ry, axis_to_u8(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHTY)));
        const std::int16_t lt = SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER);
        const std::int16_t rt = SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER);
        p.l2 = std::max(p.l2, static_cast<std::uint8_t>(lt / 128));
        p.r2 = std::max(p.r2, static_cast<std::uint8_t>(rt / 128));
        if (lt > 8000) {
            p.buttons |= kL2;
        }
        if (rt > 8000) {
            p.buttons |= kR2;
        }
        // The touchpad, for the DualShock 4 and DualSense that have one. SDL
        // reports fingers normalised; the pad reports them in the surface's
        // own pixels, which is the space scePadGetControllerInformation told
        // the game about. Ids count up as fingers arrive and are held for as
        // long as a finger stays down, so a swipe is one id, not a new one a
        // frame.
        static std::uint8_t next_touch_id = 1;
        static std::uint8_t held_id[2] = {0, 0};
        if (!touch_done && SDL_GetNumGamepadTouchpads(g_gamepad) > 0) {
            touch_done = true;  // the first pad with a touchpad
            const int fingers = SDL_GetNumGamepadTouchpadFingers(g_gamepad, 0);
            for (int f = 0; f < fingers && p.touch_count < 2; ++f) {
                bool down = false;
                float tx = 0.0f, ty = 0.0f, pressure = 0.0f;
                if (!SDL_GetGamepadTouchpadFinger(g_gamepad, 0, f, &down, &tx, &ty, &pressure)) {
                    continue;
                }
                const int slot = f < 2 ? f : 1;
                if (!down) {
                    held_id[slot] = 0;
                    continue;
                }
                if (!held_id[slot]) {
                    held_id[slot] = next_touch_id++;
                    if (!next_touch_id) next_touch_id = 1;  // 0 is "no finger"
                }
                auto clampf = [](float v, float hi) { return v < 0.0f ? 0.0f : (v > hi ? hi : v); };
                PadState::Touch& t = p.touch[p.touch_count++];
                t.x = static_cast<std::uint16_t>(clampf(tx * kPadTouchW, kPadTouchW - 1.0f));
                t.y = static_cast<std::uint16_t>(clampf(ty * kPadTouchH, kPadTouchH - 1.0f));
                t.id = held_id[slot];
                t.down = true;
            }
        }
    }
    // Back / Select / Share is the touchpad's left side - Gestures - on pads
    // without a touchpad and on those with one alike: the press, and a finger
    // where the keyboard's Gestures key puts one (bindings.cpp), since the game
    // reads the side from the finger. Its own stable id, so holding it is one
    // touch and it can be held with the key.
    if (back_held) {
        p.buttons |= kTouchPad;
        if (p.touch_count < 2) {
            PadState::Touch& t = p.touch[p.touch_count++];
            t.x = static_cast<std::uint16_t>(kPadTouchW / 4);
            t.y = static_cast<std::uint16_t>(kPadTouchH / 2);
            t.id = 201;
            t.down = true;
        }
    }
    // The keyboard, DS3-shaped and rebindable (host/bindings.h): each action
    // held sets its pad bit or stick direction.
    //
    // A pad plugged in keeps working: this **ors** into the same PadState the
    // gamepad filled and only touches a stick axis when its key is held, so
    // whichever device the player reaches for wins without a mode anywhere.
    int nkeys = 0;
    const bool* keys = SDL_GetKeyboardState(&nkeys);
    // The key that ended a text entry stays out of the pad until released.
    const int end_key = g_text_end_key.load(std::memory_order_relaxed);
    if (end_key >= 0 && !(keys && end_key < nkeys && keys[end_key])) g_text_end_key.store(-1, std::memory_order_relaxed);
    if (keys && SDL_GetKeyboardFocus() == g_window) {
        auto down = [&](SDL_Scancode sc) { return sc < nkeys && keys[sc] && static_cast<int>(sc) != g_text_end_key.load(std::memory_order_relaxed); };
        if (down(SDL_SCANCODE_UP)) p.menu_buttons |= kUp;
        if (down(SDL_SCANCODE_DOWN)) p.menu_buttons |= kDown;
        if (down(SDL_SCANCODE_LEFT)) p.menu_buttons |= kLeft;
        if (down(SDL_SCANCODE_RIGHT)) p.menu_buttons |= kRight;
        // OK and Return are whichever buttons the game's region decides and
        // returns with (engine/menu_pointer.h).
        if (down(SDL_SCANCODE_RETURN) || down(SDL_SCANCODE_KP_ENTER)) p.menu_buttons |= menu_confirm_button();
        if (down(SDL_SCANCODE_BACKSPACE)) p.menu_buttons |= menu_back_button();
        if (down(SDL_SCANCODE_ESCAPE)) p.menu_buttons |= kOptions;
        if (p.menu_buttons) p.connected = true;
        bool held[kBindCount];
        const int masked_key = g_text_end_key.load(std::memory_order_relaxed);
        if (masked_key >= 0 && masked_key < nkeys && nkeys <= SDL_SCANCODE_COUNT) {
            bool masked[SDL_SCANCODE_COUNT];
            std::memcpy(masked, keys, static_cast<std::size_t>(nkeys) * sizeof(bool));
            masked[masked_key] = false;
            host_bindings_keys_held(masked, nkeys, held);
        } else {
            host_bindings_keys_held(keys, nkeys, held);
        }
        const bool strong = held[kBindStrongMod];
        host_bindings_apply(held, strong, p);
        g_key_strong.store(strong, std::memory_order_relaxed);
    } else {
        g_key_strong.store(false, std::memory_order_relaxed);
    }
    // BBHOST_TOUCH_TEST=1: the middle mouse button reports a finger on the
    // touchpad at the pointer's position, so the whole path - PadState through
    // ScePadData's touchData - can be exercised without a DualShock 4.
    static const bool touch_test = [] {
        const char* e = std::getenv("BBHOST_TOUCH_TEST");
        return e && e[0] == '1';
    }();
    if (touch_test && p.touch_count == 0) {
        std::lock_guard<std::mutex> lock(g_mouse_mu);
        if ((g_mouse.buttons & 4u) && g_mouse.in_window && g_display_w && g_display_h) {
            PadState::Touch& t = p.touch[p.touch_count++];
            t.x = static_cast<std::uint16_t>(g_mouse.x / g_display_w * kPadTouchW);
            t.y = static_cast<std::uint16_t>(g_mouse.y / g_display_h * kPadTouchH);
            t.id = 1;
            t.down = true;
        }
    }
    if (!g_pads.empty() || g_window) {
        p.connected = true;  // a window means a player is present
    }
    std::lock_guard<std::mutex> lock(g_pad_mu);
    g_pad = p;
}
#endif

}  // namespace


void host_window_set_fullscreen(bool on) {
#if defined(BBHOST_HAVE_SDL3)
    if (g_want_fullscreen.exchange(on) == on) {
        return;
    }
    if (g_window) {
        SDL_SetWindowFullscreen(g_window, on);
        host_log("window: %s", on ? "fullscreen" : "windowed");
    }
#else
    (void)on;
#endif
}

float host_window_refresh_hz() {
#if defined(BBHOST_HAVE_SDL3)
    if (!g_active.load() || !g_window) return 0.0f;
    // Cached: the flip clock asks once a second, and this walks SDL's display
    // list. A mode change or a drag to another monitor is picked up then.
    static std::atomic<float> cached{0.0f};
    static std::atomic<std::uint64_t> asked_ns{0};
    const auto now = static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::uint64_t last = asked_ns.load(std::memory_order_relaxed);
    if (last && now - last < 1000000000ull) return cached.load(std::memory_order_relaxed);
    asked_ns.store(now, std::memory_order_relaxed);
    float hz = 0.0f;
    if (const SDL_DisplayID id = SDL_GetDisplayForWindow(g_window)) {
        if (const SDL_DisplayMode* m = SDL_GetCurrentDisplayMode(id)) hz = m->refresh_rate;
    }
    cached.store(hz, std::memory_order_relaxed);
    return hz;
#else
    return 0.0f;
#endif
}

std::uint64_t host_present_last_ns(std::uint64_t* serial) {
#if defined(BBHOST_HAVE_SDL3)
    *serial = g_present_serial.load(std::memory_order_acquire);
    return g_present_last_ns.load(std::memory_order_relaxed);
#else
    *serial = 0;
    return 0;
#endif
}

void host_window_set_vsync(bool on) {
#if defined(BBHOST_HAVE_SDL3)
    if (g_want_vsync.exchange(on) == on) {
        return;
    }
    g_swap_dirty.store(true);
    host_log("present: v-sync %s", on ? "on" : "off");
#else
    (void)on;
#endif
}

#if defined(BBHOST_HAVE_SDL3)
namespace {
// Set once SDL has made the window: what is left (the shared GPU device and
// the presenter) takes as long as the GPU's start does.
std::atomic<bool> g_window_made{false};
bool window_start_impl(int width, int height, const char* title);
}
#endif

// Some window managers never acknowledge the map from a sandboxed client
// and SDL blocks forever in ShowWindow. Create the window on a helper
// thread and give up after a few seconds; the run then continues headless.
// Not on Windows: a window there belongs to the thread that made it - its
// messages go to that thread's queue, and the system destroys it when that
// thread exits. Made on the helper, it showed for a moment and was gone (a
// Windows 11 laptop, every run), while the game ran on without it; the
// caller is the thread that pumps it (run_guest_start).
bool host_window_start(int width, int height, const char* title) {
#if defined(BBHOST_HAVE_SDL3) && defined(_WIN32)
    return window_start_impl(width, height, title);
#elif defined(BBHOST_HAVE_SDL3)
    // The shared state lives past a return from here (the thread may not
    // have finished), so it is the thread's as much as ours.
    struct Start {
        std::mutex mu;
        std::condition_variable cv;
        bool finished = false, result = false;
    };
    auto st = std::make_shared<Start>();
    std::thread([st, width, height, title = std::string(title)] {
        const bool r = window_start_impl(width, height, title.c_str());
        std::lock_guard<std::mutex> lock(st->mu);
        st->finished = true;
        st->result = r;
        st->cv.notify_all();
    }).detach();
    // Six seconds for SDL to make the window (a display whose window manager
    // never maps it). The rest - the shared GPU device, its pipeline cache
    // (400 MB from a network share took over 8 s), the presenter - is waited
    // for: giving up there ran the game "headless" in a window that appeared
    // anyway, with nothing reading the keyboard or the controller.
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(st->mu);
    const auto made = [&] { return st->finished || g_window_made.load(); };
    while (!made() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(6)) {
        st->cv.wait_for(lk, std::chrono::milliseconds(50));
    }
    if (!made()) {
        host_log("window creation did not complete in 6 s (window manager did not map it); running headless");
        return false;
    }
    st->cv.wait(lk, [&] { return st->finished; });
    const auto s = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count();
    if (s >= 6) host_log("window: ready after %lld s (the GPU's start: device, pipeline cache, presenter)", static_cast<long long>(s));
    return st->result;
#else
    (void)width;
    (void)height;
    (void)title;
    return false;
#endif
}

#if defined(BBHOST_HAVE_SDL3)
namespace {
bool window_start_impl(int width, int height, const char* title) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        host_log("SDL_Init failed: %s", SDL_GetError());
        return false;
    }
    // The options are read before the window exists, so a saved Fullscreen
    // is only a wish until here (host_window_set_fullscreen).
    const bool fullscreen = g_want_fullscreen.load();
    g_window = SDL_CreateWindow(title, width, height,
                                SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | (fullscreen ? SDL_WINDOW_FULLSCREEN : 0));
    if (g_window && fullscreen) host_log("window: fullscreen");
    if (!g_window) {
        host_log("SDL_CreateWindow failed: %s", SDL_GetError());
        SDL_Quit();
        return false;
    }
    g_window_made.store(true);
    // BBHOST_NO_GAMEPAD=1: ignore controllers, so a test run on a virtual
    // display takes input only from its own script and not from a controller
    // plugged into the machine it runs on (pads_ignored).
    int n = 0;
    if (SDL_JoystickID* ids = pads_ignored() ? nullptr : SDL_GetGamepads(&n)) {
        for (int k = 0; k < n; ++k) pad_open(ids[k]);
        SDL_free(ids);
    }
    // A pad in at the start is what the player is holding until they touch
    // something else - the prompts start as its glyphs, as they always did.
    if (!g_pads.empty()) note_input(InputDevice::Pad);
    if (!vk_start()) {
        host_log("vulkan presenter unavailable; window without presentation");
    }
    g_active = true;
    host_log("window %dx%d", width, height);
    return true;
}
}  // namespace
#endif

MouseState host_mouse_state();

bool host_window_pump() {
#if defined(BBHOST_HAVE_SDL3)
    if (!g_active) {
        return true;
    }
    // Apply the pointer mode where the window lives, not from the pad thread.
    if (const bool want = g_mouse_relative.load(std::memory_order_relaxed);
        want != g_mouse_relative_applied.load(std::memory_order_relaxed)) {
        g_mouse_relative_applied.store(want, std::memory_order_relaxed);
        SDL_SetWindowRelativeMouseMode(g_window, want);
        // Motion from before the switch was the pointer's, not the camera's.
        mouse_camera::g_counts.store(0, std::memory_order_relaxed);
        if (!want) {
            // Leaving camera mode, the absolute position is whatever SDL
            // restored the pointer to; take it now so the first menu frame
            // does not hit-test against a stale one.
            float wx = 0.0f, wy = 0.0f;
            SDL_GetMouseState(&wx, &wy);
            std::lock_guard<std::mutex> lock(g_mouse_mu);
            mouse_moved_locked(wx, wy);
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_text_mu);
        if (g_text_want != g_text_on) {
            g_text_on = g_text_want;
            if (g_text_on) {
                SDL_StartTextInput(g_window);
                ime_off_for_ascii_locked();
            } else {
                SDL_StopTextInput(g_window);
            }
            text_title_locked();
        }
    }
    // The plugin menu (F9) takes typing while it is open.
    {
        static bool menu_text = false;
        const bool want = ingame_menu_open();
        std::lock_guard<std::mutex> lock(g_text_mu);
        if (want != menu_text && !g_text_on) {
            if (want) SDL_StartTextInput(g_window); else SDL_StopTextInput(g_window);
        }
        menu_text = want;
    }
    int win_w = 1, win_h = 1, px_w = 1, px_h = 1;
    SDL_GetWindowSize(g_window, &win_w, &win_h);
    SDL_GetWindowSizeInPixels(g_window, &px_w, &px_h);
    const float pixel_ratio = win_w > 0 ? static_cast<float>(px_w) / static_cast<float>(win_w) : 1.0f;
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        // F10's screen keeps its keys; the plugin menu takes the rest while open.
        if (!host_options_open() && ingame_menu_event(e, pixel_ratio)) continue;
        switch (e.type) {
            case SDL_EVENT_QUIT:
                return false;
            case SDL_EVENT_TEXT_INPUT: {
                // Not while F10's screen is up over the game's box, unless
                // one of the screen's own fields is being typed: it is modal.
                std::lock_guard<std::mutex> lock(g_text_mu);
                if (g_text_on && e.text.text && (!host_options_open() || host_options_editing())) {
                    g_entry.insert(e.text.text);
                    text_title_locked();
                }
                break;
            }
            case SDL_EVENT_KEY_DOWN: {
                if (!e.key.repeat) note_input(InputDevice::KeyboardMouse);
                if (!e.key.repeat) warn_deck_double_input();
                // A key binding being chosen takes the next key, whatever it
                // is, and the change is saved at once like any other row's.
                if (host_bind_capturing() >= 0) {
                    if (!e.key.repeat && host_bind_capture_key(e.key.scancode)) host_options_save_now();
                    break;
                }
                // Delete outside a capture clears the selected binding, when
                // the Key Bindings screen has one selected to take it.
                if (!e.key.repeat && e.key.scancode == SDL_SCANCODE_DELETE) {
                    host_bind_note_clear_key();
                }
                // F10 opens the port's options screen, and while it is open it
                // takes every key: it is modal, so nothing leaks to the pad.
                if (host_options_key(e.key.scancode, e.key.repeat)) {
                    break;
                }
                if (!e.key.repeat && e.key.scancode == SDL_SCANCODE_F11) {
                    g_ime_test_reopen.store(true, std::memory_order_relaxed);
                    break;
                }
                if (!e.key.repeat && e.key.scancode == SDL_SCANCODE_F12) {
                    host_gpu_request_dump();  // it logs the capture's folder
                    break;
                }
                std::lock_guard<std::mutex> lock(g_text_mu);
                if (g_text_on) {
                    const SDL_Keycode k = e.key.key;
                    if (k == SDLK_BACKSPACE) {
                        g_entry.backspace();
                        text_title_locked();
                    } else if (k == SDLK_DELETE) {
                        g_entry.erase_selection();
                        text_title_locked();
                    } else if (k == SDLK_RIGHT || k == SDLK_END) {
                        g_entry.keep();  // to the suggestion's end, to type after it
                    } else if ((k == SDLK_RETURN || k == SDLK_KP_ENTER || k == SDLK_ESCAPE) && !e.key.repeat) {
                        // A fresh press only. The Enter that opened the box is
                        // often still down when it appears, and held into the
                        // key's auto-repeat it accepted the box before a
                        // letter could be typed.
                        g_text_result = k == SDLK_ESCAPE ? 2 : 1;
                        g_text_end_key.store(e.key.scancode, std::memory_order_relaxed);
                    } else if ((k == SDLK_V && (e.key.mod & SDL_KMOD_CTRL)) ||
                               (k == SDLK_INSERT && (e.key.mod & SDL_KMOD_SHIFT))) {
                        // Paste: a chalice glyph copied from a web page or a
                        // chat, its surrounding spaces and line ends dropped.
                        if (char* clip = SDL_GetClipboardText()) {
                            std::string s = clip;
                            SDL_free(clip);
                            const std::size_t b = s.find_first_not_of(" \t\r\n");
                            const std::size_t e2 = s.find_last_not_of(" \t\r\n");
                            s = b == std::string::npos ? std::string() : s.substr(b, e2 - b + 1);
                            g_entry.insert(s.c_str());
                            text_title_locked();
                        }
                    }
                }
                break;
            }
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                // Not every driver reports a resized window as an out-of-date
                // swapchain (NVIDIA on X11 scales the old one into it), so a
                // resize - a tiling window manager's, its fullscreen toggle -
                // rebuilds it here rather than waiting to be told.
                g_swap_dirty.store(true);
                break;
            case SDL_EVENT_MOUSE_MOTION: {
                // A mouse that is touched, not one that twitches by a pixel.
                if (std::fabs(e.motion.xrel) + std::fabs(e.motion.yrel) >= 3.0f) note_input(InputDevice::KeyboardMouse);
                std::lock_guard<std::mutex> lock(g_mouse_mu);
                // Relative motion is accumulated whatever the mode: SDL
                // reports it either way, and a consumer that wants deltas
                // should not have to care which mode the window is in.
                g_mouse.dx += e.motion.xrel;
                g_mouse.dy += e.motion.yrel;
                // The camera's own, in raw counts, while the mouse turns it
                // (engine/mouse_camera_step.h): the camera takes them itself.
                if (g_mouse_relative_applied.load(std::memory_order_relaxed)) {
                    mouse_camera::add(mouse_camera::g_counts, e.motion.xrel, e.motion.yrel);
                }
                if (!g_mouse_relative.load(std::memory_order_relaxed)) {
                    mouse_moved_locked(e.motion.x, e.motion.y);
                } else {
                    g_mouse.moved = g_mouse.moved || e.motion.xrel != 0.0f || e.motion.yrel != 0.0f;
                }
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP: {
                const std::uint32_t bit = mouse_bit(e.button.button);
                if (!bit) break;
                if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                    note_input(InputDevice::KeyboardMouse);
                    warn_deck_double_input();
                }
                // The same for a mouse button, and the press is the capture's
                // alone: no click reaches the menu behind it.
                if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && host_bind_capturing() >= 0) {
                    const int n = bit == 1u ? 1 : bit == 2u ? 2 : bit == 4u ? 3 : bit == 8u ? 4 : 5;
                    if (host_bind_capture_mouse(n)) host_options_save_now();
                    break;
                }
                std::lock_guard<std::mutex> lock(g_mouse_mu);
                mouse_moved_locked(e.button.x, e.button.y);
                if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                    g_mouse.buttons |= bit;
                    g_mouse.pressed |= bit;
                } else {
                    g_mouse.buttons &= ~bit;
                    g_mouse.released |= bit;
                }
                break;
            }
            case SDL_EVENT_MOUSE_WHEEL: {
                note_input(InputDevice::KeyboardMouse);
                // Rebinding: the wheel is bindable too, up and down apart
                // (host/bindings.h), and the notch is the capture's alone.
                if (host_bind_capturing() >= 0 && e.wheel.y != 0.0f) {
                    // The same sign the wheel's presses read (g_mouse.wheel).
                    if (host_bind_capture_mouse(e.wheel.y > 0.0f ? 6 : 7)) host_options_save_now();
                    break;
                }
                std::lock_guard<std::mutex> lock(g_mouse_mu);
                g_mouse.wheel += e.wheel.y;
                g_mouse_last_motion = std::chrono::steady_clock::now();
                break;
            }
            case SDL_EVENT_WINDOW_MOUSE_LEAVE: {
                std::lock_guard<std::mutex> lock(g_mouse_mu);
                g_mouse.in_window = false;
                break;
            }
            // Keyboard focus decides whether SDL delivers input at all: no
            // keys without it, and the controller's events are dropped too
            // (SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS is off), so a game
            // that ignores every input may be a window that lost focus.
            case SDL_EVENT_WINDOW_FOCUS_GAINED:
            case SDL_EVENT_WINDOW_FOCUS_LOST: {
                host_log("window: keyboard focus %s", e.type == SDL_EVENT_WINDOW_FOCUS_GAINED ? "gained" : "lost");
                // SDL restarted text input with the focus, and the IME with it.
                std::lock_guard<std::mutex> lock(g_text_mu);
                if (e.type == SDL_EVENT_WINDOW_FOCUS_GAINED && g_text_on) ime_off_for_ascii_locked();
                break;
            }
            case SDL_EVENT_GAMEPAD_ADDED:
                pad_open(e.gdevice.which);
                break;
            case SDL_EVENT_GAMEPAD_BUTTON_DOWN: {
                note_input(InputDevice::Pad);
                g_last_pad_press = std::chrono::steady_clock::now();
                // A text box takes a controller's confirm as Enter and its
                // back as Escape (Options accepts too), the game's region
                // deciding which button is which: a player with only a
                // controller in hand is never left in a box that only a
                // keyboard can close. A press, so the button that opened the
                // box - still held when it appears - does not close it.
                std::lock_guard<std::mutex> lock(g_text_mu);
                if (!g_text_on || host_options_open()) break;
                const std::uint32_t bit = e.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH  ? kCross
                                          : e.gbutton.button == SDL_GAMEPAD_BUTTON_EAST ? kCircle
                                                                                         : 0u;
                if (e.gbutton.button == SDL_GAMEPAD_BUTTON_START || (bit && bit == menu_confirm_button())) {
                    g_text_result = 1;
                } else if (bit && bit == menu_back_button()) {
                    g_text_result = 2;
                } else {
                    break;
                }
                g_text_end_button.store(e.gbutton.button, std::memory_order_relaxed);
                break;
            }
            case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
                // Past the dead zone: half a stick's throw, a quarter of a
                // trigger's - a resting stick reads a few hundred either way.
                const bool trigger = e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER;
                if (std::abs(static_cast<int>(e.gaxis.value)) > (trigger ? 8000 : 16000)) note_input(InputDevice::Pad);
                break;
            }
            case SDL_EVENT_GAMEPAD_REMOVED:
                pad_close(e.gdevice.which);
                break;
            default:
                break;
        }
    }
    update_pad();
    // BBHOST_IME_TEST=1: open the text box at startup, so the overlay's font
    // and layout can be seen without reaching character creation. It opens as
    // the name box does (hle/dialog.cpp); =glyph and =password open it as the
    // other two of the game's boxes, which are deeper in than a test reaches.
    static const int ime_test = [] {
        const char* e = std::getenv("BBHOST_IME_TEST");
        return !e ? 0 : !std::strcmp(e, "1") ? 1 : !std::strcmp(e, "glyph") ? 2 : !std::strcmp(e, "password") ? 3 : 0;
    }();
    if (ime_test) {
        // It has to close the way a real dialog does. While text entry is
        // open, update_pad() stops the keyboard driving the pad - so a test box
        // that never ends takes the keyboard away for the rest of the run, and
        // the game looks frozen to every key. Enter and Escape end it, and F11
        // brings it back.
        static bool opened = false;
        if (!opened && g_ime_test_reopen.exchange(false)) {
            opened = true;
            if (ime_test == 1) host_text_entry_begin("Hunter", 16, false, "Please Enter Name", TextCharset::BasicLatin, true);
            if (ime_test == 2) host_text_entry_begin("", 8, false, "Enter Chalice Glyph", TextCharset::Glyph);
            if (ime_test == 3) host_text_entry_begin("", 8, false, "Enter password", TextCharset::BasicLatin);
        }
        if (opened) {
            std::string typed;
            if (host_text_entry_poll(typed) != 0) {
                host_text_entry_end();
                opened = false;
                host_log("ime-test: closed with \"%s\"; F11 opens it again", typed.c_str());
            }
        }
    }
    // The host overlay's contents for this frame. Rebuilt from scratch each
    // pump; when nothing draws, the presenter's path is exactly as it was.
    host_overlay_reset();
    // The FPS counter, top right, under everything else the overlay draws.
    // The number is refreshed twice a second: every frame, it flickers
    // between neighbours and cannot be read.
    if (host_settings().fps_counter) {
        static int shown = 0;
        static std::chrono::steady_clock::time_point shown_at;
        // Beside it, how busy our work kept the GPU (host/gpu_busy.cpp) over
        // the last two refreshes - about the same second the count covers.
        // Task Manager's GPU column cannot say it on AMD's driver.
        static int gpu_pct = -1;
        static GpuBusy busy_was[2];
        static std::chrono::steady_clock::time_point busy_at[2];
        const auto now = std::chrono::steady_clock::now();
        if (now - shown_at >= std::chrono::milliseconds(500)) {
            shown = presents_in_last_second(now);
            shown_at = now;
            const GpuBusy b = host_gpu_busy();
            if (b.on && busy_was[0].on) {
                const double secs = std::chrono::duration<double>(now - busy_at[0]).count();
                const double pct = secs > 0.0 ? static_cast<double>(b.busy_ns - busy_was[0].busy_ns) / 1e7 / secs : 0.0;
                gpu_pct = static_cast<int>(std::lround(std::clamp(pct, 0.0, 100.0)));
            }
            busy_was[0] = busy_was[1];
            busy_at[0] = busy_at[1];
            busy_was[1] = b;
            busy_at[1] = now;
            if (!busy_was[0].on) {  // the first refresh: the window starts here
                busy_was[0] = b;
                busy_at[0] = now;
            }
        }
        float dw = 0.0f;
        {
            std::lock_guard<std::mutex> ml(g_mouse_mu);
            dw = g_display_w ? static_cast<float>(g_display_w) : 1920.0f;
        }
        char text[48];
        if (gpu_pct >= 0) {
            std::snprintf(text, sizeof(text), "%d FPS  GPU %d%%", shown, gpu_pct);
        } else {
            std::snprintf(text, sizeof(text), "%d FPS", shown);
        }
        const float scale = 0.8f, w = host_overlay_text_width(scale, text);
        const float x = dw - w - 20.0f, y = 14.0f;
        host_overlay_rect(x - 8.0f, y - 4.0f, w + 16.0f, 24.0f * scale + 8.0f, 0x000000a0u);
        host_overlay_text(x, y, scale, 0xe6e6e6ffu, text);
    }
    {
        // A plugin's banner (host_message_show): centred in the upper third,
        // fading over its last half second.
        std::string text;
        float alpha = 0.0f;
        {
            std::lock_guard<std::mutex> bl(g_banner_mu);
            const auto now = std::chrono::steady_clock::now();
            if (!g_banner_text.empty() && now < g_banner_until) {
                text = g_banner_text;
                const float left = std::chrono::duration<float>(g_banner_until - now).count();
                alpha = left < 0.5f ? left / 0.5f : 1.0f;
            }
        }
        if (!text.empty()) {
            float dw = 0.0f, dh = 0.0f;
            {
                std::lock_guard<std::mutex> ml(g_mouse_mu);
                dw = g_display_w ? static_cast<float>(g_display_w) : 1920.0f;
                dh = g_display_h ? static_cast<float>(g_display_h) : 1080.0f;
            }
            const float scale = 1.25f, w = host_overlay_text_width(scale, text.c_str());
            const float x = (dw - w) * 0.5f, y = dh * 0.22f;
            const auto a = [&](std::uint32_t rgba) {
                return (rgba & 0xffffff00u) | static_cast<std::uint32_t>(static_cast<float>(rgba & 0xffu) * alpha);
            };
            host_overlay_rect(x - 24.0f, y - 12.0f, w + 48.0f, 24.0f * scale + 24.0f, a(0x000000b0u));
            host_overlay_text(x, y, scale, a(0xe8dcc0ffu), text.c_str());
        }
    }
    {
        // The first start after an update that changed the shader translator
        // (and the first ever) builds every shader again. On a few cores that
        // runs past the title into the first minute of play, which stutters
        // while it lasts (on the Steam Deck the start's 3,000 compiles take its
        // four workers 26 s): say so, top right, below the FPS counter. A
        // backlog gone within two seconds - most PCs' - never shows, and the
        // line does not start over in the short lulls between GX's bursts of
        // creations.
        static std::chrono::steady_clock::time_point busy_from{}, busy_last{};
        const std::size_t left = host_gpu_shader_backlog();
        const auto now = std::chrono::steady_clock::now();
        if (left >= 32) {
            if (busy_from == std::chrono::steady_clock::time_point{} || now - busy_last > std::chrono::seconds(3)) busy_from = now;
            busy_last = now;
        }
        if (left > 0 && busy_from != std::chrono::steady_clock::time_point{} && now - busy_from >= std::chrono::seconds(2) &&
            now - busy_last <= std::chrono::seconds(3)) {
            float dw = 0.0f;
            {
                std::lock_guard<std::mutex> ml(g_mouse_mu);
                dw = g_display_w ? static_cast<float>(g_display_w) : 1920.0f;
            }
            char text[64];
            std::snprintf(text, sizeof(text), "Building shaders: %zu left", left);
            const float scale = 0.8f, w = host_overlay_text_width(scale, text);
            const float x = dw - w - 20.0f, y = host_settings().fps_counter ? 14.0f + 24.0f * scale + 16.0f : 14.0f;
            host_overlay_rect(x - 8.0f, y - 4.0f, w + 16.0f, 24.0f * scale + 8.0f, 0x000000a0u);
            host_overlay_text(x, y, scale, 0xe8dcc0ffu, text);
        }
    }
    {
        // The text box that stands in for the PS4's on-screen keyboard. The
        // game never owns a text field - sceImeDialog is a system overlay, so
        // the system is the one that has to show what is being typed. The
        // window title used to do it, which was a placeholder.
        std::lock_guard<std::mutex> lock(g_text_mu);
        if (g_text_on) {
            float dw = 0.0f, dh = 0.0f;
            {
                std::lock_guard<std::mutex> ml(g_mouse_mu);
                dw = g_display_w ? static_cast<float>(g_display_w) : 1920.0f;
                dh = g_display_h ? static_cast<float>(g_display_h) : 1080.0f;
            }
            const float bw = dw * 0.5f, bh = 150.0f;
            const float bx = (dw - bw) * 0.5f, by = dh * 0.62f;
            host_overlay_rect(bx - 3.0f, by - 3.0f, bw + 6.0f, bh + 6.0f, 0xc8a05aff);  // the game's gold, as a border
            host_overlay_rect(bx, by, bw, bh, 0x0a0a0aee);
            host_overlay_text(bx + 20.0f, by + 16.0f, 0.8f, 0xc8a05aff, g_text_label.c_str());
            const float tx = bx + 20.0f, ty = by + 58.0f;
            const std::string shown = g_text_hidden ? std::string(g_entry.length(), '*') : g_entry.text;
            // A suggestion is drawn selected, the way a PC text field shows
            // text the first key will type over.
            if (g_entry.selected) {
                host_overlay_rect(tx - 3.0f, ty - 2.0f, host_overlay_text_width(1.4f, shown.c_str()) + 6.0f, 38.0f, 0xc8a05a80u);
            }
            const float w = host_overlay_text(tx, ty, 1.4f, 0xffffffffu, shown.c_str());
            // A caret that blinks, so an empty field still looks like one that
            // is waiting for typing rather than one that is broken.
            static std::uint64_t ticks = 0;
            if ((++ticks / 30) % 2 == 0) host_overlay_rect(tx + w + 2.0f, ty, 2.0f, 34.0f, 0xffffffffu);
            // With a controller in hand, the buttons that close the box; the
            // typing itself is the keyboard's, or Steam's on-screen one.
            const char* help = "Enter accepts    Esc cancels    Backspace deletes    Ctrl+V pastes";
            if (host_input_device() == InputDevice::Pad) {
                help = menu_confirm_button() == kCross ? "Cross / A accepts    Circle / B cancels    Type on a keyboard"
                                                       : "Circle / B accepts    Cross / A cancels    Type on a keyboard";
            }
            host_overlay_text(bx + 20.0f, by + bh - 30.0f, 0.62f, 0x9a9a9aff, help);
        }
    }
    // The rows the port adds to the game's own System menu write their bytes
    // as the player moves a cursor; this is where such a change becomes a
    // setting (engine/option_menu.h).
    option_menu_poll();
    // The port's own options screen (F10), under the pointer.
    {
        float dw = 0.0f, dh = 0.0f;
        {
            std::lock_guard<std::mutex> ml(g_mouse_mu);
            dw = g_display_w ? static_cast<float>(g_display_w) : 1920.0f;
            dh = g_display_h ? static_cast<float>(g_display_h) : 1080.0f;
        }
        host_options_frame(dw, dh);
    }
    // The pointer. The game has no cursor asset and needs none - this is the
    // host's own. It draws only once the pointer has been in the window, so a
    // run nobody touches is exactly as it was.
    {
        // Ours is the only pointer on screen: two of them, one lagging the
        // other by a frame, reads as a bug. The setting can change mid-run, so
        // the window system's follows it back.
        // Not in camera mode: there is nothing to point at, the position has
        // stopped moving, and a pointer frozen mid-screen while the camera
        // turns is the clearest way to look broken.
        //
        // And not once it has been idle: DS3's CSMouseMan (sub_140f135d0)
        // hides the pointer five seconds after the mouse last moved or
        // clicked (its +0x08, 5.0f) and shows it on the next movement, so a
        // player back on the pad has no arrow parked over the menu. The
        // port's own options screen always keeps it.
        std::chrono::steady_clock::time_point last;
        {
            std::lock_guard<std::mutex> ml(g_mouse_mu);
            last = g_mouse_last_motion;
        }
        // Or at once when a controller is the device in use: DS3 hides the
        // arrow the moment the pad moves and brings it back with the mouse.
        const bool idle = !host_options_open() && !ingame_menu_open() && (std::chrono::steady_clock::now() - last > std::chrono::seconds(5) ||
                                                   host_input_device() == InputDevice::Pad);
        const bool pointer = !host_mouse_relative() && !idle;
        const bool draw_cursor = host_settings().draw_cursor;
        const bool ours = draw_cursor && pointer && !ingame_menu_open();  // the menu draws its own on top
        // The window system's, when it is the one in use. Relative mode hides
        // it by itself; this keeps the state it returns to right.
        const bool system = !draw_cursor && pointer;
        static int shown = -1;
        if (shown != (system ? 1 : 0)) {
            shown = system ? 1 : 0;
            if (system) SDL_ShowCursor(); else SDL_HideCursor();
        }
        float mx = 0.0f, my = 0.0f;
        if (ours && host_mouse_position(mx, my)) host_overlay_cursor(mx, my);
    }
    // The frame's overlay, whole, to the presenter: a present never sees one
    // half rebuilt. Nothing on it (no counter, pointer, box or screen) and the
    // presenter's pass draws nothing over the picture.
    host_overlay_commit();
    // BBHOST_MOUSE_LOG=1: the pointer as the host sees it, for bringing it up
    // before there is a consumer. This **consumes** the edges, the wheel and
    // `moved` the way a real reader does - without that the first click stays
    // latched and re-logs on every pump - so it is a bring-up aid, not
    // something to leave on beside a consumer.
    static const bool mouse_log = [] {
        const char* e = std::getenv("BBHOST_MOUSE_LOG");
        return e && e[0] == '1';
    }();
    if (mouse_log) {
        const MouseState m = host_mouse_state();
        static int logged = 0;
        if ((m.moved || m.pressed || m.released || m.wheel != 0.0f) && logged < 40) {
            ++logged;
            host_log("mouse: %.1f,%.1f%s buttons %u pressed %u released %u wheel %+.0f", m.x, m.y,
                     m.in_window ? "" : " (outside)", m.buttons, m.pressed, m.released, m.wheel);
        }
    }
    return true;
#else
    return true;
#endif
}

void host_message_show(const char* text, float seconds) {
    std::lock_guard<std::mutex> bl(g_banner_mu);
    g_banner_text = text ? text : "";
    g_banner_until = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(static_cast<long long>((seconds > 0.0f ? seconds : 3.0f) * 1000.0f));
}

void host_text_entry_begin(const char* initial_utf8, unsigned max_chars, bool hidden, const char* label, TextCharset charset,
                           bool select_initial) {
    std::lock_guard<std::mutex> lock(g_text_mu);
    g_entry.begin(initial_utf8, max_chars, charset, select_initial);
    g_text_label = label && label[0] ? label : "Enter name";
    g_text_hidden = hidden;
    g_text_result = 0;
    g_text_want = true;
}

int host_text_entry_poll(std::string& text_utf8) {
    std::lock_guard<std::mutex> lock(g_text_mu);
    text_utf8 = g_entry.text;
    if (!g_active) return 2;
    return g_text_result;
}

void host_text_entry_end() {
    std::lock_guard<std::mutex> lock(g_text_mu);
    g_text_want = false;
    g_text_result = 0;
}

bool host_text_entry_open() {
    std::lock_guard<std::mutex> lock(g_text_mu);
    return g_text_want;
}

// Presentation runs on its own thread. vkAcquireNextImageKHR on a FIFO
// swapchain blocks until the display's next vblank, and host_present is called
// from the synthetic 60 Hz vblank thread that paces the game's flips: doing the
// present there stretched every tick that carried one, dragging the whole
// cadence down and making windowed runs - the movie most visibly - choppy while
// a headless run of the same build was a flat 30 fps. The pending slot holds
// only the newest frame; the game triple-buffers, so a dropped frame means the
// display could not keep up anyway.
namespace {
#if defined(BBHOST_HAVE_SDL3)
struct PendingPresent {
    bool has = false;
    int buffer = 0;
    std::uint64_t va = 0;
    unsigned w = 0, h = 0;
    PresentFlip flip;
};
std::mutex g_present_mu;
std::condition_variable g_present_cv;
PendingPresent g_present_next;
bool g_present_quit = false;
std::thread g_present_thread;

void present_thread_main() {
    host_thread_set_name("bb-present");
    host_thread_set_class(HostThreadClass::GpuFeed, "bb-present");  // core/host_clock.h
    for (;;) {
        PendingPresent p;
        {
            std::unique_lock<std::mutex> lk(g_present_mu);
            g_present_cv.wait(lk, [] { return g_present_quit || g_present_next.has; });
            if (g_present_quit) return;
            p = g_present_next;
            g_present_next.has = false;
        }
        vk_present(p.buffer, p.va, p.w, p.h, p.flip);
        // With V-Sync on this returned on the display's vblank: the flip clock
        // uses it to stay in phase with the display.
        g_present_last_ns.store(static_cast<std::uint64_t>(
                                    std::chrono::steady_clock::now().time_since_epoch().count()),
                                std::memory_order_relaxed);
        g_present_serial.fetch_add(1, std::memory_order_release);
    }
}
#endif
}  // namespace

// The present step the presenting thread is on and how long it has been there,
// for the hang watchdog (gpu.cpp).
const char* host_present_step(double& seconds) {
    const std::chrono::steady_clock::time_point t0{
        std::chrono::steady_clock::duration(g_present_step_ns.load(std::memory_order_relaxed))};
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return g_present_step.load(std::memory_order_relaxed);
}

void host_window_error_box(const char* title, const char* text) {
#if defined(BBHOST_HAVE_SDL3)
    if (!SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, title, text, nullptr)) host_log("window: the message box failed: %s", SDL_GetError());
#else
    (void)title;
    (void)text;
#endif
}

void host_window_stop() {
#if defined(BBHOST_HAVE_SDL3)
    if (!g_active) {
        return;
    }
    g_active = false;
    {
        std::lock_guard<std::mutex> lk(g_present_mu);
        g_present_quit = true;
        g_present_cv.notify_all();
    }
    if (g_present_thread.joinable()) g_present_thread.join();
    if (g_vk.ok) {
        std::lock_guard<std::mutex> lock(g_vk.mu);
        g_vk.ok = false;
        host_gpu_lock();
        host_gpu_queue_lock();
        vkDeviceWaitIdle(g_vk.device);
        host_gpu_queue_unlock();
        host_gpu_unlock();
    }
    {
        std::lock_guard<std::mutex> lock(g_pads_mu);
        for (SDL_Gamepad* g : g_pads) SDL_CloseGamepad(g);
        g_pads.clear();
    }
    // The guest's audio threads are still running: SDL_Quit would destroy the
    // streams under their writes.
    host_audio_shutdown();
    SDL_DestroyWindow(g_window);
    SDL_Quit();
#endif
}

bool host_window_active() { return g_active.load(); }

MouseState host_mouse_state() {
    std::lock_guard<std::mutex> lock(g_mouse_mu);
    MouseState m = g_mouse;
    // The edges are the caller's now; holding them a second time would report
    // one click twice.
    g_mouse.pressed = g_mouse.released = 0;
    g_mouse.wheel = 0.0f;
    g_mouse.moved = false;
    g_mouse.dx = g_mouse.dy = 0.0f;
    return m;
}

bool host_mouse_position(float& x, float& y) {
    std::lock_guard<std::mutex> lock(g_mouse_mu);
    x = g_mouse.x;
    y = g_mouse.y;
    return g_mouse.in_window;
}

void host_mouse_describe(char* out, std::size_t n) {
#if defined(BBHOST_HAVE_SDL3)
    std::lock_guard<std::mutex> lock(g_mouse_mu);
    std::snprintf(out, n, "window %.0f,%.0f of %dx%d, display %.0f,%.0f of %ux%u%s", g_mouse_win_x, g_mouse_win_y, g_mouse_win_w,
                  g_mouse_win_h, g_mouse.x, g_mouse.y, g_display_w, g_display_h, g_mouse.in_window ? "" : " (outside)");
#else
    std::snprintf(out, n, "no window");
#endif
}

bool host_mouse_stage_position(float& x, float& y) {
#if defined(BBHOST_HAVE_SDL3)
    float px = 0.0f, py = 0.0f;
    if (!host_mouse_position(px, py)) {
        return false;
    }
    unsigned bw = 0, bh = 0;
    {
        std::lock_guard<std::mutex> lock(g_mouse_mu);
        bw = g_display_w;
        bh = g_display_h;
    }
    if (!bw || !bh) {
        return false;
    }
    // The menu movies are authored on a 1920x1080 stage. host_mouse_position
    // already reports display-buffer pixels rather than window pixels, so
    // window size and fullscreen need no handling here; only the buffer's own
    // size does. Scaleform shows the stage whole and centred (show-all): in a
    // buffer wider than 16:9 - the 21:9 and 32:9 entries - it is the middle
    // 16:9 of the picture, the sides outside it, and in a taller one (16:10)
    // the band between the top and the bottom (engine/live_resolution.cpp).
    const float s = std::min(static_cast<float>(bw) / 1920.0f, static_cast<float>(bh) / 1080.0f);
    x = (px - (static_cast<float>(bw) - 1920.0f * s) * 0.5f) / s;
    y = (py - (static_cast<float>(bh) - 1080.0f * s) * 0.5f) / s;
    return true;
#else
    (void)x;
    (void)y;
    return false;
#endif
}

// Whether the strong-attack modifier (Shift by default) is held. DS3 PC uses
// it the same way: the mouse buttons are the light attacks and the modifier
// makes them the heavy ones, which is one key instead of four bindings.
bool host_key_strong() { return g_key_strong.load(std::memory_order_relaxed); }

void host_mouse_set_relative(bool on) { g_mouse_relative.store(on, std::memory_order_relaxed); }
bool host_mouse_relative() { return g_mouse_relative.load(std::memory_order_relaxed); }

void host_mouse_set_visible(bool visible) {
#if defined(BBHOST_HAVE_SDL3)
    g_mouse_visible.store(visible, std::memory_order_relaxed);
    // Hide the window system's pointer when we draw our own, so there are not
    // two of them.
    if (visible) SDL_HideCursor(); else SDL_ShowCursor();
#else
    (void)visible;
#endif
}

PadState host_pad_state() {
    std::lock_guard<std::mutex> lock(g_pad_mu);
    return g_pad;
}

InputDevice host_input_device() {
#if defined(BBHOST_HAVE_SDL3)
    return static_cast<InputDevice>(g_input_device.load(std::memory_order_relaxed));
#else
    return InputDevice::KeyboardMouse;
#endif
}
std::uint64_t host_input_serial() {
#if defined(BBHOST_HAVE_SDL3)
    return g_input_serial.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}

bool host_pad_present() {
#if defined(BBHOST_HAVE_SDL3)
    std::lock_guard<std::mutex> lock(g_pads_mu);
    return !g_pads.empty();
#else
    return false;
#endif
}

void host_pad_rumble(std::uint8_t small, std::uint8_t large) {
#if defined(BBHOST_HAVE_SDL3)
    std::lock_guard<std::mutex> lock(g_pads_mu);
    for (SDL_Gamepad* g : g_pads) {
        SDL_RumbleGamepad(g, static_cast<Uint16>(small) * 257, static_cast<Uint16>(large) * 257, 100);
    }
#else
    (void)small;
    (void)large;
#endif
}

void host_present(int buffer_index, std::uint64_t display_va, unsigned display_w, unsigned display_h, const PresentFlip& flip) {
#if defined(BBHOST_HAVE_SDL3)
    if (g_active) {
        std::lock_guard<std::mutex> lk(g_present_mu);
        if (!g_present_thread.joinable()) g_present_thread = std::thread(present_thread_main);
        if (g_present_next.has && g_present_dropped.fetch_add(1) % 300 == 0) {
            host_log("present: display behind the game; dropped %llu frames so far",
                     static_cast<unsigned long long>(g_present_dropped.load()));
        }
        g_present_next = PendingPresent{true, buffer_index, display_va, display_w, display_h, flip};
        g_present_cv.notify_one();
        {
            // What the pointer's window position scales by.
            std::lock_guard<std::mutex> ml(g_mouse_mu);
            g_display_w = display_w;
            g_display_h = display_h;
        }
    }
#else
    (void)buffer_index;
    (void)display_va;
    (void)display_w;
    (void)display_h;
    (void)flip;
#endif
}

bool host_window_pixels(int* w, int* h) {
    const std::uint64_t v = g_window_pixels.load(std::memory_order_relaxed);
    if (!v) return false;
    *w = static_cast<int>(v >> 32);
    *h = static_cast<int>(v & 0xffffffffu);
    return true;
}

void host_window_request_quit() {
#if defined(BBHOST_HAVE_SDL3)
    // The event closing the window sends: the pump returns false and the
    // run ends the way it does then.
    SDL_Event e{};
    e.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&e);
#endif
}
