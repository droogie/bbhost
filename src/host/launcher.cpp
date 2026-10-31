#define IMGUI_DEFINE_MATH_OPERATORS
#include "host/launcher.h"

#include "bbhost_version.h"
#include "core/sha256.h"
#include "engine/addr.h"
#include "engine/patch_manifest.h"
#include "host/launcher_ui.h"
#include "host/options.h"
#include "net/account.h"
#include "host/plugin_ui.h"
#include "host/shader_compile.h"
#include "host/updater.h"
#include "log.h"

#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_sdlrenderer3.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace ui = launcher_ui;

// The name the sidebar shows.
constexpr const char kName[] = "BBHOST";

// The online servers the window offers; anything else is "Custom".
struct Server {
    const char* label;
    const char* host;
    const char* scheme;
    bool verify_tls, require_account;
    const char* auth_server;  // where /auth/* goes ("": the NP server's origin)
};
// Both take only https and signed-in players; the playtest server as the
// playtest kit's bbhost-playtest.toml has it.
constexpr Server kServers[] = {
    {"Live server (thehuntersdream.com)", "thehuntersdream.com", "https", true, true, "https://thehuntersdream.com"},
    {"Playtest server (dev.thehuntersdream.com)", "dev.thehuntersdream.com", "https", true, true, "https://dev.thehuntersdream.com"},
};

// Menu text size's values (menu_text_size in host/options.cpp, the same list).
constexpr int kTextSizes[] = {100, 125, 150, 175, 200, 250, 300, 350, 400};
const char* const kTextSizeLabels[] = {"100%", "125%", "150%", "175%", "200%", "250%", "300%", "350%", "400%"};

// The window's pages, as the sidebar lists them.
enum Page { kPageGame, kPageOnline, kPageEnhancements, kPagePatches, kPageShaders, kPagePlugins, kPageSettings, kPageCount };
struct PageInfo {
    const char* name;
    const char* slug;  // its picture's name (BBHOST_SETUP_TEST=shot)
    launcher_ui::Icon icon;
};
constexpr PageInfo kPages[kPageCount] = {
    {"Game", "game", launcher_ui::Icon::Game},
    {"Online", "online", launcher_ui::Icon::Online},
    {"PC Enhancements", "pc-enhancements", launcher_ui::Icon::Enhance},
    {"Patches", "patches", launcher_ui::Icon::Patches},
    {"Shaders", "shaders", launcher_ui::Icon::Shaders},
    {"Plugins", "plugins", launcher_ui::Icon::Plugins},
    {"Settings", "settings", launcher_ui::Icon::Settings},
};

// A text field's buffer.
struct Field {
    char text[1024] = {};
    void set(const std::string& s) { std::snprintf(text, sizeof(text), "%s", s.c_str()); }
    std::string str() const { return text; }
};

// What the system's pickers returned: they may answer on another thread.
std::mutex g_pick_mu;
std::string g_picked;
int g_pick_for = -1;  // which field the answer is for (-1 none pending)

void SDLCALL on_pick(void* userdata, const char* const* files, int) {
    std::lock_guard<std::mutex> lk(g_pick_mu);
    g_pick_for = static_cast<int>(reinterpret_cast<std::intptr_t>(userdata));
    g_picked = files && files[0] ? files[0] : "";
}

bool is_dir(const std::string& p) {
    std::error_code ec;
    return !p.empty() && fs::is_directory(p, ec);
}
// A name config_set_values can write as a key as it is (TOML's bare keys).
bool bare_key(const std::string& k) {
    if (k.empty()) return false;
    for (char ch : k) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-')) return false;
    }
    return true;
}

bool is_file(const std::string& p) {
    std::error_code ec;
    return !p.empty() && fs::is_regular_file(p, ec);
}

std::string generic(std::string p) {
    for (char& ch : p) {
        if (ch == '\\') ch = '/';
    }
    return p;
}

// A path as the config keeps it: absolute. The window checks a relative one
// against the working directory (an eboot found beside bbhost), but the config
// reads relative paths against its own folder, where it is not.
std::string saved_path(const std::string& p) {
    std::error_code ec;
    const std::filesystem::path a = std::filesystem::absolute(p, ec);
    return generic(ec ? p : a.lexically_normal().string());
}

// What the window knows about the eboot it was last given.
struct EbootCheck {
    std::string path, sha;
    bool ok = false;
};

EbootCheck check_eboot(const std::string& path) {
    EbootCheck c;
    c.path = path;
    if (!is_file(path)) return c;
    std::ifstream in(path, std::ios::binary);
    std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    c.sha = sha256_hex(data.data(), data.size());
    c.ok = c.sha == kEboot109Sha256;
    return c;
}

std::string toml_string(const std::string& s) {
    std::string out = "\"";
    for (char ch : s) {
        if (ch == '"') out += "\\\"";
        else out += ch;
    }
    return out + "\"";
}

std::string capitalised(std::string s) {
    if (!s.empty()) s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    return s;
}

bool contains_nocase(const std::string& hay, const char* needle) {
    if (!*needle) return true;
    auto low = [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); };
    std::string h, n;
    for (char ch : hay) h += low(static_cast<unsigned char>(ch));
    for (const char* p = needle; *p; ++p) n += low(static_cast<unsigned char>(*p));
    return h.find(n) != std::string::npos;
}

// The title bar in the window's colours (Windows): dark, and on Windows 11
// the night's ink with ice letters and a violet edge - turned by Menu hue
// and Menu vibrance (launcher_ui::tint) as the rest of the window is. dwmapi's
// DwmSetWindowAttribute, loaded as SDL loads anything, so nothing more is
// linked; a Windows without the attribute leaves its bar as it was.
void dark_title_bar(SDL_Window* win) {
#if defined(_WIN32)
    using SetAttribute = long (*)(void* hwnd, unsigned attribute, const void* value, unsigned size);
    SDL_SharedObject* dwm = SDL_LoadObject("dwmapi.dll");
    const auto set = dwm ? reinterpret_cast<SetAttribute>(SDL_LoadFunction(dwm, "DwmSetWindowAttribute")) : nullptr;
    void* hwnd = SDL_GetPointerProperty(SDL_GetWindowProperties(win), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    if (set && hwnd) {
        const int dark = 1;
        // COLORREFs, 0x00BBGGRR: an ImU32's byte order without its alpha.
        const auto colour = [](unsigned rgb) { return launcher_ui::tint(0xff000000u | rgb) & 0x00ffffffu; };
        const unsigned caption = colour(0x00240C10), text = colour(0x00FFD8BF), edge = colour(0x00FF4D7C);
        set(hwnd, 20, &dark, sizeof(dark));           // DWMWA_USE_IMMERSIVE_DARK_MODE
        set(hwnd, 35, &caption, sizeof(caption));     // DWMWA_CAPTION_COLOR
        set(hwnd, 36, &text, sizeof(text));           // DWMWA_TEXT_COLOR
        set(hwnd, 34, &edge, sizeof(edge));           // DWMWA_BORDER_COLOR
    }
    if (dwm) SDL_UnloadObject(dwm);
#else
    (void)win;
#endif
}

// Dim text, wrapped, in a smaller size when `small`.
void dim_text(const std::string& s, ImU32 col = launcher_ui::kTextDim, bool small = true) {
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    if (small) ImGui::PushFont(nullptr, 14.0f);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(s.c_str());
    ImGui::PopTextWrapPos();
    if (small) ImGui::PopFont();
    ImGui::PopStyleColor();
}

}  // namespace

LauncherResult launcher_run(const HostConfig& cfg, const std::string& reason, bool plugins_tab) {
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
        host_log("setup: no window (%s); edit %s instead", SDL_GetError(), config_user_file().c_str());
        return LauncherResult::Quit;
    }
    // The text size, Menu text size (menu_text_size, the same setting as
    // F10's): 100% is the window as it has always been, 1120x720 at Windows'
    // display scaling, and every size in it - the window's own included -
    // grows with the percent (a 1440p or 4K screen: 150% or 200%). A window
    // larger than the screen stops at the screen; the pages then scroll over
    // Play, Save and Quit, which stay in view.
    host_options_load();
    const SDL_DisplayID display = SDL_GetPrimaryDisplay();
    const float screen_scale = std::max(1.0f, SDL_GetDisplayContentScale(display));
    // Its size, or less where the screen is smaller.
    SDL_Rect usable{};
    const bool have_usable = SDL_GetDisplayUsableBounds(display, &usable) && usable.w > 0 && usable.h > 0;
    const auto scale_for = [&](int choice) {
        const int i = std::clamp(choice, 0, static_cast<int>(std::size(kTextSizes)) - 1);
        return screen_scale * static_cast<float>(kTextSizes[i]) / 100.0f;
    };
    int text_choice = std::max(0, host_opt_index("menu_text_size"));
    float scale = scale_for(text_choice);
    const auto window_size = [&](float s, int* w, int* h) {
        *w = static_cast<int>(1120 * s);
        *h = static_cast<int>(720 * s);
        if (have_usable) {
            *w = std::min(*w, usable.w * 19 / 20);
            *h = std::min(*h, usable.h * 9 / 10);
        }
    };
    int win_w = 0, win_h = 0;
    window_size(scale, &win_w, &win_h);
    host_log("setup: menu text size %s (x%.2f; the screen's own x%.2f)", kTextSizeLabels[text_choice], static_cast<double>(scale),
             static_cast<double>(screen_scale));
    SDL_Window* win = SDL_CreateWindow("bbhost launcher", win_w, win_h, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    SDL_Renderer* ren = win ? SDL_CreateRenderer(win, nullptr) : nullptr;
    if (!win || !ren) {
        host_log("setup: no window (%s); edit %s instead", SDL_GetError(), config_user_file().c_str());
        if (win) SDL_DestroyWindow(win);
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
        return LauncherResult::Quit;
    }
    SDL_SetWindowMinimumSize(win, std::min(win_w, static_cast<int>(860 * scale)), std::min(win_h, static_cast<int>(580 * scale)));
    dark_title_bar(win);
    SDL_SetRenderVSync(ren, 1);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // nothing written beside the exe
    ImGui_ImplSDL3_InitForSDLRenderer(win, ren);
    ImGui_ImplSDLRenderer3_Init(ren);
    // The look (host/launcher_ui.h): the fonts, the style, the painting.
    ui::init(ren, scale);

    // The fields, from what the configuration holds now.
    enum { kApp0, kEboot, kData, kToken, kHost, kFields };
    Field f[kFields];
    f[kApp0].set(cfg.app0);
    f[kEboot].set(cfg.eboot);
    f[kData].set(cfg.data == config_default_data_dir() ? std::string() : cfg.data);
    f[kToken].set(cfg.update_token_file);
    f[kHost].set(cfg.online_host);
    int server = cfg.online_host.empty() ? 0 : 2;  // nothing set yet: the live server; else custom unless it is one of these
    for (int i = 0; i < 2; ++i) {
        if (cfg.online_host == kServers[i].host) server = i;
    }
    char scheme[8] = {};
    std::snprintf(scheme, sizeof(scheme), "%s", cfg.online_scheme.empty() ? "http" : cfg.online_scheme.c_str());
    bool verify_tls = cfg.online_verify_tls, require_account = cfg.online_require_account;
#if defined(BBHOST_RELEASE_BUILD)
    bool check_updates = cfg.update_check != 0;
#else
    bool check_updates = cfg.update_check == 1;
#endif
    bool show_every_start = cfg.setup_always;
    bool animations = cfg.setup_animations;
    int text_applied = text_choice;
    float title_hue = 0.0f, title_vibrance = 0.0f;  // the title bar's colours as last set (dark_title_bar)  // the size the window has; a new choice goes in before the next frame
    bool tooling = cfg.tooling;
    // The F10 screen's start-up settings, through the same calls it uses.
    host_options_load();
    bool fullscreen = host_opt_index("window_mode") == 1;
    bool skip_logos = host_opt_get("skip_logos");
    bool performance_mode = host_opt_get("performance_mode");
    bool debug_camera = host_opt_get("debug_camera");
    int frame_cap = host_opt_frame_cap();
    int resolution = host_opt_resolution_index();
    // In the setting's order (host/options.cpp): by size within each shape.
    static const char* const kResolutions[] = {"640x360",           "960x540",           "1280x720",          "1440x810",
                                               "1600x900",          "1920x1080",         "2560x1440",         "2880x1620",
                                               "3200x1800",         "3840x2160",         "960x600 (16:10, upscaled)",
                                               "1024x640 (16:10, upscaled)",             "1280x800 (16:10)",  "1920x1200 (16:10)",
                                               "2560x1600 (16:10)", "1280x960 (4:3)",    "1440x1080 (4:3)",   "2560x1080 (21:9)",
                                               "3440x1440 (21:9)",  "5120x2160 (21:9)",  "3840x1080 (32:9)",  "5120x1440 (32:9)"};
    // The game's frame rate (engine/frame_rate.h): 0 is Display-Limited, the display's refresh rate.
    static const int kCaps[] = {30, 60, 90, 120, 150, 180, 240, 360, 480, 960, 0};
    constexpr int kCapCount = static_cast<int>(sizeof(kCaps) / sizeof(kCaps[0]));
    static const char* const kCapLabels[] = {"30 fps",  "60 fps",  "90 fps",  "120 fps", "150 fps", "180 fps",
                                             "240 fps", "360 fps", "480 fps", "960 fps", "Display-Limited"};
    static_assert(sizeof(kCapLabels) / sizeof(kCapLabels[0]) == sizeof(kCaps) / sizeof(kCaps[0]), "a label for every rate");
    // The bubble the open list shows over a rate: Display-Limited says what it is.
    static const char* const kCapTips[] = {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        "Caps the frame rate at your monitor's refresh rate, read when the game starts: 180 fps on a 180 Hz monitor, 144 on a "
        "144 Hz one."};
    static_assert(sizeof(kCapTips) / sizeof(kCapTips[0]) == sizeof(kCaps) / sizeof(kCaps[0]), "a bubble, or none, for every rate");
    int cap_choice = kCapCount - 1;
    for (int k = 0; k < kCapCount; ++k) {
        if (kCaps[k] == frame_cap) cap_choice = k;
    }
    // The follow camera's tuning above 30 fps: the setting's index, 0 FPS++, 1 Original.
    int camera_tuning = host_opt_index("camera_tuning") == 1 ? 1 : 0;
    // The Shaders page: the Shaders setting (0 HLSL, 1 Lifted), the caches'
    // state and "Compile shaders now" (host/shader_compile.h). This machine's
    // Vulkan devices are listed once, off the window's thread.
    int shaders = host_opt_index("shaders") == 1 ? 1 : 0;
    ShaderCompileJob compile_job;
    std::mutex devices_mu;
    std::vector<std::string> devices;
    std::atomic<bool> devices_ready{false};
    std::thread devices_thread([&] {
        std::vector<std::string> d = shader_vulkan_devices();
        std::lock_guard<std::mutex> lk(devices_mu);
        devices = std::move(d);
        devices_ready.store(true);
    });
    ShaderCacheState cache_state;
    std::chrono::steady_clock::time_point cache_state_at{};
    bool cache_state_devices = false, compile_was_running = false;
    // The PC enhancements, through the same calls: each is read when the
    // game starts (main.cpp, config_set_enhancements). The same four the
    // game's System > PC Enhancements and F10 > PC ENHANCEMENTS have.
    struct Enhancement {
        const char* key;
        const char* label;
        const char* sub;  // the line under it
        const char* text;  // the whole of it, in its bubble
        bool on;
    };
    Enhancement enhancements[] = {
        {"change_appearance", "Hunter's Dream mirror", "The workshop's mirror opens the appearance editor.",
         "The workshop's mirror opens the appearance editor: the developers' unused \"Put on Disguise\".", false},
        {"rebirth", "Rebirth at the Altar of Despair", "Back to your origin's level, your blood echoes to spend again.",
         "With the Yharnam Stone, the altar returns the hunter to their origin's level and gives the blood echoes back "
         "to spend again; leaving asks to accept or undo.",
         false},
        {"five_players", "Five players in the late areas", "Two invaders in the late and Old Hunters areas.",
         "The Nightmare of Mensis, Mergo's Loft, the Nightmare Frontier and the Old Hunters' areas take two invaders, as "
         "the chalice dungeons do. Every player in a session needs it on.",
         false},
        {"skill_issue", "SKILL ISSUE", "The death message says SKILL ISSUE.",
         "When the hunter dies, the screen says SKILL ISSUE instead of YOU DIED, in every language.", false},
    };
    for (Enhancement& e : enhancements) e.on = host_opt_get(e.key);
    // The Patches page: every manifest the loader would read
    // (engine/patch_manifest.h), each with its switch. One that follows a
    // setting (its `option`) shows that setting - the Game page's Skip company
    // logos, the debug camera - so a patch has one switch; the rest (The Old
    // Hunters among them) are [patches] <name> = true or false in bbhost.toml.
    bool quick_reentry = cfg.quick_reentry, all_post_processors = cfg.all_post_processors;
    bool quick_reentry_was = quick_reentry, all_post_processors_was = all_post_processors;
    struct PatchRow {
        PatchListing p;
        bool on = true, was = true;  // its [patches] switch, and what the file says
    };
    std::vector<PatchRow> patch_rows;
    for (PatchListing& p : patch_manifests_list(cfg.mods.empty() ? cfg.data + "/mods" : cfg.mods)) {
        PatchRow r;
        r.on = r.was = patch_manifest_on(p.name, p.enabled);
        r.p = std::move(p);
        patch_rows.push_back(std::move(r));
    }
    // Two of a group on (a bbhost.toml edited by hand): the one the loader
    // applies, the last in the list's order, stays on.
    for (PatchRow& r : patch_rows) {
        if (r.p.group.empty() || !r.on) continue;
        for (const PatchRow& o : patch_rows) {
            if (&o == &r || o.p.group != r.p.group || !o.on || !o.p.option.empty()) continue;
            if (std::make_pair(o.p.order, o.p.name) > std::make_pair(r.p.order, r.p.name)) r.on = false;
        }
    }
    auto patch_switch = [&](PatchRow& r) -> bool* {
        const std::string& o = r.p.option;
        if (o == "startup.skip_intro") return &skip_logos;
        if (o == "startup.performance_mode") return &performance_mode;
        if (o == "debug.free_camera") return &debug_camera;
        if (o == "loading.quick_reentry") return &quick_reentry;
        if (o == "streaming.all_post_processors") return &all_post_processors;
        return &r.on;
    };
    char patch_find[64] = {};

    EbootCheck eboot = check_eboot(f[kEboot].str());
    // The game folder's version, read again when the folder changes.
    std::string app0_seen;
    App0Version app0_ver;
    PluginUi plugin_ui;
    bool plugin_seen = false;
    std::string saved_note;
    bool saved_ok = true;
    double saved_at = -100.0;
    LauncherResult result = LauncherResult::Quit;
    bool done = false;
    int page = plugins_tab ? kPagePlugins : kPageGame;
    double page_at = 0.0;  // when it was chosen: it fades in
    // BBHOST_SETUP_TEST=play|quit: the window answers itself after a few
    // frames (a test that it opens, draws and hands over). create:NAME makes
    // an account held by this PC through the account section, as its button
    // does, then closes once the server has answered.
    const char* test = std::getenv("BBHOST_SETUP_TEST");
    int frames = 0;
    // link: starts a Discord link as its button does and, once the code is
    // up, cancels it and closes.
    const std::string test_create = test && !std::strncmp(test, "create:", 7) ? test + 7 : "";
    const bool test_link = test && !std::strcmp(test, "link");
    bool test_created = false;
    // shot:DIR saves each page as DIR/NN-<page>.png, then the Patches page
    // again with a patch's bubble up (the pointer put on a row), and closes,
    // saving nothing: a look at the window without a person, and the docs'
    // pictures.
    const std::string test_shot = test && !std::strncmp(test, "shot:", 5) ? test + 5 : "";
    int shot = 0;  // the picture being waited for: a page, then the bubble
    double shot_at = 1.2;
    ImVec2 patch_row_at(-FLT_MAX, -FLT_MAX);  // a row's middle, for the bubble
    if (!test_shot.empty()) page = kPageGame;

    // The account section's fields (never saved: the name is the account's,
    // the recovery code is typed once to recover).
    char acct_name[64] = {};
    char acct_code[64] = {};
    std::snprintf(acct_name, sizeof(acct_name), "%s", net::account_name().c_str());
    // The server the dropdown names, in memory: the account calls go there
    // whether or not it has been saved yet.
    auto apply_server = [&]() {
        const bool custom = server == 2;
        config_set_online(custom ? f[kHost].str() : kServers[server].host, custom ? scheme : kServers[server].scheme,
                          custom ? verify_tls : kServers[server].verify_tls, custom ? require_account : kServers[server].require_account,
                          custom ? cfg.auth_server : kServers[server].auth_server);
    };
    auto save = [&]() -> bool {
        std::vector<ConfigEdit> e;
        e.push_back({"paths", "app0", toml_string(saved_path(f[kApp0].str()))});
        e.push_back({"paths", "eboot", toml_string(saved_path(f[kEboot].str()))});
        if (!f[kData].str().empty()) e.push_back({"paths", "data", toml_string(saved_path(f[kData].str()))});
        const bool custom = server == 2;
        e.push_back({"online", "host", toml_string(custom ? f[kHost].str() : kServers[server].host)});
        e.push_back({"online", "scheme", toml_string(custom ? scheme : kServers[server].scheme)});
        e.push_back({"online", "verify_tls", (custom ? verify_tls : kServers[server].verify_tls) ? "true" : "false"});
        e.push_back({"online", "require_account", (custom ? require_account : kServers[server].require_account) ? "true" : "false"});
        if (!custom) e.push_back({"online", "auth_server", toml_string(kServers[server].auth_server)});
        e.push_back({"update", "check", check_updates ? "true" : "false"});
        if (!f[kToken].str().empty()) e.push_back({"update", "token_file", toml_string(saved_path(f[kToken].str()))});
        e.push_back({"startup", "setup_window", show_every_start ? "true" : "false"});
        e.push_back({"startup", "setup_animations", animations ? "true" : "false"});
        e.push_back({"debug", "tooling", tooling ? "true" : "false"});
        // What this bbhost's file looks like (core/config.h): a value saved
        // from here is a choice.
        e.push_back({"bbhost", "config_version", std::to_string(kUserConfigVersion)});
        // The Patches page's own switches, each only once changed.
        for (const PatchRow& r : patch_rows) {
            if (r.p.option.empty() && r.on != r.was && bare_key(r.p.name)) e.push_back({"patches", r.p.name, r.on ? "true" : "false"});
        }
        if (quick_reentry != quick_reentry_was) e.push_back({"loading", "quick_reentry", quick_reentry ? "true" : "false"});
        if (all_post_processors != all_post_processors_was)
            e.push_back({"streaming", "all_post_processors", all_post_processors ? "true" : "false"});
        const std::string file = config_user_file();
        saved_at = ImGui::GetTime();
        if (!config_set_values(file, e)) {
            saved_note = "Could not write " + file;
            saved_ok = false;
            return false;
        }
        for (PatchRow& r : patch_rows) r.was = r.on;
        quick_reentry_was = quick_reentry;
        all_post_processors_was = all_post_processors;
        host_opt_set_index("window_mode", fullscreen ? 1 : 0);
        host_opt_set("skip_logos", skip_logos);
        host_opt_set("performance_mode", performance_mode);
        host_opt_set("debug_camera", debug_camera);
        host_opt_set_frame_cap(kCaps[cap_choice]);
        host_opt_set_index("camera_tuning", camera_tuning);
        host_opt_set_index("shaders", shaders);
        host_opt_set_resolution_index(resolution);
        for (const Enhancement& x : enhancements) host_opt_set(x.key, x.on);
        host_options_save_now();
        saved_note = "Saved";
        saved_ok = true;
        host_log("setup: saved %s", file.c_str());
        return true;
    };

    while (!done) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL3_ProcessEvent(&ev);
            if (ev.type == SDL_EVENT_QUIT || ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) done = true;
        }
        // A sign-in or sign-out in the account section: the token is kept in
        // the options file, written at once rather than at Save.
        if (host_account_take_save()) {
            host_options_save_now();
            if (net::account_logged_in()) std::snprintf(acct_name, sizeof(acct_name), "%s", net::account_name().c_str());
        }
        {
            std::lock_guard<std::mutex> lk(g_pick_mu);
            if (g_pick_for >= 0) {
                if (!g_picked.empty()) f[g_pick_for].set(generic(g_picked));
                if (g_pick_for == kEboot) eboot = check_eboot(f[kEboot].str());
                g_pick_for = -1;
            }
        }
        // A new text size: the style and fonts take it, and the window
        // grows or shrinks to what it opens at with that size.
        if (text_choice != text_applied) {
            text_applied = text_choice;
            host_opt_set_index("menu_text_size", text_choice);  // saved at once, as F10 saves it
            scale = scale_for(text_choice);
            ui::set_scale(scale);
            window_size(scale, &win_w, &win_h);
            SDL_SetWindowMinimumSize(win, std::min(win_w, static_cast<int>(860 * scale)), std::min(win_h, static_cast<int>(580 * scale)));
            if (!(SDL_GetWindowFlags(win) & (SDL_WINDOW_MAXIMIZED | SDL_WINDOW_FULLSCREEN))) SDL_SetWindowSize(win, win_w, win_h);
            host_log("setup: menu text size %s (x%.2f)", kTextSizeLabels[text_choice], static_cast<double>(scale));
        }
        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        // A test's pictures: the pointer where the test puts it, not where the
        // desk's is.
        if (!test_shot.empty()) io.AddMousePosEvent(shot == kPageCount ? patch_row_at.x : -FLT_MAX, shot == kPageCount ? patch_row_at.y : -FLT_MAX);
        ImGui::NewFrame();
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const ImGuiStyle& st = ImGui::GetStyle();
        // At most a third of the window: a large Menu text size leaves the page room.
        const float side_w = std::round(std::min(236 * scale, vp->WorkSize.x * 0.34f));
        ui::set_animations(animations);
        ui::backdrop(side_w);
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("setup", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground |
                         ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::PopStyleVar();

        // While the Updates switch is on: one check, on a thread of its own
        // (host/updater.h); a newer release's page opens in the browser.
        if (check_updates) updater::check_once();
        const std::string app0 = f[kApp0].str();
        const bool app0_ok = is_dir(app0 + "/dvdroot_ps4");
        // Its version: the 1.09 update's files copied over it, or in an
        // update folder beside it (core/config.h), or the game stops when it
        // reads one. A warning, not a stop: Play stays on.
        if (app0_ok && app0_seen != app0) {
            app0_seen = app0;
            app0_ver = config_app0_version(app0);
        }
        // The compile's progress and the caches' state, which the Shaders page
        // shows and the Game page's recommendation reads: the state again
        // every few seconds, and when a compile ends.
        const ShaderCompileJob::View cv = compile_job.view();
        {
            const auto now = std::chrono::steady_clock::now();
            if (now - cache_state_at > std::chrono::seconds(3) || (compile_was_running && !cv.running) ||
                (devices_ready.load() && !cache_state_devices)) {
                std::lock_guard<std::mutex> lk(devices_mu);
                cache_state = shader_cache_state(f[kData].str(), devices);
                cache_state_at = now;
                cache_state_devices = devices_ready.load();
            }
            compile_was_running = cv.running;
        }
        // Caches that are not this bbhost's or this driver's: Compile shaders
        // now is recommended (never started by itself) until a compile has run
        // in this window.
        const ShaderCompileAdvice advice = shader_compile_advice(cache_state, host_steam_deck(), std::thread::hardware_concurrency());
        const bool recommend_compile = advice.recommend && !cv.running && !cv.finished && !cv.cancelled && cv.error.empty();
        auto start_compile = [&] {
            // The child reads the saved settings: save first.
            if (save()) {
                std::string err;
                compile_job.start(err);
            }
        };
        auto go = [&](int p) {
            if (page == p) return;
            page = p;
            page_at = ImGui::GetTime();
        };
        const bool compiling = compile_job.running();
        const bool can_play = app0_ok && eboot.ok && !compiling;
        bool play = false;

        // The sidebar: the name, the pages, and Play.
        {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 o = vp->WorkPos;
            ui::glass(dl, o, o + ImVec2(side_w, vp->WorkSize.y), 0.0f, IM_COL32(10, 8, 26, 160), 0);
            dl->AddLine(o + ImVec2(side_w - 0.5f, 0), o + ImVec2(side_w - 0.5f, vp->WorkSize.y), ui::alpha(ui::kViolet, 64));
            // Play, Save and Quit at the foot, always in view; over them the
            // name and the pages, which scroll when they are taller than the
            // room left (a large Menu text size), and the keys of the menus in
            // the game (host/ingame_menu.h) where there is room.
            const float play_h = std::round(52 * scale), note_h = std::round(13.5f * scale * 1.35f);
            const float hint_h = std::round(std::round(ImGui::GetFontSize() * 0.78f) + 8 * scale);
            const float hints = 2 * (hint_h + 6 * scale) + 6 * scale + st.ItemSpacing.y;  // the two keys, the gap and the spacing under it
            const float side_h = vp->WorkSize.y - 38 * scale;
            float foot = play_h + 4 * scale + ImGui::GetFrameHeight() + note_h + 3 * st.ItemSpacing.y;
            // The keys where the name and the pages (last frame's height) leave room.
            static float nav_h = 0.0f;
            const bool keys = side_h - foot - hints - 8 * scale >= nav_h;
            if (keys) foot += hints;
            ImGui::SetCursorScreenPos(o + ImVec2(16 * scale, 20 * scale));
            ImGui::BeginChild("##side", ImVec2(side_w - 32 * scale, std::max(side_h - foot - 8 * scale, 40 * scale)), 0,
                              ImGuiWindowFlags_NoBackground);
            const float w = ImGui::GetContentRegionAvail().x;
            ui::logo(kName, BBHOST_VERSION, w, vp->WorkSize.y < 700 * scale);
            if (check_updates && updater::update_available()) {
                // A newer release: its tag, a click away from its page.
                const std::string tag = updater::latest_tag() + " is out";
                const float bw = ImGui::CalcTextSize(tag.c_str()).x + st.FramePadding.x * 2;
                ImGui::SetCursorPosX((w - bw) * 0.5f);
                ImGui::PushStyleColor(ImGuiCol_Text, ui::kWarn);
                if (ui::ghost_button(tag.c_str())) updater::open_release_page();
                ImGui::PopStyleColor();
                ui::tooltip_for_item(nullptr, "Open the release page");
            }
            ImGui::Dummy(ImVec2(0, 8 * scale));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(st.ItemSpacing.x, 3 * scale));
            for (int i = 0; i < kPageCount; ++i) {
                ImU32 dot = 0;
                char badge[16] = {};
                if (i == kPageGame && !(app0_ok && eboot.ok)) dot = ui::kWarn;
                if (i == kPageOnline && net::account_logged_in()) dot = ui::kGood;
                if (i == kPageShaders) {
                    if (cv.running && cv.total) std::snprintf(badge, sizeof(badge), "%d%%", static_cast<int>(100 * cv.done / cv.total));
                    else if (recommend_compile) dot = ui::kWarn;
                }
                if (ui::nav_item(kPages[i].name, kPages[i].icon, page == i, dot, badge[0] ? badge : nullptr)) go(i);
            }
            ImGui::PopStyleVar();
            nav_h = ImGui::GetCursorPosY();
            ImGui::EndChild();
            ImGui::SetCursorScreenPos(o + ImVec2(16 * scale, 20 * scale + side_h - foot));
            ImGui::BeginChild("##foot", ImVec2(w, foot), 0, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            if (keys) {
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(st.ItemSpacing.x, 6 * scale));
                ui::key_hint("F9", "In-game patches & plugins");
                ui::key_hint("F10", "In-game settings");
                ImGui::PopStyleVar();
                ImGui::Dummy(ImVec2(0, 4 * scale));
            }
            play = ui::play_button("PLAY", ImVec2(w, play_h), can_play) || (test && !std::strcmp(test, "play") && ++frames > 30);
            if (!can_play && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ui::tooltip(nullptr, compiling ? "Play waits for the shader compile (Shaders)." : "Play needs the game folder and the 1.09 eboot (Game).");
            ImGui::Dummy(ImVec2(0, 4 * scale));
            const float half = (w - st.ItemSpacing.x) * 0.5f;
            if (ui::ghost_button("Save", ImVec2(half, 0))) save();
            ui::tooltip_for_item(nullptr, "Save to " + config_user_file() +
                                              (config_saves_locally() ? ", beside bbhost: it has the last word over " + config_user_dir() + "/bbhost.toml."
                                                                      : ", which every copy of bbhost reads."));
            ImGui::SameLine();
            if (ui::ghost_button("Quit", ImVec2(half, 0)) || (test && !std::strcmp(test, "quit") && ++frames > 30)) done = true;
            // One line under them: why Play waits, or what Save did.
            std::string note;
            ImU32 note_col = ui::kTextDim;
            if (!saved_note.empty() && (!saved_ok || ImGui::GetTime() - saved_at < 4.0)) {
                note = saved_note;
                note_col = saved_ok ? ui::kGood : ui::kBad;
            } else if (compiling) {
                note = "Waiting for the shader compile";
            } else if (!can_play) {
                note = "Needs the game folder and 1.09 eboot";
                note_col = ui::kWarn;
            }
            if (!note.empty()) {
                ImGui::PushFont(nullptr, 13.5f);
                const float nw = ImGui::CalcTextSize(note.c_str()).x;
                ImGui::SetCursorPosX(std::max(0.0f, (w - nw) * 0.5f));
                ImGui::PushStyleColor(ImGuiCol_Text, note_col);
                ImGui::TextUnformatted(note.c_str());
                ImGui::PopStyleColor();
                ImGui::PopFont();
                if (!saved_ok) ui::tooltip_for_item(nullptr, saved_note);
            }
            ImGui::EndChild();
        }
        if (test_link && ++frames > 30) {
            if (!test_created) {
                test_created = true;
                apply_server();
                host_account_action(0, "", "");
            } else if (host_account_detail().rfind("Code ", 0) == 0 || !host_account_busy()) {
                host_log("setup: link test: %s; %s", host_account_outcome().c_str(), host_account_detail().c_str());
                host_account_action(4, "", "");
                done = true;
            }
        }
        if (!test_create.empty() && ++frames > 30) {
            if (!test_created) {
                test_created = true;
                apply_server();
                host_account_action(1, test_create, "");
            } else if (!host_account_busy()) {
                host_log("setup: account test: %s; %s", host_account_outcome().c_str(), host_account_detail().c_str());
                if (host_account_take_save()) host_options_save_now();
                done = true;
            }
        }

        // The page: its heading, then itself, fading in.
        ImGui::SetCursorScreenPos(vp->WorkPos + ImVec2(side_w, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(30 * scale, 24 * scale));
        ImGui::BeginChild("##page", ImVec2(vp->WorkSize.x - side_w, vp->WorkSize.y), ImGuiChildFlags_AlwaysUseWindowPadding,
                          ImGuiWindowFlags_NoBackground);
        ImGui::PopStyleVar();
        const ImVec2 head = ImGui::GetCursorScreenPos();
        const float head_right = head.x + ImGui::GetContentRegionAvail().x;
        ImGui::PushFont(ui::font_strong(), 27.0f);
        ImGui::TextUnformatted(kPages[page].name);
        const float title_h = ImGui::GetTextLineHeight();
        ImGui::PopFont();
        ImGui::Dummy(ImVec2(0, 6 * scale));
        // Buttons at the heading's right, `w` wide in all: the cursor goes
        // there and comes back with the returned position.
        auto head_actions = [&](float w) {
            const ImVec2 back = ImGui::GetCursorScreenPos();
            ImGui::SetCursorScreenPos(ImVec2(head_right - w, head.y + (title_h - ImGui::GetFrameHeight()) * 0.5f));
            return back;
        };
        const float shown = animations ? static_cast<float>(ImGui::GetTime() - page_at) : 1.0f;
        const float fade = 1.0f - std::pow(1.0f - std::min(1.0f, shown / 0.22f), 3.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, std::max(0.02f, fade));
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (1.0f - fade) * 10 * scale);
        const float gap = 14 * scale;
        const float label_w = std::min(118 * scale, (head_right - head.x) * 0.3f);  // a large Menu text size leaves the fields room

        // A path's row: its state (-1 none, 0 wrong, 1 right; `tip` says
        // which), its label (`label_tip` in its bubble), the field and the
        // picker's button.
        auto path_row = [&](const char* label, int field, bool folder, const char* hint, int state, const std::string& tip,
                            const std::string& label_tip = {}) {
            ImGui::PushID(field);
            if (state >= 0) {
                ui::status_icon(state == 1, tip.empty() ? nullptr : tip.c_str());
                ImGui::SameLine(0, 8 * scale);
            }
            ui::row_label(label, label_w);
            if (!label_tip.empty()) ui::tooltip_for_item(nullptr, label_tip);  // the label is still the last item
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - ImGui::GetFrameHeight() - st.ItemSpacing.x);
            const bool changed = ImGui::InputTextWithHint("##p", hint, f[field].text, sizeof(f[field].text));
            ImGui::SameLine();
            if (ui::icon_button("##browse", folder ? ui::Icon::Folder : ui::Icon::File, 0.0f, folder ? "Choose a folder" : "Choose a file")) {
                void* ud = reinterpret_cast<void*>(static_cast<std::intptr_t>(field));
                const std::string at = f[field].str();
                if (folder) {
                    SDL_ShowOpenFolderDialog(on_pick, ud, win, at.empty() ? nullptr : at.c_str(), false);
                } else {
                    SDL_ShowOpenFileDialog(on_pick, ud, win, nullptr, 0, at.empty() ? nullptr : at.c_str(), false);
                }
            }
            ImGui::PopID();
            return changed;
        };
        // A problem under a row, lined up with its field.
        auto problem = [&](const std::string& text) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetFrameHeight() * 0.72f + 8 * scale + label_w);
            dim_text(text, ui::kWarn);
        };
        // Equal cards side by side: the height of a caption and `rows` rows.
        auto card_h = [&](float rows) {
            return 2 * 14 * scale + 11.5f * scale + 4 * scale + st.ItemSpacing.y + rows * ImGui::GetFrameHeightWithSpacing() - st.ItemSpacing.y;
        };

        if (page == kPageGame) {
            const float w = ImGui::GetContentRegionAvail().x;
            // Why the window opened, until the paths are right: a welcome
            // when nothing is set yet.
            if (!reason.empty() && !(app0_ok && eboot.ok)) {
                if (cfg.app0.empty() && cfg.eboot.empty())
                    ui::banner(ui::Icon::Moon, ui::kIce, "Welcome, hunter. Choose the game folder and the 1.09 eboot below, then press Play.");
                else
                    ui::banner(ui::Icon::Warning, ui::kWarn, capitalised(reason));
                ImGui::Dummy(ImVec2(0, 2 * scale));
            }
            ui::card_begin("##files", ImVec2(w, 0), "GAME FILES");
            {
                std::string bad;
                if (app0.empty()) bad = "";
                else if (!is_dir(app0)) bad = "That folder does not exist.";
                else if (!app0_ok && (is_dir(app0 + "/../dvdroot_ps4") || fs::path(app0).filename() == "dvdroot_ps4"))
                    bad = "Pick the folder above dvdroot_ps4, not dvdroot_ps4 itself.";
                else if (!app0_ok) bad = "No dvdroot_ps4 in it.";
                else if (app0_ver.app_ver.empty()) bad = "No sce_sys/param.sfo in it: is it the whole game folder?";
                else if (app0_ver.app_ver != "01.09")
                    bad = "Version " + app0_ver.app_ver +
                          (app0_ver.update.empty() ? std::string() : " (in " + fs::path(app0_ver.update).filename().string() + ")") +
                          ": copy the 1.09 update's files over this folder, or the game stops while loading.";
                const bool good = app0_ok && app0_ver.app_ver == "01.09";
                const std::string tip = good ? (app0_ver.update.empty() ? std::string("Has dvdroot_ps4, version 01.09.")
                                                                        : "Has dvdroot_ps4, version 01.09 (the update read from " +
                                                                              fs::path(app0_ver.update).filename().string() + ").")
                                             : app0.empty() ? std::string("Not set: the folder that contains dvdroot_ps4.") : bad;
                path_row("Game folder", kApp0, true, "The folder that contains dvdroot_ps4", good ? 1 : 0, tip);
                if (!bad.empty()) problem(bad);
            }
            {
                if (path_row("Eboot", kEboot, false, "The 1.09 update's eboot.bin, decrypted", eboot.ok ? 1 : 0,
                             eboot.ok ? "The 1.09 eboot."
                             : f[kEboot].str().empty() ? "Not set: the 1.09 update's eboot.bin, decrypted."
                                                       : "Not the 1.09 eboot.") ||
                    (eboot.path != f[kEboot].str() && !ImGui::IsAnyItemActive())) {
                    eboot = check_eboot(f[kEboot].str());
                }
                if (!f[kEboot].str().empty() && !eboot.ok)
                    problem(!is_file(f[kEboot].str()) ? std::string("That file does not exist.")
                                                      : "Not the 1.09 eboot (SHA-256 " + eboot.sha.substr(0, 16) + "...).");
            }
            ui::card_end();
            ImGui::Dummy(ImVec2(0, gap - st.ItemSpacing.y));

            // The display, and online at a glance, side by side.
            const float row_lw = std::min(104 * scale, (w - gap) * 0.19f);  // a third of the card at most
            ui::card_begin("##display", ImVec2(std::round((w - gap) * 0.56f), card_h(5)), "DISPLAY");
            {
                ui::row_label("Window", row_lw);
                static const char* const kModes[] = {"Windowed", "Fullscreen"};
                int mode = fullscreen ? 1 : 0;
                if (ui::segmented("##mode", &mode, kModes, 2, ImGui::GetContentRegionAvail().x)) fullscreen = mode == 1;
                ui::row_label("Frame rate", row_lw);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ui::combo("##cap", &cap_choice, kCapLabels, kCapCount, kCapTips);
                ui::tooltip_for_item("Frame rate",
                                     "The game's own speed. Display-Limited caps it at your monitor's refresh rate when the game starts (180 fps on a "
                                     "180 Hz monitor).");
                ui::row_label("Resolution", row_lw);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ui::combo("##res", &resolution, kResolutions, static_cast<int>(sizeof(kResolutions) / sizeof(kResolutions[0])));
                ui::tooltip_for_item("Render resolution", "The size the game draws at. F10 in the game has every other setting.");
                ui::toggle_row("Skip company logos", nullptr, &skip_logos,
                               "The three company logos and the Exit Game warning are skipped as the game starts.");
                ui::toggle_row("Performance mode", nullptr, &performance_mode,
                               "Requires a restart. Every post-effect and YEBIS pass off for frame rate: SSAO, motion blur, anti-aliasing, depth of field, chromatic aberration, bloom, and the dynamic lights' shadows (lamps, torches, the lantern; the sun's and the moon's stay). The map's fog stays.");
            }
            ui::card_end();
            ImGui::SameLine(0, gap);
            ui::card_begin("##online", ImVec2(0, card_h(5)), "ONLINE");
            {
                ui::row_label("Server", 72 * scale);
                static const char* const kShort[] = {"Live", "Playtest", "Custom"};
                ImGui::SetNextItemWidth(-FLT_MIN);
                ui::combo("##server", &server, kShort, 3);
                ui::tooltip_for_item(nullptr, server == 2 ? (f[kHost].str().empty() ? std::string("A server of your own: set it on the Online page.")
                                                                                    : f[kHost].str())
                                                          : std::string(kServers[server].host));
                ImGui::Dummy(ImVec2(0, 4 * scale));
                const bool in = net::account_logged_in();
                const float av = std::round(ImGui::GetFrameHeight() * 1.5f);
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float right = at.x + ImGui::GetContentRegionAvail().x;
                const char* go_label = in ? "Manage" : "Sign in";
                const float bw = ImGui::CalcTextSize(go_label).x + st.FramePadding.x * 3;
                ui::avatar(in ? net::account_name() : std::string(), av);
                ImGui::SameLine(0, 12 * scale);
                ImGui::BeginGroup();
                ImGui::SetCursorScreenPos(ImVec2(ImGui::GetCursorScreenPos().x, at.y + 1 * scale));
                // The words stop short of the button.
                ImGui::PushTextWrapPos(right - bw - 10 * scale - ImGui::GetWindowPos().x);
                ImGui::PushFont(ui::font_strong(), 0.0f);
                ImGui::TextUnformatted(in ? net::account_name().c_str() : "Not signed in");
                ImGui::PopFont();
                ImGui::PushFont(nullptr, 14.0f);
                ImGui::PushStyleColor(ImGuiCol_Text, in ? ui::kGood : ui::kTextDim);
                ImGui::TextUnformatted(in ? "Signed in" : "Sign in to play online");
                ImGui::PopStyleColor();
                ImGui::PopFont();
                ImGui::PopTextWrapPos();
                ImGui::EndGroup();
                ImGui::SetCursorScreenPos(ImVec2(right - bw, at.y + (av - ImGui::GetFrameHeight()) * 0.5f));
                if (in ? ui::ghost_button(go_label) : ui::primary_button(go_label, ImVec2(bw, 0))) go(kPageOnline);
                ImGui::SetCursorScreenPos(ImVec2(at.x, at.y + av));
                ImGui::Dummy(ImVec2(0, 0));
            }
            ui::card_end();

            // Caches that are not this bbhost's or this driver's: one line,
            // once there is a game to compile them for.
            if (recommend_compile && app0_ok && eboot.ok) {
                ImGui::Dummy(ImVec2(0, gap - st.ItemSpacing.y));
                ui::card_begin("##advice", ImVec2(w, 0));
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float fh = ImGui::GetFrameHeight();
                ImGui::Dummy(ImVec2(fh * 0.8f, fh));
                ui::draw_icon(ImGui::GetWindowDrawList(), ui::Icon::Shaders, at + ImVec2(fh * 0.4f, fh * 0.5f), fh * 0.62f, ui::kWarn);
                ImGui::SameLine(0, 8 * scale);
                const char* btn = "Compile now";
                const float bw = ImGui::CalcTextSize(btn).x + st.FramePadding.x * 3;
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted((advice.why + ".").c_str());
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, ui::kTextDim);
                ImGui::TextUnformatted((capitalised(advice.time) + ", optional").c_str());
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::SetCursorScreenPos(ImVec2(head_right - 18 * scale - bw, at.y));
                if (ui::primary_button(btn, ImVec2(bw, 0), app0_ok && eboot.ok)) {
                    start_compile();
                    go(kPageShaders);
                }
                ui::tooltip_for_item(nullptr, "Builds every shader up front, so the first start does not. Otherwise the game builds them while you play.");
                ui::card_end();
            }
        } else if (page == kPageOnline) {
            const float w = ImGui::GetContentRegionAvail().x;
            ui::card_begin("##server", ImVec2(w, 0), "SERVER");
            {
                ui::row_label("Server", label_w);
                static const char* const kServerLabels[] = {kServers[0].label, kServers[1].label, "Custom"};
                ImGui::SetNextItemWidth(-FLT_MIN);
                ui::combo("##server", &server, kServerLabels, 3);
                if (server == 2) {
                    ui::row_label("Host", label_w);
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::InputTextWithHint("##host", "server.example.com", f[kHost].text, sizeof(f[kHost].text));
                    ui::row_label("Scheme", label_w);
                    ImGui::SetNextItemWidth(120 * scale);
                    ImGui::InputText("##scheme", scheme, sizeof(scheme));
                    ui::toggle_row("Check the server's certificate", nullptr, &verify_tls);
                    ui::toggle_row("Require an account", nullptr, &require_account, "Players sign in (here, or F10 > Account) to play online.");
                }
                if (!cfg.profile.empty()) dim_text("This run uses " + cfg.profile_config + ", whose own settings may choose another server.");
            }
            ui::card_end();
            ImGui::Dummy(ImVec2(0, gap - st.ItemSpacing.y));
            // The private server's account, as F10 > ACCOUNT in the game has
            // it: no password - this PC linked to the account signed into on
            // the website with Discord, or an account held by this PC with a
            // recovery code. The calls go to the server chosen above.
            ui::card_begin("##account", ImVec2(w, 0), "ACCOUNT");
            {
                const bool busy = host_account_busy();
                const bool in = net::account_logged_in();
                const float av = std::round(ImGui::GetFrameHeight() * 1.6f);
                ui::avatar(in ? net::account_name() : std::string(), av);
                ImGui::SameLine(0, 12 * scale);
                ImGui::BeginGroup();
                ImGui::PushFont(ui::font_strong(), 18.0f);
                ImGui::TextUnformatted(in ? net::account_name().c_str() : "Not signed in");
                ImGui::PopFont();
                const std::string outcome = host_account_outcome();
                dim_text(!outcome.empty() && outcome != host_account_signed_in() ? outcome
                         : in                                                    ? std::string("Signed in")
                                                                                 : std::string("Log in with Discord, or keep an account on this PC."),
                         in && (outcome.empty() || outcome == host_account_signed_in()) ? ui::kGood : ui::kTextDim);
                ImGui::EndGroup();
                const std::string detail = host_account_detail();
                if (!detail.empty()) {
                    ImGui::Dummy(ImVec2(0, 2 * scale));
                    ImGui::AlignTextToFramePadding();
                    ImGui::PushFont(ui::font_strong(), 18.0f);
                    ImGui::PushStyleColor(ImGuiCol_Text, ui::kIce);
                    ImGui::TextUnformatted(detail.c_str());
                    ImGui::PopStyleColor();
                    ImGui::PopFont();
                    ImGui::SameLine();
                    if (ui::icon_button("##copy", ui::Icon::Copy, 0.0f, "Copy the code")) {
                        // The code alone: the word after "Code" or "code".
                        std::string code = detail;
                        for (const char* lead : {"Code ", "Recovery code ", "Website code "}) {
                            if (detail.rfind(lead, 0) == 0) code = detail.substr(std::strlen(lead));
                        }
                        code = code.substr(0, code.find("  "));
                        SDL_SetClipboardText(code.c_str());
                    }
                }
                ImGui::Dummy(ImVec2(0, 4 * scale));
                ImGui::BeginDisabled(busy);
                // The way in for most players: lit until someone is signed in.
                if (in ? ImGui::Button("Log in with Discord") : ui::primary_button("Log in with Discord", ImVec2(0, 0))) {
                    apply_server();
                    host_account_action(0, "", "");
                }
                ImGui::EndDisabled();
                ui::tooltip_for_item(nullptr, "Opens the website: sign in there with Discord and approve the code.");
                ImGui::Dummy(ImVec2(0, 2 * scale));
                ui::caption("OR AN ACCOUNT KEPT ON THIS PC");
                const float field_w = std::round(240 * scale);
                ImGui::BeginDisabled(busy);
                ImGui::SetNextItemWidth(field_w);
                ImGui::InputTextWithHint("##acct_name", "Name", acct_name, sizeof(acct_name));
                ImGui::SameLine();
                if (ImGui::Button("Create account")) {
                    apply_server();
                    host_account_action(1, acct_name, "");
                }
                ui::tooltip_for_item(nullptr, "An account held by this PC, under that name. Keep the recovery code it shows.");
                ImGui::SetNextItemWidth(field_w);
                ImGui::InputTextWithHint("##acct_code", "Recovery code", acct_code, sizeof(acct_code));
                ImGui::SameLine();
                if (ImGui::Button("Recover")) {
                    apply_server();
                    host_account_action(2, acct_name, acct_code);
                    acct_code[0] = 0;
                }
                ui::tooltip_for_item(nullptr, "The account of that name, back on this PC with its recovery code.");
                ImGui::BeginDisabled(!in);
                if (ImGui::Button("Website sign-in code")) {
                    apply_server();
                    host_account_action(3, "", "");
                }
                ImGui::EndDisabled();
                ui::tooltip_for_item(nullptr, "A code to sign in to the website as this account.");
                ImGui::EndDisabled();
                ImGui::SameLine();
                // Also stops a Discord link that is waiting for approval.
                ImGui::BeginDisabled(!busy && !in);
                if (ImGui::Button(busy ? "Cancel" : "Sign out")) {
                    apply_server();
                    host_account_action(4, "", "");
                }
                ImGui::EndDisabled();
            }
            ui::card_end();
        } else if (page == kPageEnhancements) {
            ui::card_begin("##enhancements", ImVec2(ImGui::GetContentRegionAvail().x, 0));
            for (std::size_t i = 0; i < std::size(enhancements); ++i) {
                Enhancement& e = enhancements[i];
                if (i) {
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    ImGui::GetWindowDrawList()->AddLine(ImVec2(at.x, at.y - st.ItemSpacing.y * 0.5f + 1),
                                                        ImVec2(at.x + ImGui::GetContentRegionAvail().x, at.y - st.ItemSpacing.y * 0.5f + 1),
                                                        ui::alpha(ui::kViolet, 40));
                    ImGui::Dummy(ImVec2(0, 2 * scale));
                }
                ui::toggle_row(e.label, e.sub, &e.on, e.text);
                ImGui::Dummy(ImVec2(0, 2 * scale));
            }
            ui::card_end();
        } else if (page == kPagePatches) {
            // Whether a row's switch can be changed here.
            auto switchable = [&](const PatchRow& r) {
                const bool other_eboot = eboot.ok && !r.p.eboot.empty() && r.p.eboot != eboot.sha;
                const bool unknown_option = !r.p.option.empty() && !patch_manifest_option_known(r.p.option);
                return r.p.error.empty() && !other_eboot && !unknown_option && (!r.p.option.empty() || bare_key(r.p.name));
            };
            // All off, or each patch as its own file has it (a patch that
            // says enabled = false off, the rest on); one that follows a
            // setting keeps that setting's own default.
            auto set_all = [&](bool defaults) {
                for (PatchRow& r : patch_rows) {
                    if (!switchable(r) || (defaults && !r.p.option.empty())) continue;
                    *patch_switch(r) = defaults && r.p.enabled;
                }
            };
            {
                const float find_w = std::round(200 * scale);
                const float b1 = ImGui::CalcTextSize("Defaults").x + st.FramePadding.x * 2, b2 = ImGui::CalcTextSize("All off").x + st.FramePadding.x * 2;
                const ImVec2 back = head_actions(find_w + b1 + b2 + st.ItemSpacing.x * 2);
                const ImVec2 fp = ImGui::GetCursorScreenPos();
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(30 * scale, st.FramePadding.y));
                ImGui::SetNextItemWidth(find_w);
                ImGui::InputTextWithHint("##find", "Search", patch_find, sizeof(patch_find));
                ImGui::PopStyleVar();
                ui::draw_icon(ImGui::GetWindowDrawList(), ui::Icon::Search, fp + ImVec2(15 * scale, ImGui::GetFrameHeight() * 0.5f),
                              14 * scale, ui::kTextDim);
                ImGui::SameLine();
                ImGui::BeginDisabled(patch_rows.empty());
                if (ui::ghost_button("Defaults")) set_all(true);
                ui::tooltip_for_item(nullptr, "Each patch as its own file has it.");
                ImGui::SameLine();
                if (ui::ghost_button("All off")) set_all(false);
                ui::tooltip_for_item(nullptr, "The game's code as it shipped.");
                ImGui::EndDisabled();
                ImGui::SetCursorScreenPos(back);
            }
            std::vector<std::size_t> shown;
            for (std::size_t i = 0; i < patch_rows.size(); ++i) {
                if (contains_nocase(patch_rows[i].p.name, patch_find) || contains_nocase(patch_rows[i].p.title, patch_find) ||
                    contains_nocase(patch_rows[i].p.description, patch_find))
                    shown.push_back(i);
            }
            // One line each, in two columns; the bubble has the rest. The
            // card fits the rows, or the page's height with them scrolling.
            const std::size_t per_col = (shown.size() + 1) / 2;
            const float row_h = std::round(ImGui::GetFontSize() + 2 * 3 * scale + 9 * scale);  // a box with the list's padding, and air
            const float fit = 2 * 14 * scale + std::max<float>(1.0f, static_cast<float>(per_col)) * row_h + 4 * scale;
            ui::card_begin("##patchlist", ImVec2(ImGui::GetContentRegionAvail().x, std::min(fit, std::max(ImGui::GetContentRegionAvail().y, 160 * scale))));
            if (patch_rows.empty()) dim_text("No patches found.");
            else if (shown.empty()) dim_text("No patch matches.");
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4 * scale, 3 * scale));
            const float col_gap = 22 * scale;
            const float col_w = (ImGui::GetContentRegionAvail().x - col_gap) * 0.5f;
            const ImVec2 o = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            for (std::size_t k = 0; k < shown.size(); ++k) {
                const std::size_t i = shown[k];
                PatchRow& r = patch_rows[i];
                const float col = static_cast<float>(k / per_col), row = static_cast<float>(k % per_col);
                const ImVec2 a = o + ImVec2(col * (col_w + col_gap), row * row_h);
                const ImVec2 b = a + ImVec2(col_w, row_h - 4 * scale);
                const bool hot = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(a - ImVec2(6 * scale, 0), b);
                if (hot) dl->AddRectFilled(a - ImVec2(6 * scale, 0), b, ImGui::GetColorU32(IM_COL32(255, 255, 255, 16)), 7 * scale);
                if (k == std::min<std::size_t>(2, shown.size() - 1)) patch_row_at = (a + b) * 0.5f;
                const bool other_eboot = eboot.ok && !r.p.eboot.empty() && r.p.eboot != eboot.sha;
                const bool unknown_option = !r.p.option.empty() && !patch_manifest_option_known(r.p.option);
                const bool fixed = !switchable(r);
                bool* on = patch_switch(r);
                bool off = false;
                // Its marks at the right: a gameplay change, or why it cannot be switched.
                float right = b.x - 4 * scale;
                const float fh = ImGui::GetFrameHeight();
                if (fixed) {
                    ui::draw_icon(dl, ui::Icon::Warning, ImVec2(right - fh * 0.32f, (a.y + b.y) * 0.5f), fh * 0.6f, ImGui::GetColorU32(ui::kWarn));
                    right -= fh * 0.7f + 6 * scale;
                }
                if (r.p.gameplay) {
                    const char* tag = "gameplay";
                    const float tfs = std::round(ImGui::GetFontSize() * 0.74f);
                    const ImVec2 ts = ui::font_strong()->CalcTextSizeA(tfs, FLT_MAX, 0.0f, tag);
                    const ImVec2 pa(right - ts.x - 14 * scale, (a.y + b.y) * 0.5f - ts.y * 0.5f - 2 * scale);
                    const ImVec2 pb(right, (a.y + b.y) * 0.5f + ts.y * 0.5f + 2 * scale);
                    dl->AddRectFilled(pa, pb, ImGui::GetColorU32(ui::alpha(ui::kPink, 46)), (pb.y - pa.y) * 0.5f);
                    dl->AddText(ui::font_strong(), tfs, pa + ImVec2(7 * scale, 2 * scale), ImGui::GetColorU32(ui::kPink), tag);
                    right = pa.x - 6 * scale;
                }
                // Its name up to the marks, cut short in a narrow window.
                const float name_w = right - a.x - fh - ImGui::GetStyle().ItemInnerSpacing.x;
                const std::string name = ui::fit_text(patch_display_name(r.p), name_w) + "###patch";
                ImGui::SetCursorScreenPos(ImVec2(a.x, a.y + (b.y - a.y - fh) * 0.5f));
                ImGui::PushID(static_cast<int>(i));
                ImGui::BeginDisabled(fixed);
                // Variants of one patch (a group) are one at a time: turning
                // one on turns the others off.
                if (ImGui::Checkbox(name.c_str(), fixed ? &off : on) && !fixed && *on && !r.p.group.empty()) {
                    for (PatchRow& o2 : patch_rows) {
                        if (&o2 != &r && o2.p.group == r.p.group && switchable(o2)) *patch_switch(o2) = false;
                    }
                }
                ImGui::EndDisabled();
                ImGui::PopID();
                if (hot) {
                    std::string note;
                    auto add = [&](const std::string& s) {
                        if (s.empty()) return;
                        if (!note.empty()) note += "\n";
                        note += s;
                    };
                    if (!r.p.error.empty()) add("Not read: " + r.p.error + ".");
                    else if (other_eboot) add("Made for another eboot: not applied.");
                    else if (unknown_option) add("Follows " + r.p.option + ", which this bbhost does not know: not applied.");
                    else if (r.p.option.empty() && !bare_key(r.p.name)) add("Its name cannot be a bbhost.toml key, so it is always on.");
                    // The patch's name and what it does; a note only for one that cannot be switched.
                    ui::tooltip(patch_display_name(r.p).c_str(), capitalised(r.p.description.empty() ? std::string("No description.") : r.p.description), note);
                }
            }
            ImGui::PopStyleVar();
            ImGui::SetCursorScreenPos(o + ImVec2(0, static_cast<float>(per_col) * row_h));
            ImGui::Dummy(ImVec2(1, 1));
            ui::card_end();
        } else if (page == kPageShaders) {
            const float w = ImGui::GetContentRegionAvail().x;
            // The caches at a glance, and the Shaders setting beside them.
            ui::card_begin("##state", ImVec2(w, 0), "SHADER CACHES");
            {
                // What the caches are: a mark, a headline and a line under it.
                struct State {
                    ui::Icon icon;
                    ImU32 col;
                    std::string head, sub;
                };
                State state{ui::Icon::Check, ui::kGood, "Ready", "Made by this bbhost, for this graphics driver."};
                if (cv.running) state = {ui::Icon::Refresh, ui::kBlue, "Compiling", cv.phase};
                else if (!cache_state.have_caches) state = {ui::Icon::Warning, ui::kWarn, "No shader caches yet", "The first start compiles everything."};
                else if (!cache_state.translator_current)
                    state = {ui::Icon::Warning, ui::kWarn, "Made by another bbhost version", "The next start compiles everything again."};
                else if (cache_state.device_current == 0)
                    state = {ui::Icon::Warning, ui::kWarn, "Made for another GPU or driver", "The driver compiles its part again."};
                else if (cache_state.device_current != 1) state = {ui::Icon::Info, ui::kIce, "Checking the graphics driver", "Made by this bbhost."};
                if (recommend_compile)
                    state = {ui::Icon::Warning, ui::kWarn, "Compile recommended",
                             advice.why + ". Compiling now takes " + advice.time + "; otherwise the game compiles them while you play."};
                const ui::Icon icon = state.icon;
                const ImU32 col = state.col;
                const std::string& head_line = state.head;
                const std::string& sub = state.sub;
                const float fh = ImGui::GetFrameHeight();
                const ImVec2 at = ImGui::GetCursorScreenPos();
                ImGui::Dummy(ImVec2(fh * 1.3f, fh * 1.6f));
                ui::draw_icon(ImGui::GetWindowDrawList(), icon, at + ImVec2(fh * 0.6f, fh * 0.8f), fh * 0.95f, ImGui::GetColorU32(col));
                ImGui::SameLine(0, 10 * scale);
                const float seg_w = std::round(190 * scale);
                ImGui::BeginGroup();
                ImGui::PushFont(ui::font_strong(), 19.0f);
                ImGui::TextUnformatted(head_line.c_str());
                ImGui::PopFont();
                ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + std::max(100.0f, ImGui::GetContentRegionAvail().x - seg_w - 20 * scale));
                ImGui::PushStyleColor(ImGuiCol_Text, ui::kTextDim);
                ImGui::TextUnformatted(sub.c_str());
                ImGui::PopStyleColor();
                ImGui::PopTextWrapPos();
                ImGui::EndGroup();
                const ImVec2 after = ImGui::GetCursorScreenPos();
                // The Shaders setting, at the right.
                ImGui::SetCursorScreenPos(ImVec2(head_right - 18 * scale - seg_w, at.y + (fh * 1.6f - fh) * 0.5f));
                static const char* const kShaderModes[] = {"HLSL", "Lifted"};
                ImGui::BeginDisabled(cv.running);
                ui::segmented("##shaders", &shaders, kShaderModes, 2, seg_w, kHostShaderModeTips,
                              "Saved with Save or Play; the game reads it at its start.");
                ImGui::EndDisabled();
                ImGui::SetCursorScreenPos(after);
                ImGui::Dummy(ImVec2(0, 0));
                if (cache_state.have_caches) {
                    char sizes[256];
                    std::snprintf(sizes, sizeof(sizes), "Translations and modules %.0f MiB  |  pipelines %.0f MiB  |  drawn %.1f + %.1f MiB",
                                  cache_state.translation_bytes / 1048576.0, cache_state.pipeline_bytes / 1048576.0,
                                  cache_state.stage_manifest_bytes / 1048576.0, cache_state.pipeline_manifest_bytes / 1048576.0);
                    dim_text(sizes, ui::kTextFaint);
                    ui::tooltip_for_item(nullptr, cache_state.dir);
                } else {
                    dim_text(cache_state.dir, ui::kTextFaint);
                }
            }
            ui::card_end();
            ImGui::Dummy(ImVec2(0, gap - st.ItemSpacing.y));
            ui::card_begin("##compile", ImVec2(w, 0), "COMPILE NOW");
            {
                if (!cv.running) {
                    if (ui::primary_button("Compile shaders now", ImVec2(0, 0), app0_ok && eboot.ok)) start_compile();
                    ImGui::SameLine();
                    ImGui::AlignTextToFramePadding();
                    ImGui::PushStyleColor(ImGuiCol_Text, ui::kTextDim);
                    ImGui::TextUnformatted(app0_ok && eboot.ok ? "Every shader the game uses, with this PC's driver. It can be cancelled."
                                                               : "Needs the game folder and the 1.09 eboot.");
                    ImGui::PopStyleColor();
                } else {
                    if (ui::ghost_button("Cancel")) compile_job.cancel();
                    ImGui::SameLine();
                    ImGui::AlignTextToFramePadding();
                    ImGui::PushStyleColor(ImGuiCol_Text, ui::kTextDim);
                    ImGui::TextUnformatted(cv.phase.c_str());
                    ImGui::PopStyleColor();
                }
                if (cv.running || cv.finished || cv.cancelled || !cv.error.empty()) {
                    const float frac = cv.total ? static_cast<float>(cv.done) / static_cast<float>(cv.total) : 0.0f;
                    char over[96];
                    std::snprintf(over, sizeof(over), "%llu of %llu", static_cast<unsigned long long>(cv.done),
                                  static_cast<unsigned long long>(cv.total));
                    ImGui::Dummy(ImVec2(0, 2 * scale));
                    ui::progress(frac, over);
                    char line[256];
                    if (cv.eta >= 0 && cv.running) std::snprintf(line, sizeof(line), "%.0f s so far, about %.0f s left", cv.seconds, cv.eta);
                    else std::snprintf(line, sizeof(line), "%.0f s", cv.seconds);
                    std::string stats = line;
                    std::snprintf(line, sizeof(line), "  |  %llu took DXC's module, %llu kept the lift, %llu failed",
                                  static_cast<unsigned long long>(cv.hlsl), static_cast<unsigned long long>(cv.lift),
                                  static_cast<unsigned long long>(cv.failed));
                    dim_text(stats + line);
                    if (cv.recipes_total) {
                        std::snprintf(line, sizeof(line), "Pipelines from your earlier sessions and the shipped list: %llu of %llu",
                                      static_cast<unsigned long long>(cv.recipes_done), static_cast<unsigned long long>(cv.recipes_total));
                        dim_text(line);
                    }
                    if (cv.finished) dim_text("Done: the next start takes these from the caches.", ui::kGood, false);
                    if (cv.cancelled) dim_text("Cancelled: what was compiled is kept; the game compiles the rest.", ui::kWarn, false);
                    if (!cv.error.empty()) dim_text(cv.error, ui::kWarn, false);
                    for (const std::string& r : cv.reasons) dim_text("- " + r);
                }
            }
            ui::card_end();
            if (!cache_state.last.empty()) {
                ImGui::Dummy(ImVec2(0, gap - st.ItemSpacing.y));
                // shader-compile.txt's "key value" lines, as sentences: one
                // line, the rest in its bubble.
                std::string res, when, version, stages, hlsl, seconds, device, pipelines;
                std::vector<std::string> failed, lift;
                for (const std::string& l : cache_state.last) {
                    const std::size_t sp = l.find(' ');
                    const std::string k = l.substr(0, sp), v = sp == std::string::npos ? "" : l.substr(sp + 1);
                    if (k == "result") res = v;
                    else if (k == "when") when = v;
                    else if (k == "bbhost") version = v;
                    else if (k == "stages") stages = v;
                    else if (k == "hlsl") hlsl = v;
                    else if (k == "seconds") seconds = v;
                    else if (k == "device") device = v.substr(v.find(' ') == std::string::npos ? 0 : v.find(' ') + 1);
                    else if (k == "failed") failed.push_back(v);
                    else if (k == "pipelines") pipelines = v;
                    else if (k == "lift") lift.push_back(v);
                }
                ui::card_begin("##last", ImVec2(w, 0), "THE LAST COMPILE");
                std::string details = "On " + device + ": shaders " + stages + "; " + hlsl + ".";
                if (!pipelines.empty() && pipelines.rfind("0 of 0", 0) != 0) details += "\nPipelines " + pipelines + ".";
                for (const std::string& r : failed) {
                    if (r.find("LDS use outside a compute shader") != std::string::npos)
                        details += "\nNot compiled alone (" + r.substr(r.rfind(' ') + 1) +
                                   "): vertex shaders that write LDS, which only run inside another stage (tessellation); they are "
                                   "compiled when the game draws them.";
                    else
                        details += "\nFailed: " + r;
                }
                for (const std::string& r : lift) details += "\nKept the lift: " + r;
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted((capitalised(res == "done" ? "finished" : res) + ", " + when + "  |  " + seconds + " s  |  bbhost " + version).c_str());
                ImGui::SameLine();
                const ImVec2 ip = ImGui::GetCursorScreenPos();
                const float fh = ImGui::GetFrameHeight();
                ImGui::Dummy(ImVec2(fh * 0.7f, fh));
                ui::draw_icon(ImGui::GetWindowDrawList(), ui::Icon::Info, ip + ImVec2(fh * 0.35f, fh * 0.5f), fh * 0.55f, ImGui::GetColorU32(ui::kTextDim));
                ui::tooltip_for_item("The last compile", details);
                ui::card_end();
            }
        } else if (page == kPagePlugins) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(0, 0, 0, 0));
            plugin_ui_draw(plugin_ui, scale);
            ImGui::PopStyleColor();
            ui::child_cards();
            plugin_seen = true;
        } else if (page == kPageSettings) {
            const float w = ImGui::GetContentRegionAvail().x;
            ui::card_begin("##folders", ImVec2(w, 0), "FOLDERS");
            path_row("Data folder", kData, true, config_default_data_dir().c_str(), -1, "",
                     "Saves and caches. Empty for the default: " + config_default_data_dir());
            ui::row_label("Settings file", label_w);
            {
                const std::string file = config_user_file();
                const float bw = ImGui::GetFrameHeight();
                ImGui::PushStyleColor(ImGuiCol_Text, ui::kTextDim);
                ImGui::TextUnformatted(ui::fit_left(generic(file), head_right - 18 * scale - bw - st.ItemSpacing.x - ImGui::GetCursorScreenPos().x).c_str());
                ImGui::PopStyleColor();
                ui::tooltip_for_item(nullptr, generic(file) + (config_saves_locally() ? "\nBeside bbhost: this copy reads it after the per-user one, and saves here."
                                                                                      : "\nEvery copy of bbhost reads it."));
                ImGui::SameLine(0, 0);
                ImGui::SetCursorScreenPos(ImVec2(head_right - 18 * scale - bw, ImGui::GetCursorScreenPos().y));
                if (ui::icon_button("##open", ui::Icon::Folder, 0.0f, "Open its folder")) {
                    std::string url = "file:///" + generic(fs::path(file).parent_path().string());
                    SDL_OpenURL(url.c_str());
                }
            }
            ui::card_end();
            ImGui::Dummy(ImVec2(0, gap - st.ItemSpacing.y));
            ui::card_begin("##startup", ImVec2(w, 0), "START-UP");
            ui::toggle_row("Show this window at every start", "Or start bbhost with --setup to open it once.", &show_every_start);
            ImGui::Dummy(ImVec2(0, 2 * scale));
            ui::toggle_row("Menu animations", "The drifting background, its motes, the Play button's shine.", &animations,
                           "The window's motion: the background's slow drift and the motes rising over it, the moon's glow, the Play "
                           "button's pulse and the light sweeping across it, the page fade and the sliding highlights. Off, the "
                           "window stands still.");
            ImGui::Dummy(ImVec2(0, 2 * scale));
            // As a switch's row: its label and a line under it, the control (`cw` wide) at the right.
            const auto choice_row = [&](const char* label, const char* sub, float cw, const auto& control) {
                const ImVec2 p = ImGui::GetCursorScreenPos();
                const float w = ImGui::GetContentRegionAvail().x;
                const float fs = ImGui::GetFontSize(), sub_fs = std::round(fs * 0.86f);
                ImFont* font = ImGui::GetFont();
                const float lh = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, label).y;
                const float sub_h = font->CalcTextSizeA(sub_fs, FLT_MAX, w - cw - 16 * scale, sub).y + 2 * scale;
                const float h = std::max(ImGui::GetFrameHeight(), lh + sub_h), ty = p.y + (h - lh - sub_h) * 0.5f;
                ImDrawList* dl = ImGui::GetWindowDrawList();
                dl->AddText(font, fs, ImVec2(p.x, ty), ImGui::GetColorU32(ui::kText), label);
                dl->AddText(font, sub_fs, ImVec2(p.x, ty + lh + 2 * scale), ImGui::GetColorU32(ui::kTextDim), sub, nullptr, w - cw - 16 * scale);
                ImGui::SetCursorScreenPos(ImVec2(p.x + w - cw, p.y + (h - ImGui::GetFrameHeight()) * 0.5f));
                ImGui::SetNextItemWidth(cw);
                control();
                ImGui::SetCursorScreenPos(p);
                ImGui::Dummy(ImVec2(w, h));
            };
            choice_row("Menu text size", "This window, F9 and F10; the window grows with it.", 128 * scale, [&] {
                ui::combo("##textsize", &text_choice, kTextSizeLabels, static_cast<int>(std::size(kTextSizeLabels)));
                ui::tooltip_for_item("Menu text size",
                                     "The text in bbhost's own menus - this window and the in-game F9 and F10 menus - and everything in "
                                     "them with it, this window's own size included: 150% or 200% for a 1440p or 4K screen. 100% is the "
                                     "usual size (with Windows' display scaling). A window larger than the screen stops at the screen and "
                                     "scrolls. Applies at once; F10 has the same setting.");
            });
            // Menu hue and Menu vibrance: F10's sliders (a step each, its value shown),
            // saved at once; a double click puts one back to 0.
            const auto colour_slider = [&](const char* id, const char* key, int steps, int origin, int step, const char* unit,
                                           const char* title, const char* tip) {
                int i = std::max(0, host_opt_index(key));
                const int value = origin + step * i;
                char shown[24];
                std::snprintf(shown, sizeof(shown), value > 0 ? "+%d%s" : "%d%s", value, unit);
                if (ImGui::SliderInt(id, &i, 0, steps - 1, shown, ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput))
                    host_opt_set_index(key, i);
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                    host_opt_set_index(key, -origin / step);
                ui::tooltip_for_item(title, tip);
            };
            ImGui::Dummy(ImVec2(0, 2 * scale));
            choice_row("Menu hue", "The colours of this window, F9 and F10.", 200 * scale, [&] {
                colour_slider("##hue", "menu_hue", 24, -165, 15, "°", "Menu hue",
                              "Turns the colours of bbhost's own menus - this window and the in-game F9 and F10 menus - round the "
                              "colour wheel: 0 is their own blue and violet. The green, amber and red that mean ok, a warning and a "
                              "problem stay as they are. Double-click for 0; F10 has the same setting.");
            });
            ImGui::Dummy(ImVec2(0, 2 * scale));
            choice_row("Menu vibrance", "-100% is black and white.", 200 * scale, [&] {
                colour_slider("##vibrance", "menu_vibrance", 21, -100, 10, "%%", "Menu vibrance",
                              "How much colour bbhost's own menus have: -100% is black and white, 0 their own, +100% the most. "
                              "Double-click for 0; F10 has the same setting.");
            });
            ImGui::Dummy(ImVec2(0, 2 * scale));
            {
                const std::string sub = check_updates ? updater::status() : std::string("A newer bbhost, from its GitHub releases.");
                const std::string tip = check_updates ? updater::detail() : std::string();
                ui::toggle_row("Check for updates", sub.c_str(), &check_updates, tip.empty() ? nullptr : tip.c_str());
            }
            ImGui::Dummy(ImVec2(0, 2 * scale));
            path_row("GitHub token", kToken, false, "Only while the releases are private", -1, "",
                     "A file holding a GitHub token, for update checks while the releases are private.");
            ui::card_end();
            ImGui::Dummy(ImVec2(0, gap - st.ItemSpacing.y));
            ui::card_begin("##diagnostics", ImVec2(w, 0), "DIAGNOSTICS");
            ui::toggle_row("Tooling and logging", "Leave it on when reporting a problem.", &tooling,
                           "The log and its window, the periodic reports in it, the stall detector, the GPU busy meter (GPU % beside "
                           "the FPS counter) and the recent-draw history. Off gives the game that headroom: nothing is logged and no "
                           "log window opens.");
            ui::card_end();
        }
        ImGui::PopStyleVar();  // the fade
        ImGui::EndChild();
        if (play && can_play && save()) {
            result = LauncherResult::Play;
            done = true;
        }
        ImGui::End();

        ImGui::Render();
        // Menu hue and Menu vibrance, on everything the frame draws.
        float hue = 0.0f, vibrance = 0.0f;
        host_opt_menu_colour(&hue, &vibrance);
        ui::set_colour(hue, vibrance);
        ui::recolour(ImGui::GetDrawData());
        if (hue != title_hue || vibrance != title_vibrance) {
            title_hue = hue;
            title_vibrance = vibrance;
            dark_title_bar(win);
        }
        const ImU32 ink = ui::tint(ui::kInk);
        SDL_SetRenderDrawColor(ren, static_cast<Uint8>(ink & 0xff), static_cast<Uint8>(ink >> 8 & 0xff), static_cast<Uint8>(ink >> 16 & 0xff), 255);
        SDL_SetRenderScale(
            ren,
            io.DisplayFramebufferScale.x,
            io.DisplayFramebufferScale.y
        );
        SDL_RenderClear(ren);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), ren);
        if (!test_shot.empty() && ImGui::GetTime() >= shot_at) {
            const std::string name = shot < kPageCount ? kPages[shot].slug : "patches-bubble";
            char path[64];
            std::snprintf(path, sizeof(path), "/%02d-", shot + 1);
            const std::string file = test_shot + path + name + ".png";
            if (SDL_Surface* s = SDL_RenderReadPixels(ren, nullptr)) {
                if (!SDL_SavePNG(s, file.c_str())) host_log("setup: shot: %s: %s", file.c_str(), SDL_GetError());
                SDL_DestroySurface(s);
            }
            host_log("setup: shot %s", file.c_str());
            ++shot;
            if (shot > kPageCount) {
                done = true;
            } else {
                go(shot < kPageCount ? shot : static_cast<int>(kPagePatches));
                shot_at = ImGui::GetTime() + (shot == kPageCount ? 0.6 : 1.0);
            }
        }
        SDL_RenderPresent(ren);
    }

    // Closing the window while a compile runs: it is told to stop and write
    // what it has; the destructor waits for it.
    compile_job.cancel();
    if (devices_thread.joinable()) devices_thread.join();
    if (plugin_seen && test_shot.empty()) plugin_ui_mark_seen(plugin_ui);
    ui::shutdown();
    ui::set_animations(true);  // the switch is the setup window's; the in-game screens keep theirs
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
    host_log("setup: %s", result == LauncherResult::Play ? "play" : "closed");
    return result;
}
