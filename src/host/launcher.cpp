#include "host/launcher.h"

#include "bbhost_version.h"
#include "core/sha256.h"
#include "engine/addr.h"
#include "engine/patch_manifest.h"
#include "host/options.h"
#include "net/account.h"
#include "host/plugin_ui.h"
#include "host/updater.h"
#include "log.h"

#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_sdlrenderer3.h"

#include <SDL3/SDL.h>

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

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

void status_line(bool ok, const char* good, const std::string& bad) {
    if (ok) {
        ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1.0f), "OK  %s", good);
    } else {
        ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.30f, 1.0f), "!   %s", bad.c_str());
    }
}

}  // namespace

LauncherResult launcher_run(const HostConfig& cfg, const std::string& reason, bool plugins_tab) {
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
        host_log("setup: no window (%s); edit %s instead", SDL_GetError(), config_user_file().c_str());
        return LauncherResult::Quit;
    }
    SDL_Window* win = SDL_CreateWindow("bbhost setup", 980, 760,
                                       SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    const float scale = win ? std::max(1.0f, SDL_GetWindowDisplayScale(win)) : 1.0f;
    SDL_Renderer* ren = win ? SDL_CreateRenderer(win, nullptr) : nullptr;
    if (!win || !ren) {
        host_log("setup: no window (%s); edit %s instead", SDL_GetError(), config_user_file().c_str());
        if (win) SDL_DestroyWindow(win);
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
        return LauncherResult::Quit;
    }
    SDL_SetRenderVSync(ren, 1);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // nothing written beside the exe
    ImGui::StyleColorsDark();
    ImGui::GetStyle().FontSizeBase = 17.0f;
    ImGui_ImplSDL3_InitForSDLRenderer(win, ren);
    ImGui_ImplSDLRenderer3_Init(ren);

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
    // The F10 screen's start-up settings, through the same calls it uses.
    host_options_load();
    bool fullscreen = host_opt_index("window_mode") == 1;
    bool skip_logos = host_opt_get("skip_logos");
    bool debug_camera = host_opt_get("debug_camera");
    int frame_cap = host_opt_frame_cap();
    int resolution = host_opt_resolution_index();
    static const char* const kResolutions[] = {"1280x720",         "1600x900",         "1920x1080",        "2560x1440",
                                               "3200x1800",        "3840x2160",        "2560x1080 (21:9)", "3440x1440 (21:9)",
                                               "5120x2160 (21:9)", "3840x1080 (32:9)", "5120x1440 (32:9)", "1280x800 (16:10)",
                                               "960x600 (16:10, upscaled)", "1024x640 (16:10, upscaled)"};
    // The game's frame rate (engine/frame_rate.h): 0 is uncapped.
    static const int kCaps[] = {30, 60, 90, 0};
    int cap_choice = frame_cap == 30 ? 0 : frame_cap == 60 ? 1 : frame_cap == 90 ? 2 : 3;
    // The PC enhancements, through the same calls: each is read when the
    // game starts (main.cpp, config_set_enhancements).
    struct Enhancement {
        const char* key;
        const char* label;
        const char* text;
        bool on;
    };
    Enhancement enhancements[] = {
        {"change_appearance", "Hunter's Dream mirror",
         "The workshop's mirror opens the appearance editor: the developers' unused \"Put on Disguise\".", false},
        {"rebirth", "Rebirth at the Altar of Despair",
         "With the Yharnam Stone, the altar returns the hunter to their origin's level and gives the blood echoes back "
         "to spend again; leaving asks to accept or undo.",
         false},
        {"five_players", "Five players in the late areas",
         "The Nightmare of Mensis, Mergo's Loft, the Nightmare Frontier and the Old Hunters' areas take two invaders, as "
         "the chalice dungeons do. Every player in a session needs it on.",
         false},
    };
    for (Enhancement& e : enhancements) e.on = host_opt_get(e.key);
    // The Patches tab: every manifest the loader would read
    // (engine/patch_manifest.h), each with its switch. One that follows a
    // setting (its `option`) shows that setting - the Game tab's Skip company
    // logos, its Debug camera - so a patch has one switch; the rest (The Old
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
        r.on = r.was = !patch_manifest_off(p.name);
        r.p = std::move(p);
        patch_rows.push_back(std::move(r));
    }
    auto patch_switch = [&](PatchRow& r) -> bool* {
        const std::string& o = r.p.option;
        if (o == "startup.skip_intro") return &skip_logos;
        if (o == "debug.free_camera") return &debug_camera;
        if (o == "loading.quick_reentry") return &quick_reentry;
        if (o == "streaming.all_post_processors") return &all_post_processors;
        return &r.on;
    };
    auto patch_note = [&](const PatchRow& r) -> std::string {
        const std::string& o = r.p.option;
        if (o == "startup.skip_intro") return "The same switch as the Game tab's Skip company logos.";
        if (o == "debug.free_camera") return "The same switch as the Game tab's Debug camera.";
        if (o == "loading.quick_reentry") return "Saved as quick_reentry under [loading] in bbhost.toml.";
        if (o == "streaming.all_post_processors") return "Saved as all_post_processors under [streaming] in bbhost.toml.";
        return "Saved as " + r.p.name + " under [patches] in bbhost.toml.";
    };

    EbootCheck eboot = check_eboot(f[kEboot].str());
    PluginUi plugin_ui;
    bool plugin_seen = false;
    std::string saved_note;
    LauncherResult result = LauncherResult::Quit;
    bool done = false;
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
        // What this bbhost's file looks like (core/config.h): a value saved
        // from here is a choice.
        e.push_back({"bbhost", "config_version", std::to_string(kUserConfigVersion)});
        // The Patches tab's own switches, each only once changed.
        for (const PatchRow& r : patch_rows) {
            if (r.p.option.empty() && r.on != r.was && bare_key(r.p.name)) e.push_back({"patches", r.p.name, r.on ? "true" : "false"});
        }
        if (quick_reentry != quick_reentry_was) e.push_back({"loading", "quick_reentry", quick_reentry ? "true" : "false"});
        if (all_post_processors != all_post_processors_was)
            e.push_back({"streaming", "all_post_processors", all_post_processors ? "true" : "false"});
        const std::string file = config_user_file();
        if (!config_set_values(file, e)) {
            saved_note = "Could not write " + file;
            return false;
        }
        for (PatchRow& r : patch_rows) r.was = r.on;
        quick_reentry_was = quick_reentry;
        all_post_processors_was = all_post_processors;
        host_opt_set_index("window_mode", fullscreen ? 1 : 0);
        host_opt_set("skip_logos", skip_logos);
        host_opt_set("debug_camera", debug_camera);
        host_opt_set_frame_cap(kCaps[cap_choice]);
        host_opt_set_resolution_index(resolution);
        for (const Enhancement& x : enhancements) host_opt_set(x.key, x.on);
        host_options_save_now();
        saved_note = "Saved to " + file;
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
        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("setup", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);

        ImGui::Text("bbhost %s - setup", BBHOST_VERSION);
        ImGui::TextDisabled("Saved to %s, which every copy of bbhost reads.", config_user_file().c_str());
        if (!reason.empty()) ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.30f, 1.0f), "%s", reason.c_str());
        // While the Updates box is checked: one check, on a thread of its own
        // (host/updater.h), and a newer release's page opens in the browser.
        if (check_updates) updater::check_once();
        if (check_updates && updater::update_available()) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(ImVec4(0.95f, 0.80f, 0.40f, 1.0f), "bbhost %s is available", updater::latest_tag().c_str());
            ImGui::SameLine();
            if (ImGui::Button("Open the release page")) updater::open_release_page();
        }
        ImGui::Separator();

        const std::string app0 = f[kApp0].str();
        const bool app0_ok = is_dir(app0 + "/dvdroot_ps4");
        // The tabs above the buttons: the game's settings, and the plugins.
        const float footer = ImGui::GetFrameHeightWithSpacing() * 4.2f;
        bool game_tab = false;
        if (ImGui::BeginTabBar("##tabs")) {
            if (ImGui::BeginTabItem("Game")) {
                game_tab = true;
                ImGui::EndTabItem();
            }
            // Additions to the game itself, for players who want it as it
            // shipped: the same three the game's System > PC Enhancements and
            // F10 > PC ENHANCEMENTS have.
            if (ImGui::BeginTabItem("PC enhancements")) {
                ImGui::BeginChild("##enhancements", ImVec2(0, ImGui::GetContentRegionAvail().y - footer));
                ImGui::TextWrapped("Additions to the game itself, all on by default. Turn them off for the game as it shipped.");
                ImGui::TextDisabled("Each is read when the game starts: one changed in the game waits for the next start.");
                ImGui::Spacing();
                for (Enhancement& e : enhancements) {
                    ImGui::Checkbox(e.label, &e.on);
                    ImGui::Indent();
                    ImGui::PushTextWrapPos(0.0f);
                    ImGui::TextDisabled("%s", e.text);
                    ImGui::PopTextWrapPos();
                    ImGui::Unindent();
                    ImGui::Spacing();
                }
                if (ImGui::Button("All off (the game as it shipped)")) {
                    for (Enhancement& e : enhancements) e.on = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("All on")) {
                    for (Enhancement& e : enhancements) e.on = true;
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            ImGuiTabItemFlags pflags = 0;
            if (plugins_tab) {
                pflags = ImGuiTabItemFlags_SetSelected;
                plugins_tab = false;
            }
            // The patch manifests, as the Plugins tab has the plugins.
            if (ImGui::BeginTabItem("Patches")) {
                ImGui::BeginChild("##patches", ImVec2(0, ImGui::GetContentRegionAvail().y - footer));
                ImGui::TextWrapped("Byte patches to the game's code, applied when it starts: bbhost's own (the patches folder beside "
                                   "it) and any a mod brings (<mods>/patches).");
                ImGui::TextDisabled("A patch made for another eboot is never applied.");
                ImGui::Spacing();
                if (patch_rows.empty()) ImGui::TextDisabled("No patches found.");
                for (std::size_t i = 0; i < patch_rows.size(); ++i) {
                    PatchRow& r = patch_rows[i];
                    const bool other_eboot = eboot.ok && !r.p.eboot.empty() && r.p.eboot != eboot.sha;
                    const bool unknown_option = !r.p.option.empty() && !patch_manifest_option_known(r.p.option);
                    const bool fixed = !r.p.error.empty() || !r.p.enabled || other_eboot || unknown_option ||
                                       (r.p.option.empty() && !bare_key(r.p.name));
                    bool* on = patch_switch(r);
                    bool off = false;
                    ImGui::PushID(static_cast<int>(i));
                    ImGui::BeginDisabled(fixed);
                    ImGui::Checkbox(r.p.name.c_str(), fixed ? &off : on);
                    ImGui::EndDisabled();
                    ImGui::PopID();
                    ImGui::Indent();
                    ImGui::PushTextWrapPos(0.0f);
                    if (!r.p.description.empty()) ImGui::TextDisabled("%s", r.p.description.c_str());
                    const std::string why = !r.p.error.empty()        ? "Not read: " + r.p.error + "."
                                            : !r.p.enabled            ? "Turned off in its own file ([patch] enabled = false)."
                                            : other_eboot             ? "Made for another eboot: not applied."
                                            : unknown_option          ? "Follows " + r.p.option + ", which this bbhost does not know: not applied."
                                            : r.p.option.empty() && !bare_key(r.p.name) ? "Its name cannot be a bbhost.toml key, so it is always on."
                                                                                       : patch_note(r);
                    const std::string where = generic(fs::path(r.p.path).lexically_normal().string());
                    ImGui::TextDisabled("%s %s", why.c_str(), where.c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::Unindent();
                    ImGui::Spacing();
                }
                if (!patch_rows.empty()) {
                    auto set_all = [&](bool v) {
                        for (PatchRow& r : patch_rows) {
                            const bool other_eboot = eboot.ok && !r.p.eboot.empty() && r.p.eboot != eboot.sha;
                            if (!r.p.error.empty() || !r.p.enabled || other_eboot) continue;
                            if (!r.p.option.empty() && !patch_manifest_option_known(r.p.option)) continue;
                            if (r.p.option.empty() && !bare_key(r.p.name)) continue;
                            *patch_switch(r) = v;
                        }
                    };
                    if (ImGui::Button("All off (the game's code as it shipped)")) set_all(false);
                    ImGui::SameLine();
                    if (ImGui::Button("All on##patches")) set_all(true);
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Plugins", nullptr, pflags)) {
                plugin_ui_draw(plugin_ui, scale, ImGui::GetContentRegionAvail().y - footer);
                plugin_seen = true;
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        if (game_tab) ImGui::BeginChild("##game", ImVec2(0, ImGui::GetContentRegionAvail().y - footer));

        const float browse_w = ImGui::CalcTextSize("Browse...").x + ImGui::GetStyle().FramePadding.x * 2;
        auto path_row = [&](const char* label, int field, bool folder, const char* hint) {
            ImGui::TextUnformatted(label);
            ImGui::PushID(field);
            ImGui::SetNextItemWidth(-browse_w - ImGui::GetStyle().ItemSpacing.x);
            const bool changed = ImGui::InputTextWithHint("##p", hint, f[field].text, sizeof(f[field].text));
            ImGui::SameLine();
            if (ImGui::Button("Browse...")) {
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

        if (game_tab) {
        ImGui::SeparatorText("Game files");
        path_row("Game folder (the one that contains dvdroot_ps4)", kApp0, true, "C:/Games/Bloodborne/CUSA00900");
        status_line(app0_ok, "has dvdroot_ps4",
                    app0.empty()           ? "not set"
                    : !is_dir(app0)        ? "that folder does not exist"
                    : is_dir(app0 + "/../dvdroot_ps4") || fs::path(app0).filename() == "dvdroot_ps4"
                                           ? "pick the folder above dvdroot_ps4, not dvdroot_ps4 itself"
                                           : "no dvdroot_ps4 in it");
        if (app0_ok) {
            // Its version: the 1.09 update's files copied over it, or in an
            // update folder beside it (core/config.h), or the game stops when
            // it reads one. A warning, not a stop: Play stays on.
            static std::string seen, good;
            static App0Version ver;
            if (seen != app0) {
                seen = app0;
                ver = config_app0_version(app0);
                good = ver.update.empty() ? std::string("version 01.09 (the update's files are in it)")
                                          : "version 01.09, the update read from " + fs::path(ver.update).filename().string();
            }
            status_line(ver.app_ver == "01.09", good.c_str(),
                        ver.app_ver.empty() ? std::string("no sce_sys/param.sfo in it - is it the whole game folder?")
                                            : "version " + ver.app_ver +
                                                  (ver.update.empty() ? std::string()
                                                                      : " (in " + fs::path(ver.update).filename().string() + ")") +
                                                  " - copy the 1.09 update's files over this folder, or the game stops "
                                                  "while loading");
        }
        if (path_row("Eboot (the 1.09 update's eboot.bin, decrypted)", kEboot, false, "C:/Games/Bloodborne/eboot-109-decrypted.bin") ||
            (eboot.path != f[kEboot].str() && !ImGui::IsAnyItemActive())) {
            eboot = check_eboot(f[kEboot].str());
        }
        status_line(eboot.ok, "the 1.09 eboot",
                    f[kEboot].str().empty() ? "not set"
                    : !is_file(f[kEboot].str()) ? "that file does not exist"
                                                : "not the 1.09 eboot (SHA-256 " + eboot.sha.substr(0, 16) + "...)");
        path_row("Data folder (saves and caches; empty for the default)", kData, true, config_default_data_dir().c_str());

        ImGui::SeparatorText("Online");
        ImGui::SetNextItemWidth(-1);
        static const char* const kServerLabels[] = {kServers[0].label, kServers[1].label, "Custom"};
        ImGui::Combo("##server", &server, kServerLabels, 3);
        if (server == 2) {
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.6f);
            ImGui::InputTextWithHint("Host", "server.example.com", f[kHost].text, sizeof(f[kHost].text));
            ImGui::SetNextItemWidth(100 * scale);
            ImGui::InputText("Scheme", scheme, sizeof(scheme));
            ImGui::Checkbox("Check the server's certificate", &verify_tls);
            ImGui::Checkbox("Require an account (F10 > Account)", &require_account);
        }
        if (!cfg.profile.empty()) {
            ImGui::TextDisabled("This run uses %s, whose own settings may choose another server.", cfg.profile_config.c_str());
        }

        // The private server's account, as F10 > ACCOUNT in the game has it:
        // no password - this PC linked to the account signed into on the
        // website with Discord, or an account held by this PC with a
        // recovery code. The calls go to the server chosen above.
        ImGui::SeparatorText("Account");
        {
            const bool busy = host_account_busy();
            const std::string signed_in = host_account_signed_in();
            if (net::account_logged_in()) ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1.0f), "%s", signed_in.c_str());
            else ImGui::TextUnformatted(signed_in.c_str());
            const std::string outcome = host_account_outcome();
            if (!outcome.empty() && outcome != signed_in) ImGui::TextWrapped("%s", outcome.c_str());
            const std::string detail = host_account_detail();
            if (!detail.empty()) {
                ImGui::TextColored(ImVec4(0.95f, 0.80f, 0.40f, 1.0f), "%s", detail.c_str());
                ImGui::SameLine();
                if (ImGui::SmallButton("Copy")) {
                    // The code alone: the word after "Code" or "code".
                    std::string code = detail;
                    for (const char* lead : {"Code ", "Recovery code ", "Website code "}) {
                        if (detail.rfind(lead, 0) == 0) code = detail.substr(std::strlen(lead));
                    }
                    code = code.substr(0, code.find("  "));
                    SDL_SetClipboardText(code.c_str());
                }
            }
            ImGui::BeginDisabled(busy);
            if (ImGui::Button("Log in with Discord")) {
                apply_server();
                host_account_action(0, "", "");
            }
            ImGui::SameLine();
            ImGui::TextDisabled("opens the website: sign in there with Discord and approve the code");
            const float field_w = 220 * scale;
            ImGui::SetNextItemWidth(field_w);
            ImGui::InputTextWithHint("##acct_name", "Name", acct_name, sizeof(acct_name));
            ImGui::SameLine();
            if (ImGui::Button("Create account on this PC")) {
                apply_server();
                host_account_action(1, acct_name, "");
            }
            ImGui::SetNextItemWidth(field_w);
            ImGui::InputTextWithHint("##acct_code", "Recovery code", acct_code, sizeof(acct_code));
            ImGui::SameLine();
            if (ImGui::Button("Recover account")) {
                apply_server();
                host_account_action(2, acct_name, acct_code);
                acct_code[0] = 0;
            }
            ImGui::BeginDisabled(!net::account_logged_in());
            if (ImGui::Button("Website sign-in code")) {
                apply_server();
                host_account_action(3, "", "");
            }
            ImGui::EndDisabled();
            ImGui::EndDisabled();
            ImGui::SameLine();
            // Also stops a Discord link that is waiting for approval.
            ImGui::BeginDisabled(!busy && !net::account_logged_in());
            if (ImGui::Button(busy ? "Cancel" : "Sign out")) {
                apply_server();
                host_account_action(4, "", "");
            }
            ImGui::EndDisabled();
        }

        ImGui::SeparatorText("Start-up and display");
        ImGui::Checkbox("Fullscreen", &fullscreen);
        ImGui::SameLine(260 * scale);
        ImGui::Checkbox("Skip company logos", &skip_logos);
        ImGui::Checkbox("Debug camera", &debug_camera);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Lance McDonald's free camera: hold Interact (E) and press L3 (Left Ctrl). The debug menu's LOAD TEST crashes while it is on.");
        ImGui::SetNextItemWidth(200 * scale);
        static const char* const kCapLabels[] = {"30 fps", "60 fps", "90 fps", "Uncapped"};
        ImGui::Combo("Frame rate", &cap_choice, kCapLabels, 4);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The game's own speed. Uncapped steps it by the frame time: the display's refresh rate with V-Sync on, up to 240 fps off.");
        ImGui::SetNextItemWidth(200 * scale);
        ImGui::Combo("Render resolution", &resolution, kResolutions, static_cast<int>(sizeof(kResolutions) / sizeof(kResolutions[0])));
        ImGui::TextDisabled("F10 in the game has every other setting.");

        ImGui::SeparatorText("Updates");
        ImGui::Checkbox("Check for a newer bbhost at start", &check_updates);
        if (check_updates) {
            ImGui::TextDisabled("%s", updater::status().c_str());
            if (const std::string d = updater::detail(); !d.empty()) ImGui::TextDisabled("%s", d.c_str());
        }
        path_row("GitHub token file (only while the releases are private)", kToken, false, "");

        }
        if (game_tab) ImGui::EndChild();
        ImGui::Separator();
        ImGui::Checkbox("Show this window every time bbhost starts", &show_every_start);
        ImGui::TextDisabled("Or start bbhost with --setup to open it once.");
        ImGui::Spacing();
        const bool can_play = app0_ok && eboot.ok;
        ImGui::BeginDisabled(!can_play);
        const bool play = ImGui::Button("Play", ImVec2(140 * scale, 0)) || (test && !std::strcmp(test, "play") && ++frames > 30);
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Save", ImVec2(140 * scale, 0))) save();
        ImGui::SameLine();
        if (ImGui::Button("Quit", ImVec2(140 * scale, 0)) || (test && !std::strcmp(test, "quit") && ++frames > 30)) done = true;
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
        if (!can_play) {
            ImGui::SameLine();
            ImGui::TextDisabled("Play needs the game folder and the 1.09 eboot.");
        }
        if (!saved_note.empty()) ImGui::TextDisabled("%s", saved_note.c_str());
        if (play && can_play && save()) {
            result = LauncherResult::Play;
            done = true;
        }
        ImGui::End();

        ImGui::Render();
        SDL_SetRenderDrawColor(ren, 20, 20, 24, 255);
        SDL_SetRenderScale(
            ren,
            io.DisplayFramebufferScale.x,
            io.DisplayFramebufferScale.y
        );
        SDL_RenderClear(ren);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), ren);
        SDL_RenderPresent(ren);
    }

    if (plugin_seen) plugin_ui_mark_seen(plugin_ui);
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
    host_log("setup: %s", result == LauncherResult::Play ? "play" : "closed");
    return result;
}
