#include "host/japanese_font.h"
#include "host/launcher.h"

#include "bbhost_version.h"
#include "core/sha256.h"
#include "engine/addr.h"
#include "engine/patch_manifest.h"
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

#include <cctype>
#include <chrono>
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
    {"本番サーバー (thehuntersdream.com)", "thehuntersdream.com", "https", true, true, "https://thehuntersdream.com"},
    {"テストサーバー (dev.thehuntersdream.com)", "dev.thehuntersdream.com", "https", true, true, "https://dev.thehuntersdream.com"},
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
    const float scale = std::max(1.0f, SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay()));
    SDL_Window* win = SDL_CreateWindow("bbhost 日本語版の設定", static_cast<int>(980 * scale), static_cast<int>(760 * scale),
                                       SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
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
    ImFontConfig jp_font;
    jp_font.FontDataOwnedByAtlas = false;
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(kJapaneseFont),
        static_cast<int>(kJapaneseFontSize), 18.0f, &jp_font);
    io.IniFilename = nullptr;  // nothing written beside the exe
    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(scale);
    ImGui::GetStyle().FontScaleDpi = scale;
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
    bool tooling = cfg.tooling;
    // The F10 screen's start-up settings, through the same calls it uses.
    host_options_load();
    bool fullscreen = host_opt_index("window_mode") == 1;
    bool skip_logos = host_opt_get("skip_logos");
    bool debug_camera = host_opt_get("debug_camera");
    int frame_cap = host_opt_frame_cap();
    int resolution = host_opt_resolution_index();
    // In the setting's order (host/options.cpp): by size within each shape.
    static const char* const kResolutions[] = {"640x360",           "960x540",           "1280x720",          "1440x810",
                                               "1600x900",          "1920x1080",         "2560x1440",         "2880x1620",
                                               "3200x1800",         "3840x2160",         "960x600 (16:10, 拡大表示)",
                                               "1024x640 (16:10, 拡大表示)",             "1280x800 (16:10)",  "1920x1200 (16:10)",
                                               "2560x1600 (16:10)", "1280x960 (4:3)",    "1440x1080 (4:3)",   "2560x1080 (21:9)",
                                               "3440x1440 (21:9)",  "5120x2160 (21:9)",  "3840x1080 (32:9)",  "5120x1440 (32:9)"};
    // The game's frame rate (engine/frame_rate.h): 0 is uncapped.
    static const int kCaps[] = {30, 60, 90, 120, 0};
    constexpr int kCapCount = static_cast<int>(sizeof(kCaps) / sizeof(kCaps[0]));
    int cap_choice = kCapCount - 1;
    for (int k = 0; k < kCapCount; ++k) {
        if (kCaps[k] == frame_cap) cap_choice = k;
    }
    // The follow camera's tuning above 30 fps: the setting's index, 0 FPS++, 1 Original.
    int camera_tuning = host_opt_index("camera_tuning") == 1 ? 1 : 0;
    // The Shaders tab: the Shaders setting (0 HLSL, 1 Lifted), the caches'
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
    // The main tab's "Compile shaders now" starts the compile and shows the
    // Shaders tab, where its progress is.
    bool select_shaders_tab = false;
    // The PC enhancements, through the same calls: each is read when the
    // game starts (main.cpp, config_set_enhancements).
    struct Enhancement {
        const char* key;
        const char* label;
        const char* text;
        bool on;
    };
    Enhancement enhancements[] = {
        {"change_appearance", "狩人の夢の鏡",
         "工房の鏡で容姿の編集画面を開きます。未使用の「変装」機能です。", false},
        {"rebirth", "嘆きの祭壇で生まれ変わる",
         "ヤーナムの石を持っていると、出自の初期レベルに戻し、血の遺志を返還して能力値を振り直せます。終了時に確定か取り消しを選べます。",
         false},
        {"five_players", "終盤エリアの５人プレイ",
         "メンシスの悪夢、メルゴーの高楼、悪夢の辺境、DLCエリアで聖杯ダンジョンと同様に侵入者を２人まで許可します。全参加者の有効化が必要です。",
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
        if (o == "debug.free_camera") return &debug_camera;
        if (o == "loading.quick_reentry") return &quick_reentry;
        if (o == "streaming.all_post_processors") return &all_post_processors;
        return &r.on;
    };
    auto patch_note = [&](const PatchRow& r) -> std::string {
        const std::string& o = r.p.option;
        if (o == "startup.skip_intro") return "ゲームタブの「企業ロゴをスキップ」と同じ設定です。";
        if (o == "debug.free_camera") return "ゲームタブの「デバッグカメラ」と同じ設定です。";
        if (o == "loading.quick_reentry") return "bbhost.toml の [loading] の quick_reentry に保存します。";
        if (o == "streaming.all_post_processors") return "bbhost.toml の [streaming] の all_post_processors に保存します。";
        return "保存先: " + r.p.name + " (bbhost.toml の [patches])";
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
        e.push_back({"debug", "tooling", tooling ? "true" : "false"});
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
            saved_note = "書き込めません: " + file;
            return false;
        }
        for (PatchRow& r : patch_rows) r.was = r.on;
        quick_reentry_was = quick_reentry;
        all_post_processors_was = all_post_processors;
        host_opt_set_index("window_mode", fullscreen ? 1 : 0);
        host_opt_set("skip_logos", skip_logos);
        host_opt_set("debug_camera", debug_camera);
        host_opt_set_frame_cap(kCaps[cap_choice]);
        host_opt_set_index("camera_tuning", camera_tuning);
        host_opt_set_index("shaders", shaders);
        host_opt_set_resolution_index(resolution);
        for (const Enhancement& x : enhancements) host_opt_set(x.key, x.on);
        host_options_save_now();
        saved_note = "保存しました: " + file;
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

        ImGui::Text("bbhost %s - 設定", BBHOST_VERSION);
        ImGui::TextDisabled("%s に保存しました。すべての bbhost で共有されます。", config_user_file().c_str());
        if (!reason.empty()) ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.30f, 1.0f), "%s", reason.c_str());
        // While the Updates box is checked: one check, on a thread of its own
        // (host/updater.h), and a newer release's page opens in the browser.
        if (check_updates) updater::check_once();
        if (check_updates && updater::update_available()) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(ImVec4(0.95f, 0.80f, 0.40f, 1.0f), "bbhost %s が公開されています", updater::latest_tag().c_str());
            ImGui::SameLine();
            if (ImGui::Button("リリースページを開く")) updater::open_release_page();
        }
        ImGui::Separator();

        const std::string app0 = f[kApp0].str();
        const bool app0_ok = is_dir(app0 + "/dvdroot_ps4");
        // The tabs above the buttons: the game's settings, and the plugins.
        const float footer = ImGui::GetFrameHeightWithSpacing() * 4.2f;
        bool game_tab = false;
        // The compile's progress and the caches' state, which the Shaders tab
        // shows and the main tab's recommendation reads: the state again every
        // few seconds, and when a compile ends.
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
        const ImVec4 good(0.55f, 0.85f, 0.55f, 1.0f), warn(0.95f, 0.75f, 0.35f, 1.0f);
        auto start_compile = [&] {
            // The child reads the saved settings: save first.
            if (save()) {
                std::string err;
                compile_job.start(err);
            }
        };
        if (ImGui::BeginTabBar("##tabs")) {
            if (ImGui::BeginTabItem("ゲーム")) {
                game_tab = true;
                ImGui::EndTabItem();
            }
            // Additions to the game itself, for players who want it as it
            // shipped: the same three the game's System > PC Enhancements and
            // F10 > PC ENHANCEMENTS have.
            if (ImGui::BeginTabItem("PC 追加機能")) {
                ImGui::BeginChild("##enhancements", ImVec2(0, ImGui::GetContentRegionAvail().y - footer));
                ImGui::TextWrapped("ゲーム本体への追加機能です。初期設定ではすべて有効です。無効にすると原作の動作になります。");
                ImGui::TextDisabled("各設定はゲーム起動時に読み込まれます。プレイ中の変更は次回起動時に反映されます。");
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
                if (ImGui::Button("すべて無効（原作の動作）")) {
                    for (Enhancement& e : enhancements) e.on = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("すべて有効")) {
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
            if (ImGui::BeginTabItem("パッチ")) {
                ImGui::BeginChild("##patches", ImVec2(0, ImGui::GetContentRegionAvail().y - footer));
                ImGui::TextWrapped("ゲーム起動時にコードへ適用するパッチです。本体横の patches フォルダーと MOD の <mods>/patches を読み込みます。");
                ImGui::TextDisabled("異なる eboot 用のパッチは適用しません。");
                ImGui::Spacing();
                if (patch_rows.empty()) ImGui::TextDisabled("パッチがありません。");
                // Whether a row's switch can be changed here.
                auto switchable = [&](const PatchRow& r) {
                    const bool other_eboot = eboot.ok && !r.p.eboot.empty() && r.p.eboot != eboot.sha;
                    const bool unknown_option = !r.p.option.empty() && !patch_manifest_option_known(r.p.option);
                    return r.p.error.empty() && !other_eboot && !unknown_option && (!r.p.option.empty() || bare_key(r.p.name));
                };
                for (std::size_t i = 0; i < patch_rows.size(); ++i) {
                    PatchRow& r = patch_rows[i];
                    const bool other_eboot = eboot.ok && !r.p.eboot.empty() && r.p.eboot != eboot.sha;
                    const bool unknown_option = !r.p.option.empty() && !patch_manifest_option_known(r.p.option);
                    const bool fixed = !switchable(r);
                    bool* on = patch_switch(r);
                    bool off = false;
                    ImGui::PushID(static_cast<int>(i));
                    ImGui::BeginDisabled(fixed);
                    // Variants of one patch (a group) are one at a time: turning
                    // one on turns the others off.
                    if (ImGui::Checkbox(r.p.name.c_str(), fixed ? &off : on) && !fixed && *on && !r.p.group.empty()) {
                        for (PatchRow& o : patch_rows) {
                            if (&o != &r && o.p.group == r.p.group && switchable(o)) *patch_switch(o) = false;
                        }
                    }
                    ImGui::EndDisabled();
                    ImGui::PopID();
                    ImGui::Indent();
                    ImGui::PushTextWrapPos(0.0f);
                    if (!r.p.description.empty()) ImGui::TextDisabled("%s", r.p.description.c_str());
                    std::string why = !r.p.error.empty()        ? "読み込めません: " + r.p.error + "."
                                      : other_eboot             ? "別の eboot 用のため適用しません。"
                                      : unknown_option          ? "連動先: " + r.p.option + "。この bbhost では未対応のため適用しません。"
                                      : r.p.option.empty() && !bare_key(r.p.name) ? "名前が設定キーとして使えないため常に有効です。"
                                                                                 : patch_note(r);
                    if (!fixed && !r.p.enabled) why = "ここで選択するまで無効です。 " + why;
                    if (!fixed && !r.p.group.empty()) why = "種類: " + r.p.group + "。同時に有効にできるのは１つです。 " + why;
                    if (!fixed && r.p.gameplay)
                        why = "ゲーム内容を変更します。有効時はオンラインルールに記録され、血痕・手記・統計は未改造のプレイヤーと分離されます。 " + why;
                    const std::string where = generic(fs::path(r.p.path).lexically_normal().string());
                    ImGui::TextDisabled("%s %s", why.c_str(), where.c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::Unindent();
                    ImGui::Spacing();
                }
                if (!patch_rows.empty()) {
                    // All off, or each patch as its own file has it (a patch that
                    // says enabled = false off, the rest on); one that follows a
                    // setting keeps that setting's own default.
                    auto set_all = [&](bool defaults) {
                        for (PatchRow& r : patch_rows) {
                            if (!switchable(r) || (defaults && !r.p.option.empty())) continue;
                            *patch_switch(r) = defaults && r.p.enabled;
                        }
                    };
                    if (ImGui::Button("すべて無効（原作のコード）")) set_all(false);
                    ImGui::SameLine();
                    if (ImGui::Button("初期設定##patches")) set_all(true);
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            const ImGuiTabItemFlags sflags = select_shaders_tab ? ImGuiTabItemFlags_SetSelected : 0;
            select_shaders_tab = false;
            if (ImGui::BeginTabItem("シェーダー", nullptr, sflags)) {
                ImGui::BeginChild("##shaders", ImVec2(0, ImGui::GetContentRegionAvail().y - footer));
                if (recommend_compile) {
                    ImGui::TextColored(warn, "今すぐシェーダーをコンパイルすることを推奨します。");
                    ImGui::PushTextWrapPos(0.0f);
                    ImGui::TextUnformatted((advice.why + "。今すぐコンパイルする場合の所要時間: " + advice.time +
                                            "。実行しなければプレイ中にコンパイルします。")
                                               .c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::BeginDisabled(!app0_ok || !eboot.ok);
                    if (ImGui::Button("今すぐコンパイル##advice")) start_compile();
                    ImGui::EndDisabled();
                    ImGui::Separator();
                    ImGui::Spacing();
                }
                ImGui::TextWrapped("ゲームのシェーダーをこの PC で HLSL に変換し、DXC でコンパイルします。起動時は完成済みのキャッシュを使い、残りはバックグラウンドで処理して次回起動時に反映します。ここで事前にすべて処理することもできます。");
                ImGui::Spacing();
                ImGui::SetNextItemWidth(200 * scale);
                static const char* const kShaderModes[] = {"HLSL", "変換済み SPIR-V"};
                ImGui::BeginDisabled(cv.running);
                ImGui::Combo("シェーダー", &shaders, kShaderModes, 2);
                ImGui::EndDisabled();
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextDisabled("HLSL はこの PC の DXC でコンパイルします。変換済み SPIR-V は独自変換のみを使います。「保存」または「プレイ」で保存し、次回起動時に反映します。");
                ImGui::PopTextWrapPos();
                ImGui::SeparatorText("キャッシュ");
                ImGui::TextDisabled("%s", cache_state.dir.c_str());
                if (!cache_state.have_caches) {
                    ImGui::TextColored(warn, "シェーダーキャッシュがありません。初回起動時にすべてコンパイルします。");
                } else {
                    if (cache_state.translator_current) ImGui::TextColored(good, "この bbhost のシェーダー変換器で作成されています。");
                    else ImGui::TextColored(warn, "別バージョンの bbhost で作成されています。次回起動時に再コンパイルします。");
                    if (cache_state.device_current == 1) ImGui::TextColored(good, "現在のグラフィックスドライバー用です。");
                    else if (cache_state.device_current == 0) ImGui::TextColored(warn, "別の GPU またはドライバー用です。ドライバー側の処理をやり直します。");
                    else ImGui::TextDisabled("グラフィックスドライバーを確認中...");
                    ImGui::PushTextWrapPos(0.0f);
                    ImGui::TextDisabled("変換・DXC モジュール %.0f MiB、ドライバーのパイプライン %.0f MiB、描画履歴 %.1f + %.1f MiB。",
                                        cache_state.translation_bytes / 1048576.0, cache_state.pipeline_bytes / 1048576.0,
                                        cache_state.stage_manifest_bytes / 1048576.0, cache_state.pipeline_manifest_bytes / 1048576.0);
                    ImGui::PopTextWrapPos();
                }
                if (!cache_state.last.empty()) {
                    // shader-compile.txt's "key value" lines, as sentences.
                    std::string result, when, version, stages, hlsl, seconds, device, pipelines;
                    std::vector<std::string> failed, lift;
                    for (const std::string& l : cache_state.last) {
                        const std::size_t sp = l.find(' ');
                        const std::string k = l.substr(0, sp), v = sp == std::string::npos ? "" : l.substr(sp + 1);
                        if (k == "result") result = v;
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
                    ImGui::Text("前回のコンパイル: %s、%s (%s 秒)、bbhost %s", result == "done" ? "完了" : result == "cancelled" ? "中止" : result.c_str(), when.c_str(),
                                seconds.c_str(), version.c_str());
                    ImGui::Indent();
                    ImGui::PushTextWrapPos(0.0f);
                    ImGui::TextDisabled("%s: シェーダー %s、%s。", device.c_str(), stages.c_str(), hlsl.c_str());
                    if (!pipelines.empty() && pipelines.rfind("0 of 0", 0) != 0) ImGui::TextDisabled("パイプライン %s。", pipelines.c_str());
                    for (const std::string& r : failed) {
                        if (r.find("LDS use outside a compute shader") != std::string::npos)
                            ImGui::TextDisabled("単独ではコンパイルしません (%s)。LDS に書き込む頂点シェーダーは別のステージ内で動作するため、描画時にコンパイルします。",
                                                r.substr(r.rfind(' ') + 1).c_str());
                        else
                            ImGui::TextDisabled("失敗: %s", r.c_str());
                    }
                    for (const std::string& r : lift) ImGui::TextDisabled("変換済み SPIR-V を維持: %s", r.c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::Unindent();
                }
                ImGui::SeparatorText("今すぐコンパイル");
                if (!cv.running) {
                    ImGui::BeginDisabled(!app0_ok || !eboot.ok);
                    if (ImGui::Button("今すぐシェーダーをコンパイル")) start_compile();
                    ImGui::EndDisabled();
                    ImGui::SameLine();
                    ImGui::TextDisabled("ゲームで使うすべてのシェーダーを、この PC のドライバーで処理します。途中で中止できます。");
                } else {
                    if (ImGui::Button("キャンセル")) compile_job.cancel();
                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", cv.phase.c_str());
                }
                if (cv.running || cv.finished || cv.cancelled || !cv.error.empty()) {
                    const float frac = cv.total ? static_cast<float>(cv.done) / static_cast<float>(cv.total) : 0.0f;
                    char over[96];
                    std::snprintf(over, sizeof(over), "%llu / %llu", static_cast<unsigned long long>(cv.done),
                                  static_cast<unsigned long long>(cv.total));
                    ImGui::ProgressBar(frac, ImVec2(-1, 0), over);
                    if (cv.eta >= 0 && cv.running)
                        ImGui::Text("経過 %.0f 秒、残り約 %.0f 秒", cv.seconds, cv.eta);
                    else
                        ImGui::Text("%.0f 秒", cv.seconds);
                    ImGui::Text("DXC 採用 %llu、変換済み SPIR-V 維持 %llu、失敗 %llu", static_cast<unsigned long long>(cv.hlsl),
                                static_cast<unsigned long long>(cv.lift), static_cast<unsigned long long>(cv.failed));
                    if (cv.recipes_total)
                        ImGui::Text("過去のプレイと同梱リストのパイプライン: %llu / %llu",
                                    static_cast<unsigned long long>(cv.recipes_done), static_cast<unsigned long long>(cv.recipes_total));
                    if (cv.finished) ImGui::TextColored(good, "完了しました。次回起動時はキャッシュを使用します。");
                    if (cv.cancelled) ImGui::TextColored(warn, "中止しました。完成済みのキャッシュは保持し、残りはゲーム中に処理します。");
                    if (!cv.error.empty()) ImGui::TextColored(warn, "%s", cv.error.c_str());
                    for (const std::string& r : cv.reasons) ImGui::BulletText("%s", r.c_str());
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("プラグイン", nullptr, pflags)) {
                plugin_ui_draw(plugin_ui, scale, ImGui::GetContentRegionAvail().y - footer);
                plugin_seen = true;
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        if (game_tab) ImGui::BeginChild("##game", ImVec2(0, ImGui::GetContentRegionAvail().y - footer));

        const float browse_w = ImGui::CalcTextSize("参照...").x + ImGui::GetStyle().FramePadding.x * 2;
        auto path_row = [&](const char* label, int field, bool folder, const char* hint) {
            ImGui::TextUnformatted(label);
            ImGui::PushID(field);
            ImGui::SetNextItemWidth(-browse_w - ImGui::GetStyle().ItemSpacing.x);
            const bool changed = ImGui::InputTextWithHint("##p", hint, f[field].text, sizeof(f[field].text));
            ImGui::SameLine();
            if (ImGui::Button("参照...")) {
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
        if (recommend_compile) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(warn, "%s.", advice.why.c_str());
            ImGui::SameLine();
            ImGui::BeginDisabled(!app0_ok || !eboot.ok);
            if (ImGui::SmallButton("今すぐコンパイル##main")) {
                start_compile();
                select_shaders_tab = true;
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("%s（任意）", advice.time.c_str());
        }
        ImGui::SeparatorText("ゲームファイル");
        path_row("ゲームフォルダー (dvdroot_ps4 を含むフォルダー)", kApp0, true, "C:/Games/Bloodborne/CUSA03014");
        status_line(app0_ok, "dvdroot_ps4 を確認しました",
                    app0.empty()           ? "未設定"
                    : !is_dir(app0)        ? "フォルダーが存在しません"
                    : is_dir(app0 + "/../dvdroot_ps4") || fs::path(app0).filename() == "dvdroot_ps4"
                                           ? "dvdroot_ps4 自体ではなく、その親フォルダーを選んでください"
                                           : "dvdroot_ps4 がありません");
        if (app0_ok) {
            // Its version: the 1.09 update's files copied over it, or in an
            // update folder beside it (core/config.h), or the game stops when
            // it reads one. A warning, not a stop: Play stays on.
            static std::string seen, good;
            static App0Version ver;
            if (seen != app0) {
                seen = app0;
                ver = config_app0_version(app0);
                good = ver.update.empty() ? std::string("バージョン 01.09（更新データを確認しました）")
                                          : "バージョン 01.09、更新データの読み込み先: " + fs::path(ver.update).filename().string();
            }
            status_line(ver.app_ver == "01.09", good.c_str(),
                        ver.app_ver.empty() ? std::string("sce_sys/param.sfo がありません。ゲーム全体のフォルダーを選んでください")
                                            : "バージョン " + ver.app_ver +
                                                  (ver.update.empty() ? std::string()
                                                                      : "（" + fs::path(ver.update).filename().string() + " 内）") +
                                                  " - 1.09 の更新データを重ねてください。古いままではロード中に停止します");
        }
        if (path_row("eboot（更新 1.09 の復号済み eboot.bin）", kEboot, false, "C:/Games/Bloodborne/eboot-109-decrypted.bin") ||
            (eboot.path != f[kEboot].str() && !ImGui::IsAnyItemActive())) {
            eboot = check_eboot(f[kEboot].str());
        }
        status_line(eboot.ok, "1.09 の eboot を確認しました",
                    f[kEboot].str().empty() ? "未設定"
                    : !is_file(f[kEboot].str()) ? "ファイルが存在しません"
                                                : "1.09 の eboot ではありません (SHA-256 " + eboot.sha.substr(0, 16) + "...)");
        path_row("データフォルダー（セーブとキャッシュ。空欄で標準の場所）", kData, true, config_default_data_dir().c_str());

        ImGui::SeparatorText("オンライン");
        ImGui::SetNextItemWidth(-1);
        static const char* const kServerLabels[] = {kServers[0].label, kServers[1].label, "カスタム"};
        ImGui::Combo("##server", &server, kServerLabels, 3);
        if (server == 2) {
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.6f);
            ImGui::InputTextWithHint("ホスト", "server.example.com", f[kHost].text, sizeof(f[kHost].text));
            ImGui::SetNextItemWidth(100 * scale);
            ImGui::InputText("通信方式", scheme, sizeof(scheme));
            ImGui::Checkbox("サーバーの証明書を検証する", &verify_tls);
            ImGui::Checkbox("アカウントを必須にする (F10 > アカウント)", &require_account);
        }
        if (!cfg.profile.empty()) {
            ImGui::TextDisabled("今回は %s を使用しています。その設定で別のサーバーが指定される場合があります。", cfg.profile_config.c_str());
        }

        // The private server's account, as F10 > ACCOUNT in the game has it:
        // no password - this PC linked to the account signed into on the
        // website with Discord, or an account held by this PC with a
        // recovery code. The calls go to the server chosen above.
        ImGui::SeparatorText("アカウント");
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
                if (ImGui::SmallButton("コピー")) {
                    // The code alone: the word after "Code" or "code".
                    std::string code = detail;
                    for (const char* lead : {"コード ", "復旧コード ", "ウェブ用コード "}) {
                        if (detail.rfind(lead, 0) == 0) code = detail.substr(std::strlen(lead));
                    }
                    code = code.substr(0, code.find("  "));
                    SDL_SetClipboardText(code.c_str());
                }
            }
            ImGui::BeginDisabled(busy);
            if (ImGui::Button("Discord でログイン")) {
                apply_server();
                host_account_action(0, "", "");
            }
            ImGui::SameLine();
            ImGui::TextDisabled("ウェブサイトで Discord にログインし、コードを承認してください");
            const float field_w = 220 * scale;
            ImGui::SetNextItemWidth(field_w);
            ImGui::InputTextWithHint("##acct_name", "名前", acct_name, sizeof(acct_name));
            ImGui::SameLine();
            if (ImGui::Button("この PC でアカウントを作成")) {
                apply_server();
                host_account_action(1, acct_name, "");
            }
            ImGui::SetNextItemWidth(field_w);
            ImGui::InputTextWithHint("##acct_code", "復旧コード", acct_code, sizeof(acct_code));
            ImGui::SameLine();
            if (ImGui::Button("アカウントを復旧")) {
                apply_server();
                host_account_action(2, acct_name, acct_code);
                acct_code[0] = 0;
            }
            ImGui::BeginDisabled(!net::account_logged_in());
            if (ImGui::Button("ウェブサイトのログインコード")) {
                apply_server();
                host_account_action(3, "", "");
            }
            ImGui::EndDisabled();
            ImGui::EndDisabled();
            ImGui::SameLine();
            // Also stops a Discord link that is waiting for approval.
            ImGui::BeginDisabled(!busy && !net::account_logged_in());
            if (ImGui::Button(busy ? "キャンセル" : "ログアウト")) {
                apply_server();
                host_account_action(4, "", "");
            }
            ImGui::EndDisabled();
        }

        ImGui::SeparatorText("起動と画面");
        ImGui::Checkbox("全画面", &fullscreen);
        ImGui::SameLine(260 * scale);
        ImGui::Checkbox("企業ロゴをスキップ", &skip_logos);
        ImGui::Checkbox("デバッグカメラ", &debug_camera);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Lance McDonald の自由カメラ。調べる (E) を押しながら L3 (左 Ctrl) で切り替えます。有効時にデバッグメニューの LOAD TEST を使うとクラッシュします。");
        ImGui::SetNextItemWidth(200 * scale);
        static const char* const kCapLabels[] = {"30 fps", "60 fps", "90 fps", "120 fps", "無制限"};
        static_assert(sizeof(kCapLabels) / sizeof(kCapLabels[0]) == sizeof(kCaps) / sizeof(kCaps[0]), "a label for every rate");
        ImGui::Combo("フレームレート", &cap_choice, kCapLabels, kCapCount);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("ゲームの動作速度です。無制限では実時間に合わせます。垂直同期時は画面の更新頻度、無効時はワールド内で最大 240 fps です。");
        ImGui::SetNextItemWidth(200 * scale);
        ImGui::Combo("描画解像度", &resolution, kResolutions, static_cast<int>(sizeof(kResolutions) / sizeof(kResolutions[0])));
        ImGui::TextDisabled("その他の設定はゲーム中に F10 で開きます。");

        ImGui::SeparatorText("診断");
        ImGui::Checkbox("診断機能と詳細ログ", &tooling);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("定期ログ、停止検出、GPU 使用率 (GPU %%)、描画履歴を記録します。無効にすると負荷が減り、起動・警告・エラーのみ記録します。不具合の報告時は有効にしてください。");
        ImGui::PopTextWrapPos();

        ImGui::SeparatorText("更新");
        ImGui::Checkbox("起動時に新しい bbhost を確認する", &check_updates);
        if (check_updates) {
            ImGui::TextDisabled("%s", updater::status().c_str());
            if (const std::string d = updater::detail(); !d.empty()) ImGui::TextDisabled("%s", d.c_str());
        }
        path_row("GitHub トークンのファイル（非公開リリース用）", kToken, false, "");

        }
        if (game_tab) ImGui::EndChild();
        ImGui::Separator();
        ImGui::Checkbox("起動するたびにこの画面を表示", &show_every_start);
        ImGui::TextDisabled("一度だけ開くには --setup を付けて起動します。");
        ImGui::Spacing();
        const bool compiling = compile_job.running();
        const bool can_play = app0_ok && eboot.ok && !compiling;
        ImGui::BeginDisabled(!can_play);
        const bool play = ImGui::Button("プレイ", ImVec2(140 * scale, 0)) || (test && !std::strcmp(test, "play") && ++frames > 30);
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("保存", ImVec2(140 * scale, 0))) save();
        ImGui::SameLine();
        if (ImGui::Button("終了", ImVec2(140 * scale, 0)) || (test && !std::strcmp(test, "quit") && ++frames > 30)) done = true;
        if (test_link && ++frames > 30) {
            if (!test_created) {
                test_created = true;
                apply_server();
                host_account_action(0, "", "");
            } else if (host_account_detail().rfind("コード ", 0) == 0 || !host_account_busy()) {
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
        if (compiling) {
            ImGui::SameLine();
            ImGui::TextDisabled("コンパイル完了後にプレイできます（シェーダータブ）。");
        } else if (!can_play) {
            ImGui::SameLine();
            ImGui::TextDisabled("プレイにはゲームフォルダーと 1.09 の eboot が必要です。");
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
        // Test-only screenshot before presentation; never captures game data.
        if (test && done) {
            if (const char* shot = std::getenv("BBHOST_SETUP_SCREENSHOT")) {
                if (SDL_Surface* pixels = SDL_RenderReadPixels(ren, nullptr)) {
                    SDL_SaveBMP(pixels, shot);
                    SDL_DestroySurface(pixels);
                }
            }
        }
        SDL_RenderPresent(ren);
    }

    // Closing the window while a compile runs: it is told to stop and write
    // what it has; the destructor waits for it.
    compile_job.cancel();
    if (devices_thread.joinable()) devices_thread.join();
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
