// plugins/boss_rush on a host of our own: no game, no GPU. The plugin is
// loaded as the host loads plugins and plays one round (Ludwig, the Holy
// Blade) in a pretend game whose loads are slow - the title's, the travel's
// and a death's respawn each keep their loading screen up for thousands of
// frames - and which builds the new player early in each load, so
// on_world_load comes with the screen still up, as in the game. Checked:
// nothing moves the player while a loading screen is up or before the world
// has settled; the countdown, the travel and the fog warp come after it; a
// death's retry finds the fog wall again; a travel the game drops is asked
// again, a death meanwhile holding the asks through its respawn; a stop
// during a load sends the player home once its screen has gone; only a
// travel the game never begins - or one bbhost refuses with the player there
// - stops the run, with a message that says so; on an older host (plugin API
// 11) the SDK's own loading test does the same.
//
//   boss_rush_test <boss_rush.so>
#include "bbhost_plugin.h"

#include <dlfcn.h>
#include <unistd.h>

#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;
#define CHECK(c)                                                                \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::fprintf(stderr, "%s:%d: FAILED %s\n", __FILE__, __LINE__, #c); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

constexpr int kFps = 60;               // the pretend game's frames a second (boss_rush.test_frame_seconds)
constexpr int kLoadFrames = 3000;      // a slow load's screen once the player is built: 50 s
constexpr int kDarkFrames = 200;       // a load's screen before it builds the player
constexpr int kFadeFrames = 60;        // a lamp travel's fade before its screen comes
constexpr std::int32_t kLudwigLamp = 3402950, kDream = 2102950;
constexpr std::uint32_t kLudwigBlock = 0x22000000, kDreamBlock = 0x15000000, kLudwigDefeat = 13401800;
const float kFog[3] = {-407.93f, 1503.85f, -715.45f};  // 1 m outside Ludwig's fog wall, as the plugin stands the player

struct Call {
    std::uint64_t frame;
    bool loading, player;
    std::int32_t lamp;
    float pos[3];
};

struct Game {
    std::uint64_t frame = 0;
    bool loading = false, player = false;
    bool loading_seen = false;  // the host's answer: its test runs after the plugins' frame, so they see the frame before's
    std::uint32_t block = 0;
    float pos[3] = {}, yaw = 0;
    std::int64_t hp = 100, deaths = 3;
    std::map<std::uint32_t, int> flags;
    int fade = 0, dark = 0, lit = 0, dying = 0;  // frames left of each phase of a load
    std::int32_t return_point = kDream;          // where a load (a travel, a respawn) puts the player
    int drop = 0;                                // lamp travels the game drops
    bool refuse = false;                         // bbhost refuses lamp travel (travel off)
    std::vector<std::uint64_t> screen_ends;      // frames whose screen had just gone
    std::vector<Call> travels, warps, refused;
    std::vector<std::pair<Call, std::string>> messages;
    std::vector<std::string> log;
    std::map<std::string, std::string> config;
    std::string dir, config_value;
    void (*frame_fn)(void*) = nullptr;
    void* frame_user = nullptr;
    void (*load_fn)(std::uint32_t, void*) = nullptr;
    void* load_user = nullptr;
    void (*action_fn)(const char*, void*) = nullptr;
    void* action_user = nullptr;
    // What API 11's SDK reads instead: the NowLoading request and the
    // CSNowLoadingHelper's in-game bytes (+0x50, +0x51).
    std::uint8_t request = 0;
    alignas(8) std::uint8_t helper[0x58] = {};
};
Game G;

Call now_call(std::int32_t lamp = 0, const float* xyz = nullptr) {
    Call c{G.frame, G.loading, G.player, lamp, {}};
    if (xyz) std::memcpy(c.pos, xyz, sizeof c.pos);
    return c;
}

std::uint32_t block_of(std::int32_t rp) {
    return static_cast<std::uint32_t>(rp / 100000) << 24 | static_cast<std::uint32_t>((rp / 10000) % 10) << 16;
}

void fake_log(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    G.log.push_back(buf);
}
const char* fake_config(const char* key) {
    const auto it = G.config.find(key);
    G.config_value = it == G.config.end() ? "" : it->second;
    return G.config_value.c_str();
}
int fake_eboot_is_109() { return 1; }
std::uint64_t fake_guest_addr(std::uint64_t) { return 0; }  // no GameDataMan, no camera: the refill and the camera reset do nothing
int fake_read(std::uint64_t bn, void* out, std::size_t n) {
    if (bn == 0x596286b && n == 1) {
        std::memcpy(out, &G.request, 1);
        return 0;
    }
    if (bn == 0x593e8b8 && n == 8) {
        const std::uint64_t p = reinterpret_cast<std::uint64_t>(G.helper);
        std::memcpy(out, &p, 8);
        return 0;
    }
    return 1;
}
int fake_on_frame(void (*fn)(void*), void* user) {
    G.frame_fn = fn;
    G.frame_user = user;
    return 0;
}
void* fake_param_row(const char*, std::uint32_t, std::size_t*) { return nullptr; }
int fake_event_flag_get(std::uint32_t id, int* value) {
    *value = G.flags[id];
    return 0;
}
int fake_event_flag_set(std::uint32_t id, int value) {
    G.flags[id] = value ? 1 : 0;
    return 0;
}
int fake_player_stat_get(const char* name, std::int64_t* value) {
    if (!G.player) return 1;
    *value = std::strcmp(name, "hp") == 0 ? G.hp : std::strcmp(name, "deaths") == 0 ? G.deaths : 0;
    return 0;
}
int fake_player_position(float xyz[3], float* yaw) {
    if (!G.player) return 1;
    std::memcpy(xyz, G.pos, sizeof G.pos);
    if (yaw) *yaw = G.yaw;
    return 0;
}
int fake_player_block(std::uint32_t* block) {
    if (!G.player) return 1;
    *block = G.block;
    return 0;
}
int fake_player_warp(std::uint32_t, const float xyz[3], float yaw) {
    G.warps.push_back(now_call(0, xyz));
    if (!G.player) return 1;
    std::memcpy(G.pos, xyz, sizeof G.pos);
    G.yaw = yaw;
    return 0;
}
int fake_lamp_warp(std::int32_t rp) {
    if (!G.player || G.refuse) {  // as bbhost: no player (a load), or travel off
        G.refused.push_back(now_call(rp));
        return 1;
    }
    G.travels.push_back(now_call(rp));
    if (G.drop > 0) {
        --G.drop;  // queued, and the game never begins it
        return 0;
    }
    G.return_point = rp;
    G.fade = kFadeFrames;
    return 0;
}
int fake_call_guest(std::uint64_t, BbCallRegs* regs) {
    regs->ret = 0;
    return 0;
}
int fake_on_world_load(void (*fn)(std::uint32_t, void*), void* user) {
    G.load_fn = fn;
    G.load_user = user;
    return 0;
}
int fake_chr_list(BbChr*, std::size_t, std::size_t* count) {
    *count = 0;
    return 0;
}
const char* fake_plugin_dir(const char*) { return G.dir.c_str(); }
int fake_show_message(const char* text, float) {
    G.messages.push_back({now_call(), text});
    return 0;
}
int fake_on_action(const char*, void (*fn)(const char*, void*), void* user) {
    G.action_fn = fn;
    G.action_user = user;
    return 0;
}
int fake_loading_screen() { return G.loading_seen ? 1 : 0; }

void set_loading(bool on) {
    G.loading = on;
    G.request = on ? 1 : 0;
    G.helper[0x50] = on ? 0 : 1;  // the in-game update ran
    G.helper[0x51] = 0;
}

// A load: the screen comes and the old player goes; kDarkFrames later the new
// one is built in the return point's block (on_world_load, the screen still
// up), and the screen stays kLoadFrames more.
void begin_load() {
    set_loading(true);
    G.player = false;
    G.dark = kDarkFrames;
}

// One frame of the pretend game, then the plugin's: the host fires
// on_world_load first, then the frame callbacks.
void step() {
    ++G.frame;
    if (G.fade > 0 && --G.fade == 0) begin_load();
    if (G.dying > 0 && --G.dying == 0) begin_load();  // the death's respawn
    if (G.dark > 0 && --G.dark == 0) {
        G.player = true;
        G.block = block_of(G.return_point);
        G.pos[0] = kFog[0] + 30.0f;  // at the lamp, 30 m from the fog wall
        G.pos[1] = kFog[1];
        G.pos[2] = kFog[2] + 40.0f;
        G.hp = 100;
        G.lit = kLoadFrames;
        if (G.load_fn) G.load_fn(G.block, G.load_user);
    } else if (G.lit > 0 && --G.lit == 0) {
        set_loading(false);
        G.screen_ends.push_back(G.frame);
    }
    G.frame_fn(G.frame_user);
    G.loading_seen = G.loading;
}

bool logged(const std::string& s) {
    for (const std::string& l : G.log) {
        if (l.find(s) != std::string::npos) return true;
    }
    return false;
}
std::size_t count_logged(const std::string& s) {
    std::size_t n = 0;
    for (const std::string& l : G.log) n += l.find(s) != std::string::npos;
    return n;
}
const std::pair<Call, std::string>* message(const std::string& s) {
    for (const auto& m : G.messages) {
        if (m.second.find(s) != std::string::npos) return &m;
    }
    return nullptr;
}
std::size_t count_messages(const std::string& s) {
    std::size_t n = 0;
    for (const auto& m : G.messages) n += m.second.find(s) != std::string::npos;
    return n;
}
// The frame the n-th loading screen ended (0 when it has not).
std::uint64_t screen_end(std::size_t n) { return n < G.screen_ends.size() ? G.screen_ends[n] : 0; }
bool near_fog(const float* p) {
    return std::fabs(p[0] - kFog[0]) < 0.01f && std::fabs(p[1] - kFog[1]) < 0.01f && std::fabs(p[2] - kFog[2]) < 0.01f;
}
// Runs frames until done() or `limit` more frames.
template <typename F>
bool run_until(F done, std::uint64_t limit) {
    for (const std::uint64_t end = G.frame + limit; G.frame < end;) {
        if (done()) return true;
        step();
    }
    return done();
}

// Nothing moved the player, or asked to, inside a loading screen or with no
// player.
void check_never_inside_a_screen() {
    for (const Call& c : G.travels) CHECK(!c.loading && c.player);
    for (const Call& c : G.warps) CHECK(!c.loading && c.player);
}

struct Loaded {
    void* handle = nullptr;
};

// A fresh copy of the plugin (its own globals), loaded and started on the
// pretend host with `config`, the title showing.
Loaded start(const char* so, const fs::path& dir, const std::string& name, std::uint32_t api_version,
             std::map<std::string, std::string> config) {
    G = Game{};
    G.dir = (dir / name).string();
    fs::create_directories(G.dir);
    G.config = std::move(config);
    G.config["boss_rush.test_frame_seconds"] = std::to_string(1.0 / kFps);
    // The save: Ludwig beaten once (its defeat flag and "met" on), and an
    // unrelated flag in the family left on.
    G.flags[kLudwigDefeat] = 1;
    G.flags[kLudwigDefeat + 1] = 1;
    G.flags[kLudwigDefeat + 5] = 1;
    const fs::path copy = dir / (name + ".so");
    fs::copy_file(so, copy, fs::copy_options::overwrite_existing);
    Loaded l;
    l.handle = dlopen(copy.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!l.handle) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        ++g_failures;
        return l;
    }
    auto image = reinterpret_cast<int (*)(const BbHostApi*)>(dlsym(l.handle, "bb_plugin_image"));
    static BbHostApi api;
    std::memset(&api, 0, sizeof api);
    api.version = api_version;
    // API 11's table ends before loading_screen.
    api.size = api_version >= 12 ? sizeof api : static_cast<std::uint32_t>(offsetof(BbHostApi, loading_screen));
    api.log = fake_log;
    api.config = fake_config;
    api.eboot_is_109 = fake_eboot_is_109;
    api.guest_addr = fake_guest_addr;
    api.read = fake_read;
    api.on_frame = fake_on_frame;
    api.param_row = fake_param_row;
    api.event_flag_get = fake_event_flag_get;
    api.event_flag_set = fake_event_flag_set;
    api.player_stat_get = fake_player_stat_get;
    api.player_position = fake_player_position;
    api.player_block = fake_player_block;
    api.player_warp = fake_player_warp;
    api.lamp_warp = fake_lamp_warp;
    api.call_guest = fake_call_guest;
    api.on_world_load = fake_on_world_load;
    api.chr_list = fake_chr_list;
    api.plugin_dir = fake_plugin_dir;
    api.show_message = fake_show_message;
    api.on_action = fake_on_action;
    if (api_version >= 12) api.loading_screen = fake_loading_screen;
    CHECK(image && image(&api) == 0);
    CHECK(G.frame_fn && G.load_fn);
    if (!G.frame_fn || !G.load_fn) return l;
    // The title, then the first load (from the save: the Hunter's Dream).
    for (int i = 0; i < 100; ++i) step();
    G.return_point = kDream;
    begin_load();
    return l;
}

void finish(Loaded& l, const char* name) {
    if (g_failures) {
        std::fprintf(stderr, "--- %s: the log\n", name);
        for (const std::string& s : G.log) std::fprintf(stderr, "  %s\n", s.c_str());
        for (const auto& m : G.messages)
            std::fprintf(stderr, "  [frame %llu%s] message: %s\n", static_cast<unsigned long long>(m.first.frame),
                         m.first.loading ? ", loading" : "", m.second.c_str());
    }
    if (l.handle) dlclose(l.handle);
}

std::map<std::string, std::string> one_round() {
    return {{"boss_rush.auto_start", "true"}, {"boss_rush.start_delay", "1"}, {"boss_rush.order", "story"},
            {"boss_rush.first", "15"},       {"boss_rush.rounds", "1"},      {"boss_rush.dlc", "true"}};
}

// Start by itself on a slow first load; the travel's slow load; a death and
// its slow respawn; the kill and the way home.
void slow_loads(const char* so, const fs::path& dir) {
    const int failures = g_failures;
    Loaded l = start(so, dir, "slow-loads", BB_PLUGIN_API_VERSION, one_round());
    if (!G.frame_fn) return;
    // The title's load: the player is built inside it (on_world_load) and
    // its screen stays up 50 s. The run waits for it to go.
    CHECK(run_until([] { return screen_end(0) != 0; }, 10000));
    CHECK(!message("ボスラッシュ: 1 体"));
    CHECK(G.travels.empty());
    // The countdown (1 s), seen in the world, and the travel once the world
    // has been up 3 s.
    CHECK(run_until([] { return !G.travels.empty(); }, 2000));
    const auto* begun = message("ボスラッシュ: 1 体 - 開始まで 1 秒");
    CHECK(begun && !begun->first.loading && begun->first.frame > screen_end(0));
    CHECK(!G.travels.empty() && G.travels[0].lamp == kLudwigLamp && G.travels[0].frame >= screen_end(0) + 3 * kFps);
    CHECK(G.flags[kLudwigDefeat] == 0 && G.flags[kLudwigDefeat + 1] == 1 && G.flags[kLudwigDefeat + 5] == 0);  // woken, "met"
    // The travel's load: the arrival comes inside it; the fog warp after it.
    CHECK(run_until([] { return logged("boss_rush: arrived: 聖剣のルドウイーク"); }, 2000));
    CHECK(G.loading && G.warps.empty() && G.block == kLudwigBlock);
    CHECK(run_until([] { return logged("at 聖剣のルドウイーク's fog wall"); }, kLoadFrames + 500));
    CHECK(!G.warps.empty() && G.warps[0].frame == screen_end(1) + 1 && near_fog(G.warps[0].pos));
    const auto* round = message("ラウンド 1/1 - 聖剣のルドウイーク");
    CHECK(round && !round->first.loading && round->first.frame > screen_end(1));
    // A death at the fog: the respawn's slow load puts the player back at
    // the lamp (on_world_load in the boss's block, inside the screen), and
    // the fog warp comes once it has gone.
    for (int i = 0; i < 120; ++i) step();
    G.hp = 0;
    ++G.deaths;
    G.dying = 240;
    const std::size_t warps_before = G.warps.size();
    CHECK(run_until([] { return logged("死亡後に復帰"); }, 1000));
    CHECK(G.loading && G.warps.size() == warps_before);
    CHECK(run_until([] { return count_logged("at 聖剣のルドウイーク's fog wall") == 2; }, kLoadFrames + 500));
    CHECK(G.warps.size() > warps_before && G.warps[warps_before].frame == screen_end(2) + 1 && near_fog(G.warps[warps_before].pos));
    const auto* again = message("再挑戦 - 聖剣のルドウイーク");
    CHECK(again && !again->first.loading && again->first.frame > screen_end(2));
    // The kill, ten seconds to pick up the drops, then the result and the
    // travel home; the save's flags as they were.
    for (int i = 0; i < 300; ++i) step();
    G.flags[kLudwigDefeat] = 1;
    CHECK(run_until([] { return logged("聖剣のルドウイーク defeated in"); }, 10));
    const std::uint64_t killed = G.frame;
    CHECK(run_until([] { return G.travels.size() == 2; }, 20 * kFps));
    CHECK(G.travels.size() == 2 && G.travels[1].lamp == kDream && G.travels[1].frame >= killed + 10 * kFps);
    const auto* done = message("ボスラッシュ完了 - ");
    CHECK(done && done->second.find(", 1 回死亡") != std::string::npos);
    std::ifstream results(G.dir + "/results.txt");
    std::stringstream text;
    text << results.rdbuf();
    CHECK(text.str().find("ボスラッシュ完了 - ") != std::string::npos && text.str().find("  聖剣のルドウイーク: ") != std::string::npos);
    CHECK(G.flags[kLudwigDefeat] == 1 && G.flags[kLudwigDefeat + 1] == 1 && G.flags[kLudwigDefeat + 5] == 1 &&
          G.flags[kLudwigDefeat + 2] == 0);
    CHECK(run_until([] { return screen_end(3) != 0; }, kFadeFrames + kDarkFrames + kLoadFrames + 10));
    CHECK(G.block == kDreamBlock);
    // Each travel asked once, nothing inside a screen, nothing refused, no
    // failure said.
    CHECK(G.travels.size() == 2 && G.refused.empty());
    check_never_inside_a_screen();
    CHECK(!message("stopped") && !message("travel is off") && !message("could not"));
    CHECK(!logged("again") && !logged("could not"));
    finish(l, "slow loads");
    std::printf("slow loads: %s\n", g_failures == failures ? "ok" : "FAILED");
}

// The game drops the first travel: asked again 10 s of the world later, it
// lands; nothing said.
void one_dropped(const char* so, const fs::path& dir, std::uint32_t api_version, const char* name) {
    const int failures = g_failures;
    Loaded l = start(so, dir, name, api_version, one_round());
    if (!G.frame_fn) return;
    G.drop = 1;
    CHECK(run_until([] { return logged("at 聖剣のルドウイーク's fog wall"); }, 20000));
    CHECK(G.travels.size() == 2 && G.travels[0].lamp == kLudwigLamp && G.travels[1].lamp == kLudwigLamp);
    CHECK(G.travels.size() == 2 && G.travels[1].frame >= G.travels[0].frame + 10 * kFps &&
          G.travels[1].frame <= G.travels[0].frame + 10 * kFps + 2);
    CHECK(logged("the travel to 聖剣のルドウイーク's arena 移動が始まりませんでした; again"));
    CHECK(!G.warps.empty() && G.warps[0].frame == screen_end(1) + (api_version >= 12 ? 1 : 0) && near_fog(G.warps[0].pos));
    check_never_inside_a_screen();
    CHECK(!message("stopped") && G.refused.empty());
    finish(l, name);
    std::printf("%s: %s\n", name, g_failures == failures ? "ok" : "FAILED");
}

// The game drops the first travel and the player dies before it is asked
// again: the respawn's slow load holds the asks; the next comes 3 s of the
// world after its screen (and 10 s after the first), and lands.
void death_while_travelling(const char* so, const fs::path& dir) {
    const int failures = g_failures;
    Loaded l = start(so, dir, "death-while-travelling", BB_PLUGIN_API_VERSION, one_round());
    if (!G.frame_fn) return;
    G.drop = 1;
    CHECK(run_until([] { return G.travels.size() == 1; }, 10000));
    for (int i = 0; i < 2 * kFps; ++i) step();
    G.hp = 0;
    ++G.deaths;
    G.dying = 120;  // the respawn: the Hunter's Dream again, its screen up 50 s
    CHECK(run_until([] { return screen_end(1) != 0; }, 120 + kDarkFrames + kLoadFrames + 10));
    CHECK(G.travels.size() == 1);
    CHECK(run_until([] { return G.travels.size() == 2; }, 20 * kFps));
    CHECK(G.travels.size() == 2 && G.travels[1].frame >= screen_end(1) + 3 * kFps &&
          G.travels[1].frame >= G.travels[0].frame + 10 * kFps);
    CHECK(logged("the travel to 聖剣のルドウイーク's arena 別のマップに到着しました; again"));
    CHECK(run_until([] { return logged("at 聖剣のルドウイーク's fog wall"); }, 10000));
    check_never_inside_a_screen();
    CHECK(!message("ボスラッシュ 停止") && G.refused.empty());
    finish(l, "death while travelling");
    std::printf("death while travelling: %s\n", g_failures == failures ? "ok" : "FAILED");
}

// Started from the menu in the world, stopped from it while the travel's
// load is up: the save's flags back at once, the travel home asked only once
// the screen has gone and the world has run 3 s.
void stopped_in_a_load(const char* so, const fs::path& dir) {
    const int failures = g_failures;
    auto config = one_round();
    config["boss_rush.auto_start"] = "false";
    Loaded l = start(so, dir, "stopped-in-a-load", BB_PLUGIN_API_VERSION, config);
    if (!G.frame_fn) return;
    CHECK(run_until([] { return screen_end(0) != 0; }, 10000));
    for (int i = 0; i < 5 * kFps; ++i) step();
    CHECK(G.travels.empty() && G.action_fn);
    if (!G.action_fn) return;
    G.action_fn("start", G.action_user);
    CHECK(message("ボスラッシュ: 1 体 - 開始まで 1 秒"));
    CHECK(run_until([] { return !G.travels.empty(); }, 3 * kFps));
    CHECK(G.travels.size() == 1 && G.travels[0].frame >= G.messages.back().first.frame + kFps);
    CHECK(run_until([] { return logged("boss_rush: arrived: 聖剣のルドウイーク"); }, 2000));
    G.action_fn("stop", G.action_user);
    CHECK(message("ボスラッシュ 停止") && G.flags[kLudwigDefeat] == 1 && G.flags[kLudwigDefeat + 5] == 1);
    CHECK(run_until([] { return G.travels.size() == 2; }, kLoadFrames + 10 * kFps));
    CHECK(G.travels.size() == 2 && G.travels[1].lamp == kDream && G.travels[1].frame >= screen_end(1) + 3 * kFps);
    CHECK(G.warps.empty());  // stopped before the fog warp
    check_never_inside_a_screen();
    CHECK(!message("ボスラッシュ停止:") && G.refused.empty());
    finish(l, "stopped in a load");
    std::printf("stopped in a load: %s\n", g_failures == failures ? "ok" : "FAILED");
}

// The game never begins the travel: five asks, 10 s of the world apart, then
// the run stops and says why, with the save's flags put back. Once.
void never_begun(const char* so, const fs::path& dir) {
    const int failures = g_failures;
    Loaded l = start(so, dir, "never-begun", BB_PLUGIN_API_VERSION, one_round());
    if (!G.frame_fn) return;
    G.drop = 1000;
    CHECK(run_until([] { return message("ボスラッシュ停止:") != nullptr; }, 20000));
    const auto* m = message("ボスラッシュ停止:");
    CHECK(m && m->second == "ボスラッシュ停止: ゲームが移動を開始しませんでした / 聖剣のルドウイークの闘技場");
    CHECK(G.travels.size() == 5);
    for (std::size_t i = 1; i < G.travels.size(); ++i) CHECK(G.travels[i].frame - G.travels[i - 1].frame >= 10 * kFps);
    CHECK(m && G.travels.size() == 5 && m->first.frame >= G.travels[4].frame + 10 * kFps);
    CHECK(G.flags[kLudwigDefeat] == 1 && G.flags[kLudwigDefeat + 1] == 1 && G.flags[kLudwigDefeat + 5] == 1);
    for (int i = 0; i < 60 * kFps; ++i) step();
    CHECK(G.travels.size() == 5 && count_messages("ボスラッシュ停止:") == 1);
    CHECK(logged("ゲームが移動を開始しませんでした to 聖剣のルドウイーク's arena (lamp 3402950, asked 5 times); the run stops"));
    CHECK(logged("boss_rush: 11 flags back as the save had them") && !logged("read back otherwise"));  // the defeat family and +3000
    check_never_inside_a_screen();
    finish(l, "never begun");
    std::printf("never begun: %s\n", g_failures == failures ? "ok" : "FAILED");
}

// bbhost refuses the travel with the player there (travel off): said at
// once, and the run stops.
void refused(const char* so, const fs::path& dir) {
    const int failures = g_failures;
    Loaded l = start(so, dir, "refused", BB_PLUGIN_API_VERSION, one_round());
    if (!G.frame_fn) return;
    G.refuse = true;
    CHECK(run_until([] { return message("ボスラッシュ停止:") != nullptr; }, 20000));
    const auto* m = message("ボスラッシュ停止:");
    CHECK(m && m->second == "ボスラッシュ停止: bbhost が移動を拒否しました / 聖剣のルドウイークの闘技場");
    CHECK(G.refused.size() == 1 && !G.refused[0].loading && G.refused[0].player);
    for (int i = 0; i < 30 * kFps; ++i) step();
    CHECK(G.refused.size() == 1 && count_messages("ボスラッシュ停止:") == 1);
    CHECK(G.flags[kLudwigDefeat] == 1 && G.flags[kLudwigDefeat + 5] == 1);
    finish(l, "refused");
    std::printf("refused: %s\n", g_failures == failures ? "ok" : "FAILED");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: boss_rush_test <boss_rush.so>\n");
        return 2;
    }
    const fs::path dir = fs::temp_directory_path() / ("boss-rush-test-" + std::to_string(getpid()));
    fs::create_directories(dir);
    slow_loads(argv[1], dir);
    one_dropped(argv[1], dir, BB_PLUGIN_API_VERSION, "one-dropped");
    one_dropped(argv[1], dir, 11, "one-dropped-api11");
    death_while_travelling(argv[1], dir);
    stopped_in_a_load(argv[1], dir);
    never_begun(argv[1], dir);
    refused(argv[1], dir);
    fs::remove_all(dir);
    std::printf("%s\n", g_failures ? "FAILED" : "ok");
    return g_failures ? 1 : 0;
}
