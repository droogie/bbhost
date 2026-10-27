// Boss rush: an official bbhost plugin (docs/plugins.md).
//
// The game's bosses one after another: each round wakes a boss (its defeat
// flag cleared), carries the player to a lamp of its map and on to just
// outside its fog wall, and waits for the defeat flag to come back on. A
// death retries the round, from the fog wall again. When the last one falls,
// the time and the deaths are on screen and in
// <data>/plugins/boss_rush/results.txt, every flag the run changed is put
// back as the save had it, and the player is back in the Hunter's Dream.
//
// Start, skip and stop it from the in-game plugin menu (its settings are
// there and in the plugin manager at launch; the file form:)
//   [plugins]
//   boss_rush = true
//   [boss_rush]
//   order = "story"     # or "random" (seeded by `seed`)
//   seed = ""
//   dlc = true          # The Old Hunters' bosses
//   chalice = false     # and the ten only the chalice dungeons have, in their fixed dungeons
//   start_delay = 10    # seconds from Start to the first round (in the world)
//   auto_start = false  # begin once the first load's screen has gone, without the menu
//   first = 1           # start at the Nth boss of the order (practice one: first = N, rounds = 1)
//   rounds = 0          # only N bosses (0: all)
//   awake = true        # the fog wall stands as on a return; false: the first time's entrances
//   refill = true       # blood vials and quicksilver bullets back to the most after a death
//
// Play it on a save of its own: bosses give their rewards again, and while a
// run is under way the save holds woken bosses (restore.txt puts them back if
// the game is closed mid-run).
//
// The player is moved only in the world: each travel and fog warp waits for
// the load's screen to go (plugin API 12, loading_screen), and the waits are
// seconds, not frames (clocks()), so a slow load or a loading screen that
// flips hundreds of times a second changes nothing. A travel the game drops
// is asked again; one it will not take stops the run, and the message says so.
//
// Every boss's defeat flag and arena lamp come from the game's own data: the
// boss fights' event scripts (the health bar's NpcName and its entity) and
// the lamps each defeat lights (tools/mapval_lamps.py's "requires" column);
// the fog walls from the same scripts and the maps' layouts.
#include "bbhost/sdk.hpp"
#include "bbhost/engine/sprj/camera.hpp"
#include "bbhost/engine/sprj/game_data_man.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN | BB_PLUGIN_GAMEPLAY,
    "boss_rush",
    "ボスラッシュ",
    "1.1.1",
    "bbhost",
    "ボスと連続で戦い、時間を計測します。死亡時はそのラウンドを再挑戦し、終了時はボスの状態を元に戻します。",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, nullptr, "ゲーム内のプラグインメニューから開始", "専用セーブを使ってください。ボス報酬を再び受け取ります。"},
    {BB_OPT_ACTION, "start", "ボスラッシュ開始", "現在地から数秒後に開始します。"},
    {BB_OPT_ACTION, "skip", "このラウンドをスキップ"},
    {BB_OPT_ACTION, "stop", "停止", "挑戦を終了し、セーブ内のボスの状態を元に戻します。"},
    {BB_OPT_HEADING, nullptr, "挑戦の設定", "挑戦開始時に読み込みます。"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "order", "順番", nullptr, "story", 0, 0, "story|random"},
    {BB_OPT_TEXT | BB_OPT_LIVE, "seed", "ランダム順のシード", nullptr, ""},
    {BB_OPT_BOOL | BB_OPT_LIVE, "dlc", "DLC のボス", "そのマップに入れないセーブではスキップします。", "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "chalice", "聖杯ダンジョンのボス",
     "他のボスの後に、固定聖杯の固有ボス10体と戦います。祭壇の枠を一時使用し、終了後に戻します。", "false"},
    {BB_OPT_INT | BB_OPT_LIVE, "first", "開始するボス", "1 は最初のボスです。ラウンド数を 1 にすると１体を練習できます。", "1", 1, 29},
    {BB_OPT_INT | BB_OPT_LIVE, "rounds", "ラウンド数", "0 で全ボス。", "0", 0, 29},
    {BB_OPT_BOOL | BB_OPT_LIVE, "awake", "ボスの初回演出をスキップ", "再戦時と同様に霧の前から開始します。無効では初回の演出を再生します。", "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "refill", "死亡時に輸血液と水銀弾を補充", "無効では原作と同じく保管箱から補充します。", "true"},
    {BB_OPT_BOOL, "auto_start", "自動開始", "ワールドの初回読み込み時にメニュー操作なしで開始します。", "false"},
    {BB_OPT_INT | BB_OPT_LIVE, "start_delay", "最初のラウンドまでの秒数", nullptr, "10", 3, 60},
    {BB_OPT_END}};

namespace {

struct Boss {
    const char* name;
    std::uint32_t defeat_flag;  // set by the game when it dies
    std::int32_t lamp;          // a lamp in its arena's block that no boss gates (ReturnPointParam row: lamp entity + 1000)
    bool dlc;
    std::int32_t name_id;       // its health bar's NpcName id, which its NpcParam rows carry as nameId
    std::int32_t alt_name_id;   // another nameId its characters carry (0: none), for the test kill
    float fog[3];               // its fog wall: the map object its defeat event switches off (MSB position)
    float fog_yaw;              // that object's rotation, degrees: it faces into the arena
    std::uint8_t met = 2;       // defeat flag + met: "fought before", which the fog's event waits for (1 for the DLC's)
    std::uint32_t set_flags[2] = {};  // more flags the round turns on, put back after the run: a door on the way, a first visit's cutscene
    std::int32_t keep_out[2] = {};    // characters (entity ids) switched off for the round: NPCs a save's quests put in the arena
    std::uint32_t dungeon = 0;        // a chalice boss: its fixed dungeon, by map id (m29_10_90_00: 0x1d0a5a00)
    std::uint32_t begun = 0;          // a chalice boss: its "the fight has begun" flag, which stays on
};

// Story order (the usual first playthrough's). Each round travels to the
// block's first lamp - outside the arena, so neither the arrival nor a death's
// respawn lands in the fight - and from there to just outside the fog wall,
// facing it: the player walks in, and the game stages the fight as it does on
// any return (the fog's own event wakes the boss). The fog walls are the
// objects each defeat event switches off (2005[3]), whose action button
// (3[24]) turns the player toward the arena's "entry target" region. A
// thing at yaw y faces (-sin y, -cos y) - the player as the warp sets it and
// player_position reads it, and the fog objects, which all 19 face into their
// arenas. Positions are the MSB's frame, which the warp takes as it is.
const Boss kBosses[] = {
    {"聖職者の獣", 12411700, 2412951, false, 500000, 0, {-154.14f, -27.02f, 35.16f}, -135.0f},
    {"ガスコイン神父", 12411800, 2412951, false, 271000, 0, {-50.18f, -40.82f, 71.02f}, -135.0f},
    {"血に渇いた獣", 12301800, 2302950, false, 209000, 0, {-87.654f, -126.44f, 13.512f}, 129.825f},
    // The Grand Cathedral is where Eileen's quest ends (m24_00's event 12400650
    // and its flags 1360-1379): her placement there (2400902) and the hunter
    // she comes to fight (2400901) are switched off.
    {"教区長エミーリア", 12401800, 2402950, false, 502000, 0, {51.51f, 35.71f, 311.57f}, -150.0f, 2, {}, {2400901, 2400902}},
    {"ヘムウィックの魔女", 12201800, 2202950, false, 210000, 0, {-329.75f, 2.466f, 698.21f}, 180.0f},
    {"黒獣パール", 12301700, 2302950, false, 508000, 0, {135.444f, -119.83f, -85.772f}, 113.5f},
    {"ヤーナムの影", 12701800, 2702950, false, 212010, 0, {-322.923f, -185.868f, 494.393f}, 44.779f},
    // Rom's fog stands in the doorway to the lake the player drops into: the
    // door (3201120) is opened as its own event keeps it (m32_00's 13200040,
    // slot 0: its flag, and 13200120, the opening's).
    {"白痴の蜘蛛、ロマ", 13201800, 3202950, false, 510000, 0, {-452.874f, -175.03f, 392.65f}, 70.0f, 2, {13200040, 13200120}},
    {"再誕者", 12801800, 2802950, false, 507000, 0, {381.167f, -122.49f, -215.583f}, -45.0f},
    // A save's first time in Cainhurst plays the carriage's arrival (m25_00's
    // event 12500000, which its own flag ends) - over the round's first
    // seconds.
    {"殉教者ローゲリウス", 12501800, 2502950, false, 232000, 0, {33.93f, 112.32f, -330.56f}, -35.0f, 2, {12500000}},
    {"アメンドーズ", 13301800, 3302950, false, 512000, 0, {-47.183f, 1461.853f, -69.041f}, 112.0f},
    {"星界からの使者", 12421700, 2422950, false, 257000, 0, {-5.47f, 55.336f, 293.41f}, -60.0f},
    {"星の娘、エーブリエタース", 12421800, 2422950, false, 251000, 0, {94.0f, 4.2f, 387.0f}, -150.741f},
    // The Wet Nurse (entity 2600802) is made when the player walks into her
    // arena; 515000/516000 are the doubles that come with her.
    {"メルゴーの乳母", 12601800, 2602950, false, 551000, 0, {122.14f, 1125.184f, -37.839f}, -90.0f},
    {"聖剣のルドウイーク", 13401800, 3402950, true, 451000, 0, {-407.93f, 1503.8f, -716.45f}, 0.0f, 1},
    // The Living Failures' fog stands in the balcony's doorway (door 3501120,
    // m35_00's 13501200 slot 0, and 13504220), as Rom's does.
    {"失敗作たち", 13501850, 3502950, true, 403000, 0, {-378.54f, 1592.9f, -824.55f}, 90.0f, 1, {13501200, 13504220}},
    {"時計塔のマリア", 13501800, 3502950, true, 452000, 0, {-452.0f, 1595.51f, -824.41f}, 90.0f, 1},
    {"初代教区長ローレンス", 13401850, 3402950, true, 450000, 0, {-450.9f, 1535.71f, -292.69f}, -150.0f, 1},
    {"ゴースの遺子", 13601800, 3602950, true, 453000, 0, {-700.2f, 1580.5f, -906.2f}, 0.0f, 1},
    // The chalice dungeons' own bosses (the "chalice" setting), in the fixed
    // dungeons the game ships as whole layouts (m29_XX_90_00, "固定" in
    // map/mapviewlist_fordungeon.loadlistlist), the shallowest that has each.
    // A fixed dungeon's script starts the shared m29 events once per boss
    // slot: its fog (12906796: the fog objects, the defeat flag, "met" - which
    // also opens the boss door, 12901713 - and "begun"), its health bar
    // (12906806, whose name id is the bar's) and its battle (12901686). All
    // its layers are one block, entered at its lamp (2901950: return point
    // 2902950). name_id is the bosses' NpcParam nameId, which can differ from
    // the bar's (the Abhorrent Beast's rows say Loran Darkbeast).
    {"死体の巨人", 12901800, 2902950, false, 313000, 0, {-22.8f, 31.53f, -91.0f}, 90.0f, 0, {12900532}, {}, 0x1d0a5a00, 12905502},
    {"遺跡の番人", 12901801, 2902950, false, 304000, 0, {-130.0f, 10.5f, -55.0f}, 180.0f, 0, {12900536}, {}, 0x1d0a5a00, 12905511},
    {"旧主の番犬", 12901802, 2902950, false, 501000, 0, {-22.637f, -17.5f, 52.0f}, 270.0f, 0, {12900538}, {}, 0x1d0a5a00, 12905516},
    {"獣憑き", 12901800, 2902950, false, 75000, 0, {78.027f, 26.403f, 110.5f}, 270.0f, 0, {12900530}, {}, 0x1d145a00, 12905504},
    {"旧主の番人", 12901801, 2902950, false, 216000, 0, {156.0f, 10.1f, 84.5f}, 90.0f, 0, {12900533}, {}, 0x1d145a00, 12905511},
    {"トゥメルの末裔", 12901802, 2902950, false, 305000, 0, {-42.25f, -17.5f, 22.784f}, 0.0f, 0, {12900534}, {}, 0x1d145a00, 12905516},
    {"獣血の主", 12901803, 2902950, false, 509000, 0, {-32.422f, -66.5f, -35.75f}, 270.0f, 0, {12900541}, {}, 0x1d1e5a00, 12905525},
    {"トゥメルの古老", 12901802, 2902950, false, 305000, 0, {-55.25f, -17.5f, 68.161f}, 0.0f, 0, {12900540}, {}, 0x1d1f5a00, 12905513},
    {"恐ろしい獣", 12901802, 2902950, false, 504000, 0, {55.254f, -17.5f, -52.0f}, 90.0f, 0, {12900528}, {}, 0x1d2a5a00, 12905516},
    {"トゥメルの女王、ヤーナム", 12901802, 2902950, false, 306000, 0, {34.388f, -28.072f, -204.75f}, 90.0f, 0, {12900536}, {}, 0x1d325a00, 12905516},
};

bb::Plugin g;

enum class State { Waiting, Countdown, Warping, Fighting, Retry, Cleared, Done };
State g_state = State::Waiting;
std::vector<const Boss*> g_run;
std::size_t g_round = 0;
double g_wait = 0;          // the state's wait, s: the countdown's in real time, Retry's and Cleared's in the world's
double g_run_start = 0;     // real time when the first round began
double g_round_start = -1;  // real time when this round's fight began (its first stand at the fog); -1: not yet
std::int64_t g_deaths_at_start = 0;
std::vector<double> g_round_times;  // s
std::map<std::uint32_t, bool> g_original;  // every flag the run changes, as the save had them
std::string g_dir;
int g_start_delay = 10;
bool g_auto_start = false, g_auto_started = false, g_auto_pending = false, g_restored = false;
// boss_rush.test_kill = N (seconds): a test switch - N seconds into a round
// the boss (the character whose NpcParam names it) is put at 0 HP.
int g_retries = 0;
int g_refused = 0;                 // times this round's map sent the player away without a death
std::int64_t g_deaths_at_arrival = 0;
std::int64_t g_deaths_refilled = 0;  // the deaths count the last refill answered
int g_test_kill = 0;
double g_fight_time = 0;  // the world's time since this round's arrival, or its last retry
// boss_rush.test_watch = 1: a test switch - the player's and the boss's
// positions every two seconds of a round, to see the boss move once the
// player has gone through the fog.
bool g_test_watch = false;
double g_watch_at = 0, g_quiet_at = 0;  // the world's time of the next watch line, the next quiet_arena
bool g_skip_intro = true;

// Time, in seconds, from two clocks read once a frame (clocks()):
//  - real time: the run's and the rounds' times, loading screens in, and the
//    countdown the player is told about (which runs only in the world);
//  - the world's time, the game's own, which the waits are about (a travel
//    the game is given time to begin, a fade, a pick-up): a frame adds its
//    length but at most 1/30 s - the game's step, so a renderer too slow for
//    30 frames a second slows it as it slows the game - and nothing while a
//    loading screen is up or there is no world (the title and its menus).
// They were frames, at 60 a second. At another frame rate every wait was off,
// and a loading screen flips hundreds of times a second: a load of a few
// seconds used up the travel's "25 s", the travel was asked again inside the
// load, refused there (no player), and the run stopped saying travel was off.
// boss_rush.test_frame_seconds = s: a test switch - each frame is s long, for
// a pretend host whose frames come as fast as it makes them.
double g_frame_seconds = 0;
std::chrono::steady_clock::time_point g_t0;
std::uint64_t g_frame_no = 0;
double g_now = 0;          // real time since the plugin started
double g_dt = 0;           // this frame's length
double g_world_time = 0;   // the world's time
double g_step = 0;         // this frame's share of it
double g_world_since = 0;  // g_world_time when the world last came back
bool g_loading = false;    // this frame: a loading screen is up
bool g_in_world = false;   // this frame: no loading screen, and the player in a map
void clocks() {
    const double now = g_frame_seconds > 0 ? static_cast<double>(g_frame_no) * g_frame_seconds
                                           : std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count();
    g_dt = g_frame_no++ ? std::max(0.0, now - g_now) : 0.0;
    g_now = now;
    g_loading = g.loading_screen();
    std::uint32_t block = 0;
    const bool world = !g_loading && g.api->player_block(&block) == 0;
    if (world && !g_in_world) g_world_since = g_world_time;
    g_in_world = world;
    g_step = world ? std::min(g_dt, 1.0 / 30.0) : 0.0;
    g_world_time += g_step;
}
// How long the world has been up since it last came back; 0 outside it.
double settled() { return g_in_world ? g_world_time - g_world_since : 0.0; }
// A lamp travel is asked only in the world, this long after a loading screen
// (or the title) has gone: past the fade-in, during which the game can drop
// one (the first asked after a death's respawn often went).
constexpr double kSettle = 3.0;

// A message for the player, shown once the world is up: on_world_load comes
// while a load's screen is still up.
std::string g_banner;
float g_banner_s = 0;
void banner(const std::string& text, float seconds) {
    g_banner = text;
    g_banner_s = seconds;
}

std::string clock_text(double seconds) {
    const auto s = static_cast<unsigned long long>(seconds > 0 ? seconds : 0);
    char t[32];
    std::snprintf(t, sizeof(t), "%llu:%02llu", s / 60, s % 60);
    return t;
}

void save_restore_file() {
    std::ofstream f(g_dir + "/restore.txt");
    for (const auto& [flag, on] : g_original) f << flag << ' ' << (on ? 1 : 0) << '\n';
}

// A run cut short last time: its flags as they were before it.
void restore_leftovers() {
    std::ifstream f(g_dir + "/restore.txt");
    std::uint32_t flag = 0;
    int on = 0;
    int n = 0;
    while (f >> flag >> on) {
        g.set_flag(flag, on != 0);
        ++n;
    }
    if (n) {
        g.log("boss_rush: put back %d flags from a run that did not finish", n);
        std::remove((g_dir + "/restore.txt").c_str());
    }
}

// A run ends in the Hunter's Dream, at its first return point
// (ReturnPointParam), travelled to in the world and settled (dream_tick).
constexpr std::int32_t kHuntersDream = 2102950;
bool g_to_dream = false;
void to_dream() { g_to_dream = true; }
void dream_tick() {
    if (!g_to_dream || settled() < kSettle) return;
    g_to_dream = false;
    if (g.api->lamp_warp(kHuntersDream) != 0) g.log("boss_rush: could not travel to the Hunter's Dream");
}

void give_back();
void finish() {
    give_back();
    const std::int64_t deaths = g.stat("deaths") - g_deaths_at_start;
    std::ostringstream text;
    text << "ボスラッシュ完了 - " << clock_text(g_now - g_run_start) << ", " << deaths << (deaths == 1 ? " 回死亡" : " 回死亡");
    g.message(text.str(), 10.0f);
    std::ofstream f(g_dir + "/results.txt", std::ios::app);
    f << text.str() << '\n';
    for (std::size_t i = 0; i < g_run.size() && i < g_round_times.size(); ++i) {
        f << "  " << g_run[i]->name << ": " << clock_text(g_round_times[i]) << '\n';
    }
    g.log("boss_rush: %s", text.str().c_str());
    g_state = State::Done;
    to_dream();
}

// A boss fight is a small family of events in its map's script, and an
// event's flag is its own id, set when it has run: the defeat event (the
// defeat flag itself) and the ones after it (defeat + 1..9), among them the
// first encounter's staging (+2; +1 in The Old Hunters), whose flag says the
// boss was met. Clearing the family reopens the fight. With "met" on, the
// fight is staged as on any return: the fog wall stands (its event waits for
// that flag), and going through it wakes the boss; with it off, as the first
// time (the entrances and their cutscenes). "The fight has begun" (defeat +
// 3000) is the map's own and starts each load cleared.
//
// A chalice dungeon's flags (1290xxxx) are numbered by boss slot, so every
// fixed dungeon uses the same ones: a chalice round first puts back those an
// earlier round changed, then clears its boss's defeat and "begun" and turns
// "met" on (its set_flags) - always, there being no entrance to skip - and
// has the borrowed altar slot hold its dungeon.
std::uint32_t read_bits(std::uint32_t first, int n) {  // n flags as a number, the first the highest bit (as the game reads them)
    std::uint32_t v = 0;
    for (int i = 0; i < n; ++i) v = v << 1 | (g.flag(first + static_cast<std::uint32_t>(i)) ? 1u : 0u);
    return v;
}
void write_bits(std::uint32_t first, int n, std::uint32_t v) {
    for (int i = 0; i < n; ++i) g.set_flag(first + static_cast<std::uint32_t>(i), (v >> (n - 1 - i)) & 1);
}
// The Hunter's Dream's altars keep a chalice dungeon each, in event flags:
// which altar is in use (9020..9026, the first one on picks slot n = flag -
// 9020), and per slot, from 94000000 + n * 100000, its dungeon's map id (32
// flags, area byte first) and the rest of it (to +0x600). A stage change into
// area 29 loads the dungeon of the slot in use - a fixed one by its map id
// alone, as the debug menu's dungeon list does (sub_1d62940) - so a lamp
// travel to the dungeon's lamp enters it. The run takes the last empty slot
// (map id all ones), else the seventh, and puts the slot and the choice back.
constexpr std::uint32_t kAltarInUse = 9020, kAltarSlots = 94000000, kAltarSlotStride = 100000, kAltarSlotFlags = 0x600;
int g_altar = -1;
void borrow_altar() {
    g_altar = 6;
    for (int n = 6; n >= 0; --n) {
        if (read_bits(kAltarSlots + static_cast<std::uint32_t>(n) * kAltarSlotStride, 32) == 0xffffffffu) {
            g_altar = n;
            break;
        }
    }
    for (std::uint32_t i = 0; i < 7; ++i) g_original[kAltarInUse + i] = g.flag(kAltarInUse + i);
    const std::uint32_t slot = kAltarSlots + static_cast<std::uint32_t>(g_altar) * kAltarSlotStride;
    for (std::uint32_t i = 0; i < kAltarSlotFlags; ++i) g_original[slot + i] = g.flag(slot + i);
    g.log("boss_rush: altar slot %d holds the chalice rounds' dungeons%s", g_altar + 1,
          read_bits(slot, 32) == 0xffffffffu ? "" : " (none was empty: its own dungeon comes back after the run)");
}

// The save's flags as they were before the run (the altar slot among them).
void give_back() {
    int missed = 0;  // read back otherwise: a flag whose block is not loaded
    for (const auto& [flag, on] : g_original) {
        g.set_flag(flag, on);
        missed += g.flag(flag) != on;
    }
    const std::string note = missed ? " (" + std::to_string(missed) + " read back otherwise)" : "";
    g.log("boss_rush: %zu flags back as the save had them%s", g_original.size(), note.c_str());
    std::remove((g_dir + "/restore.txt").c_str());
    if (g_altar >= 0) {
        g.log("boss_rush: altar slot %d given back (map id %08x), the altar in use as before (%02x)", g_altar + 1,
              read_bits(kAltarSlots + static_cast<std::uint32_t>(g_altar) * kAltarSlotStride, 32), read_bits(kAltarInUse, 7));
    }
}

bool wake(const Boss* b) {
    int failed = 0;
    if (b->dungeon) {
        for (const auto& [flag, on] : g_original) {
            if (flag / 10000 == 1290) g.set_flag(flag, on);
        }
        failed += g.api->event_flag_set(b->defeat_flag, 0) != 0;
        failed += g.api->event_flag_set(b->begun, 0) != 0;
        write_bits(kAltarInUse, 7, 1u << (6 - g_altar));
        write_bits(kAltarSlots + static_cast<std::uint32_t>(g_altar) * kAltarSlotStride, 32, b->dungeon);
    } else {
        for (std::uint32_t i = 0; i < 10; ++i) failed += g.api->event_flag_set(b->defeat_flag + i, 0) != 0;
        if (g_skip_intro) failed += g.api->event_flag_set(b->defeat_flag + b->met, 1) != 0;
    }
    for (const std::uint32_t f : b->set_flags) {
        if (f) failed += g.api->event_flag_set(f, 1) != 0;
    }
    if (failed) g.log("boss_rush: %d of %s's flags could not be written here", failed, b->name);
    return failed == 0;
}

void log_flags(const Boss* b, const char* when) {
    std::uint32_t block = 0;
    g.api->player_block(&block);
    if (b->dungeon) {
        g.log("boss_rush: %s: %s flags %u = %d, met %u = %d, begun %u = %d; block %08x, altar slot %d holds %08x, deaths %lld", when, b->name,
              b->defeat_flag, g.flag(b->defeat_flag) ? 1 : 0, b->set_flags[0], g.flag(b->set_flags[0]) ? 1 : 0, b->begun, g.flag(b->begun) ? 1 : 0,
              block, g_altar + 1, read_bits(kAltarSlots + static_cast<std::uint32_t>(g_altar) * kAltarSlotStride, 32),
              static_cast<long long>(g.stat("deaths")));
        return;
    }
    std::string t;
    for (std::uint32_t i = 0; i < 4; ++i) t += g.flag(b->defeat_flag + i) ? '1' : '0';
    g.log("boss_rush: %s: %s flags %u..+3 = %s, +3000 = %d; block %08x, deaths %lld", when, b->name, b->defeat_flag, t.c_str(),
          g.flag(b->defeat_flag + 3000) ? 1 : 0, block, static_cast<long long>(g.stat("deaths")));
}

// The map block a boss's lamp is in, as player_block packs it: a return
// point (lamp entity + 1000) is AABxxxx - area AA, block B (2412951: m24_01).
std::uint32_t block_of(const Boss* b) {
    const std::uint32_t area = static_cast<std::uint32_t>(b->lamp) / 100000;
    const std::uint32_t blk = (static_cast<std::uint32_t>(b->lamp) / 10000) % 10;
    return area << 24 | blk << 16;
}

// Just outside the fog wall, facing it: within reach of its action button (a
// box 1.2 m deep before the fog: ActionButtonParam regionType 1, allowAngle 90).
constexpr float kPi = 3.14159265f, kFogStandOff = 1.0f;
void fog_spot(const Boss* b, float pos[3], float* yaw) {
    const float t = b->fog_yaw * kPi / 180.0f;
    pos[0] = b->fog[0] + std::sin(t) * kFogStandOff;  // behind the fog: out of the arena
    pos[1] = b->fog[1] + 0.05f;
    pos[2] = b->fog[2] + std::cos(t) * kFogStandOff;
    *yaw = t;  // facing as the fog does: into the arena
}

// Blood Vials (goods 1000) and Quicksilver Bullets (900) back to the most
// the player can carry, with the game's own helpers - the ones its debug
// menu's "fill to the maximum" calls (sub_18d3dc0): the goods row
// (sub_231e3d0: {id, row}), the most one can hold (sub_231e7e0: for these
// two the live caps runes raise), the inventory slot (sub_18d9f60 on
// EquipInventoryData; its entries are 16 bytes, the quantity at +8, the first
// *(+0x24) at +0x58 and the rest at +0x48) and the add (sub_18cdfd0 on
// EquipGameData, PlayerGameData + 0x1d0, which holds the inventory at +0x158).
constexpr std::uint64_t kGoodsRow = 0x231e3d0, kGoodsMax = 0x231e7e0, kInventorySlot = 0x18d9f60, kInventoryAdd = 0x18cdfd0;
constexpr std::uint32_t kGoods = 0x40000000, kBloodVial = 1000, kQuicksilverBullets = 900;
std::uint64_t call_at(std::uint64_t bn, std::initializer_list<std::uint64_t> args, std::uint64_t stack0 = 0) {
    BbCallRegs r{};
    std::size_t i = 0;
    for (std::uint64_t a : args) r.arg[i++] = a;
    r.stack[0] = stack0;
    return g.api->call_guest(bn, &r) == 0 ? r.ret : 0;
}
std::uint8_t* equip_game_data() {
    const std::uint64_t slot = g.api->guest_addr(bb::GAME_DATA_MAN_SINGLETON_PTR.bn());
    auto* man = slot ? *reinterpret_cast<bb::GameDataMan**>(slot) : nullptr;
    auto* player = man ? reinterpret_cast<std::uint8_t*>(man->local_player_record) : nullptr;
    return player ? player + 0x1d0 : nullptr;
}
// How many of goods `id` the inventory holds, and the most it can.
bool goods_count(std::uint32_t id, int* have, int* max) {
    std::uint8_t* egd = equip_game_data();
    if (!egd) return false;
    struct {
        std::int32_t id;
        std::int32_t pad;
        void* row;
    } h{-1, 0, nullptr};
    call_at(kGoodsRow, {reinterpret_cast<std::uint64_t>(&h), id});
    if (!h.row) return false;
    *max = static_cast<std::int32_t>(call_at(kGoodsMax, {reinterpret_cast<std::uint64_t>(&h)}));
    std::uint8_t* inv = egd + 0x158;
    std::uint32_t item[4] = {kGoods | id, 0, 0, 0};
    const auto slot = static_cast<std::int32_t>(call_at(kInventorySlot, {reinterpret_cast<std::uint64_t>(inv), reinterpret_cast<std::uint64_t>(item)}));
    *have = 0;
    const auto rd32 = [&](std::size_t off) { return *reinterpret_cast<const std::uint32_t*>(inv + off); };
    const auto rd64 = [&](std::size_t off) { return *reinterpret_cast<std::uint8_t* const*>(inv + off); };
    if (slot >= 0 && static_cast<std::uint32_t>(slot) < rd32(0x88) + 1) {
        const std::uint32_t first = rd32(0x24);
        const std::uint8_t* e = static_cast<std::uint32_t>(slot) < first ? rd64(0x58) + slot * 16 : rd64(0x48) + (slot - first) * 16;
        if (e && *reinterpret_cast<const std::uint32_t*>(e)) *have = *reinterpret_cast<const std::int32_t*>(e + 8);
    }
    return true;
}
void refill() {
    std::string t;
    for (const std::uint32_t id : {kBloodVial, kQuicksilverBullets}) {
        int have = 0, max = 0;
        if (!goods_count(id, &have, &max)) continue;
        if (max > have) call_at(kInventoryAdd, {reinterpret_cast<std::uint64_t>(equip_game_data()), kGoods, id, static_cast<std::uint64_t>(max - have), 1, 1}, 0);
        int now = have;
        goods_count(id, &now, &max);
        t += (t.empty() ? "" : ", ") + std::string(id == kBloodVial ? "vials " : "bullets ") + std::to_string(have) + " -> " + std::to_string(now);
    }
    g.log("boss_rush: refilled: %s", t.c_str());
}

// The follow camera keeps the view it had at the lamp through a warp inside
// the block; the camera owner's one-shot reset (consumed by the camera
// blend) puts it behind the player, looking at the fog.
void reset_camera() {
    const std::uint64_t slot = g.api->guest_addr(bb::ChrExFollowCam::OWNER_ROOT_PTR.bn());
    const std::uint64_t root = slot ? *reinterpret_cast<const std::uint64_t*>(slot) : 0;
    auto* owner = root ? *reinterpret_cast<bb::ChrCamOwner**>(root + bb::CHR_CAM_OWNER_GLOBAL_OFFSET) : nullptr;
    if (owner) owner->camera_reset_requested = true;
}

// Characters a round switches off, with the game's own "disable character"
// (EMEVD 2004[05] calls this on the ChrIns: its collision off, and bit 0 of its
// ChrSetEntry's +0x20, which keeps it out of the world's update and draw, set),
// found by entity id with the lookup the event instructions use. A map load
// makes them again, so each arrival switches them off anew; nothing is saved.
constexpr std::uint64_t kChrByEntity = 0x17c97a0, kChrSetDisabled = 0x1cc6390;
constexpr std::int32_t kTeamEnemy = 23;  // an ordinary enemy's team (NpcParam teamType; bosses have their own)
bool chr_disabled(void* ins) {
    const auto* entry = *reinterpret_cast<const std::uint8_t* const*>(static_cast<const std::uint8_t*>(ins) + 0x18);
    return entry && (entry[0x20] & 1);
}
bool switch_off(void* ins) {
    if (!ins || chr_disabled(ins)) return false;
    call_at(kChrSetDisabled, {reinterpret_cast<std::uint64_t>(ins), 1});
    return true;
}
bool boss_name(const Boss* b, std::int32_t name_id) {
    return name_id / 100 == b->name_id / 100 || (b->alt_name_id && name_id / 100 == b->alt_name_id / 100);
}
// The arena for the fight alone, and the way to it clear: the NPCs a save's
// quests may have in the arena (keep_out), and ordinary enemies outside the
// fog within 30 m of it - the arena's side is left alone, its adds being the
// fight's.
void quiet_arena(const Boss* b) {
    int off = 0;
    for (const std::int32_t entity : b->keep_out) {
        if (entity && switch_off(reinterpret_cast<void*>(call_at(kChrByEntity, {static_cast<std::uint64_t>(entity), 0, 0})))) {
            g.log("boss_rush: %d switched off for %s", entity, b->name);
            ++off;
        }
    }
    const float t = b->fog_yaw * kPi / 180.0f, fx = -std::sin(t), fz = -std::cos(t);  // into the arena
    for (const BbChr& c : g.characters()) {
        if (c.is_player || c.team_type != kTeamEnemy || c.hp <= 0 || c.npc_param <= 0) continue;
        const float dx = c.pos[0] - b->fog[0], dy = c.pos[1] - b->fog[1], dz = c.pos[2] - b->fog[2];
        if (dx * fx + dz * fz > -1.0f || dx * dx + dz * dz > 30.0f * 30.0f || std::fabs(dy) > 12.0f) continue;
        const auto* npc = g.param<bb::params::NpcParam>(static_cast<std::uint32_t>(c.npc_param));
        if (npc && boss_name(b, npc->nameId)) continue;
        if (switch_off(c.ins)) ++off;
    }
    if (off) g.log("boss_rush: %d characters switched off around %s's fog wall", off, b->name);
}

// The fog warp, inside the arena's block: asked once the world is up, never
// while the arrival's loading screen is (on_world_load comes with it still
// up, and a warp then lands before the world is ready: the streamer can ask
// for the neighbouring blocks inside the screen, and the arrival comes out
// partial, its effects left over for the visit). Checked a second of the
// world's time later and asked again, for up to kFogTries of them.
constexpr int kFogTries = 10;
int g_fog_tries = 0;
int g_fog = 0;  // 0: none; 1: wanted, once the world is up; 2: asked, checked at g_fog_at
double g_fog_at = 0;
void to_fog(const Boss* b) {
    float p[3], yaw = 0;
    fog_spot(b, p, &yaw);
    g.api->player_warp(0, p, yaw);
    g_fog = 2;
    g_fog_at = g_world_time + 1.0;
}
void check_fog(const Boss* b) {
    float p[3], at[3], yaw = 0;
    fog_spot(b, p, &yaw);
    if (g.api->player_position(at, nullptr) != 0) {
        g_fog_at = g_world_time + 1.0;
        return;
    }
    const float dx = at[0] - p[0], dz = at[2] - p[2];
    if (dx * dx + dz * dz < 4.0f) {
        g_fog = 0;
        reset_camera();
        g.log("boss_rush: at %s's fog wall (%.1f %.1f %.1f)", b->name, at[0], at[1], at[2]);
        quiet_arena(b);
        if (g_test_watch) {
            const float t = b->fog_yaw * kPi / 180.0f;
            for (const BbChr& c : g.characters()) {
                const float cx = c.pos[0] - b->fog[0], cy = c.pos[1] - b->fog[1], cz = c.pos[2] - b->fog[2];
                if (c.is_player || cx * cx + cz * cz > 40.0f * 40.0f || std::fabs(cy) > 20.0f) continue;
                const auto* npc = c.npc_param > 0 ? g.param<bb::params::NpcParam>(static_cast<std::uint32_t>(c.npc_param)) : nullptr;
                g.log("boss_rush: near the fog: c%04u npc %d name %d team %d chr_type %d hp %d at %.1f m, %s%s", c.model, c.npc_param,
                      npc ? npc->nameId : -1, c.team_type, c.chr_type, c.hp, std::sqrt(cx * cx + cz * cz),
                      cx * -std::sin(t) + cz * -std::cos(t) < 0 ? "outside" : "inside", chr_disabled(c.ins) ? ", off" : "");
            }
        }
        return;
    }
    if (++g_fog_tries >= kFogTries) {
        g_fog = 0;
        g.log("boss_rush: could not reach %s's fog wall (at %.1f %.1f %.1f)", b->name, at[0], at[1], at[2]);
        g.message("ボスラッシュ: 到達できません: " + std::string(b->name) + "の霧。灯りから歩いて向かってください", 6.0f);
        return;
    }
    to_fog(b);
}
void fog_tick(const Boss* b) {
    if (g_fog == 1) {
        if (g_round_start < 0) g_round_start = g_now;  // the round's time runs from its first stand at the fog
        g.log("boss_rush: to %s's fog wall, %.2f s of the world after the loading screen", b->name, settled());
        to_fog(b);
    } else if (g_fog == 2 && g_world_time >= g_fog_at) {
        check_fog(b);
    }
}

// A round's travel, a lamp travel into the boss's map: asked in the world,
// settled (kSettle), never inside a loading screen - asked there, the game
// began it as the screen went, or bbhost refused it (the load had no player
// yet). The game not beginning it - no loading screen kTravelWait of the
// world's time after the ask - means it dropped it: asked again. Only when
// it has dropped kTravelAsks of them, or bbhost refuses one with the player
// there (travel is off), has it truly failed: the run stops, saying so.
constexpr double kTravelWait = 10.0;
constexpr int kTravelAsks = 5;
int g_travel_asks = 0;        // this round's asks
double g_travel_at = -1;      // the world's time of the last ask; -1: none yet
bool g_travel_began = false;  // a loading screen since the last ask
// boss_rush.test_drop_travel = N: a test switch - the first N asks of each
// round are not passed on, as the game drops a travel.
int g_test_drop_travel = 0;

void travel_failed(const Boss* b, const std::string& why) {
    g.log("boss_rush: %s to %s's arena (lamp %d, asked %d times); the run stops", why.c_str(), b->name, b->lamp, g_travel_asks);
    give_back();
    g.message("ボスラッシュ停止: " + why + " / " + b->name + "の闘技場", 8.0f);
    g_state = State::Done;
}

void travel_tick(const Boss* b) {
    if (g_loading) {  // begun (or another load): the arrival is on_world_load's
        g_travel_began = true;
        return;
    }
    if (settled() < kSettle || (g_travel_at >= 0 && g_world_time - g_travel_at < kTravelWait)) return;
    if (g_travel_at >= 0) {
        g.log("boss_rush: the travel to %s's arena %s; again", b->name,
              g_travel_began ? "別のマップに到着しました" : "移動が始まりませんでした");
    }
    if (g_travel_asks >= kTravelAsks) {
        travel_failed(b, g_travel_began ? "ゲームが移動を受け付けませんでした" : "ゲームが移動を開始しませんでした");
        return;
    }
    ++g_travel_asks;
    g_travel_at = g_world_time;
    g_travel_began = false;
    g.log("boss_rush: travel to %s's arena (lamp %d, ask %d), %.2f s of the world after the loading screen", b->name, b->lamp,
          g_travel_asks, settled());
    if (g_travel_asks <= g_test_drop_travel) {
        g.log("boss_rush: test: travel ask %d to %s's arena dropped", g_travel_asks, b->name);
        return;
    }
    if (g.api->lamp_warp(b->lamp) != 0) {
        std::uint32_t block = 0;
        if (g.api->player_block(&block) != 0) {  // no player this frame after all: not a refusal
            --g_travel_asks;
            g_travel_at = -1;
            return;
        }
        travel_failed(b, "bbhost が移動を拒否しました");
    }
}

void begin();
void start_round() {
    const Boss* b = g_run[g_round];
    wake(b);
    g.log("boss_rush: round %zu - %s", g_round + 1, b->name);
    g_state = State::Warping;
    g_travel_asks = 0;
    g_travel_at = -1;
    g_travel_began = false;
    travel_tick(b);
}

void tick() {
    clocks();
    if (g_auto_pending && g_in_world) {  // Start by itself: once the first load's screen has gone
        g_auto_pending = false;
        begin();
    }
    if (g_in_world && !g_banner.empty()) {
        g.message(g_banner, g_banner_s);
        g_banner.clear();
    }
    dream_tick();
    switch (g_state) {
    case State::Countdown:
        if (g_in_world) g_wait -= g_dt;
        if (g_wait <= 0 && settled() >= kSettle) {
            g_run_start = g_now;
            g_deaths_at_start = g_deaths_refilled = g.stat("deaths");
            g_altar = -1;
            for (const Boss* b : g_run) {
                if (b->dungeon) {
                    g_original[b->defeat_flag] = g.flag(b->defeat_flag);
                    g_original[b->begun] = g.flag(b->begun);
                    if (g_altar < 0) borrow_altar();
                } else {
                    for (std::uint32_t i = 0; i < 10; ++i) g_original[b->defeat_flag + i] = g.flag(b->defeat_flag + i);
                    g_original[b->defeat_flag + 3000] = g.flag(b->defeat_flag + 3000);
                }
                for (const std::uint32_t f : b->set_flags) {
                    if (f) g_original[f] = g.flag(f);
                }
            }
            save_restore_file();
            start_round();
        }
        break;
    case State::Fighting: {
        const Boss* b = g_run[g_round];
        const double fought = g_fight_time;
        if (g_in_world) {
            g_fight_time += g_step;
            fog_tick(b);
            if (g_world_time >= g_quiet_at) {  // whoever the map made since
                g_quiet_at = g_world_time + 1.0;
                quiet_arena(b);
            }
        }
        if (g_test_watch && g_in_world && g_world_time >= g_watch_at) {
            g_watch_at = g_world_time + 2.0;
            float at[3] = {}, yaw = 0;
            g.api->player_position(at, &yaw);
            std::string t;
            for (const BbChr& c : g.characters()) {
                const auto* npc = c.npc_param > 0 ? g.param<bb::params::NpcParam>(static_cast<std::uint32_t>(c.npc_param)) : nullptr;
                if (!npc || npc->nameId / 100 != b->name_id / 100 || c.hp <= 0) continue;
                char one[96];
                std::snprintf(one, sizeof(one), " c%04u %.1f %.1f %.1f hp %d", c.model, c.pos[0], c.pos[1], c.pos[2], c.hp);
                t += one;
            }
            g.log("boss_rush: watch %.0fs: player %.1f %.1f %.1f yaw %.2f; boss%s", g_fight_time, at[0], at[1], at[2], yaw,
                  t.empty() ? " not found" : t.c_str());
        }
        // The defeat flag is the defeat event's own: the event system sets it
        // when the event ends, not through SetEventFlag, so on_event_flag
        // does not see it - it is read here instead.
        if (g.flag(b->defeat_flag)) {
            const double took = g_round_start >= 0 ? g_now - g_round_start : 0.0;
            g_round_times.push_back(took);
            g.message(std::string(b->name) + " 撃破 - " + clock_text(took), 6.0f);
            g.log("boss_rush: %s defeated in %s", b->name, clock_text(took).c_str());
            g_state = State::Cleared;
            g_wait = 10.0;  // time to pick up what it dropped
            break;
        }
        // Each test step once, as the world's time passes its second.
        const auto passes = [&](double at) { return fought < at && g_fight_time >= at; };
        if (g_test_kill) {
            const auto find_boss = [](std::size_t* among) -> BbChr {
                const std::vector<BbChr> chrs = g.characters();
                if (among) *among = chrs.size();
                for (const BbChr& c : chrs) {
                    const auto* npc =
                        c.npc_param > 0 ? g.param<bb::params::NpcParam>(static_cast<std::uint32_t>(c.npc_param)) : nullptr;
                    if (npc && npc->nameId == g_run[g_round]->name_id && c.hp > 0) return c;
                }
                return BbChr{};
            };
            if (!g_skip_intro && passes(10.0)) {
                // Wake it the way a player does: walk up to it.
                const BbChr boss = find_boss(nullptr);
                if (boss.ins) {
                    const float near[3] = {boss.pos[0] + 4.0f, boss.pos[1] + 0.5f, boss.pos[2]};
                    g.api->player_warp(0, near, 0.0f);
                }
            }
            if (passes(static_cast<double>(g_test_kill))) {
                // Every character of the fight: a boss can be several (the two
                // witches of Hemwick; the three Shadows, whose names are 212010,
                // 212020 and 212030 - one hundred).
                std::size_t among = 0, killed = 0;
                for (const BbChr& c : g.characters()) {
                    ++among;
                    const auto* npc =
                        c.npc_param > 0 ? g.param<bb::params::NpcParam>(static_cast<std::uint32_t>(c.npc_param)) : nullptr;
                    if (!npc || !boss_name(g_run[g_round], npc->nameId) || c.hp <= 0) continue;
                    // ChrIns -> modules (+0x3b0) -> data module (+0x20) -> HP (+0xf8): the
                    // game's own death handling takes it from there (rewards, the
                    // defeat event).
                    auto* modules = *reinterpret_cast<std::uint8_t**>(static_cast<std::uint8_t*>(c.ins) + 0x3b0);
                    auto* data = *reinterpret_cast<std::uint8_t**>(modules + 0x20);
                    g.log("boss_rush: test kill: c%04u (NpcParam %d) at %d/%d HP", c.model, c.npc_param, c.hp, c.max_hp);
                    *reinterpret_cast<std::int32_t*>(data + 0xf8) = 0;
                    ++killed;
                }
                g.log("boss_rush: test kill: %zu of %zu characters carry %s's name", killed, among, g_run[g_round]->name);
                if (!killed) {
                    std::map<std::int32_t, int> names;
                    for (const BbChr& c : g.characters()) {
                        const auto* npc =
                            c.npc_param > 0 ? g.param<bb::params::NpcParam>(static_cast<std::uint32_t>(c.npc_param)) : nullptr;
                        if (npc && c.max_hp > 500) ++names[npc->nameId];
                    }
                    std::string t;
                    for (const auto& [id, n] : names) t += " " + std::to_string(id) + "x" + std::to_string(n);
                    g.log("boss_rush: test kill: the names of characters over 500 HP here:%s", t.c_str());
                }
            }
        }
        break;
    }
    case State::Warping:
        travel_tick(g_run[g_round]);
        break;
    case State::Retry:  // in the world: the next round's flags are written where the map is
        if (g_in_world && (g_wait -= g_step) <= 0) start_round();
        break;
    case State::Cleared:
        if (g_in_world && (g_wait -= g_step) <= 0) {
            g_refused = 0;
            if (++g_round >= g_run.size()) finish();
            else start_round();
        }
        break;
    default:
        break;
    }
}

// The run as the settings say now (a run reads them when it starts).
void build_run() {
    g_run.clear();
    g_start_delay = static_cast<int>(g.config_number("boss_rush.start_delay", 10));
    g_skip_intro = g.config_bool("boss_rush.awake", true);
    const bool dlc = g.config_bool("boss_rush.dlc", true);
    const bool chalice = g.config_bool("boss_rush.chalice", false);
    for (const Boss& b : kBosses) {
        if ((dlc || !b.dlc) && (chalice || !b.dungeon)) g_run.push_back(&b);
    }
    if (g.config("boss_rush.order") == "random") {
        bb::Rng rng(bb::seed_of(g.config("boss_rush.seed", "ボスラッシュ")));
        rng.shuffle(g_run);
    }
    if (const int first = static_cast<int>(g.config_number("boss_rush.first", 1)); first > 1 && static_cast<std::size_t>(first) <= g_run.size())
        g_run.erase(g_run.begin(), g_run.begin() + (first - 1));
    if (const int rounds = static_cast<int>(g.config_number("boss_rush.rounds", 0)); rounds > 0 && static_cast<std::size_t>(rounds) < g_run.size())
        g_run.resize(static_cast<std::size_t>(rounds));
}

void begin() {
    if (g_state != State::Waiting && g_state != State::Done) {
        g.message("ボスラッシュは既に進行中です", 3.0f);
        return;
    }
    if (g.stat("hp", -1) < 0) {
        g.message("先にワールドへ入ってください", 3.0f);
        return;
    }
    build_run();
    if (g_run.empty()) return;
    g_round = 0;
    g_retries = g_refused = 0;
    g_round_times.clear();
    g_original.clear();
    g_to_dream = false;
    g_state = State::Countdown;
    g_wait = g_start_delay;  // seconds in the world: a loading screen holds it
    g.log("boss_rush: %zu bosses, %s order", g_run.size(), g.config("boss_rush.order", "story").c_str());
    g.message("ボスラッシュ: " + std::to_string(g_run.size()) + " 体 - 開始まで " + std::to_string(g_start_delay) +
                  " 秒",
              6.0f);
}

// Ends a run before its last boss: the save's flags as they were, and back to
// the Hunter's Dream once a round has begun.
void stop(const char* why) {
    if (g_state == State::Waiting || g_state == State::Done) return;
    const bool travelled = g_state != State::Countdown;
    give_back();
    g.log("boss_rush: %s at round %zu", why, g_round + 1);
    g.message("ボスラッシュ " + std::string(why), 4.0f);
    g_state = State::Done;
    if (travelled) to_dream();
}

void skip() {
    if (g_state != State::Fighting && g_state != State::Warping && g_state != State::Retry) return;
    g.log("boss_rush: %s skipped", g_run[g_round]->name);
    g_round_times.push_back(0);
    g_state = State::Cleared;
    g_wait = 0;
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    if (!g.attach(api, 9)) return 1;
    if (!api->eboot_is_109()) return 1;
    g_dir = api->plugin_dir("boss_rush");
    g_test_kill = static_cast<int>(g.config_number("boss_rush.test_kill", 0));
    g_test_watch = g.config_bool("boss_rush.test_watch", false);
    g_test_drop_travel = static_cast<int>(g.config_number("boss_rush.test_drop_travel", 0));
    g_frame_seconds = g.config_number("boss_rush.test_frame_seconds", 0);
    g_t0 = std::chrono::steady_clock::now();
    g_auto_start = g.config_bool("boss_rush.auto_start", false);
    build_run();
    g.api->on_action("boss_rush", [](const char* action, void*) {
        const std::string a = action;
        if (a == "start") begin();
        else if (a == "stop") stop("停止");
        else if (a == "skip") skip();
    }, nullptr);
    g.every_frame(tick);
    g.on_world_load([](std::uint32_t block) {
        if (g_state == State::Waiting) {
            if (!g_restored) {
                g_restored = true;
                restore_leftovers();
            }
            // Start by itself: once this load's screen has gone (tick), so the
            // countdown is seen and the travel asked in the world.
            if (g_auto_start && !g_auto_started) {
                g_auto_started = true;
                g_auto_pending = true;
            }
        } else if (g_state == State::Warping || g_state == State::Fighting) {
            const Boss* b = g_run[g_round];
            // A death puts the player back at the lamp the round travelled to
            // (its block's, where the fog warp below picks them up); anywhere
            // but the boss's map, the round starts again.
            if (block != block_of(b)) {
                if (g_state == State::Warping) return;  // the map being left, not the arena yet
                // Sent away without dying: the game will not keep the player
                // in that map (The Old Hunters' maps, on a save without their
                // way in). Twice, and the round is skipped.
                if (g.stat("deaths") == g_deaths_at_arrival && ++g_refused >= 2) {
                    g.log("boss_rush: %s's map sends the player away on this save; skipped", b->name);
                    banner("到達できません: " + std::string(b->name) + " （このセーブではスキップ）", 6.0f);
                    g_round_times.push_back(0);
                    g_state = State::Cleared;
                    g_wait = 3.0;
                    return;
                }
                ++g_retries;
                if (g.stat("deaths") == g_deaths_at_arrival) {
                    g.log("boss_rush: sent away from %s's map without a death; once more", b->name);
                } else {
                    g.log("boss_rush: died to %s; back to its arena (retry %d)", b->name, g_retries);
                    banner("再挑戦 - " + std::string(b->name), 4.0f);
                }
                // The travel back, asked once the respawn's screen has gone and
                // the world has settled (travel_tick): past its fade-in.
                g_state = State::Retry;
                g_wait = 0;
                return;
            }
            if (g_state == State::Warping) {
                g_round_start = -1;
                g_deaths_at_arrival = g.stat("deaths");
                log_flags(b, "arrived");
                banner("ラウンド " + std::to_string(g_round + 1) + "/" + std::to_string(g_run.size()) + " - " + b->name, 5.0f);
            } else {
                // A death in the boss's own map: back at the block's lamp.
                ++g_retries;
                log_flags(b, "死亡後に復帰");
                banner("再挑戦 - " + std::string(b->name), 4.0f);
            }
            // A death since the last refill (here, or elsewhere before this
            // round's travel): vials and bullets back to the most.
            if (const std::int64_t deaths = g.stat("deaths"); deaths > g_deaths_refilled) {
                g_deaths_refilled = deaths;
                if (g.config_bool("boss_rush.refill", true)) refill();
            }
            // To the fog wall, whichever way the player came into the map, once
            // this load's screen has gone (fog_tick).
            g_fog_tries = 0;
            g_fog = 1;
            g_state = State::Fighting;
            g_fight_time = 0;
            g_watch_at = g_quiet_at = g_world_time;
        }
    });
    return 0;
}
