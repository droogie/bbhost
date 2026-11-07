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
//   refill = true       # blood vials and quicksilver bullets back to the most after a death (a practice: before every attempt)
//   practice_boss = "Ludwig, the Holy Blade"   # Practice: one boss, by name (any in kBosses)
//   practice_repeat = "Loop until stopped"     # or "Once", "N times" (practice_times attempts)
//   practice_times = 3
//   practice_phase = "2"                       # start its fights at phase 2 or 3 ("1": as the game does)
//   practice_boss_hp = "Normal"                # or "Can't die", "Infinite": the boss for practising against
//   practice_player_hp = "Normal"              # or "Can't die", "Infinite": the player, for learning its moves
//
// Practice (the menu's "Practice this boss") fights one boss again and
// again: each fight starts at its fog wall as a rush's round does, a death
// tries again from there, and a win wakes the boss and starts the next fight
// - once, N times, or until Stop. An attempt is a fight that ended, in a win
// or a death: Once and N times end after that many, whichever way each went.
// Restart (Skip does the same in a practice) starts the fight under way over
// from the fog wall, the boss woken anew, and is not an attempt. With Refill
// on, every attempt starts with the most vials and bullets: they are topped
// up for the first, after each win and after each death or Restart. Its
// attempts, wins, deaths, best and last clear times (from the fight's start,
// the boss's own "fight begun" event, to its defeat) and, from phase 1, the
// time the boss took to reach its phase 2, are on screen after each fight
// and in results.txt. Started at
// phase 2 or 3, each fight's boss is put just under that phase's HP threshold
// as the fight begins - the HP its event script (an HP-ratio condition) or
// its AI (GetHpRate) changes at - and the game plays its own change: the
// cutscene, the second character, the arena's change. kFights has each
// boss's thresholds and characters, from those scripts; a boss without a
// later phase (or a chalice boss: their phase events are not mapped yet)
// fights from the latest it has, and the message says so.
//
// Boss HP and Your HP are practice's cheats, for learning a fight rather than
// winning it: Can't die is the debug menu's No Dead on the boss's characters
// (or the player) - hits still take HP, down to 1 - and Infinite puts the HP
// back after every hit as well: the boss stays in the phase its fight began
// in, the player at full health. Neither is in a rush. A boss that cannot die
// is never defeated, so an attempt ends only in a death, a Restart or Stop.
//
// Play it on a save of its own: bosses give their rewards again, and while a
// run is under way the save holds woken bosses (restore.txt puts them back if
// the game is closed mid-run). Practice puts the save's flags back as a rush
// does, when it ends or is stopped.
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
#include "bbhost/engine/sprj/chr_ins.hpp"
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
    "Boss rush",
    "1.3.0",
    "bbhost",
    "Fight the game's bosses back to back, timed, or practise one again and again, from its later phases too, with a boss "
    "or a player that cannot die; deaths retry the fight and the save's bosses are restored at the end.",
};

// The practice list: every boss in kBosses by name, in its order (checked
// against kBosses below).
#define BOSS_RUSH_PRACTICE_BOSSES                                                                                              \
    "Cleric Beast|Father Gascoigne|Blood-starved Beast|Vicar Amelia|The Witch of Hemwick|Darkbeast Paarl|Shadow of Yharnam|"   \
    "Rom, the Vacuous Spider|The One Reborn|Martyr Logarius|Amygdala|Celestial Emissary|Ebrietas, Daughter of the Cosmos|"     \
    "Mergo's Wet Nurse|Ludwig, the Holy Blade|Living Failures|Lady Maria of the Astral Clocktower|Laurence, the First Vicar|" \
    "Orphan of Kos|Undead Giant|Merciless Watchers|Watchdog of the Old Lords|Beast-possessed Soul|Keeper of the Old Lords|"     \
    "Pthumerian Descendant|Bloodletting Beast|Pthumerian Elder|Abhorrent Beast|Yharnam, Pthumerian Queen"

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, nullptr, "Start it from the in-game plugin menu", "Play it on a save of its own: bosses give their rewards again."},
    {BB_OPT_ACTION, "start", "Start the boss rush", "Begins in a few seconds, from wherever you are."},
    {BB_OPT_ACTION, "skip", "Skip this round", "On to the next boss. In a practice it restarts the fight, as Restart does."},
    {BB_OPT_ACTION, "stop", "Stop", "Ends the run and puts the save's bosses back as they were."},
    {BB_OPT_HEADING, nullptr, "The run", "Read when a run starts."},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "order", "Order", nullptr, "story", 0, 0, "story|random"},
    {BB_OPT_TEXT | BB_OPT_LIVE, "seed", "Seed for the random order", nullptr, ""},
    {BB_OPT_BOOL | BB_OPT_LIVE, "dlc", "The Old Hunters' bosses", "Skipped on a save that cannot enter their maps.", "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "chalice", "Chalice dungeon bosses",
     "Ten found only in the chalice dungeons, in their fixed dungeons, after the others. An altar's slot is borrowed and given back.", "false"},
    {BB_OPT_INT | BB_OPT_LIVE, "first", "Start at boss", "1 is the first of the order; with Rounds 1, practice one boss.", "1", 1, 29},
    {BB_OPT_INT | BB_OPT_LIVE, "rounds", "Rounds", "0: every boss.", "0", 0, 29},
    {BB_OPT_BOOL | BB_OPT_LIVE, "awake", "Skip the bosses' first entrances", "Each round starts at the fog wall, as on a return. Off: as the first time (cutscenes).", "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "refill", "Refill vials and bullets",
     "To the most after each death; in a practice, before every attempt. Off: what you have left, as the game restocks it from storage.",
     "true"},
    {BB_OPT_BOOL, "auto_start", "Start by itself", "Begin a run when the world first loads, without the menu.", "false"},
    {BB_OPT_INT | BB_OPT_LIVE, "start_delay", "Seconds before the first round", nullptr, "10", 3, 60},
    {BB_OPT_HEADING, nullptr, "Practice",
     "One boss again and again, from its fog wall: a death tries again, a win wakes it for the next fight. Restart starts the "
     "fight over; Stop ends it. Skip the bosses' first entrances, Refill and the seconds before the first round apply."},
    {BB_OPT_ACTION, "practice", "Practice this boss", "Begins in a few seconds, from wherever you are."},
    {BB_OPT_ACTION, "restart", "Restart",
     "The practice's fight again from its fog wall, the boss woken anew. It is not counted as an attempt."},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "practice_boss", "Boss", "The Old Hunters' and the chalice bosses when the save can reach them.",
     "Cleric Beast", 0, 0, BOSS_RUSH_PRACTICE_BOSSES},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "practice_repeat", "Repeat",
     "Once or N times: that many attempts, each a win or a death (a Restart is not one). Loop: until Stop.", "Loop until stopped", 0,
     0, "Once|N times|Loop until stopped"},
    {BB_OPT_INT | BB_OPT_LIVE, "practice_times", "N", "Attempts, with Repeat N times: a win or a death each.", "3", 1, 99},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "practice_phase", "Start at phase",
     "2 or 3: as each fight begins the boss is put just under the HP its next phase starts at, and the game plays its own "
     "change. A boss without that phase starts at the latest it has (the message says so).",
     "1", 0, 0, "1|2|3"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "practice_boss_hp", "Boss HP",
     "Can't die: hits take its HP down to 1 and no further, its phases still come. Infinite: its HP goes back up after every "
     "hit, and it stays in the phase the fight began in. Either way the fight is not won: Restart or Stop ends it.",
     "Normal", 0, 0, "Normal|Can't die|Infinite"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "practice_player_hp", "Your HP",
     "Can't die: hits still take your HP, down to 1 and no further. Infinite: back to full after every hit. For learning a "
     "boss's moves and when to dodge them.",
     "Normal", 0, 0, "Normal|Can't die|Infinite"},
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
    // A flag outside the family that the boss's own events take for its
    // defeat: Ludwig's defeat event sets 9471, and while it is on every event
    // of his fight ends at once and the defeat event kills him as the map
    // loads (2004[4]) - woken with it on, he came back dead. Cleared with the
    // family, put back after the run.
    std::uint32_t defeated = 0;
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
constexpr Boss kBosses[] = {
    {"Cleric Beast", 12411700, 2412951, false, 500000, 0, {-154.14f, -27.02f, 35.16f}, -135.0f},
    {"Father Gascoigne", 12411800, 2412951, false, 271000, 0, {-50.18f, -40.82f, 71.02f}, -135.0f},
    {"Blood-starved Beast", 12301800, 2302950, false, 209000, 0, {-87.654f, -126.44f, 13.512f}, 129.825f},
    // The Grand Cathedral is where Eileen's quest ends (m24_00's event 12400650
    // and its flags 1360-1379): her placement there (2400902) and the hunter
    // she comes to fight (2400901) are switched off.
    {"Vicar Amelia", 12401800, 2402950, false, 502000, 0, {51.51f, 35.71f, 311.57f}, -150.0f, 2, {}, {2400901, 2400902}},
    {"The Witch of Hemwick", 12201800, 2202950, false, 210000, 0, {-329.75f, 2.466f, 698.21f}, 180.0f},
    {"Darkbeast Paarl", 12301700, 2302950, false, 508000, 0, {135.444f, -119.83f, -85.772f}, 113.5f},
    {"Shadow of Yharnam", 12701800, 2702950, false, 212010, 0, {-322.923f, -185.868f, 494.393f}, 44.779f},
    // Rom's fog stands in the doorway to the lake the player drops into: the
    // door (3201120) is opened as its own event keeps it (m32_00's 13200040,
    // slot 0: its flag, and 13200120, the opening's).
    {"Rom, the Vacuous Spider", 13201800, 3202950, false, 510000, 0, {-452.874f, -175.03f, 392.65f}, 70.0f, 2, {13200040, 13200120}},
    {"The One Reborn", 12801800, 2802950, false, 507000, 0, {381.167f, -122.49f, -215.583f}, -45.0f},
    // A save's first time in Cainhurst plays the carriage's arrival (m25_00's
    // event 12500000, which its own flag ends) - over the round's first
    // seconds.
    {"Martyr Logarius", 12501800, 2502950, false, 232000, 0, {33.93f, 112.32f, -330.56f}, -35.0f, 2, {12500000}},
    {"Amygdala", 13301800, 3302950, false, 512000, 0, {-47.183f, 1461.853f, -69.041f}, 112.0f},
    {"Celestial Emissary", 12421700, 2422950, false, 257000, 0, {-5.47f, 55.336f, 293.41f}, -60.0f},
    {"Ebrietas, Daughter of the Cosmos", 12421800, 2422950, false, 251000, 0, {94.0f, 4.2f, 387.0f}, -150.741f},
    // The Wet Nurse (entity 2600802) is made when the player walks into her
    // arena; 515000/516000 are the doubles that come with her.
    {"Mergo's Wet Nurse", 12601800, 2602950, false, 551000, 0, {122.14f, 1125.184f, -37.839f}, -90.0f},
    {"Ludwig, the Holy Blade", 13401800, 3402950, true, 451000, 0, {-407.93f, 1503.8f, -716.45f}, 0.0f, 1, {}, {}, 0, 0, 9471},
    // The Living Failures' fog stands in the balcony's doorway (door 3501120,
    // m35_00's 13501200 slot 0, and 13504220), as Rom's does.
    {"Living Failures", 13501850, 3502950, true, 403000, 0, {-378.54f, 1592.9f, -824.55f}, 90.0f, 1, {13501200, 13504220}},
    {"Lady Maria of the Astral Clocktower", 13501800, 3502950, true, 452000, 0, {-452.0f, 1595.51f, -824.41f}, 90.0f, 1},
    {"Laurence, the First Vicar", 13401850, 3402950, true, 450000, 0, {-450.9f, 1535.71f, -292.69f}, -150.0f, 1},
    {"Orphan of Kos", 13601800, 3602950, true, 453000, 0, {-700.2f, 1580.5f, -906.2f}, 0.0f, 1},
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
    {"Undead Giant", 12901800, 2902950, false, 313000, 0, {-22.8f, 31.53f, -91.0f}, 90.0f, 0, {12900532}, {}, 0x1d0a5a00, 12905502},
    {"Merciless Watchers", 12901801, 2902950, false, 304000, 0, {-130.0f, 10.5f, -55.0f}, 180.0f, 0, {12900536}, {}, 0x1d0a5a00, 12905511},
    {"Watchdog of the Old Lords", 12901802, 2902950, false, 501000, 0, {-22.637f, -17.5f, 52.0f}, 270.0f, 0, {12900538}, {}, 0x1d0a5a00, 12905516},
    {"Beast-possessed Soul", 12901800, 2902950, false, 75000, 0, {78.027f, 26.403f, 110.5f}, 270.0f, 0, {12900530}, {}, 0x1d145a00, 12905504},
    {"Keeper of the Old Lords", 12901801, 2902950, false, 216000, 0, {156.0f, 10.1f, 84.5f}, 90.0f, 0, {12900533}, {}, 0x1d145a00, 12905511},
    {"Pthumerian Descendant", 12901802, 2902950, false, 305000, 0, {-42.25f, -17.5f, 22.784f}, 0.0f, 0, {12900534}, {}, 0x1d145a00, 12905516},
    {"Bloodletting Beast", 12901803, 2902950, false, 509000, 0, {-32.422f, -66.5f, -35.75f}, 270.0f, 0, {12900541}, {}, 0x1d1e5a00, 12905525},
    {"Pthumerian Elder", 12901802, 2902950, false, 305000, 0, {-55.25f, -17.5f, 68.161f}, 0.0f, 0, {12900540}, {}, 0x1d1f5a00, 12905513},
    {"Abhorrent Beast", 12901802, 2902950, false, 504000, 0, {55.254f, -17.5f, -52.0f}, 90.0f, 0, {12900528}, {}, 0x1d2a5a00, 12905516},
    {"Yharnam, Pthumerian Queen", 12901802, 2902950, false, 306000, 0, {34.388f, -28.072f, -204.75f}, 90.0f, 0, {12900536}, {}, 0x1d325a00, 12905516},
};

constexpr bool same(const char* a, const char* b) {
    while (*a && *a == *b) ++a, ++b;
    return *a == *b;
}
// The practice list is kBosses' names, in order, '|' between.
constexpr bool practice_list_is_kbosses(const char* list) {
    for (const Boss& b : kBosses) {
        const char* n = b.name;
        while (*n && *n == *list) ++n, ++list;
        if (*n || (*list != '|' && *list != '\0')) return false;
        if (*list == '|') ++list;
    }
    return *list == '\0';
}
static_assert(practice_list_is_kbosses(BOSS_RUSH_PRACTICE_BOSSES), "the practice list must name kBosses in order");

// A boss's later phases, as its fight's own logic starts them: an HP ratio
// its event script waits for (condition 4[2], "a character's HP ratio",
// which then sends its AI a command, 2004[17], or plays the change itself)
// or its AI compares GetHpRate with (the battle program's Activate), from
// event/<map>.emevd.dcx and script/<map>.luabnd.dcx (bossprac notes have the
// events, programs and animations). chars: the fight's characters by entity
// id - the one the thresholds are on first, then those the fight's start
// links to it (2004[33]: they take the same damage, the second form a boss
// changes into) or that share its threshold (the Shadows: all three under
// it). A practice fight from phase 2 or 3 puts every one of them just under
// that phase's threshold once the fight has begun, as the damage would have.
// then: phase 3 is set only once phase 2's change has begun - that event
// flag on (> 0) or that SpEffect on chars[0] (< 0, negated) - where the
// game does not order the two itself (0: its own event or AI takes phase 2
// first whatever the HP).
struct Fight {
    const char* boss;     // kBosses' name
    std::int32_t chars[3];
    float below[2];       // the HP ratio phase 2 / phase 3 starts under; 0: no such phase
    float any_below = 0;  // phase 2 also starts when any one of chars is under this (the Shadows)
    std::int32_t then = 0;
};
constexpr Fight kFights[] = {
    // 12414707: under 0.7 the AI is told 1 (slot 1): its roar (3010).
    {"Cleric Beast", {2410800}, {0.7f, 0}},
    // The AI's own axe (Act28, 3028; SpEffect 5401) at 0.67; 12414807 under
    // 0.34: command 40, the change (Act29, 3030), the beast (2410811) put in
    // his place.
    {"Father Gascoigne", {2410810, 2410811}, {0.67f, 0.34f}, 0, -5401},
    // 12304807 under 0.67 (7010, command 100), 12304808 under 0.33 after it.
    {"Blood-starved Beast", {2300800}, {0.67f, 0.33f}},
    // 12404807 under 0.5: command 100 (Act20); her heal from 0.5 (AI).
    {"Vicar Amelia", {2400800}, {0.5f, 0}},
    // 12204810 at 0.5: the second witch (2200801) comes in.
    {"The Witch of Hemwick", {2200800}, {0.5f, 0}},
    // 12304707 under 0.67 (with its SpEffect 5402 window): command 100 (Act17).
    {"Darkbeast Paarl", {2300810}, {0.67f, 0}},
    // 12704812, each Shadow, once all three are at 0.5 or one is at 0.3.
    {"Shadow of Yharnam", {2700800, 2700801, 2700802}, {0.5f, 0}, 0.3f},
    // 13204807 at 0.75 (the teleport, 3021, flag 13204811), then at 0.5 (13204812).
    {"Rom, the Vacuous Spider", {3200800}, {0.75f, 0.5f}},
    // 12804871 under 0.5: the bell ringers' command 2.
    {"The One Reborn", {2800803}, {0.5f, 0}},
    // The AI at 0.7 (Act07, 3010; its number 0 set to 1).
    {"Martyr Logarius", {2500800}, {0.7f, 0}},
    // 13304807 under 0.7 (command 100), 13304808 under 0.3 after it (110).
    {"Amygdala", {3300800}, {0.7f, 0.3f}},
    // 12424790 under 0.6: command 40, the giant (2420811) put in its place (3025).
    {"Celestial Emissary", {2420810, 2420811}, {0.6f, 0}},
    // 12424980 under 0.5: command 100.
    {"Ebrietas, Daughter of the Cosmos", {2420800}, {0.5f, 0}},
    // The AI at 0.5: the combos and the counter the darkness comes with.
    {"Mergo's Wet Nurse", {2600802}, {0.5f, 0}},
    // 13404824 under 0.5, 13404825: the cutscene (34000030) and the Holy
    // Blade (3400801); 13404835 under 0.35 on the Holy Blade (7002, command 110).
    {"Ludwig, the Holy Blade", {3400800, 3400801}, {0.5f, 0.35f}, 0, 13404825},
    // 13505655 under 0.6, at a spawn cycle's end: the meteors (13504869).
    {"Living Failures", {3500850}, {0.6f, 0}},
    // The AI: Act30 at 0.6 (3030, SpEffect 5402), Act23 at 0.3 (3016, 5403).
    {"Lady Maria of the Astral Clocktower", {3500800}, {0.6f, 0.3f}},
    // The AI: at 0.8 (3010, 3033), at 0.4 (3036; SpEffect 5402).
    {"Laurence, the First Vicar", {3400850}, {0.8f, 0.4f}},
    // 13604820 under 0.5: command 100, the second form (3600801) put in its place (3031).
    {"Orphan of Kos", {3600800, 3600801}, {0.5f, 0}},
};
constexpr bool fights_are_kbosses() {
    for (const Fight& f : kFights) {
        bool found = false;
        for (const Boss& b : kBosses) found = found || (same(f.boss, b.name) && !b.dungeon);
        if (!found || !f.chars[0] || !(f.below[0] > 0 && f.below[0] < 1) || !(f.below[1] < f.below[0])) return false;
    }
    return true;
}
static_assert(fights_are_kbosses(), "every kFights row names a story boss of kBosses, phase 3 under phase 2");

const Fight* fight_of(const Boss* b) {
    for (const Fight& f : kFights) {
        if (same(f.boss, b->name)) return &f;
    }
    return nullptr;
}
// The flags a story boss's defeat event sets outside its family (2003[2] in
// the event the defeat flag is the id of, from the event scripts): the
// map's progress (2411, 3400..3403), the boss's own global flag (9456 ..
// 9471) and others (72400512, 5910, 6675). A win sets them again; a run keeps
// them as the save had them and puts them back with the family, so the save
// is left exactly as it was.
struct Aftermath {
    const char* boss;  // kBosses' name
    std::uint32_t flags[5];
};
constexpr Aftermath kAftermath[] = {
    {"Cleric Beast", {2411, 9456}},
    {"Father Gascoigne", {2412, 5910, 9457}},
    {"Blood-starved Beast", {2300, 9453, 72400512}},
    {"Vicar Amelia", {2400, 2401, 2402, 9455, 72400512}},
    {"The Witch of Hemwick", {2200, 5911, 9452, 72400512}},
    {"Darkbeast Paarl", {2301, 9454}},
    {"Shadow of Yharnam", {2700, 2701, 9463}},
    {"Rom, the Vacuous Spider", {3200, 9465}},
    {"The One Reborn", {2800, 9464}},
    {"Martyr Logarius", {2500, 9460}},
    {"Amygdala", {3300, 9466}},
    {"Celestial Emissary", {2420, 9458}},
    {"Ebrietas, Daughter of the Cosmos", {2421, 9459}},
    {"Mergo's Wet Nurse", {2601, 9462}},
    {"Ludwig, the Holy Blade", {3400, 3401, 3402, 3403, 9471}},
    {"Living Failures", {3510, 3511, 3512, 3517}},
    {"Lady Maria of the Astral Clocktower", {3510, 3511, 3512, 3513, 6675}},
    {"Laurence, the First Vicar", {3400, 3401, 3402, 3403}},
    {"Orphan of Kos", {3600}},
};
constexpr bool aftermath_is_story_bosses() {
    for (const Aftermath& a : kAftermath) {
        bool found = false;
        for (const Boss& b : kBosses) found = found || (same(a.boss, b.name) && !b.dungeon);
        if (!found) return false;
    }
    for (const Boss& b : kBosses) {
        if (b.defeated) {  // its defeated flag among its defeat's flags
            bool in = false;
            for (const Aftermath& a : kAftermath)
                for (const std::uint32_t f : a.flags) in = in || (same(a.boss, b.name) && f == b.defeated);
            if (!in) return false;
        }
    }
    return true;
}
static_assert(aftermath_is_story_bosses(), "kAftermath names story bosses; a boss's defeated flag is one of its defeat's");
const Aftermath* aftermath_of(const Boss* b) {
    for (const Aftermath& a : kAftermath) {
        if (same(a.boss, b->name)) return &a;
    }
    return nullptr;
}

// The latest phase, up to `wanted`, the game can be started at for this boss.
int phase_for(const Boss* b, int wanted) {
    const Fight* f = fight_of(b);
    if (!f || wanted < 2) return 1;
    if (wanted >= 3 && f->below[1] > 0) return 3;
    return 2;
}

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

// Practice: a run of one boss (g_run[0]) whose round starts again after a
// win or a death, until the Repeat setting is met or Stop.
bool g_practice = false;
enum class Repeat { Once, Times, Loop };
Repeat g_repeat = Repeat::Loop;
int g_times = 3;        // attempts (a win or a death each), with Repeat::Times
int g_phase = 1;        // the phase each fight is started at (phase_for: what the boss has)
int g_test_stop = 0;    // boss_rush.test_stop = N: a test switch - the practice is stopped, as Stop does, after N wins
bool g_test_practice = false;  // boss_rush.test_practice = true: a test switch - Start by itself begins a practice
// boss_rush.test_restart = S: a test switch - once, the first fight begun
// after an attempt has ended is restarted, as Restart does, S seconds of the
// world after it began (none ending meanwhile).
int g_test_restart = 0;
bool g_test_restarted = false;
// A practice's refill still to make at the next arrival, and why: its first
// attempt's, or a Restart's (a death's goes by the deaths count, a win's is
// made as the win's wait ends). nullptr: none.
const char* g_refill_due = nullptr;
struct PracticeStats {
    int wins = 0;
    std::int64_t deaths_at_start = 0;
    double best = -1, last = -1;                    // clear times, s
    int to_phase2_n = 0;                            // fights from phase 1 that reached phase 2
    double to_phase2_best = -1, to_phase2_last = -1;  // their times from the fight's start, s
};
PracticeStats g_ps;
// The fight under way, since the round's arrival (or its last death):
bool g_fight_open = false;  // its attempt counted: the stand at the fog, or the fight begun
bool g_begun = false;       // the boss's own "fight begun" flag has come on
double g_begun_at = 0;      // the world's time it came on
std::int64_t g_begun_attempts = 0;  // the attempts made by then
double g_fight_t0 = -1;     // real time it began (the begun flag; the fog stand until then)
int g_phase_set = 1;        // the phase its boss's HP has been put at
double g_phase_at = 0;      // the world's time of the next try at it
int g_phase_tries = 0;
bool g_phase2_seen = false;  // from phase 1: its boss has come under phase 2's threshold
// The HP Infinite keeps each of the fight's characters at, by data module:
// what it had once the fight had begun and its phase was set.
std::map<const bb::ChrDataModule*, std::int32_t> g_hold;

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
// Travels wait while the player is dead: one asked during the death screen cuts
// the respawn short, and the game plays the death again after it (a second
// death counted, the respawn after the travel).
bool dead() { return g.stat("hp", 1) <= 0; }
void dream_tick() {
    if (!g_to_dream || settled() < kSettle || dead()) return;
    g_to_dream = false;
    if (g.api->lamp_warp(kHuntersDream) != 0) g.log("boss_rush: could not travel to the Hunter's Dream");
}

void give_back();
void finish() {
    give_back();
    const std::int64_t deaths = g.stat("deaths") - g_deaths_at_start;
    std::ostringstream text;
    text << "Boss rush complete - " << clock_text(g_now - g_run_start) << ", " << deaths << (deaths == 1 ? " death" : " deaths");
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

std::string practice_title(const Boss* b);
std::string practice_tally();
// A practice's summary: on screen as it ends, and in results.txt after the
// lines its wins wrote as they fell.
std::string practice_summary(const Boss* b) {
    std::string t = "Practice - " + practice_title(b) + ": " + practice_tally();
    if (g_ps.wins) t += "; best " + clock_text(g_ps.best) + ", last " + clock_text(g_ps.last);
    if (g_ps.to_phase2_n)
        t += "; phase 2 after " + clock_text(g_ps.to_phase2_last) + " (best " + clock_text(g_ps.to_phase2_best) + ")";
    return t;
}
void results_line(const std::string& line) {
    std::ofstream f(g_dir + "/results.txt", std::ios::app);
    f << line << '\n';
}
// The practice over - its wins, or Stop: the summary, the save's flags back,
// and (once a fight has been travelled to) home to the Hunter's Dream.
void cheats_clear(const Boss* b, bool boss, bool player);
void practice_end(const char* how, bool home, const std::string& say = "") {
    const Boss* b = g_run[0];
    cheats_clear(b, true, true);
    give_back();
    const std::string sum = practice_summary(b);
    results_line(sum);
    g.log("boss_rush: practice %s: %s", how, sum.c_str());
    const std::string text = std::string("Practice ") + how + " - " + b->name + ": " + practice_tally() +
                             (g_ps.wins ? ", best " + clock_text(g_ps.best) : "");
    if (!say.empty()) banner(say, 8.0f);  // shown once the world is up
    else if (g_loading || !g_in_world) banner(text, 10.0f);  // over in a load: a last attempt's death, as it respawns
    else g.message(text, 10.0f);
    g_state = State::Done;
    if (home) to_dream();
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

std::uint32_t begun_flag(const Boss* b);
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
        if (b->defeated) failed += g.api->event_flag_set(b->defeated, 0) != 0;
        if (g_skip_intro) failed += g.api->event_flag_set(b->defeat_flag + b->met, 1) != 0;
        // A practice fight's start is that flag coming on: off for each one.
        if (g_practice) failed += g.api->event_flag_set(begun_flag(b), 0) != 0;
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
void refill(const char* why) {
    std::string t;
    for (const std::uint32_t id : {kBloodVial, kQuicksilverBullets}) {
        int have = 0, max = 0;
        if (!goods_count(id, &have, &max)) continue;
        if (max > have) call_at(kInventoryAdd, {reinterpret_cast<std::uint64_t>(equip_game_data()), kGoods, id, static_cast<std::uint64_t>(max - have), 1, 1}, 0);
        int now = have;
        goods_count(id, &now, &max);
        t += (t.empty() ? "" : ", ") + std::string(id == kBloodVial ? "vials " : "bullets ") + std::to_string(have) + " -> " + std::to_string(now);
    }
    g.log("boss_rush: refilled %s: %s", why, t.c_str());
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

// --- practice ------------------------------------------------------------
//
// A boss fight's own "fight begun": the event that switches its AI on and
// shows its health bar once the player has gone through the fog (defeat +
// 3002 for each story boss: 13404802 is Ludwig's), whose flag comes on as it
// ends; a chalice boss's `begun`.
std::uint32_t begun_flag(const Boss* b) { return b->dungeon ? b->begun : b->defeat_flag + 3002; }

std::string plural(std::int64_t n, const char* one) { return std::to_string(n) + " " + one + (n == 1 ? "" : "s"); }
std::int64_t practice_deaths() { return g.stat("deaths") - g_ps.deaths_at_start; }
// The attempts made: fights that ended, in a win or a death (a Restart ends
// none).
std::int64_t practice_attempts() { return g_ps.wins + practice_deaths(); }
// Once and N times end after that many attempts, whichever way each went.
bool practice_over() {
    return g_repeat != Repeat::Loop && practice_attempts() >= (g_repeat == Repeat::Once ? 1 : g_times);
}
// "2 wins in 3 attempts, 1 death"; with N times, "1 win in 2 of 3 attempts, 1 death"
std::string practice_tally() {
    const std::int64_t n = practice_attempts();
    const std::string made = g_repeat == Repeat::Times ? std::to_string(n) + " of " + plural(g_times, "attempt") : plural(n, "attempt");
    return plural(g_ps.wins, "win") + " in " + made + ", " + plural(practice_deaths(), "death");
}
std::string practice_title(const Boss* b) {
    return std::string(b->name) + (g_phase > 1 ? " from phase " + std::to_string(g_phase) : "");
}

// The fight from its round's arrival, or a death's respawn: nothing begun.
void reset_fight() {
    g_hold.clear();  // the next fight's HP is its own
    g_fight_open = g_begun = g_phase2_seen = false;
    g_fight_t0 = -1;
    g_phase_set = 1;
    g_phase_tries = 0;
}
// A fight opened once: at the stand at the fog, or as it begins. It is the
// next attempt (the same one again after a Restart) until a win or a death
// ends it.
void open_attempt(const Boss* b, const char* at) {
    if (g_fight_open) return;
    g_fight_open = true;
    g_fight_t0 = g_now;
    g.log("boss_rush: practice: %s, attempt %lld (%s)", b->name, static_cast<long long>(practice_attempts() + 1), at);
}

// A character by entity id, with the lookup the event instructions use, and
// its HP (the data module's, ChrIns +0x3b0 -> +0x20: what the health bar and
// the scripts' HP-ratio conditions read).
bb::ChrDataModule* chr_data_of(std::int32_t entity) {
    auto* ins = reinterpret_cast<bb::ChrIns*>(call_at(kChrByEntity, {static_cast<std::uint64_t>(entity), 0, 0}));
    return ins ? ins->chr_data() : nullptr;
}
float hp_ratio(const bb::ChrDataModule* d) { return static_cast<float>(d->hp) / static_cast<float>(d->max_hp); }

// Every character of the fight a little under `below` of its HP - as the
// damage would have left it, the ones linked to the first taking the same -
// and none raised or brought back. False while one is not there yet.
bool put_under(const Boss* b, const Fight* f, int phase) {
    const float below = f->below[phase - 2];
    std::string t;
    for (const std::int32_t e : f->chars) {
        if (!e) continue;
        bb::ChrDataModule* d = chr_data_of(e);
        if (!d || d->max_hp <= 0) return false;
        const auto to = static_cast<std::int32_t>(std::floor((below - 0.005f) * static_cast<float>(d->max_hp)));
        const std::int32_t was = d->hp;
        if (d->hp > to && to >= 1) d->hp = to;
        char one[96];
        std::snprintf(one, sizeof(one), "%s%d %d -> %d/%d", t.empty() ? "" : ", ", e, was, d->hp, d->max_hp);
        t += one;
    }
    g.log("boss_rush: practice: %s put at phase %d (HP under %.2f): %s", b->name, phase, static_cast<double>(below), t.c_str());
    return true;
}
// Phase 2 under way in its HP: every character of the fight under its
// threshold (or one under any_below), as the game's own conditions test it.
bool phase2_reached(const Fight* f) {
    bool all = true, any = false;
    int n = 0;
    for (const std::int32_t e : f->chars) {
        if (!e) continue;
        const bb::ChrDataModule* d = chr_data_of(e);
        if (!d || d->max_hp <= 0) {
            all = false;
            continue;
        }
        ++n;
        const float r = hp_ratio(d);
        all = all && r < f->below[0] && d->hp > 0;  // alive under it: a kill that skips phase 2 is not phase 2
        any = any || (f->any_below > 0 && r < f->any_below && d->hp > 0);
    }
    return n && (all || any);
}
// Phase 2's change begun, for a boss whose phase 3 waits for it (Fight::then).
bool then_met(const Fight* f) {
    if (f->then > 0) return g.flag(static_cast<std::uint32_t>(f->then));
    void* ins = reinterpret_cast<void*>(call_at(kChrByEntity, {static_cast<std::uint64_t>(f->chars[0]), 0, 0}));
    return ins && g.api->sp_effect_has(ins, -f->then) != 0;
}

// Each frame of a practice fight, in the world: its start, the phase it is
// put at, and from phase 1 the time to phase 2.
void practice_watch(const Boss* b) {
    if (!g_begun && g.flag(begun_flag(b))) {
        g_begun = true;
        open_attempt(b, "the fight begun");
        g_fight_t0 = g_now;  // the fight's own start, not the stand at the fog
        g_phase_at = g_begun_at = g_world_time;
        g_begun_attempts = practice_attempts();
        g.log("boss_rush: practice: %s's fight has begun (flag %u)", b->name, begun_flag(b));
    }
    const Fight* f = fight_of(b);
    if (!g_begun || !f) return;
    if (g_phase == 1 && !g_phase2_seen && phase2_reached(f)) {
        g_phase2_seen = true;
        const double t = g_now - g_fight_t0;
        ++g_ps.to_phase2_n;
        g_ps.to_phase2_last = t;
        if (g_ps.to_phase2_best < 0 || t < g_ps.to_phase2_best) g_ps.to_phase2_best = t;
        g.log("boss_rush: practice: %s under phase 2's HP (%.2f) %.1f s into the fight", b->name, static_cast<double>(f->below[0]), t);
        g.text_at("Phase 2 after " + clock_text(t), 5.0f, 0.03f, 0.84f, 1.0f, BB_TEXT_LEFT);
    }
    if (g_phase_set >= g_phase || g_world_time < g_phase_at) return;
    int next = g_phase_set + 1;
    if (next == 2 && g_phase == 3 && f->then == 0) next = 3;  // the game takes phase 2 first itself
    if (next == 3 && g_phase_set == 2 && !then_met(f)) {
        g_phase_at = g_world_time + 0.25;
        return;
    }
    if (put_under(b, f, next)) {
        g_phase_set = next;
        g_phase_at = g_world_time + 0.25;
    } else if (++g_phase_tries >= 40) {  // ten seconds of the world without its characters
        g.log("boss_rush: practice: %s's characters were not found; this fight is from phase %d", b->name, g_phase_set);
        g_phase_set = g_phase;
    } else {
        g_phase_at = g_world_time + 0.25;
    }
}

// Practice's cheats. No Dead is bit 4 of a character data module's flags
// (+0x200): sub_1e577d0, which the death code asks, tests it before the
// event system's immortality (bit 1, the fights' own scripts') and the debug
// menu's All and Player No Dead - so it is the debug menu's per-character No
// Dead, and nothing of the game's own sets or clears it.
constexpr std::uint16_t kNoDead = 4;
enum class Cheat { Normal, NoDeath, Infinite };
Cheat cheat_of(const char* key) {
    const std::string v = g.config(key, "Normal");
    return v == "Can't die" ? Cheat::NoDeath : v == "Infinite" ? Cheat::Infinite : Cheat::Normal;
}
bool g_boss_cheat = false, g_player_cheat = false;  // No Dead set on them this practice

bb::ChrDataModule* player_data() {
    for (const BbChr& c : g.characters()) {
        if (c.is_player && c.ins) return static_cast<bb::ChrIns*>(c.ins)->chr_data();
    }
    return nullptr;
}
// The boss's characters: its kFights row's, or those whose NpcParam names it
// (a chalice boss).
template <class Fn>
void each_boss_chr(const Boss* b, Fn&& fn) {
    if (const Fight* f = fight_of(b)) {
        for (const std::int32_t e : f->chars) {
            if (!e) continue;
            if (bb::ChrDataModule* d = chr_data_of(e)) fn(d);
        }
        return;
    }
    for (const BbChr& c : g.characters()) {
        const auto* npc = c.npc_param > 0 ? g.param<bb::params::NpcParam>(static_cast<std::uint32_t>(c.npc_param)) : nullptr;
        if (!npc || !boss_name(b, npc->nameId) || !c.ins) continue;
        if (bb::ChrDataModule* d = static_cast<bb::ChrIns*>(c.ins)->chr_data()) fn(d);
    }
}
// No Dead off again: from the player, and from the boss's characters while
// they are there (a boss unloaded took its flags with it).
void cheats_clear(const Boss* b, bool boss, bool player) {
    if (boss && g_boss_cheat) {
        if (b) each_boss_chr(b, [](bb::ChrDataModule* d) { d->flags = static_cast<std::uint16_t>(d->flags & ~kNoDead); });
        g_boss_cheat = false;
        g_hold.clear();
    }
    if (player && g_player_cheat) {
        if (bb::ChrDataModule* d = player_data()) d->flags = static_cast<std::uint16_t>(d->flags & ~kNoDead);
        g_player_cheat = false;
    }
}
// Each frame of a practice's fight in the world.
void cheats_tick(const Boss* b) {
    const Cheat boss = cheat_of("boss_rush.practice_boss_hp"), player = cheat_of("boss_rush.practice_player_hp");
    cheats_clear(b, boss == Cheat::Normal, player == Cheat::Normal);
    if (boss != Cheat::Normal) {
        // Infinite holds its HP only once the fight has begun and its phase is
        // set (practice_watch): before that the phase's HP is still to come.
        const bool hold = boss == Cheat::Infinite && g_begun && g_phase_set >= g_phase;
        if (boss != Cheat::Infinite) g_hold.clear();
        each_boss_chr(b, [&](bb::ChrDataModule* d) {
            d->flags = static_cast<std::uint16_t>(d->flags | kNoDead);
            if (d->hp < 1) d->hp = 1;
            if (!hold) return;
            const auto it = g_hold.try_emplace(d, d->hp).first;
            if (d->hp < it->second) d->hp = it->second;
        });
        g_boss_cheat = true;
    }
    if (player != Cheat::Normal) {
        if (bb::ChrDataModule* d = player_data()) {
            d->flags = static_cast<std::uint16_t>(d->flags | kNoDead);
            if (d->hp < 1) d->hp = 1;
            if (player == Cheat::Infinite && d->hp < d->max_hp) d->hp = d->max_hp;
            g_player_cheat = true;
        }
    }
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
        if (g_practice) open_attempt(b, "at the fog wall");
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
        g.message("Boss rush: could not reach " + std::string(b->name) + "'s fog wall - walk to it from the lamp", 6.0f);
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
    if (g_practice) {  // its summary on file too, the message the same
        practice_end("stopped", false, "Boss rush stopped: " + why + " to " + b->name + "'s arena");
        return;
    }
    give_back();
    g.message("Boss rush stopped: " + why + " to " + b->name + "'s arena", 8.0f);
    g_state = State::Done;
}

void travel_tick(const Boss* b) {
    if (g_loading) {  // begun (or another load): the arrival is on_world_load's
        g_travel_began = true;
        return;
    }
    if (settled() < kSettle || (g_travel_at >= 0 && g_world_time - g_travel_at < kTravelWait) || dead()) return;
    if (g_travel_at >= 0) {
        g.log("boss_rush: the travel to %s's arena %s; again", b->name,
              g_travel_began ? "did not arrive (a load, but to another map)" : "was not begun (no loading screen)");
    }
    if (g_travel_asks >= kTravelAsks) {
        travel_failed(b, g_travel_began ? "the game would not travel" : "the game did not begin the travel");
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
        travel_failed(b, "bbhost refused the travel");
    }
}

void begin();
void practice();
void stop(const char* why);
void restart();
void start_round() {
    const Boss* b = g_run[g_round];
    wake(b);
    if (g_practice)
        g.log("boss_rush: practice: %s, attempt %lld (%s)", b->name, static_cast<long long>(practice_attempts() + 1),
              practice_tally().c_str());
    else g.log("boss_rush: round %zu - %s", g_round + 1, b->name);
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
        if (g_test_practice) practice();
        else begin();
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
                    if (g_practice) g_original[begun_flag(b)] = g.flag(begun_flag(b));
                    if (b->defeated) g_original[b->defeated] = g.flag(b->defeated);
                    if (const Aftermath* a = aftermath_of(b)) {
                        for (const std::uint32_t f : a->flags) {
                            if (f) g_original[f] = g.flag(f);
                        }
                    }
                }
                for (const std::uint32_t f : b->set_flags) {
                    if (f) g_original[f] = g.flag(f);
                }
            }
            save_restore_file();
            g_ps.deaths_at_start = g_deaths_at_start;
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
            if (g_practice) {
                practice_watch(b);
                cheats_tick(b);
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
        if (g.flag(b->defeat_flag) && g_practice) {
            // Its time from the fight's start (the stand at the fog until the
            // fight began), the best and the last, on screen and on file.
            const double took = g_fight_t0 >= 0 ? g_now - g_fight_t0 : 0.0;
            if (!g_fight_open) open_attempt(b, "the boss's defeat");
            ++g_ps.wins;
            g_ps.last = took;
            const bool best = g_ps.best < 0 || took < g_ps.best;
            if (best) g_ps.best = took;
            g.message(std::string(b->name) + " defeated - " + clock_text(took) + (best ? " (best)" : " (best " + clock_text(g_ps.best) + ")") +
                          " - " + practice_tally(),
                      6.0f);
            results_line("Practice - " + practice_title(b) + ": win " + std::to_string(g_ps.wins) + " in " + clock_text(took) + ", attempt " +
                         std::to_string(practice_attempts()));
            g.log("boss_rush: practice: %s defeated in %s (%.1f s), %s", b->name, clock_text(took).c_str(), took, practice_tally().c_str());
            reset_fight();
            g_state = State::Cleared;
            g_wait = 10.0;  // time to pick up what it dropped
            break;
        }
        if (g.flag(b->defeat_flag)) {
            const double took = g_round_start >= 0 ? g_now - g_round_start : 0.0;
            g_round_times.push_back(took);
            g.message(std::string(b->name) + " defeated - " + clock_text(took), 6.0f);
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
        if (g_practice && g_test_restart && !g_test_restarted && g_begun && g_begun_attempts >= 1 &&
            practice_attempts() == g_begun_attempts && g_world_time - g_begun_at >= g_test_restart) {
            g_test_restarted = true;
            g.log("boss_rush: test restart: %.1f s into attempt %lld", g_world_time - g_begun_at,
                  static_cast<long long>(practice_attempts() + 1));
            restart();
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
            if (g_practice) {
                // The win's refill, its drops picked up; then the next fight,
                // until the Repeat setting is met.
                if (g.config_bool("boss_rush.refill", true)) refill("after a win");
                if (g_test_stop && g_ps.wins >= g_test_stop) stop("stopped");
                else if (!practice_over()) start_round();
                else practice_end("done", true);
            } else if (++g_round >= g_run.size()) {
                finish();
            } else {
                start_round();
            }
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
        bb::Rng rng(bb::seed_of(g.config("boss_rush.seed", "boss rush")));
        rng.shuffle(g_run);
    }
    if (const int first = static_cast<int>(g.config_number("boss_rush.first", 1)); first > 1 && static_cast<std::size_t>(first) <= g_run.size())
        g_run.erase(g_run.begin(), g_run.begin() + (first - 1));
    if (const int rounds = static_cast<int>(g.config_number("boss_rush.rounds", 0)); rounds > 0 && static_cast<std::size_t>(rounds) < g_run.size())
        g_run.resize(static_cast<std::size_t>(rounds));
}

void begin() {
    if (g_state != State::Waiting && g_state != State::Done) {
        g.message("A boss rush is already under way", 3.0f);
        return;
    }
    if (g.stat("hp", -1) < 0) {
        g.message("Load into the world first", 3.0f);
        return;
    }
    build_run();
    if (g_run.empty()) return;
    g_practice = false;
    g_round = 0;
    g_retries = g_refused = 0;
    g_round_times.clear();
    g_original.clear();
    g_to_dream = false;
    g_state = State::Countdown;
    g_wait = g_start_delay;  // seconds in the world: a loading screen holds it
    g.log("boss_rush: %zu bosses, %s order", g_run.size(), g.config("boss_rush.order", "story").c_str());
    g.message("Boss rush: " + std::to_string(g_run.size()) + " bosses - the first in " + std::to_string(g_start_delay) +
                  " seconds",
              6.0f);
}

// Practice: the boss the settings name, its run of one, the round's
// countdown as a rush's.
void practice() {
    if (g_state != State::Waiting && g_state != State::Done) {
        g.message("A boss rush or practice is already under way", 3.0f);
        return;
    }
    if (g.stat("hp", -1) < 0) {
        g.message("Load into the world first", 3.0f);
        return;
    }
    build_run();  // the shared settings (start_delay, awake)
    const std::string name = g.config("boss_rush.practice_boss", kBosses[0].name);
    const Boss* b = nullptr;
    for (const Boss& k : kBosses) {
        if (name == k.name) b = &k;
    }
    if (!b) {
        g.message("Boss rush: no boss is called " + name, 4.0f);
        return;
    }
    const std::string repeat = g.config("boss_rush.practice_repeat", "Loop until stopped");
    g_repeat = repeat == "Once" ? Repeat::Once : repeat == "N times" ? Repeat::Times : Repeat::Loop;
    g_times = std::max(1, static_cast<int>(g.config_number("boss_rush.practice_times", 3)));
    const int wanted = std::clamp(static_cast<int>(g.config_number("boss_rush.practice_phase", 1)), 1, 3);
    g_phase = phase_for(b, wanted);
    g_run.assign(1, b);
    g_practice = true;
    g_ps = PracticeStats{};
    g_refill_due = "for the first attempt";
    g_test_restarted = false;
    reset_fight();
    g_round = 0;
    g_retries = g_refused = 0;
    g_round_times.clear();
    g_original.clear();
    g_to_dream = false;
    g_state = State::Countdown;
    g_wait = g_start_delay;
    // Said once, here: a phase the boss does not have, and what it starts at.
    std::string note;
    if (wanted > g_phase) {
        note = b->dungeon ? " - chalice bosses start at phase 1" : g_phase == 1 ? " - it has no later phase" : " - it has no phase 3";
    }
    const std::string goal = g_repeat == Repeat::Once ? "once" : g_repeat == Repeat::Times ? plural(g_times, "attempt") : "until stopped";
    g.log("boss_rush: practice: %s from phase %d (asked %d), %s", b->name, g_phase, wanted, goal.c_str());
    g.message("Practice: " + practice_title(b) + note + ", " + goal + " - the first fight in " + std::to_string(g_start_delay) + " seconds",
              6.0f);
}

// Ends a run before its last boss: the save's flags as they were, and back to
// the Hunter's Dream once a round has begun.
void stop(const char* why) {
    if (g_state == State::Waiting || g_state == State::Done) return;
    const bool travelled = g_state != State::Countdown;
    if (g_practice) {
        g.log("boss_rush: practice %s", why);
        practice_end(why, travelled);
        return;
    }
    give_back();
    g.log("boss_rush: %s at round %zu", why, g_round + 1);
    g.message("Boss rush " + std::string(why), 4.0f);
    g_state = State::Done;
    if (travelled) to_dream();
}

// Practice's Restart: the fight under way over, neither a win nor a death and
// so not an attempt - the same one again from the fog wall, the boss woken
// anew (wake), and refilled as it arrives. In a rush there is none: Skip moves
// it on.
void restart() {
    if (!g_practice || g_state == State::Waiting || g_state == State::Done) {
        g.message(g_state == State::Waiting || g_state == State::Done ? "No practice is under way"
                                                                      : "Restart is for a practice - Skip moves the rush on",
                  3.0f);
        return;
    }
    if (g_state != State::Fighting && g_state != State::Warping && g_state != State::Retry) return;
    if (dead()) {
        // Dead: the death is this attempt, and its respawn brings the next one
        // to the fog wall (travels wait for it, dream_tick).
        g.log("boss_rush: practice: %s's restart refused - the player is dead; the respawn restarts it", g_run[0]->name);
        g.message("A death restarts it - back at the fog wall after the respawn", 3.0f);
        return;
    }
    g.log("boss_rush: practice: %s's fight restarted, not counted; attempt %lld again from the fog wall", g_run[0]->name,
          static_cast<long long>(practice_attempts() + 1));
    g.message("Restart - " + std::string(g_run[0]->name), 3.0f);
    g_refill_due = "after a restart";
    reset_fight();
    g_state = State::Retry;
    g_wait = 0;
}

void skip() {
    if (g_practice) {
        restart();  // in a practice, Skip is its Restart
        return;
    }
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
    g_test_stop = static_cast<int>(g.config_number("boss_rush.test_stop", 0));
    g_test_practice = g.config_bool("boss_rush.test_practice", false);
    g_test_restart = static_cast<int>(g.config_number("boss_rush.test_restart", 0));
    g_t0 = std::chrono::steady_clock::now();
    g_auto_start = g.config_bool("boss_rush.auto_start", false);
    build_run();
    g.api->on_action("boss_rush", [](const char* action, void*) {
        const std::string a = action;
        if (a == "start") begin();
        else if (a == "practice") practice();
        else if (a == "stop") stop("stopped");
        else if (a == "skip") skip();
        else if (a == "restart") restart();
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
            // A practice whose last attempt was a death: over as the respawn
            // loads (a win's last attempt ends it in Cleared).
            if (g_practice && practice_over()) {
                g.log("boss_rush: practice: %s's attempt %lld ended in a death, the last", b->name,
                      static_cast<long long>(practice_attempts()));
                practice_end("done", true);
                return;
            }
            // A death puts the player back at the lamp the round travelled to
            // (its block's, where the fog warp below picks them up); anywhere
            // but the boss's map, the round starts again.
            if (block != block_of(b)) {
                if (g_state == State::Warping) return;  // the map being left, not the arena yet
                // Sent away without dying: the game will not keep the player
                // in that map (The Old Hunters' maps, on a save without their
                // way in). Twice, and the round is skipped.
                if (g.stat("deaths") == g_deaths_at_arrival && ++g_refused >= 2 && g_practice) {
                    g.log("boss_rush: %s's map sends the player away on this save; the practice ends", b->name);
                    practice_end("ended", true, "Cannot reach " + std::string(b->name) + " on this save - practice ended");
                    return;
                }
                if (g.stat("deaths") == g_deaths_at_arrival && g_refused >= 2) {
                    g.log("boss_rush: %s's map sends the player away on this save; skipped", b->name);
                    banner("Cannot reach " + std::string(b->name) + " on this save - skipped", 6.0f);
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
                    banner("Again - " + std::string(b->name) + (g_practice ? " (" + practice_tally() + ")" : ""), 4.0f);
                }
                reset_fight();
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
                if (!g_practice) banner("Round " + std::to_string(g_round + 1) + "/" + std::to_string(g_run.size()) + " - " + b->name, 5.0f);
                else banner("Practice - " + practice_title(b) + (practice_attempts() ? " (" + practice_tally() + ")" : ""), 5.0f);
            } else {
                // A death in the boss's own map: back at the block's lamp.
                ++g_retries;
                log_flags(b, "back after a death");
                banner("Again - " + std::string(b->name) + (g_practice ? " (" + practice_tally() + ")" : ""), 4.0f);
            }
            if (g_practice) {
                // A fight anew. Its "fight begun" is the map's own and goes with
                // each load; one still on is taken off, so its start is seen.
                reset_fight();
                if (g.flag(begun_flag(b))) {
                    g.log("boss_rush: practice: %s's fight-begun flag %u was on at the arrival; off", b->name, begun_flag(b));
                    g.set_flag(begun_flag(b), false);
                }
            }
            // A death since the last refill (here, or elsewhere before this
            // round's travel): vials and bullets back to the most. A practice
            // also refills its first attempt and a Restart's here (a win's is
            // made as the win's wait ends): every attempt starts full.
            const std::int64_t deaths = g.stat("deaths");
            const char* why = deaths > g_deaths_refilled ? "after a death" : g_practice ? g_refill_due : nullptr;
            g_deaths_refilled = std::max(g_deaths_refilled, deaths);
            g_refill_due = nullptr;
            if (why && g.config_bool("boss_rush.refill", true)) refill(why);
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
