/* Map validation probe (tools/map_validation.sh): ground truth for the
 * private server's world map. Test-only: it moves the player, so it needs
 * the host's BBHOST_TEST_WARP=1 and is never installed by default.
 *
 * Every second it logs where the player is:
 *     mapval pos <epoch s> <block hex> <x> <y> <z> <yaw> <hp>
 * as they happen, a loading screen coming and going (mapval loading-begins,
 * loading-ends), the player's death (mapval death) and each time the game
 * places the player in a map - a load, a death's respawn, a lamp (mapval
 * placed: the host's on_world_load);
 * and, once the player has been in the world BBHOST_MAPVAL_START seconds
 * (default 55: the frozen seed's scripted death and respawn are over), it
 * runs the tour in BBHOST_MAPVAL_TOUR, one step a line:
 *     warp <name> <x> <y> <z> <yaw> <hold s>  put the player there, hold
 *     warpb <name> <block hex> <x> <y> <z> <yaw> <hold s>
 *                                             the same into another (resident)
 *                                             block, 18010000 for m24_01_00_00
 *     kill <radius m>                         the nearest live enemy: its HP
 *                                             to 1, the player next to it
 *                                             facing it until it dies (the
 *                                             driver presses attack)
 *     falldeath <name> <x> <y> <z> <height>   HP to 1, dropped from height
 *                                             above the point: a death there
 *     travel <return point> <block hex> <name> [<event flag>[=<0|1>]...]
 *                                             lamp travel (a full map load,
 *                                             api->lamp_warp); done once the
 *                                             host has placed the player in a
 *                                             map since the request and the
 *                                             player stands still in that block
 *                                             with the loading screen gone. Up
 *                                             to four event flags are set
 *                                             first, as the load reads them:
 *                                             <flag> on (a door's "opened",
 *                                             say), <flag>=0 off (a cutscene's
 *                                             "seen", so it plays). Failed
 *                                             after BBHOST_MAPVAL_TRAVEL_S
 *                                             seconds (150; a slow renderer
 *                                             such as lavapipe needs more)
 *     walk <name> <s>                         log walk-ready and give the
 *                                             driver <s> seconds to walk
 *     wait <s>
 * and, for tools/area_check.py, which measures each place it stands in:
 *     hold <s>                                wait with the player's HP kept full
 *     flips <n>                               wait, HP kept full, until the
 *                                             loading screen has been gone for
 *                                             <n> frames: on a slow renderer
 *                                             (lavapipe, ~0.4 frames a second)
 *                                             its fade-out takes 20-30 frames,
 *                                             far past a settle in seconds
 *     mark <name>                             log `mapval mark <name> <t> <block>
 *                                             <x> <y> <z> yaw <yaw> hp <hp> loading
 *                                             <0|1>`: where the player is then
 *     camera                                  the follow camera back behind the
 *                                             player, facing as the player does
 *     quiet <radius m>                        switch off the ordinary enemies
 *                                             within the radius (the game's own
 *                                             "disable character"; a load makes
 *                                             them again), so the place holds still
 *     dump <name>                             this frame to
 *                                             $BBHOST_MAPVAL_DUMPS/<name>.ppm
 *     f12 <name>                              an F12 capture of the next frame
 *                                             (the host logs its folder)
 *     capture <n> [<name>]                    let BBHOST_CAPTURE_DRAW take up to n
 *                                             draws, into
 *                                             $BBHOST_MAPVAL_CAPTURES/<name>; 0 stops
 * (the last two ask the host through BBHOST_TEST_REQUESTS, hle/video.cpp),
 * and for tools/full_tour.py, which tours every map:
 *     god [hide|off]                          the debug menu's Player No Dead
 *                                             (CHR DBG, 0x593e888) on, and with
 *                                             "hide" Player Hide (0x593e88e): the
 *                                             enemies do not notice the player;
 *                                             "off": both off again
 *     pitch <radians>|off                     hold the follow camera's pitch
 *                                             (+0x140, and the reset's +0x148 and
 *                                             the return's +0x150 so it does not
 *                                             ease back; positive looks down)
 *                                             every frame until "pitch off"
 *     turn <n> <s>                            the follow camera through 360
 *                                             degrees in n headings, s seconds
 *                                             each (its yaw, ChrExFollowCam
 *                                             +0x144, and the reset's +0x14c)
 *     flags <id>=<0|1> ...                    event flags set now (up to 8): the
 *                                             world's time (9800-9802) before a
 *                                             travel, say
 *     flaglog <first> <n>                     log n flags from first as a number
 *     chalice <map id hex> <slot> <name>      a fixed chalice dungeon: the altar
 *                                             slot (0-6) holds its map id and is
 *                                             the one in use, then the lamp travel
 *                                             to 2902950 enters it
 *     bits <first> <n> <hex>                  n event flags from first written as
 *                                             one number, the first flag its top
 *                                             bit (as the game reads them): a Root
 *                                             dungeon's altar slot, say, before its
 *                                             chalice step
 *     remo <id> <mode> <skip s> <max s>       a cutscene, as the event scripts'
 *                                             2002[03] asks for one (0x171a9f0);
 *                                             Options pressed (the game's skip)
 *                                             once it has played skip s; done
 *                                             when it ends or after max s
 *     movie <id> <skip s> <max s>             a movie, as RequestPlayMovie asks
 *                                             (0x1734d90), Options after skip s
 *     exit                                    the run ends the way
 *                                             BBHOST_EXIT_SECONDS ends one
 * Each step logs what it did with the time, so the dataset script can find
 * the play-log samples of the same seconds. While a warp or kill step runs
 * the player's HP is kept full (the seed stands among enemies).
 *
 * Build: cc -shared -fPIC -O2 -Iinclude -o mapval.so tools/mapval_plugin.c -lm
 */
#include "bbhost_plugin.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const BbHostApi* api;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}

static int ok(uint64_t p) {
    return ((p >= 0x100000000ull && p < 0x800000000ull) || (p >= 0x7e0000000000ull && p < 0x7ffffffff000ull)) && (p & 7) == 0;
}
static uint64_t rd64(uint64_t a) {
    return ok(a & ~7ull) ? *(const uint64_t*)(uintptr_t)a : 0;
}

/* --- loading screens -------------------------------------------------------- */

/* A loading screen is up: the game's NowLoading request (a byte at
 * 0x596286b) with no in-game frame run since (CSNowLoadingHelper, *0x593e8b8,
 * +0x50 or +0x51) - the host's own test (src/engine/loading.cpp), which
 * prints `loading:`. 0 when it cannot be read. A lamp travel puts the player
 * in its block well before the screen goes: a load's flips complete at once,
 * so the travel's ninety still frames came and went under it. */
static int loading_screen(void) {
    uint8_t requested = 0;
    uint64_t helper = 0;
    if (api->read(0x596286b, &requested, 1) || !requested) return 0;
    if (api->read(0x593e8b8, &helper, 8) || !ok(helper)) return 1;
    const volatile uint8_t* h = (const volatile uint8_t*)(uintptr_t)helper;
    return !(h[0x50] || h[0x51]);
}
static int loading;        /* this frame's answer */
static double loading_t0;  /* when the screen came up */
static uint64_t frame_no;           /* frames seen (on_frame calls) */
static uint64_t loading_end_frame;  /* frame_no when the loading screen last went ("flips") */

/* --- the tour ------------------------------------------------------------ */

enum { S_WARP, S_KILL, S_FALL, S_WAIT, S_TRAVEL, S_WALK, S_HOLD, S_MARK, S_CAMERA, S_QUIET, S_DUMP, S_CAPTURE, S_F12,
       S_GOD, S_TURN, S_FLAGS, S_FLAGLOG, S_REMO, S_MOVIE, S_EXIT, S_BITS, S_PITCH, S_FLIPS };
#define MAX_STEPS 4096
typedef struct {
    int kind;
    char name[64];
    float x, y, z, yaw, hold;
    int team;
    uint32_t block; /* warp into this block (0: the current one) */
    uint32_t flags[8]; /* travel: event flags turned on before it; flags: the flags */
    uint8_t values[8]; /* flags: their values */
    int nflags;
    uint32_t dungeon; /* chalice: the fixed dungeon's map id (m29_10_90_00: 1d0a5a00) */
} Step;
static Step steps[MAX_STEPS];
static int nsteps, cur = -1;
static double step_t0, last_pos_t, first_seen_t, start_after = 55, last_seen_t;
static int phase;         /* inside a step */
static int tour_done;
static uint64_t kill_chr; /* the enemy a kill step picked */
static double kill_last_warp;
static int fall_died;
static double near_last;
static int travel_failed, travel_still;
static int tour_rp;          /* the last travel's return point and block: */
static uint32_t tour_block;  /* a warp outside that block travels back first */
static int retravel;
static int retravel_step = -1, retravel_count;
static int warp_retries;
static float travel_last[3];
static double travel_still_t;  /* when the player came to rest for the travel's stillness */
static double travel_loading;  /* loading-screen seconds within the current travel step */
/* When the host last placed the player in a map (its on_world_load), on now_s()'s clock. A travel is done only after
 * a placement that came after its request: a player already standing still in the travel's block - a save made
 * there, or the tour's last place in the same block - would otherwise end it before the warp's own load wherever
 * the warp starts late (lavapipe: over a minute after the request; a GPU at 60 fps: under a second). */
static volatile double placed_t = -1;
static int travel_wait_logged;
static int dump_chrs = 0, want_team = -1;
static int warp_tries = 10; /* BBHOST_MAPVAL_WARP_TRIES */
static double travel_limit = 150; /* BBHOST_MAPVAL_TRAVEL_S: the seconds a travel may take */
/* BBHOST_MAPVAL_DEADLINE=<s>: that long after the probe loads, a tour still
 * running skips to its exit step (a harness's lock hold has a limit; the run
 * still ends with its reports and cache saves). */
static double deadline_s, loaded_t;
static int deadline_hit;
/* A warp onto ground whose collision is not loaded yet drops the player
 * through the world (Player No Dead does not stop the kill plane under it):
 * the warp is made again, and after two falls the player goes back to the
 * last place that held them. */
static int fall_retries;
static float good_pos[3];
static int have_good;
/* Falling, not standing lower than the navmesh face (a slope, a ledge under
 * a bridge): the height sampled every 0.25 s, more than 2 m lost in 0.5 s. */
static float fall_hist[3];
static int fall_n;
static double fall_sample_t;
/* BBHOST_MAPVAL_FLOAT=1: the Havok world's gravity (*(*(*0x593d700 + 0x28) +
 * 8) + 0x5b0, the vector plugins/mutators scales) held at 0 through a run of
 * warps, so a player put where the collision has not streamed in yet stays
 * there and sees the place; back to the game's before a travel, a cutscene, a
 * movie and the exit. */
static int float_mode, grav_off;
static float grav_saved = -9.8f;
static float* world_gravity(void) {
    uint64_t sys = 0;
    if (api->read(0x593d700, &sys, 8) || !ok(sys)) return NULL;
    uint64_t wrap = rd64(sys + 0x28), world = ok(wrap) ? rd64(wrap + 8) : 0;
    if (!ok(world)) return NULL;
    float* v = (float*)(uintptr_t)(world + 0x5b0);
    if (v[0] != 0.0f || v[2] != 0.0f || !(v[1] <= 0.0f && v[1] > -1000.0f)) return NULL;
    return v;
}
static void gravity_hold(int off, double t) {
    if (!float_mode || off == grav_off) return;
    float* v = world_gravity();
    if (!v) return;
    if (off) {
        if (v[1] < 0.0f) grav_saved = v[1];
        v[1] = 0.0f;
    } else {
        v[1] = grav_saved;
    }
    grav_off = off;
    api->log("mapval gravity %.3f %s", t, off ? "held at 0" : "back");
}

static void load_tour(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        api->log("mapval: no tour file %s", path);
        return;
    }
    char line[512];
    while (fgets(line, sizeof line, f) && nsteps < MAX_STEPS) {
        Step s;
        memset(&s, 0, sizeof s);
        char kind[32] = "";
        if (line[0] == '#' || sscanf(line, "%31s", kind) != 1) continue;
        if (!strcmp(kind, "warp") && sscanf(line, "%*s %63s %f %f %f %f %f", s.name, &s.x, &s.y, &s.z, &s.yaw, &s.hold) == 6)
            s.kind = S_WARP;
        else if (!strcmp(kind, "warpb") && sscanf(line, "%*s %63s %x %f %f %f %f %f", s.name, &s.block, &s.x, &s.y, &s.z, &s.yaw, &s.hold) == 7)
            s.kind = S_WARP;
        else if (!strcmp(kind, "kill") && sscanf(line, "%*s %f", &s.hold) == 1) {
            s.kind = S_KILL, strcpy(s.name, "kill");
            /* kill <radius> [x y z [team]]: around that point instead of the
             * player, and only characters of that team type (23: the
             * hostile ones seen in m24_01) */
            float team = -1;
            if (sscanf(line, "%*s %*f %f %f %f %f", &s.x, &s.y, &s.z, &team) >= 3) s.yaw = 1;
            s.hold = s.hold + 0; /* radius */
            s.team = (int)team;
        }
        else if (!strcmp(kind, "falldeath") && sscanf(line, "%*s %63s %f %f %f %f", s.name, &s.x, &s.y, &s.z, &s.hold) == 5)
            s.kind = S_FALL;
        else if (!strcmp(kind, "travel") && sscanf(line, "%*s %d %x %63s", &s.team, &s.block, s.name) == 3) {
            s.kind = S_TRAVEL; /* travel <return point id> <block hex> <name> [<event flag>[=<0|1>]...] */
            int used = 0;
            sscanf(line, "%*s %*d %*x %*63s%n", &used);
            const char* q = used > 0 ? line + used : "";
            while (s.nflags < 4) {
                unsigned id = 0, v = 1;
                int n = 0;
                if (sscanf(q, " %u=%u%n", &id, &v, &n) != 2 && (v = 1, sscanf(q, " %u%n", &id, &n) != 1)) break;
                s.flags[s.nflags] = id, s.values[s.nflags] = (uint8_t)(v != 0), ++s.nflags;
                q += n;
            }
        }
        else if (!strcmp(kind, "walk") && sscanf(line, "%*s %63s %f", s.name, &s.hold) == 2)
            s.kind = S_WALK; /* walk <name> <seconds>: the driver walks */
        else if (!strcmp(kind, "wait") && sscanf(line, "%*s %f", &s.hold) == 1)
            s.kind = S_WAIT, strcpy(s.name, "wait");
        else if (!strcmp(kind, "hold") && sscanf(line, "%*s %f", &s.hold) == 1)
            s.kind = S_HOLD, strcpy(s.name, "hold");
        else if (!strcmp(kind, "flips") && sscanf(line, "%*s %f", &s.hold) == 1)
            s.kind = S_FLIPS, strcpy(s.name, "flips");
        else if (!strcmp(kind, "mark") && sscanf(line, "%*s %63s", s.name) == 1)
            s.kind = S_MARK;
        else if (!strcmp(kind, "camera"))
            s.kind = S_CAMERA, strcpy(s.name, "camera");
        else if (!strcmp(kind, "quiet") && sscanf(line, "%*s %f", &s.hold) == 1)
            s.kind = S_QUIET, strcpy(s.name, "quiet"); /* hold: the radius */
        else if (!strcmp(kind, "dump") && sscanf(line, "%*s %63s", s.name) == 1)
            s.kind = S_DUMP;
        else if (!strcmp(kind, "f12") && sscanf(line, "%*s %63s", s.name) == 1)
            s.kind = S_F12;
        else if (!strcmp(kind, "capture") && sscanf(line, "%*s %d", &s.team) == 1) {
            s.kind = S_CAPTURE; /* team: the draws */
            if (sscanf(line, "%*s %*d %63s", s.name) != 1) s.name[0] = 0;
        }
        else if (!strcmp(kind, "god")) {
            char opt[16] = "";
            s.kind = S_GOD, strcpy(s.name, "god");
            if (sscanf(line, "%*s %15s", opt) == 1) s.team = !strcmp(opt, "hide") ? 1 : !strcmp(opt, "off") ? 2 : 0;
        }
        else if (!strcmp(kind, "turn") && sscanf(line, "%*s %d %f", &s.team, &s.hold) == 2 && s.team > 0)
            s.kind = S_TURN, strcpy(s.name, "turn");
        else if (!strcmp(kind, "pitch")) {
            char opt[16] = "";
            s.kind = S_PITCH, strcpy(s.name, "pitch");
            if (sscanf(line, "%*s %15s", opt) != 1 || !strcmp(opt, "off")) s.team = 0;
            else s.team = 1, s.hold = (float)atof(opt);
        }
        else if (!strcmp(kind, "flags")) {
            s.kind = S_FLAGS, strcpy(s.name, "flags");
            const char* q = line + 5;
            unsigned id = 0, v = 0;
            int used = 0;
            while (s.nflags < 8 && sscanf(q, " %u=%u%n", &id, &v, &used) == 2) {
                s.flags[s.nflags] = id, s.values[s.nflags] = (uint8_t)(v != 0), ++s.nflags;
                q += used;
            }
        }
        else if (!strcmp(kind, "flaglog") && sscanf(line, "%*s %u %d", &s.flags[0], &s.team) == 2)
            s.kind = S_FLAGLOG, strcpy(s.name, "flaglog");
        else if (!strcmp(kind, "bits") && sscanf(line, "%*s %u %d %x", &s.flags[0], &s.team, &s.dungeon) == 3 && s.team > 0 &&
                 s.team <= 32)
            s.kind = S_BITS, strcpy(s.name, "bits"); /* flags[0]: the first flag, team: how many, dungeon: the value */
        else if (!strcmp(kind, "chalice") && sscanf(line, "%*s %x %d %63s", &s.dungeon, &s.nflags, s.name) == 3 && s.nflags >= 0 &&
                 s.nflags < 7) {
            /* the travel to the dungeon's lamp; nflags: the altar slot */
            s.kind = S_TRAVEL, s.team = 2902950, s.block = 0x1d000000;
            s.yaw = (float)s.nflags;
            s.nflags = 0;
        }
        else if (!strcmp(kind, "remo") && sscanf(line, "%*s %d %f %f %f", &s.team, &s.x, &s.y, &s.hold) == 4)
            s.kind = S_REMO, snprintf(s.name, sizeof s.name, "remo%d", s.team);
        else if (!strcmp(kind, "movie") && sscanf(line, "%*s %d %f %f", &s.team, &s.y, &s.hold) == 3)
            s.kind = S_MOVIE, snprintf(s.name, sizeof s.name, "movie%d", s.team);
        else if (!strcmp(kind, "exit"))
            s.kind = S_EXIT, strcpy(s.name, "exit");
        else {
            api->log("mapval: bad tour line: %s", line);
            continue;
        }
        steps[nsteps++] = s;
    }
    fclose(f);
    api->log("mapval: tour of %d steps from %s", nsteps, path);
}

/* --- characters ----------------------------------------------------------- */

static int chr_pos(uint64_t chr, float v[3]) {
    uint64_t mods = ok(chr) ? rd64(chr + 0x3b0) : 0, t = ok(mods) ? rd64(mods + 0x68) : 0;
    if (!ok(t)) return 0;
    memcpy(v, (const void*)(uintptr_t)(t + 0x1e0), 12);
    return isfinite(v[0]) && isfinite(v[1]) && isfinite(v[2]);
}
static uint64_t chr_data(uint64_t chr) {
    uint64_t mods = ok(chr) ? rd64(chr + 0x3b0) : 0;
    return ok(mods) ? rd64(mods + 0x20) : 0;
}
static int chr_hp(uint64_t chr, int* hp, int* max) {
    uint64_t d = chr_data(chr);
    if (!ok(d)) return 0;
    *hp = *(const int*)(uintptr_t)(d + 0xf8);
    *max = *(const int*)(uintptr_t)(d + 0xfc);
    return 1;
}
/* The character's id ("c1000_0003") at +0x158 of its data module: a
 * UTF-16 string kept inline below 8 code units, else through a pointer;
 * length at +0x168, capacity at +0x170. */
static void chr_name(uint64_t chr, char* out, size_t n) {
    uint64_t d = chr_data(chr);
    size_t i = 0;
    out[0] = 0;
    if (!ok(d)) return;
    uint64_t len = *(const uint64_t*)(uintptr_t)(d + 0x168), cap = *(const uint64_t*)(uintptr_t)(d + 0x170);
    uint64_t at = d + 0x158;
    if (cap >= 8) at = rd64(d + 0x158);
    if (!ok(at & ~7ull) || len > 32) return;
    const uint16_t* w = (const uint16_t*)(uintptr_t)at;
    for (; i + 1 < n && i < len && w[i] && w[i] < 128; ++i) out[i] = (char)w[i];
    out[i] = 0;
}

/* WorldChrMan (0x593e878): +0x278 the number of character sets, +0x280 an
 * array of 0x18-byte slots whose first word points at a set {int count; ...;
 * entries at +8, 0x38 bytes each, the ChrIns first} - the walk the game's
 * own world update does over every character (0x1948740). */
static uint64_t nearest_enemy(const float p[3], float radius, float* dist) {
    uint64_t wcm = 0, best = 0;
    api->read(0x593e878, &wcm, 8);
    if (!ok(wcm)) return 0;
    uint64_t player = rd64(wcm + 0x60);
    int nsets = 0;
    nsets = *(const int*)(uintptr_t)(wcm + 0x278);
    float bd = radius;
    for (int s = 0; s < nsets && s < 64; ++s) {
        uint64_t set = rd64(wcm + 0x280 + (uint64_t)s * 0x18);
        if (!ok(set)) continue;
        int n = 0;
        n = *(const int*)(uintptr_t)set;
        uint64_t ents = rd64(set + 8);
        if (!ok(ents)) continue;
        for (int i = 0; i < n && i < 4096; ++i) {
            uint64_t c = rd64(ents + (uint64_t)i * 0x38);
            float v[3];
            int hp = 0, mx = 0;
            if (!ok(c) || c == player || !chr_pos(c, v) || !chr_hp(c, &hp, &mx) || hp <= 0 || mx <= 0) continue;
            char nm[20];
            chr_name(c, nm, sizeof nm);
            int team = *(const int*)(uintptr_t)(c + 0x88);
            if (dump_chrs) api->log("mapval chr %s team %d hp %d/%d at %.2f %.2f %.2f", nm, team, hp, mx, v[0], v[1], v[2]);
            if (!nm[0] || !strncmp(nm, "c0000", 5) || !strncmp(nm, "c9", 2)) continue; /* players, NPC hunters, c9xxx helpers */
            if (want_team >= 0 && team != want_team) continue;
            float dx = v[0] - p[0], dy = v[1] - p[1], dz = v[2] - p[2];
            float d = sqrtf(dx * dx + dy * dy + dz * dz);
            if (fabsf(dy) < 4 && d > 0.3f && d < bd) bd = d, best = c;
        }
    }
    if (dist) *dist = bd;
    return best;
}

/* Every live non-player character within r of p, one line each: during a
 * kill step, so a second enemy the attack happens to kill has a known
 * position too (mapval near <id> <t> <x> <y> <z> <hp>). */
static void log_near(const float p[3], float r, double t) {
    uint64_t wcm = 0;
    api->read(0x593e878, &wcm, 8);
    if (!ok(wcm)) return;
    uint64_t player = rd64(wcm + 0x60);
    int nsets = *(const int*)(uintptr_t)(wcm + 0x278);
    for (int s = 0; s < nsets && s < 64; ++s) {
        uint64_t set = rd64(wcm + 0x280 + (uint64_t)s * 0x18), ents = ok(set) ? rd64(set + 8) : 0;
        if (!ok(ents)) continue;
        int n = *(const int*)(uintptr_t)set;
        for (int i = 0; i < n && i < 4096; ++i) {
            uint64_t c = rd64(ents + (uint64_t)i * 0x38);
            float v[3];
            int hp = 0, mx = 0;
            char nm[20];
            if (!ok(c) || c == player || !chr_pos(c, v) || !chr_hp(c, &hp, &mx) || mx <= 0) continue;
            float dx = v[0] - p[0], dy = v[1] - p[1], dz = v[2] - p[2];
            if (dx * dx + dy * dy + dz * dz > r * r) continue;
            chr_name(c, nm, sizeof nm);
            if (!nm[0] || !strncmp(nm, "c0000", 5)) continue;
            api->log("mapval near %s %.3f %.3f %.3f %.3f %d", nm, t, v[0], v[1], v[2], hp);
        }
    }
}

/* The player's live HP is its ChrIns data module's (+0xf8; the
 * GameDataMan record that player_stat_get reads is a copy). */
static uint64_t player_chr(void) {
    uint64_t wcm = 0;
    api->read(0x593e878, &wcm, 8);
    return ok(wcm) ? rd64(wcm + 0x60) : 0;
}
static void set_player_hp(int v) {
    uint64_t d = chr_data(player_chr());
    int hp = 0, mx = 0;
    if (!ok(d) || !chr_hp(player_chr(), &hp, &mx) || hp <= 0) return; /* never revive */
    *(int*)(uintptr_t)(d + 0xf8) = v < 0 ? mx : v;
}
static void heal(void) { set_player_hp(-1); }
static int64_t player_hp(void) {
    int hp = 0, mx = 0;
    return chr_hp(player_chr(), &hp, &mx) ? hp : 0;
}

/* --- for tools/area_check.py ------------------------------------------------ */

/* A request to the host (BBHOST_TEST_REQUESTS, hle/video.cpp): one line in a
 * file of its own, written beside and renamed into place, so the host never
 * reads half of it. */
static int request_seq;
static void request(const char* line) {
    const char* dir = getenv("BBHOST_TEST_REQUESTS");
    char tmp[1024], req[1024];
    if (!dir || !*dir) {
        api->log("mapval: no BBHOST_TEST_REQUESTS for \"%s\"", line);
        return;
    }
    snprintf(tmp, sizeof tmp, "%s/%06d.tmp", dir, request_seq);
    snprintf(req, sizeof req, "%s/%06d.req", dir, request_seq);
    ++request_seq;
    FILE* f = fopen(tmp, "w");
    if (!f) {
        api->log("mapval: cannot write %s", tmp);
        return;
    }
    fprintf(f, "%s\n", line);
    fclose(f);
    if (rename(tmp, req) != 0) api->log("mapval: cannot rename %s", tmp);
}

/* The follow camera's owner (CHR_CAM_OWNER_ROOT_PTR's object, +0x2830): its
 * byte +0x80 asks for the camera to be put back behind the player
 * (include/bbhost/engine/sprj/camera.hpp, ChrCamOwner). */
static int reset_camera(void) {
    uint64_t root = 0;
    if (api->read(0x593e860, &root, 8) || !ok(root)) return 0;
    uint64_t owner = rd64(root + 0x2830);
    if (!ok(owner)) return 0;
    *(volatile uint8_t*)(uintptr_t)(owner + 0x80) = 1;
    return 1;
}

/* The game's "disable character" (EMEVD 2004[05]: its collision off and bit
 * 0 of its ChrSetEntry's +0x20, which keeps it out of the world's update and
 * draw), as plugins/boss_rush does it to clear an arena. */
static int chr_disabled(const void* ins) {
    const uint8_t* entry = (const uint8_t*)(uintptr_t)rd64((uint64_t)(uintptr_t)ins + 0x18);
    return entry && ok((uint64_t)(uintptr_t)entry & ~7ull) && (entry[0x20] & 1);
}
static void quiet(float radius, double t) {
    static BbChr chrs[1024]; /* ~56 KiB: not on the game's stack */
    size_t n = 0;
    float p[3];
    int off = 0, left = 0;
    if (api->version < 8 || api->chr_list(chrs, 1024, &n) || api->player_position(p, NULL)) {
        api->log("mapval quiet %.3f unavailable", t);
        return;
    }
    if (n > 1024) n = 1024;
    for (size_t i = 0; i < n; ++i) {
        const BbChr* c = &chrs[i];
        float dx = c->pos[0] - p[0], dy = c->pos[1] - p[1], dz = c->pos[2] - p[2];
        if (c->is_player || c->hp <= 0 || dx * dx + dz * dz > radius * radius || fabsf(dy) > 30) continue;
        /* 23: an ordinary enemy's team (NpcParam teamType); bosses have their own */
        if (c->team_type != 23 || c->npc_param <= 0 || !c->ins) {
            ++left;
            continue;
        }
        if (chr_disabled(c->ins)) continue;
        BbCallRegs regs;
        memset(&regs, 0, sizeof regs);
        regs.arg[0] = (uint64_t)(uintptr_t)c->ins;
        regs.arg[1] = 1;
        if (api->call_guest(0x1cc6390, &regs) == 0) ++off;
    }
    api->log("mapval quiet %.3f %d enemies within %.0f m switched off, %d other characters left", t, off, radius, left);
}

/* --- for tools/full_tour.py ------------------------------------------------- */

/* The extended follow camera (ChrCamOwner +0x60): +0x144 its yaw, +0x14c the
 * yaw a reset takes (src/decomp/camera/follow_camera.cpp). */
static uint64_t ex_follow_cam(void) {
    uint64_t root = 0;
    if (api->read(0x593e860, &root, 8) || !ok(root)) return 0;
    uint64_t owner = rd64(root + 0x2830);
    return ok(owner) ? rd64(owner + 0x60) : 0;
}
static double turn_base;
static int turn_k = -1;
static int pitch_on;    /* "pitch": the camera's pitch held every frame */
static float pitch_hold;

/* A cutscene plays while bit 0 of byte +0x168 of the remo manager
 * (*(*0x5940058 + 8)) is set: what the event scripts' wait for one polls
 * (0x1bb7700). */
static int remo_playing(void) {
    uint64_t holder = 0;
    if (api->read(0x5940058, &holder, 8) || !ok(holder)) return 0;
    uint64_t man = rd64(holder + 8);
    return ok(man) ? (*(const volatile uint8_t*)(uintptr_t)(man + 0x168) & 1) : 0;
}
static int remo_started, remo_taps;
static double remo_start_t, remo_tap_t;

/* n event flags from first as one number, the first the top bit (as the game
 * reads multi-flag values); -1 when one cannot be read. */
static int64_t flags_value(uint32_t first, int n) {
    int64_t v = 0;
    for (int i = 0; i < n; ++i) {
        int on = 0;
        if (api->event_flag_get(first + (uint32_t)i, &on)) return -1;
        v = v << 1 | (on ? 1 : 0);
    }
    return v;
}
static int flags_write(uint32_t first, int n, uint32_t v) {
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += api->event_flag_set(first + (uint32_t)i, (int)((v >> (n - 1 - i)) & 1)) != 0;
    return bad;
}

/* --- per frame ------------------------------------------------------------- */

/* The travel's block: a chalice dungeon is all of area 29 (the player can
 * read as its own map id there). */
static int in_tour_block(uint32_t b) { return b == tour_block || (tour_block >> 24 == 0x1d && b >> 24 == 0x1d); }

static void begin(int i, double t) {
    Step* s = &steps[i];
    cur = i;
    step_t0 = t;
    phase = 0;
    if ((s->kind == S_WARP || s->kind == S_FALL) && tour_block && !s->block) {
        uint32_t b = 0;
        if (api->player_block(&b) == 0 && !in_tour_block(b)) {
            /* A death (a kill plane under a navmesh point) respawned the
             * player elsewhere: back to the block's lamp, then this step. */
            /* The same step sending the player out twice (a kill plane
             * under it): leave it and go on with the next. */
            if (retravel_step == i && ++retravel_count >= 2) {
                api->log("mapval skip %s %.3f (the player left the block twice here)", s->name, t);
                retravel_count = 0;
                retravel_step = -1;
                if (i + 1 < nsteps) begin(i + 1, t);
                return;
            }
            if (retravel_step != i) retravel_step = i, retravel_count = 0;
            retravel = 1;
            travel_still = 0;
            api->lamp_warp(tour_rp);
            api->log("mapval retravel %s %.3f in %08x, back to %d", s->name, t, b, tour_rp);
            return;
        }
    }
    warp_retries = 0;
    fall_retries = 0;
    fall_n = 0;
    if (s->kind == S_WARP) gravity_hold(1, t);
    else if (s->kind == S_TRAVEL || s->kind == S_REMO || s->kind == S_MOVIE || s->kind == S_EXIT || s->kind == S_FLAGS) gravity_hold(0, t);
    if (s->kind == S_WARP) {
        float v[3] = {s->x, s->y, s->z};
        int r = api->player_warp(s->block, v, s->yaw);
        api->log("mapval step %d warp %s %.3f target %.3f %.3f %.3f yaw %.3f -> %s", i, s->name, t, s->x, s->y, s->z, s->yaw,
                 r ? "REFUSED" : "queued");
    } else if (s->kind == S_FALL) {
        float v[3] = {s->x, s->y + s->hold, s->z};
        set_player_hp(1);
        int r = api->player_warp(0, v, 0);
        fall_died = 0;
        api->log("mapval step %d falldeath %s %.3f target %.3f %.3f %.3f from %.1f -> %s", i, s->name, t, s->x, s->y, s->z, s->hold,
                 r ? "REFUSED" : "queued");
    } else if (s->kind == S_KILL) {
        kill_chr = 0;
        kill_last_warp = 0;
        api->log("mapval step %d kill %.3f radius %.1f", i, t, s->hold);
    } else if (s->kind == S_TRAVEL) {
        if (s->dungeon) {
            /* The altar slot in use (9020 + slot on, the others off) holds the
             * dungeon (94000000 + slot * 100000: its map id, 32 flags, area
             * byte first): a stage change into area 29 loads it
             * (plugins/boss_rush does the same). */
            int slot = (int)s->yaw;
            int bad = flags_write(9020, 7, 1u << (6 - slot)) + flags_write(94000000u + (uint32_t)slot * 100000u, 32, s->dungeon);
            api->log("mapval chalice %s %.3f slot %d holds %08x%s", s->name, t, slot, s->dungeon, bad ? " (flags REFUSED)" : "");
        }
        for (int k = 0; k < s->nflags; ++k) {
            int r = api->event_flag_set(s->flags[k], s->values[k]);
            api->log("mapval flag %u %s %.3f -> %s", s->flags[k], s->values[k] ? "on" : "off", t, r ? "REFUSED" : "set");
        }
        int r = api->size >= sizeof(BbHostApi) ? api->lamp_warp(s->team) : 1;
        travel_still = 0;
        travel_loading = 0;
        travel_wait_logged = 0;
        tour_rp = s->team;
        tour_block = s->block;
        api->log("mapval step %d travel %s %.3f return point %d block %08x -> %s", i, s->name, t, s->team, s->block, r ? "REFUSED" : "queued");
        if (r) travel_failed = 1;
    } else if (s->kind == S_WALK) {
        float p[3] = {0, 0, 0};
        api->player_position(p, NULL);
        api->log("mapval walk-ready %s %.3f %.3f %.3f %.3f", s->name, t, p[0], p[1], p[2]);
    } else if (s->kind == S_HOLD) {
        api->log("mapval step %d hold %.3f %.1f s", i, t, s->hold);
    } else if (s->kind == S_FLIPS) {
        api->log("mapval step %d flips %.3f %.0f after the loading screen (%llu so far%s)", i, t, s->hold,
                 (unsigned long long)(frame_no - loading_end_frame), loading ? ", the screen is up" : "");
    } else if (s->kind == S_MARK) {
        float p[3] = {0, 0, 0}, yaw = 0;
        uint32_t b = 0;
        api->player_position(p, &yaw);
        api->player_block(&b);
        api->log("mapval mark %s %.3f %08x %.3f %.3f %.3f yaw %.3f hp %lld loading %d", s->name, t, b, p[0], p[1], p[2], yaw,
                 (long long)player_hp(), loading);
    } else if (s->kind == S_CAMERA) {
        api->log("mapval camera %.3f %s", t, reset_camera() ? "reset" : "unavailable");
    } else if (s->kind == S_QUIET) {
        quiet(s->hold, t);
    } else if (s->kind == S_GOD) {
        uint8_t v = s->team == 2 ? 0 : 1;
        int r = api->write(0x593e888, &v, 1);
        if (s->team) r |= api->write(0x593e88e, &v, 1);
        api->log("mapval god %.3f no-dead%s -> %s", t, s->team ? " hide" : "", r ? "REFUSED" : v ? "on" : "off");
    } else if (s->kind == S_PITCH) {
        uint64_t cam = ex_follow_cam();
        pitch_on = s->team, pitch_hold = s->hold;
        api->log("mapval pitch %.3f %s%s", t, s->team ? "held at" : "off", cam ? "" : " (no camera)");
        if (s->team) api->log("mapval pitch-to %.3f (was %.3f)", pitch_hold, cam ? *(const volatile float*)(uintptr_t)(cam + 0x140) : 0.f);
    } else if (s->kind == S_TURN) {
        uint64_t cam = ex_follow_cam();
        turn_k = -1;
        turn_base = cam ? *(const volatile float*)(uintptr_t)(cam + 0x144) : 0;
        api->log("mapval turn %.3f %d headings %.1f s each from yaw %.3f%s", t, s->team, s->hold, turn_base, cam ? "" : " (no camera)");
    } else if (s->kind == S_FLAGS) {
        for (int k = 0; k < s->nflags; ++k) {
            int r = api->event_flag_set(s->flags[k], s->values[k]);
            api->log("mapval flag %u %s %.3f -> %s", s->flags[k], s->values[k] ? "on" : "off", t, r ? "REFUSED" : "set");
        }
    } else if (s->kind == S_FLAGLOG) {
        api->log("mapval flags %u+%d %.3f = %lld", s->flags[0], s->team, t, (long long)flags_value(s->flags[0], s->team));
    } else if (s->kind == S_BITS) {
        int bad = flags_write(s->flags[0], s->team, s->dungeon);
        api->log("mapval bits %u+%d %.3f = %08x%s (reads %08x)", s->flags[0], s->team, t, s->dungeon, bad ? " REFUSED" : "",
                 (uint32_t)flags_value(s->flags[0], s->team));
    } else if (s->kind == S_REMO) {
        BbCallRegs regs;
        memset(&regs, 0, sizeof regs);
        regs.arg[1] = (uint32_t)s->team;   /* the cutscene: s<area>_<block>_<nnnn> */
        regs.arg[2] = (uint32_t)(int)s->x; /* its play mode (0: skippable) */
        regs.arg[3] = 0xffffffffu;
        regs.arg[4] = 0xffffffffu;
        regs.arg[5] = 10000; /* the player */
        int r = api->call_guest(0x171a9f0, &regs);
        remo_started = 0, remo_taps = 0, remo_tap_t = 0;
        api->log("mapval remo-request %d %.3f mode %d -> %s (%d)", s->team, t, (int)s->x, r ? "REFUSED" : "queued", (int)(regs.ret & 0xff));
        if (r) s->hold = 0;
    } else if (s->kind == S_MOVIE) {
        BbCallRegs regs;
        memset(&regs, 0, sizeof regs);
        regs.arg[1] = (uint32_t)s->team;
        regs.arg[2] = 1;
        int r = api->call_guest(0x1734d90, &regs);
        remo_taps = 0, remo_tap_t = 0;
        api->log("mapval movie-request %d %.3f -> %s", s->team, t, r ? "REFUSED" : "queued");
    } else if (s->kind == S_EXIT) {
        request("exit");
        api->log("mapval exit %.3f", t);
    } else if (s->kind == S_DUMP) {
        const char* dir = getenv("BBHOST_MAPVAL_DUMPS");
        char line[1100];
        snprintf(line, sizeof line, "dump %s/%s.ppm", dir && *dir ? dir : ".", s->name);
        request(line);
        api->log("mapval dump %s %.3f", s->name, t);
    } else if (s->kind == S_F12) {
        char line[160];
        snprintf(line, sizeof line, "f12 %s", s->name);
        request(line);
        api->log("mapval f12 %s %.3f", s->name, t);
    } else if (s->kind == S_CAPTURE) {
        const char* dir = getenv("BBHOST_MAPVAL_CAPTURES");
        char line[1100];
        if (s->name[0] && dir && *dir)
            snprintf(line, sizeof line, "capture %d %s/%s", s->team, dir, s->name);
        else
            snprintf(line, sizeof line, "capture %d", s->team);
        request(line);
        api->log("mapval capture %d %s %.3f", s->team, s->name[0] ? s->name : "-", t);
    } else {
        api->log("mapval step %d wait %.3f %.1f s", i, t, s->hold);
    }
}

static void put_next_to(uint64_t chr, double t) {
    float e[3], p[3];
    if (!chr_pos(chr, e) || api->player_position(p, NULL)) return;
    float dx = p[0] - e[0], dz = p[2] - e[2], d = sqrtf(dx * dx + dz * dz);
    if (d < 0.01f) dx = 1, dz = 0, d = 1;
    float at[3] = {e[0] + dx / d * 1.3f, e[1] + 0.2f, e[2] + dz / d * 1.3f};
    /* Facing: in one run a light attack at yaw 0 stepped the player towards
     * -z, so this faces the enemy if yaw 0 looks down -z; the driver also
     * locks on (Q), which turns the attacks to the target either way. */
    float yaw = atan2f(at[0] - e[0], at[2] - e[2]);
    api->player_warp(0, at, yaw);
    kill_last_warp = t;
}

static void tick(double t) {
    if (cur < 0 || tour_done) return;
    if (deadline_s > 0 && !deadline_hit && t - loaded_t > deadline_s && steps[cur].kind != S_EXIT) {
        int j = cur + 1;
        while (j < nsteps && steps[j].kind != S_EXIT) ++j;
        deadline_hit = 1;
        api->log("mapval deadline %.3f after %.0f s: steps %d-%d skipped (from %s)", t, t - loaded_t, cur, j - 1, steps[cur].name);
        if (j < nsteps) {
            begin(j, t);
            return;
        }
    }
    Step* s = &steps[cur];
    double el = t - step_t0;
    float p[3], yaw = 0;
    int have = api->player_position(p, &yaw) == 0;
    int64_t hp = player_hp();
    int next = 0;
    if (retravel) {
        uint32_t b = 0;
        int in = have && api->player_block(&b) == 0 && in_tour_block(b);
        if (in && !loading && fabsf(p[0] - travel_last[0]) + fabsf(p[1] - travel_last[1]) + fabsf(p[2] - travel_last[2]) < 0.05f) travel_still++;
        else travel_still = 0;
        if (have) memcpy(travel_last, p, sizeof travel_last);
        /* A request made while a load or respawn is still going can be
         * dropped: ask again every 20 s. */
        if (!in && el > retravel * 20.0) {
            ++retravel;
            api->lamp_warp(tour_rp);
        }
        if (travel_still > 90 || el > travel_limit) {
            retravel = 0;
            api->log("mapval retravel-%s %.3f %08x", travel_still > 90 ? "done" : "failed", t, b);
            begin(cur, t);
        }
        return;
    }
    if (s->kind == S_WARP) {
        heal();
        if (have && t - fall_sample_t >= 0.25) {
            fall_hist[fall_n++ % 3] = p[1];
            fall_sample_t = t;
        }
        int falling = have && fall_n >= 3 && p[1] < fall_hist[fall_n % 3] - 2.0f;  /* the oldest of the three: ~0.5 s ago */
        /* with gravity held at 0 nothing falls: a drop is the player put down on the floor under the target */
        if (have && !loading && el >= 1.2 && p[1] < s->y - 6.0f && falling && !grav_off) {
            if (fall_retries < 2) {
                float v[3] = {s->x, s->y, s->z};
                ++fall_retries;
                api->player_warp(s->block, v, s->yaw);
                step_t0 = t; /* the arrival checked again from now */
                phase = 0;
                fall_n = 0;
                api->log("mapval rewarp-fall %s %.3f try %d (at %.3f %.3f %.3f)", s->name, t, fall_retries, p[0], p[1], p[2]);
                return;
            }
            if (fall_retries == 2) {
                ++fall_retries;
                api->log("mapval fell %s %.3f at %.3f %.3f %.3f, back to %.3f %.3f %.3f", s->name, t, p[0], p[1], p[2], good_pos[0], good_pos[1],
                         good_pos[2]);
                if (have_good) api->player_warp(0, good_pos, 0);
                next = 1;
            }
        }
        /* A warp made while a load is still on screen is dropped: warp
         * again (up to 10 times, 3 s apart) until the player is there. */
        if (phase == 0 && el >= 1.5 && have && !loading && warp_retries < warp_tries && hypotf(p[0] - s->x, p[2] - s->z) > 2.5f) {
            float v[3] = {s->x, s->y, s->z};
            ++warp_retries;
            api->player_warp(s->block, v, s->yaw);
            step_t0 = t - 1.5 + 3.0; /* look again in 3 s */
            api->log("mapval rewarp %s %.3f try %d (at %.3f %.3f %.3f)", s->name, t, warp_retries, p[0], p[1], p[2]);
            return;
        }
        if (phase == 0 && el >= 1.5 && have && !loading && !next) {
            uint32_t b = 0;
            api->player_block(&b);
            api->log("mapval at %s %.3f %08x %.3f %.3f %.3f yaw %.3f", s->name, t, b, p[0], p[1], p[2], yaw);
            phase = 1;
        }
        if (phase == 1 && el >= 1.5 + s->hold && p[1] >= s->y - 6.0f) {
            memcpy(good_pos, p, sizeof good_pos); /* held: a place to come back to */
            good_pos[1] += 0.3f;
            have_good = 1;
        }
        if (!next && el >= 1.5 + s->hold) {
            api->log("mapval hold-end %s %.3f %.3f %.3f %.3f", s->name, t, p[0], p[1], p[2]);
            next = 1;
        }
    } else if (s->kind == S_KILL) {
        heal();
        if (!kill_chr && have) {
            float d = 0, c[3] = {s->x, s->y, s->z};
            dump_chrs = 1;
            want_team = s->team;
            kill_chr = nearest_enemy(s->yaw ? c : p, s->hold, &d);
            /* Nothing there (this save killed them, or they are not loaded
             * after the tour): the nearest one to the player, farther out,
             * and then of any team. */
            if (!kill_chr) kill_chr = nearest_enemy(p, 250, &d);
            if (!kill_chr) want_team = -1, kill_chr = nearest_enemy(p, 250, &d);
            dump_chrs = 0;
            if (!kill_chr) {
                api->log("mapval kill none within %.1f m", s->hold);
                next = 1;
            } else {
                char nm[20];
                float e[3];
                chr_name(kill_chr, nm, sizeof nm);
                chr_pos(kill_chr, e);
                *(int*)(uintptr_t)(chr_data(kill_chr) + 0xf8) = 1;
                api->log("mapval kill-target %s %.3f %.3f %.3f %.3f dist %.1f", nm, t, e[0], e[1], e[2], d);
                put_next_to(kill_chr, t);
                api->log("mapval kill-ready %s %.3f", nm, t);
            }
        } else if (kill_chr) {
            if (have && t - near_last >= 0.5) near_last = t, log_near(p, 10, t);
            int ehp = 0, emx = 0;
            float e[3] = {0, 0, 0};
            chr_hp(kill_chr, &ehp, &emx);
            chr_pos(kill_chr, e);
            if (ehp <= 0) {
                char nm[20];
                chr_name(kill_chr, nm, sizeof nm);
                api->log("mapval kill-done %s %.3f %.3f %.3f %.3f player %.3f %.3f %.3f", nm, t, e[0], e[1], e[2], p[0], p[1], p[2]);
                next = 1;
            } else if (el > 60) {
                api->log("mapval kill-timeout %.3f hp %d", t, ehp);
                next = 1;
            } else {
                if (ehp > 1) *(int*)(uintptr_t)(chr_data(kill_chr) + 0xf8) = 1;
                if (t - kill_last_warp > 2) {
                    float dx = e[0] - p[0], dz = e[2] - p[2];
                    if (dx * dx + dz * dz > 2.5f * 2.5f) put_next_to(kill_chr, t);
                    else kill_last_warp = t;
                }
            }
        }
    } else if (s->kind == S_FALL) {
        if (!fall_died && (hp <= 0 || !have)) {
            fall_died = 1;
            api->log("mapval fall-dead %s %.3f last %.3f %.3f %.3f", s->name, t, p[0], p[1], p[2]);
        }
        if (phase == 0 && have && el > 0.5 && el < 1.0) api->log("mapval fall-from %s %.3f %.3f %.3f %.3f", s->name, t, p[0], p[1], p[2]);
        if (have && el > 0.5 && !fall_died) phase = 1;
        /* Done once the player is back (respawned) or after 60 s. */
        if ((fall_died && have && hp > 0 && el > 8) || el > 60) {
            api->log("mapval fall-end %s %.3f died %d back at %.3f %.3f %.3f", s->name, t, fall_died, p[0], p[1], p[2]);
            next = 1;
        }
    } else if (s->kind == S_TRAVEL) {
        uint32_t b = 0;
        int in = have && api->player_block(&b) == 0 && (b == s->block || (s->dungeon && b >> 24 == 0x1d));
        if (in) heal();
        if (in && !loading && fabsf(p[0] - travel_last[0]) + fabsf(p[1] - travel_last[1]) + fabsf(p[2] - travel_last[2]) < 0.05f) {
            if (travel_still++ == 0) travel_still_t = t;
        } else {
            travel_still = 0;
        }
        if (have) memcpy(travel_last, p, sizeof travel_last);
        /* In the block with the loading screen gone, and still for ~3 s (90
         * frames at 30 fps, and 2 s however fast the frames come) - or, where
         * frames come slower than ~7 a second (lavapipe: a chalice dungeon
         * draws one frame in ~2 s), still for 12 s over at least 8 frames. */
        int still = (travel_still > 90 && t - travel_still_t >= 2.0) || (travel_still > 8 && t - travel_still_t >= 12.0);
        int placed_since = placed_t >= step_t0; /* the warp's own placement, not one before the request */
        if (still && placed_since && (fabsf(p[0]) > 0.01f || fabsf(p[2]) > 0.01f)) { /* not the (0, y, 0) a load starts at */
            api->log("mapval travel-done %s %.3f %08x %.3f %.3f %.3f after %.1f s, %.1f s of it loading", s->name, t, b, p[0], p[1], p[2],
                     el, travel_loading);
            memcpy(good_pos, p, sizeof good_pos);
            good_pos[1] += 0.3f;
            have_good = 1;
            next = 1;
        } else if (travel_failed || el > travel_limit) {
            api->log("mapval travel-failed %s %.3f in %08x after %.1f s", s->name, t, b, el);
            /* The rest of this block's steps would run in the wrong place:
             * skip to the next travel. */
            travel_failed = 0;
            int j = cur + 1;
            while (j < nsteps && steps[j].kind != S_TRAVEL) ++j;
            if (j < nsteps) {
                begin(j, t);
                return;
            }
            next = 1;
            cur = nsteps - 1;
        } else if (still && !placed_since && !travel_wait_logged) {
            travel_wait_logged = 1;
            api->log("mapval travel-wait %s %.3f: still in %08x for %.1f s, not placed since the request", s->name, t, b,
                     t - travel_still_t);
        }
    } else if (s->kind == S_WALK) {
        heal();
        if (el >= s->hold) {
            api->log("mapval walk-end %s %.3f %.3f %.3f %.3f", s->name, t, p[0], p[1], p[2]);
            next = 1;
        }
    } else if (s->kind == S_HOLD) {
        heal();
        if (el >= s->hold) next = 1;
    } else if (s->kind == S_FLIPS) {
        heal();
        if (!loading && frame_no - loading_end_frame >= (uint64_t)s->hold) {
            api->log("mapval flips-done %.3f %llu frames after the loading screen, %.1f s waited", t,
                     (unsigned long long)(frame_no - loading_end_frame), el);
            next = 1;
        } else if (el > 900) { /* a screen that never goes: the tour goes on */
            api->log("mapval flips-gave-up %.3f after %.0f s, %llu frames after the loading screen%s", t, el,
                     (unsigned long long)(frame_no - loading_end_frame), loading ? ", the screen still up" : "");
            next = 1;
        }
    } else if (s->kind == S_TURN) {
        heal();
        int k = (int)(el / (s->hold > 0.05f ? s->hold : 0.05f));
        uint64_t cam = ex_follow_cam();
        if (k >= s->team || !cam) {
            next = 1;
        } else {
            float yaw = (float)(turn_base + k * 6.283185307179586 / s->team);
            volatile float* f = (volatile float*)(uintptr_t)cam;
            if (k != turn_k) {
                api->log("mapval turn-to %d %.3f yaw %.3f (was %.3f)", k, t, yaw, f[0x144 / 4]);
                turn_k = k;
            }
            f[0x144 / 4] = yaw; /* every frame: the camera's own update would turn it back */
            f[0x14c / 4] = yaw;
        }
    } else if (s->kind == S_REMO) {
        heal();
        int playing = remo_playing();
        if (playing && !remo_started) {
            remo_started = 1, remo_start_t = t;
            api->log("mapval remo-start %d %.3f after %.1f s", s->team, t, el);
        }
        if (remo_started && !playing) {
            api->log("mapval remo-end %d %.3f played %.1f s, skip pressed %d times", s->team, t, t - remo_start_t, remo_taps);
            next = 1;
        } else if (!remo_started && el > 25) {
            api->log("mapval remo-none %d %.3f (it did not start)", s->team, t);
            next = 1;
        } else if (el > s->hold) {
            api->log("mapval remo-timeout %d %.3f still playing after %.1f s", s->team, t, el);
            next = 1;
        } else if (remo_started && t - remo_start_t >= s->y && t - remo_tap_t >= 3.0 && remo_taps < 4) {
            /* the game's skip: Options during a cutscene */
            request("tap options 250");
            remo_tap_t = t, ++remo_taps;
            api->log("mapval remo-skip %d %.3f press %d", s->team, t, remo_taps);
        }
    } else if (s->kind == S_MOVIE) {
        heal();
        if (el > s->hold) {
            api->log("mapval movie-end %d %.3f after %.1f s, skip pressed %d times", s->team, t, el, remo_taps);
            next = 1;
        } else if (el >= s->y && t - remo_tap_t >= 3.0 && remo_taps < 3) {
            request("tap options 250");
            remo_tap_t = t, ++remo_taps;
            api->log("mapval movie-skip %d %.3f press %d", s->team, t, remo_taps);
        }
    } else if (s->kind == S_EXIT) {
        if (el > 30) next = 1; /* the host ends the run before this */
    } else if (s->kind == S_MARK || s->kind == S_CAMERA || s->kind == S_QUIET || s->kind == S_DUMP || s->kind == S_CAPTURE ||
               s->kind == S_F12 || s->kind == S_GOD || s->kind == S_FLAGS || s->kind == S_FLAGLOG || s->kind == S_BITS ||
               s->kind == S_PITCH) {
        next = 1; /* done in begin() */
    } else if (el >= s->hold) {
        next = 1;
    }
    if (next) {
        if (cur + 1 < nsteps)
            begin(cur + 1, t);
        else {
            tour_done = 1;
            api->log("mapval tour-done %.3f", t);
        }
    }
}

/* A death: the player's HP gone to 0 (the same character as the last frame
 * - a load builds a new one, whose HP reads 0 until it is set up). */
static void watch_death(double t, const float p[3], int have) {
    static uint64_t last_chr;
    static int last_hp = -1;
    int hp = 0, mx = 0;
    uint64_t chr = have ? player_chr() : 0;
    if (chr != last_chr) {
        last_chr = chr;
        last_hp = -1;
    }
    if (!chr || !chr_hp(chr, &hp, &mx) || mx <= 0) return;
    if (hp <= 0 && last_hp > 0) {
        uint32_t b = 0;
        api->player_block(&b);
        api->log("mapval death %.3f %08x %.3f %.3f %.3f", t, b, p[0], p[1], p[2]);
    }
    last_hp = hp;
}

/* The host's on_world_load: the player placed in a map (a load, a death's
 * respawn, a lamp, a warp into another block). */
static void placed(uint32_t block, void* u) {
    (void)u;
    double t = now_s();
    placed_t = t;
    api->log("mapval placed %.3f %08x", t, block);
}

static void on_frame(void* u) {
    (void)u;
    double t = now_s();
    float p[3], yaw = 0;
    int have = api->player_position(p, &yaw) == 0;
    int was = loading;
    ++frame_no;
    loading = loading_screen();
    if (loading && !was) {
        loading_t0 = t;
        api->log("mapval loading-begins %.3f", t);
    } else if (!loading && was) {
        loading_end_frame = frame_no;
        if (cur >= 0 && !tour_done && (steps[cur].kind == S_TRAVEL || retravel)) travel_loading += t - loading_t0;
        api->log("mapval loading-ends %.3f after %.1f s", t, t - loading_t0);
    }
    watch_death(t, p, have);
    if (have) {
        if (!first_seen_t || t - last_seen_t > 30) first_seen_t = first_seen_t ? first_seen_t : t;
        last_seen_t = t;
    }
    if (t - last_pos_t >= 1.0) {
        last_pos_t = t;
        if (have) {
            uint32_t b = 0;
            int64_t hp = player_hp();
            api->player_block(&b);
            api->log("mapval pos %.3f %08x %.3f %.3f %.3f %.3f %lld", t, b, p[0], p[1], p[2], yaw, (long long)hp);
        }
    }
    if (pitch_on) {
        uint64_t cam = ex_follow_cam();
        if (cam) {
            volatile float* f = (volatile float*)(uintptr_t)cam;
            f[0x140 / 4] = pitch_hold; /* every frame: the camera's return would ease it back */
            f[0x148 / 4] = pitch_hold;
            f[0x150 / 4] = pitch_hold;
        }
    }
    if (cur < 0 && nsteps && have && first_seen_t && t - first_seen_t >= start_after) {
        if (player_hp() > 0) {
            api->log("mapval tour-start %.3f", t);
            begin(0, t);
            return;
        }
    }
    tick(t);
}

BB_PLUGIN_EXPORT int bb_plugin_init(const BbHostApi* a) {
    api = a;
    return a->version < 6; /* lamp_warp (7) is checked where it is used */
}

BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* a) {
    if (!a->eboot_is_109()) return 1;
    const char* tour = getenv("BBHOST_MAPVAL_TOUR");
    const char* st = getenv("BBHOST_MAPVAL_START");
    if (st && *st) start_after = atof(st);
    const char* wt = getenv("BBHOST_MAPVAL_WARP_TRIES");
    if (wt && *wt) warp_tries = atoi(wt);
    const char* fm = getenv("BBHOST_MAPVAL_FLOAT");
    float_mode = fm && fm[0] == '1';
    const char* dl = getenv("BBHOST_MAPVAL_DEADLINE");
    if (dl && *dl) deadline_s = atof(dl);
    const char* tl = getenv("BBHOST_MAPVAL_TRAVEL_S");
    if (tl && *tl && atof(tl) > 0) travel_limit = atof(tl);
    loaded_t = now_s();
    if (tour && *tour) load_tour(tour);
    a->on_frame(on_frame, 0);
    if (a->size >= offsetof(BbHostApi, on_world_load) + sizeof(a->on_world_load) && a->on_world_load) a->on_world_load(placed, 0);
    a->log("mapval: loaded (start after %.0f s in the world)", start_after);
    return 0;
}
