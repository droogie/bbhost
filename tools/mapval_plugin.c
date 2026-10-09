/* Map validation probe (tools/map_validation.sh): ground truth for the
 * private server's world map. Test-only: it moves the player, so it needs
 * the host's BBHOST_TEST_WARP=1 and is never installed by default.
 *
 * Every second it logs where the player is:
 *     mapval pos <epoch s> <block hex> <x> <y> <z> <yaw> <hp>
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
 *     travel <return point> <block hex> <name> [<event flag>...]
 *                                             lamp travel (a full map load,
 *                                             api->lamp_warp); done once the
 *                                             player stands in that block. Up
 *                                             to four event flags are turned on
 *                                             first, as the load reads them: a
 *                                             door's "opened", say
 *     walk <name> <s>                         log walk-ready and give the
 *                                             driver <s> seconds to walk
 *     wait <s>
 * and, for tools/area_check.py, which measures each place it stands in:
 *     hold <s>                                wait with the player's HP kept full
 *     mark <name>                             log `mapval mark <name> <t>`
 *     camera                                  the follow camera back behind the
 *                                             player, facing as the player does
 *     quiet <radius m>                        switch off the ordinary enemies
 *                                             within the radius (the game's own
 *                                             "disable character"; a load makes
 *                                             them again), so the place holds still
 *     dump <name>                             this frame to
 *                                             $BBHOST_MAPVAL_DUMPS/<name>.ppm
 *     capture <n> [<name>]                    let BBHOST_CAPTURE_DRAW take up to n
 *                                             draws, into
 *                                             $BBHOST_MAPVAL_CAPTURES/<name>; 0 stops
 * (the last two ask the host through BBHOST_TEST_REQUESTS, hle/video.cpp).
 * Each step logs what it did with the time, so the dataset script can find
 * the play-log samples of the same seconds. While a warp or kill step runs
 * the player's HP is kept full (the seed stands among enemies).
 *
 * Build: cc -shared -fPIC -O2 -Iinclude -o mapval.so tools/mapval_plugin.c -lm
 */
#include "bbhost_plugin.h"

#include <math.h>
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

/* --- the tour ------------------------------------------------------------ */

enum { S_WARP, S_KILL, S_FALL, S_WAIT, S_TRAVEL, S_WALK, S_HOLD, S_MARK, S_CAMERA, S_QUIET, S_DUMP, S_CAPTURE };
typedef struct {
    int kind;
    char name[64];
    float x, y, z, yaw, hold;
    int team;
    uint32_t block; /* warp into this block (0: the current one) */
    uint32_t flags[4]; /* travel: event flags turned on before it */
    int nflags;
} Step;
static Step steps[1024];
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
static int dump_chrs = 0, want_team = -1;

static void load_tour(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        api->log("mapval: no tour file %s", path);
        return;
    }
    char line[512];
    while (fgets(line, sizeof line, f) && nsteps < 1024) {
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
            s.kind = S_TRAVEL; /* travel <return point id> <block hex> <name> [<event flag>...] */
            int used = 0;
            unsigned f[4];
            sscanf(line, "%*s %*d %*x %*63s%n", &used);
            if (used > 0) s.nflags = sscanf(line + used, "%u %u %u %u", &f[0], &f[1], &f[2], &f[3]);
            if (s.nflags < 0) s.nflags = 0;
            for (int k = 0; k < s.nflags; ++k) s.flags[k] = f[k];
        }
        else if (!strcmp(kind, "walk") && sscanf(line, "%*s %63s %f", s.name, &s.hold) == 2)
            s.kind = S_WALK; /* walk <name> <seconds>: the driver walks */
        else if (!strcmp(kind, "wait") && sscanf(line, "%*s %f", &s.hold) == 1)
            s.kind = S_WAIT, strcpy(s.name, "wait");
        else if (!strcmp(kind, "hold") && sscanf(line, "%*s %f", &s.hold) == 1)
            s.kind = S_HOLD, strcpy(s.name, "hold");
        else if (!strcmp(kind, "mark") && sscanf(line, "%*s %63s", s.name) == 1)
            s.kind = S_MARK;
        else if (!strcmp(kind, "camera"))
            s.kind = S_CAMERA, strcpy(s.name, "camera");
        else if (!strcmp(kind, "quiet") && sscanf(line, "%*s %f", &s.hold) == 1)
            s.kind = S_QUIET, strcpy(s.name, "quiet"); /* hold: the radius */
        else if (!strcmp(kind, "dump") && sscanf(line, "%*s %63s", s.name) == 1)
            s.kind = S_DUMP;
        else if (!strcmp(kind, "capture") && sscanf(line, "%*s %d", &s.team) == 1) {
            s.kind = S_CAPTURE; /* team: the draws */
            if (sscanf(line, "%*s %*d %63s", s.name) != 1) s.name[0] = 0;
        }
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

/* --- per frame ------------------------------------------------------------- */

static void begin(int i, double t) {
    Step* s = &steps[i];
    cur = i;
    step_t0 = t;
    phase = 0;
    if ((s->kind == S_WARP || s->kind == S_FALL) && tour_block && !s->block) {
        uint32_t b = 0;
        if (api->player_block(&b) == 0 && b != tour_block) {
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
        for (int k = 0; k < s->nflags; ++k) {
            int r = api->event_flag_set(s->flags[k], 1);
            api->log("mapval flag %u on %.3f -> %s", s->flags[k], t, r ? "REFUSED" : "set");
        }
        int r = api->size >= sizeof(BbHostApi) ? api->lamp_warp(s->team) : 1;
        travel_still = 0;
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
    } else if (s->kind == S_MARK) {
        float p[3] = {0, 0, 0}, yaw = 0;
        uint32_t b = 0;
        api->player_position(p, &yaw);
        api->player_block(&b);
        api->log("mapval mark %s %.3f %08x %.3f %.3f %.3f yaw %.3f", s->name, t, b, p[0], p[1], p[2], yaw);
    } else if (s->kind == S_CAMERA) {
        api->log("mapval camera %.3f %s", t, reset_camera() ? "reset" : "unavailable");
    } else if (s->kind == S_QUIET) {
        quiet(s->hold, t);
    } else if (s->kind == S_DUMP) {
        const char* dir = getenv("BBHOST_MAPVAL_DUMPS");
        char line[1100];
        snprintf(line, sizeof line, "dump %s/%s.ppm", dir && *dir ? dir : ".", s->name);
        request(line);
        api->log("mapval dump %s %.3f", s->name, t);
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
    Step* s = &steps[cur];
    double el = t - step_t0;
    float p[3], yaw = 0;
    int have = api->player_position(p, &yaw) == 0;
    int64_t hp = player_hp();
    int next = 0;
    if (retravel) {
        uint32_t b = 0;
        int in = have && api->player_block(&b) == 0 && b == tour_block;
        if (in && fabsf(p[0] - travel_last[0]) + fabsf(p[1] - travel_last[1]) + fabsf(p[2] - travel_last[2]) < 0.05f) travel_still++;
        else travel_still = 0;
        if (have) memcpy(travel_last, p, sizeof travel_last);
        /* A request made while a load or respawn is still going can be
         * dropped: ask again every 20 s. */
        if (!in && el > retravel * 20.0) {
            ++retravel;
            api->lamp_warp(tour_rp);
        }
        if (travel_still > 90 || el > 150) {
            retravel = 0;
            api->log("mapval retravel-%s %.3f %08x", travel_still > 90 ? "done" : "failed", t, b);
            begin(cur, t);
        }
        return;
    }
    if (s->kind == S_WARP) {
        heal();
        /* A warp made while a load is still on screen is dropped: warp
         * again (up to 10 times, 3 s apart) until the player is there. */
        if (phase == 0 && el >= 1.5 && have && warp_retries < 10 && hypotf(p[0] - s->x, p[2] - s->z) > 2.5f) {
            float v[3] = {s->x, s->y, s->z};
            ++warp_retries;
            api->player_warp(s->block, v, s->yaw);
            step_t0 = t - 1.5 + 3.0; /* look again in 3 s */
            api->log("mapval rewarp %s %.3f try %d (at %.3f %.3f %.3f)", s->name, t, warp_retries, p[0], p[1], p[2]);
            return;
        }
        if (phase == 0 && el >= 1.5 && have) {
            uint32_t b = 0;
            api->player_block(&b);
            api->log("mapval at %s %.3f %08x %.3f %.3f %.3f yaw %.3f", s->name, t, b, p[0], p[1], p[2], yaw);
            phase = 1;
        }
        if (el >= 1.5 + s->hold) {
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
        int in = have && api->player_block(&b) == 0 && b == s->block;
        if (in) heal();
        if (in && fabsf(p[0] - travel_last[0]) + fabsf(p[1] - travel_last[1]) + fabsf(p[2] - travel_last[2]) < 0.05f)
            travel_still++;
        else
            travel_still = 0;
        if (have) memcpy(travel_last, p, sizeof travel_last);
        /* In the block and still for ~3 s (90 frames at 30 fps). */
        if (travel_still > 90 && (fabsf(p[0]) > 0.01f || fabsf(p[2]) > 0.01f)) { /* not the (0, y, 0) a load starts at */
            api->log("mapval travel-done %s %.3f %08x %.3f %.3f %.3f after %.1f s", s->name, t, b, p[0], p[1], p[2], el);
            next = 1;
        } else if (travel_failed || el > 150) {
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
    } else if (s->kind == S_MARK || s->kind == S_CAMERA || s->kind == S_QUIET || s->kind == S_DUMP || s->kind == S_CAPTURE) {
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

static void on_frame(void* u) {
    (void)u;
    double t = now_s();
    float p[3], yaw = 0;
    int have = api->player_position(p, &yaw) == 0;
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
    if (tour && *tour) load_tour(tour);
    a->on_frame(on_frame, 0);
    a->log("mapval: loaded (start after %.0f s in the world)", start_after);
    return 0;
}
