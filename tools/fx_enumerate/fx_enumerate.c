/* fx_enumerate: a test plugin that plays every effect animation of a boss
 * (tools/fx_enumerate/plan.py's plan, FXE_PLAN) one at a time and logs every
 * effect request the game's effect system holds - made, retired, gone, and
 * how long each lived - for tools/fx_enumerate/report.py. Never shipped.
 *
 * plugins/boss_rush (one round, the boss's) stands the player 1 m short of
 * the fog wall, facing it (FXE_FOG=x,y,z: that stand): Cross there until the
 * player is through - 1.5 m on from where it first pressed, the way it faced
 * there (the traversal takes it ~1.8 m, past the wall's plane). Then the
 * debug menu's CHR DBG flags (0x593e880..): Player No Dead (0x593e888), All
 * No Dead (0x593e890) and All No Update Ai (0x593e895; FXE_AI=1 leaves it
 * off) - the boss plays only what it is made to; the player is placed
 * FXE_NEAR (7) m in front of it. Each animation: bit 9 of ChrIns+0x1e1
 * cleared and the event module's request set (0x1e59190, the call
 * ForcePlayAnimation 0x17cbfd0 makes); the next after FXE_MIN_S (2.5) once
 * the requests made since have gone, or after FXE_MAX_S (14). At the end,
 * FXE_TAIL_S (20) for the last ones, then "fxe: done".
 *
 * The requests: SprjSfx (0x5940370) +0x10, a list (next +0, request +0x10);
 * a request's effect id +0x68, state +0xc (3 retired), its instance +0x88. */
#include "bbhost_plugin.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const BbHostApi* api;
static uint64_t frames;

static int rd(uint64_t a, void* out, size_t n) { return api->read(a, out, n); }
static uint64_t r64(uint64_t a) { uint64_t v = 0; return rd(a, &v, 8) ? 0 : v; }
static uint32_t r32(uint64_t a) { uint32_t v = 0; return rd(a, &v, 4) ? 0 : v; }
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}
static double env_f(const char* n, double d) { const char* e = getenv(n); return e && *e ? atof(e) : d; }

/* the plan */
#define MAXA 1024
static int nanim, boss_model;
static int32_t anim_id[MAXA];
static float anim_span[MAXA];

/* the requests */
#define MAXR 4096
typedef struct {
    uint64_t req;
    uint32_t id, state;
    int anim;  /* the plan's index current when it was made; -1 before the first */
    double born, retired;
    int live, seen;
} R;
static R rq[MAXR];
static int cur = -1;  /* the animation playing */

static R* find(uint64_t req) {
    for (int i = 0; i < MAXR; ++i)
        if (rq[i].live && rq[i].req == req) return &rq[i];
    return 0;
}
static R* fresh(void) {
    for (int i = 0; i < MAXR; ++i)
        if (!rq[i].live) return &rq[i];
    return 0;
}
static int aid(int i) { return i >= 0 && i < nanim ? anim_id[i] : -1; }

static void track(double t) {
    const uint64_t sfx = r64(api->guest_addr(0x5940370));
    const uint64_t head = sfx ? r64(sfx + 0x10) : 0;
    if (!head) return;
    for (int i = 0; i < MAXR; ++i) rq[i].seen = 0;
    uint64_t n = r64(head);
    for (int k = 0; n && n != head && k < 20000; ++k, n = r64(n)) {
        const uint64_t req = r64(n + 0x10);
        if (!req) continue;
        const uint32_t id = r32(req + 0x68), state = r32(req + 0xc);
        R* e = find(req);
        if (!e || e->id != id) {
            if (e) e->live = 0;  /* the slot went to another request between two looks */
            e = fresh();
            if (!e) continue;
            e->req = req; e->id = id; e->state = state; e->anim = cur; e->born = t; e->retired = 0; e->live = 1;
            api->log("fxe: f%llu new a%d %u req %llx state %u", (unsigned long long)frames, aid(cur), id, (unsigned long long)req, state);
        } else if (state != e->state) {
            if (state == 3 && !e->retired) {
                e->retired = t;
                api->log("fxe: f%llu retired a%d %u req %llx after %.2f s", (unsigned long long)frames, aid(e->anim), id, (unsigned long long)req, t - e->born);
            }
            e->state = state;
        }
        e->seen = 1;
    }
    for (int i = 0; i < MAXR; ++i) {
        if (!rq[i].live || rq[i].seen) continue;
        api->log("fxe: f%llu gone a%d %u req %llx after %.2f s (state %u%s)", (unsigned long long)frames, aid(rq[i].anim), rq[i].id,
                 (unsigned long long)rq[i].req, t - rq[i].born, rq[i].state, rq[i].retired ? ", retired" : "");
        rq[i].live = 0;
    }
}
static int live_of(int a) {
    int n = 0;
    for (int i = 0; i < MAXR; ++i) n += rq[i].live && rq[i].anim == a;
    return n;
}

/* the boss, by model, in WorldChrMan's list */
static const BbChr* boss(BbChr* list) {
    size_t count = 0;
    if (api->chr_list(list, 128, &count) != 0) return 0;
    for (size_t i = 0; i < count && i < 128; ++i)
        if ((int)list[i].model == boss_model && !list[i].is_player) return &list[i];
    return 0;
}

static void request(const char* tag, int n, const char* line) {
    const char* dir = getenv("BBHOST_TEST_REQUESTS");
    if (!dir || !*dir) return;
    char tmp[1024], req[1024];
    snprintf(tmp, sizeof tmp, "%s/fxe-%s-%03d.tmp", dir, tag, n % 1000);
    snprintf(req, sizeof req, "%s/fxe-%s-%03d.req", dir, tag, n % 1000);
    FILE* f = fopen(tmp, "w");
    if (!f) return;
    fputs(line, f);
    fclose(f);
    rename(tmp, req);
}

enum { FOG, THROUGH, PLAY, TAIL, DONE };
static int stage = FOG, presses = 0, near_frames = 0;
static float fog[3], pressed_at[3], pressed_yaw;
static double stage_t, played_t, next_press;

static void play(const BbChr* b, int i, double t) {
    const uint64_t ins = (uint64_t)(uintptr_t)b->ins;
    const uint64_t mods = r64(ins + 0x3b0), ev = mods ? r64(mods + 0x58) : 0;
    uint32_t f1e1 = 0;
    if (!ev || rd(ins + 0x1e1, &f1e1, 4)) return;
    f1e1 &= ~0x200u;
    *(volatile uint32_t*)(uintptr_t)(ins + 0x1e1) = f1e1;
    BbCallRegs r;
    memset(&r, 0, sizeof r);
    r.arg[0] = ev;
    r.arg[1] = (uint32_t)anim_id[i];
    const int rc = api->call_guest(0x1e59190, &r);
    cur = i;
    played_t = t;
    api->log("fxe: f%llu play %d/%d a%d (TAE span %.2f s; call %d) at (%.1f %.1f %.1f)", (unsigned long long)frames, i + 1, nanim, anim_id[i],
             anim_span[i], rc, b->pos[0], b->pos[1], b->pos[2]);
}

static void on_frame(void* user) {
    (void)user;
    ++frames;
    const double t = now_s();
    track(t);
    float at[3], yaw = 0;
    if (api->player_position(at, &yaw) != 0) return;
    BbChr list[128];
    if (stage == FOG) {
        const float dx = at[0] - fog[0], dz = at[2] - fog[2], d2 = dx * dx + dz * dz;
        /* the way on from the first press along the facing then, (-sin, -cos) of the yaw: the traversal never
         * takes the player 3 m from the stand, so distance alone never says it is through */
        const float on = presses ? (at[0] - pressed_at[0]) * -sinf(pressed_yaw) + (at[2] - pressed_at[2]) * -cosf(pressed_yaw) : 0.0f;
        if (presses && (d2 > 9.0f || on > 1.5f)) {
            stage = THROUGH;
            stage_t = t;
            api->log("fxe: f%llu through the fog after %d presses (%.1f %.1f %.1f)", (unsigned long long)frames, presses, at[0], at[1], at[2]);
            return;
        }
        if (d2 > 4.0f) { near_frames = 0; return; }
        if (++near_frames < 60 || t < next_press || presses >= 8) return;
        request("fog", presses, "tap cross 300\n");
        if (!presses) { memcpy(pressed_at, at, sizeof pressed_at); pressed_yaw = yaw; }
        ++presses;
        next_press = t + 3.0;
        api->log("fxe: f%llu Cross at the fog (%d)", (unsigned long long)frames, presses);
        return;
    }
    if (stage == THROUGH) {
        if (t - stage_t < 3.0) return;
        const BbChr* b = boss(list);
        if (!b) {
            if (t - stage_t > 30.0) { api->log("fxe: no c%04d in the world - done", boss_model); stage = DONE; }
            return;
        }
        uint8_t one = 1;
        int r = api->write(0x593e888, &one, 1);  /* Player No Dead */
        r |= api->write(0x593e890, &one, 1);     /* All No Dead */
        const char* ai = getenv("FXE_AI");
        if (!(ai && ai[0] == '1')) r |= api->write(0x593e895, &one, 1);  /* All No Update Ai */
        const float near = (float)env_f("FXE_NEAR", 7.0);
        float ux = at[0] - b->pos[0], uz = at[2] - b->pos[2];
        const float ul = sqrtf(ux * ux + uz * uz);
        if (ul < 0.1f) { ux = 0; uz = 1; } else { ux /= ul; uz /= ul; }
        const float p[3] = {b->pos[0] + ux * near, b->pos[1] + 0.3f, b->pos[2] + uz * near};
        const int w = api->player_warp(0, p, atan2f(ux, uz));
        api->log("fxe: f%llu c%04d at (%.1f %.1f %.1f) hp %d/%d; flags %s (AI %s); the player %.0f m from it: %s", (unsigned long long)frames, boss_model,
                 b->pos[0], b->pos[1], b->pos[2], b->hp, b->max_hp, r ? "REFUSED" : "on", ai && ai[0] == '1' ? "running" : "stopped", near,
                 w == 0 ? "queued" : "refused");
        stage = PLAY;
        stage_t = t + 2.0;  /* the first animation after the warp */
        cur = -1;
        return;
    }
    if (stage == PLAY) {
        if (t < stage_t) return;
        const BbChr* b = boss(list);
        if (!b) return;
        const double min_s = env_f("FXE_MIN_S", 2.5), max_s = env_f("FXE_MAX_S", 14.0);
        if (cur >= 0) {
            const double el = t - played_t;
            const int live = live_of(cur);
            if (el < min_s || (live && el < max_s)) return;
            if (live) api->log("fxe: f%llu a%d: %d request(s) live at %.1f s, the cap - on to the next", (unsigned long long)frames, anim_id[cur], live, el);
        }
        if (cur + 1 >= nanim) {
            stage = TAIL;
            stage_t = t;
            api->log("fxe: f%llu all %d played; %.0f s for the last requests", (unsigned long long)frames, nanim, env_f("FXE_TAIL_S", 20.0));
            return;
        }
        play(b, cur + 1, t);
        return;
    }
    if (stage == TAIL) {
        int live = 0;
        for (int i = 0; i < MAXR; ++i) live += rq[i].live && rq[i].anim >= 0;
        if (live && t - stage_t < env_f("FXE_TAIL_S", 20.0)) return;
        for (int i = 0; i < MAXR; ++i)
            if (rq[i].live && rq[i].anim >= 0)
                api->log("fxe: f%llu still a%d %u req %llx alive %.2f s (state %u%s)", (unsigned long long)frames, aid(rq[i].anim), rq[i].id,
                         (unsigned long long)rq[i].req, t - rq[i].born, rq[i].state, rq[i].retired ? ", retired" : "");
        api->log("fxe: f%llu done: c%04d, %d animations", (unsigned long long)frames, boss_model, nanim);
        stage = DONE;
    }
}

BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {BB_PLUGIN_API_VERSION, BB_PLUGIN_OPT_IN | BB_PLUGIN_GAMEPLAY, "fx_enumerate", "fx_enumerate",
                                                      "0", "bbhost", "plays every effect animation of a boss and logs the effects (a test)"};

BB_PLUGIN_EXPORT int bb_plugin_init(const BbHostApi* a) {
    api = a;
    return 0;
}

BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* a) {
    api = a;
    if (!a->eboot_is_109() || a->version < 8) return 1;
    const char* fe = getenv("FXE_FOG");
    if (!fe || sscanf(fe, "%f,%f,%f", &fog[0], &fog[1], &fog[2]) != 3) { a->log("fxe: FXE_FOG=x,y,z is not set"); return 1; }
    const char* path = getenv("FXE_PLAN");
    FILE* f = path ? fopen(path, "r") : 0;
    if (!f) { a->log("fxe: no plan (FXE_PLAN)"); return 1; }
    char line[8192];
    const char* want = getenv("FXE_ANIMS");  /* attacks (3000-3999, the default) or all */
    const int all = want && !strcmp(want, "all");
    while (fgets(line, sizeof line, f)) {
        int id = 0, m = 0;
        float span = 0;
        if (sscanf(line, "boss %*d %d", &m) == 1) {
            if (boss_model) break;  /* the plan's first boss only */
            boss_model = m;
        }
        else if (sscanf(line, "anim %d %f", &id, &span) == 2 && nanim < MAXA && (all || (id >= 3000 && id < 4000)))
            anim_id[nanim] = id, anim_span[nanim++] = span;
    }
    fclose(f);
    a->log("fxe: c%04d, %d animations to play (%s), fog (%.1f %.1f %.1f)", boss_model, nanim, all ? "all" : "attacks, 3000-3999", fog[0], fog[1], fog[2]);
    a->on_frame(on_frame, 0);
    return 0;
}
