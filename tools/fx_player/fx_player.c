/* fx_player: a test plugin that makes the player's own combat effects through
 * the game's paths, one shot after another (tools/fx_player/plan.py's plan,
 * FXP_PLAN), and measures every effect instance each shot makes, for
 * tools/fx_player/report.py. Never shipped.
 *
 * The shots, each from the main thread at a frame's start:
 *   sfx       an effect on one of the player's dummy polys as the character's
 *             own effects are made (sfx_at below: a SprjSfx request for the
 *             character, bound to the dummy poly's model, as sub_1ca6410 does
 *             it); a weapon's dummy poly with no such weapon in hand is the
 *             body's dummy poly 1, and when no effect of that id appears
 *             within FXP_SPAWN_FRAMES (6) frames, once more at dummy poly 1.
 *             The request is ended when the shot's TAE event length (the
 *             plan's hold) has passed, as a character's effect components end
 *             theirs when their event ends (the request's flags +0x10 |= 1);
 *             FXP_TAE_END=0 leaves it to run, so a looping effect lives on.
 *   behavior  a BehaviorParam_PC row, and
 *   goods     an EquipParamGoods row, both through the character's behaviour
 *             call sub_1cb61c0(emit, chr + 0x178, chr), where every bullet a
 *             character makes goes (the TAE's bullet events through
 *             sub_1cc7530, item use through sub_1cc9160, the event scripts'
 *             CastPointSpell through sub_1e40780): a goods row's refCategory
 *             1 is a bullet (its refId), 2 a SpEffect; a behaviour's refType
 *             the same. The emit record is laid out as those three make it
 *             (no costs; no dummy poly, so the bullet starts at the record's
 *             launch transform +0x30: the player's own rotation turned down by
 *             the shot's pitch, 1.4 m up and 0.5 m ahead of its feet).
 *   speffect  a SpEffectParam row on the player (the host's sp_effect_apply,
 *             the game's own SpEffect call), taken off again after the shot's
 *             seconds through the game's removal (sub_1be1d90 on the
 *             player's SpEffect list, as EraseEventSpecialEffect does).
 * A shot ends once FXP_MIN_S (1) game seconds have gone, its TAE effects were
 * ended and every effect it made has gone, or after FXP_MAX_S (15);
 * FXP_GAP_S (0.5) later the next.
 * Game seconds are frames / FXP_FPS (the run's rate: each frame one step of
 * it - the game's own 1/30 below 30 fps, BBHOST_SIM_FRAME_TIME's target at a
 * converted rate).
 *
 * What is measured, per request (SprjSfx 0x5940370 +0x10: a ring, the request
 * at +0x10; its state +0xc, 3 retired; its effect id +0x68; its instance
 * +0x88) made while a shot was the last one made, ids alive before the first
 * shot left out (the place's own):
 *   - its life, in frames and in frames with a whole effect step (the step
 *     slot SprjSfx::Update's load at 0x2377d4a reads, not 0);
 *   - its instance's node tree (+0x78; a node's child +0x70, sibling +0x68,
 *     parent +0x60, as sub_2c4f0e0 walks it): the nodes seen, and for staged
 *     nodes (flags +0x78 bit 3) the state +0xdc's changes (edges) and the
 *     emission count +0xe0's adds (emissions);
 *   - the container updates (sub_2c4c8c0, hooked: (node + 0x80, &step, ...),
 *     which every node update calls, the game's or a decomp leaf's) of its
 *     nodes, by frame: a frame with a whole step, or a frame between whole
 *     steps on which the containers got a step (the smooth steps' frame
 *     domain) or none (held, as effect steps 30 hold them) - "kept";
 *   - its classes: the vtables of each node's motion (+0x90; a composite's
 *     two parts at +8 / +0x10), modules (+0x98, +0xa0, +0xa8) and pending
 *     binders' targets (the container's list +0xb0: next +0x18, target +0x20;
 *     a GPU binder's (0x56e3100) type instance *(target + 0x10) + 8);
 *   - frame words: each of those objects' first FXP_WORD_BYTES (0x200) bytes
 *     at every frame's start, against the frame before (the same object), by
 *     word, filed under the frame between's kind (whole or between) - for
 *     report.py's "the frame words change on every frame".
 * The log lines (fxp: ...) are what report.py reads; FXP_WORDS (a path) gets
 * the word table at the end and every FXP_WORDS_EVERY (3000) frames.
 * FXP_DUMPS (a folder) with FXP_DUMP_S (game seconds after a shot, "0.25,1")
 * asks BBHOST_TEST_REQUESTS for the displayed frame then. FXP_EXIT=1 ends
 * the run (the `exit` request) when the plan is done. */
#include "bbhost_plugin.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const BbHostApi* api;
static uint64_t frames;
static uint64_t img_rt;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static int rd(uint64_t a, void* out, size_t n) { return api->read(a, out, n); }
static uint64_t r64(uint64_t a) { uint64_t v = 0; return rd(a, &v, 8) ? 0 : v; }
static uint32_t r32(uint64_t a) { uint32_t v = 0; return rd(a, &v, 4) ? 0 : v; }
static float f32of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static int heap_ptr(uint64_t v) { return v > 0x100000000ull && v < 0x800000000000ull; }
static uint32_t bn_of(uint64_t rt) { return rt >= img_rt && rt < img_rt + 0x6000000ull ? (uint32_t)(rt - img_rt + 0x400000) : 0; }
static double env_f(const char* n, double d) { const char* e = getenv(n); return e && *e ? atof(e) : d; }
static int env_i(const char* n, int d) { const char* e = getenv(n); return e && *e ? atoi(e) : d; }

/* the effects' step slot: the address the load at 0x2377d4a reads */
static uint64_t step_slot;
static float slot_value(void) {
    uint32_t u = 0;
    if (!step_slot || rd(step_slot, &u, 4)) return -1.0f;
    return f32of(u);
}

/* --- the plan ------------------------------------------------------------------------------- */
enum { K_SFX, K_BEHAVIOR, K_GOODS, K_SPEFFECT };
typedef struct {
    char family[48], name[96];
    int kind, id, dummy, repeat;
    float arg, pitch;  /* arg: a SpEffect's seconds */
    float hold;        /* an sfx shot's TAE event length, game seconds: its effect is ended then (0: at once) */
} Shot;
#define MAXS 512
static Shot plan[MAXS];
static int nshots;

static int load_plan(const char* path) {
    FILE* f = path ? fopen(path, "r") : 0;
    if (!f) return 0;
    char line[512], fam[48], kind[16], name[96];
    while (fgets(line, sizeof line, f) && nshots < MAXS) {
        int id = 0, rep = 1;
        float dmy = 0, pitch = 0, hold = 0;
        if (sscanf(line, "shot %47s %15s %d %f %f %d %95s %f", fam, kind, &id, &dmy, &pitch, &rep, name, &hold) < 7) continue;
        Shot* s = &plan[nshots];
        memset(s, 0, sizeof *s);
        snprintf(s->family, sizeof s->family, "%s", fam);
        snprintf(s->name, sizeof s->name, "%s", name);
        s->id = id;
        s->pitch = pitch;
        s->hold = hold > 0 ? hold : 0;
        s->repeat = rep < 1 ? 1 : rep;
        if (!strcmp(kind, "sfx")) s->kind = K_SFX, s->dummy = (int)dmy;
        else if (!strcmp(kind, "behavior")) s->kind = K_BEHAVIOR, s->dummy = -1;
        else if (!strcmp(kind, "goods")) s->kind = K_GOODS, s->dummy = -1;
        else if (!strcmp(kind, "speffect")) s->kind = K_SPEFFECT, s->arg = dmy;
        else continue;
        ++nshots;
    }
    fclose(f);
    return nshots;
}

/* --- effect instances ------------------------------------------------------------------------ */
/* Every effect instance whose nodes run their container updates (the hook below sees each call: (node + 0x80, &step,
 * ...); node +0x58 its instance), whatever made it: a SprjSfx request (a character's effects, map effects), a bullet
 * (its effects are made in the manager directly, sub_23786e0), the event scripts' CreateSfx. Its effect id: the
 * instance +0 is the definition sub_2c53020 hands it (*(definition data + 8)); the FXR's root object holds the id
 * (type 0x85 at +0, the id at +8; in a whole file, "FXR\0" at +0 and the id at +0x38). An instance not seen for
 * GONE_FRAMES frames has gone; one whose definition changed is a new instance in the old one's memory. */
#define MAXI 4096
#define MAXN 512  /* nodes remembered per instance */
#define MAXC 16   /* classes remembered per instance */
#define GONE_FRAMES 8
typedef struct {
    uint64_t node;
    uint32_t state, count;
    uint8_t staged, seen;
} NodeRec;
typedef struct {
    uint64_t inst, def;
    uint32_t id;
    int shot, plan, ambient, finished;  /* the shot made (counting repeats) and its line in the plan; ambient: the place's own */
    uint64_t born, born_ticks, last_seen;
    /* node tree */
    NodeRec nodes[MAXN];
    int nnodes, nodes_total, nodes_lost;
    uint32_t edges, emissions;
    uint32_t classes[MAXC];
    int nclasses;
    /* container calls (the hook, under mu) and the frames they make */
    uint32_t cur_whole, cur_held, cur_smooth;  /* this frame's calls */
    double cur_step_sum;
    uint32_t calls_whole, calls_held, calls_smooth;
    uint32_t frames_whole, frames_kept, frames_smooth;
    double step_sum_whole, step_sum_between;
    /* the first frames' root node position (the flight of a bullet's effect) */
    float path[8][3];
    int npath;
} Inst;
static Inst* in;  /* MAXI, dense: the first ninst used */
static int ninst;
static int tracking;  /* the hook makes slots for new instances */
static uint64_t lost_insts;
static int shot_cur = -1, shot_rep = 0, shot_index = 0;  /* the plan's shot, its repeat; shot_index counts every shot made */
static uint64_t shot_at, shot_end_at;
#define MAXAMB 2048
static uint32_t ambient[MAXAMB];
static int nambient;
static int is_ambient(uint32_t id) {
    for (int i = 0; i < nambient; ++i)
        if (ambient[i] == id) return 1;
    return 0;
}
static void add_ambient(uint32_t id) {
    if (id && !is_ambient(id) && nambient < MAXAMB) ambient[nambient++] = id;
}

/* instance -> slot (rebuilt each frame; the hook adds the new) */
#define IMAP 16384
static uint64_t imap_inst[IMAP];
static int imap_slot[IMAP];
static void imap_put(uint64_t inst, int slot) {
    uint32_t h = (uint32_t)((inst >> 4) * 2654435761u) & (IMAP - 1);
    for (int k = 0; k < 256; ++k, h = (h + 1) & (IMAP - 1))
        if (!imap_inst[h] || imap_inst[h] == inst) {
            imap_inst[h] = inst;
            imap_slot[h] = slot;
            return;
        }
}
static int imap_find(uint64_t inst) {
    uint32_t h = (uint32_t)((inst >> 4) * 2654435761u) & (IMAP - 1);
    for (int k = 0; k < 256; ++k, h = (h + 1) & (IMAP - 1)) {
        if (imap_inst[h] == inst) return imap_slot[h];
        if (!imap_inst[h]) return -1;
    }
    return -1;
}
static void imap_rebuild(void) {
    memset(imap_inst, 0, sizeof imap_inst);
    for (int i = 0; i < ninst; ++i) imap_put(in[i].inst, i);
}

/* An instance's effect id: the FXR's root object (type 0x85 at +0, the id at +8) or the whole file ("FXR\0" at +0,
 * the id at +0x38), reached from the instance by one pointer (+o1) or two (+o1, then +o2). The path is the one the
 * place's own requests agree with at the start (calibrate_ids: each request's instance +0x88 and id +0x68) - the
 * instance +0 is what sub_2c53020 gives it from the effect's definition. */
static int id_o1 = 0, id_o2 = -1, id_o3 = -1;
static uint32_t id_from(uint64_t inst, int o1, int o2, int o3) {
    uint64_t p = r64(inst + (uint64_t)o1);
    if (o2 >= 0) p = heap_ptr(p) ? r64(p + (uint64_t)o2) : 0;
    if (o3 >= 0) p = heap_ptr(p) ? r64(p + (uint64_t)o3) : 0;
    if (!heap_ptr(p)) return 0;
    const uint32_t w = r32(p);
    if (w == 0x85) return r32(p + 8);                             /* the root object: its type, then its id */
    if (w == 0x00525846 || r32(p + 0x30) == 0x85) return r32(p + 0x38);  /* the file: "FXR\0", the root at +0x30 */
    return 0;
}
/* An instance's id from its creation: every effect made by id goes through sub_2c53020 (system, id, ...), whose new
 * instance (+0x38 the system) is handed to sub_2c505f0 before its nodes are built - the id is kept from the first call
 * (per thread) and put on the instance at the second. */
#define CMAP 65536
static uint64_t cmap_inst[CMAP];
static uint32_t cmap_id[CMAP];
static pthread_mutex_t cmu = PTHREAD_MUTEX_INITIALIZER;
static __thread uint64_t pend_sys;
static __thread uint32_t pend_id;
static uint64_t created_named;
static int use_def_path = 1;
static int h_create(BbHookCtx* ctx, void* user) {
    (void)user;
    const uint32_t id = (uint32_t)ctx->arg[1];
    if (id && id != 0xfffffffeu) pend_sys = ctx->arg[0], pend_id = id;
    return 0;
}
static int h_created(BbHookCtx* ctx, void* user) {
    (void)user;
    const uint64_t inst = ctx->arg[0];
    const uint32_t id = pend_id;
    const uint64_t sys = pend_sys;
    pend_id = 0;
    if (!id || !heap_ptr(inst) || r64(inst + 0x38) != sys) return 0;
    pthread_mutex_lock(&cmu);
    uint32_t h = (uint32_t)((inst >> 4) * 2654435761u) & (CMAP - 1);
    for (int k = 0; k < 256; ++k, h = (h + 1) & (CMAP - 1))
        if (!cmap_inst[h] || cmap_inst[h] == inst) {
            cmap_inst[h] = inst, cmap_id[h] = id;
            ++created_named;
            break;
        }
    pthread_mutex_unlock(&cmu);
    return 0;
}
static uint32_t created_id(uint64_t inst) {
    uint32_t id = 0;
    pthread_mutex_lock(&cmu);
    uint32_t h = (uint32_t)((inst >> 4) * 2654435761u) & (CMAP - 1);
    for (int k = 0; k < 256; ++k, h = (h + 1) & (CMAP - 1)) {
        if (cmap_inst[h] == inst) { id = cmap_id[h]; break; }
        if (!cmap_inst[h]) break;
    }
    pthread_mutex_unlock(&cmu);
    return id;
}
/* An instance's id from the request that holds it (+0x88), for when its definition gives none: the requests' effects
 * only - the ones a manager makes directly have no request. */
static uint32_t req_id_of_inst(uint64_t inst) {
    const uint64_t sfx = r64(api->guest_addr(0x5940370));
    const uint64_t head = sfx ? r64(sfx + 0x10) : 0;
    uint64_t q = head ? r64(head) : 0;
    for (int k = 0; q && q != head && k < 20000; ++k, q = r64(q)) {
        const uint64_t req = r64(q + 0x10);
        if (req && r64(req + 0x88) == inst) return r32(req + 0x68);
    }
    return 0;
}
static uint32_t id_of_inst(uint64_t inst) { return heap_ptr(inst) ? id_from(inst, id_o1, id_o2, id_o3) : 0; }
static void calibrate_ids(void) {
    uint64_t inst[64];
    uint32_t rid[64];
    int n = 0;
    const uint64_t sfx = r64(api->guest_addr(0x5940370));
    const uint64_t head = sfx ? r64(sfx + 0x10) : 0;
    uint64_t q = head ? r64(head) : 0;
    for (int k = 0; q && q != head && k < 20000 && n < 64; ++k, q = r64(q)) {
        const uint64_t req = r64(q + 0x10);
        const uint64_t i = req ? r64(req + 0x88) : 0;
        if (heap_ptr(i)) inst[n] = i, rid[n] = r32(req + 0x68), ++n;
    }
    /* the shortest path that agrees with the most: one pointer, then two, then three */
    int best = 0, b1 = 0, b2 = -1, b3 = -1;
    for (int depth = 1; depth <= 3 && best < n; ++depth)
        for (int o1 = 0; o1 < 0x98; o1 += 8)
            for (int o2 = depth >= 2 ? 0 : -1; o2 < (depth >= 2 ? 0x40 : 0); o2 += depth >= 2 ? 8 : 1)
                for (int o3 = depth >= 3 ? 0 : -1; o3 < (depth >= 3 ? 0x40 : 0); o3 += depth >= 3 ? 8 : 1) {
                    int m = 0;
                    for (int k = 0; k < n; ++k) m += rid[k] && id_from(inst[k], o1, o2, o3) == rid[k];
                    if (m > best) best = m, b1 = o1, b2 = o2, b3 = o3;
                }
    if (best > 0) id_o1 = b1, id_o2 = b2, id_o3 = b3;
    else use_def_path = 0;  /* the ids from the instances' creation (and their requests) only */
    api->log("fxp: f%llu effect ids from instances: the pointer at +0x%x, then %d, then %d (-1: none); %d of %d of the place's requests agree%s",
             (unsigned long long)frames, id_o1, id_o2, id_o3, best, n, best > 0 ? "" : " - NO PATH FOUND, ids unread");
}

/* this frame's kind, from the container calls: 1 a whole step, 2 between steps, 0 no call seen */
static volatile int frame_kind;
static uint64_t frames_whole, frames_between;
static uint64_t ticks;  /* frames with a whole effect step so far */

static int h_container(BbHookCtx* ctx, void* user) {
    (void)user;
    const uint64_t c = ctx->arg[0];
    if (!heap_ptr(c) || r64(c + 8) != c - 0x80 || !ctx->arg[1]) return 0;
    const float step = f32of(r32(ctx->arg[1]));
    const float sv = slot_value();
    const int between = sv == 0.0f;
    const uint64_t inst = r64(c - 0x80 + 0x58);
    pthread_mutex_lock(&mu);
    if (between) {
        if (frame_kind != 1) frame_kind = 2;
    } else {
        frame_kind = 1;
    }
    int i = heap_ptr(inst) ? imap_find(inst) : -1;
    if (i < 0 && tracking && heap_ptr(inst)) {
        if (ninst < MAXI) {
            i = ninst++;
            Inst* n = &in[i];
            memset(n, 0, sizeof *n);
            n->inst = inst;
            n->born = frames + 1;  /* the frame now running: on_frame counts it when it ends */
            n->born_ticks = ticks;
            n->shot = -2;  /* to be attributed at the frame's end */
            imap_put(inst, i);
        } else {
            ++lost_insts;
        }
    }
    if (i >= 0) {
        Inst* r = &in[i];
        if (!between) ++r->cur_whole, r->cur_step_sum += step;
        else if (step != 0.0f) ++r->cur_smooth, r->cur_step_sum += step;
        else ++r->cur_held;
    }
    pthread_mutex_unlock(&mu);
    return 0;
}

/* --- frame words ------------------------------------------------------------------------------ */
enum { O_MOTION, O_MODA, O_MODB, O_MODC, O_TARGET, O_GXTYPE, O_N };
static const char* oname[O_N] = {"motion", "moduleA", "moduleB", "moduleC", "binder", "gxtype"};
static int word_bytes = 0x200;
#define SNAP 8192
typedef struct {
    uint64_t obj, owner;
    uint64_t frame;
    uint32_t cls, id;
    uint8_t kind, used;
    uint8_t b[0x200];
} Snap;
static Snap* snaps;
#define WSTAT 131072
typedef struct {
    uint64_t okey;
    uint32_t off, used;
    uint32_t chg[2];  /* whole, between */
} WStat;
static WStat* wstats;
#define OSTAT 8192
typedef struct {
    uint64_t key;
    uint32_t obs[2];
} OStat;
static OStat* ostats;
static uint64_t wlost;

static uint64_t okey(uint32_t id, int kind, uint32_t cls) { return ((uint64_t)id << 32) ^ ((uint64_t)kind << 28) ^ (uint64_t)cls; }
static OStat* ostat(uint64_t key) {
    uint32_t h = (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 40) & (OSTAT - 1);
    for (int i = 0; i < 256; ++i, h = (h + 1) & (OSTAT - 1)) {
        if (ostats[h].key == key) return &ostats[h];
        if (!ostats[h].key) { ostats[h].key = key; return &ostats[h]; }
    }
    return 0;
}
static WStat* wstat(uint64_t okey_, uint32_t off) {
    uint32_t h = (uint32_t)(((okey_ ^ ((uint64_t)off << 52)) * 0x9E3779B97F4A7C15ull) >> 40) & (WSTAT - 1);
    for (int i = 0; i < 512; ++i, h = (h + 1) & (WSTAT - 1)) {
        WStat* w = &wstats[h];
        if (w->used && w->okey == okey_ && w->off == off) return w;
        if (!w->used) { w->used = 1, w->okey = okey_, w->off = off; return w; }
    }
    return 0;
}
/* the transition that just ended: kind 0 whole, 1 between, -1 unknown (no snapshot filed) */
static int transition = -1;
static void observe(uint32_t id, int kind, uint64_t owner, uint64_t obj) {
    if (!heap_ptr(obj)) return;
    const uint64_t vt = r64(obj);
    const uint32_t cls = bn_of(vt);
    if (!cls) return;
    uint8_t cur[0x200];
    const int n = word_bytes;
    if (rd(obj, cur, (size_t)n)) return;
    const uint64_t k = obj ^ (owner << 7) ^ ((uint64_t)kind << 60) ^ ((uint64_t)id * 0x100000001b3ull);
    uint32_t h = (uint32_t)((k * 0x9E3779B97F4A7C15ull) >> 40) & (SNAP - 1);
    Snap* s = 0;
    for (int i = 0; i < 128; ++i, h = (h + 1) & (SNAP - 1)) {
        Snap* c = &snaps[h];
        if (c->used && c->obj == obj && c->owner == owner && c->kind == kind && c->id == id) { s = c; break; }
        if (!c->used || frames > c->frame + 3) { if (!s) s = c; if (!c->used) break; }
    }
    if (!s) { ++wlost; return; }
    if (s->used && s->obj == obj && s->owner == owner && s->kind == kind && s->cls == cls && s->id == id && s->frame + 1 == frames &&
        transition >= 0) {
        OStat* o = ostat(okey(id, kind, cls));
        if (o) ++o->obs[transition];
        for (int w = 0; w < n / 4; ++w) {
            uint32_t a, b;
            memcpy(&a, s->b + 4 * w, 4);
            memcpy(&b, cur + 4 * w, 4);
            if (a == b) continue;
            WStat* st = wstat(okey(id, kind, cls), (uint32_t)(4 * w));
            if (st) ++st->chg[transition];
            else ++wlost;
        }
    }
    s->used = 1;
    s->obj = obj;
    s->owner = owner;
    s->kind = (uint8_t)kind;
    s->cls = cls;
    s->id = id;
    s->frame = frames;
    memcpy(s->b, cur, (size_t)n);
}
static void dump_words(void) {
    const char* out = getenv("FXP_WORDS");
    if (!out || !*out) return;
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp", out);
    FILE* f = fopen(tmp, "w");
    if (!f) return;
    fprintf(f, "# fx_player words: frames %llu, whole %llu, between %llu, word bytes %d, lost %llu\n", (unsigned long long)frames,
            (unsigned long long)frames_whole, (unsigned long long)frames_between, word_bytes, (unsigned long long)wlost);
    fprintf(f, "id\tkind\tclass\toffset\twhole_obs\tbetween_obs\twhole_chg\tbetween_chg\n");
    for (int i = 0; i < WSTAT; ++i) {
        const WStat* w = &wstats[i];
        if (!w->used) continue;
        const uint64_t ok = w->okey;
        const uint32_t off = w->off;
        const uint32_t id = (uint32_t)(ok >> 32), kind = (uint32_t)((ok >> 28) & 0xf), cls = (uint32_t)(ok & 0xfffffff);
        OStat* o = ostat(ok);
        fprintf(f, "%u\t%s\t%x\t%x\t%u\t%u\t%u\t%u\n", id, kind < O_N ? oname[kind] : "?", cls, off, o ? o->obs[0] : 0, o ? o->obs[1] : 0,
                w->chg[0], w->chg[1]);
    }
    for (int i = 0; i < OSTAT; ++i) {
        const OStat* o = &ostats[i];
        if (!o->key) continue;
        const uint32_t id = (uint32_t)(o->key >> 32), kind = (uint32_t)((o->key >> 28) & 0xf), cls = (uint32_t)(o->key & 0xfffffff);
        fprintf(f, "#obs\t%u\t%s\t%x\t%u\t%u\n", id, kind < O_N ? oname[kind] : "?", cls, o->obs[0], o->obs[1]);
    }
    fclose(f);
    rename(tmp, out);
}

/* --- an instance's node tree, once a frame ---------------------------------------------------- */
static void add_class(Inst* r, uint64_t obj) {
    if (!heap_ptr(obj)) return;
    const uint32_t cls = bn_of(r64(obj));
    if (!cls) return;
    for (int i = 0; i < r->nclasses; ++i)
        if (r->classes[i] == cls) return;
    if (r->nclasses < MAXC) r->classes[r->nclasses++] = cls;
}
static void walk(Inst* r) {
    const uint64_t inst = r->inst;
    if (!heap_ptr(inst)) return;
    for (int i = 0; i < r->nnodes; ++i) r->nodes[i].seen = 0;
    uint64_t node = r64(inst + 0x78);
    for (int steps = 0; heap_ptr(node) && steps < 4096; ++steps) {
        const uint32_t flags = r32(node + 0x78);
        NodeRec* nr = 0;
        for (int i = 0; i < r->nnodes; ++i)
            if (r->nodes[i].node == node) { nr = &r->nodes[i]; break; }
        const int staged = (flags & 8) != 0;
        const uint32_t state = r32(node + 0xdc), count = r32(node + 0xe0);
        if (!nr) {
            ++r->nodes_total;
            if (r->nnodes < MAXN) {
                nr = &r->nodes[r->nnodes++];
                nr->node = node, nr->state = state, nr->count = count, nr->staged = (uint8_t)staged;
            } else {
                ++r->nodes_lost;
            }
        } else if (staged) {
            if (state != nr->state) ++r->edges;
            if (count > nr->count && count - nr->count < 1000) r->emissions += count - nr->count;
            else if (count < nr->count && count < 1000) r->emissions += count;  /* the count restarted */
            nr->state = state, nr->count = count, nr->staged = 1;
        }
        if (nr) nr->seen = 1;
        /* the classes and their words */
        const uint64_t motion = r64(node + 0x90);
        add_class(r, motion);
        observe(r->id, O_MOTION, node, motion);
        if (heap_ptr(motion) && bn_of(r64(motion)) == 0x570e750) {  /* a composite motion: its two parts */
            add_class(r, r64(motion + 8));
            add_class(r, r64(motion + 0x10));
            observe(r->id, O_MOTION, node ^ 1, r64(motion + 8));
            observe(r->id, O_MOTION, node ^ 2, r64(motion + 0x10));
        }
        static const uint8_t mods[3] = {0x98, 0xa0, 0xa8};
        for (int k = 0; k < 3; ++k) {
            const uint64_t m = r64(node + mods[k]);
            add_class(r, m);
            observe(r->id, O_MODA + k, node, m);
        }
        uint64_t item = r64(node + 0xb0);
        for (int k = 0; heap_ptr(item) && k < 32; ++k, item = r64(item + 0x18)) {
            const uint64_t target = r64(item + 0x20);
            add_class(r, target);
            observe(r->id, O_TARGET, node ^ ((uint64_t)k << 2), target);
            if (heap_ptr(target) && bn_of(r64(target)) == 0x56e3100) {
                const uint64_t holder = r64(target + 0x10);
                const uint64_t type = heap_ptr(holder) ? r64(holder + 8) : 0;
                add_class(r, type);
                observe(r->id, O_GXTYPE, type, type);
            }
        }
        /* next: the child, else the sibling of this node or of the nearest parent with one (sub_2c4f0e0) */
        uint64_t next = r64(node + 0x70);
        if (!next) {
            uint64_t up = node;
            while (up && !(next = r64(up + 0x68))) up = r64(up + 0x60);
        }
        node = next;
    }
    /* a node no longer in the tree: its state's last change counted already */
    int k = 0;
    for (int i = 0; i < r->nnodes; ++i)
        if (r->nodes[i].seen) r->nodes[k++] = r->nodes[i];
    r->nnodes = k;
    if (r->npath < 8) {
        const uint64_t root = r64(inst + 0x78);
        if (heap_ptr(root)) {
            float m[12];
            if (!rd(root + 0x20, m, sizeof m)) {
                r->path[r->npath][0] = m[3], r->path[r->npath][1] = m[7], r->path[r->npath][2] = m[11];
                ++r->npath;
            }
        }
    }
}

/* --- the instances, once a frame ------------------------------------------------------------- */
static const char* fam_of(int p) { return p >= 0 && p < nshots ? plan[p].family : "-"; }
static void finish(Inst* r, const char* how) {
    char cls[MAXC * 9 + 2] = "";
    int o = 0;
    for (int i = 0; i < r->nclasses && o < (int)sizeof cls - 10; ++i) o += snprintf(cls + o, sizeof cls - (size_t)o, "%s%x", i ? "," : "", r->classes[i]);
    char path[8 * 3 * 12 + 2] = "";
    o = 0;
    for (int i = 0; i < r->npath && o < (int)sizeof path - 40; ++i)
        o += snprintf(path + o, sizeof path - (size_t)o, "%s%.2f/%.2f/%.2f", i ? ";" : "", r->path[i][0], r->path[i][1], r->path[i][2]);
    const uint64_t life = r->last_seen >= r->born ? r->last_seen - r->born + 1 : 0;
    api->log("fxp: f%llu %s s%d %s %u inst %llx life %llu frames %llu steps retired -1 nodes %d lost %d edges %u emissions %u "
             "frames whole %u kept %u smooth %u none 0 calls whole %u held %u smooth %u step whole %.4f between %.4f classes %s path %s",
             (unsigned long long)frames, how, r->shot, fam_of(r->plan), r->id, (unsigned long long)r->inst, (unsigned long long)life,
             (unsigned long long)(ticks - r->born_ticks), r->nodes_total, r->nodes_lost, r->edges, r->emissions, r->frames_whole, r->frames_kept,
             r->frames_smooth, r->calls_whole, r->calls_held, r->calls_smooth, r->step_sum_whole, r->step_sum_between, cls[0] ? cls : "-",
             path[0] ? path : "-");
}
static int shot_live(int shot) {
    int n = 0;
    for (int i = 0; i < ninst; ++i) n += !in[i].finished && in[i].shot == shot;
    return n;
}
/* The SprjSfx requests (+0x10: a ring, the request at +0x10; its effect id +0x68, its instance +0x88), each frame:
 * their ids checked against their instances' (the id from the definition), and the requests a shot made logged as
 * they come and go ("req new" / "req gone") - a request whose instance never runs its nodes (none of its frames reach
 * the hook) is in the log only this way. */
static uint64_t id_checked, id_differ;
#define MAXQ 1024
typedef struct {
    uint64_t req, inst;
    uint32_t id;
    int shot, plan, seen, live;
    uint64_t born;
} Rq;
static Rq rqs[MAXQ];
static void check_requests(void) {
    const uint64_t sfx = r64(api->guest_addr(0x5940370));
    const uint64_t head = sfx ? r64(sfx + 0x10) : 0;
    uint64_t n = head ? r64(head) : 0;
    for (int i = 0; i < MAXQ; ++i) rqs[i].seen = 0;
    for (int k = 0; n && n != head && k < 20000; ++k, n = r64(n)) {
        const uint64_t req = r64(n + 0x10);
        if (!req) continue;
        const uint64_t inst = r64(req + 0x88);
        const uint32_t rid = r32(req + 0x68);
        const int i = heap_ptr(inst) ? imap_find(inst) : -1;
        if (i >= 0 && !in[i].ambient && in[i].shot >= 0) {
            ++id_checked;
            if (rid != in[i].id && id_differ++ < 8)
                api->log("fxp: f%llu instance %llx: its definition says %u, its request %u", (unsigned long long)frames, (unsigned long long)inst, in[i].id, rid);
        }
        Rq* q = 0;
        for (int j = 0; j < MAXQ; ++j)
            if (rqs[j].live && rqs[j].req == req && rqs[j].id == rid) { q = &rqs[j]; break; }
        if (!q) {
            if (shot_cur < 0 || is_ambient(rid)) continue;
            for (int j = 0; j < MAXQ; ++j)
                if (!rqs[j].live) { q = &rqs[j]; break; }
            if (!q) continue;
            q->req = req, q->inst = inst, q->id = rid, q->shot = shot_index - 1, q->plan = shot_cur, q->live = 1, q->born = frames;
            api->log("fxp: f%llu req new s%d %s %u req %llx inst %llx", (unsigned long long)frames, q->shot, fam_of(q->plan), rid, (unsigned long long)req,
                     (unsigned long long)inst);
        }
        q->seen = 1;
    }
    for (int j = 0; j < MAXQ; ++j)
        if (rqs[j].live && !rqs[j].seen) {
            api->log("fxp: f%llu req gone s%d %s %u req %llx life %llu frames", (unsigned long long)frames, rqs[j].shot, fam_of(rqs[j].plan), rqs[j].id,
                     (unsigned long long)rqs[j].req, (unsigned long long)(frames - rqs[j].born));
            rqs[j].live = 0;
        }
}
static int req_live(int shot) {
    int n = 0;
    for (int j = 0; j < MAXQ; ++j) n += rqs[j].live && rqs[j].shot == shot;
    return n;
}
static int seen_id_since(uint32_t id, uint64_t since) {
    for (int i = 0; i < ninst; ++i)
        if (in[i].id == id && in[i].born >= since) return 1;
    for (int j = 0; j < MAXQ; ++j)
        if (rqs[j].live && rqs[j].id == id && rqs[j].born >= since) return 1;
    return 0;
}
/* At a frame's end (mu held): fold the frame's calls, name and attribute the new instances, end the gone. */
static void instances_frame(void) {
    for (int i = 0; i < ninst; ++i) {
        Inst* r = &in[i];
        if (r->finished) continue;
        const int seen = r->cur_whole || r->cur_held || r->cur_smooth;
        if (seen) {
            r->last_seen = frames;
            r->calls_whole += r->cur_whole, r->calls_held += r->cur_held, r->calls_smooth += r->cur_smooth;
            if (r->cur_whole) ++r->frames_whole, r->step_sum_whole += r->cur_step_sum;
            else if (r->cur_smooth) ++r->frames_smooth, r->step_sum_between += r->cur_step_sum;
            else if (r->cur_held) ++r->frames_kept;
        }
        r->cur_whole = r->cur_held = r->cur_smooth = 0;
        r->cur_step_sum = 0;
        const uint64_t def = r64(r->inst + (uint64_t)id_o1);
        if (r->shot == -2) {  /* new: its id, the shot it is part of, or the place's own */
            r->def = def;
            r->id = use_def_path ? id_of_inst(r->inst) : 0;
            if (!r->id) r->id = created_id(r->inst);
            static int from_req;
            if (!r->id && shot_cur >= 0 && (r->id = req_id_of_inst(r->inst)) && from_req++ < 4)
                api->log("fxp: f%llu instance %llx: its id %u from its request (its definition gives none)", (unsigned long long)frames,
                         (unsigned long long)r->inst, r->id);
            if (shot_cur < 0) {
                r->ambient = 1;
                add_ambient(r->id);
                r->shot = -1;
            } else if (is_ambient(r->id) || !r->id) {
                static int unnamed;
                if (!r->id && unnamed++ < 8)
                    api->log("fxp: f%llu instance %llx: no effect id in its definition %llx (%08x %08x %08x %08x)", (unsigned long long)frames,
                             (unsigned long long)r->inst, (unsigned long long)def, r32(def), r32(def + 4), r32(def + 8), r32(def + 12));
                r->ambient = 1;
                r->shot = -1;
            } else {
                r->shot = shot_index - 1;
                r->plan = shot_cur;
                api->log("fxp: f%llu new s%d %s %u inst %llx", (unsigned long long)frames, r->shot, fam_of(r->plan), r->id, (unsigned long long)r->inst);
            }
        } else if (def != r->def || (!seen && frames - r->last_seen >= GONE_FRAMES)) {
            if (!r->ambient) finish(r, "gone");
            r->finished = 1;
        }
    }
    /* the finished out, the rest packed */
    int k = 0;
    for (int i = 0; i < ninst; ++i)
        if (!in[i].finished) {
            if (k != i) in[k] = in[i];
            ++k;
        }
    ninst = k;
    imap_rebuild();
}

/* --- the shots -------------------------------------------------------------------------------- */
/* The player's ChrIns, from WorldChrMan's list. */
static uint64_t player_chr(void) {
    BbChr list[64];
    size_t count = 0;
    if (api->chr_list(list, 64, &count) != 0) return 0;
    for (size_t i = 0; i < count && i < 64; ++i)
        if (list[i].is_player) return (uint64_t)(uintptr_t)list[i].ins;
    return 0;
}

/* The character's transform as the bullet request takes it (sub_1cb6680): *(chr + 0x48) + 0x20 holds a 4x4 whose
 * rows +0x60/+0x70/+0x80 are the X, Y, Z axes and +0x90 the place; with no such object, chr + 0x58's +0x90.. are the
 * three rows of the 3x4 form already. Out: axes X, Y, Z and the place P. */
static int chr_axes(uint64_t chr, float X[3], float Y[3], float Z[3], float P[3]) {
    const uint64_t a = r64(chr + 0x48);
    float m[16];
    if (heap_ptr(a) && heap_ptr(r64(a + 0x20)) && !rd(r64(a + 0x20) + 0x60, m, sizeof m)) {
        for (int i = 0; i < 3; ++i) X[i] = m[i], Y[i] = m[4 + i], Z[i] = m[8 + i], P[i] = m[12 + i];
        return 1;
    }
    const uint64_t b = r64(chr + 0x58);
    float t[12];
    if (heap_ptr(b) && !rd(b + 0x90, t, sizeof t)) {
        for (int i = 0; i < 3; ++i) X[i] = t[4 * i], Y[i] = t[4 * i + 1], Z[i] = t[4 * i + 2], P[i] = t[4 * i + 3];
        return 1;
    }
    return 0;
}

/* The emit record the character's behaviour call reads (sub_1cb61c0), as sub_1cc9160 / sub_1cc7530 / sub_1e40780 make it. */
typedef struct __attribute__((aligned(16))) {
    int32_t magic, goods, behavior, bullet; /* +0x00: -1 for none; the bullet -1: the behaviour's refId */
    int32_t target, hand, slot;             /* +0x10 a lock target's handle, +0x14 / +0x18 the weapon's hand and slot */
    uint8_t player, event, pad1e[2];        /* +0x1c 1 as the player's own paths set it, +0x1d CastPointSpell's 1 */
    int32_t dummy;                          /* +0x20: -1, the launch transform below */
    uint8_t flag24, pad25[11];
    float launch[3][4];                     /* +0x30: three rows, the axes' components, the place in the fourth */
    float mult[4];                          /* +0x60 */
    int32_t cost[6];                        /* +0x70..+0x84: stamina, items, bullets, HP, durability - none */
    uint8_t pad88[8];
} Emit;
static Emit emit;
static float forward_sign = 1.0f;  /* +Z or -Z is where the player faces */

static void fill_launch(uint64_t chr, float pitch_deg) {
    float X[3], Y[3], Z[3], P[3];
    if (!chr_axes(chr, X, Y, Z, P)) return;
    /* the facing axis F = sign * Z, turned down by the pitch about X: F' = cos p F - sin p Y, Y' = cos p Y + sin p F */
    const float p = pitch_deg * 3.14159265f / 180.0f, c = cosf(p), s = sinf(p);
    float F[3], Fn[3], Yn[3], Zn[3];
    for (int i = 0; i < 3; ++i) F[i] = forward_sign * Z[i];
    for (int i = 0; i < 3; ++i) Fn[i] = c * F[i] - s * Y[i], Yn[i] = c * Y[i] + s * F[i];
    for (int i = 0; i < 3; ++i) Zn[i] = forward_sign * Fn[i];
    /* FXP_FLIP=1: the bullets' flight is the other Z (the transform turned half round its up axis) */
    const float flip = env_i("FXP_FLIP", 0) ? -1.0f : 1.0f;
    for (int i = 0; i < 3; ++i) {
        emit.launch[i][0] = flip * X[i];
        emit.launch[i][1] = Yn[i];
        emit.launch[i][2] = flip * Zn[i];
        emit.launch[i][3] = P[i] + 1.4f * Y[i] + 0.5f * F[i];
    }
}

static int call_behavior(uint64_t chr, int goods, int behavior, int bullet, float pitch) {
    memset(&emit, 0, sizeof emit);
    emit.magic = -1, emit.goods = goods, emit.behavior = behavior, emit.bullet = bullet;
    emit.target = emit.hand = emit.slot = -1;
    emit.player = 1;
    emit.dummy = -1;
    for (int i = 0; i < 4; ++i) emit.mult[i] = 1.0f;
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 4; ++k) emit.launch[i][k] = i == k ? 1.0f : 0.0f;
    fill_launch(chr, pitch);
    BbCallRegs r;
    memset(&r, 0, sizeof r);
    r.arg[0] = (uint64_t)(uintptr_t)&emit;
    r.arg[1] = chr + 0x178;
    r.arg[2] = chr;
    const int rc = api->call_guest(0x1cb61c0, &r);
    return rc ? -1 : (int)(r.ret & 0xff);
}

/* An effect on one of the player's dummy polys as the character's own effects are made (sub_1ca6410's pattern): a
 * SprjSfx request for the character (sub_2378160(&request, SprjSfx, chr, &effect id, *0x57bf090 + 0x30), queued for
 * SprjSfx's next update, which moves it to its list +0x10), its offset +0x30..+0x4f from 0x593e970, then bound to the
 * dummy poly of the model PlayerIns vfunc +0x6b0 names for the id (0 body - 999, a weapon's or armour's by the
 * thousands) through sub_2372fd0, and the reference handed back dropped (the list keeps its own). A dummy poly whose
 * model is not there (a weapon not in hand) is the body's dummy poly 1. Returns the dummy poly used, -1 for none made.
 * (CreateSfx_DummyPoly's work, sub_1719e10, makes its effects in the manager directly, with no request a harness can
 * follow: not used.) */
/* The TAE effects' requests a shot holds (FXP_TAE_END, on by default): each is ended when the shot's event length has
 * passed, the way a character's effect components end theirs (sub_1cac8b0, sub_1ca6410: the request's flags +0x10 |= 1,
 * which SprjSfx's request update, sub_2373850, takes as the end of its effect), and our reference dropped. Off: the
 * request is left to run (a looping effect then lives on into the shots after it). */
#define MAXTAE 8
static uint64_t tae_req[MAXTAE];
static int ntae;
static int tae_end_on = 1;
static void req_unref(uint64_t req) {
    volatile int32_t* refs = (volatile int32_t*)(uintptr_t)(req + 8);
    if (*refs > 1) {
        *refs -= 1;
        return;
    }
    if (*refs != 1) return;
    /* ours was the last reference: the object's release, its vtable's first slot, as DLReferenceCountObject's Unref calls it */
    *refs = 0;
    const uint32_t fn = bn_of(r64(r64(req)));
    if (fn) {
        BbCallRegs r;
        memset(&r, 0, sizeof r);
        r.arg[0] = req;
        api->call_guest(fn, &r);
    }
}
static void tae_end_all(const char* why) {
    for (int i = 0; i < ntae; ++i) {
        const uint64_t req = tae_req[i];
        if (!heap_ptr(req)) continue;
        volatile uint8_t* flags = (volatile uint8_t*)(uintptr_t)(req + 0x10);
        const uint32_t state = r32(req + 0xc);
        *flags |= 1;
        api->log("fxp: f%llu shot s%d: its TAE effect %u ended %s (request state %u)", (unsigned long long)frames, shot_index - 1, r32(req + 0x68), why,
                 state);
        req_unref(req);
    }
    ntae = 0;
}
static int sfx_at(uint64_t chr, int dummy, int id) {
    static int32_t eid, dmy_model, dmy_bind;
    static uint64_t request;
    const uint64_t sfx = r64(api->guest_addr(0x5940370));
    const uint64_t params = r64(api->guest_addr(0x57bf090));
    const uint64_t model_fn = bn_of(r64(r64(chr) + 0x6b0));
    if (!sfx || !params || !model_fn) return -1;
    BbCallRegs r;
    memset(&r, 0, sizeof r);
    dmy_model = dummy;
    r.arg[0] = chr;
    r.arg[1] = (uint64_t)(uintptr_t)&dmy_model;
    if (api->call_guest(model_fn, &r)) return -1;
    uint64_t model = r.ret;
    if (!heap_ptr(model) && dummy != 1) {
        dummy = 1;
        dmy_model = 1;
        memset(&r, 0, sizeof r);
        r.arg[0] = chr;
        r.arg[1] = (uint64_t)(uintptr_t)&dmy_model;
        if (api->call_guest(model_fn, &r)) return -1;
        model = r.ret;
    }
    if (!heap_ptr(model)) return -1;
    eid = id;
    request = 0;
    memset(&r, 0, sizeof r);
    r.arg[0] = (uint64_t)(uintptr_t)&request;
    r.arg[1] = sfx;
    r.arg[2] = chr;
    r.arg[3] = (uint64_t)(uintptr_t)&eid;
    r.arg[4] = params + 0x30;
    if (api->call_guest(0x2378160, &r) || !heap_ptr(request)) return -1;
    uint8_t offset[32];
    if (!api->read(0x593e970, offset, sizeof offset)) memcpy((void*)(uintptr_t)(request + 0x30), offset, sizeof offset);
    dmy_bind = dummy;
    memset(&r, 0, sizeof r);
    r.arg[0] = (uint64_t)(uintptr_t)&request;
    r.arg[1] = (uint64_t)(uintptr_t)&dmy_bind;
    r.arg[2] = model;
    api->call_guest(0x2372fd0, &r);
    if (tae_end_on && ntae < MAXTAE) {
        tae_req[ntae++] = request;  /* ours, ended and dropped when the shot's event length has passed */
    } else {
        volatile uint32_t* refs = (volatile uint32_t*)(uintptr_t)(request + 8);
        if (*refs > 1) *refs -= 1;  /* ours; SprjSfx's list keeps its own */
    }
    return dummy;
}

/* the SpEffect's node on the player's list (ChrIns +0x1c8 -> +8, next +0x58, id +0x40), taken off by the game's removal */
static int speffect_off(uint64_t chr, int id) {
    const uint64_t list = r64(chr + 0x1c8);
    uint64_t node = list ? r64(list + 8) : 0;
    for (int g = 0; heap_ptr(node) && g < 1024; ++g, node = r64(node + 0x58)) {
        if ((int32_t)r32(node + 0x40) != id) continue;
        BbCallRegs r;
        memset(&r, 0, sizeof r);
        r.arg[0] = list;
        r.arg[1] = node;
        r.arg[2] = 1;
        api->call_guest(0x1be1d90, &r);
        return 1;
    }
    return 0;
}

static void request(const char* line) {
    const char* dir = getenv("BBHOST_TEST_REQUESTS");
    if (!dir || !*dir) return;
    static int n;
    char tmp[1024], req[1024];
    snprintf(tmp, sizeof tmp, "%s/fxp-%05d.tmp", dir, n);
    snprintf(req, sizeof req, "%s/fxp-%05d.req", dir, n);
    ++n;
    FILE* f = fopen(tmp, "w");
    if (!f) return;
    fputs(line, f);
    fclose(f);
    rename(tmp, req);
}

/* --- the stages ------------------------------------------------------------------------------- */
enum { WAIT, SETTLE, SHOOT, AFTER, TAIL, FINAL, DONE };
static int stage = WAIT;
static uint64_t stage_at, settled;
static double fps = 30.0;
static int spawn_retry_done, speffect_on;
static uint32_t want_block;
static double dump_s[8];
static int ndump, dumps_made;

static int loading_screen(void) {
    uint8_t requested = 0, h[2] = {0, 0};
    uint64_t helper = 0;
    if (api->read(0x596286b, &requested, 1) || !requested) return 0;
    if (api->read(0x593e8b8, &helper, 8) || !helper) return 1;
    if (api->read(helper + 0x50, h, 2)) return 1;
    return !(h[0] || h[1]);
}
static double game_s(uint64_t since) { return (double)(frames - since) / fps; }

static void start_shot(uint64_t chr) {
    Shot* s = &plan[shot_cur];
    shot_at = frames;
    spawn_retry_done = 0;
    speffect_on = 0;
    dumps_made = 0;
    ++shot_index;
    int rc = 0;
    if (s->kind == K_SFX) {
        rc = sfx_at(chr, s->dummy, s->id);  /* the dummy poly used */
    } else if (s->kind == K_BEHAVIOR) {
        rc = call_behavior(chr, -1, s->id, -1, s->pitch);
    } else if (s->kind == K_GOODS) {
        /* a goods row's bullet is its refId (+0), refCategory at +0x3b; its SpEffect the same call applies */
        size_t bytes = 0;
        const uint8_t* g = (const uint8_t*)api->param_row("EquipParamGoods", (uint32_t)s->id, &bytes);
        int32_t ref = -1;
        if (g && bytes > 0x3b) memcpy(&ref, g, 4);
        rc = call_behavior(chr, s->id, -1, g && g[0x3b] == 1 ? ref : -1, s->pitch);
    } else if (s->kind == K_SPEFFECT) {
        rc = api->sp_effect_apply(NULL, s->id);
        speffect_on = rc == 0;
    }
    float at[3] = {0, 0, 0}, yaw = 0;
    api->player_position(at, &yaw);
    api->log("fxp: f%llu shot s%d %d/%d %s %s %s %d dummy %d pitch %.1f repeat %d/%d: %d at (%.2f %.2f %.2f) yaw %.3f", (unsigned long long)frames,
             shot_index - 1, shot_cur + 1, nshots, s->family, s->name,
             s->kind == K_SFX ? "sfx" : s->kind == K_BEHAVIOR ? "behavior" : s->kind == K_GOODS ? "goods" : "speffect", s->id, s->dummy, s->pitch,
             shot_rep + 1, s->repeat, rc, at[0], at[1], at[2], yaw);
}

static void on_frame(void* user) {
    (void)user;
    /* the frame that just ended: its kind, and every instance's container calls in it */
    pthread_mutex_lock(&mu);
    const int fk = frame_kind;
    frame_kind = 0;
    transition = fk == 1 ? 0 : fk == 2 ? 1 : -1;
    if (fk == 1) ++frames_whole, ++ticks;
    else if (fk == 2) ++frames_between;
    pthread_mutex_unlock(&mu);
    ++frames;
    if (tracking) {
        pthread_mutex_lock(&mu);
        instances_frame();
        pthread_mutex_unlock(&mu);
    }

    if (stage == DONE) return;
    float at[3], yaw = 0;
    if (api->player_position(at, &yaw) != 0) return;
    const uint64_t chr = player_chr();
    if (!chr) return;

    if (stage == WAIT) {
        uint32_t block = 0;
        const int here = (!want_block || (api->player_block(&block) == 0 && block == want_block)) && !loading_screen();
        settled = here ? settled + 1 : 0;
        if (frames > (uint64_t)env_i("FXP_WAIT_FRAMES", 40000)) {
            api->log("fxp: f%llu never in block %08x with the loading screen gone - done", (unsigned long long)frames, want_block);
            stage = DONE;
            if (env_i("FXP_EXIT", 0)) request("exit\n");
            return;
        }
        if (settled < (uint64_t)env_i("FXP_SETTLE", 90)) return;
        uint8_t one = 1;
        int w = api->write(0x593e888, &one, 1);  /* Player No Dead */
        w |= api->write(0x593e890, &one, 1);     /* All No Dead */
        float X[3], Y[3], Z[3], P[3];
        const int axes = chr_axes(chr, X, Y, Z, P);
        /* the facing (-sin yaw, -cos yaw): which of +Z and -Z it is */
        const float fx = -sinf(yaw), fz = -cosf(yaw);
        if (axes && Z[0] * fx + Z[2] * fz < 0) forward_sign = -1.0f;
        api->log("fxp: f%llu ready in %08x at (%.2f %.2f %.2f) yaw %.3f; flags %s; axes X (%.2f %.2f %.2f) Y (%.2f %.2f %.2f) Z (%.2f %.2f %.2f) "
                 "place (%.2f %.2f %.2f): the player faces %sZ; %d shots, %.0f frames a game second",
                 (unsigned long long)frames, block, at[0], at[1], at[2], yaw, w ? "REFUSED" : "on", X[0], X[1], X[2], Y[0], Y[1], Y[2], Z[0], Z[1],
                 Z[2], P[0], P[1], P[2], forward_sign > 0 ? "+" : "-", nshots, fps);
        /* the place's own effects: every instance seen while it settles (their ids stay out), and the requests' */
        const uint64_t sfx = r64(api->guest_addr(0x5940370));
        const uint64_t head = sfx ? r64(sfx + 0x10) : 0;
        uint64_t n = head ? r64(head) : 0;
        for (int k = 0; n && n != head && k < 20000; ++k, n = r64(n)) {
            const uint64_t req = r64(n + 0x10);
            if (req) add_ambient(r32(req + 0x68));
        }
        calibrate_ids();
        pthread_mutex_lock(&mu);
        tracking = 1;
        pthread_mutex_unlock(&mu);
        stage = SETTLE;
        stage_at = frames;
        return;
    }
    if (stage == SETTLE) {
        if (game_s(stage_at) < 1.0) return;
        api->log("fxp: f%llu the place's own effects: %d ids left out", (unsigned long long)frames, nambient);
        stage = SHOOT;
        shot_cur = 0, shot_rep = 0;
        start_shot(chr);
        stage = AFTER;
        return;
    }

    pthread_mutex_lock(&mu);
    check_requests();
    for (int i = 0; i < ninst; ++i)
        if (!in[i].ambient && in[i].shot >= 0) walk(&in[i]);
    pthread_mutex_unlock(&mu);
    if (frames % (uint64_t)env_i("FXP_WORDS_EVERY", 3000) == 0) {
        pthread_mutex_lock(&mu);
        dump_words();
        pthread_mutex_unlock(&mu);
    }

    if (stage == AFTER) {
        Shot* s = &plan[shot_cur];
        const double t = game_s(shot_at);
        /* an effect that did not appear at its dummy poly: once more at the body's */
        if (s->kind == K_SFX && !spawn_retry_done && frames - shot_at >= (uint64_t)env_i("FXP_SPAWN_FRAMES", 6)) {
            spawn_retry_done = 1;
            if (!seen_id_since((uint32_t)s->id, shot_at) && s->dummy != 1) {
                const int used = sfx_at(chr, 1, s->id);
                api->log("fxp: f%llu shot s%d: no %d at dummy poly %d; at dummy poly 1 (%d)", (unsigned long long)frames, shot_index - 1, s->id, s->dummy,
                         used);
            }
        }
        if (ntae && t >= s->hold) tae_end_all("at its event's length");
        if (speffect_on && t >= s->arg) {
            const int off = speffect_off(chr, s->id);
            speffect_on = 0;
            api->log("fxp: f%llu shot s%d: SpEffect %d off after %.2f s (%s)", (unsigned long long)frames, shot_index - 1, s->id, t,
                     off ? "removed" : "not on the list");
        }
        if (dumps_made < ndump && t >= dump_s[dumps_made]) {
            const char* dir = getenv("FXP_DUMPS");
            if (dir && *dir) {
                char line[1200];
                snprintf(line, sizeof line, "dump %s/s%03d-%s-%s-%.2f.ppm\n", dir, shot_index - 1, s->family, s->name, dump_s[dumps_made]);
                request(line);
            }
            ++dumps_made;
        }
        const int live = shot_live(shot_index - 1) + req_live(shot_index - 1);
        const double min_s = env_f("FXP_MIN_S", 1.0), max_s = env_f("FXP_MAX_S", 15.0);
        if (t < min_s || speffect_on || ntae || (live && t < max_s)) return;
        if (live) api->log("fxp: f%llu shot s%d: %d instance(s) live at %.1f s, the cap - on to the next", (unsigned long long)frames, shot_index - 1, live, t);
        api->log("fxp: f%llu shot s%d done after %.2f s", (unsigned long long)frames, shot_index - 1, t);
        shot_end_at = frames;
        stage = TAIL;  /* the gap */
        return;
    }
    if (stage == TAIL) {
        if (game_s(shot_end_at) < env_f("FXP_GAP_S", 0.5)) return;
        if (++shot_rep >= plan[shot_cur].repeat) shot_rep = 0, ++shot_cur;
        if (shot_cur < nshots) {
            start_shot(chr);
            stage = AFTER;
            return;
        }
        shot_cur = nshots - 1;  /* what comes now is the last shot's */
        stage = FINAL;
        return;
    }
    if (stage == FINAL) {
        /* every shot made: the last requests, then done */
        int live = 0;
        for (int i = 0; i < ninst; ++i) live += !in[i].ambient && in[i].shot >= 0;
        if (live && game_s(shot_end_at) < env_f("FXP_TAIL_S", 10.0)) return;
        pthread_mutex_lock(&mu);
        for (int i = 0; i < ninst; ++i)
            if (!in[i].ambient && in[i].shot >= 0) finish(&in[i], "still");
        tracking = 0;
        dump_words();
        pthread_mutex_unlock(&mu);
        api->log("fxp: f%llu done: %d shots made, %llu frames with a whole step, %llu between, %llu instances not tracked, %llu word snapshots lost; "
                 "ids checked against requests %llu, differing %llu; instances named at their creation %llu",
                 (unsigned long long)frames, shot_index, (unsigned long long)frames_whole, (unsigned long long)frames_between,
                 (unsigned long long)lost_insts, (unsigned long long)wlost, (unsigned long long)id_checked, (unsigned long long)id_differ,
                 (unsigned long long)created_named);
        stage = DONE;
        if (env_i("FXP_EXIT", 0)) request("exit\n");
    }
}

BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {BB_PLUGIN_API_VERSION, BB_PLUGIN_OPT_IN | BB_PLUGIN_GAMEPLAY, "fx_player", "fx_player", "0",
                                                      "bbhost", "makes the player's combat effects one after another and logs them (a test)"};

BB_PLUGIN_EXPORT int bb_plugin_init(const BbHostApi* a) {
    api = a;
    return 0;
}

BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* a) {
    api = a;
    if (!a->eboot_is_109() || a->version < 8) return 1;
    if (!load_plan(getenv("FXP_PLAN"))) {
        a->log("fxp: no plan (FXP_PLAN)");
        return 1;
    }
    fps = env_f("FXP_FPS", 30.0);
    tae_end_on = env_i("FXP_TAE_END", 1) != 0;
    if (getenv("FXP_BLOCK")) want_block = (uint32_t)strtoul(getenv("FXP_BLOCK"), 0, 16);
    word_bytes = env_i("FXP_WORD_BYTES", 0x200) & ~3;
    if (word_bytes <= 0 || word_bytes > 0x200) word_bytes = 0x200;
    const char* d = getenv("FXP_DUMP_S");
    while (d && *d && ndump < 8) {
        char* end = 0;
        const double v = strtod(d, &end);
        if (end == d) break;
        dump_s[ndump++] = v;
        d = *end == ',' ? end + 1 : end;
    }
    img_rt = a->guest_addr(0x400000);
    int32_t rel = 0;
    if (!a->read(0x2377d4e, &rel, 4)) step_slot = a->guest_addr(0x2377d4e) + 4 + (int64_t)rel;
    in = calloc(MAXI, sizeof(Inst));
    snaps = calloc(SNAP, sizeof(Snap));
    wstats = calloc(WSTAT, sizeof(WStat));
    ostats = calloc(OSTAT, sizeof(OStat));
    if (!in || !snaps || !wstats || !ostats) return 1;
    /* push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; sub rsp, 0x128: the container update's own first bytes */
    static const uint8_t p_cont[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x28, 0x01, 0x00, 0x00};
    const int hk = a->hook2(0x2c4c8c0, p_cont, sizeof p_cont, h_container, 0);
    /* the instance's creation by id (sub_2c53020: the same first bytes, sub rsp 0x98) and the call that gets the new
     * instance (sub_2c505f0: push rbp; mov rbp, rsp; push r15, r14, r12, rbx; mov rbx, rsi; mov r14, rdi) */
    static const uint8_t p_create[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x98, 0x00, 0x00, 0x00};
    static const uint8_t p_created[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x89, 0xf3, 0x49, 0x89, 0xfe};
    const int hc = a->hook2(0x2c53020, p_create, sizeof p_create, h_create, 0);
    const int hd = a->hook2(0x2c505f0, p_created, sizeof p_created, h_created, 0);
    a->log("fxp: creation hooks %d %d (0 in); TAE effects %s", hc, hd, tae_end_on ? "ended at their event's length" : "left to run (FXP_TAE_END=0)");
    a->on_frame(on_frame, 0);
    a->log("fxp: %d shots (FXP_PLAN), %.0f frames a game second, block %08x; container hook %d (0 in); the step slot %s; word bytes 0x%x; %d dumps a shot",
           nshots, fps, want_block, hk, step_slot ? "found" : "NOT found", word_bytes, ndump);
    return 0;
}
