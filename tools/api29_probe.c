/* Plugin API 29 probe: world_pause, world_step, pad, pad_take, world_state and
 * post_look in a running game, for a test harness. Test-only: built beside the
 * tests, never into plugins/.
 *
 * From the first frame in the world it logs, once a second and whenever one
 * changes,
 *     api29 t <s> pos <x> <y> <z> state <flags> session <n> others <n> arriving <n>
 * and every controller press it reads (pad), as it reads it,
 *     api29 t <s> pad pressed <hex> held <hex> l <x> <y> r <x> <y> l2 <v> r2 <v>
 * and it carries out BBHOST_API29_PROBE, a list of "<s>:<step>" separated by
 * commas, <s> seconds after that first frame:
 *     pause / run         world_pause(name, 1 / 0)
 *     step <n>            world_step(name, n)
 *     take <hex> <axes>   pad_take(buttons, axes); "take 0 0" gives them back
 *     look <group> <numbers...>   post_look with that one group (the rest the
 *                         area's): tone <exposure gamma saturation brightness
 *                         contrast>, vignette <power>, dof <on focus aperture
 *                         near far>, ca <on lateral0 lateral1 uniform0
 *                         uniform1>, blur <on time length scale samples>, fog
 *                         <distance density contrast distance_offset
 *                         height_offset>; "look off" gives the area's own
 * For example "8:pause,14:step 2,16:step 2,20:run,24:take 4000 1,32:take 0 0".
 * Each step's result is logged: api29 t <s> step <text> -> <return>. */
#include "bbhost_plugin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const BbHostApi* g_api;
static double g_world_at = -1.0;
static double g_last_log = -10.0;
static BbWorldState g_last_state;
static float g_last_pos[3];

BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION, 0, "api29_probe", "API 29 probe", "1", "bbhost",
    "Test-only: plugin API 29's world_pause, world_step, pad, pad_take, world_state and post_look in a running game."};

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

/* The steps, parsed once. */
typedef struct {
    double at;
    char text[64];
    int done;
} Step;
static Step g_steps[64];
static int g_nsteps;

static void parse_steps(void) {
    const char* e = getenv("BBHOST_API29_PROBE");
    while (e && *e && g_nsteps < 64) {
        char* end = NULL;
        const double at = strtod(e, &end);
        if (!end || *end != ':') break;
        const char* text = end + 1;
        const char* comma = strchr(text, ',');
        const size_t n = comma ? (size_t)(comma - text) : strlen(text);
        Step* s = &g_steps[g_nsteps++];
        s->at = at;
        snprintf(s->text, sizeof(s->text), "%.*s", (int)n, text);
        e = comma ? comma + 1 : NULL;
    }
}

static int run_look(const char* spec) {
    BbLook l;
    float v[5] = {0, 0, 0, 0, 0};
    char group[16] = "";
    uint32_t avail = 0;
    int rc;
    memset(&l, 0, sizeof(l));
    l.size = sizeof(l);
    if (sscanf(spec, "%15s %f %f %f %f %f", group, &v[0], &v[1], &v[2], &v[3], &v[4]) < 1) return -1;
    if (strcmp(group, "off") == 0) {
        rc = g_api->post_look("api29_probe", NULL, &avail);
        g_api->log("api29 look off -> %d (available %x)", rc, avail);
        return rc;
    }
    if (strcmp(group, "tone") == 0) {
        l.set = BB_LOOK_TONE;
        l.exposure = v[0], l.gamma = v[1], l.saturation = v[2], l.brightness = v[3], l.contrast = v[4];
    } else if (strcmp(group, "vignette") == 0) {
        l.set = BB_LOOK_VIGNETTE;
        l.vignette = v[0];
    } else if (strcmp(group, "dof") == 0) {
        l.set = BB_LOOK_DOF;
        l.dof = (int32_t)v[0], l.dof_focus = v[1], l.dof_aperture = v[2], l.dof_near_levels = (int32_t)v[3],
        l.dof_far_levels = (int32_t)v[4];
    } else if (strcmp(group, "ca") == 0) {
        l.set = BB_LOOK_ABERRATION;
        l.aberration = (int32_t)v[0], l.lateral[0] = v[1], l.lateral[1] = v[2], l.uniform[0] = v[3], l.uniform[1] = v[4];
    } else if (strcmp(group, "blur") == 0) {
        l.set = BB_LOOK_MOTION_BLUR;
        l.motion_blur = (int32_t)v[0], l.blur_time = v[1], l.blur_length = v[2], l.blur_scale = v[3],
        l.blur_samples = (int32_t)v[4];
    } else if (strcmp(group, "fog") == 0) {
        l.set = BB_LOOK_FOG;
        l.fog_distance = v[0], l.fog_density = v[1], l.fog_contrast = v[2], l.fog_distance_offset = v[3],
        l.fog_height_offset = v[4];
    } else {
        return -1;
    }
    rc = g_api->post_look("api29_probe", &l, &avail);
    g_api->log("api29 look %s -> %d (available %x)", spec, rc, avail);
    return rc;
}

static int run_step(const char* text) {
    unsigned a = 0, b = 0;
    if (strcmp(text, "pause") == 0) return g_api->world_pause("api29_probe", 1);
    if (strcmp(text, "run") == 0) return g_api->world_pause("api29_probe", 0);
    if (sscanf(text, "step %u", &a) == 1) return g_api->world_step("api29_probe", a);
    if (sscanf(text, "take %x %u", &a, &b) == 2) return g_api->pad_take(a, b);
    if (strncmp(text, "look ", 5) == 0) return run_look(text + 5);
    return -1;
}

static void on_frame(void* user) {
    (void)user;
    BbWorldState w;
    memset(&w, 0, sizeof(w));
    w.size = sizeof(w);
    if (g_api->world_state(&w) != 0) return;
    const double now = now_s();
    if (g_world_at < 0.0) {
        if (!(w.flags & BB_WORLD_IN) || (w.flags & BB_WORLD_LOADING)) return;
        g_world_at = now;
        g_api->log("api29 world: the first frame in the world; %d step(s) to run", g_nsteps);
    }
    const double t = now - g_world_at;
    for (int i = 0; i < g_nsteps; ++i) {
        Step* s = &g_steps[i];
        if (s->done || t < s->at) continue;
        s->done = 1;
        g_api->log("api29 t %.1f step %s -> %d", t, s->text, run_step(s->text));
    }
    BbPad pad;
    memset(&pad, 0, sizeof(pad));
    pad.size = sizeof(pad);
    if (g_api->pad(&pad) == 0 && pad.pressed)
        g_api->log("api29 t %.1f pad pressed %05x held %05x l %.2f %.2f r %.2f %.2f l2 %.2f r2 %.2f", t, pad.pressed, pad.buttons,
                   (double)pad.lx, (double)pad.ly, (double)pad.rx, (double)pad.ry, (double)pad.l2, (double)pad.r2);
    float pos[3] = {0, 0, 0};
    g_api->player_position(pos, NULL);
    const int changed = w.flags != g_last_state.flags || w.session != g_last_state.session || w.others != g_last_state.others ||
                        w.arriving != g_last_state.arriving;
    if (changed || t - g_last_log >= 1.0) {
        g_api->log("api29 t %.1f pos %.3f %.3f %.3f state %x session %d others %u arriving %u pad %05x l %.2f %.2f", t, (double)pos[0],
                   (double)pos[1], (double)pos[2], w.flags, w.session, w.others, w.arriving, pad.buttons, (double)pad.lx, (double)pad.ly);
        g_last_log = t;
        g_last_state = w;
        memcpy(g_last_pos, pos, sizeof(pos));
    }
}

BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    if (api->version < 29 || api->size < offsetof(BbHostApi, post_look) + sizeof(void*)) {
        api->log("api29 probe: this host is older than plugin API 29");
        return 1;
    }
    g_api = api;
    parse_steps();
    api->on_frame(on_frame, NULL);
    api->log("api29 probe: ready, %d step(s)", g_nsteps);
    return 0;
}
