/* shader_tint: a test of plugin API 27's shaders (docs/plugins.md, "Shaders").
 * It takes one of the game's shaders' own source (shader_source) once the
 * game has met it, patches it to tint what it draws red - its strength read
 * from the plugin's shader data (bbhost.hlsli's bb_data_float) - and hands it
 * back (shader_override); then turns the tint on and off by turns, and pulses
 * the strength every frame, which builds nothing. Every state change of the
 * override is logged. An example and a test of the host, not a mod: built
 * beside the tests (CMake target shader_tint_probe, shader_tint.so / .dll),
 * never into plugins/ and never packaged, and opt-in should it be put there.
 *
 *   [plugins] shader_tint = true
 *   [shader_tint]
 *   # the shader, as BBHOST_HLSL_DUMP names its file: the scene's copy into the post chain
 *   shader = "PointSampleColorClamp1.ppo@ccbf44a6.ps"
 *   toggle_seconds = 10      # on for this long, then off as long (0: on throughout)
 *   pulse_seconds = 4        # the strength from 0 to 1 and back over this long (0: 1 throughout)
 */
#include "bbhost_plugin.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

static const BbHostApi* g_api;
static char g_shader[256] = "PointSampleColorClamp1.ppo@ccbf44a6.ps";
static double g_toggle = 10.0, g_pulse = 4.0, g_start = -1.0;
static char* g_text;   /* the patched source, once made ("" when it could not be) */
static int g_on = -1;  /* the override as last set: 1 on, 0 off */
static int g_state = -1;
static unsigned g_frames;

BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION, BB_PLUGIN_OPT_IN, "shader_tint", "Shader tint (test)", "1.0.0", "bbhost",
    "A test of the host's shader overrides (plugin API 27): one shader tinted live, on and off, its strength pulsing."};

static double seconds(void) {
#if defined(_WIN32)
    return (double)GetTickCount64() / 1000.0;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
#endif
}

/* The source with the tint after main's call of lifted_main: the pixel
 * shader's mrt0 a float4. 0 when it has no such output. */
static int patch(const char* src) {
    static const char kCall[] = "    lifted_main();\n";
    static const char kTint[] = "    mrt0.rgb = lerp(mrt0.rgb, mrt0.rgb * float3(1.0f, 0.35f, 0.35f), saturate(bb_data_float(0)));\n";
    static const char kInclude[] = "#include \"bbhost.hlsli\"\n";
    const char* at = NULL;
    const char* p;
    size_t n, head, all;
    if (!strstr(src, "static float4 mrt0;")) return 0;
    for (p = strstr(src, kCall); p; p = strstr(p + 1, kCall)) at = p;
    if (!at) return 0;
    n = strlen(src);
    head = (size_t)(at - src) + sizeof(kCall) - 1;
    all = sizeof(kInclude) - 1 + n + sizeof(kTint) - 1 + 1;
    g_text = (char*)malloc(all);
    if (!g_text) return 0;
    memcpy(g_text, kInclude, sizeof(kInclude) - 1);
    memcpy(g_text + sizeof(kInclude) - 1, src, head);
    memcpy(g_text + sizeof(kInclude) - 1 + head, kTint, sizeof(kTint) - 1);
    memcpy(g_text + sizeof(kInclude) - 1 + head + sizeof(kTint) - 1, src + head, n - head + 1);
    return 1;
}

static void on_frame(void* user) {
    const char* src = NULL;
    size_t n = 0;
    double t, phase;
    float strength;
    int on;
    (void)user;
    ++g_frames;
    if (!g_text) {
        const int r = g_api->shader_source(g_shader, &src, &n);
        if (r == 2) {
            g_api->log("shader_tint: shader overrides do not apply here (the HLSL path or the overrides off)");
            g_text = (char*)calloc(1, 1);
            return;
        }
        if (r != 0) return; /* not met yet: again next frame */
        if (!patch(src)) {
            g_api->log("shader_tint: %s has no float4 output mrt0 after a call of lifted_main: nothing to tint", g_shader);
            g_text = (char*)calloc(1, 1);
            return;
        }
        g_api->log("shader_tint: %s's own source (%zu bytes) patched, at frame %u", g_shader, n, g_frames);
        g_start = seconds();
    }
    if (!g_text[0]) return;
    t = seconds() - g_start;
    on = g_toggle <= 0.0 || ((long long)(t / g_toggle)) % 2 == 0;
    if (on != g_on) {
        g_api->shader_override("shader_tint", g_shader, on ? g_text : NULL);
        g_api->log("shader_tint: tint %s at %.1f s (frame %u)", on ? "on" : "off", t, g_frames);
        g_on = on;
    }
    phase = g_pulse > 0.0 ? t / g_pulse - (double)(long long)(t / g_pulse) : 0.5;
    strength = (float)(g_pulse > 0.0 ? (phase < 0.5 ? phase * 2.0 : 2.0 - phase * 2.0) : 1.0);
    g_api->shader_data("shader_tint", 0, &strength, sizeof(strength));
    if (g_frames % 30 == 0) {
        char why[256];
        const int st = g_api->shader_state("shader_tint", g_shader, why, sizeof(why));
        if (st != g_state) {
            static const char* const names[] = {"none", "waiting", "taken", "refused", "shadowed", "off"};
            g_api->log("shader_tint: %s %s%s%s (frame %u)", g_shader, st >= 0 && st <= 5 ? names[st] : "?", why[0] ? ": " : "", why, g_frames);
            g_state = st;
        }
    }
}

BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    const char* v;
    if (api->version < 27 || api->size < offsetof(BbHostApi, shader_data) + sizeof(api->shader_data)) {
        api->log("shader_tint: needs plugin API 27 (this host has %u)", api->version);
        return 1;
    }
    g_api = api;
    if ((v = api->config("shader_tint.shader")) && v[0]) snprintf(g_shader, sizeof(g_shader), "%s", v);
    if ((v = api->config("shader_tint.toggle_seconds")) && v[0]) g_toggle = atof(v);
    if ((v = api->config("shader_tint.pulse_seconds")) && v[0]) g_pulse = atof(v);
    api->log("shader_tint: %s, on and off every %.1f s, its strength pulsing over %.1f s", g_shader, g_toggle, g_pulse);
    return api->on_frame(on_frame, NULL);
}
