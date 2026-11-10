/* The example plugin: what a plugin can do, and the test that the
 * API works on both platforms. It logs the config's online id at init, and
 * after the image is loaded reads the ELF header, counts calls into one of
 * the game's allocators through a prologue hook, and says so at 1, 1000 and
 * every 100,000 calls. With a version-2 host it also counts frames through
 * on_frame and, once the params have loaded, reads a field of one by the
 * game's own name; with version 3, once a character is loaded, its level and
 * echoes and an event flag; with version 4, the first flags the world
 * changes; with version 5, where the player and the camera are. Built beside
 * bbhost as build/hello.so (hello.dll), never into build/plugins/: copy it
 * into a plugins/ folder to load it.
 *
 * A host only ever appends to its API: check that `version` is at least the
 * one whose calls you make, not that it is equal. */
#include "bbhost_plugin.h"

#include <stdint.h>

static const BbHostApi* api;
static uint64_t calls, frames;
static int read_param, read_player, read_position, flags_seen;

static int on_alloc(const BbHookFrame* frame, void* user) {
    (void)user;
    ++calls;
    if (calls == 1 || calls == 1000 || calls % 100000 == 0) {
        api->log("hello: %llu allocator calls (last size %llu)", (unsigned long long)calls,
                 (unsigned long long)frame->rsi);
    }
    return 0; /* run the function */
}

/* Version 2: once a frame, on the game's main thread. */
static void on_frame(void* user) {
    char fov[32];
    (void)user;
    ++frames;
    if (!read_param && api->param_get("LockCamParam", 0, "camFovY", fov, sizeof(fov)) == 0) {
        read_param = 1;
        api->log("hello: frame %llu, LockCamParam row 0 camFovY = %s (by name)", (unsigned long long)frames, fov);
    }
    if (api->version >= 3 && !read_player) {
        int64_t level = 0, echoes = 0, deaths = -1, played = -1, cycle = -1;
        int online = 0;
        if (api->player_stat_get("level", &level) == 0 && level > 0 && api->player_stat_get("echoes", &echoes) == 0) {
            read_player = 1;
            const int flag = api->event_flag_get(2000, &online);
            api->player_stat_get("deaths", &deaths);
            api->player_stat_get("play_time_ms", &played);
            api->player_stat_get("ng_cycle", &cycle);
            api->log("hello: frame %llu, level %lld, %lld echoes, %lld deaths, NG+%lld, played %lld s; event flag 2000 %s %d",
                     (unsigned long long)frames, (long long)level, (long long)echoes, (long long)deaths, (long long)cycle,
                     (long long)(played / 1000), flag == 0 ? "=" : "not there,", online);
        }
    }
    if (api->version >= 3 && frames == 3000) {
        int64_t deaths = -1, played = -1, cycle = -1;
        api->player_stat_get("deaths", &deaths);
        api->player_stat_get("play_time_ms", &played);
        api->player_stat_get("ng_cycle", &cycle);
        api->log("hello: frame %llu, in the world: %lld deaths, NG+%lld, played %lld s", (unsigned long long)frames,
                 (long long)deaths, (long long)cycle, (long long)(played / 1000));
    }
    if (api->version >= 5 && !read_position && frames % 600 == 0) {
        float at[3], yaw = 0, cam[3], focus[3];
        if (api->player_position(at, &yaw) == 0 && api->camera_get(cam, focus) == 0) {
            read_position = 1;
            api->log("hello: frame %llu, the player at %.2f %.2f %.2f facing %.2f rad; the camera at %.2f %.2f %.2f",
                     (unsigned long long)frames, at[0], at[1], at[2], yaw, cam[0], cam[1], cam[2]);
        }
    }
}

/* Version 4: a flag the game (or a plugin) flipped, on the main thread. */
static void on_flag(uint32_t id, int value, void* user) {
    (void)user;
    if (++flags_seen <= 5) api->log("hello: event flag %u -> %d (frame %llu)", id, value, (unsigned long long)frames);
}

BB_PLUGIN_EXPORT int bb_plugin_init(const BbHostApi* a) {
    api = a;
    if (a->version < 1) return 1;
    a->log("hello: init, online id '%s', data in %s", a->config("online.online_id"), a->data_dir());
    return 0;
}

BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* a) {
    unsigned char head[4] = {0};
    a->log("hello: eboot %.12s... (1.09: %d)", a->eboot_sha256(), a->eboot_is_109());
    if (a->read(0x400000, head, sizeof(head)) == 0) {
        a->log("hello: image starts %02x %02x %02x %02x", head[0], head[1], head[2], head[3]);
    }
    if (!a->eboot_is_109()) return 0;
    /* sub_2486810, an allocator of the game's DL heap: push rbp; mov rbp, rsp;
     * push r15..r12, rbx; push rax (14 bytes, none rip-relative). */
    static const unsigned char prologue[14] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41,
                                               0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50};
    const int r = a->hook(0x2486810, prologue, sizeof(prologue), on_alloc, 0);
    a->log("hello: allocator hook %s", r == 0 ? "installed" : r == 1 ? "refused (bytes differ)" : "not placed");
    if (a->version >= 2 && a->on_frame(on_frame, 0) == 0) a->log("hello: on_frame registered");
    if (a->version >= 4 && a->on_event_flag(on_flag, 0) == 0) a->log("hello: on_event_flag registered");
    return 0;
}
