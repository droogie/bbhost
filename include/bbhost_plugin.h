/* bbhost plugin API, version 12.
 *
 * A plugin is a shared library (.so on Linux, .dll on Windows) in
 * <exe dir>/plugins or <data>/plugins. The host loads every one at start and
 * calls, in this order:
 *
 *   int bb_plugin_init(const BbHostApi* api);    before the eboot is loaded:
 *                                                log, config, register_hle
 *   int bb_plugin_image(const BbHostApi* api);   after the eboot is loaded and
 *                                                patched: read, write, patch,
 *                                                hook, guest_addr (optional)
 *
 * and reads `bb_plugin_info` (a BbPluginInfo, optional) before either. C++
 * plugins get the engine's layouts, symbols and typed params from
 * include/bbhost/ (bbhost/sdk.hpp is the entry point).
 *
 * Both return 0 to stay loaded; anything else unloads the plugin with a log
 * line. Everything is plain C with the platform's default calling
 * convention, except a function handed to register_hle: the game calls it
 * with the PS4's SysV ABI, so it must carry BB_GUEST_ABI (Clang and GCC on
 * Windows; MSVC cannot express it, so register_hle is not for MSVC-built
 * plugins). Hook callbacks are ordinary C functions; the host adapts.
 *
 * Addresses are Binary Ninja's for the 1.09 eboot (the image at its preferred
 * base 0x400000). guest_addr() turns one into the runtime address; the read,
 * write, patch and hook calls take Binary Ninja addresses themselves.
 */
#ifndef BBHOST_PLUGIN_H
#define BBHOST_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BB_PLUGIN_API_VERSION 12

#if defined(_WIN32)
#define BB_PLUGIN_EXPORT __declspec(dllexport)
#if defined(__clang__) || defined(__GNUC__)
#define BB_GUEST_ABI __attribute__((sysv_abi))
#else
#define BB_GUEST_ABI /* not expressible: do not use register_hle from MSVC */
#endif
#else
#define BB_PLUGIN_EXPORT __attribute__((visibility("default")))
#define BB_GUEST_ABI
#endif

/* The guest's argument registers as a hooked function was entered. */
typedef struct BbHookFrame {
    uint64_t rdi, rsi, rdx, rcx, r8, r9;
} BbHookFrame;

/* Returns nonzero to skip the hooked function (its caller sees 0), zero to
 * run it. */
typedef int (*BbHookFn)(const BbHookFrame* frame, void* user);

/* Version 8: a hook that sees and rewrites everything a call carries. */
typedef struct BbHookCtx {
    uint64_t arg[6];  /* rdi rsi rdx rcx r8 r9: the integer arguments, writable */
    uint64_t xmm[8];  /* the low 64 bits of xmm0-7 (a float in the low 32), writable */
    uint64_t ret;     /* what the caller gets when the hook skips the function (0 to start) */
} BbHookCtx;
/* Returns nonzero to skip the function (its caller gets ctx->ret), zero to
 * run it with the arguments as ctx leaves them. */
typedef int (*BbHookFn2)(BbHookCtx* ctx, void* user);

/* A call into the game: the integer and float argument registers, four
 * stack arguments (the 7th integer argument on), and what came back. */
typedef struct BbCallRegs {
    uint64_t arg[6];    /* rdi rsi rdx rcx r8 r9 */
    uint64_t xmm[8];    /* low 64 bits of xmm0-7; bb_f32() / bb_f64() fill one */
    uint64_t stack[4];  /* [rsp+8].. at the callee's entry */
    uint64_t ret;       /* rax */
    uint64_t ret_xmm0;  /* xmm0's low 64 bits: a float result */
} BbCallRegs;

static inline uint64_t bb_f32(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    return v.u;
}
static inline uint64_t bb_f64(double d) {
    union { double d; uint64_t u; } v;
    v.d = d;
    return v.u;
}

/* One character the world holds (chr_list). */
typedef struct BbChr {
    void* ins;           /* the game's ChrIns (bbhost/engine/sprj/chr_ins.hpp) */
    uint32_t handle;     /* its ChrHandle: what the game's own lookups take */
    int32_t npc_param;   /* NpcParam row; -1 for a player */
    int32_t chr_type;    /* the multiplayer class (0 = the local player's kind) */
    int32_t team_type;   /* combat relationship */
    int32_t hp, max_hp;
    float pos[3];        /* in its block's frame, y up */
    uint32_t model;      /* the model number: 2560 for c2560 */
    int32_t is_player;   /* 1 for the main player */
} BbChr;

/* What a plugin says about itself: export
 *   BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {...};
 * The host shows it in the log and the setup window, and honours flags. */
#define BB_PLUGIN_OPT_IN 1u       /* loaded only when the player enables it ([plugins] enable) */
#define BB_PLUGIN_GAMEPLAY 2u     /* changes the game (items, enemies, rules): flagged on the play log */
#define BB_PLUGIN_ADOPTS_RULES 4u /* can play by a host's rules when joining that host's world
                                     (on_world_rules): loaded even when off, as a visitor (visitor()), so a
                                     vanilla player can be summoned into a world this plugin made; the
                                     server lets players advertising it (X-BBHost-Adopt) meet across
                                     this plugin's rules */
typedef struct BbPluginInfo {
    uint32_t api_version;    /* BB_PLUGIN_API_VERSION it was built against */
    uint32_t flags;          /* BB_PLUGIN_* */
    const char* name;        /* "randomizer" - the [plugins] enable name and its config section */
    const char* title;       /* "Randomizer" */
    const char* version;     /* "1.0.0" */
    const char* author;
    const char* description; /* one or two sentences */
} BbPluginInfo;

/* Version 9: what a plugin's settings are, so the host can draw them - in
 * the plugin manager at launch (read without starting the plugin) and in
 * the in-game plugin menu. Export a table ended by a zero entry:
 *   BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
 *       {BB_OPT_TEXT, "seed", "Seed", "Empty: a new one", ""},
 *       {BB_OPT_BOOL, "shops", "Shuffle shops", 0, "true"},
 *       {BB_OPT_FLOAT, "speed", "Game speed", 0, "1.0", 0.25, 2.0},
 *       {BB_OPT_CHOICE, "order", "Order", 0, "story", 0, 0, "story|random"},
 *       {BB_OPT_ACTION, "start", "Start the boss rush"},
 *       {0}};
 * Values live in the per-user bbhost.toml under [<plugin name>] and come
 * back through config("<plugin name>.<key>"). An action is a button in the
 * in-game menu; pressing it calls the on_action callback. */
#define BB_OPT_END 0
#define BB_OPT_BOOL 1      /* def "true"/"false" */
#define BB_OPT_INT 2       /* min..max */
#define BB_OPT_FLOAT 3     /* min..max */
#define BB_OPT_CHOICE 4    /* one of choices, "a|b|c" */
#define BB_OPT_TEXT 5
#define BB_OPT_ACTION 6    /* a button (in game only) */
#define BB_OPT_HEADING 7   /* a section title: label (and help) only */
#define BB_OPT_LIVE 0x100u /* or-ed into type: takes effect at once in game
                              (on_option); without it, at the next start */
typedef struct BbOption {
    uint32_t type;        /* BB_OPT_*, maybe | BB_OPT_LIVE */
    const char* key;      /* the config key; an action's id */
    const char* label;
    const char* help;     /* a line under it (may be NULL) */
    const char* def;      /* the default, as text */
    double min, max;      /* INT / FLOAT range */
    const char* choices;  /* CHOICE: "a|b|c" */
} BbOption;

/* Version 11: a string of the game's and what to show in its place
 * (replace_text). */
typedef struct BbText {
    uint64_t bn_addr;      /* the game's UTF-16 string, by its Binary Ninja address */
    const uint16_t* text;  /* UTF-16, ended by a 0 */
} BbText;

typedef struct BbHostApi {
    uint32_t version; /* BB_PLUGIN_API_VERSION */
    uint32_t size;    /* sizeof(BbHostApi): a newer host may append fields */

    /* Text to the host log, printf-style, one line. */
    void (*log)(const char* fmt, ...);
    /* A value from bbhost.toml as "section.key" ("online.online_id"); "" when
     * absent. The pointer is valid until the next call. */
    const char* (*config)(const char* key);
    /* Host paths: the data directory (saves, caches) and the mods overlay
     * ("" when there is none). */
    const char* (*data_dir)(void);
    const char* (*mods_dir)(void);
    /* Replaces the HLE implementation of a PS4 import (its exported name,
     * e.g. "sceKernelUsleep") with fn, a BB_GUEST_ABI function taking the
     * import's own arguments. Only from bb_plugin_init. 0 on success. */
    int (*register_hle)(const char* name, void* fn);

    /* From bb_plugin_image on. */
    /* SHA-256 of the loaded eboot, hex. */
    const char* (*eboot_sha256)(void);
    /* 1 when the loaded eboot is the 1.09 build every address here means. */
    int (*eboot_is_109)(void);
    /* Runtime address of a Binary Ninja address, 0 when not the 1.09 build
     * or outside the image. */
    uint64_t (*guest_addr)(uint64_t bn_addr);
    /* Copies n bytes at a Binary Ninja address (any mapped guest memory when
     * the address is a runtime one above the image). 0 on success. */
    int (*read)(uint64_t bn_addr, void* out, size_t n);
    /* Writes n bytes to a data page. 0 on success. */
    int (*write)(uint64_t bn_addr, const void* data, size_t n);
    /* Writes n code bytes at a Binary Ninja address after checking that
     * `expect` is there; the page ends read-only executable. 0 on success,
     * 1 when the bytes differ, 2 when the page cannot be written. */
    int (*patch)(uint64_t bn_addr, const void* expect, const void* bytes, size_t n);
    /* Hooks a function's entry: `prologue` (n >= 14 bytes, none of them
     * rip-relative) must be the function's first bytes; fn runs on every
     * call with the argument registers before the function does. 0 on
     * success, 1 when the bytes differ, 2 when the hook cannot be placed. */
    int (*hook)(uint64_t bn_addr, const void* prologue, size_t n, BbHookFn fn, void* user);

    /* Version 2: engine services, from bb_plugin_image
     * on. A version-1 host's table ends above: check `size` (or `version`)
     * before calling them. They name the game's things, not its addresses. */
    /* fn(user) once a game frame, on the game's main thread, from the
     * frame-time manager - the first thing each frame. 0 on success. */
    int (*on_frame)(void (*fn)(void* user), void* user);
    /* A param row by table name, as the log names the tables
     * ("EquipParamWeapon"), and row id: where it lies in the game's memory,
     * and its size into *bytes; NULL when the table is not loaded yet (they
     * load during the boot) or has no such row. */
    void* (*param_row)(const char* table, uint32_t id, size_t* bytes);
    /* A field of a param row by the game's own field name
     * ("correctStrength"; "name[2]" for an array element), as text into out
     * (n bytes). 0 on success. */
    int (*param_get)(const char* table, uint32_t id, const char* field, char* out, size_t n);
    /* Writes a field of a param row from text. 0 on success. */
    int (*param_set)(const char* table, uint32_t id, const char* field, const char* value);
    /* Raises a named Lua event the way the game's scripts do, on the main
     * thread at the next frame. 0 when queued. */
    int (*lua_event)(const char* name);

    /* Version 3: the world's state and the player, by name. */
    /* An event flag - the world's state as numbered bits: bosses, doors,
     * NPC steps - read into *value (0 or 1) / written. 0 on success; 1 when
     * the flag's block does not exist (yet). Write from on_frame: the game
     * reads them on its main thread. */
    int (*event_flag_get)(uint32_t id, int* value);
    int (*event_flag_set)(uint32_t id, int value);
    /* The local player's numbers by name: "level", "echoes", "insight",
     * "vitality", "endurance", "strength", "skill", "bloodtinge", "arcane",
     * "hp", "max_hp", "stamina", "max_stamina", and the save's counters
     * "ng_cycle", "deaths", "true_deaths", "coop_helps", "invader_kills",
     * "play_time_ms" (these load with the world: at the title they read
     * 0). 0 on success; 1 for an unknown name or when no character is
     * loaded. Write from on_frame. */
    int (*player_stat_get)(const char* name, int64_t* value);
    int (*player_stat_set)(const char* name, int64_t value);

    /* Version 4: world events as they happen. */
    /* fn(id, value, user) for every event flag a setter flips - the game's
     * own (a boss dies, a door opens, an NPC moves on) or a plugin's - on the
     * main thread at the next frame, oldest first. The flags a save loads
     * with are not changes. 0 on success. */
    int (*on_event_flag)(void (*fn)(uint32_t id, int value, void* user), void* user);

    /* Version 5: where things are. */
    /* The main player's position (x, y, z; y is up) and facing (radians)
     * into xyz / *yaw (either may be NULL). 0 on success; 1 before the world
     * has a player (the title, a load). */
    int (*player_position)(float xyz[3], float* yaw);
    /* The follow camera: where it is and the point it looks at (the player
     * raised by the camera's height). 0 on success; 1 when there is none. */
    int (*camera_get)(float pos[3], float focus[3]);

    /* Version 6: the player's map block, and a test-only move. */
    /* The map block the main player is in, as the game packs it: area,
     * block, then two sub-ids, a byte each (m24_01_00_00 = 0x18010000).
     * 0 on success; 1 before the world has a player. */
    int (*player_block)(uint32_t* block);
    /* Moves the main player to xyz facing yaw (radians, as player_position
     * reads it) through the game's own warp, at the start of the next frame:
     * inside the block it is in (block 0), or into another block given as
     * player_block packs it, which must be resident. Off unless the host
     * runs with BBHOST_TEST_WARP=1 (tools/map_validation.sh) or the player
     * turned on a plugin flagged BB_PLUGIN_OPT_IN | BB_PLUGIN_GAMEPLAY (a
     * boss rush). 0 when queued; 1 when off or there is no player. */
    int (*player_warp)(uint32_t block, const float xyz[3], float yaw);

    /* Version 7: lamp travel. */
    /* Travels to a lamp the way the Hunter's Dream headstones do - the
     * game's WarpNextStage_Bonfire, a full map load - by its
     * ReturnPointParam row id (the lamp's entity id + 1000: 2412951 is
     * Central Yharnam's lamp). Gated as player_warp is;
     * at the start of the next frame. 0 when queued. The host's
     * BBHOST_TEST_UNLOCK_LAMPS=1 lights every lamp at each load. */
    int (*lamp_warp)(int32_t return_point);

    /* Version 8: the engine SDK (include/bbhost/), and hooks and calls that
     * carry everything. */
    /* The Binary Ninja address of a named engine symbol - any name in
     * bbhost/engine/symbols.hpp's ALL_SYMBOLS ("WORLD_CHR_MAN_SINGLETON_PTR")
     * - 0 when unknown or not the 1.09 build. */
    uint64_t (*symbol)(const char* name);
    /* Calls a game function at a Binary Ninja address with regs, under the
     * guest's thread state; regs->ret / ret_xmm0 hold the result. From the
     * game's main thread (on_frame, a hook) only. 0 on success. */
    int (*call_guest)(uint64_t bn_addr, BbCallRegs* regs);
    /* hook() with a BbHookCtx: float arguments, rewritable arguments and a
     * chosen return value. `prologue` may be NULL to take the bytes that are
     * there (n of them, still whole instructions, none rip-relative). */
    int (*hook2)(uint64_t bn_addr, const void* prologue, size_t n, BbHookFn2 fn, void* user);
    /* A param table's row ids in order: up to max into ids, the table's row
     * count into *count (ids may be NULL to ask for the count). 0 on success;
     * 1 when the table is not loaded. */
    int (*param_ids)(const char* table, uint32_t* ids, size_t max, size_t* count);
    /* fn(block, user) on the main thread each time the player is placed in a
     * map - a load, a warp, a death, a lamp - with the block player_block
     * packs, once the world holds them: for a load, while its screen is
     * still up (loading_screen, version 12). */
    int (*on_world_load)(void (*fn)(uint32_t block, void* user), void* user);
    /* The characters WorldChrMan holds: up to max into out, the total into
     * *count. From the main thread; the pointers are good until the next
     * frame. 0 on success; 1 before the world exists. */
    int (*chr_list)(BbChr* out, size_t max, size_t* count);
    /* Applies a SpEffectParam row to a character (BbChr.ins; NULL for the
     * main player) the way the game's scripts do. Main thread. 0 on success. */
    int (*sp_effect_apply)(void* chr, int32_t sp_effect);
    /* Whether a SpEffectParam row is on a character now. */
    int (*sp_effect_has)(void* chr, int32_t sp_effect);
    /* The calling plugin's own folder, for its saves and generated files:
     * <data>/plugins/<name>/ (made on first call). `name` is the plugin's
     * BbPluginInfo name. */
    const char* (*plugin_dir)(const char* name);
    /* A line of text on the screen for `seconds`, over the game (a banner:
     * "Round 2 - Father Gascoigne"). Any thread. 0 when shown. */
    int (*show_message)(const char* text, float seconds);
    /* Game time against real time: 0.5 is slow motion at half speed, 2.0
     * double speed as far as the machine keeps up (clamped to 0.1-4; 1 is
     * normal). At 30 fps the game advances a fixed step each frame, so this
     * paces the frames; at the other frame rates it scales the step (at most
     * 1/30 s a frame) and the frames keep their pace. Any thread. 0 on
     * success. */
    int (*set_time_scale)(float scale);

    /* Version 9: settings and actions from the plugin menus, and files. */
    /* fn(key, value, user) when the player changes one of the calling
     * plugin's options in the in-game menu (key without the section; value
     * as text). `name` is the plugin's BbPluginInfo name. Main thread. */
    int (*on_option)(const char* name, void (*fn)(const char* key, const char* value, void* user), void* user);
    /* fn(action, user) when the player presses one of the plugin's
     * BB_OPT_ACTION buttons in the in-game menu. Main thread. */
    int (*on_action)(const char* name, void (*fn)(const char* action, void* user), void* user);
    /* A game file by its /app0 path ("/dvdroot_ps4/map/mapstudio/m24_01_00_00.msb.dcx"),
     * as the game would read it without the plugins' overlays (the player's
     * mods first, never a plugin's own output), DCX-decompressed,
     * into a buffer the host owns until the next call: *data and *size. 0 on
     * success. From bb_plugin_image (before the game reads it) on. */
    int (*game_file)(const char* path, const void** data, size_t* size);
    /* Writes a file into the calling plugin's overlay - which the game reads
     * in place of its own file of that path, behind the player's mods and
     * ahead of the dump - DCX-compressed again when the path ends in .dcx.
     * Write before the game opens the file (bb_plugin_image is early
     * enough for maps). An overlay is cleared at each start: write what
     * this run needs. 0 on success. */
    int (*overlay_file)(const char* name, const char* path, const void* data, size_t size);

    /* Version 10: rules that change while the game runs, and other players'. */
    /* The plugin's entry in the session's rules, which every request
     * carries (X-BBHost-Ruleset): "randomizer;seed=<token>" (bb::seed_token).
     * Set it again when the plugin's seed changes. Any thread. */
    int (*set_rules)(const char* name, const char* rules);
    /* fn(rules, user) when the player joins another's world as a guest whose
     * host plays by other rules - with the host's entry for this plugin
     * ("randomizer;seed=<token>"), or "" when the host plays without it (a
     * vanilla world) - and fn(NULL, user) when they go back to their own. Called before that world loads, on the network thread: a
     * plugin with BB_PLUGIN_ADOPTS_RULES rewrites what the next load reads
     * (overlay_file) and returns; it may not touch game objects here. */
    int (*on_world_rules)(const char* name, void (*fn)(const char* rules, void* user), void* user);
    /* 1 when the plugin is loaded only as a visitor: the player has it off,
     * but it has BB_PLUGIN_ADOPTS_RULES, so it is there to play other
     * players' worlds (on_world_rules) and must change nothing of the
     * player's own - no params, no layouts, no set_rules. */
    int (*visitor)(const char* name);

    /* Version 11: the game's own text. */
    /* Every reference to each string - the lea instructions that load its
     * address, the pointers to it in the image's data - names a copy of the
     * new text instead (the host keeps it); the string itself stays as it
     * was. For the 1.09 image's read-only UTF-16 strings, by Binary Ninja
     * address; from bb_plugin_image only, before the game runs. The number of
     * references moved, or -1 outside the image phase or for another eboot.
     * A string the code reaches some other way keeps its text. */
    int (*replace_text)(const BbText* texts, size_t count);

    /* Version 12: the loading screen. */
    /* 1 while a loading screen is up - the game's NowLoading request with no
     * in-game frame run since, the host's own test, which its log's
     * "loading:" lines come from - and 0 in the world, at the title and in
     * its menus. The host reads it once a frame, after the on_frame
     * callbacks: they see the frame before's. A plugin that moves the player
     * waits for 0: on_world_load comes as soon as a load has built the
     * player, with its screen still up, and a warp asked then lands before
     * the world is ready, a lamp travel is refused once the load replaces
     * the player. Any thread. */
    int (*loading_screen)(void);
} BbHostApi;

#ifdef __cplusplus
}
#endif

#endif
