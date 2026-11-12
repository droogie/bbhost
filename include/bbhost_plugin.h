/* bbhost plugin API, version 29.
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
 * write, patch and hook calls take Binary Ninja addresses themselves. read()
 * also takes the runtime address of any guest memory outside the image - the
 * pointers a plugin reads out of the game's heaps. Under Windows the image
 * may be slid (to 0x300000000), and the heaps lie below it as well as above.
 */
#ifndef BBHOST_PLUGIN_H
#define BBHOST_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BB_PLUGIN_API_VERSION 29

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
#define BB_PLUGIN_LIVE_SWITCH 8u  /* version 21: turned on and off while the game runs. With BB_PLUGIN_OPT_IN it
                                     is loaded even when off - bb_plugin_init and bb_plugin_image run, and
                                     config("plugins.<name>") says whether it is on ("true") - and asks for
                                     nothing until it is; the player's switch then comes to on_option as
                                     the key "" ("true" or "false"), on the main thread, and the plugin
                                     applies or takes back what it does. Not for BB_PLUGIN_GAMEPLAY
                                     plugins (their rules are the session's): the host ignores it there. */
#define BB_PLUGIN_AA_LEAVE 16u    /* version 27: starts with aa_claim's BB_AA_LEAVE ... */
#define BB_PLUGIN_AA_TAKE 32u     /* ... or BB_AA_TAKE (both: TAKE) - the setup window reads it before the plugin runs */
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
#define BB_OPT_WHEN 0x200u /* version 21, or-ed into the type of any option but a
                              CHOICE: its `choices` is "key=value" ("key=a|b" for any
                              of them), and the menus grey the row out unless the
                              plugin's option `key` holds that value (older hosts
                              show it as always). On a HEADING it covers the rows
                              under it, up to the next heading. */
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

/* Version 13: the keyboard and mouse as a plugin with controls of its own
 * reads them (input). */
typedef struct BbInput {
    uint8_t key[512];  /* by SDL scancode, which is the USB HID usage (4 = A, 26 = W,
                          44 = Space, 224 = Left Ctrl): 1 while held */
    uint32_t buttons;  /* the mouse's buttons held: bit 0 left, 1 right, 2 middle, 3 back, 4 forward */
    uint32_t pressed;  /* the buttons that went down since the last read - latched, so a
                          click shorter than a frame is still seen */
    float dx, dy;      /* the mouse's motion since the last read while playing, in its own
                          counts (right and down positive) */
    float wheel;       /* the wheel's notches since the last read, up positive */
    int32_t playing;   /* 1 while the mouse turns the camera: in the world with no menu or
                          text box up, the window focused */
} BbInput;

/* Version 13: the player's camera (on_camera) - its basis and where it is, in
 * the world (y up): x to the right, y up and z the way it looks, each w 0; the
 * position's w 1. */
typedef struct BbCamera {
    float right[4], up[4], forward[4], position[4];
    float fov_y;           /* vertical field of view, radians */
    float aspect;          /* width over height */
    float near_z, far_z;   /* the clip planes' distances */
} BbCamera;

/* Version 23: view_cutaway2's numbers. */
typedef struct BbCutaway {
    float distance;      /* metres from the camera to the point it looks at */
    float pitch;         /* radians the view looks down below the horizon */
    float radius;        /* metres about the view's axis at that point (0: off) */
    float widen;         /* metres wider for each metre nearer the camera */
    float height;        /* metres above the point looked at where the cut starts */
    float margin;        /* metres in front of the point looked at where the cut starts */
    float feather;       /* metres over which it fades in */
} BbCutaway;

/* Version 27: the frame as bbhost draws it, for effects of a plugin's own
 * recorded into it (frame_info, on_render), and who does the anti-aliasing
 * (aa_claim).
 *
 * The conventions. A matrix is 16 floats in column-major order (m[column * 4
 * + row]: GLSL's mat4 and GLM's layout, HLSL's column_major) applied to
 * column vectors: clip = proj * view * (x, y, z, 1). The world is the game's,
 * in metres, y up (player_position's, BbCamera's). The view: x right, y up, z
 * the way the camera looks, in metres. Clip space is Vulkan's over the 3D's
 * images: divided by w (the view's z), x runs from -1 at their left edge to +1
 * at their right, y from -1 at their top to +1 at their bottom, and z is the
 * value the G-buffer's depth holds - 0 at the near plane and 1 at the far one
 * as the game draws (the other way round with BB_FRAME_DEPTH_REVERSED). A
 * pixel (px, py) of a w x h image is centred at ((px + 0.5) / w * 2 - 1,
 * (py + 0.5) / h * 2 - 1); the picture at the output size covers the same
 * view. Angles are radians, sizes pixels, times seconds. */

/* A camera as the frame drew it - an on_view plugin's change included -
 * without the jitter. */
typedef struct BbFrameCamera {
    BbCamera camera;          /* its basis (right, up, forward) and eye in the world - the four in a row are its
                                 view-to-world matrix, column by column - fov_y, aspect, and near_z and far_z: the
                                 view distances the depth's 0 and 1 stand for (1e6 at most: an infinite far plane) */
    float view[16];           /* world to view */
    float proj[16];           /* view to clip. The frame's camera draws were moved by the jitter: their projection is
                                 this one with m[8] = 2 * jitter_x / render_width and m[9] = 2 * jitter_y / render_height */
    float view_proj[16];      /* proj * view: world to clip */
    float inv_view_proj[16];  /* clip to world: a pixel's (x, y, depth, 1) to a world point times w */
} BbFrameCamera;

#define BB_FRAME_CAMERA 1u          /* `now` is this frame's: its camera draws were seen (else the last known; reset) */
#define BB_FRAME_PREV 2u            /* `prev` is the last frame's (else this frame's again) */
#define BB_FRAME_CUT 4u             /* the camera jumped since the last frame - moved more than 4 m, turned more than 75
                                       degrees or its field of view by more than 10: a cut (reset) */
#define BB_FRAME_DEPTH_REVERSED 8u  /* the depth holds 1 at the near plane and 0 at the far one */
#define BB_FRAME_AA_PLUGIN 16u      /* a plugin does the anti-aliasing (aa_claim's BB_AA_TAKE): bbhost's is off */

#define BB_UPSCALE_NONE 0u  /* BbFrameInfo.method: the 3D is the picture as drawn - Upscaling Off, an upscaler this
                               machine cannot run, or a plugin's anti-aliasing */
#define BB_UPSCALE_DLSS 1u
#define BB_UPSCALE_FSR 2u   /* FSR 3 or FSR 4 */
#define BB_UPSCALE_TAA 3u   /* bbhost's own */

#define BB_QUALITY_NATIVE 0u      /* the 3D at the picture's size: DLAA, FSR's native AA, TAA */
#define BB_QUALITY_QUALITY 1u     /* two thirds each way - also a Resolution below the window's, whatever its share */
#define BB_QUALITY_BALANCED 2u    /* about 59% */
#define BB_QUALITY_PERFORMANCE 3u /* half */

/* What a 3D frame is (frame_info, BbRenderCall.frame). Size first: a newer
 * host appends fields after `prev`. */
typedef struct BbFrameInfo {
    uint32_t size;            /* the bytes filled */
    uint32_t flags;           /* BB_FRAME_* */
    uint64_t frame;           /* 1 for the first 3D frame described, one more for each after it (a frame without the
                                 3D - a loading screen - is none) */
    uint64_t flip;            /* the host's flips so far as it was recorded: its log's flip numbers */
    uint32_t render_width, render_height;  /* the 3D's: the scene's, the depth's, the motion's and the glow's size */
    uint32_t output_width, output_height;  /* the picture's: the upscaler's output, the 3D's size without one */
    uint32_t method;          /* BB_UPSCALE_*: what takes the 3D to the picture this frame */
    uint32_t quality;         /* BB_QUALITY_*, as the upscaler is told it (the sizes are exact) */
    float jitter_x, jitter_y; /* render pixels this frame's camera draws were moved by, +x right, +y down: what the
                                 upscaler is told; 0 when none runs */
    float frame_seconds;      /* real time since the last frame described */
    uint32_t reset;           /* 1: nothing to blend with - the first frame, a new render or output size or method,
                                 no camera this frame or the last, a cut, or more than a quarter of a second since the
                                 last frame */
    BbFrameCamera now;        /* this frame's camera */
    BbFrameCamera prev;       /* the last frame's */
} BbFrameInfo;

/* on_render's points, and what a hook may ask for. */
#define BB_RENDER_BEFORE_UPSCALE 1u /* the 3D drawn - lit, its transparency on it - before the upscaler, TAA, the
                                       embers, the sharpening and frame generation read it; with none running too */
#define BB_RENDER_AFTER_UPSCALE 2u  /* the final picture, after all of them, before the post-processing copies it */
#define BB_RENDER_MOTION 0x100u     /* or-ed in: the frame's motion made for the hooks (BbRenderCall.motion) whether
                                       or not an upscaler or frame generation runs */

#define BB_IMAGE_WRITE 1u  /* BbRenderImage.flags: the hook may write it */

/* An image of the frame as Vulkan has it (cast to the Vulkan types). */
typedef struct BbRenderImage {
    uint64_t image;       /* VkImage; 0: not in this call */
    uint64_t view;        /* VkImageView: 2D, its one level and layer, its format (a depth image's depth aspect) */
    uint64_t generation;  /* a new number whenever the host makes the image or its view again (a Resolution change):
                             while it stays the same, the handles are the same image and view in every call */
    uint32_t format;      /* VkFormat */
    uint32_t layout;      /* VkImageLayout it is in - and must be in again when the hook returns */
    uint32_t usage;       /* VkImageUsageFlags it was made with: what the hook can do with it */
    uint32_t width, height;
    uint32_t flags;       /* BB_IMAGE_* */
} BbRenderImage;

typedef void (*BbVkVoidFunction)(void);

/* The Vulkan device bbhost draws with: one for the run. */
typedef struct BbRenderDevice {
    uint32_t size;              /* sizeof(BbRenderDevice) as the host fills it */
    uint32_t api_version;       /* the Vulkan version it is used at (1.3), with synchronization2, buffer device
                                   address and storage images read and written without a format */
    void* instance;             /* VkInstance */
    void* physical_device;      /* VkPhysicalDevice */
    void* device;               /* VkDevice */
    BbVkVoidFunction (*get_instance_proc_addr)(void* instance, const char* name); /* vkGetInstanceProcAddr */
    BbVkVoidFunction (*get_device_proc_addr)(void* device, const char* name);     /* vkGetDeviceProcAddr */
    uint32_t queue_family;      /* the family of the queue the frames go to: graphics, compute and transfer */
    uint32_t frames_in_flight;  /* what a hook records may still be read by the GPU this many frames later: a
                                   resource its commands used is destroyed (or written again) no sooner */
} BbRenderDevice;

/* One call of a render hook. Size first: a newer host appends fields after
 * `picture`. */
typedef struct BbRenderCall {
    uint32_t size;                 /* sizeof(BbRenderCall) as the host fills it */
    uint32_t point;                /* BB_RENDER_BEFORE_UPSCALE or BB_RENDER_AFTER_UPSCALE */
    void* command_buffer;          /* VkCommandBuffer: the frame's, recording, outside any render pass (this call only) */
    const BbFrameInfo* frame;      /* this frame's (this call only) */
    const BbRenderDevice* device;  /* the run's (valid for the whole run) */
    BbRenderImage scene;           /* the lit 3D at the render size in linear light, before the post-processing:
                                      RGBA16F, or B10G11R11 with the Performance patch. Written before the upscale
                                      (BB_IMAGE_WRITE), read after it. A render target: sampled and transfer, never
                                      storage - a hook copies or blits its result in */
    BbRenderImage depth;           /* the G-buffer's depth at the render size, jittered as the scene (read) */
    BbRenderImage motion;          /* with BB_RENDER_MOTION: R16G16_SFLOAT at the render size, the render pixels from
                                      each pixel to where its surface was in the last frame, x right, y down, without
                                      the jitter (read) */
    BbRenderImage glow;            /* the scene's second target - the effects' glow, which the bloom reads - at the
                                      render size, when the frame has one (read) */
    BbRenderImage picture;         /* after the upscale: the final picture at the output size - what the
                                      post-processing reads (written); the scene itself when no upscaler runs; 0
                                      before the upscale */
} BbRenderCall;
typedef void (*BbRenderFn)(const BbRenderCall* call, void* user);

/* aa_claim's claims. */
#define BB_AA_NONE 0   /* no claim (the default) */
#define BB_AA_LEAVE 1  /* the plugin does none and moves no draws: bbhost's runs as the settings say - also beside a
                          Vulkan layer that would otherwise keep its TAA off */
#define BB_AA_TAKE 2   /* the plugin does it: while it is on, bbhost moves no draws (no jitter) and runs no TAA, FSR or
                          DLSS nor the game's own anti-aliasing pass, and draws the 3D at the Resolution, as with
                          Upscaling Off; the menus name the plugin */

/* Version 29: the controller as a plugin with controls of its own reads it
 * (pad) - its own buttons, sticks and triggers, whatever the player's bindings
 * make of them in the game - and what it takes from the game (pad_take). */
#define BB_PAD_L3 0x0002u
#define BB_PAD_R3 0x0004u
#define BB_PAD_OPTIONS 0x0008u
#define BB_PAD_UP 0x0010u
#define BB_PAD_RIGHT 0x0020u
#define BB_PAD_DOWN 0x0040u
#define BB_PAD_LEFT 0x0080u
#define BB_PAD_L2 0x0100u        /* the trigger past a quarter of its travel */
#define BB_PAD_R2 0x0200u
#define BB_PAD_L1 0x0400u
#define BB_PAD_R1 0x0800u
#define BB_PAD_TRIANGLE 0x1000u
#define BB_PAD_CIRCLE 0x2000u
#define BB_PAD_CROSS 0x4000u
#define BB_PAD_SQUARE 0x8000u
#define BB_PAD_TOUCHPAD 0x100000u /* the touchpad's press, or a controller's Back (Select, Share) */
#define BB_PAD_LEFT_STICK 1u     /* pad_take's axes */
#define BB_PAD_RIGHT_STICK 2u
#define BB_PAD_TRIGGERS 4u       /* L2's and R2's travel (their presses are BB_PAD_L2 and BB_PAD_R2) */

typedef struct BbPad {
    uint32_t size;        /* set to sizeof(BbPad) before the call: the host fills as much as both know */
    uint32_t buttons;     /* BB_PAD_* held */
    uint32_t pressed;     /* BB_PAD_* that went down since the last read - latched, so a press shorter than a
                             frame is still seen */
    int32_t connected;    /* 1 while a controller is there */
    float lx, ly, rx, ry; /* the sticks, -1..1, right and down positive, the player's dead zones applied */
    float l2, r2;         /* the triggers' travel, 0..1 */
} BbPad;

/* Version 29: what the world is doing and who else is in it (world_state). */
#define BB_WORLD_IN 1u         /* a world is loaded and running - or held by world_pause */
#define BB_WORLD_LOADING 2u    /* a loading screen is up */
#define BB_WORLD_CUTSCENE 4u   /* a cutscene plays, or one is queued */
#define BB_WORLD_MENU 8u       /* one of the game's menus or dialogs has the controls */
#define BB_WORLD_PAUSED 16u    /* world_pause holds the world (any plugin's) */
#define BB_WORLD_VISITING 32u  /* the player is in another player's world, or on the way there */
#define BB_WORLD_OPEN 64u      /* the player's world is open to others: a sign or a bell is up, or others are in it */

/* BbWorldState.session: the game's own session role. */
#define BB_SESSION_NONE 0
#define BB_SESSION_HOSTING 1       /* the world being opened to others (a bell rung, a sign seen) */
#define BB_SESSION_HOST_FAILED 2
#define BB_SESSION_HOST 3          /* the world's own session, others in it or let in */
#define BB_SESSION_JOINING 4       /* on the way into another player's world */
#define BB_SESSION_JOIN_FAILED 5
#define BB_SESSION_GUEST 6         /* in another player's world */
#define BB_SESSION_LEAVING 7

typedef struct BbWorldState {
    uint32_t size;      /* set to sizeof(BbWorldState) before the call: the host fills as much as both know */
    uint32_t flags;     /* BB_WORLD_* */
    uint32_t others;    /* other players' characters in this world now: co-operators, invaders, the host one visits */
    uint32_t arriving;  /* players on their way in: members of the world's session who are not in it yet */
    int32_t session;    /* BB_SESSION_* */
} BbWorldState;

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
    /* Copies n bytes from a Binary Ninja address inside the image, or from a
     * runtime address the guest has mapped (its heaps, above the image or,
     * under Windows, below the slid image) or above the image. 0 on success. */
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
     * as text). `name` is the plugin's BbPluginInfo name. Main thread.
     * Version 21: a BB_PLUGIN_LIVE_SWITCH plugin also gets the key "" when
     * the player turns it on ("true") or off ("false"). */
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

    /* Version 13: controls and a view of a plugin's own (a first-person mode). */
    /* What the keyboard and mouse hold now, and the presses, motion and wheel
     * since the last read (one reader: two plugins reading would share them).
     * Main thread. 0 on success; 1 without a window (headless). */
    int (*input)(BbInput* out);
    /* Takes inputs from the game while the player is playing
     * (BbInput.playing): the keys listed (scancodes) no longer press what they
     * are bound to, nor the mouse buttons in `buttons` (BbInput's bits);
     * `wheel` nonzero keeps the wheel's notches from the game, and `motion`
     * nonzero keeps the mouse from turning the game's camera. The menus keep
     * every key. A call replaces the last; nothing taken gives everything
     * back. Any thread. 0 on success. */
    int (*input_take)(const uint16_t* scancodes, size_t n, uint32_t buttons, int wheel, int motion);
    /* fn(cam, user) each frame once the game has placed the player's camera
     * (its follow camera and the blend of the cameras it owns) and before
     * anything is drawn from it or a scripted camera (a cutscene's) takes its
     * place: the plugin may move it, turn it and change its projection. The
     * character's own update has run by then; the next frame's movement reads
     * the direction the camera looks. Main thread. 0 on success; 1 for another
     * eboot. */
    int (*on_camera)(void (*fn)(BbCamera* cam, void* user), void* user);
    /* 1 when the player's key bindings put a game action on this key (an SDL
     * scancode), 0 when the key is free - for a plugin that offers keys of its
     * own only where they press nothing else. Actions that work in menus only
     * (Menu Confirm, the menu pages and options) do not count: in play they
     * press nothing. */
    int (*key_bound)(uint16_t scancode);

    /* Version 14: text a plugin places itself. */
    /* A line of text over the game for `seconds`, fading over its last half
     * second: its box's corner at (x, y), fractions of the window (0,0 the
     * top left, 1,1 the bottom right), `align` the box's side the point is
     * (BB_TEXT_LEFT: the box runs right of it, BB_TEXT_CENTRE: centred on it,
     * BB_TEXT_RIGHT: to its left; with BB_TEXT_BOTTOM the box rises from the
     * point instead of hanging under it), `scale` the host's text size (1 is
     * the FPS counter's). One line for all plugins - a new one replaces the
     * last - apart from show_message's banner. Any thread. 0 on success. */
    int (*show_text)(const char* text, float seconds, float x, float y, float scale, uint32_t align);

    /* Version 15: drawing inside the scene. */
    /* fn(cc, user) once a frame on the GX worker that records the first chunk
     * of the scene's render queue at or after `pass` - a sort key's category
     * and sub-category, (bits 22-27 << 6) | bits 16-21: 0xf85 is 3e/05;
     * BBHOST_SCENE_CATEGORIES=1 logs those a frame holds - in a GX command
     * context of its own: the pass of that chunk's first item set up on it
     * (its targets bound), queued just ahead of the chunk, so what fn records
     * there draws into the scene before it. `cc` is the game's GX command
     * context - what its draw code records into: the debug draw lists' play
     * takes it as its parameters' first field. fn runs under the guest's
     * thread state, on a worker: call only the game's draw code (BB_GUEST_ABI
     * pointers from guest_addr) and nothing of the host's but log. Not called
     * on a frame whose scene has nothing at or after the pass. One pass for
     * each fn and user (version 17, up to four in all; one for all plugins
     * before) - a later call with the same fn and user replaces it; -1 takes
     * it away. 0 on success; 1 for another eboot. */
    int (*on_scene_category)(int32_t pass, void (*fn)(uint64_t cc, void* user), void* user);

    /* Version 16. */
    /* Inside an on_scene_category fn: the draws the game's code records on
     * this thread from then on, restyled until set back to BB_DRAW_AS_IS,
     * which fn must do before it returns. The or of: BB_DRAW_LINES - each
     * triangle's edges, a pixel wide, pulled toward the viewer so they show
     * over the faces they outline (BBHOST_WIREFRAME_BIAS), the depth not
     * written; BB_DRAW_BLACK - zero written to every target; BB_DRAW_SUBTRACT -
     * the draw's colour less the target's, once a pixel (the nearest face
     * there; pulled toward the viewer a little, its depth written). A shader
     * that shades from the triangle's slope on screen draws its lines dark:
     * white lines are drawn as black ones over white faces, then the faces
     * again subtracting. Version 17: BB_DRAW_DEPTH_ONLY - no colour written,
     * the depth tested and written; BB_DRAW_DEPTH_EQUAL - the depth tested
     * less-or-equal and not written, pulled toward the viewer a little: a draw
     * made depth-only early in the scene (an occluder for the game's
     * occlusion queries) and again later lands on its own depth. Version 19:
     * BB_DRAW_GBUFFER - drawn into the G-buffer, where the game's lighting
     * lights it as it does the map (the sun, the lamps and their shadows, the
     * fog), with the host's pixel shader in place of the draw's: for the
     * game's simple coloured draw (the debug draw's), rgb the albedo (linear)
     * and alpha the face's world normal when negative - -(1 + u + 2048 v), u
     * and v its octahedral coordinates in 11 bits each (gbuffer_flat.frag).
     * The draw marks the stencil as the map's G-buffer draws do, which the
     * lights test. Only where the G-buffer is bound (a pass of the scene
     * ahead of 3e/13, such as 3e/0c); elsewhere the draw is drawn as it is. */
    void (*draw_style)(int style);

    /* Version 18: the 3D's look. What the camera's draws - the 3D world, its
     * characters, its sky and effects: the draws with the scene's depth that
     * are not full-screen rect lists, as the upscaler's jitter takes them - are
     * drawn with. The shadow maps, the post-processing and the HUD and menus
     * keep the game's own. scene_scale and scene_colour_bits decide the scene's
     * targets: call them from bb_plugin_init for the start, or once the game
     * runs from the main thread (on_frame, on_option), which makes the targets
     * again at the next quiet point, as a Resolution change does - from
     * bb_plugin_image they change nothing until then. The others: any thread,
     * from the next frame. */
    /* The 3D's vertices moved, as their stage writes them, to the nearest
     * corner of a grid of the screen `lines` cells high and as many across as
     * the 3D's shape gives (240: 427 x 240 at 16:9), as the PS1's landed on
     * its pixels; BB_SNAP_PIXELS the 3D's own pixels; 0 off. */
    void (*scene_snap)(uint32_t lines);
    /* How the 3D samples its textures: BB_FILTER_NEAREST - the nearest texel of
     * the nearest mip, no anisotropy (depth comparisons keep their own) - or
     * BB_FILTER_AS_IS. */
    void (*scene_filter)(int filter);
    /* The 3D's size against the game's: BB_SCALE_NEAREST draws the game, its
     * HUD and menus at the window's size and the 3D at the Resolution - or
     * `lines` high (0: the Resolution) as wide as the window's shape - brought
     * up to the window texel for texel, with no smoothing: no upscaler and no
     * anti-aliasing; a Resolution as large as the window or larger draws the
     * 3D at the window's size.
     * BB_SCALE_AS_IS gives it back to the Resolution, Upscaling and
     * anti-aliasing settings. */
    void (*scene_scale)(int scale, uint32_t lines);
    /* The scene's colour targets - the lit scene, its light and its
     * reflections - in 8 bits a channel (RGBA8, as the rgba8f patch makes
     * them), or (0) in the game's and its patches' formats. */
    void (*scene_colour_bits)(int bits);
    /* The colour grading's table - the area's CGrading textures
     * (map/mXX/mXX_cgrading.tpf: m24_00_CGrading0001 ...), copied each frame
     * into a 16x16x16 table that the post-processing's grading pass samples -
     * sampled BB_FILTER_NEAREST, the grading stepping from entry to entry (16
     * shades a channel), or BB_FILTER_AS_IS, smoothly between them. */
    void (*grading_filter)(int filter);
    /* Version 19: BB_DRAW_GBUFFER draws' indirect light - what a lightmap
     * gives the map, which they have none of - from the next frame (0.03 by
     * default; 0 leaves what no lamp reaches black). */
    void (*gbuffer_ambient)(float level);

    /* Version 20: the camera the scene is drawn from. */
    /* fn(cam, user) each frame once the game has copied the player's camera
     * (on_camera's, after its callbacks) into the scene's, before a scripted
     * camera (a cutscene's) replaces it: what fn writes is what is drawn, while
     * the game's own camera - the one the follow camera keeps, the controls
     * move by and the aim reads - stays as on_camera left it. Not called in
     * the game state on_camera reaches through its second hook (a warp's).
     * Main thread. 0 on success; 1 for another eboot. */
    int (*on_view)(void (*fn)(BbCamera* cam, void* user), void* user);

    /* Version 21: old-hardware sound. Everything the game plays - its final
     * mix, as it reaches the device - low-passed at `lowpass_hz` (a one-pole
     * filter: the muffled top end of a PS1-era sound chip), held down to
     * `rate_hz` (sample-and-hold, the steps kept across buffers), rounded to
     * `bits` a sample (no dither: the crunch; quieter than half a step is
     * silence) and, after a hold, smoothed at 0.45 of `rate_hz` as a DAC's
     * output filter smoothed its steps. 0 turns each off; all three
     * 0 is the game's own sound, which costs nothing. Any thread, from the
     * next buffer. */
    void (*audio_crush)(int rate_hz, int bits, float lowpass_hz);

    /* Version 22: a cutaway for a high camera (the Isometric's). The camera's
     * draws - the world, its characters, the effects, not the shadows - are
     * clipped where they lie within `radius` of the view's axis and nearer
     * than `near_z` along it (both in metres, view space): what stands between
     * the camera and the character at the screen's middle, `near_z` the
     * character's distance less a margin. `tan_x` and `tan_y` are the tangents
     * of half the drawn view's field of view across and up (on_view's fov_y
     * and aspect). radius 0 turns it off. Each new set of numbers makes the
     * camera draws' pipelines again (rounded to 1/4 m), so set it as the
     * distance or the field of view change, not every frame. Any thread. */
    void (*view_cutaway)(float tan_x, float tan_y, float near_z, float radius);

    /* Version 23: the cutaway per pixel, faded (BbCutaway). The map's own
     * camera draws (never the characters, the objects or their parts) leave out, in their pixel
     * stage, what lies within `radius` of the view's axis - `widen` metres
     * wider for each metre nearer the camera than the point looked at - more
     * than `height` metres above that point and more than `margin` in front of
     * it, fading in over `feather` metres at each of those edges (a dither:
     * the G-buffer has no blending). The view only turns and moves: `pitch`
     * (radians below the horizon) and `distance` (to the point looked at, at
     * the screen's middle) place it; the field of view and the depth come from
     * the drawn view itself. NULL or radius 0 turns it off. Turning it on
     * makes the camera draws' pipelines again once; after that (since version
     * 26) the numbers are read live, so a new set each frame - a zoom easing
     * in - builds nothing. Any thread. */
    void (*view_cutaway2)(const struct BbCutaway* cut);
    /* Version 24: view_cutaway2's cone for the map's objects too (never the
     * characters or what they carry), its tip `start` of the way from the
     * point looked at toward the camera - 0.5 halfway - and the rest of it
     * as view_cutaway2 sets it. Below 0 (the default) the objects are never
     * cut. Read live since version 26: it may change each frame. Any
     * thread. */
    void (*view_cutaway_objects)(float start);
    /* Version 25: the finished picture - the 3D and its post effects, before
     * the HUD and the menus are drawn over it - rounded to `bits` a channel
     * through the PS1 GPU's 4x4 ordered dither, a cell of the pattern a pixel
     * of the 3D (scene_scale's lines: as coarse as the 3D). 5 is the PS1's
     * 15-bit colour; 0 (the default) off. Any thread. */
    void (*scene_dither)(uint32_t bits);
    /* Version 26: the player's Camera Zoom presses (the wheel and the middle
     * button by default, as bound; in the world only) since the last call:
     * `steps` notches out (in when negative), `reset` 1 when Camera Zoom
     * Reset was pressed. From the first call on they are the plugin's and
     * the game's camera no longer zooms. Any thread. 0 on success. */
    int (*camera_zoom)(int* steps, int* reset);

    /* Version 27: the frame as it is drawn, plugins' compute work in it,
     * and who does the anti-aliasing (the conventions above BbFrameCamera). */
    /* The 3D frame being recorded (inside a hook), else the last one
     * described, into *out: set out->size to sizeof(*out) first; the host
     * fills as much as both know and sets out->size to that. From the first
     * call on (or the first on_render) the host describes every 3D frame.
     * Any thread. 0 on success; 1 before a frame is described, or for a size
     * under 8. */
    int (*frame_info)(BbFrameInfo* out);
    /* fn(call, user) at each of `points` (BB_RENDER_BEFORE_UPSCALE,
     * BB_RENDER_AFTER_UPSCALE, or-ed; | BB_RENDER_MOTION for the motion) of
     * every 3D frame, on the thread that records the frame - not the game's
     * main thread - with the renderer's lock held: the command buffer is
     * outside any render pass, every image in `call` is in the layout it
     * gives, and what the frame wrote to them is visible to compute and
     * transfer. fn records compute and transfer work only - dispatches,
     * copies, blits, clears and the barriers between its own steps: no render
     * pass, draw, graphics pipeline or dynamic state - and leaves each image
     * in its layout. It never submits, ends the command buffer or waits for
     * the GPU (no queue, fence, semaphore or query-wait calls), and calls
     * nothing of the host's but log, frame_info and on_render. The host makes
     * what fn wrote visible to what follows and binds its own state again
     * after it. A later call with the same fn and user replaces its points;
     * no points take it away. `name` is the plugin's BbPluginInfo name: its
     * hooks are not called while it is off (BB_PLUGIN_LIVE_SWITCH), and go if
     * it is unloaded. Several hooks run in the order they were registered.
     * From bb_plugin_init on, any thread. 0 on success; 1 for a bad
     * argument. */
    int (*on_render)(const char* name, uint32_t points, BbRenderFn fn, void* user);
    /* The calling plugin's anti-aliasing claim: BB_AA_NONE, BB_AA_LEAVE or
     * BB_AA_TAKE (its BbPluginInfo.flags' BB_PLUGIN_AA_* is the claim it
     * starts with). The strongest claim of the plugins that are on holds -
     * BB_AA_TAKE before BB_AA_LEAVE - and a plugin that is off has none. From
     * bb_plugin_init for the start, or once the game runs from the main
     * thread (on_frame, on_option): the 3D's size follows at the next quiet
     * point, as at a Resolution change. Never from a render hook. 0 on
     * success; 1 for a bad argument or from a hook. */
    int (*aa_claim)(const char* name, int claim);

    /* Version 27: shader overrides (docs/plugins.md, "Shaders") - a plugin's
     * HLSL in place of one of the game's shaders, as an edited file in
     * <data>/bbhost/shaders replaces one (docs/modding.md). `shader` is named
     * as BBHOST_HLSL_DUMP=1 names the shader's file, less ".hlsl":
     * "GXGuiShader_Tex.ppo@6d921b0b.ps" - its bundle entry and GX's name, then
     * the stage (vs, ps, hs, ds, gs or cs) - for every variant of the shader in
     * that stage whose interface the source declares;
     * "GXGuiShader_Tex.ppo@6d921b0b.ps~7de085b59d21219a" for that one variant;
     * "<lift hash>.ps" for a shader whose name is not known. The source is
     * compiled by DXC for each variant met, and drawn with where it declares
     * the stage's interface (the dumped file's bindings, inputs and outputs)
     * and its SPIR-V validates; anywhere else the shader keeps its own. Only on
     * the HLSL path (the Shaders setting's default). The player's own file of
     * the same name comes first, then the plugin loaded first (by file name).
     * A change is compiled for the variants met so far, then takes effect at
     * the shader's next draw, which builds the pipelines drawing with it again
     * - once: going back to a source drawn with before takes the pipelines made
     * for it and compiles nothing. A value that changes goes through
     * shader_data, not the text. Any thread. */
    /* `hlsl` (UTF-8, ended by a 0, at most 4 MiB) for `shader`; NULL takes the
     * plugin's away. The same text again changes nothing. 0 on success; 1 for
     * a name that is no shader's, an unknown plugin or too long a text. */
    int (*shader_override)(const char* name, const char* shader, const char* hlsl);
    /* Every <shader>.hlsl file in `dir` - absolute, or relative to the
     * plugin's own folder (plugin_dir) - as shader_override takes a text, each
     * read when its shader is built (and again when it changes, with
     * BBHOST_SHADER_WATCH=1). The plugin's shader_override texts come before
     * its folders' files, and a folder before the ones added after it. NULL
     * takes all its folders away. The number of .hlsl files in it, or -1 when
     * it is not a folder. */
    int (*shader_folder)(const char* name, const char* dir);
    /* What became of the plugin's override of `shader` (BB_SHADER_*), and why
     * into `why` (n bytes with its ending 0; may be NULL): DXC's first error,
     * the interface that differs, the file or plugin that comes first, why
     * overrides are off. */
    int (*shader_state)(const char* name, const char* shader, char* why, size_t n);
    /* The HLSL this build prints for `shader`, as BBHOST_HLSL_DUMP writes it -
     * "<name>.<stage>": the first variant this run met; "...~<lift hash>":
     * that one - so a plugin can patch the game's own source rather than ship
     * whole copies, which an update that changes the shader's interface
     * refuses. Into a buffer the host owns until the calling thread's next
     * call: *hlsl, and its length without the ending 0 in *size. 0 on success;
     * 1 while no stage of it has been built this run (as the game creates or
     * first draws it: ask again from on_frame); 2 when overrides do not apply
     * (BB_SHADER_OFF). */
    int (*shader_source)(const char* shader, const char** hlsl, size_t* size);
    /* `size` bytes at `offset` into the plugin's shader data -
     * BB_SHADER_DATA_BYTES of its own, 0 at the start - which its overrides
     * read with bbhost.hlsli's bb_data_* (#include "bbhost.hlsli"). Each frame's
     * draws read one copy: the data as it stood when the frame before had been
     * recorded, so write what goes together in one call. Changing it builds
     * nothing. Any thread. 0 on success; 1 past the end or for an unknown
     * plugin. */
    int (*shader_data)(const char* name, uint32_t offset, const void* data, uint32_t size);

    /* Version 28: the recompiled follow camera's visual target. Register once
     * from bb_plugin_image. fn receives the character whose controls the
     * camera uses and returns a live ChrIns to follow, or NULL for unchanged.
     * Called on the game thread for each update; never retain a dead actor.
     * Only the model and moving-floor position are redirected, not input.
     * Native-camera builds still require the plugin's native camera hooks.
     * 0 on success; 1 outside image setup, for NULL, or if already owned. */
    int (*on_camera_target)(void* (*fn)(void* character, void* user), void* user);

    /* Version 29: the world held still, the controller, and who else is in
     * the world (a photo mode's). */
    /* The world held still - its characters, physics, AI, events, effects and
     * cloth - while frames go on being drawn and on_view goes on being called:
     * the game's own pause, the one it holds while it leaves a world. `paused`
     * 1 asks for it for the plugin named `name` (its BbPluginInfo name), 0 lets
     * it go; the world is held while any plugin asks. It takes hold at the
     * world's next update. A load lets every hold go (a new world starts
     * running), and so does the plugin being turned off. Any thread. 0 on
     * success; 1 for another eboot or an unknown plugin. */
    int (*world_pause)(const char* name, int paused);
    /* While the plugin holds the world: `updates` more of its updates run, one
     * a frame, and then it is held again (a frame advance). A later call adds
     * to what is left. Any thread. 0 on success; 1 when the plugin holds no
     * pause. */
    int (*world_step)(const char* name, uint32_t updates);
    /* The controller, into *out (set out->size first): its own buttons and
     * sticks - the dead zones applied, the player's key bindings not - pressed
     * or not, as the game would read them before pad_take takes any; a test's
     * taps (BBHOST_AUTOPRESS, the "tap" request) count as the controller's. One
     * reader: the presses are the caller's once read. While bbhost's own menus
     * or a text box have the controller it reads nothing held. Main thread. 0
     * on success; 1 for a size under 8. */
    int (*pad)(BbPad* out);
    /* Takes the controller's buttons (BB_PAD_*) and axes (BB_PAD_LEFT_STICK,
     * BB_PAD_RIGHT_STICK, BB_PAD_TRIGGERS) from the game: the game reads them
     * let go and at rest - whatever they are bound to in play is not pressed -
     * while pad still reads them. Keys are input_take's. A call replaces the last;
     * 0, 0 gives everything back. Any thread. 0 on success. */
    int (*pad_take)(uint32_t buttons, uint32_t axes);
    /* What the world is doing and who else is in it (BbWorldState), into *out
     * (set out->size first). Main thread. 0 on success; 1 for another eboot or
     * a size under 8. */
    int (*world_state)(BbWorldState* out);
} BbHostApi;

#define BB_SNAP_PIXELS 0xffffffffu /* version 18: scene_snap's grid the 3D's own pixels */

#define BB_SHADER_DATA_BYTES 65536u /* version 27: a plugin's shader data (shader_data) */
/* Version 27: shader_state. */
#define BB_SHADER_NONE 0     /* the plugin has no override of that shader */
#define BB_SHADER_WAITING 1  /* none of the shader's stages built with it yet: not met since it was set (or compiling) */
#define BB_SHADER_TAKEN 2    /* compiled and drawn with (why: a variant it does not fit, if any) */
#define BB_SHADER_REFUSED 3  /* refused - DXC, another interface, invalid SPIR-V, the driver: the shader keeps its own */
#define BB_SHADER_SHADOWED 4 /* not used: the player's own file of that name, or an earlier plugin's, comes first */
#define BB_SHADER_OFF 5      /* overrides do not apply: the lift alone (the Shaders setting at Lifted,
                                BBHOST_HLSL=0) or video.shader_overrides = false (without DXC every
                                override is refused instead: "DXC not found") */
#define BB_FILTER_AS_IS 0          /* version 18: scene_filter, grading_filter: the game's filtering */
#define BB_FILTER_NEAREST 1        /* version 18: the nearest texel of the nearest mip */
#define BB_SCALE_AS_IS 0           /* version 18: scene_scale: the settings' sizes */
#define BB_SCALE_NEAREST 1         /* version 18: the 3D at its own size, brought up nearest */

#define BB_DRAW_AS_IS 0
#define BB_DRAW_LINES 1
#define BB_DRAW_BLACK 2
#define BB_DRAW_SUBTRACT 4
#define BB_DRAW_DEPTH_ONLY 8   /* version 17: no colour written, the depth tested and written */
#define BB_DRAW_DEPTH_EQUAL 16 /* version 17: the depth tested less-or-equal, not written, pulled toward the viewer a little */
#define BB_DRAW_GBUFFER 32     /* version 19: into the G-buffer, lit by the game (see draw_style) */

#define BB_TEXT_LEFT 0u
#define BB_TEXT_CENTRE 1u
#define BB_TEXT_RIGHT 2u
#define BB_TEXT_BOTTOM 4u

#ifdef __cplusplus
}
#endif

#endif
