# Plugins

A plugin is a shared library (`.so` on Linux, `.dll` on Windows) that bbhost
loads at start. Because the game runs natively inside bbhost's process, a
plugin can read and write game objects directly, hook game functions, call
them, and use the services bbhost builds on top of the engine (params, event
flags, the player, warps, characters, settings, file overlays).

Like the rest of bbhost, the plugin API (`include/bbhost_plugin.h`,
`include/bbhost/`) is licensed under the GNU GPL, version 3 or later. A
plugin built against it is a derivative work: if you distribute one, it must
be under a GPL-compatible license, with its source.

## Where plugins live

bbhost loads every plugin in `<exe dir>/plugins` and `<data>/plugins`. A file
in a later directory replaces one with the same name in an earlier one.
`BBHOST_PLUGINS=0` loads none.

Plugins are enabled and disabled by name under `[plugins]` in `bbhost.toml`:

```toml
[plugins]
randomizer = true     # an opt-in plugin, turned on
mutators = false      # any plugin can be turned off
signed_only = false   # true: load only signed official plugins
```

The setup window has a Plugins tab with the same switches and each plugin's
settings. In game, F9 opens the patch and plugin menu over the game.

## A first plugin

`plugins/hello/hello.c` is the example. bbhost's own build produces it as
`build/plugins/hello.so` (or `hello.dll`). A plugin needs only the header:

```sh
# Linux
cc -shared -fPIC -I<bbhost>/include -o plugins/mine.so mine.c
# Windows, cross-compiled with Clang
clang --target=x86_64-w64-mingw32 -shared -I<bbhost>/include -o plugins/mine.dll mine.c
```

The smallest useful plugin counts frames:

```c
#include "bbhost_plugin.h"

static const BbHostApi* g_api;
static unsigned g_frames;

static void on_frame(void* user) {
    (void)user;
    if (++g_frames % 300 == 0) g_api->log("hello: another 300 frames");
}

BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION, 0, "mine", "My plugin", "1.0.0", "me", "Counts frames."};

BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    if (api->version < 2) return 1;  /* on_frame arrived in version 2 */
    g_api = api;
    api->on_frame(on_frame, 0);
    return 0;
}
```

## Life cycle

| Entry point | Called | Typical use |
|---|---|---|
| `bb_plugin_info` (data, optional) | read before anything else, also by the setup window | name, title, version, flags |
| `bb_plugin_options` (data, optional) | read without starting the plugin | the settings the menus draw |
| `int bb_plugin_init(const BbHostApi*)` | before the eboot is loaded | `log`, `config`, `register_hle` |
| `int bb_plugin_image(const BbHostApi*)` | after the eboot is loaded and bbhost's patches are in | everything else |

Either entry point may be missing. Returning nonzero unloads the plugin with a
log line. Everything a plugin logs is prefixed with `plugin:`.

The API only ever grows. A plugin checks that `api->version` is at least the
version whose calls it makes, and `api->size` for fields a newer host
appended. The current version is 21 (`BB_PLUGIN_API_VERSION`).

## Addresses

Addresses in the API are Binary Ninja addresses of the 1.09 eboot, with the
image at its preferred base `0x400000`. `guest_addr()` converts one to the
runtime address. `read`, `write`, `patch` and `hook` take Binary Ninja
addresses directly and refuse them when the loaded eboot is not the 1.09
build. A runtime address above the image (the game's heap) is passed as is.

The engine SDK expresses addresses as image-relative `bb::Rva` values; `.bn()`
gives the Binary Ninja address.

## The API by area

**Memory and code**

- `read(addr, out, n)`: a safe copy; 0 on success. `addr` is a Binary Ninja
  address inside the image, or the runtime address of guest memory outside it
  (a pointer read from the game's heaps). Under Windows the image may be slid,
  and the heaps lie below it as well as above.
- `write(addr, data, n)`: writes data pages.
- `patch(addr, expect, bytes, n)`: writes code bytes only if `expect` is there,
  then makes the page read-only and executable again. A patch manifest
  (see [modding.md](modding.md)) is simpler when the bytes are fixed.

**Hooks and calls**

- `hook(addr, prologue, n, fn, user)`: a prologue hook. `fn` sees the six
  integer argument registers and returns nonzero to skip the function.
  `prologue` is the function's first `n` bytes (at least 14, none of them
  rip-relative).
- `hook2(...)` (v8): the same with a `BbHookCtx`: integer and float argument
  registers, both writable, and the value a skipped call returns.
- `call_guest(addr, regs)` (v8): calls a game function with up to six integer
  arguments, eight float registers and four stack arguments; returns rax and
  xmm0. Call it from the main thread.
- `symbol(name)` (v8): the address of a named engine symbol from
  `bbhost/engine/symbols.hpp`.
- `register_hle(name, fn)`: replaces bbhost's implementation of a PS4 system
  function by its export name. Only from `bb_plugin_init`. The game calls `fn`
  with the SysV ABI, so it must be declared `BB_GUEST_ABI` (Clang or GCC; not
  possible with MSVC).

**The engine's state**

- `on_frame(fn, user)` (v2): once per game frame, on the game's main thread.
  This is where game state may be touched safely.
- `param_row`, `param_get`, `param_set`, `param_ids` (v2, v8): param rows by
  table name and id, and fields by the game's own field names, checked against
  the paramdefs the game ships.
- `lua_event(name)` (v2): raises a named Lua event the way the game's scripts do.
- `event_flag_get`, `event_flag_set` (v3) and `on_event_flag` (v4): the world's
  state as numbered flags (bosses killed, doors opened, NPC progress), and a
  callback for every flag a setter changes.
- `player_stat_get`, `player_stat_set` (v3): level, echoes, insight, the six
  attributes, HP and stamina, and the save's counters, by name.
- `player_position`, `camera_get` (v5), `player_block` (v6): where the player
  and the camera are, and the map block.
- `player_warp` (v6) and `lamp_warp` (v7): move the player within a block or
  travel to a lamp. Only for opt-in gameplay plugins the player has turned on.
- `on_world_load` (v8): called whenever the player is placed in a map. For a
  load that is as soon as the load has built the player, with its loading
  screen still up.
- `loading_screen()` (v12): 1 while a loading screen is up - the host's own
  test, the one behind its log's `loading:` lines. A plugin that moves the
  player waits for 0: a warp asked inside the screen lands before the world
  is ready, and a lamp travel is refused once the load has replaced the
  player.
- `chr_list` (v8): every character the world holds, with its `ChrIns` pointer,
  NpcParam row, model, team, HP and position.
- `sp_effect_apply`, `sp_effect_has` (v8): apply a SpEffectParam row to a
  character the way the game's scripts do.

**Controls and the view** (v13)

- `input(out)`: the keyboard by SDL scancode (the USB HID usage: 26 is W, 44
  Space, 224 Left Ctrl), the mouse's buttons held and pressed since the last
  read, its motion in its own counts and the wheel's notches - and `playing`,
  1 while the mouse turns the camera (in the world, no menu up). One reader:
  the motion, presses and notches are the caller's once read.
- `input_take(scancodes, n, buttons, wheel, motion)`: while the player is
  playing, those keys and mouse buttons stop pressing what they are bound to,
  the wheel's notches stay out of the game, and the mouse stops turning the
  game's camera - the plugin reads them all through `input`. The menus keep
  every key. A call replaces the last; nothing taken gives everything back.
- `on_camera(fn, user)`: each frame once the game has placed the player's
  camera - the follow camera's update and the camera manager's blend of the
  cameras it owns - and before the scene is drawn from it or a scripted camera
  (a cutscene's) takes its place, with a `BbCamera`: its basis (right, up,
  forward) and position in the world, its vertical field of view, aspect and
  clip planes, all writable. The character's update has run by then; its next
  one moves by where this camera looks. The Quake plugin is a first-person
  view this way.
- `on_view(fn, user)` (v20): each frame after the game has copied that camera
  into the scene's, before a scripted camera replaces it: what `fn` writes is
  drawn, while the game's own camera - which the follow camera keeps, the
  controls move by and the aim reads - stays as `on_camera` left it. The
  Isometric plugin draws its view this way.

**Settings, UI and files**

- `bb_plugin_options` (v9): a table of headings, switches, numbers, choices,
  text fields and action buttons. Values are stored under `[<plugin name>]` in
  the per-user `bbhost.toml` and read back with `config("<name>.<key>")`.
- `on_option`, `on_action` (v9): callbacks for changes and button presses made
  in the in-game menu.
- `show_message(text, seconds)` (v8): a banner over the game.
- `show_text(text, seconds, x, y, scale, align)` (v14): a line of text the
  plugin places: its box's corner at (x, y), fractions of the window (0,0 the
  top left), `align` `BB_TEXT_LEFT`, `BB_TEXT_CENTRE` or `BB_TEXT_RIGHT` of the
  point, `| BB_TEXT_BOTTOM` for a box that rises from it, `scale` the overlay
  text's (1 the FPS counter's). It fades over its last half second; a new one
  replaces the last, apart from the banner. The SDK's `text_at` falls back to
  the banner on an older host. The Quake plugin names the weapon it switches to
  this way, small, in the bottom left corner.
- `on_scene_category(pass, fn, user)` (v15): drawing inside the scene.
  `fn(cc, user)` runs once a frame on the GX worker that records the first
  chunk of the scene's render queue at or after `pass` - a sort key's category
  and sub-category, `(bits 22-27 << 6) | bits 16-21` (`BBHOST_SCENE_CATEGORIES=1`
  logs those a frame holds) - in a GX command context of its own with that
  chunk's pass set up, queued just ahead of it: what it records draws into the
  scene's targets before that pass does. One pass for each `fn` and `user`
  (v17, up to four; v15-16 one for all plugins); -1 takes it away. The Lo-Hit
  plugin plays the game's collision debug draw there, under the effects.
- `draw_style(style)` (v16): inside an `on_scene_category` fn, the draws the
  game's code records on that thread from then on restyled, until set back to
  `BB_DRAW_AS_IS` before fn returns: `BB_DRAW_LINES` (each triangle's edges a
  pixel wide, pulled toward the viewer, no depth written;
  `BBHOST_WIREFRAME_BIAS` -512 and `BBHOST_WIREFRAME_SLOPE` -1 by default),
  `BB_DRAW_BLACK` (zero written), `BB_DRAW_SUBTRACT` (the draw's colour less the
  target's, once a pixel), `BB_DRAW_DEPTH_ONLY` (v17: no colour, the depth
  written), `BB_DRAW_DEPTH_EQUAL` (v17: tested less-or-equal against the depth
  an earlier depth-only draw of the same geometry wrote), or-ed.
  `BB_DRAW_GBUFFER` (v19) draws into the G-buffer (a pass ahead of `3e/13`,
  such as `3e/0c`) with the host's pixel shader (`gbuffer_flat.frag`) in
  place of the draw's, marking the stencil as the map's G-buffer draws do,
  so the game's lighting lights it as it does the map: the sun, the lamps
  and their shadows, the fog. For the game's simple coloured draw (the debug
  draw's): rgb the albedo (linear), alpha the face's world normal when
  negative, `-(1 + u + 2048 v)` with u and v its octahedral coordinates in
  11 bits. The Lo-Hit plugin's "Lit by the game" draws the collision this way.
- `gbuffer_ambient(level)` (v19): `BB_DRAW_GBUFFER` draws' indirect light,
  which the map gets from its lightmaps (0.03 by default).
- `audio_crush(rate_hz, bits, lowpass_hz)` (v21): old-hardware sound. The
  game's final mix, as it reaches the device, low-passed at `lowpass_hz` (one
  pole), held down to `rate_hz` (sample-and-hold, kept across buffers so their
  boundaries don't click) and truncated to `bits` a sample with no dither; 0
  turns each off, and all three 0 costs nothing. Any thread, from the next
  buffer. The Bloodborne PS1 plugin's Sound.
- The 3D's look (v18): how the camera's draws - the world, its characters, the
  sky and the effects, the draws with the scene's depth that are not
  full-screen rect lists - are drawn; the shadow maps, the post-processing and
  the HUD and menus keep the game's own. `scene_snap(lines)` moves each vertex,
  as its stage writes it, to the nearest corner of a grid of the screen `lines`
  cells high (as many across as the 3D's shape gives; `BB_SNAP_PIXELS` the
  3D's own pixels; 0 off) - the PS1's wobble. `scene_filter(BB_FILTER_NEAREST)`
  samples their textures nearest (the nearest mip, no anisotropy; depth
  comparisons keep theirs). `scene_scale(BB_SCALE_NEAREST, lines)` draws the
  game, its HUD and menus at the window's size and the 3D at the Resolution, or
  `lines` high, brought up to the window texel for texel with no upscaler.
  `scene_colour_bits(8)` makes the scene's colour targets RGBA8, as the rgba8f
  patch does. `grading_filter(BB_FILTER_NEAREST)` samples the colour grading's
  table nearest - the area's `CGrading` textures (`map/mXX/mXX_cgrading.tpf`)
  are copied each frame into a 16x16x16 table that the grading pass samples -
  so the grading steps from entry to entry, 16 shades a channel.
  `scene_scale` and `scene_colour_bits` decide the scene's targets: call them
  from `bb_plugin_init` for the start, or once the game runs from the main
  thread (`on_frame`, `on_option`), which makes the targets again at the next
  quiet point; the others work from any thread. With `scene_scale` on, the
  game and its HUD are always the window's size and the 3D at most that. The
  Bloodborne PS1 plugin is all of these.
- `plugin_dir(name)` (v8): a data folder of the plugin's own.
- `game_file(path, ...)` (v9): a game file by its `/app0` path, decompressed.
- `overlay_file(name, path, data, size)` (v9): a file the game reads instead
  of its own, compressed again when the path ends in `.dcx`. The randomizer
  rewrites map layouts this way.
- `replace_text(texts, count)` (v11): the game's own text. Every reference to
  each UTF-16 string in the image - the lea instructions that load its address
  and the pointers to it in the image's data - names the new text instead; the
  string itself stays. From `bb_plugin_image` only. The Debug Menu plugin's
  English is made this way. Replace only text the game shows: a string it also
  looks up by name stops matching. The character scripts call the game's
  functions by their Japanese names, and with those names in English every
  character died at load.

**Online rules** (v10)

A gameplay plugin is announced to the server with every request
(`X-BBHost-Ruleset`), so modded runs stay apart from vanilla ones and players
only meet others with the same rules. `set_rules` updates the plugin's entry
(for example when its seed changes); `on_world_rules` and `visitor` let a
plugin flagged `BB_PLUGIN_ADOPTS_RULES` play by a host's rules when summoned
into another player's modded world.

## The C++ SDK

C++ plugins can use `include/bbhost/`:

| Header | Contents |
|---|---|
| `bbhost/sdk.hpp` | `bb::Plugin`: the API with lambdas (`every_frame`, `on_world_load`, `on_event_flag`, `on_camera`), typed params, characters, flags, stats, messages, `loading_screen()` (on a host older than v12 it makes the same test from the game's memory), `gravity()`, `call("SYMBOL", args...)`; `bb::Rng` |
| `bbhost/params.hpp` | a struct for every param row type and a tag for every table, generated from the game's own paramdefs, with offsets checked by `static_assert` |
| `bbhost/engine/` | engine layouts and addresses for 1.09: characters, the world, matchmaking and sessions, GX, YEBIS and more |

```cpp
#include "bbhost/sdk.hpp"

static bb::Plugin g;

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    if (!g.attach(api, 8)) return 1;
    g.every_frame([] {
        // The Saw Cleaver weighs nothing.
        if (auto* w = g.param<bb::params::EquipParamWeapon>(1000000)) w->weight = 0;
    });
    return 0;
}
```

`tests/sdk_headers_test.cpp` compiles every header as part of the test suite.

## Which tool for which change

- A number the game reads from a table (damage, enemy HP, item lots, shops):
  change the param row once the table has loaded. Characters are recomputed
  from their params, so writing a character's own copy does not last.
- Something that should happen now (a buff, a status): apply a SpEffectParam
  row.
- World state (bosses, doors, NPC progress): event flags. A boss fight is a
  family of flags around its defeat flag.
- Moving the player: `player_warp` inside a block, `lamp_warp` between maps,
  each once `loading_screen()` is 0. Count waits in seconds, not frames: a
  loading screen can flip hundreds of times a second.
- Physics and time: `gravity()` and the time scale.
- Anything else: the engine layouts, `hook2` and `call_guest`.

## Official plugins

The plugins in `plugins/` are built with bbhost and shipped in its packages,
each with a `.sig` file: an Ed25519 signature by the release key. bbhost checks
it before loading; a plugin whose signature does not match is refused. The
setup window can download the latest signed plugins from the newest release.

| Plugin | What it does |
|---|---|
| `randomizer` | Shuffles one-time pickups and shop stock from a seed, and optionally enemies: each regular enemy becomes another kind (model, AI and gear together), drawn from its own area or from the whole game, with optional roaming bosses. Writes a spoiler file. |
| `boss_rush` | Fights the 19 story bosses back to back, optionally with the ten chalice-dungeon bosses, starting each round outside the boss's fog wall. Tracks time and deaths. Practice fights one boss, picked by name, again and again: once, N times, or until stopped, a death trying again from the fog wall and a win waking the boss for the next fight. Each fight that ends in a win or a death is an attempt, so Once and N times end after that many attempts; Restart starts the fight under way over from the fog wall without counting it. With Refill on, every attempt starts with the most blood vials and quicksilver bullets. A practice can start each fight at the boss's phase 2 or 3: once the fight begins, the boss is put just under the HP at which its own event script or AI changes phase, and the game plays the change itself (Ludwig's cutscene and Holy Blade, Gascoigne's beast, Lady Maria's blood and fire). Attempts, wins, deaths, the best and last clear times and, from phase 1, the time to phase 2 are shown after each fight and written to `results.txt`. Boss HP and Your HP are practice's cheats, for learning a fight: Can't die keeps hits taking HP down to 1 and no further (the debug menu's No Dead), and Infinite also puts the HP back after every hit, so the boss stays in the phase its fight began in and the player stays at full health. A boss that cannot die is not defeated, so its fights end in a death, a Restart or Stop. |
| `mutators` | Switchable rules: gravity, game speed, bullet time at low HP, one-hit deaths, blood drain, enemy HP, echo rate, random effects. |
| `quake` | Off until turned on. First person with Quake 3's movement and view: strafe jumping, bunny hopping (hold jump to keep hopping), crouching, the walls only characters meet (the invisible ones along ledges) as solid as the rest, creatures as solid as Quake 3's players, and a choice of physics - Quake 3 (VQ3, the default), CPMA, Quake 1 + CPMA, QuakeWorld and DeFRaG's. Five guns made of Bloodborne's (Cannon, Church Cannon, Piercing Rifle, Blunderbuss, Gatling Gun), always +10, firing from the view with each gun's own flash and sound: rockets and grenades at Quake 3's speeds (the grenade bouncing until its fuse), the Blunderbuss's and the Gatling Gun's own bullets, and a plasma of A Call Beyond's star or Blacksky Eye's shard; their blasts throw the player (rocket and grenade jumps, plasma climbing), and the gun that is up is drawn at the lower right with Quake 3's bob and kick. Field of view, mouse sensitivity, gravity and knockback are settings. |
| `lohit` | Off until turned on. The map's lo-hit collision drawn in place of the map's own geometry (the game's own WorldHitMan debug draw, with GROUP MASK > Map off): every part loaded (Whole map, the default), or the part you stand on and the parts connected to it, one step out; Hi-hit shows the detailed collision characters and objects use instead. Each mesh is built once into a vertex buffer of its own and drawn with two draws a frame, where the game's debug draw rebuilt every triangle each frame and copied it twice (the triangles' edges the game drew over the faces, a second draw costing the GPU more than the faces, are left out). Its depth (plugin API 17) is drawn ahead of the scene's objects too, so the game's occlusion culling skips what is behind its walls, as the map's own geometry would. It is drawn inside the scene ahead of its effects so fire, sparks, blood and mist show over it, by default in the launcher's deep violet (walls) and dark blue (floors); the Colours sliders set both (0-255, as they come out on the screen) for the lo-hit and the hi-hit. Pair it with Performance mode (the setup window's main page) for the post-effects off. |
| `ps1` | Off until turned on. Bloodborne PS1: the 3D drawn as the first PlayStation drew it, the HUD and the menus left sharp at the window's size: its vertices snapped to a grid of the screen as it moves (Vertex snap: 240p, the PS1's own screen, by default; 480p, 360p, 120p, Native - the 3D's own pixels - or off), its textures unfiltered (no bilinear, trilinear or anisotropic filtering), its colour targets 8 bits a channel (8-bit colour, as the rgba8f patch makes them; the colour grading then steps between its table's entries too, where it is smooth without it), and the 3D at its own Resolution - Native (the Resolution setting, at most the window's), 480p, 360p or 240p, each as wide as the window's shape gives - brought up to the window pixel for pixel, with no smoothing. Upscaling and the anti-aliasing stand aside while it is on. Old-hardware sound (off by default) makes everything the game plays sound like an old console's sound chip: PS1 (22050 Hz, 12 bits, muffled above 10 kHz), Crunchy (11025 Hz, 8 bits, unfiltered) or Custom (a rate and bits of its own). Each option changes in the game (F9). |
| `debug_menu` | Restores the developers' debug menu (`` ` `` opens it), in English or the developers' Japanese, and makes the debug font it draws with from the public-domain X11 fonts k14 and 7x14, so nothing has to be downloaded. Cheating against other players with it gets an account banned; trying it out online with friends is fine. |

The gameplay plugins - all five - announce themselves in the session's
ruleset, so the server keeps their players' messages, bloodstains and
statistics apart from those of players without them. Lo-Hit and Bloodborne PS1
change only the picture: they are not in the rules, and their players meet
everyone.
