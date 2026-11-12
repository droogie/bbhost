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

`examples/plugins/hello/hello.c` is the example, and `examples/plugins/` holds
the others. bbhost's own build compiles it as `build/hello.so` (or
`hello.dll`), beside bbhost and not in `build/plugins/`, so no run loads or
lists it until you copy it into a `plugins/` folder. A plugin needs only the
header:

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
appended. The current version is 28 (`BB_PLUGIN_API_VERSION`).

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
- `on_camera_target(fn, user)` (v28): the recompiled follow camera's target.
  Registered once, from `bb_plugin_image`; on the game thread at each of the
  camera's updates `fn` gets the character whose controls the camera uses and
  returns a live `ChrIns` to follow instead, or NULL to leave it. Only the
  model and the moving floor's position are followed elsewhere, never the
  input, and a plugin keeps no pointer to an actor that may die. A spectator
  camera this way; where the game's own follow camera runs instead of the
  recompiled one, a plugin still needs its own camera hooks. One plugin owns
  it: 1 outside image setup, for NULL, or once taken; 0 otherwise.

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
  `fn(cc, user)` runs once a frame on the GX worker that records the scene's
  render queue's first item at or after `pass` - a sort key's category and
  sub-category, `(bits 22-27 << 6) | bits 16-21` (`BBHOST_SCENE_CATEGORIES=1`
  logs those a frame holds) - in a GX command context of its own with that
  item's pass set up, queued just ahead of it (the worker's chunk is cut
  there): what it records draws into the scene's targets after the passes
  before it and before that pass does. One pass for each `fn` and `user`
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
  11 bits. Lo-Hit draws the collision this way (always, on a host with it).
- `gbuffer_ambient(level)` (v19): `BB_DRAW_GBUFFER` draws' indirect light,
  which the map gets from its lightmaps (0.03 by default).
- `audio_crush(rate_hz, bits, lowpass_hz)` (v21): old-hardware sound. The
  game's final mix, as it reaches the device, low-passed at `lowpass_hz` (one
  pole), held down to `rate_hz` (sample-and-hold, kept across buffers so their
  boundaries don't click), rounded to `bits` a sample with no dither (quieter
  than half a step is silence) and, after a hold, smoothed by a second-order
  low-pass at 0.45 of `rate_hz`, as a DAC's output filter took out the steps'
  images above the rate; 0 turns each off, and all three 0 costs nothing. Any thread, from the next
  buffer. The Bloodborne PS1 plugin's Sound.
- `view_cutaway(tan_x, tan_y, near_z, radius)` (v22): a cutaway for a high
  camera. The camera's draws are clipped (a clip distance added to their last
  stage before the rasterizer) within `radius` metres of the view's axis and
  nearer than `near_z` along it - the roof, ceiling or near wall between the
  camera and the character it centres. Camera-relative, so it is set when the
  distance or field of view change, not as the camera moves; each new set
  makes the camera draws' pipelines again.
- `view_cutaway2(const BbCutaway*)` (v23): the cutaway per pixel, faded. The
  map's own draws into the G-buffer (never the characters, the objects or
  their parts: the G-buffer's stencil marks them 8 and 12, the map 9) leave
  out, in their pixel stage, what lies within `radius` of the view's axis -
  `widen` metres wider for each metre nearer the camera than the point looked
  at, a cone opening toward the camera - more than `height` metres above that
  point and more than `margin` in front of it, faded over `feather` metres at
  those edges by a screen-door dither (the G-buffer has no blending). `pitch`
  and `distance` place the point; the field of view and the depth's decode
  come from the drawn view's own block. Turning it on makes the map's
  pipelines again once; since v26 the numbers (and the view's field of view
  and depth) are read live from a small buffer, so a new set each frame - a
  zoom easing in - builds nothing (before, each new set made them all again:
  the Isometric's zoom stutter). NULL or radius 0 turns it off. The Isometric plugin's Cutaway; it costs a third of a
  millisecond of GPU time a frame at 1080p.
- `view_cutaway_objects(float start)` (v24): `view_cutaway2`'s cone for the
  draws the stencil marks 8 and 12 too, its tip `start` of the way from the
  point looked at toward the camera (0.5 halfway); read live since v26, so it
  may change each frame. Those marks hold the characters and the map's
  objects alike - neither the stencil, the vertex shader nor what moves tells
  them apart - so a character standing into the cone above its tip is cut
  with the objects. Below 0, the default, they are never cut. The Isometric
  plugin's Cut objects (its tip 1.4 m over the hunter's feet); no GPU time to
  measure beyond the map's cut.
- `scene_dither(bits)` (v25): the finished picture - the 3D and its post
  effects, before the HUD and the menus are drawn over it - rounded to `bits`
  a channel through the PS1 GPU's 4x4 ordered dither, a cell of the pattern a
  pixel of the 3D (`scene_scale`'s lines). 5 is the PS1's 15-bit colour; 0,
  the default, off. The Bloodborne PS1's Dithering.
- `camera_zoom(int* steps, int* reset)` (v26): the player's Camera Zoom
  presses since the last call - the wheel and the middle button by default,
  as bound, in the world only: `steps` notches out (in when negative), `reset`
  1 for Camera Zoom Reset. From the first call they are the plugin's and the
  game's camera no longer zooms. The Isometric's zoom.
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

**Rendering effects** (v27)

- `frame_info`, `on_render` and `aa_claim`: the frame as bbhost draws it,
  compute work of the plugin's own recorded into it before and after the
  upscale, and who does the anti-aliasing. See
  [Rendering effects](#rendering-effects-v27).

**Shaders** (v27)

A plugin can put HLSL of its own in place of any of the game's shaders, as an
edited file in `<data>/bbhost/shaders` does ([modding.md](modding.md#shaders)):
the same names, compiled and checked the same way, and applied while the game
runs.

- **Names.** A shader is named as `BBHOST_HLSL_DUMP=1` names its file, less
  `.hlsl`: `GXGuiShader_Tex.ppo@6d921b0b.ps` for every variant of that pixel
  shader whose inputs and outputs the source declares,
  `GXGuiShader_Tex.ppo@6d921b0b.ps~7de085b59d21219a` for one variant, and a
  lift's hash, `7de085b59d21219a.ps`, for a shader whose name is not known.
  The stages are `vs`, `ps`, `hs`, `ds`, `gs` and `cs`.
- `shader_override(name, shader, hlsl)`: the source for a shader (UTF-8, at
  most 4 MiB); `NULL` gives the shader back to the game. DXC compiles it for
  each variant of the shader met so far, off the thread that draws, and then
  the shader's next draw uses it: its pipelines are built again, once. Going
  back to a source drawn with before takes the pipelines built for it and
  compiles nothing, and the same text again changes nothing. A source is
  taken where it declares the stage's interface - the dumped file's bindings,
  inputs and outputs - and its SPIR-V validates; elsewhere the shader keeps
  its own. A value that changes goes through `shader_data`, not a new text:
  each new text is a compile and new pipelines.
- `shader_folder(name, dir)`: every `<shader>.hlsl` in a folder, relative to
  the plugin's own folder (`plugin_dir`) or absolute, as `shader_override`
  takes a text; `NULL` takes the plugin's folders away. The plugin's own texts
  come first, then its folders in the order it added them. It returns how many
  files it found, or -1 for no such folder.
- `shader_state(name, shader, why, n)`: what became of it, and why:
  `BB_SHADER_WAITING` (its shader not met since, compiling, or the plugin is
  off), `BB_SHADER_TAKEN`, `BB_SHADER_REFUSED` (DXC's error, citing the
  source's own line; another interface than the shader's; invalid SPIR-V; a
  pipeline holding it that the driver refused - that pipeline draws with the
  game's shader), `BB_SHADER_SHADOWED` (a file of the same name in the
  player's folder, or an earlier plugin's, comes first) or `BB_SHADER_OFF`.
- `shader_source(shader, &hlsl, &size)`: the HLSL this build prints for a
  shader, as the dump writes it - the first variant met this run, or the one a
  `~<lift hash>` name gives - so a plugin can patch the game's own source
  instead of shipping whole copies, which an update that changes the shader's
  interface would refuse. It returns 1 until the game has built a stage of
  that shader in this run (ask again from `on_frame`), and 2 when overrides do
  not apply. The text is printed for the player's GPU, whose features change
  it (half-precision values among them), so a file made from another GPU's
  dump can be refused there: patch what `shader_source` gives on the player's
  machine.
- `shader_data(name, offset, data, size)`: numbers of the plugin's own that
  its shaders read with no binding: `BB_SHADER_DATA_BYTES` (64 KiB), zero at
  the start, written from any thread. A shader includes `bbhost.hlsli` - the
  host provides it; the dump folder has a copy - and reads it with
  `bb_data_float(offset)`, `bb_data_float2` to `bb_data_float4`,
  `bb_data_float4x4`, `bb_data_int`, `bb_data_uint` to `bb_data_uint4` (byte
  offsets: 4-aligned, 8 for two elements, 16 for three, four or a matrix,
  rows first). Each frame's draws read one copy of it, as it stood when the
  frame before had been recorded, so write what belongs together in one call.
  Changing a value builds nothing.

```hlsl
#include "bbhost.hlsli"
// ... the shader's own source, as shader_source gave it, and in its main:
    lifted_main();
    mrt0.rgb = lerp(mrt0.rgb, mrt0.rgb * float3(1.0f, 0.35f, 0.35f), saturate(bb_data_float(0)));
```

```c
float strength = 0.5f;
api->shader_data("mine", 0, &strength, sizeof(strength));  /* from the next frame */
```

The player's own file of a name comes first, then the plugin loaded first (by
file name); the one not used is logged once. A plugin turned off in game keeps
its shaders until it is on again. Overrides apply on the HLSL path only (the
Shaders setting at HLSL, the default) and to the stages it compiles: a draw
drawn with the shader's translation instead (one whose constant buffers do not
all bind) keeps the game's. A compute shader's
new source is compiled where it is first dispatched when it was not met before
(a short stall). The log's start and exit reports list each override, who gave
it and what became of it - the exit's every one set during the run, one given
back with the flip it went at; `BBHOST_SHADER_WATCH=1` reads the plugins' folders
and the player's again while the game runs; `video.shader_overrides = false` or
`BBHOST_SHADER_OVERRIDES=0` turns all of them off. Not yet: textures or
samplers of an override's own, and an override for some draws of a shader and
not others. `examples/plugins/shader_tint` is a worked example: a plugin that
tints one shader live, on and off, its strength pulsing through its shader
data (built beside the tests, in no package).

**Online rules** (v10)

A gameplay plugin is announced to the server with every request
(`X-BBHost-Ruleset`), so modded runs stay apart from vanilla ones and players
only meet others with the same rules. `set_rules` updates the plugin's entry
(for example when its seed changes); `on_world_rules` and `visitor` let a
plugin flagged `BB_PLUGIN_ADOPTS_RULES` play by a host's rules when summoned
into another player's modded world.

## Rendering effects (v27)

A plugin can draw effects of its own into the frame - ambient occlusion,
fog, a colour pass - from the frame bbhost describes and the images it hands
over, in the Vulkan command buffer the frame is recorded into. It needs only
Vulkan's headers (`#define VK_NO_PROTOTYPES` before `vulkan/vulkan.h`): every
function comes from the device it is given, so it never links the Vulkan
loader. `examples/plugins/render_probe` is a working example: it registers
both hooks, writes a marker into each, and checks everything below on the
GPU.

### The frame: `frame_info`

`frame_info(&info)` fills a `BbFrameInfo` with the last 3D frame described -
inside a hook, the one being recorded - from any thread. Set `info.size` to
`sizeof(info)` first: the host fills as much as both of you know and sets
`size` to that, and a newer host only ever appends fields. From the first call
on (or the first `on_render`) the host describes every 3D frame; before that
it costs nothing, and the first call returns 1.

- `frame` counts the 3D frames from 1 (a loading screen is none); `flip` is
  the host's flip count, as its log has it.
- `render_width` x `render_height` is the 3D's size - the scene's, the depth's,
  the motion's and the glow's; `output_width` x `output_height` the picture's,
  larger than the render size when an upscaler draws the 3D smaller (an
  Upscaling quality below Native AA, or a Resolution below the window's).
- `method` is what takes the 3D to the picture this frame: `BB_UPSCALE_NONE`
  (Upscaling Off, an upscaler this machine cannot run, or a plugin's own
  anti-aliasing), `BB_UPSCALE_DLSS`, `BB_UPSCALE_FSR` or `BB_UPSCALE_TAA`, and
  `quality` the Upscaling quality the upscaler is told (`BB_QUALITY_*`; work
  from the sizes, which are exact).
- `jitter_x`, `jitter_y`: the render pixels this frame's camera draws were
  moved by - +x right, +y down - exactly what the upscaler is told; 0 when none
  runs.
- `now` and `prev`: this frame's camera and the last frame's, as drawn (an
  `on_view` plugin's change included) and without the jitter. Each is a
  `BbFrameCamera`: a `BbCamera` - the basis (right, up, forward) and the eye,
  which in a row are the view-to-world matrix column by column, the vertical
  field of view, the aspect, and `near_z`/`far_z`, the view distances the
  depth's 0 and 1 stand for - and the matrices `view` (world to view), `proj`
  (view to clip), `view_proj` and `inv_view_proj` (clip to world).
- `reset` is 1 when there is nothing to blend with: the first frame, a new
  render or output size or method, no camera this frame or the last, a cut
  (`BB_FRAME_CUT`: the eye moved more than 4 m, the view turned more than 75
  degrees or its field of view changed by more than 10), or more than a
  quarter of a second since the last frame.
- `flags`: `BB_FRAME_CAMERA` (`now` is this frame's), `BB_FRAME_PREV` (`prev`
  is the last frame's; else it repeats `now`), `BB_FRAME_CUT`,
  `BB_FRAME_DEPTH_REVERSED` and `BB_FRAME_AA_PLUGIN` (a plugin does the
  anti-aliasing).

The conventions:

- Matrices are 16 floats, column-major (`m[column * 4 + row]`: GLSL's `mat4`
  and GLM's layout; HLSL's `column_major`), applied to column vectors:
  `clip = proj * view * (x, y, z, 1)`.
- The world is the game's, in metres, y up - `player_position`'s and
  `BbCamera`'s. The view: x right, y up, z the way the camera looks.
- Clip space is Vulkan's over the 3D's images: after the divide by w (the
  view's z) x runs from -1 at their left edge to +1 at their right, y from -1
  at their top to +1 at their bottom, and z is the value the depth image holds
  there: 0 at the near plane and 1 at the far one as the game draws (the other
  way round with `BB_FRAME_DEPTH_REVERSED`). The pixel (px, py) of a w x h
  image is centred at `((px + 0.5) / w * 2 - 1, (py + 0.5) / h * 2 - 1)`; the
  picture at the output size covers the same view.
- The frame's draws were moved by the jitter: their projection is `proj` with
  `m[8] = 2 * jitter_x / render_width` and `m[9] = 2 * jitter_y / render_height`.
- Angles are radians, sizes pixels, times seconds.

### The hooks: `on_render`

`on_render(name, points, fn, user)` has `fn(call, user)` called at each of
`points` of every 3D frame:

- `BB_RENDER_BEFORE_UPSCALE`: the 3D is drawn - lit, its transparency on it -
  and nothing has read it yet: the upscaler, TAA, the embers, the sharpening
  and frame generation come after. With Upscaling Off it is called all the
  same.
- `BB_RENDER_AFTER_UPSCALE`: the final picture, after all of them, before the
  post-processing (exposure, bloom, depth of field, tone mapping, grading) and
  the HUD. With no upscaler the picture is the scene itself.
- `| BB_RENDER_MOTION` asks for the frame's motion, made for the hooks whether
  or not an upscaler or frame generation needs it.

A later call with the same `fn` and `user` replaces its points; no points take
it away. `name` is the plugin's `BbPluginInfo` name: its hooks are not called
while it is off, and go if it is unloaded. Several plugins' hooks run in the
order they were registered.

`call` (a `BbRenderCall`, size first) holds the command buffer, the frame's
`BbFrameInfo`, the device (`BbRenderDevice`: the instance, physical device and
device, `get_instance_proc_addr` and `get_device_proc_addr`, the queue family
and `frames_in_flight`), and the images, each as its `VkImage`, view,
generation, format, layout, usage, size and `BB_IMAGE_WRITE` when the hook may
write it:

| Image | What it is | Size | Layout | Written |
|---|---|---|---|---|
| `scene` | the lit 3D in linear light: RGBA16F, or B10G11R11 with the Performance patch | render | `GENERAL` | before the upscale |
| `depth` | the G-buffer's depth, through a view of its depth aspect; jittered as the scene | render | read-only optimal, or `GENERAL` | never |
| `motion` | with `BB_RENDER_MOTION`: R16G16_SFLOAT, render pixels from each pixel to where its surface was in the last frame, x right, y down, without the jitter | render | `GENERAL` | never |
| `glow` | the scene's second target, the effects' glow the bloom reads, when the frame has one | render | `GENERAL` | never |
| `picture` | after the upscale: the final picture - the upscaler's RGBA16F image (a storage image), or the scene when no upscaler runs | output | `GENERAL` | after the upscale |

The scene is a render target: it can be sampled, copied and blitted, never
bound as a storage image - write your result into an image of your own and
copy or blit it in. Every writable image takes copies; check `usage` for
storage.

The contract:

- Record compute and transfer work only: dispatches, copies, blits, clears,
  and the barriers between your own steps. No render pass, draw, graphics
  pipeline or dynamic state.
- Leave every image in the layout the call gives (change one only to put it
  back before you return).
- Never submit, end the command buffer or wait for the GPU: no queue, fence,
  semaphore or query-wait calls. The host submits the frame.
- The host puts the barriers around each hook - what the frame wrote is
  visible to your compute and transfer, what you wrote is visible to
  everything after - and binds all of its own state again afterwards.
- Hooks run on the thread that records the frame, not the game's main thread,
  with the renderer's lock held: the host's GPU work waits while one runs. Keep
  it to recording; build pipelines once. From a hook, call nothing of the
  host's but `log`, `frame_info` and `on_render`.

How long things last:

- The command buffer and the frame info: for the call.
- The device: for the whole run (bbhost makes one).
- An image's handle and view: in every call while its `generation` stays the
  same - so views and descriptors of your own can be kept by generation. A new
  generation (a Resolution change makes one) is a new image.
- What a hook records may still be read by the GPU `frames_in_flight` frames
  later: free or reuse a resource your commands used no sooner.

### Who does the anti-aliasing: `aa_claim`

`aa_claim(name, claim)` says whether your plugin leaves the anti-aliasing to
bbhost or does its own; `BB_PLUGIN_AA_LEAVE` or `BB_PLUGIN_AA_TAKE` in
`BbPluginInfo.flags` is the claim it starts with, which the setup window can
read before the plugin runs.

- `BB_AA_LEAVE`: your plugin does none and moves no draws. bbhost's runs as
  the settings say - also beside a Vulkan layer that would otherwise keep its
  TAA off.
- `BB_AA_TAKE`: your plugin does it. While it is on, bbhost moves no draws (no
  jitter) and runs no TAA, FSR or DLSS, nor the game's own anti-aliasing pass,
  and draws the 3D at the Resolution, as with Upscaling Off. The Upscaling and
  Anti-aliasing rows name your plugin, in bbhost's menus and the game's; the
  player's settings are kept for when it is off.

Of the plugins that are on, the strongest claim holds: TAKE before LEAVE. If
two plugins take it over, both run their own, and the menus name the first
loaded. Claim from `bb_plugin_init` for the start, or once the game runs from
the main thread (`on_frame`, `on_option`): the 3D's size follows at the next
quiet point, as at a Resolution change.

### What it costs

With no hook registered and `frame_info` never called, nothing runs. A hook
costs a frame description, a barrier on each side, and with Upscaling Off one
camera read a frame. `BB_RENDER_MOTION` with no upscaler makes the motion as
frame generation does - the motion pass, the characters' capture - and beside
TAA one motion pass more. Each hook's GPU time shows in `BBHOST_GPU_PROFILE`
as `plugin <name> before` and `plugin <name> after`.

## From a Vulkan layer to a plugin

A mod that sits beside `bbhost.exe` as `vulkan-1.dll` and finds the camera,
the scene and the depth by watching draws breaks whenever bbhost's rendering
changes. As a bbhost plugin it gets them from bbhost, on Windows and Linux:

1. Export `bb_plugin_info` - opt-in, and `BB_PLUGIN_LIVE_SWITCH` so the
   player turns it on and off in game - and register from `bb_plugin_init`.
2. Take the device from the first hook's `call->device`, and every Vulkan
   function from its `get_device_proc_addr` and `get_instance_proc_addr`.
3. Read the camera from `frame_info`'s `now` and `prev` instead of finding it
   in the draws.
4. Take the scene, the depth, the motion and the glow from the call; keep
   your views of them by `generation` instead of finding them again.
5. Move your passes to `BB_RENDER_BEFORE_UPSCALE` (they are then upscaled and
   anti-aliased with the scene) or `BB_RENDER_AFTER_UPSCALE` (on the final
   picture).
6. Never jitter the camera: claim `BB_AA_LEAVE` and bbhost's TAA, FSR or DLSS
   runs as the player chose, or `BB_AA_TAKE` to do the anti-aliasing yourself.
7. Remove the `vulkan-1.dll` from beside `bbhost.exe`. As a plugin nothing
   needs it there, and while a `vulkan-1.dll` that is not the Vulkan loader
   sits there the upscaler bridge keeps DLSS and FSR off, and bbhost's TAA
   stays off unless a plugin leaves the anti-aliasing to bbhost.
8. Check your plugin with `BBHOST_VK_VALIDATE=1`: the Vulkan validation layer
   reports any layout or barrier it gets wrong.
9. Build the same source as a `.so` for Linux.

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
| `ps1` | Off until turned on. Bloodborne PS1: the 3D drawn as the first PlayStation drew it, the HUD and the menus left sharp at the window's size: its vertices snapped to a grid of the screen as it moves (Vertex snap: 240p, the PS1's own screen, by default; 480p, 360p, 120p or off), its textures unfiltered (no bilinear, trilinear or anisotropic filtering), its colour targets 8 bits a channel (8-bit colour, as the rgba8f patch makes them; the colour grading then steps between its table's entries too, where it is smooth without it), and the 3D at its own Resolution - Native (the Resolution setting, at most the window's), 480p, 360p or 240p, each as wide as the window's shape gives - brought up to the window pixel for pixel, with no smoothing. Upscaling and the anti-aliasing stand aside while it is on. Old-hardware sound (off by default) makes everything the game plays sound like an old console's sound chip: PS1 (22050 Hz, 12 bits, muffled above 10 kHz), Crunchy (11025 Hz, 8 bits, no low-pass going in, so it aliases) or Custom (a rate and bits of its own). Each option changes in the game (F9). |
| `shoulder` | Off until turned on. Over the Shoulder: a close third-person camera over the hunter's right or left shoulder, as Silent Hill 2's remake frames its hero - the hunter to one side, the way ahead in the middle - pulled in at walls by the game's own sphere cast, with a key (Tab by default; Z, B, H, M or a mouse side button) that swaps shoulders while playing. Only where the camera stands moves: the stick, the lock-on and the aim keep the game's turning. Always face the camera keeps the hunter in the monocular's precision-shooting state (a walk that strafes, the hunter always facing where the camera looks, the crosshair up) with the shoulder camera still placed behind them. Dodge is a slider: 0 the game's roll, 1 a sidestep in the stick's direction (the omni-roll), which wins over the Modern Movement Lite patch while the plugin is on. Distance, side offset, height, field of view and swap time are settings, all live. |
| `debug_menu` | Restores the developers' debug menu (`` ` `` opens it), in English or the developers' Japanese, and makes the debug font it draws with from the public-domain X11 fonts k14 and 7x14, so nothing has to be downloaded. Cheating against other players with it gets an account banned; trying it out online with friends is fine. |

The gameplay plugins - all five - announce themselves in the session's
ruleset, so the server keeps their players' messages, bloodstains and
statistics apart from those of players without them. Lo-Hit and Bloodborne PS1
change only the picture: they are not in the rules, and their players meet
everyone.
