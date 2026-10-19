# Modding

bbhost never modifies the game files. Every change is layered on top at run
time, from a few places:

| What | Where | Needs code? |
|---|---|---|
| Replacement files (menus, text, textures, layouts) | the mods folder | no |
| Param values (weapons, enemies, item lots, shops) | `<mods>/params/*.toml` | no |
| Byte patches to the game's code | `patches/*.toml` | no |
| Shaders, as HLSL source | `<data>/bbhost/shaders/*.hlsl` | no (HLSL) |
| Anything else | a plugin ([plugins.md](plugins.md)) | yes |

## The mods folder

`paths.mods` in `bbhost.toml` (default `<data>/mods`) is checked before the
game dump for every file the game opens. A file at
`<mods>/dvdroot_ps4/<path>` replaces the game's `dvdroot_ps4/<path>`. The log
lists the files served from it (`FS overlay: ...`).

Files are matched by their path as the game requests it, so a replacement
keeps the original's name and container format (`.dcx` stays compressed).

## Params

The game's numbers live in 62 param tables: weapons, armour, enemies, attacks,
magic, item lots, shops, special effects, the lock-on camera and more. bbhost
reads the field definitions the game ships (`dvdroot_ps4/paramdef`), so fields
are named the way FromSoftware named them.

Put one TOML file per table in `<mods>/params/<TableName>.toml`:

```toml
# <mods>/params/EquipParamWeapon.toml - the Saw Cleaver is row 1000000
row.1000000.correctStrength = 250
row.1000000.weight = 0.5

# <mods>/params/LockCamParam.toml - every row
row.all.camDistTarget = 5.5
row.all.chrLockRangeMaxRadius = 30

# An array element is row.<id>.<name>[<i>]. Where no name fits, use a raw
# type and byte offset: f32 i32 u32 i16 u16 i8 u8.
row.1000000.f32_0x10 = 1.5
```

Table names are the ones the log prints (`params: slot N <TableName>: <TYPE>,
R rows`). To see a row before changing it:

```sh
BBHOST_PARAM_DUMP="EquipParamWeapon:1000000,LockCamParam:0" ./bbhost
```

This prints every field with its value, type and the game's own description.
Overrides apply whenever a table loads, including when the game reloads one.
A table whose rows do not match its definition is refused with a log line
instead of being written wrongly.

`tools/bbparam.py` reads and searches the param tables of a dump from the
command line (set `BBHOST_APP0` to the game folder).

## Lighting and post-processing

Per-area lighting, fog, tone mapping and post effects are data in
`param/drawparam/*.gparam.dcx`. `tools/gparam.py --list <file>` shows a
file's parameters; `tools/gparam.py --set "Name=value" <dump> <mods>` writes
edited copies of every file into the mods folder.

## Shaders

bbhost turns each of the game's shaders into HLSL on your machine and
compiles it with DXC ([rendering.md](rendering.md#hlsl)). You can edit
that HLSL and have the game use your version.

1. Write out the HLSL. Start the game once with `BBHOST_HLSL_DUMP=1`. Each
   shader the game uses is saved to `<data>/bbhost/hlsl-dump/`, named after
   the shader and its stage. For example,
   `GXGuiShader_Tex.ppo@6d921b0b.ps.hlsl` is the pixel shader of
   `GXGuiShader_Tex.ppo`: its name in the game's shader bundles, then GX's
   name for it.
   - The game's own built-in shaders (the menus' Scaleform, YEBIS) have only
     GX's name, as in `38d65b32.ps.hlsl`.
   - The stages are `ps`, `vs`, `hs`, `ds` and `cs`.
   - Beside each file is the lift it came from (`.lift.spv`).
   - The first line names the shader again; the second gives this build's lift
     hash.

   So `ls <data>/bbhost/hlsl-dump/GXFlver_Sword*` finds the weapon shaders. A
   file is written the first time the game compiles or loads that shader, so
   play through the places you want first.

   One shader can be compiled more than one way: a variant with a clip plane,
   another input layout or another depth test. The first variant met gets the
   plain name. Each other one gets its lift hash after a `~`, as in
   `GXGuiShader_Tex.ppo@6d921b0b.ps~7de085b59d21219a.hlsl`.
2. Copy the files you want to change into `<data>/bbhost/shaders/`. Keep
   each file's name, and edit the copy.
3. Start the game. The log lists the edited files it found (`render: HLSL
   overrides: N edited shaders in ...`). The first time a shader is used,
   its edited file is compiled, and the log says whether it was taken
   (`render: HLSL override <file> compiled and taken for <shader>`).

For each shader, the game looks for an edited file in this order:

1. `<name>.<stage>~<hash>.hlsl`: that one variant;
2. `<name>.<stage>.hlsl`: every variant of the shader in that stage whose
   inputs and outputs match the file. A variant that doesn't match keeps
   its own shader, and the log says so;
3. `<hash>.<stage>.hlsl`: a file named only by this build's lift hash. This
   is the fallback, for a shader whose name is not known.

An edited shader is refused, and the game uses its own version of that
shader, when:

- DXC cannot compile it (the log shows DXC's error);
- it declares other inputs, outputs, resources or bindings than the
  original;
- its SPIR-V does not validate.

Nothing else is checked. An edited shader may compute anything with what it
reads.

Things to know:

- **Names survive updates; hashes don't.** A file named after the shader
  (`<name>.<stage>.hlsl`) keeps working after a bbhost update and on another
  machine: both parts come from the game's own files. A lift hash
  identifies the shader as translated for your GPU and this bbhost version:
  the device's features (half-precision rounding, run-time sample offsets)
  and the translator are part of it. So `~<hash>` variant files and
  `<hash>` fallback files stop matching after an update; write out a fresh
  set then. A file that no shader matches does nothing.
- **An update can change the shader's interface.** If the update changes the
  inputs or outputs the game's own version declares, your file is refused,
  the log says why, and the game uses its own until you edit it again.
- **When changes take effect.** Edits are read at the start, so restart the
  game to try a change.
- **The off switch turns this off too.** Edited shaders apply only while the
  HLSL path is on. With Shaders set to Lifted (HLSL Shaders off in the game's
  PC Graphics), or `BBHOST_HLSL=0`, they are ignored.
- **Don't share the files.** They are generated from the game's own shaders,
  so they stay on your machine and are never published.

## Draw and frame hooks (planned)

Not in the plugin API yet: this is the design that GX's draw methods and
command lists, now source ([decomp.md](decomp.md#gx-srcdecompgx)), make
possible. GX is the engine's Direct3D 11 layer - the game names the
interfaces it emulates, `ID3D11DeviceContext`, `ID3D11CommandList` and the
state objects - so the hooks speak its terms.

- **Per draw and dispatch.** Called as the game issues each draw, dispatch or
  state commit, before bbhost takes it, on the thread that records it. The
  callback sees the call - its kind and arguments - and the state it draws
  with as the engine's objects: the shaders (named as the shader dump and the
  HLSL overrides name them), the textures and buffers bound, the render
  targets and depth target, the blend, depth-stencil and rasterizer states.
  It can let the draw go as it is, drop it, change what it draws with
  (another shader, texture or blend state, for that draw), or issue draws of
  its own next to it. One state object can also be swapped for every draw at
  once, where the output merger applies it.
- **Per pass.** Called as a recording thread finishes a command list and as
  the frame executes the lists, in order. Each list carries the targets it
  began and ended with, so in order they are the frame's passes: what wrote
  which target, and what read it later. A plugin can name the passes,
  capture a pass's targets, run a pass of its own between two of the game's
  (an effect after the scene and before the menus), or leave one out.
- **Per frame.** Called where a frame ends, in the swap chain's present:
  how many frames the game may run ahead of the display (two display
  buffers, so at most two flips in flight) and what it waits for before it
  hands a frame over are decided there. A plugin can run a pass of its own
  over the final image, capture it, or pace frames another way.
- **Per post-processing parameter.** Called as the game hands each frame's
  post-processing parameters to YEBIS, the post-process library, before the
  chain runs: PPFX's 69 setters ([decomp.md](decomp.md#yebis-srcdecompyebis)),
  for the scene's projections and luminance scale, auto exposure, tone
  mapping, chromatic aberration, the colour matrix (colour grading), the
  glare (bloom), light shafts, depth of field and auto focus, motion blur,
  the feedback (afterimage) and post-process antialiasing. The callback sees
  each value as the game set it for that frame and can change it before
  YEBIS takes it: turn an effect off, scale the bloom, move the focus,
  change the grading.

Hooks cost nothing while no plugin registers one. A registered one runs for
every draw on the engine's render threads, so it must be fast, thread-safe
and never wait. YEBIS's post-processing passes are covered as the D3D11
calls its own context takes (the targets each pass binds, its draws, its
flushes) and where they reach Gnm, the whole chain as the two calls the
engine runs it with (`ppfxBindFramebuffer`, `ppfxApplyEffects`), and its
parameters as the setters the game hands them through; the Scaleform menus
are not, until that code is source too.

## Text

Message banks (`msg/<lang>/*.msgbnd.dcx`, FMG files) are read through the mods
folder as well. `src/engine/menu_assets.cpp` shows how bbhost adds its own
entries to the game's banks at start, in every language.

## Patch manifests

A byte patch is a TOML file. bbhost reads every `*.toml` in these folders, a
later file replacing an earlier one with the same name:

1. `<exe dir>/patches` (the release package)
2. `<exe dir>/../patches` (a build tree: `build/bbhost` reads the repository's `patches/`)
3. `<mods>/patches` (a mod's own)

```toml
[patch]
name = "skip-intro"
eboot = "941f887a562aae054fac35af8cc8f27cf075f3d4cc2e029fb5ae2a663aaa5ae7"
description = "the three company logos at startup are skipped"
option = "startup.skip_intro"      # optional: a setting that switches it
enabled = true                      # optional, default true; false: off until turned on
gameplay = false                    # optional: true for a patch that changes the game (online rules)
order = 49                          # optional: its place in the list it came from (later wins)
# group = "analog-run-range"        # optional: variants of one patch, one on at a time

[site.logo-fromsoft]
address = "0x04d99138"              # Binary Ninja address (image at 0x400000)
expect = "4c 00 6f 00"              # the bytes that must be there
bytes = "00 00 00 00"               # the bytes written, same length
kind = "code"                       # optional: "code" (default) or "data"
yields = ["other-patch"]            # optional: left alone while one of these is on
```

- `eboot` is the SHA-256 of the decrypted eboot the addresses belong to. A
  file for another build is skipped.
- Every site's `expect` is checked before anything is written; one mismatch
  refuses the whole file. A site that already holds `bytes` counts as done.
- Manifests apply in `order`, then by name, and a site is checked against the
  eboot's own bytes under anything an earlier manifest wrote: where two write
  the same bytes, the later one's stand, as shadPS4 applies a patch list. Kyo's
  ports carry their place in his list (`Bloodborne.xml`).
- Where a manifest changed a value bbhost's own frame-rate conversion also
  writes (a constant, a redirected load's), the conversion leaves it as the
  manifest wrote it and converts the rest.
- `gameplay = true` marks a patch that changes the game rather than how it
  looks or loads (how far the player runs, the rally window, a debug move).
  While one is on, the session's online rules (`X-BBHost-Ruleset`, as for
  gameplay plugins) carry `patch.<name>`, so the server keeps that player's
  bloodstains, messages and statistics apart from vanilla players'. A mod's
  patch that changes the game must say so; the log names the rules at start
  (`plugin: this session's online rules`).
- `kind = "code"` restores the page to read-only and executable afterwards;
  `"data"` leaves it writable.
- `BBHOST_PATCHES=0` skips all manifests. The log has one line per file:
  written, skipped or refused.

The setup window's Patches tab lists every manifest with a switch. A patch
that follows an `option` is switched by that setting; any other is turned off
with `<name> = false` under `[patches]` in `bbhost.toml`, and one whose
`[patch]` says `enabled = false` is off until `<name> = true` there - the tab
writes both. Turning on one patch of a `group` turns the others off; a
`bbhost.toml` with two of a group on applies the later one in `order`.
Defaults puts every switch back to its file's `enabled`.

Two patches may write the same bytes differently (the resolution patches'
fixed sizes and the HD Motion Blur ones, say). A site that names the other
patch in `yields` steps aside while that one is on: it is neither checked nor
written, and the log line says how many sites were left to which patch.

Shipped patches:

| File | Effect |
|---|---|
| `skip-intro.toml` | skips the company logos and the "Exit Game may not have been selected" warning (illusion's 1.09 patch) |
| `fov-uncap.toml` | removes the 38-48 degree limit on the camera's field of view |
| `quick-reentry.toml` | loads after a death or lamp travel end when the area is ready instead of after 12 seconds (on by default) |
| `ingame-post-processors.toml` | three resource post-processors during play instead of two: streaming settles sooner, with more hitches meanwhile (off by default) |
| `deflate-level.toml` | the game's run-time compression at level 1 instead of 9, removing a periodic ~25 ms hitch |
| `old-hunters.toml` | makes the game treat itself as The Old Hunters edition, so the DLC is available (Kyo's 1.09 patch) |
| `debug-camera.toml` | restores the debug free camera (Lance McDonald's 1.09 patch): hold Action and press L3 to cycle its modes. Off by default; it replaces a debug-menu test step, which then must not be used |
| `low-res-aux-targets.toml` | the low-resolution renders of Kyo's shadPS4 resolution patches at every resolution: motion blur's velocity map and its depth at 640x360, the quarter-size depth at 320x180 (off by default: at 1920x1080 the game's own velocity map is 960x540). Not the patches' 80x45 glow buffer, which blacked out the lit world where glow is on |
| `screen-space-reflections.toml` | Enable Screen Space Reflections (enable only, keeps res), Kyo's patch (off by default) |
| `disable-misc-renders.toml` | Disable Misc. Renders, Kyo's patch: four of the scene's extra targets are not made (off by default) |
| `disable-dynamic-light-shadow.toml` | Disable Dynamic Light Shadow, Kyo's patch (off by default) |
| `hd-motion-blur-360p.toml`, `hd-motion-blur-4k.toml` | HD Motion Blur at 640x360 or 3840x2160, Kyo's patches (off by default; one at a time) |
| `fxaa2-gs-dlaa-gs.toml` | FXAA2_GS + DLAA_GS, Kyo's patch: both YEBIS anti-aliasing filters (off by default) |
| `glow-blur.toml` | Glow: force on + Blur 12.5/12.5/30, Kyo's patch (off by default) |
| `rally-window.toml` | Rally Window 2x + Stacking (6 s before decay), Kyo's patch (off by default; changes the game) |
| `performance-v3-decals-on.toml`, `performance-v3-velomap.toml` | Performance Patch v3, Kyo's, in its Decals On and Velomap + Camera Blur variants (off by default; one at a time) |
| `debug-dash.toml` | Restore Debug Dash (stagvant): L3 on and off, L2 up, L1 down, R2 through walls (off by default; changes the game) |
| `analog-run-range-10.toml` ... `-40.toml` | +10% to +40% analog run range, Kyo's (off by default; one at a time; changes the game) |
| `bigger-weapons-1-2x.toml` | Bigger Weapons 1.2x, Kyo's patch (off by default; changes the game) |
| `movement-roll-1-1x.toml` | Movement + Roll-Dodge 1.1x, Kyo's patch (off by default; changes the game) |
| `aerial-vectors-480.toml` | Aerial Vectors 1/480 fps, Kyo's patch: the momentum's per-frame value fixed at its 480 fps one (off by default; changes the game; bbhost follows the frame time without it) |

Patches whose bytes depend on settings, and anything that needs a hook rather
than fixed bytes, are written in code or as plugins.

## Event scripts and talk scripts

The maps' event scripts (`event/*.emevd.dcx`) and the NPC talk scripts
(`script/talk/*.talkesdbnd.dcx`) can be rewritten at start from the player's
own dump and served through the overlay, so nothing is redistributed. Two of
bbhost's PC enhancements work this way and serve as examples:

- `src/engine/maiden_events.cpp` rewrites the invasion events so the late
  areas take two invaders (the five-player sessions).
- `src/engine/rebirth.cpp` adds "Rebirth in the Nightmare" to the Altar of
  Despair's talk script and drives the level-up screen from it.

## Tools

The `tools/` folder has small, independent scripts for the game's formats:
`bbparam.py` (params), `gparam.py` (draw params), `msb.py` (map layouts),
`luabnd.py` (AI script bundles), `shaderbnd.py` (shader bundles) and
`gfx_dump.py` (Scaleform menus). Each documents its options at the top of
the file. Scripts that read the game need `BBHOST_APP0` set to the folder
that contains `dvdroot_ps4`.
