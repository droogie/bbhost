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
3. `patches` in the per-user folder (`%APPDATA%\bbhost\patches`,
   `~/.config/bbhost/patches`): your own, which no release carries and every
   copy of bbhost reads
4. `<mods>/patches` (a mod's own)

```toml
[patch]
name = "skip-intro"
title = "Skip Intro + warning message"  # optional: the name the menus show (else `name`, dashes as spaces)
eboot = "941f887a562aae054fac35af8cc8f27cf075f3d4cc2e029fb5ae2a663aaa5ae7"
description = "the three company logos at startup are skipped"
option = "startup.skip_intro"      # optional: a setting that switches it
enabled = true                      # optional, default true; false: off until turned on
gameplay = false                    # optional: true for a patch that changes the game (online rules)
order = 10                          # optional: its place in the list it came from (later wins)
# group = "hunter-size"             # optional: variants of one patch, one on at a time
# live = false                      # optional: takes effect only as the game starts

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
  the same bytes, the later one's stand, as shadPS4 applies a patch list. The
  ports carry their place in the shadPS4 patch list (`Bloodborne.xml`, ten
  apart), and the menus list them in that order; bbhost's own come after it.
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

The setup window's Patches tab lists every manifest with a switch, and so
does the Patches page of the menu in the game (F9). A patch that follows an
`option` is switched by that setting; any other is turned off with
`<name> = false` under `[patches]` in `bbhost.toml`, and one whose `[patch]`
says `enabled = false` is off until `<name> = true` there - both write it.
Turning on one patch of a `group` turns the others off; a `bbhost.toml` with
two of a group on applies the later one in `order`. Defaults puts every
switch back to its file's `enabled`.

In the game the switch also takes effect at once: the patch's bytes are
written, or the eboot's own put back, while the game runs - new code first
and the jumps to it last, code added where the eboot had zeros left in when it
goes, a site within one aligned 8-byte word written in one store. Each site is
checked first; where bbhost or another patch has changed the bytes since the
start, or bbhost's frame-rate conversion left a value to the patch, the switch
is saved and applies at the next start, and the page says so. `live = false`
marks a patch that only takes effect as the game starts (render targets sized
as they are made, code bbhost itself rewrites while it runs): the page saves
its switch for the next start and leaves the code alone.

Two patches may write the same bytes differently (the resolution patches'
fixed sizes and the HD Motion Blur ones, say). A site that names the other
patch in `yields` steps aside while that one is on: it is neither checked nor
written, and the log line says how many sites were left to which patch.

Shipped patches, in their order. Left out are the list's patches bbhost
does itself - its resolutions and FPS++ (the Resolution and Frame rate
settings), the light grids, Restore Change Appearance (PC Enhancements), the
Model LODs and Disable AA, SSAO, DoF, Bloom, Motion Blur and Chromatic
Aberration (PC Graphics), Increased camera distance (Camera distance),
Increased Graphics Heap Sizes (bbhost grows that heap for the render size),
Faster loading screens (bbhost runs loading screens unpaced) and Skip Intro
(Skip company logos, built into bbhost) - and the ones a release does not
carry: the debug patches (camera, dash, menu, Enemy Control), the rally and
weapon-size changes, and the emulator workarounds and extras (FMOD Crash Fix,
the Intel SFX workaround, Disable HTTP Requests, Disable Facegen, Enable TAA,
Bookmark and Capture outputs, the low-res motion blur buffers). A player who
wants one keeps its file in their own patches folder.

| File | Patch |
|---|---|
| `performance-v3-decals-on.toml` | **Performance Patch (Perf Increase)**: Modifies some of the game's debug parameters for better performance without affecting visuals (Kyo). (on by default; one of performance v3; at the next start) |
| `performance-v3-velomap.toml` | **Performance Patch v3 (Velomap + Camera Blur)**: Performance Patch v3 (Velomap + Camera Blur, Decals On) (Kyo) (off by default; one of performance v3; at the next start) |
| `disable-misc-renders.toml` | **Disable Misc. Renders**: Disable Misc. Renders - Kyo's patch: four of the scene's extra targets are not made, nor their passes drawn (off by default; at the next start) |
| `rgba8f-color-space.toml` | **rgba8f color space (Perf Increase)**: Affects visuals: the rgba16f colour targets become rgba8f. Less GPU use, for older or weaker hardware. (off by default; at the next start) |
| `disable-dynamic-light-shadow.toml` | **Disable Dynamic Light Shadows (Perf Increase)**: Disable Dynamic Light Shadow - Kyo's patch: dynamic lights cast no shadows (faster) (off by default) |
| `fxaa2-gs-dlaa-gs.toml` | **Better AA**: FXAA2_GS + DLAA_GS - Kyo's patch: both anti-aliasing filters (off by default) |
| `screen-space-reflections.toml` | **Enable Screen Space Reflections**: Screen space reflections on, with the depth renders back at the game's default resolutions. It loads last, so other patches' render changes give way to it. (off by default; at the next start) |
| `text-scale-50.toml` | **50% Text scale**: Smaller text, which looks good at higher resolutions. (off by default) |
| `intel-tonemap-fix.toml` | **Intel Black Tonemap Fix**: Fixes the black tonemap colours, most noticeable in the DLC's areas, when using an Intel CPU. (off by default) |
| `ds1-physics.toml` | **DS1-like physics**: Corpses and objects can be flung around, as in Dark Souls. (off by default; one of object physics) |
| `lower-object-physics.toml` | **Lower object/corpse physics**: Corpses resist being moved by the player or anything else. (off by default; one of object physics) |
| `disable-camera-auto-rotation.toml` | **Disable Camera Auto Rotation via Movement**: The camera no longer turns by itself as the character moves; turning it by hand and lock-on still work. (off by default; at the next start) |
| `easier-run-range.toml` | **Easier run range**: The stick runs from less of its travel: about two-thirds of the way out instead of 90%. (on by default) |
| `l3-run-jump.toml` | **L3 run-jump**: While running (holding Circle), L3 jumps too: the run-jump a second press of Circle makes. Circle still jumps. (off by default) |
| `auto-refill-from-storage.toml` | **Auto-Refill From Storage on Any Area Refresh/Reload**: Eligible consumables refill from your storage whenever an area is refreshed or reloaded, taking from storage as the game does, without dying or going back to the Hunter's Dream. The game's own auto-refill; no free items. (off by default) |
| `smaller-hunter.toml` | **Smaller Hunter**: A smaller hunter (smol). (off by default; one of hunter size) |
| `smallest-hunter.toml` | **Smallest Hunter**: The smallest hunter (teeny). (off by default; one of hunter size) |
| `unlock-game-region.toml` | **Unlock Game Region (DISABLES ONLINE PLAY)**: Unlocks the game's region for more language options. Online play does not work while it is on. The Cross and Circle buttons are not swapped. (off by default; at the next start) |
| `zero-gravity.toml` | **Zero-Gravity Physics**: Havok's gravity at zero: objects and corpses float. (off by default; one of gravity) |
| `anti-gravity.toml` | **Anti-Gravity Physics**: Havok's gravity reversed: objects and corpses fall upward. (off by default; one of gravity) |
| `hd-motion-blur-360p.toml` | **HD Motion Blur 360p (640x360)**: HD Motion Blur 360p (640x360) - Kyo's patch (off by default; one of hd motion blur; at the next start) |
| `hd-motion-blur-1080p.toml` | **HD Motion Blur 1080p (1920x1080)**: Motion blur's velocity map is half the render resolution by default; this makes it 1080p. Uses more video memory. (off by default; one of hd motion blur; at the next start) |
| `hd-motion-blur-1440p.toml` | **HD Motion Blur 1440p (2560x1440)**: Motion blur's velocity map is half the render resolution by default; this makes it 1440p. Uses more video memory. (off by default; one of hd motion blur; at the next start) |
| `hd-motion-blur-4k.toml` | **HD Motion Blur 4K (3840x2160)**: HD Motion Blur 4K (3840x2160) - Kyo's patch (off by default; one of hd motion blur; at the next start) |
| `lower-specific-renders.toml` | **Lower specific renders (Perf Increase)**: Lowers the resolution of some renders (the velocity map, glow, the scaled depth): faster, with no visual difference unless Enable Screen Space Reflections is on. (off by default; one of low res renders; at the next start) |
| `old-hunters.toml` | **The Old Hunters**: The Old Hunters is available: the game takes itself for the edition that includes it (on by default; at the next start) |
| `quick-reentry.toml` | **Quick re-entry**: A load after a death or lamp travel ends when the area is ready, not after a 12-second minimum (follows `[loading] quick_reentry`, on by default) |
| `ingame-post-processors.toml` | **All post-processors in play**: All three resource post-processors run in play, not two (follows `[streaming] all_post_processors`, off by default) |
| `deflate-level.toml` | **Deflate level 1**: The game's run-time deflate compresses at level 1, not 9 (a main-thread hitch every ~50 s) (off by default) |
| `fov-uncap.toml` | **Uncapped field of view**: The field of view is no longer capped at 48 degrees (off by default) |
| `glow-blur.toml` | **Glow: force on + Blur**: Glow: force on + Blur 12.5/12.5/30 (all time slots) (Kyo) (off by default) |

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
