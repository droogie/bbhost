# Example plugins

Plugins that show bbhost's plugin API at work, for anyone writing their own.
They are examples, not mods. bbhost's build compiles them beside bbhost
(`build/hello.so`, `build/shader_tint.so`, `build/render_probe.so`, or `.dll`
on Windows), never into `build/plugins/`, and no package carries them. So no
run lists or loads one until you copy it into a `plugins/` folder yourself.

| Example | What it shows | API |
|---|---|---|
| [`hello/`](hello/hello.c) | The basics, in C. It logs a config value, reads the game's image once it is loaded, hooks one of the game's functions and counts frames. It reads a param by the game's own name, then the character's level, echoes and an event flag, the world's flag changes, and where the player and the camera are. | 1-5 |
| [`shader_tint/`](shader_tint/shader_tint.c) | Shader overrides. It takes one of the game's shaders' own HLSL (`shader_source`), patches it and hands it back (`shader_override`), turns it on and off while the game runs, and pulses its strength through the plugin's shader data. | 27 |
| [`render_probe/`](render_probe/render_probe.cpp) | Render hooks, in C++ with only Vulkan's headers. It uses `frame_info`, `on_render` before and after the upscale with the motion, and `aa_claim`, and checks each of them on the GPU. | 27 |

[`docs/plugins.md`](../../docs/plugins.md) is the guide to the API, and
[`include/bbhost_plugin.h`](../../include/bbhost_plugin.h) is the API itself.

## Building and trying one

`hello` builds with bbhost. Build the others by their targets:

```sh
cmake --build build --target shader_tint_probe render_probe
```

A plugin needs only the header, so you can also build it on its own; the
compiler lines are in [`docs/plugins.md`](../../docs/plugins.md#a-first-plugin).

To try one, copy it into `plugins/` next to bbhost or into `<data>/plugins`.
`hello` loads at the next start. `shader_tint` and `render_probe` are opt-in,
so they also need turning on in `bbhost.toml`:

```toml
[plugins]
shader_tint = true
```

The options each one takes are in the comment at the top of its source.
