#pragma once

// The world held still (plugin API 29's world_pause and world_step): the game's
// own pause, the one it holds while it leaves a world.
//
// The in-game update (sub_1d3ac10, the in-game step in rdi) reads two bytes of
// the step as it starts (0x1d3b125): +0x150 holds the world - no physics, AI,
// bullets or events, the characters only posed again where they stand
// (sub_1d18880) - and +0x151 the effects and the cloth with it (SprjSfx +0x8c,
// SprjCloth +0x40). Its camera section runs either way, so the view - on_view's
// included - is drawn every frame. The game sets +0x150 itself while it leaves
// a world, with +0x153; the debug camera's community patch set it to fly a
// camera over a held world. A hook at the update's entry writes both bytes
// before they are read, for as long as any plugin asks:
//  - it clears only what it set itself, and never while +0x153 holds them;
//  - a load makes a new in-game step, and the first update of a new one lets
//    every hold go, so a world never starts held;
//  - the held update reads the step's camera root +0xd0 (the debug free
//    camera, which the game makes with the root) with no check, so a world
//    whose root has none is not held (logged).

#include <cstdint>
#include <string>

struct ElfImage;

// The hook on the in-game update's entry. Only after the eboot's SHA-256 check.
void world_pause_install(ElfImage* image);
// A plugin's hold, by its name: asked for (on) or let go. False when the hook
// is not in (another eboot, a prologue that differs).
bool world_pause_set(const std::string& owner, bool on);
// While `owner` holds the world: `updates` more updates run (one a frame), then
// it is held again. False when `owner` holds none.
bool world_pause_step(const std::string& owner, std::uint32_t updates);
// The world was held at its last update.
bool world_pause_held();
// Every hold of `owner` let go (the plugin turned off).
void world_pause_drop(const std::string& owner);
