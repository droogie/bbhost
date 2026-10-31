#pragma once

#include <cstdint>

// The follow camera, from LockCamParam - one row per area's camera setup,
// which the camera's update (sub_183ac60) reads every frame and eases toward:
// its distance (camDistTarget), the height it looks at (chrOrgOffset_Y) and
// its vertical field of view (camFovY, 43 degrees in nearly every row). The
// settings scale those fields in the table the game has loaded.
// engine/camera.cpp.

struct ElfImage;
// Lifts the camera's 48-degree ceiling on camFovY. Only after the eboot's
// hash has been checked, like every address-based patch.
void camera_install(ElfImage* image);

// Called for every view (engine/graphics_patch.cpp's render-view hook). It
// has work only when the settings changed (their listener, registered by
// camera_install), when the search for the table finished, or when the param
// repository (re)loaded it: then it finds the table if need be and writes the
// rows. Otherwise it reads one flag.
void camera_tick();
// The param repository's walk (engine/params.cpp) saw LockCamParam loaded or
// reloaded at `file` (the .param as the game keeps it): the next view writes
// its rows, if the settings ask for anything.
void camera_params_changed(std::uint64_t file);
// The camera zoom (host/bindings.h: the wheel and the middle button by
// default): `steps` notches further out (closer when negative), 20% of the
// distance each, from half the game's own distance to six times it; or with
// `reset` back to the Camera distance setting's. A multiplier on that
// setting, not saved; the camera eases there as it does for the setting.
void camera_zoom(int steps, bool reset);
