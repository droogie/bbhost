#pragma once

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

// The rows are written at their source, never in a frame: when the camera
// settings change (camera_settings_changed, from the render view's check of
// the settings' serial) and when the param repository loads or reloads
// LockCamParam (camera_table_loaded, from engine/params.cpp's walk). Each
// writes only what differs from what it wrote last.
void camera_settings_changed();
void camera_table_loaded();
// BBHOST_FOV_WATCH only (a test switch): finds the table by its type name and
// watches its reads. From params_tick, on the main thread.
void camera_debug_tick();
