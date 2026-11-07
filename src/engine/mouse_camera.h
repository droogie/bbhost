#pragma once

// The mouse turns the camera the way Dark Souls III PC does (docs/running.md,
// "Mouse"): not through the right stick, but as an angle per count of the
// mouse's own movement, added to the follow camera's pitch and yaw once a
// frame. DS3 reads the DirectInput counts and turns its ChrExFollowCam by
//
//     0.2 * (sensitivity * 0.15 + 0.5) degrees a count
//
// (0.25 at its default 5), after the stick has turned it and only when the
// stick has not - with no frame time in it, no dead zone, no cap and no
// acceleration, so the same movement always turns the same angle. Bloodborne
// has the same camera class without the mouse; this adds DS3's step where DS3
// has it, in NS_SPRJ::ChrExFollowCam::Update - in our source of it when that
// is in place (decomp/camera/follow_camera.cpp), else by hooks on the game's.
// Locked on, DS3's mouse does not turn the camera but switches the target on
// a flick, which this does through the right stick the game already switches
// with.
//
// While the keyboard and mouse were used last and the mouse turns the camera,
// the camera's own turns as the character moves are held - unless the
// "Mouse auto-rotation" setting is on. A controller always has the game's own
// camera.

#include <cstdint>

struct ElfImage;

// After decomp_install: ours turns the camera when our update is in place;
// otherwise hooks on the game's.
void mouse_camera_install(ElfImage* image);
bool mouse_camera_installed();

// DS3's degrees per count at a sensitivity of 0..10.
float mouse_camera_degrees_per_count(float sensitivity);

// For the pad read, every read: what the camera turns by, and whether the
// camera's own turns are held (engine/mouse_camera_step.h).
void mouse_camera_publish(float sensitivity, bool invert_x, bool invert_y, bool hold_auto_rotation);

// For the pad read: whether the mouse has just flicked for a lock-on target
// switch, and the right stick that switches it (-1..1, x right, y down).
bool mouse_camera_flick(float& x, float& y);

// A compare run's stand-in (decomp/camera/follow_camera.cpp), before it runs
// the game's update and ours on the same input: the update's counts, and the
// game's code holding auto-rotation when ours does.
void mouse_camera_compare_begin();

// The exit report: how the mouse turned the camera.
void mouse_camera_report();
