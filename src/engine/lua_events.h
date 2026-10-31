// Named Lua events: the game's scripts
// react to events raised by name through the Lua event manager's dispatcher -
// SprjLuaEventMan (slot at Binary Ninja 0x593b0c8), its context at +8, and
// LuaEvent_DispatchByName (0x1739870) taking (context, name), the call the
// scripts' own event raisers make (the online test raises OnEvent_Call_SOS
// through it). Raised on the game's main thread only: a request from any
// thread is queued and raised at the next frame, from the frame-time
// manager's hook.
#pragma once

#include <cstdint>

struct ElfImage;

void lua_events_install(ElfImage* image);
// Queues a named event; false when the build is not 1.09 or the queue is full.
bool lua_event_queue(const char* name);
// Queues a cutscene (Remo) request as an event script's PlayCutscene makes it
// for the player (entity 10000): the test harness's `remo ID [METHOD]`.
bool remo_request_queue(std::int32_t id, std::int32_t method);
// From the frame-time manager's hook, on the main thread: raises the queued
// events in order.
void lua_events_tick();
