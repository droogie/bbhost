#pragma once
#include <cstdint>
struct ElfImage;
void play_history_install(ElfImage* image);
void play_history_tick();  // main thread, after loading and plugin callbacks
void play_history_event_flag(std::uint32_t id, bool value);
int play_history_pause(const char* plugin, int paused);  // main-thread plugin API
