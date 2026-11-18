#pragma once
#include "core/play_history.h"
#include "core/chalice_history.h"
#include <chrono>

struct PlayHistoryUi {
    std::string directory;
    play_history::Snapshot state;
    chalice_history::Snapshot chalice_state;
    std::chrono::steady_clock::time_point refreshed{};
    std::uint64_t character = 0;
    int cycle = -1;
    bool cumulative = false, chalice = false;
};
void play_history_ui_draw(PlayHistoryUi& ui, const std::string& data_dir, bool live = false);
