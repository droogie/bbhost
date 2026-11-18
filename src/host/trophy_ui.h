#pragma once
#include "core/trophies.h"
#include <chrono>
#include <string>

struct TrophyUi {
    std::string file;
    trophies::Snapshot state;
    std::chrono::steady_clock::time_point refreshed{};
    int filter = 0;
    bool reveal = false;
};
void trophy_ui_draw(TrophyUi& ui, const std::string& data_dir, float height = 0);
// Called on the window pump between overlay reset and commit.
void trophy_notice_draw(float display_width, float display_height);
