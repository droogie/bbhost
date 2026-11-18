#include "host/trophy_ui.h"

#if defined(BBHOST_HAVE_SDL3)
#include "host/overlay.h"
#include "imgui.h"
#include <algorithm>
#include <cstdio>
#include <ctime>

namespace {
std::uint32_t colour(trophies::Grade g) {
    switch (g) {
    case trophies::Grade::Platinum: return 0xb8e9ffff;
    case trophies::Grade::Gold: return 0xe8c56aff;
    case trophies::Grade::Silver: return 0xd4dbe5ff;
    default: return 0xd99c6aff;
    }
}
ImVec4 text_colour(trophies::Grade g) {
    const auto c = colour(g);
    return ImVec4((c >> 24) / 255.f, ((c >> 16) & 255) / 255.f, ((c >> 8) & 255) / 255.f, 1);
}
}

void trophy_ui_draw(TrophyUi& ui, const std::string& data_dir, float height) {
    const auto now = std::chrono::steady_clock::now();
    const auto file = std::filesystem::u8path(data_dir) / "trophies.txt";
    if (ui.file != data_dir || now - ui.refreshed > std::chrono::seconds(1)) {
        ui.file = data_dir; ui.refreshed = now; ui.state = trophies::read(file);
    }
    ImGui::BeginChild("##trophies", ImVec2(0, height));
    ImGui::Text("Trophies  %d / %d", ui.state.count(), trophies::kCount);
    ImGui::ProgressBar(float(ui.state.count()) / trophies::kCount, ImVec2(-1, 0));
    ImGui::TextWrapped("Trophies are stored on this PC. Earn all 33 base-game trophies to unlock platinum; the expansion is not required.");
    if (!ui.state.error.empty()) ImGui::TextColored(ImVec4(1,.65f,.3f,1), "%s", ui.state.error.c_str());
    const char* filters[] = {"All", "Unlocked", "Locked"};
    ImGui::SetNextItemWidth(160);
    ImGui::Combo("Show", &ui.filter, filters, 3);
    ImGui::SameLine();
    ImGui::Checkbox("Reveal hidden trophies", &ui.reveal);
    for (int group = 0; group < 2; ++group) {
        const int begin = group ? trophies::kBaseCount : 0, end = group ? trophies::kCount : trophies::kBaseCount;
        char title[100];
        std::snprintf(title, sizeof(title), "%s  %d / %d", group ? "The Old Hunters" : "Base game", ui.state.count(begin, end), end - begin);
        ImGui::SeparatorText(title);
        for (int i = begin; i < end; ++i) {
            const auto& t = trophies::catalog()[i];
            const auto when = ui.state.earned[i];
            if ((ui.filter == 1 && !when) || (ui.filter == 2 && when)) continue;
            const bool secret = t.hidden && !when && !ui.reveal;
            ImGui::TextColored(text_colour(t.grade), "%s", trophies::grade_name(t.grade));
            ImGui::SameLine();
            ImGui::Text("%s  %s", when ? "Unlocked" : "Locked", secret ? "Hidden trophy" : t.name);
            ImGui::TextWrapped("%s", secret ? "Unlock this trophy to reveal its name and requirements." : t.description);
            if (when) {
                const std::time_t seconds = static_cast<std::time_t>(when);
                std::tm tm{};
#if defined(_WIN32)
                const bool valid = gmtime_s(&tm, &seconds) == 0;
#else
                const bool valid = gmtime_r(&seconds, &tm) != nullptr;
#endif
                char date[64] = {};
                if (valid && std::strftime(date, sizeof(date), "%Y/%m/%d %H:%M UTC", &tm)) ImGui::TextDisabled("%s", date);
            }
            ImGui::Spacing();
        }
    }
    ImGui::EndChild();
}

void trophy_notice_draw(float dw, float dh) {
    static int current = -1;
    static auto began = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    if (current >= 0 && now - began >= std::chrono::seconds(6)) current = -1;
    if (current < 0) {
        current = trophies::runtime().pop_notice();
        if (current < 0) return;
        began = now;
    }
    const auto& item = trophies::catalog()[current];
    const float elapsed = std::chrono::duration<float>(now - began).count();
    const float alpha = std::min(std::clamp(elapsed / .2f, 0.f, 1.f), std::clamp((6.f - elapsed) / .5f, 0.f, 1.f));
    const auto fade = [&](std::uint32_t c) { return (c & 0xffffff00u) | std::uint32_t((c & 255) * alpha); };
    const float s = std::clamp(dh / 1080.f, .65f, 1.5f);
    const float x = 24*s, y = 28*s, width = std::min(560*s, dw - 48*s), h = 102*s;
    host_overlay_rect(x, y, width, h, fade(0x15191ff0));
    host_overlay_rect(x, y, 4*s, h, fade(colour(item.grade)));
    // Draw the cup with overlay primitives.
    const float cx = x+26*s, cy = y+28*s;
    host_overlay_rect(cx, cy, 34*s, 24*s, fade(colour(item.grade)));
    host_overlay_tri(cx, cy+24*s, cx+34*s, cy+24*s, cx+17*s, cy+35*s, fade(colour(item.grade)));
    host_overlay_rect(cx+14*s, cy+30*s, 6*s, 14*s, fade(colour(item.grade)));
    host_overlay_rect(cx+6*s, cy+44*s, 22*s, 5*s, fade(colour(item.grade)));
    host_overlay_text(x+80*s, y+13*s, .65f*s, fade(0xcbd3dfff), "Trophy unlocked");
    const float name_scale = .9f*s * std::min(1.f, (width-95*s) / std::max(1.f, host_overlay_text_width(.9f*s, item.name)));
    host_overlay_text(x+80*s, y+37*s, name_scale, fade(0xffffffff), item.name);
    host_overlay_text(x+80*s, y+76*s, .6f*s, fade(colour(item.grade)), trophies::grade_name(item.grade));
}
#else
void trophy_ui_draw(TrophyUi&, const std::string&, float) {}
void trophy_notice_draw(float, float) {}
#endif
