#include "host/play_history_ui.h"

#if defined(BBHOST_HAVE_SDL3)
#include "imgui.h"
#include <algorithm>

namespace {
std::string label(const play_history::Character& c) {
    return "Slot " + std::to_string(c.slot + 1) + ": " + (c.name.empty() ? "Character" : c.name) +
           " (record " + std::to_string(c.id) + ")";
}
void chalice_records(const PlayHistoryUi& ui, const play_history::Character& c, int cycle) {
    const auto& state = ui.chalice_state;
    if (!state.error.empty()) ImGui::TextWrapped("%s", state.error.c_str());
    ImGui::SeparatorText(ui.cumulative ? "Chalice bosses: all cycles" : "Chalice bosses: selected cycle");
    const auto rows = chalice_history::totals(state, c.id, ui.cumulative ? -1 : cycle);
    if (rows.empty()) ImGui::TextWrapped("No chalice encounters recorded for this cycle yet. Enter a chalice dungeon to record each dungeon and layer.");
    if (!rows.empty() && ImGui::BeginTable("##chalice-totals", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn("Chalice boss"); ImGui::TableSetupColumn("Attempts"); ImGui::TableSetupColumn("Recorded kills"); ImGui::TableHeadersRow();
        for (const auto& row : rows) {
            ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextWrapped("%s", chalice_history::boss_name(row.name_id).c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%llu%s", static_cast<unsigned long long>(row.count.attempts), row.count.partial ? "+" : "");
            ImGui::TableNextColumn(); ImGui::Text("%llu", static_cast<unsigned long long>(row.count.kills));
        }
        ImGui::EndTable();
    }
    if (ImGui::CollapsingHeader("Records by dungeon and layer", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& e : state.encounters) if (e.character == c.id) {
            play_history::BossRecord total{}; bool shown = ui.cumulative && e.prior_clear;
            for (const auto& s : e.stats) if (ui.cumulative || s.cycle == cycle) {
                total.attempts += s.count.attempts; total.kills += s.count.kills; total.partial = total.partial || s.count.partial; shown = true;
            }
            if (!shown) continue;
            ImGui::PushID(static_cast<int>(e.id));
            ImGui::Separator();
            const auto place = e.altar ? "Altar " + std::to_string(e.altar) : std::string("Makeshift Altar");
            ImGui::TextWrapped("%s: %s, layer %d (record %llu)", chalice_history::boss_name(e.name_id).c_str(),
                place.c_str(), e.layer, static_cast<unsigned long long>(e.id));
            if (e.prior_clear) ImGui::TextWrapped("Cleared before recording began. The cycle is unknown, so this kill is not assigned to a cycle.");
            else ImGui::Text("Attempts %llu%s / kills %llu%s", static_cast<unsigned long long>(total.attempts), total.partial ? "+" : "",
                static_cast<unsigned long long>(total.kills), e.active ? " / in progress" : "");
            ImGui::PopID();
        }
    }
    ImGui::Spacing();
    ImGui::TextWrapped("Chalice records and totals are separate from the base game and DLC. Each dungeon and layer is tracked separately. Chalice progress carries into NG+, so an existing kill is not counted again in the next cycle.");
    ImGui::TextWrapped("Supports fixed and root chalice bosses. Earlier attempts are unknown, so initial records show a lower bound (+). Recording pauses during boss rush, practice, and visits to another player's world.");
}
}
void play_history_ui_draw(PlayHistoryUi& ui, const std::string& data_dir, bool live) {
    const auto now = std::chrono::steady_clock::now();
    if (ui.directory != data_dir || now - ui.refreshed >= std::chrono::seconds(1)) {
        ui.directory = data_dir; ui.refreshed = now;
        ui.state = live ? play_history::runtime().snapshot() : play_history::read(std::filesystem::u8path(data_dir) / "play-history.txt");
        ui.chalice_state = live ? chalice_history::runtime().snapshot() : chalice_history::read(std::filesystem::u8path(data_dir) / "chalice-history.txt");
    }
    ImGui::BeginChild("##play-history");
    ImGui::TextWrapped("Records NG and NG+ playtime, boss attempts, and kills for each character.");
    if (!ui.state.error.empty()) ImGui::TextWrapped("%s", ui.state.error.c_str());
    if (ui.state.characters.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("Recording starts automatically when you play. Records are saved in play-history.txt in the data folder and continue across sessions.");
        ImGui::EndChild(); return;
    }
    auto found = std::find_if(ui.state.characters.begin(), ui.state.characters.end(), [&](const auto& c) { return c.id == ui.character; });
    if (found == ui.state.characters.end()) { found = ui.state.characters.end() - 1; ui.character = found->id; ui.cycle = -1; }
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##character", label(*found).c_str())) {
        for (const auto& c : ui.state.characters) {
            if (ImGui::Selectable(label(c).c_str(), c.id == ui.character)) { ui.character = c.id; ui.cycle = -1; }
        }
        ImGui::EndCombo();
    }
    // A selection may have changed above.
    const auto& c = *std::find_if(ui.state.characters.begin(), ui.state.characters.end(), [&](const auto& item) { return item.id == ui.character; });
    if (c.cycles.empty()) { ImGui::EndChild(); return; }
    if (ui.cycle < 0 || ui.cycle >= static_cast<int>(c.cycles.size())) ui.cycle = static_cast<int>(c.cycles.size()) - 1;
    ImGui::SeparatorText("Playtime by cycle");
    if (ImGui::BeginTable("##cycles", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Cycle"); ImGui::TableSetupColumn("Playtime"); ImGui::TableSetupColumn("Status"); ImGui::TableHeadersRow();
        for (int i = 0; i < static_cast<int>(c.cycles.size()); ++i) {
            const auto& cy = c.cycles[i];
            ImGui::PushID(i); ImGui::TableNextRow(); ImGui::TableNextColumn();
            if (ImGui::Selectable(play_history::cycle_name(cy.number).c_str(), ui.cycle == i, ImGuiSelectableFlags_SpanAllColumns)) {
                ui.cycle = i; ui.cumulative = false;
            }
            ImGui::TableNextColumn();
            ImGui::Text("%s%s", play_history::duration(cy.elapsed_ms).c_str(), cy.partial_time ? "+" : "");
            ImGui::TableNextColumn(); ImGui::TextUnformatted(cy.complete ? "Complete" : "In progress"); ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::Checkbox("Show totals across all cycles", &ui.cumulative);
    const auto& selected = c.cycles[ui.cycle];
    if (ImGui::RadioButton("Base game and DLC", !ui.chalice)) ui.chalice = false;
    ImGui::SameLine();
    if (ImGui::RadioButton("Chalice dungeons", ui.chalice)) ui.chalice = true;
    if (ui.chalice) {
        chalice_records(ui, c, selected.number);
        ImGui::EndChild(); return;
    }
    const auto records = ui.cumulative ? play_history::totals(c) : selected.bosses;
    ImGui::SeparatorText(ui.cumulative ? "Boss fights: this character's totals across all cycles" : play_history::cycle_name(selected.number).c_str());
    if (ImGui::BeginTable("##boss-records", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Boss", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        ImGui::TableSetupColumn("Attempts", ImGuiTableColumnFlags_WidthStretch, 1.f);
        ImGui::TableSetupColumn("Kills", ImGuiTableColumnFlags_WidthStretch, .6f);
        ImGui::TableSetupColumn("Result", ImGuiTableColumnFlags_WidthStretch, 1.6f); ImGui::TableHeadersRow();
        for (int i = 0; i < play_history::kBossCount; ++i) {
            const auto& b = records[i];
            ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextWrapped("%s", play_history::bosses()[i].name);
            ImGui::TableNextColumn();
            if (b.partial && !b.attempts) ImGui::TextUnformatted("Unknown");
            else ImGui::Text("%llu%s", static_cast<unsigned long long>(b.attempts), b.partial ? "+" : "");
            ImGui::TableNextColumn(); ImGui::Text("%llu", static_cast<unsigned long long>(b.kills));
            ImGui::TableNextColumn();
            const bool fighting = !ui.cumulative && ui.cycle == static_cast<int>(c.cycles.size()) - 1 && c.active_boss == i;
            ImGui::TextWrapped("%s", fighting ? "In progress" : b.kills ? (b.partial ? "Defeated (partial record)" : "Defeated") : b.attempts ? "Not defeated" : "Not recorded");
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::TextWrapped("An attempt is counted when a fight starts, including deaths, retreats, interrupted fights, and the winning attempt. Counts for that cycle are final once the boss is defeated.");
    ImGui::TextWrapped("Time follows the game's play clock. Entering the next NG+ ends the previous cycle. Recording pauses during boss rush, practice, and visits to another player's world. Chalice bosses have their own records under Chalice dungeons.");
    ImGui::TextWrapped("Earlier attempts and NG+ start times cannot be reconstructed. A + or Unknown means the record is incomplete. If a save rollback or a new character is detected, the previous record is kept and a new one is started.");
    ImGui::EndChild();
}
#else
void play_history_ui_draw(PlayHistoryUi&, const std::string&, bool) {}
#endif
