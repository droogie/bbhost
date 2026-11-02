#define IMGUI_DEFINE_MATH_OPERATORS
#include "host/plugin_ui.h"

#include "bbhost_plugin.h"
#include "core/config.h"
#include "host/updater.h"

#include "imgui.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream in(s);
    while (std::getline(in, cur, sep)) out.push_back(cur);
    return out;
}

// The plugins.seen list: name@version of each plugin the manager has shown.
std::vector<std::string> seen_list() { return split(config_value("plugins.seen"), ','); }

std::string seen_key(const PluginEntry& p) { return p.name + "@" + p.version; }

// Eight hex digits: what the randomizer makes itself, and a plain seed token
// (bb::seed_token), so the server and other players see it as typed.
std::string random_seed() {
    std::random_device rd;
    std::mt19937 gen(rd() ^ static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
    char text[16];
    std::snprintf(text, sizeof(text), "%08x", static_cast<unsigned>(gen()));
    return text;
}

// A small rounded label beside the title, its text on the title's line.
void badge(const char* text, ImVec4 colour) {
    ImGui::SameLine();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 ts = ImGui::CalcTextSize(text);
    const ImVec2 pad(ImGui::GetFontSize() * 0.45f, ImGui::GetFontSize() * 0.1f);
    const ImVec2 p = ImGui::GetCursorScreenPos() - ImVec2(0, pad.y);
    dl->AddRectFilled(p, p + ts + pad * 2, ImGui::GetColorU32(ImVec4(colour.x, colour.y, colour.z, 0.16f)), ts.y * 0.5f + pad.y);
    dl->AddText(p + pad, ImGui::GetColorU32(colour), text);
    ImGui::Dummy(ImVec2(ts.x + pad.x * 2, ts.y));
}

// A setting's help: in a bubble over the setting, and over the (?) after it.
void help(const std::string& text) {
    auto bubble = [&] {
        if (ImGui::BeginItemTooltip()) {
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
            ImGui::TextUnformatted(text.c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    };
    bubble();
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    bubble();
}

// One setting's widget. `value` is what the config holds (or the default);
// a change goes to plugins_set_option as text.
void option_row(PluginUi& ui, const PluginEntry& p, const PluginOptionDesc& o, float scale) {
    const std::string id = p.name + "." + o.key;
    ImGui::PushID(id.c_str());
    const bool later = ui.in_game && !o.live && o.type != BB_OPT_ACTION && o.type != BB_OPT_HEADING;
    const std::string label = o.label + (later ? "  (next start)" : "");
    bool helped = false;  // the widget put its help after something of its own
    const std::string value = plugins_option_value(p, o);
    const float width = 260 * scale;
    switch (o.type) {
    case BB_OPT_HEADING:
        ImGui::SeparatorText(o.label.c_str());
        break;
    case BB_OPT_BOOL: {
        bool on = value == "true" || value == "1";
        if (ImGui::Checkbox(label.c_str(), &on)) plugins_set_option(p.name, o.key, on ? "true" : "false");
        break;
    }
    case BB_OPT_INT: {
        auto it = ui.edits.find(id);
        int v = std::atoi((it != ui.edits.end() ? it->second : value).c_str());
        ImGui::SetNextItemWidth(width);
        if (ImGui::SliderInt(label.c_str(), &v, static_cast<int>(o.min), static_cast<int>(o.max))) ui.edits[id] = std::to_string(v);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            plugins_set_option(p.name, o.key, std::to_string(v));
            ui.edits.erase(id);
        }
        break;
    }
    case BB_OPT_FLOAT: {
        auto it = ui.edits.find(id);
        float v = static_cast<float>(std::atof((it != ui.edits.end() ? it->second : value).c_str()));
        ImGui::SetNextItemWidth(width);
        if (ImGui::SliderFloat(label.c_str(), &v, static_cast<float>(o.min), static_cast<float>(o.max), "%.2f")) {
            char t[32];
            std::snprintf(t, sizeof(t), "%.2f", static_cast<double>(v));
            ui.edits[id] = t;
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            char t[32];
            std::snprintf(t, sizeof(t), "%.2f", static_cast<double>(v));
            plugins_set_option(p.name, o.key, t);
            ui.edits.erase(id);
        }
        break;
    }
    case BB_OPT_CHOICE: {
        const std::vector<std::string> choices = split(o.choices, '|');
        int cur = 0;
        for (std::size_t i = 0; i < choices.size(); ++i) {
            if (choices[i] == value) cur = static_cast<int>(i);
        }
        ImGui::SetNextItemWidth(width);
        if (ImGui::BeginCombo(label.c_str(), choices.empty() ? "" : choices[static_cast<std::size_t>(cur)].c_str())) {
            for (std::size_t i = 0; i < choices.size(); ++i) {
                if (ImGui::Selectable(choices[i].c_str(), static_cast<int>(i) == cur)) plugins_set_option(p.name, o.key, choices[i]);
            }
            ImGui::EndCombo();
        }
        break;
    }
    case BB_OPT_TEXT: {
        auto it = ui.edits.find(id);
        if (it == ui.edits.end()) it = ui.edits.emplace(id, value).first;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s", it->second.c_str());
        ImGui::SetNextItemWidth(width);
        if (ImGui::InputText(label.c_str(), buf, sizeof(buf))) it->second = buf;
        if (ImGui::IsItemDeactivatedAfterEdit()) plugins_set_option(p.name, o.key, it->second);
        const bool active = ImGui::IsItemActive();
        // A seed: a new one at the click, shown in the box and saved.
        if (o.key == "seed") {
            if (!o.help.empty()) help(o.help);
            helped = true;
            ImGui::SameLine();
            if (ImGui::Button("Random seed")) {
                it->second = random_seed();
                plugins_set_option(p.name, o.key, it->second);
            }
        }
        if (!active && it->second == value) ui.edits.erase(id);
        break;
    }
    case BB_OPT_ACTION: {
        const bool can = ui.in_game && p.loaded;
        ImGui::BeginDisabled(!can);
        if (ImGui::Button(o.label.c_str(), ImVec2(width, 0))) plugins_run_action(p.name, o.key);
        ImGui::EndDisabled();
        // Why it cannot be pressed, in its bubble with its help.
        if (!can) {
            if (ImGui::BeginItemTooltip()) {
                ImGui::TextUnformatted(ui.in_game ? "Turn the plugin on and restart." : "In the game: the plugin menu.");
                if (!o.help.empty()) ImGui::TextDisabled("%s", o.help.c_str());
                ImGui::EndTooltip();
            }
            helped = true;
        }
        break;
    }
    default:
        break;
    }
    if (!o.help.empty() && o.type == BB_OPT_HEADING) {
        // A heading's words say what the settings under it are for: shown.
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", o.help.c_str());
        ImGui::PopTextWrapPos();
    } else if (!o.help.empty() && !helped) {
        help(o.help);
    }
    ImGui::PopID();
}

}  // namespace

void plugin_ui_refresh(PluginUi& ui) {
    ui.plugins = plugins_scan();
    // In the game, which of them are running.
    for (const PluginEntry& loaded : plugins_catalog()) {
        for (PluginEntry& p : ui.plugins) {
            if (p.name != loaded.name) continue;
            p.loaded = !loaded.visitor;  // a visitor is not the player's plugin running
            p.visitor = loaded.visitor;
        }
    }
    std::sort(ui.plugins.begin(), ui.plugins.end(), [](const PluginEntry& a, const PluginEntry& b) {
        if (a.official != b.official) return a.official;
        return a.title < b.title;
    });
    if (ui.selected >= static_cast<int>(ui.plugins.size())) ui.selected = 0;
    ui.scanned = true;
}

void plugin_ui_draw(PluginUi& ui, float scale, float height) {
    if (!ui.scanned) plugin_ui_refresh(ui);
    static bool was_busy = false;
    const bool busy = updater::plugins_busy();
    if (was_busy && !busy) plugin_ui_refresh(ui);  // a fetch just ended
    was_busy = busy;

    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Get the official plugins")) updater::fetch_plugins(config_user_dir() + "/plugins");
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("The latest signed plugins, from the newest release.");
    ImGui::SameLine();
    if (ImGui::Button("Rescan")) plugin_ui_refresh(ui);
    ImGui::SetItemTooltip("Read the plugin files again.");
    ImGui::SameLine();
    // At the right of the line.
    const char* every_label = "Show at every start";
    const float every_w = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize(every_label).x;
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - every_w));
    bool every = config_value("plugins.show_settings") == "true";
    if (ImGui::Checkbox(every_label, &every)) {
        config_set_values(config_user_file(), {{"plugins", "show_settings", every ? "true" : "false"}});
        config_value_set("plugins.show_settings", every ? "true" : "false");
    }
    ImGui::SetItemTooltip("Open the launcher on the plugins at every start.");
    if (const std::string st = updater::plugins_status(); !st.empty()) ImGui::TextDisabled("%s", st.c_str());

    const float h = height > 0 ? height : ImGui::GetContentRegionAvail().y;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ImGui::GetFontSize() * 0.6f, ImGui::GetFontSize() * 0.6f));
    ImGui::BeginChild("##list", ImVec2(230 * scale, h), ImGuiChildFlags_Borders);
    ImGui::PopStyleVar();
    if (ui.plugins.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("No plugins. Get the official ones above, or put plugin files in %s/plugins.", config_user_dir().c_str());
        ImGui::PopTextWrapPos();
    }
    // Each plugin a row: a dot, lit when it is on, and its title.
    const float fs = ImGui::GetFontSize(), row_h = std::round(fs * 1.8f);
    for (std::size_t i = 0; i < ui.plugins.size(); ++i) {
        const PluginEntry& p = ui.plugins[i];
        ImGui::PushID(static_cast<int>(i));
        const ImVec2 at = ImGui::GetCursorScreenPos();
        if (ImGui::Selectable("##row", ui.selected == static_cast<int>(i), 0, ImVec2(0, row_h))) ui.selected = static_cast<int>(i);
        ImGui::SetItemTooltip("%s", p.enabled ? "On" : "Off");
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec4 dot = p.bad_signature ? ImVec4(0.95f, 0.35f, 0.40f, 1.0f) : p.enabled ? ImVec4(0.37f, 0.89f, 0.67f, 1.0f) : ImVec4(0.5f, 0.5f, 0.6f, 0.6f);
        dl->AddCircleFilled(at + ImVec2(fs * 0.55f, row_h * 0.5f), fs * 0.22f, ImGui::GetColorU32(dot));
        dl->AddText(at + ImVec2(fs * 1.3f, (row_h - fs) * 0.5f), ImGui::GetColorU32(p.enabled ? ImGuiCol_Text : ImGuiCol_TextDisabled),
                    p.title.c_str());
        ImGui::PopID();
    }
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(fs * 1.0f, fs * 0.8f));
    ImGui::BeginChild("##details", ImVec2(0, h), ImGuiChildFlags_Borders);
    ImGui::PopStyleVar();
    if (!ui.plugins.empty()) {
        PluginEntry& p = ui.plugins[static_cast<std::size_t>(ui.selected)];
        ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.3f);
        ImGui::TextUnformatted(p.title.c_str());
        ImGui::PopFont();
        if (!p.version.empty()) {
            ImGui::SameLine();
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("%s", p.version.c_str());
        }
        if (p.official) badge("official", ImVec4(0.37f, 0.89f, 0.67f, 1.0f));
        else if (p.bad_signature) badge("signature does not match - not loaded", ImVec4(0.95f, 0.35f, 0.40f, 1.0f));
        else badge("not signed", ImVec4(0.65f, 0.66f, 0.78f, 1.0f));
        if (p.flags & BB_PLUGIN_GAMEPLAY) badge("changes the game", ImVec4(1.0f, 0.54f, 0.78f, 1.0f));
        if (!p.author.empty()) ImGui::TextDisabled("by %s - %s", p.author.c_str(), p.file.c_str());
        if (!p.description.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(p.description.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::Spacing();
        ImGui::BeginDisabled(p.bad_signature);
        bool on = p.enabled;
        if (ImGui::Checkbox("Turned on", &on)) {
            plugins_set_enabled(p.name, on);
            p.enabled = on;
        }
        ImGui::EndDisabled();
        if (p.visitor && !p.enabled) {
            ImGui::SameLine();
            ImGui::TextDisabled("off - joins other players' %s worlds when summoned", p.title.c_str());
        } else if (ui.in_game && p.enabled != p.loaded) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.30f, 1.0f), p.enabled ? "starts with the next launch" : "stops at the next launch");
        }
        if (!p.options.empty()) {
            ImGui::Spacing();
            for (const PluginOptionDesc& o : p.options) option_row(ui, p, o, scale);
        } else {
            ImGui::TextDisabled("No settings.");
        }
    }
    ImGui::EndChild();
}

bool plugin_ui_wants_attention() {
    if (config_value("plugins.show_settings") == "true") return true;
    const std::vector<std::string> seen = seen_list();
    for (const PluginEntry& p : plugins_scan()) {
        if (!p.enabled || p.options.empty()) continue;
        if (std::find(seen.begin(), seen.end(), seen_key(p)) == seen.end()) return true;
    }
    return false;
}

void plugin_ui_mark_seen(const PluginUi& ui) {
    std::vector<std::string> seen = seen_list();
    bool changed = false;
    for (const PluginEntry& p : ui.plugins) {
        if (!p.enabled || p.options.empty()) continue;
        if (std::find(seen.begin(), seen.end(), seen_key(p)) == seen.end()) {
            seen.push_back(seen_key(p));
            changed = true;
        }
    }
    if (!changed) return;
    std::string v;
    for (const std::string& s : seen) v += (v.empty() ? "" : ",") + s;
    config_set_values(config_user_file(), {{"plugins", "seen", "\"" + v + "\""}});
    config_value_set("plugins.seen", "\"" + v + "\"");
}
