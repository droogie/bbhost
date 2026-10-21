#include "net/discord_presence.h"

#include "discord_rpc.h"
#include "host/options.h"

#include <cstring>
#include <string>

namespace net {
namespace {

constexpr char kApplicationId[] = "1562063078466650206";
bool g_initialized = false;
std::string g_last_activity;

void publish(const char* details, const char* state) {
    if (!details || !state) return;
    const std::string activity_key = std::string(details) + "\n" + state;
    if (g_last_activity == activity_key) return;
    DiscordRichPresence presence{};
    presence.details = details;
    presence.state = state;
    Discord_UpdatePresence(&presence);
    g_last_activity = activity_key;
}

}  // namespace

bool discord_rpc_enabled() { return host_opt_get("discord_rpc"); }

void InitDiscordRPC() {
    if (g_initialized) return;

    DiscordEventHandlers handlers{};
    Discord_Initialize(kApplicationId, &handlers, 1, nullptr);
    g_initialized = true;
}

void UpdateDiscordLauncherPresence() {
    if (!g_initialized) return;
    publish("Bloodborne", "In BBHost Launcher");
}

void UpdateDiscordMainMenuPresence() {
    if (!g_initialized) return;
    publish("Bloodborne", "Main Menu");
}

void UpdateDiscordGamePresence(const char* locationName, const char* characterName, int playerLevel) {
    if (!g_initialized) return;

    // Keep the active map and character identity in separate Discord fields.
    std::string details = "Bloodborne";
    if (locationName && locationName[0]) details += " · " + std::string(locationName);
    std::string state = characterName && characterName[0] ? characterName : playerLevel > 0 ? "Character loaded" : "Loading character...";
    if (playerLevel > 0) state += " · Level " + std::to_string(playerLevel);
    publish(details.c_str(), state.c_str());
}

void ShutdownDiscordRPC() {
    if (!g_initialized) return;

    Discord_ClearPresence();
    Discord_Shutdown();
    g_initialized = false;
    g_last_activity.clear();
}

}  // namespace net
