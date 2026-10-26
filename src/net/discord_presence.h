#pragma once

namespace net {

// True when the player enabled this feature in BBHost options.
bool discord_rpc_enabled();

// Starts the Discord IPC client once for the current BBHost process.
void InitDiscordRPC();

// Publishes a lightweight status while the setup/launcher window is open.
void UpdateDiscordLauncherPresence();

// Publishes the game's front-end/title state before a character is active.
void UpdateDiscordMainMenuPresence();

// Publishes the game's location and loaded character level when available.
void UpdateDiscordGamePresence(const char* locationName, const char* characterName, int playerLevel);

// Clears the activity and closes the IPC client.
void ShutdownDiscordRPC();

}  // namespace net
