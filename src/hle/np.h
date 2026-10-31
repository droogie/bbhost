// Shared between the NP HLE files (system.cpp, np_matching2.cpp, np_signaling.cpp).
#pragma once

#include <cstdint>
#include <string>

// 36-byte SceNpId from an online id: the handle (16 chars + NUL), the rest zero.
void hle_np_fill_npid(void* out, const char* online);
// The handle inside a SceNpId, as text.
std::string hle_np_npid_text(const void* npid);

void hle_register_np_matching2();
// One line for the F10 screen: signed in as whom, in which room as which member, how many peers.
std::string hle_np_online_status();
// True once the game has started its Matching2 context (it read its NpId before that,
// so an account logged in afterwards only takes effect on the next launch).
bool hle_np_context_started();
void hle_register_np_signaling();

// NpSignaling learns a peer's address: fires ESTABLISHED/ACTIVE for a
// connection that was activated before the peer was known.
void hle_np_signaling_peer_known(std::uint16_t member_id);
// NpMatching2 tells NpSignaling the room is gone.
void hle_np_signaling_reset();
// The peer left or timed out: DEAD (0) on its connection(s), so the game's
// SocketState closes the link instead of waiting on it.
void hle_np_signaling_peer_dead(const std::string& online_id);
// A control datagram from another bbhost's NpSignaling (hle/net.cpp,
// kSignalMagic), from addr (network order):port (network order).
void hle_np_signaling_control(const std::uint8_t* data, std::size_t len, std::uint32_t addr, std::uint16_t port);
// Whether the game has a live NpSignaling connection to this player
// (NpMatching2's leave keeps that player's DEAD until the game closes it).
bool hle_np_signaling_live(const std::string& online_id);
// NpSignaling closed the game's connection to this player: NpMatching2 sends
// the DEAD it kept for that player at its LeaveRoom, before NpSignaling's own.
void hle_np_matching2_peer_closed(const std::string& online_id);
