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
// The other members of the room this session is in (0 outside one): a player
// on the way in counts from the server's member-joined event, before the game
// has made anything of them (plugin API 29's world_state).
std::uint32_t hle_np_room_others();
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

// Matching2 signaling rides NpSignaling's transport to each player, as
// shadPS4's (signaling_handler.cpp StartMatching2 / SyncMatching2Peers /
// StopMatching2):
// A room member to connect to: its transport made (or kept) and its handshake
// started; hle_np_matching2_transport_up once it is Established.
void hle_np_signaling_m2_connect(const std::string& online_id);
// The transport's status to that player: 0 inactive, 1 pending, 2 active.
int hle_np_signaling_transport_status(const std::string& online_id);
// Our room is over for that player: its transport closed, unless the game
// still has an NpSignaling connection on it.
void hle_np_signaling_m2_stop(const std::string& online_id);
// The transport to that player has been Established for a second:
// NpMatching2's ESTABLISHED (0x5102) for that member, once.
void hle_np_matching2_transport_up(const std::string& online_id);
// The transport to that player gave up (handshake timeout): NpMatching2's
// DEAD (0x5101) for that member with this error.
void hle_np_matching2_transport_dead(const std::string& online_id, int error);
