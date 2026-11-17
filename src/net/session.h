// The session service behind the Sony NP API.
//
// Three parts, none of which knows the SDK's structs: a Dispatcher (the one
// host thread that calls guest callbacks, with a guest TCB, in due-time and
// priority order), a Poller (asks the private server's /np/events queue for
// matching2 events and hands them to whoever registered), and the PeerTable
// (member id -> addresses and signaling state, shared by NpMatching2 and
// NpSignaling). The transport is the FastAPI private server's
// /mp/matching2/* endpoints over net/http.
#pragma once

#include "replay/json.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace net {

// Priority when several callbacks are due at once: the contract's order.
enum class Prio { Context = 0, Request = 1, Signaling = 2, RoomEvent = 3 };

// Runs fn on the dispatcher thread after delay_ms. The thread holds a guest
// TCB, so fn may call guest code through hle_call_guest. Serial: no two run
// at once, and the queue is drained in (due, priority, order) order.
void dispatch_after(int delay_ms, Prio prio, std::function<void()> fn);
// True on the dispatcher thread.
bool on_dispatcher();
// Stops the dispatcher and the poller (exit).
void session_shutdown();

// The online id this host plays as (config online.online_id, "Player" if unset).
std::string online_id();
// Our P2P endpoint as the server should record it (config online.p2p_port,
// or the port this instance actually bound when another program had that one).
std::string local_addr_text();
int p2p_port();
void set_p2p_port_bound(int port);

struct Peer {
    std::uint16_t member_id = 0;
    std::string online_id;
    std::uint32_t addr = 0;        // network byte order, the address other peers reach us at
    std::uint16_t port = 0;        // host byte order
    std::uint32_t local_addr = 0;  // network byte order
    std::uint16_t local_port = 0;
    unsigned conn_id = 0;          // NpSignaling connection id once activated (0 = none)
    int sig_status = 0;            // 0 inactive, 1 pending, 2 active
};

// The peer table. All calls lock internally.
void peers_clear();
void peers_upsert(const Peer& p);          // keeps signaling state only for the same player
bool peers_get(std::uint16_t member_id, Peer* out);
bool peers_find_online(const std::string& online_id, Peer* out);
bool peers_find_conn(unsigned conn_id, Peer* out);
void peers_bind_conn(const std::string& online_id, unsigned conn_id, int sig_status);
// Status changes must still name the connection currently bound to this player.
void peers_set_conn(const std::string& online_id, unsigned conn_id, int sig_status);
std::vector<Peer> peers_all();
void peers_erase(std::uint16_t member_id);
std::int64_t now_ms();
// Parses the server's member records ({MemberId, OnlineId, Addr, Port,
// LocalAddr, LocalPort, ...}) into the table. Returns how many.
int peers_from_members(const json::Value& members);

// The server. Each returns false with `error` on failure; `reply` is the parsed JSON.
bool server_context_start(json::Value& reply, std::string& error);
bool server_create_room(int max_members, const json::Value& extra, json::Value& reply, std::string& error);
bool server_join_room(std::uint64_t room_id, json::Value& reply, std::string& error);
bool server_leave_room(const std::string& session_id, int member_id, json::Value& reply, std::string& error);
// 1 when the server keeps us in the room, 0 when it says we are not in it
// (it dropped the room or us: "InRoom": 0), -1 when it could not be asked.
int server_heartbeat(const std::string& session_id, int member_id);
// The host's KickoutRoomMember: `opt` is the request's optData (up to 16
// bytes), which the kicked player's game reads.
bool server_kick_member(const std::string& session_id, int member_id, int kicker_id, const std::uint8_t* opt,
                        std::size_t opt_len, std::string& error);
bool server_session_blob(json::Value& reply, std::string& error);
// The peer's signaling address as the server learned it at its context_start
// (/np/signaling/resolve). addr in network byte order, port in host order.
bool server_signaling_resolve(const std::string& online_id, std::uint32_t* addr, std::uint16_t* port,
                              std::string& error);
// A peer known by address before it is a room member (the host activates a
// connection to the guest it summons before the guest joins). Takes a member
// id from the provisional range 0xff00.. so the table can hold it; the real
// member record replaces nothing - both entries stay and bind to the same
// player's connection. The transport itself retains that player's endpoint.
Peer peers_provisional(const std::string& online_id, std::uint32_t addr, std::uint16_t port);

// The event poller: it asks /np/events/poll for the next matching2 event for
// our online id, hands it to the handler, and acks it. The server holds an
// empty poll until an event comes or its wait ends, and the next poll goes out
// once the last is answered (net/poll_pacing.h has the pace). The handler
// runs on the poller thread and must not call guest code; it schedules
// through dispatch_after. poller_stop never waits: a poll held at that moment
// is given up within about a second, and what it brought goes to the next
// poller_start's poller, never to this one's handler.
void poller_start(std::function<void(const json::Value& event)> handler);
void poller_stop();

// The room we are in, for the keepalive's address updates (np_matching2
// sets it when a room is made or joined and clears it when it is over).
void room_set(const std::string& session_id, int member_id);
void room_clear();
// The last STUN exchange's round trip to the server, microseconds (0 = none yet).
int stun_rtt_us();

}  // namespace net
