// What the session layer asks of the P2P port (hle/net.cpp) for players
// behind NAT: the reflexive address of the port through a STUN
// Binding exchange on the port's own socket, and hole-punch probes toward a
// peer so this side's NAT opens for the peer's datagrams before the game's
// own traffic starts.
#pragma once

#include <cstddef>
#include <cstdint>

#include "net/stun.h"

// Sends a STUN Binding Request from the P2P port (opened if the game has not
// bound it yet) to host:port and waits up to timeout_ms for the response.
// True with the mapped address (network byte order) and port (host order).
// With `relay`, the request carries the relay HELLO (net/stun.h) and the
// answer's BBHOST-RELAY lands there; a relay in the answer turns framing on
// for datagrams to that server's other ports, an answer without one off.
bool hle_net_p2p_stun(const char* host, std::uint16_t port, int timeout_ms, std::uint32_t* mapped_addr,
                      std::uint16_t* mapped_port, net::stun::Relay* relay = nullptr);

// Whether datagrams go through the server's relay: its address (network
// order) and our port on it.
bool hle_net_p2p_relay(std::uint32_t* server, std::uint16_t* vport);

// Starts sending small probe datagrams from the P2P port to addr:port and,
// when different, local_addr:local_port (network byte order, host order),
// every 500 ms for up to 20 s or until anything arrives from either. The
// probes carry their own magic and the receiving port drops them; the game
// never sees them. `label` names the peer in the log. A peer behind the
// relay gets none: the relay needs no hole.
void hle_net_p2p_punch(const char* label, std::uint32_t addr, std::uint16_t port, std::uint32_t local_addr,
                       std::uint16_t local_port);

// NpSignaling's control datagrams between two bbhost instances (activation,
// its acknowledgement, close): sent from the P2P port to addr (network
// order):port (host order), through the relay when the peer is behind it.
// They start with kSignalMagic; the receiving port hands them to
// hle_np_signaling_control (hle/np.h) and the game never sees them.
constexpr std::uint8_t kSignalMagic[4] = {0xfe, 'b', 'b', 's'};
void hle_net_p2p_signal(std::uint32_t addr, std::uint16_t port, const void* data, std::size_t len);
