#pragma once

#include "net/session.h"
#include <cstdint>

namespace signaling_test {
void reset_dispatch();
void dispatch_ready(int horizon_ms = 0);
void enter_room(std::uint64_t room, const net::Peer& peer, void* callback);
void leave_room();
void bind_member(const net::Peer& peer);
int member_status(std::uint64_t room, std::uint16_t member, std::uint32_t* addr = nullptr, std::uint16_t* port = nullptr);
}
