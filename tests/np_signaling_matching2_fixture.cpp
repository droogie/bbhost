// Exercise actual Matching2 bindings and callbacks without an HTTP room service.
#include "hle/np_matching2.cpp"
#include "np_signaling_fixture.h"

namespace signaling_test {
void enter_room(std::uint64_t room, const net::Peer& peer, void* callback) {
    std::lock_guard<std::mutex> lk(g.mu);
    forget_room_locked();
    g.ctx_id = 1;
    g.in_room = true;
    g.room_id = room;
    g.member_id = 1;
    g.sig_cb = callback;
    net::peers_upsert(peer);
    hle_np_signaling_peer_known(peer.member_id);
    connect_signaling_for_all();
}

void leave_room() {
    std::lock_guard<std::mutex> lk(g.mu);
    end_session_locked("identity regression test");
}

void bind_member(const net::Peer& peer) {
    std::lock_guard<std::mutex> lk(g.mu);
    net::peers_upsert(peer);
    hle_np_signaling_peer_known(peer.member_id);
    g.m2_established.erase(peer.member_id);
    g.m2_dead_sent.erase(peer.member_id);
    g.m2_transports[peer.member_id] = hle_np_signaling_m2_connect(peer.online_id);
}

int member_status(std::uint64_t room, std::uint16_t member, std::uint32_t* addr, std::uint16_t* port) {
    int status = -1;
    m2_signaling_get_connection_status(1, room, member, &status, addr, port);
    return status;
}
}
