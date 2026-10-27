// sceNpMatching2 over the session service.
//
// The game takes three values from this library - connection status 2, the
// peer's address, the peer's port - and its own session state machines do
// the rest, driven by the callbacks below. The layouts are the SDK's; what
// the game reads out of each callback was read from the eboot's own
// dispatch (sub_10c3210: the roomId at RoomDataInternal+0x18, the member
// list, the owner's member id; sub_10c3da0: MEMBER_JOINED 0x1101 with the
// member record at data[0]). Timings and order are the co-op contract.
#include "hle/np.h"

#include "core/config.h"
#include "core/thunk.h"
#include "engine/summon_invite.h"
#include "guest_abi.h"
#include "hle/common.h"
#include "host/plugins.h"
#include "log.h"
#include "net/http.h"
#include "net/account.h"
#include "net/session.h"

#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace {

// ---- SDK layouts (the game's build of libSceNpMatching2) --------------------

struct RoomDataInternal {
    std::uint16_t publicSlots, privateSlots, openPublicSlots, openPrivateSlots, maxSlot, serverId;
    std::uint32_t worldId;
    std::uint64_t lobbyId;
    std::uint64_t roomId;
    std::uint64_t passwdSlotMask;
    std::uint64_t joinedSlotMask;
    const void* roomGroup;
    std::uint64_t roomGroups;
    std::uint32_t flags;
    std::uint32_t pad;
    const void* roomBinAttrInternal;
    std::uint64_t roomBinAttrInternalNum;
};
static_assert(offsetof(RoomDataInternal, roomId) == 0x18);
static_assert(sizeof(RoomDataInternal) == 0x58);

struct RoomMemberDataInternal {
    RoomMemberDataInternal* next;
    std::uint64_t joinDate;
    std::uint8_t npId[36];
    std::uint8_t pad[4];
    std::uint16_t memberId;
    std::uint8_t teamId;
    std::uint8_t natType;
    std::uint32_t flagAttr;
    void* roomGroup;
    void* roomMemberBinAttrInternal;
    std::uint64_t roomMemberBinAttrInternalNum;
};
static_assert(offsetof(RoomMemberDataInternal, memberId) == 0x38);
static_assert(sizeof(RoomMemberDataInternal) == 0x58);

struct RoomMemberDataInternalList {
    RoomMemberDataInternal* members;
    std::uint64_t membersNum;
    RoomMemberDataInternal* me;
    RoomMemberDataInternal* owner;
};

struct CreateJoinRoomResponse {
    const RoomDataInternal* roomData;
    RoomMemberDataInternalList members;
};

struct World {
    World* next;
    std::uint32_t worldId;
    std::uint32_t numOfLobby;
    std::uint32_t curNumOfTotalLobby;
    std::uint32_t maxNumOfTotalLobby;
    std::uint32_t curNumOfRoom;
    std::uint32_t maxNumOfRoom;
    std::uint32_t curNumOfTotalRoomMember;
    std::uint32_t numOfRoom;
};
static_assert(sizeof(World) == 0x28);  // the game walks the list at this stride (sub_10be5c0)

struct GetWorldInfoListResponse {
    World* world;
    std::uint32_t worldNum;
    std::uint32_t pad;
};

struct RoomMemberUpdateInfo {
    RoomMemberDataInternal* roomMemberDataInternal;
    std::uint8_t eventCause;
    std::uint8_t pad[7];
    std::uint8_t optData[16];
    std::uint64_t optDataLen;
};

struct RequestOptParam {
    void* cbFunc;
    void* cbFuncArg;
    std::uint32_t timeout;
    std::uint16_t appReqId;
    std::uint16_t pad;
};

// RoomUpdateInfo, the data of KICKEDOUT and ROOM_DESTROYED: the game reads
// optData only when it is 4 bytes long - the reason the kicking host passed
// (sub_10c3da0 -> sub_1098070).
struct RoomUpdateInfo {
    std::uint8_t eventCause;
    std::uint8_t pad[3];
    std::int32_t errorCode;
    std::uint8_t optData[16];
    std::uint64_t optDataLen;
};
static_assert(offsetof(RoomUpdateInfo, optData) == 8 && offsetof(RoomUpdateInfo, optDataLen) == 0x18);

// KickoutRoomMemberRequest: roomId, the member, blockKickFlag, optData.
struct KickoutRequest {
    std::uint64_t roomId;
    std::uint16_t memberId;
    std::uint8_t blockKickFlag;
    std::uint8_t pad[5];
    std::uint8_t optData[16];
    std::uint64_t optDataLen;
};
static_assert(offsetof(KickoutRequest, optData) == 0x10 && offsetof(KickoutRequest, optDataLen) == 0x20);

struct SignalingGetPingInfoResponse {
    std::uint16_t serverId;
    std::uint16_t pad;
    std::uint32_t worldId;
    std::uint64_t roomId;
    std::uint32_t rtt;  // microseconds
    std::uint32_t pad2;
};

// Events (SceNpMatching2Event).
constexpr std::uint16_t kEvGetWorldInfoList = 0x0002;
constexpr std::uint16_t kEvSetRoomDataExternal = 0x0004;
constexpr std::uint16_t kEvCreateJoinRoom = 0x0101;
constexpr std::uint16_t kEvJoinRoom = 0x0102;
constexpr std::uint16_t kEvLeaveRoom = 0x0103;
constexpr std::uint16_t kEvKickoutRoomMember = 0x0104;
constexpr std::uint16_t kEvGrantRoomOwner = 0x0105;
constexpr std::uint16_t kEvSearchRoom = 0x0106;
constexpr std::uint16_t kEvSetRoomDataInternal = 0x0109;
constexpr std::uint16_t kEvSetRoomMemberDataInternal = 0x010B;
constexpr std::uint16_t kEvSignalingGetPingInfo = 0x0E01;
constexpr std::uint16_t kEvMemberJoined = 0x1101;
constexpr std::uint16_t kEvMemberLeft = 0x1102;
constexpr std::uint16_t kEvKickedOut = 0x1103;
constexpr std::uint16_t kEvRoomDestroyed = 0x1104;
constexpr std::uint16_t kEvSignalingDead = 0x5101;
constexpr std::uint16_t kEvSignalingEstablished = 0x5102;
constexpr std::uint16_t kEvContextStarted = 0x6F02;

// SceNpMatching2EventCause.
constexpr std::uint8_t kCauseLeave = 1, kCauseKickout = 2, kCauseServerOperation = 4, kCauseMemberDisappeared = 5;

constexpr int kErrInvalidArg = static_cast<int>(0x80550C03);
constexpr int kErrContext = static_cast<int>(0x80550C06);
constexpr int kErrServerNotAvailable = static_cast<int>(0x80550C28);
// The signaling DEAD event's error: the peer left, or we did.
constexpr int kErrTerminatedByPeer = static_cast<int>(0x80550E10);
constexpr int kErrTerminatedByMyself = static_cast<int>(0x80550E18);

// ---- state ----------------------------------------------------------------

// A response the game reads during its callback. sub_10c3210 copies the room
// id and walks the member list inside the call, so keeping the last few
// alive is generous.
struct Held {
    std::vector<std::uint8_t> bytes;
};

struct M2 {
    std::mutex mu;
    unsigned ctx_id = 0;
    bool started = false;
    void* ctx_cb = nullptr;
    void* ctx_arg = nullptr;
    void* room_cb = nullptr;
    void* room_arg = nullptr;
    void* sig_cb = nullptr;
    void* sig_arg = nullptr;
    void* lobby_cb = nullptr;
    void* lobby_arg = nullptr;
    RequestOptParam dflt{};
    std::atomic<unsigned> next_req{1};
    // the room
    std::string session_id;
    std::uint64_t room_id = 0;
    std::uint16_t member_id = 0;
    std::uint16_t owner_id = 0;
    std::uint16_t max_slot = 0;
    bool is_host = false;
    bool in_room = false;
    // The room a JoinRoom is on its way into: its member events can come in
    // on the poller before the answer is in.
    std::uint64_t joining_room = 0;
    std::int64_t room_since_ms = 0;
    // LeaveRoom, deferred until the session ends
    bool leave_pending = false;
    unsigned leave_req = 0;
    RequestOptParam leave_opt{};
    std::vector<std::unique_ptr<Held>> held;
    std::string npid;  // our handle
};
M2 g;

bool g_trace = [] {
    const char* e = std::getenv("BBHOST_NP_TRACE");
    return e && e[0] == '1';
}();

}  // namespace

bool hle_np_context_started() {
    std::lock_guard<std::mutex> lk(g.mu);
    return g.started;
}

std::string hle_np_online_status() {
    std::string s = "ログイン中: " + net::online_id();
    if (net::account_logged_in()) s += " （アカウント）";
    else if (config().online_require_account) s = "未ログイン: 下でログインしてください";
    if (g.in_room) {
        s += " / ルーム " + std::to_string(g.room_id) + " / メンバー " + std::to_string(g.member_id) +
             (g.is_host ? " （ホスト）" : " （ゲスト）") + ", " + std::to_string(net::peers_all().size()) + " 人接続";
    } else {
        s += " / ルーム未参加";
    }
    return s;
}

namespace {

// Never trust a null callback: the game registers all of them, but a
// scheduled event can outlive a DestroyContext.
void fire_request(void* fn, void* arg, unsigned req, std::uint16_t event, int error, void* data) {
    if (!fn) return;
    if (g_trace || event != kEvGetWorldInfoList) {
        host_log("np: REQUEST_CB event 0x%x req %u error 0x%x", event, req, static_cast<unsigned>(error));
    }
    hle_call_guest<std::int64_t>(fn, g.ctx_id, req, event, error, data, arg);
}
// A callback read under g.mu and made once it is released. The game's
// handlers take their own locks (the SessionOwner lock)
// and call back into sceNpMatching2, whose entry points take g.mu: made under
// it, a callback could deadlock against a game thread holding the
// SessionOwner lock, or against itself.
struct M2Call {
    void* fn = nullptr;
    std::int64_t a[6] = {};
    void run() const {
        if (fn) hle_call_guest6(fn, a[0], a[1], a[2], a[3], a[4], a[5]);
    }
};
M2Call request_call(void* fn, void* arg, unsigned req, std::uint16_t event, int error, void* data) {
    if (!fn) return {};
    if (g_trace || event != kEvGetWorldInfoList) {
        host_log("np: REQUEST_CB event 0x%x req %u error 0x%x", event, req, static_cast<unsigned>(error));
    }
    return {fn, {static_cast<std::int64_t>(g.ctx_id), static_cast<std::int64_t>(req), static_cast<std::int64_t>(event), error,
                 reinterpret_cast<std::int64_t>(data), reinterpret_cast<std::int64_t>(arg)}};
}
M2Call context_call(std::uint16_t event, int cause, int error) {
    if (!g.ctx_cb) return {};
    host_log("np: CONTEXT_CB event 0x%x", event);
    return {g.ctx_cb, {static_cast<std::int64_t>(g.ctx_id), static_cast<std::int64_t>(event), cause, error,
                       reinterpret_cast<std::int64_t>(g.ctx_arg), 0}};
}
M2Call room_event_call(std::uint16_t event, void* data, std::uint64_t room_id) {
    if (!g.room_cb) return {};
    host_log("np: ROOM_EVENT_CB event 0x%x room %llu", event, static_cast<unsigned long long>(room_id));
    return {g.room_cb, {static_cast<std::int64_t>(g.ctx_id), static_cast<std::int64_t>(room_id), static_cast<std::int64_t>(event),
                        reinterpret_cast<std::int64_t>(data), reinterpret_cast<std::int64_t>(g.room_arg), 0}};
}
M2Call room_event_call(std::uint16_t event, void* data) { return room_event_call(event, data, g.room_id); }
void fire_signaling(std::uint16_t member, std::uint16_t event, int error, std::uint64_t room_id) {
    if (!g.sig_cb) return;
    if (g_trace) host_log("np: SIGNALING_CB event 0x%x member %u", event, member);
    hle_call_guest<std::int64_t>(g.sig_cb, g.ctx_id, room_id, member, event, error, g.sig_arg);
}
void fire_signaling(std::uint16_t member, std::uint16_t event, int error) { fire_signaling(member, event, error, g.room_id); }

// The room's real members (provisional 0xff00+ entries are the same people,
// resolved before they joined).
std::vector<net::Peer> room_peers() {
    std::vector<net::Peer> v;
    for (const net::Peer& p : net::peers_all()) {
        if (p.member_id < 0xff00) v.push_back(p);
    }
    return v;
}

// The ESTABLISHED event for every member the game must see before
// signalingPoll runs (ConnObj+0xd8), self included.
void schedule_established_for_all(int delay_ms) {
    std::vector<std::uint16_t> ids;
    for (const net::Peer& p : room_peers()) ids.push_back(p.member_id);
    ids.push_back(g.member_id);
    net::dispatch_after(delay_ms, net::Prio::Signaling, [ids] {
        for (std::uint16_t m : ids) fire_signaling(m, kEvSignalingEstablished, 0);
    });
}

// Members gone for good: the signaling DEAD for each first, then the room
// event, as the SDK delivers them (DEAD 0x5101 before MEMBER_LEFT).
void schedule_dead(const std::vector<std::uint16_t>& ids, int error, std::uint64_t room_id, int delay_ms) {
    if (ids.empty()) return;
    net::dispatch_after(delay_ms, net::Prio::Signaling, [ids, error, room_id] {
        for (std::uint16_t m : ids) fire_signaling(m, kEvSignalingDead, error, room_id);
    });
}

// Builds a RoomDataInternal + member list (self + every peer) into one Held.
CreateJoinRoomResponse* build_room_response(std::uint64_t room_id, std::uint16_t self_id, std::uint16_t owner_id,
                                            std::uint16_t max_slot) {
    const std::vector<net::Peer> peers = room_peers();
    const std::size_t n = peers.size() + 1;
    auto held = std::make_unique<Held>();
    held->bytes.assign(sizeof(CreateJoinRoomResponse) + sizeof(RoomDataInternal) + n * sizeof(RoomMemberDataInternal),
                       0);
    auto* resp = reinterpret_cast<CreateJoinRoomResponse*>(held->bytes.data());
    auto* room = reinterpret_cast<RoomDataInternal*>(resp + 1);
    auto* members = reinterpret_cast<RoomMemberDataInternal*>(room + 1);
    room->maxSlot = max_slot ? max_slot : 5;
    room->publicSlots = room->maxSlot;
    room->openPublicSlots = static_cast<std::uint16_t>(room->maxSlot > n ? room->maxSlot - n : 0);
    room->serverId = 1;
    room->worldId = 1;
    room->lobbyId = 1;
    room->roomId = room_id;
    std::uint64_t joined = 0;
    std::size_t i = 0;
    auto put = [&](std::uint16_t member_id, const std::string& online) {
        RoomMemberDataInternal& m = members[i];
        m.next = i + 1 < n ? &members[i + 1] : nullptr;
        hle_np_fill_npid(m.npId, online.c_str());
        m.memberId = member_id;
        m.natType = 1;
        if (member_id && member_id <= 64) joined |= 1ull << (member_id - 1);
        if (member_id == self_id) resp->members.me = &m;
        if (member_id == owner_id) {
            resp->members.owner = &m;
            m.flagAttr |= 0x80000000u;  // SCE_NP_MATCHING2_ROOMMEMBER_FLAG_ATTR_OWNER
        }
        ++i;
    };
    put(self_id, net::online_id());
    for (const net::Peer& p : peers) put(p.member_id, p.online_id);
    room->joinedSlotMask = joined;
    resp->roomData = room;
    resp->members.members = members;
    resp->members.membersNum = n;
    if (!resp->members.owner) resp->members.owner = resp->members.me;
    g.held.push_back(std::move(held));
    if (g.held.size() > 8) g.held.erase(g.held.begin());
    return resp;
}

RoomMemberUpdateInfo* build_member_update(const net::Peer& p, std::uint8_t cause) {
    auto held = std::make_unique<Held>();
    held->bytes.assign(sizeof(RoomMemberUpdateInfo) + sizeof(RoomMemberDataInternal), 0);
    auto* info = reinterpret_cast<RoomMemberUpdateInfo*>(held->bytes.data());
    auto* m = reinterpret_cast<RoomMemberDataInternal*>(info + 1);
    hle_np_fill_npid(m->npId, p.online_id.c_str());
    m->memberId = p.member_id;
    m->natType = 1;
    if (p.member_id == g.owner_id) m->flagAttr |= 0x80000000u;
    info->roomMemberDataInternal = m;
    info->eventCause = cause;
    g.held.push_back(std::move(held));
    if (g.held.size() > 8) g.held.erase(g.held.begin());
    return info;
}

RoomUpdateInfo* build_room_update(std::uint8_t cause, const std::uint8_t* opt, std::size_t opt_len) {
    auto held = std::make_unique<Held>();
    held->bytes.assign(sizeof(RoomUpdateInfo), 0);
    auto* info = reinterpret_cast<RoomUpdateInfo*>(held->bytes.data());
    info->eventCause = cause;
    opt_len = opt_len > sizeof(info->optData) ? sizeof(info->optData) : opt_len;
    if (opt && opt_len) std::memcpy(info->optData, opt, opt_len);
    info->optDataLen = opt_len;
    g.held.push_back(std::move(held));
    if (g.held.size() > 8) g.held.erase(g.held.begin());
    return info;
}

RequestOptParam opt_or_default(const RequestOptParam* opt) {
    RequestOptParam o = opt && opt->cbFunc ? *opt : g.dflt;
    return o;
}

// A guest that played by its host's rules goes home to its own: the plugins
// rewrite what the home world's load reads (~0.3 s for the randomizer's 22
// layouts). Not under g.mu - the poller, heartbeat and requests wait on it -
// and before the game hears LeaveRoom and starts that load.
std::atomic<bool> g_restore_world{false};
void restore_world_if_pending() {
    if (g_restore_world.exchange(false)) plugins_enter_world("");
}

void forget_room_locked() {
    g.in_room = false;
    g.session_id.clear();
    g.room_id = 0;
    g.owner_id = 0;
    g.max_slot = 0;
    net::peers_clear();
    net::room_clear();
    hle_np_signaling_reset();
}

// Our LeaveRoom (or the context going away): out of the server's room,
// DEAD for every member, then 0x103 at +300 ms. The leave goes to the server
// from the dispatcher, ahead of the DEADs and 0x103 queued after it: until
// 2026-10-25 LeaveRoom's caller, a game thread, waited here under g.mu for
// the server's answer (a round trip, the whole 5 s timeout for a slow server).
void end_session_locked(const char* why) {
    if (!g.in_room && !g.leave_pending) return;
    g_restore_world = true;
    host_log("np: session over (%s): room %llu", why, static_cast<unsigned long long>(g.room_id));
    if (!g.session_id.empty()) {
        const std::string sid = g.session_id;
        const int mid = g.member_id;
        net::dispatch_after(0, net::Prio::Context, [sid, mid] {
            json::Value reply;
            std::string err;
            if (!net::server_leave_room(sid, mid, reply, err)) host_log("np: leave_room failed: %s", err.c_str());
        });
    }
    std::vector<std::uint16_t> ids;
    for (const net::Peer& p : room_peers()) ids.push_back(p.member_id);
    schedule_dead(ids, kErrTerminatedByMyself, g.room_id, 0);
    if (g.leave_pending) {
        const unsigned req = g.leave_req;
        const RequestOptParam opt = g.leave_opt;
        g.leave_pending = false;
        net::dispatch_after(300, net::Prio::Request, [req, opt] {
            restore_world_if_pending();   // the home world's layouts first, then the game may load it
            fire_request(opt.cbFunc, opt.cbFuncArg, req, kEvLeaveRoom, 0, nullptr);
        });
    } else {
        net::dispatch_after(0, net::Prio::Request, [] { restore_world_if_pending(); });
    }
    forget_room_locked();
}

// The server ended our membership (the host left or timed out, we were
// kicked, the server lost the room): the game still holds the room until
// it hears so - KICKEDOUT or ROOM_DESTROYED, after a DEAD per member.
// Before 2026-10-06 the room was dropped here silently, and a host whose
// room the server had closed went on summoning players into it: every
// join failed "Room not found" (dev, room 1059).
void room_gone_locked(std::uint16_t event, std::uint8_t cause, const std::uint8_t* opt, std::size_t opt_len,
                      const std::string& why) {
    if (!g.in_room) return;
    const std::uint64_t room = g.room_id;
    host_log("np: room %llu gone (%s): the game hears 0x%x", static_cast<unsigned long long>(room), why.c_str(), event);
    g_restore_world = true;
    std::vector<std::uint16_t> ids;
    for (const net::Peer& p : room_peers()) {
        ids.push_back(p.member_id);
        hle_np_signaling_peer_dead(p.online_id);
    }
    schedule_dead(ids, kErrTerminatedByPeer, room, 0);
    RoomUpdateInfo* info = build_room_update(cause, opt, opt_len);
    net::dispatch_after(100, net::Prio::RoomEvent, [event, info, room] {
        restore_world_if_pending();
        M2Call c;
        {
            std::lock_guard<std::mutex> lk2(g.mu);
            c = room_event_call(event, info, room);
        }
        c.run();
    });
    forget_room_locked();
}

// The server drops a member whose heartbeat is 15 s old (its
// HEARTBEAT_TIMEOUT); one every 5 s while we are in a room. Nothing sent
// these before 2026-09-22, so every session ended 15 s after its join. A
// server that answers we are not in the room any more (it dropped the room:
// a restart, our own heartbeat lapsing) ends the room for the game.
void heartbeat_loop() {
    std::string sid;
    int mid = 0;
    {
        std::lock_guard<std::mutex> lk(g.mu);
        if (g.in_room && !g.session_id.empty()) {
            sid = g.session_id;
            mid = g.member_id;
        }
    }
    if (!sid.empty()) {
        const int r = net::server_heartbeat(sid, mid);
        if (r < 0 && g_trace) host_log("np: heartbeat not acknowledged");
        if (r == 0) {
            std::lock_guard<std::mutex> lk(g.mu);
            if (g.in_room && g.session_id == sid) {
                room_gone_locked(kEvRoomDestroyed, kCauseServerOperation, nullptr, 0, "the server has no room for us");
            }
        }
    }
    net::dispatch_after(5000, net::Prio::Request, heartbeat_loop);
}

std::vector<std::uint8_t> b64_decode(const std::string& in) {
    std::vector<std::uint8_t> out;
    std::uint32_t acc = 0;
    int bits = 0;
    for (char ch : in) {
        int v;
        if (ch >= 'A' && ch <= 'Z') v = ch - 'A';
        else if (ch >= 'a' && ch <= 'z') v = ch - 'a' + 26;
        else if (ch >= '0' && ch <= '9') v = ch - '0' + 52;
        else if (ch == '+') v = 62;
        else if (ch == '/') v = 63;
        else continue;
        acc = (acc << 6) | static_cast<std::uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>(acc >> bits));
        }
    }
    return out;
}

bool is_loopback(std::uint32_t addr_nbo) { return (addr_nbo & 0xff) == 127; }
bool server_is_local() {
    const std::string base = net::np_server_base();
    return base.find("://127.") != std::string::npos || base.find("://localhost") != std::string::npos;
}

// ---- events from the server -----------------------------------------------

void on_server_event(const json::Value& ev);

// An event about the room a JoinRoom is still answering for: handled again
// once the answer is in (the poller can be quicker than the join's reply).
bool defer_for_join_locked(const json::Value& ev) {
    const auto room = static_cast<std::uint64_t>(net::int_of(ev, "RoomId", 0));
    if (g.in_room || !room || room != g.joining_room) return false;
    const int tries = static_cast<int>(net::int_of(ev, "_Deferred", 0));
    if (tries >= 20) return false;
    json::Value again = ev;
    again.set("_Deferred", tries + 1);
    net::dispatch_after(250, net::Prio::RoomEvent, [again] { on_server_event(again); });
    return true;
}

void on_server_event(const json::Value& ev) {
    const std::string name = net::str_of(ev, "Name");
    if (name == "guest_invite") {
        std::lock_guard<std::mutex> lk(g.mu);
        const std::uint64_t room = static_cast<std::uint64_t>(net::int_of(ev, "RoomId", 0));
        net::Peer host;
        host.member_id = 1;
        host.online_id = net::str_of(ev, "HostOnlineId");
        // The host's addresses as the server knows them: the peer table
        // takes them now so JoinRoom's answer only confirms, and the punch
        // opens our NAT toward the host before its first datagram. A server
        // before 2026-10-06 sent the host's self-reported 127.0.0.1 here,
        // which pointed us at ourselves: such an invite seeds nothing, and
        // the host is resolved when the game asks for it.
        json::Value one = json::Value::make_array();
        json::Value rec = json::Value::make_object();
        rec.set("MemberId", 1);
        rec.set("OnlineId", host.online_id);
        rec.set("Addr", net::str_of(ev, "HostAddr"));
        rec.set("Port", net::int_of(ev, "HostPort", 0));
        rec.set("LocalAddr", net::str_of(ev, "HostLocalAddr"));
        rec.set("LocalPort", net::int_of(ev, "HostLocalPort", 0));
        rec.set("MappedAddr", net::str_of(ev, "HostMappedAddr"));
        rec.set("MappedPort", net::int_of(ev, "HostMappedPort", 0));
        one.push(rec);
        net::Peer before;
        const bool had = net::peers_get(1, &before);
        net::peers_from_members(one);
        net::Peer seeded;
        if (net::peers_get(1, &seeded) && is_loopback(seeded.addr) && !server_is_local()) {
            if (had) net::peers_upsert(before);
            else net::peers_erase(1);
            host_log("np: guest_invite: the host's address is loopback (an older server); not seeded");
        }
        SummonInvite inv{};
        inv.room_id = room;
        inv.member_tag = static_cast<std::uint32_t>(net::int_of(ev, "MemberTag", 0));
        inv.host_area = static_cast<std::uint32_t>(net::int_of(ev, "HostArea", 0));
        inv.host_level = static_cast<std::uint32_t>(net::int_of(ev, "HostLevel", 0));
        const json::Value* pos = ev.find("HostPos");
        if (pos && pos->type == json::Value::Type::Array) {
            for (std::size_t i = 0; i < 3 && i < pos->array.size(); ++i) {
                inv.host_pos[i] = static_cast<float>(pos->array[i].number);
            }
        }
        inv.host_online_id = host.online_id;
        host_log("np: guest_invite: room %llu from %s (%s:%lld)", static_cast<unsigned long long>(room),
                 host.online_id.c_str(), net::str_of(ev, "HostAddr").c_str(), net::int_of(ev, "HostPort", 0));
        // Hand the game its type-1 item from the dispatcher thread (guest code).
        net::dispatch_after(0, net::Prio::RoomEvent, [inv] { summon_invite_deliver(inv); });
        return;
    }
    if (name == "room_member_joined") {
        std::lock_guard<std::mutex> lk(g.mu);
        const auto mid = static_cast<std::uint16_t>(net::int_of(ev, "MemberId", 0));
        if (defer_for_join_locked(ev)) return;
        if (!g.in_room || !mid || mid == g.member_id) return;
        if (static_cast<std::uint64_t>(net::int_of(ev, "RoomId", 0)) != g.room_id) return;
        // A member our JoinRoom answer already listed (a server before
        // 2026-10-06 announced the existing guests to a joiner again): the
        // address is news, the member is not - a second MEMBER_JOINED would
        // make the game build it twice.
        net::Peer known;
        const bool already = net::peers_get(mid, &known) && known.online_id == net::str_of(ev, "OnlineId");
        json::Value one = json::Value::make_array();
        one.push(ev);
        if (net::peers_from_members(one) == 0) return;
        net::Peer p;
        if (!net::peers_get(mid, &p)) return;
        hle_np_signaling_peer_known(p.member_id);
        if (already) {
            if (g_trace) host_log("np: member %u (%s) already known; address refreshed", mid, p.online_id.c_str());
            return;
        }
        host_log("np: member %u (%s) joined room %llu", p.member_id, p.online_id.c_str(),
                 static_cast<unsigned long long>(g.room_id));
        net::dispatch_after(300, net::Prio::RoomEvent, [p] {
            M2Call c;
            {
                std::lock_guard<std::mutex> lk2(g.mu);
                c = room_event_call(kEvMemberJoined, build_member_update(p, 0));
            }
            c.run();
        });
        net::dispatch_after(800, net::Prio::Signaling, [mid] {
            fire_signaling(mid, kEvSignalingEstablished, 0);
            fire_signaling(g.member_id, kEvSignalingEstablished, 0);
        });
        return;
    }
    if (name == "room_member_left") {
        std::lock_guard<std::mutex> lk(g.mu);
        const auto mid = static_cast<std::uint16_t>(net::int_of(ev, "MemberId", 0));
        if (defer_for_join_locked(ev)) return;
        if (static_cast<std::uint64_t>(net::int_of(ev, "RoomId", 0)) != g.room_id) return;
        net::Peer p;
        if (!net::peers_get(mid, &p)) return;
        const std::string reason = net::str_of(ev, "Reason");
        const std::uint8_t cause = reason == "kicked"                           ? kCauseKickout
                                   : (reason == "leave_room" || reason.empty()) ? kCauseLeave
                                                                                : kCauseMemberDisappeared;
        host_log("np: member %u (%s) left (%s)", mid, p.online_id.c_str(), reason.c_str());
        // DEAD for that member, then MEMBER_LEFT; the room stays ours. The
        // host used to leave the server's room here when one or fewer peers
        // were left: with three in a room one leaving took the third out
        // with it, and the host's game summoned into a room the server no
        // longer had (dev, 2026-10-05).
        schedule_dead({mid}, kErrTerminatedByPeer, g.room_id, 0);
        net::dispatch_after(100, net::Prio::RoomEvent, [p, cause] {
            M2Call c;
            {
                std::lock_guard<std::mutex> lk2(g.mu);
                c = room_event_call(kEvMemberLeft, build_member_update(p, cause));
            }
            c.run();
        });
        hle_np_signaling_peer_dead(p.online_id);
        net::peers_erase(mid);
        // Its provisional entry (resolved before it joined) goes too: a
        // return resolves it afresh.
        for (const net::Peer& q : net::peers_all()) {
            if (q.member_id >= 0xff00 && q.online_id == p.online_id) net::peers_erase(q.member_id);
        }
        return;
    }
    if (name == "peer_deactivated") {
        // The server timed the peer out; its connection is dead for us too.
        std::string who = net::str_of(ev, "OnlineId");
        if (who.empty()) who = net::str_of(ev, "PeerOnlineId");
        if (!who.empty()) hle_np_signaling_peer_dead(who);
        return;
    }
    if (name == "room_closed") {
        std::lock_guard<std::mutex> lk(g.mu);
        const auto room = static_cast<std::uint64_t>(net::int_of(ev, "RoomId", 0));
        if (room && room != g.room_id) return;
        const std::string reason = net::str_of(ev, "Reason");
        room_gone_locked(kEvRoomDestroyed, reason == "host_left" ? kCauseLeave : kCauseServerOperation, nullptr, 0,
                         "room closed: " + reason);
        return;
    }
    if (name == "room_member_kicked") {
        std::lock_guard<std::mutex> lk(g.mu);
        const auto room = static_cast<std::uint64_t>(net::int_of(ev, "RoomId", 0));
        const auto mid = static_cast<std::uint16_t>(net::int_of(ev, "MemberId", 0));
        if (room != g.room_id || mid != g.member_id) return;
        const std::vector<std::uint8_t> opt = b64_decode(net::str_of(ev, "OptData"));
        room_gone_locked(kEvKickedOut, kCauseKickout, opt.data(), opt.size(), "kicked by the host");
        return;
    }
    if (g_trace) host_log("np: event %s ignored", name.c_str());
}

// ---- the imports ----------------------------------------------------------

GUEST_ABI int m2_initialize(const void*) {
    host_log("sceNpMatching2Initialize");
    return 0;
}
GUEST_ABI int m2_terminate() {
    host_log("sceNpMatching2Terminate");
    std::lock_guard<std::mutex> lk(g.mu);
    net::poller_stop();
    g.started = false;
    return 0;
}
// Bloodborne's two-argument form: (const CreateContextParam*, ctxId*). The
// param's first field is the SceNpId pointer.
GUEST_ABI int m2_create_context(const void* param, unsigned* ctx) {
    if (!ctx) return kErrInvalidArg;
    std::lock_guard<std::mutex> lk(g.mu);
    if (param) {
        const void* npid = nullptr;
        std::memcpy(&npid, param, sizeof(npid));
        if (npid) g.npid = hle_np_npid_text(npid);
    }
    if (!g.ctx_id) g.ctx_id = 1;
    *ctx = g.ctx_id;
    host_log("sceNpMatching2CreateContext (%s) -> %u", g.npid.c_str(), g.ctx_id);
    return 0;
}
GUEST_ABI int m2_register_context_cb(void* fn, void* arg) {
    std::lock_guard<std::mutex> lk(g.mu);
    g.ctx_cb = fn;
    g.ctx_arg = arg;
    return 0;
}
GUEST_ABI int m2_context_start(unsigned ctx, unsigned) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (ctx != g.ctx_id) return kErrContext;
    if (g.started) return 0;
    g.started = true;
    host_log("sceNpMatching2ContextStart -> server %s", net::np_server_base().c_str());
    summon_invite_set_signaling_gate();
    net::poller_start(on_server_event);
    static bool heartbeat_started = false;
    if (!heartbeat_started) {
        heartbeat_started = true;
        net::dispatch_after(5000, net::Prio::Request, heartbeat_loop);
    }
    // Register with the server off the game's thread; the started event
    // follows at the contract's delay whether or not the server answered,
    // since an unreachable server should read as "online but alone", not a
    // stuck boot.
    net::dispatch_after(0, net::Prio::Context, [] {
        json::Value reply;
        std::string err;
        if (!net::server_context_start(reply, err)) {
            host_log("np: context_start failed: %s", err.c_str());
        }
    });
    net::dispatch_after(200, net::Prio::Context, [] {
        M2Call c;
        {
            std::lock_guard<std::mutex> lk2(g.mu);
            c = context_call(kEvContextStarted, 0, 0);
        }
        c.run();
    });
    return 0;
}
GUEST_ABI int m2_context_stop(unsigned) {
    std::lock_guard<std::mutex> lk(g.mu);
    net::poller_stop();
    g.started = false;
    return 0;
}
GUEST_ABI int m2_destroy_context(unsigned) {
    std::lock_guard<std::mutex> lk(g.mu);
    net::poller_stop();
    end_session_locked("context destroyed");
    g.started = false;
    g.ctx_cb = g.room_cb = g.sig_cb = g.lobby_cb = nullptr;
    return 0;
}
GUEST_ABI int m2_abort_context_start(unsigned) { return 0; }
GUEST_ABI int m2_get_server_id(unsigned ctx, std::uint16_t* id) {
    if (!id) return kErrInvalidArg;
    if (ctx != g.ctx_id) return kErrContext;
    *id = 1;
    return 0;
}
GUEST_ABI int m2_set_default_opt(unsigned, const RequestOptParam* opt) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (opt) g.dflt = *opt;
    return 0;
}
GUEST_ABI int m2_register_room_event_cb(unsigned, void* fn, void* arg) {
    std::lock_guard<std::mutex> lk(g.mu);
    g.room_cb = fn;
    g.room_arg = arg;
    return 0;
}
GUEST_ABI int m2_register_signaling_cb(unsigned, void* fn, void* arg) {
    std::lock_guard<std::mutex> lk(g.mu);
    g.sig_cb = fn;
    g.sig_arg = arg;
    return 0;
}
GUEST_ABI int m2_register_lobby_event_cb(unsigned, void* fn, void* arg) {
    std::lock_guard<std::mutex> lk(g.mu);
    g.lobby_cb = fn;
    g.lobby_arg = arg;
    return 0;
}

GUEST_ABI int m2_get_world_info_list(unsigned ctx, const void*, const RequestOptParam* opt, unsigned* req_out) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (ctx != g.ctx_id) return kErrContext;
    const unsigned req = g.next_req++;
    if (req_out) *req_out = req;
    const RequestOptParam o = opt_or_default(opt);
    net::dispatch_after(100, net::Prio::Request, [req, o] {
        M2Call c;
        std::unique_lock<std::mutex> lk2(g.mu);
        auto held = std::make_unique<Held>();
        held->bytes.assign(sizeof(GetWorldInfoListResponse) + sizeof(World), 0);
        auto* resp = reinterpret_cast<GetWorldInfoListResponse*>(held->bytes.data());
        auto* w = reinterpret_cast<World*>(resp + 1);
        w->worldId = 1;
        w->numOfLobby = 1;
        w->maxNumOfTotalLobby = 1;
        w->maxNumOfRoom = 1000;
        w->numOfRoom = 1000;
        resp->world = w;
        resp->worldNum = 1;
        g.held.push_back(std::move(held));
        if (g.held.size() > 8) g.held.erase(g.held.begin());
        c = request_call(o.cbFunc, o.cbFuncArg, req, kEvGetWorldInfoList, 0, resp);
        lk2.unlock();
        c.run();
    });
    return 0;
}

// CreateJoinRoomRequest: maxSlot u16 at +0, worldId u32 at +0xc.
GUEST_ABI int m2_create_join_room(unsigned ctx, const std::uint8_t* reqp, const RequestOptParam* opt,
                                  unsigned* req_out) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (ctx != g.ctx_id) return kErrContext;
    const unsigned req = g.next_req++;
    if (req_out) *req_out = req;
    std::uint16_t max_slot = 5;  // Bloodborne asks 5: host, two cooperators, two invaders
    if (reqp) std::memcpy(&max_slot, reqp, 2);
    const RequestOptParam o = opt_or_default(opt);
    host_log("sceNpMatching2CreateJoinRoom req %u maxSlot %u%s", req, max_slot,
             g.in_room ? " (the game left its room without LeaveRoom; the server closes it)" : "");
    net::dispatch_after(0, net::Prio::Request, [req, o, max_slot] {
        json::Value extra = json::Value::make_object();
        summon_invite_host_extra(extra);
        json::Value reply;
        std::string err;
        std::uint64_t room = 0;
        std::string sid;
        int error = 0;
        if (net::server_create_room(max_slot, extra, reply, err) && net::int_of(reply, "ResKind", -1) == 0) {
            room = static_cast<std::uint64_t>(net::int_of(reply, "RoomId", 0));
            sid = net::str_of(reply, "SessionId");
        } else {
            host_log("np: create_room failed: %s", err.empty() ? reply.string.c_str() : err.c_str());
            error = static_cast<int>(0x80550C0B);  // SCE_NP_MATCHING2_ERROR_CONNECTION_FAILED-ish: the game reports it
        }
        CreateJoinRoomResponse* resp = nullptr;
        {
            std::lock_guard<std::mutex> lk2(g.mu);
            if (!error) {
                g.session_id = sid;
                g.room_id = room;
                g.member_id = static_cast<std::uint16_t>(net::int_of(reply, "MemberId", 1));
                g.owner_id = g.member_id;
                g.max_slot = static_cast<std::uint16_t>(net::int_of(reply, "MaxMembers", max_slot));
                g.is_host = true;
                g.in_room = true;
                g.room_since_ms = net::now_ms();
                net::peers_clear();
                net::room_set(sid, g.member_id);
                resp = build_room_response(room, g.member_id, g.member_id, max_slot);
                host_log("np: room %llu created, we are member %u", static_cast<unsigned long long>(room),
                         g.member_id);
            }
        }
        net::dispatch_after(300, net::Prio::Request, [req, o, error, resp] {
            fire_request(o.cbFunc, o.cbFuncArg, req, kEvCreateJoinRoom, error, resp);
        });
        if (!error) {
            std::lock_guard<std::mutex> lk2(g.mu);
            schedule_established_for_all(800);
        }
    });
    return 0;
}

// JoinRoomRequest: roomId u64 at +0.
GUEST_ABI int m2_join_room(unsigned ctx, const std::uint8_t* reqp, const RequestOptParam* opt, unsigned* req_out) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (ctx != g.ctx_id) return kErrContext;
    const unsigned req = g.next_req++;
    if (req_out) *req_out = req;
    std::uint64_t room = 0;
    if (reqp) std::memcpy(&room, reqp, 8);
    const RequestOptParam o = opt_or_default(opt);
    host_log("sceNpMatching2JoinRoom req %u room %llu", req, static_cast<unsigned long long>(room));
    g.joining_room = room;
    net::dispatch_after(0, net::Prio::Request, [req, o, room] {
        json::Value reply;
        std::string err;
        int error = 0;
        if (!(net::server_join_room(room, reply, err) && net::int_of(reply, "ResKind", -1) == 0)) {
            host_log("np: join_room failed: %s", err.c_str());
            error = static_cast<int>(0x80550C0B);
        }
        CreateJoinRoomResponse* resp = nullptr;
        {
            std::lock_guard<std::mutex> lk2(g.mu);
            if (g.joining_room == room) g.joining_room = 0;
            if (!error) {
                g.session_id = net::str_of(reply, "SessionId");
                g.room_id = room;
                g.member_id = static_cast<std::uint16_t>(net::int_of(reply, "MemberId", 2));
                // The room's size and owner as the server has them (a server
                // before 2026-10-06 sent neither: the host, and the 5 the game
                // asks for - the 4 answered here before told the guest's game
                // its room held one fewer than the host's).
                g.owner_id = static_cast<std::uint16_t>(net::int_of(reply, "OwnerMemberId", 1));
                g.max_slot = static_cast<std::uint16_t>(net::int_of(reply, "MaxMembers", 5));
                g.is_host = false;
                g.in_room = true;
                g.room_since_ms = net::now_ms();
                // The member list is the room: entries from an invite or an
                // earlier room go (provisional ones stay for their connections).
                for (const net::Peer& p : net::peers_all()) {
                    if (p.member_id < 0xff00) net::peers_erase(p.member_id);
                }
                const json::Value* members = reply.find("Members");
                if (members) net::peers_from_members(*members);
                for (const net::Peer& p : net::peers_all()) hle_np_signaling_peer_known(p.member_id);
                net::room_set(g.session_id, g.member_id);
                resp = build_room_response(room, g.member_id, g.owner_id, g.max_slot);
                host_log("np: joined room %llu as member %u with %zu peers", static_cast<unsigned long long>(room),
                         g.member_id, room_peers().size());
            }
        }
        // The host's rules, when the server says them (HostRuleset) and they
        // are not ours: plugins that can (BB_PLUGIN_ADOPTS_RULES) rewrite the
        // world the guest is about to load - before the game hears it joined.
        if (!error) {
            const std::string host_rules = net::str_of(reply, "HostRuleset");
            if (!host_rules.empty() && host_rules != plugins_ruleset()) plugins_enter_world(host_rules);
        }
        net::dispatch_after(300, net::Prio::Request,
                            [req, o, error, resp] { fire_request(o.cbFunc, o.cbFuncArg, req, kEvJoinRoom, error, resp); });
        if (!error) {
            std::lock_guard<std::mutex> lk2(g.mu);
            schedule_established_for_all(500);
        }
    });
    return 0;
}

GUEST_ABI int m2_leave_room(unsigned ctx, const void*, const RequestOptParam* opt, unsigned* req_out) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (ctx != g.ctx_id) return kErrContext;
    const unsigned req = g.next_req++;
    if (req_out) *req_out = req;
    const RequestOptParam o = opt_or_default(opt);
    // In every pair run (v4-v7) the game called this only when its session
    // was over - the guest gone, or the host giving up - never right after
    // the room existed as the design once feared. Deferring the answer 10
    // minutes left the SessionOwner waiting for 0x103 that long, and a new
    // bell in the meantime found no idle session: the "cannot leave and
    // rejoin" the shadPS4 fork had. So: leave the server's room now, answer
    // 0x103 at +300 ms, and log if it ever comes within 2 s of the room.
    host_log("sceNpMatching2LeaveRoom req %u%s", req,
             g.in_room && net::now_ms() - g.room_since_ms < 2000 ? " (2 s after the room was made!)" : "");
    g.leave_pending = true;
    g.leave_req = req;
    g.leave_opt = o;
    end_session_locked("LeaveRoom");
    return 0;
}

// (ctx, roomId, memberId, status*, SceNetInAddr*, SceInPort_t*): the three values.
GUEST_ABI int m2_signaling_get_connection_status(unsigned ctx, std::uint64_t room, std::uint16_t member,
                                                  int* status, std::uint32_t* addr, std::uint16_t* port) {
    if (ctx != g.ctx_id) return kErrContext;
    net::Peer p;
    const bool known = net::peers_get(member, &p) && p.addr != 0;
    if (status) *status = known ? 2 : 0;
    if (addr) *addr = known ? p.addr : 0;
    if (port) *port = known ? static_cast<std::uint16_t>((p.port << 8) | (p.port >> 8)) : 0;
    if (g_trace) {
        host_log("sceNpMatching2SignalingGetConnectionStatus room %llu member %u -> %d",
                 static_cast<unsigned long long>(room), member, known ? 2 : 0);
    }
    return 0;
}

// The requests the summon path does not need still answer, as the SDK's
// always do: the game's room object marks a request pending until its
// callback (SessionOwner+0x458) and runs nothing else meanwhile - before
// 2026-10-06 these were bound to a function that never answered.
int answer_request(unsigned ctx, const RequestOptParam* opt, unsigned* req_out, std::uint16_t event, int error,
                   const char* what) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (ctx != g.ctx_id) return kErrContext;
    const unsigned req = g.next_req++;
    if (req_out) *req_out = req;
    const RequestOptParam o = opt_or_default(opt);
    host_log("sceNpMatching2%s req %u: answered 0x%x%s", what, req, event, error ? " with an error" : "");
    net::dispatch_after(100, net::Prio::Request, [o, req, event, error] {
        fire_request(o.cbFunc, o.cbFuncArg, req, event, error, nullptr);
    });
    return 0;
}

// The host sends a member away (sub_10c5b70: a 4-byte reason in optData,
// which the kicked game reads from KICKEDOUT). The server takes the member
// out and tells everyone, the kicked one included; without the server the
// room hears it here.
GUEST_ABI int m2_kickout(unsigned ctx, const KickoutRequest* rq, const RequestOptParam* opt, unsigned* req_out) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (ctx != g.ctx_id) return kErrContext;
    if (!rq) return kErrInvalidArg;
    const unsigned req = g.next_req++;
    if (req_out) *req_out = req;
    const RequestOptParam o = opt_or_default(opt);
    const KickoutRequest r = *rq;
    host_log("sceNpMatching2KickoutRoomMember req %u room %llu member %u (%llu-byte reason)", req,
             static_cast<unsigned long long>(r.roomId), r.memberId, static_cast<unsigned long long>(r.optDataLen));
    const std::string sid = g.session_id;
    const int me = g.member_id;
    net::dispatch_after(0, net::Prio::Request, [o, req, r, sid, me] {
        std::string err;
        const std::size_t n = r.optDataLen > sizeof(r.optData) ? sizeof(r.optData) : static_cast<std::size_t>(r.optDataLen);
        const bool sent = !sid.empty() && net::server_kick_member(sid, r.memberId, me, r.optData, n, err);
        if (!sent) {
            host_log("np: kick of member %u not sent (%s); the room hears it here", r.memberId,
                     sid.empty() ? "no room" : err.c_str());
            json::Value ev = json::Value::make_object();
            ev.set("Name", "room_member_left");
            {
                std::lock_guard<std::mutex> lk2(g.mu);
                ev.set("RoomId", static_cast<long long>(g.room_id));
            }
            ev.set("MemberId", static_cast<int>(r.memberId));
            ev.set("Reason", "kicked");
            on_server_event(ev);
        }
        net::dispatch_after(100, net::Prio::Request,
                            [o, req] { fire_request(o.cbFunc, o.cbFuncArg, req, kEvKickoutRoomMember, 0, nullptr); });
    });
    return 0;
}
GUEST_ABI int m2_grant_room_owner(unsigned ctx, const void*, const RequestOptParam* opt, unsigned* req_out) {
    return answer_request(ctx, opt, req_out, kEvGrantRoomOwner, 0, "GrantRoomOwner");
}
GUEST_ABI int m2_set_room_data_internal(unsigned ctx, const void*, const RequestOptParam* opt, unsigned* req_out) {
    return answer_request(ctx, opt, req_out, kEvSetRoomDataInternal, 0, "SetRoomDataInternal");
}
GUEST_ABI int m2_set_room_data_external(unsigned ctx, const void*, const RequestOptParam* opt, unsigned* req_out) {
    return answer_request(ctx, opt, req_out, kEvSetRoomDataExternal, 0, "SetRoomDataExternal");
}
GUEST_ABI int m2_set_room_member_data_internal(unsigned ctx, const void*, const RequestOptParam* opt,
                                               unsigned* req_out) {
    return answer_request(ctx, opt, req_out, kEvSetRoomMemberDataInternal, 0, "SetRoomMemberDataInternal");
}
// No room list here: summons go through the game's own sign server. The
// game's handler reads an error as "nothing found" (sub_10be2a0 posts 0x2a).
GUEST_ABI int m2_search_room(unsigned ctx, const void*, const RequestOptParam* opt, unsigned* req_out) {
    return answer_request(ctx, opt, req_out, kEvSearchRoom, kErrServerNotAvailable, "SearchRoom");
}
// (ctx, {roomId}, opt, reqId): the round trip through the relay, from the
// keepalive's STUN exchange (ours to the server, twice for a peer's).
GUEST_ABI int m2_signaling_get_ping_info(unsigned ctx, const std::uint64_t* rq, const RequestOptParam* opt,
                                         unsigned* req_out) {
    std::lock_guard<std::mutex> lk(g.mu);
    if (ctx != g.ctx_id) return kErrContext;
    const unsigned req = g.next_req++;
    if (req_out) *req_out = req;
    const RequestOptParam o = opt_or_default(opt);
    auto held = std::make_unique<Held>();
    held->bytes.assign(sizeof(SignalingGetPingInfoResponse), 0);
    auto* resp = reinterpret_cast<SignalingGetPingInfoResponse*>(held->bytes.data());
    resp->serverId = 1;
    resp->worldId = 1;
    resp->roomId = rq ? *rq : g.room_id;
    const int rtt = net::stun_rtt_us();
    resp->rtt = static_cast<std::uint32_t>(rtt > 0 ? 2 * rtt : 60000);
    g.held.push_back(std::move(held));
    if (g.held.size() > 8) g.held.erase(g.held.begin());
    net::dispatch_after(50, net::Prio::Request, [o, req, resp] {
        fire_request(o.cbFunc, o.cbFuncArg, req, kEvSignalingGetPingInfo, 0, resp);
    });
    return 0;
}
// Lobbies: never called by this game; bound so a call is visible.
GUEST_ABI int m2_unused(unsigned, const void*, const RequestOptParam*, unsigned* req_out) {
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 4) host_log("np: an unused NpMatching2 lobby request was called");
    if (req_out) *req_out = g.next_req++;
    return 0;
}

}  // namespace

void hle_np_fill_npid(void* out, const char* online) {
    if (!out) return;
    std::memset(out, 0, 36);
    if (!online || !online[0]) online = "Player";
    std::snprintf(static_cast<char*>(out), 17, "%s", online);
}

std::string hle_np_npid_text(const void* npid) {
    if (!npid) return "";
    char buf[17]{};
    std::memcpy(buf, npid, 16);
    return buf;
}

void hle_register_np_matching2() {
    REG("sceNpMatching2Initialize", m2_initialize);
    REG("sceNpMatching2Terminate", m2_terminate);
    REG("sceNpMatching2CreateContext", m2_create_context);
    REG("sceNpMatching2RegisterContextCallback", m2_register_context_cb);
    REG("sceNpMatching2ContextStart", m2_context_start);
    REG("sceNpMatching2ContextStop", m2_context_stop);
    REG("sceNpMatching2DestroyContext", m2_destroy_context);
    REG("sceNpMatching2GetServerId", m2_get_server_id);
    REG("sceNpMatching2SetDefaultRequestOptParam", m2_set_default_opt);
    REG("sceNpMatching2RegisterRoomEventCallback", m2_register_room_event_cb);
    REG("sceNpMatching2RegisterSignalingCallback", m2_register_signaling_cb);
    REG("sceNpMatching2RegisterLobbyEventCallback", m2_register_lobby_event_cb);
    REG("sceNpMatching2GetWorldInfoList", m2_get_world_info_list);
    REG("sceNpMatching2CreateJoinRoom", m2_create_join_room);
    REG("sceNpMatching2JoinRoom", m2_join_room);
    REG("sceNpMatching2LeaveRoom", m2_leave_room);
    REG("sceNpMatching2SignalingGetConnectionStatus", m2_signaling_get_connection_status);
    REG("sceNpMatching2SearchRoom", m2_search_room);
    REG("sceNpMatching2JoinLobby", m2_unused);
    REG("sceNpMatching2LeaveLobby", m2_unused);
    REG("sceNpMatching2GetLobbyInfoList", m2_unused);
    REG("sceNpMatching2KickoutRoomMember", m2_kickout);
    REG("sceNpMatching2GrantRoomOwner", m2_grant_room_owner);
    REG("sceNpMatching2SetRoomDataInternal", m2_set_room_data_internal);
    REG("sceNpMatching2SetRoomDataExternal", m2_set_room_data_external);
    REG("sceNpMatching2SetRoomMemberDataInternal", m2_set_room_member_data_internal);
    REG("sceNpMatching2SignalingGetPingInfo", m2_signaling_get_ping_info);
}
