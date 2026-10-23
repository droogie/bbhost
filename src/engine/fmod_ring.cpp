#include "guest_abi.h"
#include "engine/fmod_ring.h"

#include "core/elf.h"
#include "core/host_clock.h"
#include "core/portable.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "hle/modules.h"
#include "hle/sync.h"
#include "log.h"

#include <pthread.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace {

using namespace fmod_ring;

// FMOD Ex 4.44.50 in eboot 1.09 (Binary Ninja addresses).
// MemPool::free(pool, ptr, file, line): every FMOD free; ptr is the object,
// FMOD's {size, flags} header the 8 bytes before it.
constexpr std::uint64_t kFree = 0x11ab3a0;
constexpr std::uint8_t kFreePrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                          0x41, 0x55, 0x41, 0x54, 0x53, 0x50};
// SoundI::setSoundGroup(sound, group): vtable slot 0x100 of every SoundI class.
constexpr std::uint64_t kSetGroup = 0x11c0880;
constexpr std::uint8_t kSetGroupPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                              0x53, 0x50, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb};
// SoundI::release(sound, freethis): slot 0x28 of SoundI and Stream, the tail
// of SampleSoftware's (0x11ba040) and the bank container's (0x11c3190).
constexpr std::uint64_t kRelease = 0x11be120;
constexpr std::uint8_t kReleasePrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                             0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x18};
// SoundI::SoundI(sound): the base constructor every class's runs.
constexpr std::uint64_t kCtor = 0x11bd0f0;
constexpr std::uint8_t kCtorPrologue[] = {0x48, 0x8d, 0x47, 0x08, 0x48, 0x89, 0x47, 0x08, 0x48, 0x89,
                                          0x47, 0x10, 0x48, 0xc7, 0x47, 0x18, 0x00, 0x00, 0x00, 0x00};
// SoundI::setSubSound's body (parent, index, subsound, ...).
constexpr std::uint64_t kSetSub = 0x11bea30;
constexpr std::uint8_t kSetSubPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                            0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x58, 0x01, 0x00, 0x00};
// SystemI::createSound(system, name_or_data, mode, exinfo, &sound).
constexpr std::uint64_t kCreate = 0x1114de0;
constexpr std::uint8_t kCreatePrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                            0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x28, 0x01, 0x00, 0x00};
// SoundI::release frees its shared subsound (the +0xb0 block) here, after
// asking it to release itself and ignoring the answer (fmod_soundi.cpp line
// 0x367): the return address of that call to MemPool::free.
constexpr std::uint64_t kSharedFreeReturn = 0x11be687;
// FMOD's global critical section: a pointer to the mutex slot it locks.
constexpr std::uint64_t kCrit = 0x586d260;
constexpr std::uint64_t kSoundVtables[4] = {0x56d7640, 0x56d7840, 0x56d7a30, 0x56d7c30};
constexpr std::uint64_t kTextLo = 0x400000, kTextHi = 0x54d96dc;

std::uint64_t g_slide = 0;
Vtables g_vt{};
bool g_guard = false, g_check = false;
std::atomic<std::uint64_t> g_guarded{0};
// The refusing shared subsounds kept (orphan()), with the bytes each holds; one
// freed later (a release was under way on another thread) leaves the map.
std::mutex g_kept_mu;
std::unordered_map<std::uint64_t, std::uint64_t> g_kept;  // under g_kept_mu
std::atomic<std::uint64_t> g_kept_count{0}, g_kept_freed{0}, g_kept_live{0};
// The detector: the sounds events hold as programmer sounds (EventSound::prepare's
// 0x200), and how many events hold each.
std::mutex g_prog_mu;
std::unordered_map<std::uint64_t, std::uint32_t> g_prog;  // under g_prog_mu: a programmer sound -> events holding it

std::uint64_t va(std::uint64_t bn) { return g_slide + (bn - kPreferredGuestSlide); }
std::uint64_t bn(std::uint64_t a) { return a - g_slide + kPreferredGuestSlide; }
bool text(std::uint64_t a) { return a >= va(kTextLo) && a < va(kTextHi); }
bool readable(std::uint64_t a, std::size_t n) { return a >= 0x10000 && hle_kernel_va_readable(a, n); }

// FMOD's critical section, taken from here as FMOD takes it (recursive: the
// hooks run where FMOD may already hold it).
struct Crit {
    void** slot = nullptr;
    explicit Crit(bool take = true) {
        const std::uint64_t c = take ? rd(va(kCrit)) : 0;
        if (c) {
            slot = reinterpret_cast<void**>(static_cast<std::uintptr_t>(c));
            gsync::mutex_lock(slot, nullptr);
        }
    }
    ~Crit() {
        if (slot) gsync::mutex_unlock(slot);
    }
    Crit(const Crit&) = delete;
    Crit& operator=(const Crit&) = delete;
};

void thread_name(char* out, std::size_t n) {
    out[0] = 0;
#if !defined(_WIN32)
    pthread_getname_np(pthread_self(), out, n);
#else
    (void)n;
#endif
}

// Up to `n` eboot code addresses: the hooked function's return address
// (saved[6]: the prologue stub pushed six registers over it), then the first
// code addresses above it on the guest stack.
void callers(const std::uint64_t* saved, std::uint64_t* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = 0;
    out[0] = saved[6];
    int k = 1;
    for (int i = 7; i < 7 + 32 && k < n; ++i) {
        if (text(saved[i])) out[k++] = saved[i];
    }
}

// --- the detector's history -------------------------------------------------
struct Event {
    std::uint64_t ns;
    std::uint32_t tid;
    char kind;  // N born, G linked, R release, S subsound set, C createSound, F freed (sound), X guard unlinked, K kept
    char pad[3];
    std::uint64_t obj, a, b;
    std::uint64_t ret[3];
};
constexpr std::size_t kEvents = 1u << 15;
Event* g_events = nullptr;
std::atomic<std::uint64_t> g_next{0};
std::atomic<std::uint64_t> g_links{0};

void note(char kind, std::uint64_t obj, std::uint64_t a, std::uint64_t b, const std::uint64_t* saved) {
    if (!g_events) return;
    const std::uint64_t i = g_next.fetch_add(1, std::memory_order_relaxed);
    Event& e = g_events[i & (kEvents - 1)];
    e.ns = host_clock_monotonic_ns();
    e.tid = host_thread_id();
    e.kind = kind;
    e.obj = obj;
    e.a = a;
    e.b = b;
    callers(saved, e.ret, 3);
}

void log_event(const char* why, const Event& e) {
    host_log("fmod ring:   %s %c t=%.3f tid %u obj %llx a %llx b %llx from %llx %llx %llx", why, e.kind,
             static_cast<double>(e.ns) * 1e-9, e.tid, static_cast<unsigned long long>(e.obj),
             static_cast<unsigned long long>(e.a), static_cast<unsigned long long>(e.b),
             static_cast<unsigned long long>(e.ret[0] ? bn(e.ret[0]) : 0),
             static_cast<unsigned long long>(e.ret[1] ? bn(e.ret[1]) : 0),
             static_cast<unsigned long long>(e.ret[2] ? bn(e.ret[2]) : 0));
}

// The history of one sound (or of whatever touched the node at obj+0x1d0),
// oldest first, at most `cap` events.
void log_history(std::uint64_t obj, int cap) {
    if (!g_events) return;
    const std::uint64_t end = g_next.load(std::memory_order_relaxed);
    const std::uint64_t begin = end > kEvents ? end - kEvents : 0;
    int found = 0;
    std::uint64_t idx[64];
    for (std::uint64_t i = end; i > begin && found < cap && found < 64; --i) {
        const Event& e = g_events[(i - 1) & (kEvents - 1)];
        if (e.obj == obj || e.a == obj || e.b == obj) idx[found++] = i - 1;
    }
    host_log("fmod ring:   history of %llx: %d event(s)%s", static_cast<unsigned long long>(obj), found,
             found == 0 ? " (older than the history or never seen)" : "");
    for (int k = found - 1; k >= 0; --k) log_event("", g_events[idx[k] & (kEvents - 1)]);
}

void log_recent(int n) {
    if (!g_events) return;
    const std::uint64_t end = g_next.load(std::memory_order_relaxed);
    const std::uint64_t first = end > static_cast<std::uint64_t>(n) ? end - n : 0;
    host_log("fmod ring:   the last %llu events:", static_cast<unsigned long long>(end - first));
    for (std::uint64_t i = first; i < end; ++i) log_event("", g_events[i & (kEvents - 1)]);
}

void describe(const char* what, std::uint64_t obj) {
    char name[32];
    thread_name(name, sizeof(name));
    host_log("fmod ring: %s: sound %llx (vtable %llx, flags %x, open state %u, busy %u, parent %llx subsound %u of %u, "
             "shared %llx, group %llx) on %s (tid %u)",
             what, static_cast<unsigned long long>(obj), static_cast<unsigned long long>(bn(rd(obj))), rd32(obj + kFlags),
             rd32(obj + kOpenState), static_cast<unsigned>(rd(obj + kBusy) & 0xff),
             static_cast<unsigned long long>(rd(obj + kParent)), rd32(obj + kSubIndex), rd32(obj + kNumSubsounds),
             static_cast<unsigned long long>(rd(obj + kShared)), static_cast<unsigned long long>(rd(obj + kGroup)), name,
             host_thread_id());
}

// --- the master ring walk -----------------------------------------------------
std::atomic<std::uint64_t> g_master{0};
std::mutex g_seen_mu;
std::unordered_set<std::uint64_t> g_seen_stale;  // under g_seen_mu

std::uint64_t master_group(std::uint64_t sound) {
    std::uint64_t m = g_master.load(std::memory_order_relaxed);
    if (m) return m;
    const std::uint64_t sys = rd(sound + kSystem);
    if (!readable(sys + kMaster, 8)) return 0;
    m = rd(sys + kMaster);
    if (m) g_master.store(m, std::memory_order_relaxed);
    return m;
}

// Walks the master ring (under FMOD's critical section when `lock`) and
// reports a stale member once.
void check_ring(bool lock, const char* when) {
    const std::uint64_t m = g_master.load(std::memory_order_relaxed);
    if (!m) return;
    std::size_t members = 0;
    std::uint64_t stale = 0, next = 0, prev = 0;
    {
        Crit c(lock);
        stale = first_stale(g_vt, m + kHead, m, 1u << 20, &members, readable);
        if (stale && readable(stale, 24)) {
            next = rd(stale);
            prev = rd(stale + 8);
        }
    }
    static std::atomic<std::uint64_t> walks{0};
    const std::uint64_t w = walks.fetch_add(1) + 1;
    if (w == 1 || (w & (w - 1)) == 0) {
        host_log("fmod ring: master ring walk #%llu (%s): %zu members, %s", static_cast<unsigned long long>(w), when,
                 members, stale ? "a stale member" : "all live");
    }
    if (!stale) return;
    {
        std::lock_guard<std::mutex> lk(g_seen_mu);
        if (!g_seen_stale.insert(stale).second) return;
    }
    host_log("fmod ring: STALE master-ring member %llx (%s, member %zu): next %llx prev %llx data %llx; its sound would be "
             "%llx",
             static_cast<unsigned long long>(stale), when, members, static_cast<unsigned long long>(next),
             static_cast<unsigned long long>(prev),
             static_cast<unsigned long long>(readable(stale + 16, 8) ? rd(stale + 16) : 0),
             static_cast<unsigned long long>(stale - kNode));
    log_history(stale - kNode, 32);
    if (prev) log_history(prev - kNode, 8);
    if (next) log_history(next - kNode, 8);
}

// --- the hooks ------------------------------------------------------------------
GUEST_ABI std::int64_t free_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t p = saved[4];  // rsi
    if (p < 0x10000 || !is_sound(g_vt, p)) return 0;
    if (g_kept_live.load(std::memory_order_relaxed)) {
        // A kept sound freed after all: the release that was under way.
        std::lock_guard<std::mutex> lk(g_kept_mu);
        if (g_kept.erase(p)) {
            g_kept_live.fetch_sub(1, std::memory_order_relaxed);
            const std::uint64_t f = g_kept_freed.fetch_add(1) + 1;
            if (f <= 16) {
                host_log("fmod ring: kept sound %llx freed by its own release (%llu so far)",
                         static_cast<unsigned long long>(p), static_cast<unsigned long long>(f));
            }
        }
    }
    const bool linked_now = linked(p, readable);
    if (g_check) note('F', p, linked_now ? 1 : 0, rd32(p + kFlags), saved);
    if (!linked_now || !g_guard) {
        if (linked_now && g_check) {
            describe("freed while still in its sound group (guard off)", p);
            log_history(p, 32);
        }
        return 0;
    }
    // The stream's free of a shared subsound that refused: it is kept, a sound
    // of its own, for whoever still holds it (an event, or the release under
    // way on another thread) - freed, it would be their use-after-free next.
    const bool keep = saved[6] == va(kSharedFreeReturn) && refused_shared(p);
    bool done = false;
    {
        Crit c;
        done = unlink(p, readable);
        if (done && keep) orphan(p);
    }
    if (!done) return 0;
    const std::uint64_t n = g_guarded.fetch_add(1) + 1;
    if (keep) {
        const std::uint64_t bytes = owned_bytes(p, readable);
        std::lock_guard<std::mutex> lk(g_kept_mu);
        if (g_kept.emplace(p, bytes).second) g_kept_live.fetch_add(1, std::memory_order_relaxed);
        g_kept_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (g_check) note(keep ? 'K' : 'X', p, n, 0, saved);
    if (n <= 64 || (n & (n - 1)) == 0) {
        std::uint64_t r[4];
        callers(saved, r, 4);
        describe(keep ? "a stream's shared subsound that refused its release, about to be freed still in its sound group; "
                        "taken out of the ring and kept as a sound of its own that nothing plays again (a programmer "
                        "sound kept so is never freed)"
                      : "freed while still in its sound group; taken out of the ring first",
                 p);
        host_log("fmod ring:   freed from %llx, callers %llx %llx %llx (%llu so far)",
                 static_cast<unsigned long long>(bn(r[0])), static_cast<unsigned long long>(r[1] ? bn(r[1]) : 0),
                 static_cast<unsigned long long>(r[2] ? bn(r[2]) : 0), static_cast<unsigned long long>(r[3] ? bn(r[3]) : 0),
                 static_cast<unsigned long long>(n));
        if (g_check) {
            std::uint32_t held = 0;
            {
                std::lock_guard<std::mutex> lk(g_prog_mu);
                const auto it = g_prog.find(p);
                if (it != g_prog.end()) held = it->second;
            }
            host_log("fmod ring:   %u event(s) hold it as a programmer sound", held);
            log_history(p, 32);
        }
    }
    return keep ? 1 : 0;  // 1: MemPool::free is not run (its caller ignores the result)
}

GUEST_ABI std::int64_t set_group_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t s = saved[5], group = saved[4];
    note('G', s, group, rd32(s + kFlags), saved);
    if (rd32(s + kFlags) & 0x20) {
        describe("linked into a group while being released", s);
        log_history(s, 16);
    }
    if (master_group(s) && (g_links.fetch_add(1) & 63) == 0) check_ring(true, "a link");
    return 0;
}

// A stream of several subsounds shares one subsound object (+0xb0, whose
// own +0xb0 points at itself). Its release asks that object to release
// itself, ignores the answer and then frees it (fmod_soundi.cpp line 0x367,
// 0x11be682): an object that refused - flags 0x20 being released, 0x100
// preloaded, 0x200 an event's programmer sound - is freed still linked. Each
// such release is logged with the shared object's flags.
// BBHOST_FMOD_FORCE_REFUSE=1 (a test of the guard, with the detector): the
// shared object is marked 0x200 there, so that it always refuses.
bool g_force_refuse = false;
std::atomic<std::uint64_t> g_shared_releases{0}, g_shared_refusing{0};

GUEST_ABI std::int64_t release_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t s = saved[5];
    note('R', s, rd32(s + kFlags) | (static_cast<std::uint64_t>(rd32(s + kOpenState)) << 32), rd(s + kParent), saved);
    const std::uint64_t shared = rd(s + kShared);
    if (shared && shared != s && readable(shared, kNode + 24) && is_sound(g_vt, shared) && rd(shared + kShared) == shared &&
        !(rd32(s + kFlags) & 0x320)) {
        if (g_force_refuse) wr32(shared + kFlags, rd32(shared + kFlags) | 0x200u);
        const std::uint32_t f = rd32(shared + kFlags);
        const std::uint64_t n = g_shared_releases.fetch_add(1) + 1;
        const bool refusing = (f & 0x320) != 0;
        const std::uint64_t r = refusing ? g_shared_refusing.fetch_add(1) + 1 : g_shared_refusing.load();
        if (refusing || n <= 16 || (n & (n - 1)) == 0) {
            // The release's callers name its owner: the game's FSB resource
            // (0x2c0eb70), the event system's stream instance (0x112794d)...
            std::uint64_t c[4];
            callers(saved, c, 4);
            char name[32];
            thread_name(name, sizeof(name));
            host_log("fmod ring: stream %llx (%u subsounds) released, its shared subsound %llx has flags %x%s (%llu such "
                     "releases, %llu with a refusing subsound); from %llx %llx %llx %llx on %s",
                     static_cast<unsigned long long>(s), rd32(s + kNumSubsounds), static_cast<unsigned long long>(shared), f,
                     refusing ? " - it will refuse and be freed still linked" : "", static_cast<unsigned long long>(n),
                     static_cast<unsigned long long>(r), static_cast<unsigned long long>(bn(c[0])),
                     static_cast<unsigned long long>(c[1] ? bn(c[1]) : 0), static_cast<unsigned long long>(c[2] ? bn(c[2]) : 0),
                     static_cast<unsigned long long>(c[3] ? bn(c[3]) : 0), name);
        }
    }
    return 0;
}

GUEST_ABI std::int64_t ctor_hook(std::uint64_t, const std::uint64_t* saved) {
    note('N', saved[5], 0, 0, saved);
    return 0;
}

GUEST_ABI std::int64_t set_sub_hook(std::uint64_t, const std::uint64_t* saved) {
    // parent, index, subsound
    note('S', saved[3], saved[5], saved[4], saved);
    return 0;
}

GUEST_ABI std::int64_t create_hook(std::uint64_t, const std::uint64_t* saved) {
    // name or data, mode
    note('C', saved[4], saved[3], saved[1], saved);
    return 0;
}

// --- the game's FSBs and the event system's programmer sounds (the detector) --
// The game keeps each FSB it loads in a holder of its own (0x60 bytes: +0x10
// the name, +0x20 let go, +0x21 preloaded with EventSystem::preloadFSB, +0x22
// a voice set, +0x28 the Sound): a sample bank in memory (0x2c0f090), a
// preloaded stream (0x2c0f210), or a voice set (0x2c0f370: a stream whose
// subsounds are the voice lines, its line table beside it). Its voice-line
// start (0x2bf27f0) takes a line with getSubSound - for a stream of several
// subsounds, the one shared subsound object - and hands it to the line's
// event at SOUNDDEF_CREATE; EventSound::prepare marks it 0x200 (0x11f8456,
// then the getOpenState call at 0x11f8465) until EventSound::releaseSound
// clears that and calls SOUNDDEF_RELEASE back (0x11f7cfc). The holder's
// release (0x2c0eae0) releases the Sound.
constexpr std::uint64_t kBankLoad = 0x2c0f090, kStreamLoad = 0x2c0f210, kVoiceLoad = 0x2c0f370, kHolderRelease = 0x2c0eae0;
constexpr std::uint8_t kAlignedLoadPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                                 0x41, 0x54, 0x53, 0x48, 0x81, 0xe4, 0xe0, 0xff, 0xff, 0xff};
constexpr std::uint8_t kVoiceLoadPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                               0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xa8, 0x00, 0x00, 0x00};
constexpr std::uint8_t kHolderReleasePrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                   0x41, 0x54, 0x53, 0x49, 0x89, 0xfe};
constexpr std::uint64_t kProgSet = 0x11f8465, kGetOpenState = 0x11bb760;   // prepare: the call after the 0x200
constexpr std::uint64_t kProgRelease = 0x11f7cfc, kEventCallback = 0x11dc8b0;  // releaseSound: SOUNDDEF_RELEASE (7)

std::atomic<std::uint64_t> g_prog_sets{0}, g_prog_releases{0}, g_holder_loads{0}, g_holder_releases{0};

std::string guest_cstr(std::uint64_t a, std::size_t max = 96) {
    std::string s;
    for (std::size_t i = 0; a && i < max && readable(a + i, 1); ++i) {
        const char ch = static_cast<char>(*reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(a + i)));
        if (!ch) break;
        s += ch;
    }
    return s;
}

// A sound and, for a subsound, what it belongs to.
std::string sound_text(std::uint64_t s) {
    char b[200];
    if (!s || !readable(s, kNode + 24) || !is_sound(g_vt, s)) {
        std::snprintf(b, sizeof(b), "%llx (not a sound)", static_cast<unsigned long long>(s));
        return b;
    }
    const std::uint64_t p = rd(s + kParent);
    const std::uint32_t pn = p && readable(p, kNode + 24) ? rd32(p + kNumSubsounds) : 0;
    std::snprintf(b, sizeof(b), "%llx (flags %x, %u subsounds, %s; parent %llx of %u subsounds)", static_cast<unsigned long long>(s),
                  rd32(s + kFlags), rd32(s + kNumSubsounds), rd(s + kShared) == s ? "a stream's shared subsound" : "its own",
                  static_cast<unsigned long long>(p), pn);
    return b;
}

GUEST_ABI std::int64_t holder_load_hook(std::uint64_t id, const std::uint64_t* saved) {
    // (holder, ...): the names are often on the caller's stack, outside the
    // memory the detector reads; the holder's release names the FSB.
    const std::uint64_t n = g_holder_loads.fetch_add(1) + 1;
    const char* kind = id == 1 ? "a sample bank" : id == 2 ? "a preloaded stream" : "a voice set";
    if (id != 1 || n <= 256) {
        host_log("fmod ring: the game loads %s into holder %llx (%llu loads)", kind, static_cast<unsigned long long>(saved[5]),
                 static_cast<unsigned long long>(n));
    }
    return 0;
}

GUEST_ABI std::int64_t holder_release_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t h = saved[5];
    if (!readable(h, 0x60)) return 0;
    const std::uint64_t n = g_holder_releases.fetch_add(1) + 1;
    const std::uint64_t s = rd(h + 0x28);
    const std::uint64_t sh = s && readable(s, kNode + 24) ? rd(s + kShared) : 0;
    std::string held;
    if (sh) {
        std::lock_guard<std::mutex> lk(g_prog_mu);
        const auto it = g_prog.find(sh);
        if (it != g_prog.end()) held = ", its shared subsound held by " + std::to_string(it->second) + " event(s) as a programmer sound";
    }
    if (n <= 256 || !held.empty()) {
        host_log("fmod ring: the game releases holder %llx (%s; let go %u, preloaded %u, voice set %u): sound %s%s%s (%llu releases)",
                 static_cast<unsigned long long>(h), guest_cstr(rd(h + 0x10)).c_str(), rd32(h + 0x20) & 0xff, (rd32(h + 0x20) >> 8) & 0xff,
                 (rd32(h + 0x20) >> 16) & 0xff, sound_text(s).c_str(), sh ? (", shared subsound " + sound_text(sh)).c_str() : "",
                 held.c_str(), static_cast<unsigned long long>(n));
    }
    return 0;
}

GUEST_ABI std::int64_t prog_set_hook(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t s = saved[5];  // rdi: the sound just marked 0x200
    std::uint32_t holders = 0;
    {
        std::lock_guard<std::mutex> lk(g_prog_mu);
        holders = ++g_prog[s];
    }
    const std::uint64_t n = g_prog_sets.fetch_add(1) + 1;
    if (n <= 64 || (n & (n - 1)) == 0 || holders > 1) {
        host_log("fmod ring: an event takes programmer sound %s, now held by %u event(s) (%llu so far)", sound_text(s).c_str(),
                 holders, static_cast<unsigned long long>(n));
    }
    return 0;
}

GUEST_ABI std::int64_t prog_release_hook(std::uint64_t, const std::uint64_t* saved) {
    if (saved[4] != 7) return 0;  // esi: SOUNDDEF_RELEASE
    const std::uint64_t s = saved[2];  // rcx: the sound
    std::uint32_t left = 0;
    bool counted = false;
    {
        std::lock_guard<std::mutex> lk(g_prog_mu);
        const auto it = g_prog.find(s);
        if (it != g_prog.end()) {
            counted = true;
            left = --it->second;
            if (!left) g_prog.erase(it);
        }
    }
    const std::uint64_t n = g_prog_releases.fetch_add(1) + 1;
    if (n <= 64 || (n & (n - 1)) == 0) {
        host_log("fmod ring: an event lets programmer sound %llx go%s, %u event(s) still hold it (%llu so far)",
                 static_cast<unsigned long long>(s), counted ? "" : " (one no event took as a programmer sound)", left,
                 static_cast<unsigned long long>(n));
    }
    return 0;
}

bool hook(ElfImage* image, std::uint64_t at, const std::uint8_t* pro, std::size_t n, void* fn, const char* what) {
    const bool ok = engine_prologue_hook(image, va(at), pro, n, fn);
    if (!ok) host_log("fmod ring: %s (0x%llx) NOT hooked (unexpected bytes)", what, static_cast<unsigned long long>(at));
    return ok;
}

}  // namespace

void fmod_ring_install(ElfImage* image) {
    const char* g = std::getenv("BBHOST_FMOD_GUARD");
    const char* c = std::getenv("BBHOST_FMOD_CHECK");
    g_guard = !(g && *g == '0');
    g_check = c && *c == '1';
    const char* fr = std::getenv("BBHOST_FMOD_FORCE_REFUSE");
    g_force_refuse = g_check && fr && *fr == '1';
    if (!g_guard && !g_check) {
        host_log("fmod ring: the sound-group guard is off (BBHOST_FMOD_GUARD=0)");
        return;
    }
    g_slide = image->mem.slide;
    for (int i = 0; i < 4; ++i) g_vt.vt[i] = va(kSoundVtables[i]);
    if (g_check) g_events = static_cast<Event*>(std::calloc(kEvents, sizeof(Event)));
    bool ok = hook(image, kFree, kFreePrologue, sizeof(kFreePrologue), reinterpret_cast<void*>(&free_hook), "MemPool::free");
    if (g_check) {
        ok = hook(image, kSetGroup, kSetGroupPrologue, sizeof(kSetGroupPrologue), reinterpret_cast<void*>(&set_group_hook),
                  "SoundI::setSoundGroup") && ok;
        ok = hook(image, kRelease, kReleasePrologue, sizeof(kReleasePrologue), reinterpret_cast<void*>(&release_hook),
                  "SoundI::release") && ok;
        ok = hook(image, kCtor, kCtorPrologue, sizeof(kCtorPrologue), reinterpret_cast<void*>(&ctor_hook), "SoundI::SoundI") && ok;
        ok = hook(image, kSetSub, kSetSubPrologue, sizeof(kSetSubPrologue), reinterpret_cast<void*>(&set_sub_hook),
                  "SoundI::setSubSound") && ok;
        ok = hook(image, kCreate, kCreatePrologue, sizeof(kCreatePrologue), reinterpret_cast<void*>(&create_hook),
                  "SystemI::createSound") && ok;
        const auto load = [&](std::uint64_t at, const std::uint8_t* pro, std::size_t n, std::uint64_t id, const char* what) {
            const bool placed = engine_prologue_hook(image, va(at), pro, n, reinterpret_cast<void*>(&holder_load_hook), id);
            if (!placed) host_log("fmod ring: %s (0x%llx) NOT hooked (unexpected bytes)", what, static_cast<unsigned long long>(at));
            return placed;
        };
        ok = load(kBankLoad, kAlignedLoadPrologue, sizeof(kAlignedLoadPrologue), 1, "the game's bank load") && ok;
        ok = load(kStreamLoad, kAlignedLoadPrologue, sizeof(kAlignedLoadPrologue), 2, "the game's stream load") && ok;
        ok = load(kVoiceLoad, kVoiceLoadPrologue, sizeof(kVoiceLoadPrologue), 3, "the game's voice set load") && ok;
        ok = hook(image, kHolderRelease, kHolderReleasePrologue, sizeof(kHolderReleasePrologue),
                  reinterpret_cast<void*>(&holder_release_hook), "the game's FSB holder release") && ok;
        if (!engine_call_site_hook(image, va(kProgSet), va(kGetOpenState), reinterpret_cast<void*>(&prog_set_hook))) {
            host_log("fmod ring: EventSound::prepare's programmer sound (0x%llx) NOT hooked", static_cast<unsigned long long>(kProgSet));
            ok = false;
        }
        if (!engine_call_site_hook(image, va(kProgRelease), va(kEventCallback), reinterpret_cast<void*>(&prog_release_hook))) {
            host_log("fmod ring: EventSound::releaseSound's SOUNDDEF_RELEASE (0x%llx) NOT hooked",
                     static_cast<unsigned long long>(kProgRelease));
            ok = false;
        }
    }
    host_log("fmod ring: sound-group guard %s%s%s%s", g_guard ? "on" : "off",
             g_check ? ", detector on (BBHOST_FMOD_CHECK=1: history, master ring walks)" : "",
             g_force_refuse ? ", shared subsounds forced to refuse (BBHOST_FMOD_FORCE_REFUSE=1)" : "",
             ok ? "" : " - a hook was not placed");
}

void fmod_ring_report() {
    if (!g_guard) return;
    const std::uint64_t n = g_guarded.load(), k = g_kept_count.load(), f = g_kept_freed.load();
    // try_lock: the report also runs from a signal (SIGTERM, a crash) that may
    // land on a thread inside the free hook.
    std::uint64_t live = g_kept_live.load(), bytes = 0;
    {
        std::unique_lock<std::mutex> lk(g_kept_mu, std::try_to_lock);
        if (lk.owns_lock()) {
            live = g_kept.size();
            for (const auto& kv : g_kept) bytes += kv.second;
        }
    }
    host_log("fmod ring: the guard took %llu freed sound(s) out of the ring; kept %llu refusing shared subsound(s), %llu "
             "freed since by a release under way, %llu never freed (%llu bytes)",
             static_cast<unsigned long long>(n), static_cast<unsigned long long>(k), static_cast<unsigned long long>(f),
             static_cast<unsigned long long>(live), static_cast<unsigned long long>(bytes));
    if (g_check) {
        std::size_t held = 0;
        {
            std::unique_lock<std::mutex> lk(g_prog_mu, std::try_to_lock);
            if (lk.owns_lock()) held = g_prog.size();
        }
        host_log("fmod ring: the game loaded %llu FSB(s) into holders and released %llu; events took %llu programmer sound(s) "
                 "and let %llu go, %zu still held",
                 static_cast<unsigned long long>(g_holder_loads.load()), static_cast<unsigned long long>(g_holder_releases.load()),
                 static_cast<unsigned long long>(g_prog_sets.load()), static_cast<unsigned long long>(g_prog_releases.load()), held);
    }
}

void fmod_ring_crash_report() {
    if (g_guard && g_guarded.load()) fmod_ring_report();
    if (!g_check) return;
    check_ring(false, "the crash");
    log_recent(24);
}
