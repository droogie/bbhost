#pragma once

// FMOD Ex 4.44.50 in the eboot keeps every sound it has made - a bank's
// samples, a stream and its subsound - in the master sound group's ring:
// SoundI +0x1d0 is a node (next, prev, the sound), the group's head is
// SoundGroupI +0x30, the master group SystemI +0x9668. The game makes no
// other group (System::createSoundGroup's only caller is SystemI::init).
// setSoundGroup (0x11c0880) links a sound in and SoundI::release (0x11be120)
// takes it out before freeing it, both under FMOD's global critical section
// (the pointer at 0x586d260: a recursive scePthreadMutex).
//
// A sound whose memory is freed while it is still linked leaves a dead member
// in the ring, and the next sound linked or unlinked beside it writes through
// it into whatever the heap has put there since. That was a crash on
// arriving at Rom's lake from the Fishing Hamlet: the crash's core holds a
// member left between the sprj_m27 bank's samples and the lake's music
// stream; its memory had become a render item's listener node, and the
// lake's newest sample, linked beside it, wrote itself over that node's next
// pointer.
//
// FMOD's own release unlinks a sound before freeing it, with one exception:
// a stream of several subsounds (a voice bank's: sprj_sm27_eng.fsb has 166)
// shares one subsound object (+0xb0, whose +0xb0 names itself), and the
// stream's release asks it to release itself, ignores the answer and frees
// it (fmod_soundi.cpp line 0x367, 0x11be682). The object refuses - and so
// stays linked - while its flags say 0x200 (an event's programmer sound in
// use), 0x100 (preloaded) or 0x20 (a release under way): freed linked.
//
// The guard (on; BBHOST_FMOD_GUARD=0 turns it off): FMOD's MemPool free
// (0x11ab3a0) first takes a sound out of its group, under the same critical
// section, when the block it is about to free is a sound that is still
// linked, and logs who freed it. At that one free of a refusing shared
// subsound it also keeps the object, as a sound of its own that nothing can
// play again (orphan() below), so that whoever still holds it - the event,
// the other release - does not use freed memory. An event's programmer
// sound kept so is never freed: a leak of a few hundred bytes each time,
// which the exit report counts.
//
// BBHOST_FMOD_CHECK=1 adds the detector: a history of the sounds' births,
// links, releases, subsound moves and frees (thread, callers); a walk of the
// master ring at every 64th link, reporting a member that is not a live
// sound of the group with that address's history; and the last events in a
// crash report.

#include <cstddef>
#include <cstdint>
#include <cstring>

struct ElfImage;

void fmod_ring_install(ElfImage* image);
void fmod_ring_crash_report();
// The exit report's line: sounds taken out of the ring, those kept, leaked.
void fmod_ring_report();

namespace fmod_ring {

// SoundI (FMOD Ex 4.44.50 in eboot 1.09).
constexpr std::uint64_t kNode = 0x1d0;       // the group ring's node: next, prev, the sound
constexpr std::uint64_t kGroup = 0x1c8;      // SoundGroupI*
constexpr std::uint64_t kFlags = 0x50;       // 0x20 being released, 0x100 preloaded (event system), 0x200 programmer sound in use
constexpr std::uint64_t kBusy = 0x54;        // byte: the nonblocking thread is finishing an open
constexpr std::uint64_t kCodec = 0x58;
constexpr std::uint64_t kSubsounds = 0xa8;   // SoundI*[numsubsounds]
constexpr std::uint64_t kShared = 0xb0;      // the subsound object a stream of several subsounds shares
constexpr std::uint64_t kNumSubsounds = 0xb8;
constexpr std::uint64_t kParent = 0xc0;
constexpr std::uint64_t kSubIndex = 0xc8;
constexpr std::uint64_t kSystem = 0x188;     // SystemI*
constexpr std::uint64_t kOpenState = 0x1c0;  // FMOD_OPENSTATE
constexpr std::uint64_t kStreamSample = 0x238;  // a stream's sample buffer, which its subsound object shares
constexpr std::uint32_t kOpenStateError = 2;    // FMOD_OPENSTATE_ERROR
// The blocks SoundI::release frees besides the object: the name, a subsound
// table, two sync-point buffers and the sync-point list.
constexpr std::uint64_t kOwnedBlocks[] = {0x30, 0xd0, 0x198, 0x1a0, 0x1b0};
// SoundGroupI and SystemI.
constexpr std::uint64_t kHead = 0x30;
constexpr std::uint64_t kMaster = 0x9668;

inline std::uint64_t rd(std::uint64_t a) {
    std::uint64_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a)), sizeof(v));
    return v;
}
inline std::uint32_t rd32(std::uint64_t a) {
    std::uint32_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a)), sizeof(v));
    return v;
}
inline void wr(std::uint64_t a, std::uint64_t v) {
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(a)), &v, sizeof(v));
}
inline void wr32(std::uint64_t a, std::uint32_t v) {
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(a)), &v, sizeof(v));
}

// The SoundI classes' vtables, as the objects hold them (runtime addresses):
// SampleSoftware, SoundI, the bank container (Sample) and Stream.
struct Vtables {
    std::uint64_t vt[4];
};

inline bool is_sound(const Vtables& v, std::uint64_t obj) {
    const std::uint64_t t = rd(obj);
    for (const std::uint64_t x : v.vt) {
        if (t == x) return true;
    }
    return false;
}

// In a ring: the node names the sound and both neighbours point back at it.
// `readable(addr, len)` says whether guest memory there can be read.
template <typename Readable>
bool linked(std::uint64_t obj, Readable readable) {
    const std::uint64_t node = obj + kNode;
    const std::uint64_t next = rd(node), prev = rd(node + 8);
    if (rd(node + 16) != obj || next == node || prev == node) return false;
    if (!readable(next, 16) || !readable(prev, 8)) return false;
    return rd(next + 8) == node && rd(prev) == node;
}

// Takes a linked sound out of its ring and leaves the node as
// SoundI::release does (linked to itself, no data). False when it was not
// linked or its neighbours do not point back at it (then nothing is written).
template <typename Readable>
bool unlink(std::uint64_t obj, Readable readable) {
    if (!linked(obj, readable)) return false;
    const std::uint64_t node = obj + kNode, next = rd(node), prev = rd(node + 8);
    wr(prev, next);
    wr(next + 8, prev);
    wr(node, node);
    wr(node + 8, node);
    wr(node + 16, 0);
    return true;
}

// A stream's shared subsound (its +0xb0 names itself) that refused its own
// release - flags 0x20 being released, 0x100 preloaded by the event system,
// 0x200 an event's programmer sound - when the stream released it.
constexpr std::uint32_t kRefusing = 0x20 | 0x100 | 0x200;
inline bool refused_shared(std::uint64_t obj) {
    return rd(obj + kShared) == obj && (rd32(obj + kFlags) & kRefusing) != 0;
}

// What the guard makes of such a subsound instead of letting its stream free
// it: a sound of its own (no parent, no shared block). Its codec, async data
// and stream channel were the stream's, and the stream's release has cleared
// them; the stream's sample buffer it also points at was released by the
// stream too, so that is cleared here.
//
// Its open state becomes FMOD_OPENSTATE_ERROR, so that nothing plays it
// again: an event that starts its programmer sound once more gets this
// object back from the game (its sound object still holds the line), and
// with the state left READY the event's start (0x11dd4f4) and the system's
// channel allocation for a stream (0x1109fbc) would read through its cleared
// codec. getOpenState reports the error with every pointer it follows
// checked, and the event then waits on it until it is stopped; the system's
// playSound (0x1115d93), getLength and getSubSound refuse it as not ready.
//
// Its fate: an event's programmer sound (0x200) is never freed. The event
// system lets it go in EventSound::releaseSound (0x11f7c50: the channel
// stopped by handle, 0x200 cleared at 0x11f7c8b, SOUNDDEF_RELEASE to the
// game), and the game's callback (0x2c2c660) does nothing for
// SOUNDDEF_RELEASE - the stream owned it. So it leaks: the object and the
// blocks it owns, a few hundred bytes, each time the path is taken (counted
// in the exit report). One whose release was under way on another thread
// (0x20) is freed by that release, which now finds no shared block.
inline void orphan(std::uint64_t obj) {
    wr(obj + kParent, 0);
    wr(obj + kShared, 0);
    wr(obj + kStreamSample, 0);
    wr32(obj + kOpenState, kOpenStateError);
}

// The bytes FMOD's allocator recorded for a sound and the blocks it owns (its
// {size, flags} header sits 8 bytes before each block).
template <typename Readable>
std::uint64_t owned_bytes(std::uint64_t obj, Readable readable) {
    std::uint64_t n = readable(obj - 8, 4) ? rd32(obj - 8) : 0;
    for (const std::uint64_t off : kOwnedBlocks) {
        const std::uint64_t b = rd(obj + off);
        if (b && readable(b - 8, 4)) n += rd32(b - 8);
    }
    return n;
}

// The first node of the ring at `head` (a group's +0x30) that is not a live
// sound of `group` - its data is not the object it sits in, the object is no
// sound, it names another group, or its prev does not point at the node
// before it - or 0. Unreadable links and a ring longer than `limit` count as
// stale at the node that leads there. *count: the members walked.
template <typename Readable>
std::uint64_t first_stale(const Vtables& v, std::uint64_t head, std::uint64_t group, std::size_t limit,
                          std::size_t* count, Readable readable) {
    std::size_t n = 0;
    std::uint64_t prev = head;
    std::uint64_t node = rd(head);
    while (node != head) {
        if (++n > limit || !readable(node, 24)) break;
        const std::uint64_t obj = node - kNode;
        if (rd(node + 8) != prev || rd(node + 16) != obj || !readable(obj, kNode) || !is_sound(v, obj) ||
            rd(obj + kGroup) != group) {
            if (count) *count = n;
            return node;
        }
        prev = node;
        node = rd(node);
    }
    if (count) *count = n;
    return node == head ? 0 : prev;
}

}  // namespace fmod_ring
