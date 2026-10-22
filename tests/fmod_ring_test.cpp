// The FMOD sound-group ring guard (engine/fmod_ring.h) without the game:
// sounds and a group laid out as FMOD Ex 4.44.50 lays them out, linked the
// way setSoundGroup (0x11c0880) links them and taken out the way
// SoundI::release (0x11be120) does, then the sequence of a crash on arriving
// at Rom's lake from the Fishing Hamlet - a sound freed while still linked,
// its memory reused, a new sound linked beside it -
// with the guard and without it, plus the cases the guard must leave alone
// (a sound already out, a block that is no sound, a node whose neighbours do
// not point back) and the ring walk's verdicts.
#include "engine/fmod_ring.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace fmod_ring;

namespace {

int g_failed = 0;
void check(bool ok, const char* what, int line) {
    if (ok) return;
    if (++g_failed <= 40) std::printf("FAIL (line %d): %s\n", line, what);
}
#define CHECK(x) check((x), #x, __LINE__)

constexpr std::uint64_t kVtSample = 0x56d7640, kVtSound = 0x56d7840, kVtBank = 0x56d7a30, kVtStream = 0x56d7c30;
const Vtables kVt{{kVtSample, kVtSound, kVtBank, kVtStream}};

// One arena for every object, so that "readable" can be told exactly.
alignas(16) std::uint8_t g_mem[1 << 16];
std::size_t g_used = 0;
std::uint64_t base() { return reinterpret_cast<std::uint64_t>(g_mem); }
bool readable(std::uint64_t a, std::size_t n) { return a >= base() && a + n <= base() + sizeof(g_mem); }
auto kReadable = [](std::uint64_t a, std::size_t n) { return readable(a, n); };

std::uint64_t alloc(std::size_t n) {
    const std::uint64_t p = base() + g_used;
    g_used += (n + 15) & ~std::size_t{15};
    std::memset(reinterpret_cast<void*>(p), 0, n);
    return p;
}

// SoundGroupI: its ring's head at +0x30, linked to itself.
std::uint64_t make_group() {
    const std::uint64_t g = alloc(0x88);
    wr(g + kHead, g + kHead);
    wr(g + kHead + 8, g + kHead);
    return g;
}

// A sound as SoundI::SoundI leaves it: its node linked to itself.
std::uint64_t make_sound(std::uint64_t vt = kVtSample) {
    const std::uint64_t s = alloc(0x260);
    wr(s, vt);
    wr(s + kNode, s + kNode);
    wr(s + kNode + 8, s + kNode);
    return s;
}

// setSoundGroup's body: out of the old ring, before the group's head.
void link(std::uint64_t s, std::uint64_t g) {
    const std::uint64_t node = s + kNode;
    wr(rd(node + 8), rd(node));
    wr(rd(node) + 8, rd(node + 8));
    wr(s + kGroup, g);
    const std::uint64_t head = g + kHead;
    wr(node + 8, rd(head + 8));
    wr(node, head);
    wr(head + 8, node);
    wr(rd(node + 8), node);
    wr(node + 16, s);
}

// SoundI::release's unlink.
void release_unlink(std::uint64_t s) {
    const std::uint64_t node = s + kNode;
    wr(rd(node + 8), rd(node));
    wr(rd(node) + 8, rd(node + 8));
    wr(node, node);
    wr(node + 8, node);
    wr(node + 16, 0);
}

// The ring's members in order, and whether every link is consistent.
std::vector<std::uint64_t> members(std::uint64_t g, bool* consistent) {
    std::vector<std::uint64_t> out;
    const std::uint64_t head = g + kHead;
    std::uint64_t prev = head;
    *consistent = true;
    for (std::uint64_t n = rd(head); n != head && out.size() < 64; n = rd(n)) {
        if (rd(n + 8) != prev) *consistent = false;
        out.push_back(n - kNode);
        prev = n;
    }
    if (rd(head + 8) != prev) *consistent = false;
    return out;
}

void test_linked_and_unlink() {
    const std::uint64_t g = make_group();
    const std::uint64_t a = make_sound(), b = make_sound(kVtStream), c = make_sound(kVtBank);
    CHECK(!linked(a, kReadable));  // just made: linked to itself
    link(a, g);
    link(b, g);
    link(c, g);
    CHECK(linked(a, kReadable) && linked(b, kReadable) && linked(c, kReadable));
    CHECK(is_sound(kVt, a) && is_sound(kVt, b) && is_sound(kVt, c));
    // The middle one out, as the guard does it.
    CHECK(unlink(b, kReadable));
    bool ok = false;
    const auto m = members(g, &ok);
    CHECK(ok && m.size() == 2 && m[0] == a && m[1] == c);
    CHECK(rd(b + kNode) == b + kNode && rd(b + kNode + 8) == b + kNode && rd(b + kNode + 16) == 0);
    // Out already: nothing to do, nothing written.
    CHECK(!unlink(b, kReadable));
    // A release after the guard (the order FMOD's own path has) finds it out.
    release_unlink(b);
    const auto m2 = members(g, &ok);
    CHECK(ok && m2.size() == 2);
    // The last one and the first one.
    CHECK(unlink(c, kReadable));
    CHECK(unlink(a, kReadable));
    const auto m3 = members(g, &ok);
    CHECK(ok && m3.empty());
}

void test_not_a_sound() {
    // A block whose first qword is not one of the four vtables is left alone
    // by the guard (it checks is_sound first), even with sound-like bytes.
    const std::uint64_t g = make_group();
    const std::uint64_t s = make_sound();
    link(s, g);
    wr(s, 0x56d7648);  // eight bytes past a vtable: not a sound
    CHECK(!is_sound(kVt, s));
    wr(s, 0);
    CHECK(!is_sound(kVt, s));
    wr(s, kVtSound);
    CHECK(is_sound(kVt, s));
}

void test_inconsistent_neighbours() {
    // A node whose neighbours do not point back is not taken out: unlinking
    // through them would write into memory the ring no longer owns.
    const std::uint64_t g = make_group();
    const std::uint64_t a = make_sound(), b = make_sound(), c = make_sound();
    link(a, g);
    link(b, g);
    link(c, g);
    const std::uint64_t saved_next = rd(a + kNode);
    wr(a + kNode, c + kNode);  // a's next skips b, c's prev still names b
    CHECK(!linked(a, kReadable));
    std::uint8_t before[0x260 * 3];
    std::memcpy(before, reinterpret_cast<void*>(a), sizeof(before));
    CHECK(!unlink(a, kReadable));
    CHECK(std::memcmp(before, reinterpret_cast<void*>(a), sizeof(before)) == 0);
    wr(a + kNode, saved_next);
    // The node's data must name the sound.
    wr(b + kNode + 16, a);
    CHECK(!linked(b, kReadable));
    wr(b + kNode + 16, b);
    CHECK(linked(b, kReadable));
    // A neighbour outside readable memory.
    const std::uint64_t keep = rd(c + kNode);
    wr(c + kNode, 0x10);
    CHECK(!linked(c, kReadable));
    wr(c + kNode, keep);
    CHECK(linked(c, kReadable));
}

void test_ring_walk() {
    const std::uint64_t g = make_group(), other = make_group();
    std::uint64_t s[6];
    for (auto& x : s) {
        x = make_sound();
        link(x, g);
    }
    std::size_t n = 0;
    CHECK(first_stale(kVt, g + kHead, g, 100, &n, kReadable) == 0 && n == 6);
    // A member whose object is no sound any more (freed, memory reused).
    wr(s[2], 0x1234);
    CHECK(first_stale(kVt, g + kHead, g, 100, &n, kReadable) == s[2] + kNode && n == 3);
    wr(s[2], kVtSample);
    // A member that names another group.
    wr(s[4] + kGroup, other);
    CHECK(first_stale(kVt, g + kHead, g, 100, &n, kReadable) == s[4] + kNode);
    wr(s[4] + kGroup, g);
    // A member whose data is not its sound.
    wr(s[1] + kNode + 16, 0x122882650);
    CHECK(first_stale(kVt, g + kHead, g, 100, &n, kReadable) == s[1] + kNode);
    wr(s[1] + kNode + 16, s[1]);
    // A prev that does not name the node before.
    wr(s[5] + kNode + 8, s[3] + kNode);
    CHECK(first_stale(kVt, g + kHead, g, 100, &n, kReadable) == s[5] + kNode);
    wr(s[5] + kNode + 8, s[4] + kNode);
    // A next that leaves readable memory: the node leading there is named.
    const std::uint64_t keep = rd(s[3] + kNode);
    wr(s[3] + kNode, 0x40);
    CHECK(first_stale(kVt, g + kHead, g, 100, &n, kReadable) == s[3] + kNode);
    wr(s[3] + kNode, keep);
    // A ring that does not come back within the limit.
    CHECK(first_stale(kVt, g + kHead, g, 4, &n, kReadable) != 0);
    CHECK(first_stale(kVt, g + kHead, g, 100, &n, kReadable) == 0 && n == 6);
}

// The crash's core: the bank's last sample R, then a stream Y of several subsounds
// and its shared subsound S (linked in that order), later the lake's music
// stream X. Y's release frees S still linked (S refused); the heap hands S's
// block out again (a 0x18-byte list node whose header is the qword at S's
// next, its next at S's prev); Y unlinks itself at the end of its release;
// X is linked. Returns whether S's reused memory was written - both writes
// the crash's core showed: S.prev = R's node, S.next = X's node.
bool crash_sequence(bool guard) {
    const std::uint64_t g = make_group();
    const std::uint64_t r = make_sound(), y = make_sound(kVtStream), sub = make_sound(kVtStream),
                        x = make_sound(kVtStream);
    link(r, g);
    link(y, g);
    link(sub, g);
    wr(y + kShared, sub);
    wr(sub + kShared, sub);
    wr(sub + kParent, y);
    wr32(sub + kFlags, 0x200);  // an event's programmer sound: its release refuses
    // Y's release: the subsound's own release refuses (nothing happens), then
    // MemPool::free of the shared block - the guard runs at its entry.
    bool kept = false;
    if (guard && is_sound(kVt, sub) && linked(sub, kReadable)) {
        kept = refused_shared(sub);
        unlink(sub, kReadable);
        if (kept) orphan(sub);
    }
    std::uint8_t junk[0x260];
    if (!kept) {
        // Freed: the heap reuses the block.
        for (std::size_t i = 0; i < sizeof(junk); ++i) junk[i] = static_cast<std::uint8_t>(0xa5 ^ i);
        std::memcpy(reinterpret_cast<void*>(sub), junk, sizeof(junk));
    } else {
        std::memcpy(junk, reinterpret_cast<void*>(sub), sizeof(junk));
    }
    // The end of Y's release, then the music stream.
    release_unlink(y);
    link(x, g);
    bool ok = false;
    const auto m = members(g, &ok);
    const bool written = std::memcmp(junk, reinterpret_cast<void*>(sub), sizeof(junk)) != 0;
    if (guard) {
        CHECK(kept);
        CHECK(ok && m.size() == 2 && m[0] == r && m[1] == x);
        CHECK(first_stale(kVt, g + kHead, g, 100, nullptr, kReadable) == 0);
    } else {
        // Without the guard the dead member stays, and the walk finds it.
        CHECK(first_stale(kVt, g + kHead, g, 100, nullptr, kReadable) == sub + kNode);
        CHECK(rd(sub + kNode + 8) == r + kNode && rd(sub + kNode) == x + kNode);  // the two writes of the core
    }
    return written;
}

// A stream of several subsounds and its shared subsound object, linked as
// createSound links them (the stream first). The guard keeps the subsound
// only when it names itself as the shared block and its flags refuse.
void test_refused_shared() {
    const std::uint64_t g = make_group();
    const std::uint64_t stream = make_sound(kVtStream), sub = make_sound(kVtStream);
    link(stream, g);
    link(sub, g);
    wr(stream + kShared, sub);
    wr(sub + kShared, sub);
    wr(sub + kParent, stream);
    wr(sub + kStreamSample, 0x1234560);  // the stream's sample, shared
    wr32(stream + kNumSubsounds, 166);
    CHECK(!refused_shared(sub));  // flags 0: its release goes through, it unlinks itself
    for (const std::uint32_t f : {0x20u, 0x100u, 0x200u, 0x220u}) {
        wr32(sub + kFlags, f);
        CHECK(refused_shared(sub));
    }
    wr32(sub + kFlags, 0x10);  // a flag that does not refuse
    CHECK(!refused_shared(sub));
    wr32(sub + kFlags, 0x200);
    CHECK(!refused_shared(stream));  // the stream names its subsound, not itself
    wr(sub + kShared, 0);
    CHECK(!refused_shared(sub));  // a stream's only subsound has no shared block: its own release frees it
    wr(sub + kShared, sub);
    // The bytes it holds: FMOD's {size, flags} header 8 bytes before the object
    // (here in the stream's block, which ends there) and before its name block.
    wr32(sub - 8, 0x260);
    const std::uint64_t name = alloc(0x110) + 8;
    wr32(name - 8, 0x108);
    wr(sub + 0x30, name);
    CHECK(owned_bytes(sub, kReadable) == 0x260 + 0x108);
    wr(sub + 0xd0, 0x10);  // an owned pointer outside readable memory counts nothing
    CHECK(owned_bytes(sub, kReadable) == 0x260 + 0x108);
    wr(sub + 0xd0, 0);
    // What the guard does at the stream's free of it: out of the ring, kept as a sound of its own
    // that nothing plays again.
    CHECK(unlink(sub, kReadable));
    orphan(sub);
    CHECK(rd(sub + kParent) == 0 && rd(sub + kShared) == 0 && rd(sub + kStreamSample) == 0 && !linked(sub, kReadable));
    CHECK(rd32(sub + kOpenState) == kOpenStateError);  // playSound, getLength, getSubSound refuse it; an event waits
    CHECK(rd32(sub + kFlags) == 0x200 && is_sound(kVt, sub));  // still a sound: the event's release finds one
    bool ok = false;
    const auto m = members(g, &ok);
    CHECK(ok && m.size() == 1 && m[0] == stream);
    // The stream's own unlink at the end of its release leaves the ring empty
    // and writes nothing into the kept subsound.
    std::uint8_t before[0x260];
    std::memcpy(before, reinterpret_cast<void*>(sub), sizeof(before));
    release_unlink(stream);
    CHECK(std::memcmp(before, reinterpret_cast<void*>(sub), sizeof(before)) == 0);
    const auto m2 = members(g, &ok);
    CHECK(ok && m2.empty());
}

}  // namespace

int main() {
    test_linked_and_unlink();
    test_not_a_sound();
    test_inconsistent_neighbours();
    test_ring_walk();
    test_refused_shared();
    // With the guard the reused memory is never written; without it (the
    // game's own behaviour) the link of the next sound writes into it - the
    // test would not tell the two apart otherwise.
    CHECK(!crash_sequence(true));
    CHECK(crash_sequence(false));
    if (g_failed) {
        std::printf("fmod_ring_test: %d failure(s)\n", g_failed);
        return 1;
    }
    std::printf("fmod_ring_test: ok\n");
    return 0;
}
