// GX on Vulkan, stage 1: GX buffers by handle - the table, and the side GX's
// threads call (host/gxvk.h, host/gxvk_table.h). The command processor's
// side, which places, fills and binds the copies, is gxvk_vk.cpp.
#include "host/gxvk_table.h"
#include "host/gxvk_watch.h"

#include "log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace gxvk {

std::atomic<std::uint64_t> g_bound[kUses] = {}, g_fallback[kUses][kFalls] = {}, g_bound_after_end[kUses] = {};
std::atomic<std::uint64_t> g_made[2] = {}, g_not_kept[9] = {}, g_retired[kRetires] = {};

namespace {

using ull = unsigned long long;

// Why a buffer is not kept, by the not-kept counts' index.
const char* const kNotKept[9] = {"dynamic or staging", "default without initial data", "CPU access", "a constant buffer",
                                 "a bind the GPU writes (unordered access, stream output)", "no vertex or index bind",
                                 "CPU-only memory (misc 0x4000)", "the table full", "default, past the copy watch's slots"};
const char* const kRetireName[kRetires] = {"updated", "mapped", "a GPU copy's destination", "its holder released", "its memory unmapped",
                                           "replaced", "a compare disagreed", "no memory for its copy", "its bytes out of reach"};

// The table's growth, its free indices, the live entries by holder and by
// memory, and the retired ones not yet taken back. Taken on GX's threads (a
// buffer's creation, a write, a death) and once a recording on the command
// processor's; never while binding.
std::mutex g_mu;
std::atomic<BufferEntry*> g_chunks[kChunks] = {};
std::uint32_t g_next = 0;
std::vector<std::uint32_t> g_free, g_retired_q;
std::unordered_map<std::uint64_t, std::uint32_t> g_by_holder;
std::map<std::uint64_t, std::uint32_t> g_by_memory;

// The copy watch: the kept default buffers' ranges, which a GPU copy may
// write (host/gxvk_watch.h): set and cleared under the table's lock, asked
// without it by every copy (copy_written). BBHOST_GXVK_BUFFERS=2 checks its
// granules against its ranges on every copy (copy_watch_check).
CopyWatch<> g_watch;
std::atomic<std::uint64_t> g_watch_checks{0}, g_watch_near{0}, g_watch_hits{0}, g_watch_missed{0};

int watch_add_locked(std::uint64_t lo, std::uint64_t hi) { return g_watch.add(lo, hi); }

void watch_remove_locked(BufferEntry& e) {
    if (e.watch_slot < 0) return;
    g_watch.remove(e.watch_slot);
    e.watch_slot = -1;
}


// Kept: immutable buffers, which Direct3D 11 - and GX after it - lets nothing
// write after their creation (no Map, no UpdateSubresource, no copy into
// them), and default ones made with data, which only an UpdateSubresource or
// a GPU copy may write: GX's update of a buffer that is not a constant
// buffer is hooked where it starts its copy (0x2ab3560, which no constant
// buffer's update reaches; engine/gx_resources.cpp), and every copy's
// destination is tested against the default ones' ranges without a lock
// (copy_written). Nothing on GX's threads takes the table's lock for a
// buffer it does not keep. (The 1a census: 7-13 default buffers with data a
// session, and some of the most bound - a shared index buffer among them.)
int not_kept(std::uint8_t usage, std::uint8_t cpu, std::uint8_t bind, std::uint32_t misc, bool initial_data) {
    if (usage >= 2) return 0;
    if (usage == 0 && !initial_data) return 1;
    if (cpu) return 2;
    if (bind & 4) return 3;
    if (bind & 0x90) return 4;
    if (!(bind & 3)) return 5;
    if (misc & 0x4000) return 6;
    return -1;
}

// Under g_mu.
bool take_index(std::uint32_t& index) {
    if (!g_free.empty()) {
        index = g_free.back();
        g_free.pop_back();
        return true;
    }
    if (g_next > kIndexMask) return false;
    const std::uint32_t c = g_next >> kChunkBits;
    if (!g_chunks[c].load(std::memory_order_relaxed)) g_chunks[c].store(new BufferEntry[1u << kChunkBits], std::memory_order_release);
    index = g_next++;
    return true;
}

void forget_locked(std::uint32_t index, const BufferEntry& e) {
    const std::uint64_t holder = e.holder, memory = e.memory;
    if (auto h = g_by_holder.find(holder); h != g_by_holder.end() && h->second == index) g_by_holder.erase(h);
    if (auto m = g_by_memory.find(memory); m != g_by_memory.end() && m->second == index) g_by_memory.erase(m);
}

// From made or placed to retired, once; queued for the command processor.
bool retire_locked(std::uint32_t index, std::uint32_t gen, Retire why) {
    BufferEntry* e = entry_at(index);
    if (!e) return false;
    std::uint32_t w = e->word.load(std::memory_order_acquire);
    for (;;) {
        const std::uint32_t st = entry_state(w);
        if (entry_gen(w) != gen || (st != kEntryMade && st != kEntryPlaced)) return false;
        if (e->word.compare_exchange_weak(w, (gen << 8) | kEntryRetired, std::memory_order_acq_rel)) break;
    }
    e->retired_why.store(static_cast<std::uint8_t>(why), std::memory_order_relaxed);
    g_retired[why].fetch_add(1, std::memory_order_relaxed);
    watch_remove_locked(*e);
    forget_locked(index, *e);
    g_retired_q.push_back(index);
    return true;
}

bool retire_live_locked(std::uint32_t index, Retire why) {
    BufferEntry* e = entry_at(index);
    return e && retire_locked(index, entry_gen(e->word.load(std::memory_order_acquire)), why);
}

// The handle a holder's +0x14 carries now (0: none): a lock-free test of
// whether the table can know the holder at all.
std::uint32_t holder_handle(std::uint64_t holder) {
    std::uint32_t h = 0;
    std::memcpy(&h, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(holder + kHolderSpareA)), 4);
    return is_handle(h) ? h : 0;
}

// The live entries whose memory overlaps [va, va + bytes), retired.
void retire_overlapping_locked(std::uint64_t va, std::uint64_t bytes, Retire why) {
    if (g_by_memory.empty() || !bytes) return;
    auto it = g_by_memory.upper_bound(va);
    if (it != g_by_memory.begin()) --it;
    std::vector<std::uint32_t> hit;
    for (; it != g_by_memory.end() && it->first < va + bytes; ++it) {
        const BufferEntry* e = entry_at(it->second);
        if (e && it->first + e->bytes > va) hit.push_back(it->second);
    }
    for (const std::uint32_t index : hit) retire_live_locked(index, why);
}

void write_handle(std::uint64_t holder, std::uint32_t handle) {
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(holder + kHolderSpareA)), &handle, 4);
}

}  // namespace

bool buffers_on() {
    static const bool on = buffers_mode() != Buffers::kOff;
    return on;
}

bool watch_writes() {
    static const bool on = buffers_mode() == Buffers::kCompare;
    return on;
}

BufferEntry* entry_at(std::uint32_t index) {
    if (index > kIndexMask) return nullptr;
    BufferEntry* c = g_chunks[index >> kChunkBits].load(std::memory_order_acquire);
    return c ? &c[index & ((1u << kChunkBits) - 1)] : nullptr;
}

std::size_t take_retired(std::uint32_t* out, std::size_t cap) {
    std::lock_guard<std::mutex> lk(g_mu);
    const std::size_t n = std::min(cap, g_retired_q.size());
    std::memcpy(out, g_retired_q.data() + (g_retired_q.size() - n), n * sizeof(std::uint32_t));
    g_retired_q.resize(g_retired_q.size() - n);
    return n;
}

void recycle(std::uint32_t index) {
    BufferEntry* e = entry_at(index);
    if (!e) return;
    std::lock_guard<std::mutex> lk(g_mu);
    const std::uint32_t gen = entry_gen(e->word.load(std::memory_order_relaxed));
    e->word.store(((gen + 1) & 0xff) << 8 | kEntryFree, std::memory_order_release);
    g_free.push_back(index);
}

bool retire(std::uint32_t index, std::uint32_t gen, Retire why) {
    std::lock_guard<std::mutex> lk(g_mu);
    return retire_locked(index, gen, why);
}

void buffer_made(std::uint64_t holder, std::uint64_t memory, std::uint32_t bytes, std::uint8_t usage, std::uint8_t cpu, std::uint8_t bind,
                 std::uint32_t misc, bool initial_data) {
    if (!buffers_on() || !holder) return;
    int why = memory && bytes ? not_kept(usage, cpu, bind, misc, initial_data) : 5;
    // A buffer the table does not keep - nearly every one GX makes, its
    // constant buffers above all - takes no lock: no handle in its holder,
    // unless the holder's memory last held a kept buffer whose end the table
    // never heard of (its handle is still there), which is retired first.
    if (why >= 0 && !holder_handle(holder)) {
        g_not_kept[why].fetch_add(1, std::memory_order_relaxed);
        write_handle(holder, 0);
        return;
    }
    std::uint32_t handle = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        // A holder or memory the table still has as live: GX gave it to this
        // buffer without the host hearing of the old one's end.
        if (auto h = g_by_holder.find(holder); h != g_by_holder.end()) retire_live_locked(h->second, kRetireReplaced);
        if (memory && bytes) retire_overlapping_locked(memory, bytes, kRetireReplaced);
        std::uint32_t index = 0;
        int slot = -1;
        if (why < 0 && usage == 0 && (slot = watch_add_locked(memory, memory + bytes)) < 0) why = 8;
        if (why < 0 && !take_index(index)) why = 7;
        if (why >= 0 && slot >= 0) g_watch.remove(slot);
        if (why >= 0) {
            g_not_kept[why].fetch_add(1, std::memory_order_relaxed);
        } else {
            BufferEntry& e = *entry_at(index);
            e.holder = holder;
            e.memory = memory;
            e.bytes = bytes;
            e.usage = usage;
            e.bind = bind;
            e.retired_why.store(0, std::memory_order_relaxed);
            e.vk_buffer = 0;
            e.vk_offset = 0;
            e.block = -1;
            e.placed_serial = 0;
            e.hash = 0;
            e.binds = 0;
            e.watch_slot = static_cast<std::int8_t>(slot);
            const std::uint32_t gen = entry_gen(e.word.load(std::memory_order_relaxed));
            e.word.store((gen << 8) | kEntryMade, std::memory_order_release);
            g_by_holder[holder] = index;
            g_by_memory[memory] = index;
            handle = handle_of(index, gen);
            g_made[usage == 1 ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
        }
    }
    // The holder is the creator's, just made: its +0x14 is ours to write
    // (no handle for a buffer the host does not keep).
    write_handle(holder, handle);
}

// Both take the lock only for a holder that carries a handle: GX's threads
// release, map and update buffers the table does not keep thousands of times
// a frame (every constant buffer's update), and a lock taken on each - one
// for every worker - more than doubled their time.
void buffer_gone(std::uint64_t holder) {
    if (!buffers_on() || !holder || !holder_handle(holder)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    if (auto h = g_by_holder.find(holder); h != g_by_holder.end()) retire_live_locked(h->second, kRetireGone);
}

void buffer_replaced(std::uint64_t holder) {
    if (!buffers_on() || !holder) return;
    std::lock_guard<std::mutex> lk(g_mu);
    if (auto h = g_by_holder.find(holder); h != g_by_holder.end()) retire_live_locked(h->second, kRetireReplaced);
}

void buffer_written(std::uint64_t holder, Retire why) {
    if (!buffers_on() || !holder || !holder_handle(holder)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    if (auto h = g_by_holder.find(holder); h != g_by_holder.end()) retire_live_locked(h->second, why);
}

void copy_written(std::uint64_t va, std::uint64_t bytes) {
    if (!g_watch.written(va, bytes)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    retire_overlapping_locked(va, bytes, kRetireCopy);
}

void copy_watch_check(std::uint64_t va, std::uint64_t bytes) {
    if (!buffers_on() || !bytes) return;
    const bool near = g_watch.near(va, bytes), hit = g_watch.hit(va, bytes);
    g_watch_checks.fetch_add(1, std::memory_order_relaxed);
    if (near) g_watch_near.fetch_add(1, std::memory_order_relaxed);
    if (hit) g_watch_hits.fetch_add(1, std::memory_order_relaxed);
    if (hit && !near) g_watch_missed.fetch_add(1, std::memory_order_relaxed);
}

void memory_written(std::uint64_t va, std::uint64_t bytes, Retire why) {
    if (!buffers_on() || !bytes) return;
    std::lock_guard<std::mutex> lk(g_mu);
    retire_overlapping_locked(va, bytes, why);
}

void memory_released(std::uint64_t va, std::uint64_t bytes) {
    if (!buffers_on() || !bytes) return;
    std::lock_guard<std::mutex> lk(g_mu);
    retire_overlapping_locked(va, bytes, kRetireUnmap);
}

void buffers_report() {
    if (!buffers_on()) return;
    const char* const mode = buffers_mode() == Buffers::kCompare ? "compared (BBHOST_GXVK_BUFFERS=2)" : "bound by handle (BBHOST_GXVK_BUFFERS=1)";
    std::string nk;
    for (int k = 0; k < 9; ++k) {
        const std::uint64_t n = g_not_kept[k].load();
        if (!n) continue;
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %llu", nk.empty() ? "" : ", ", kNotKept[k], static_cast<ull>(n));
        nk += b;
    }
    std::string rt;
    for (int k = 0; k < kRetires; ++k) {
        const std::uint64_t n = g_retired[k].load();
        if (!n) continue;
        char b[96];
        std::snprintf(b, sizeof b, "%s%s %llu", rt.empty() ? "" : ", ", kRetireName[k], static_cast<ull>(n));
        rt += b;
    }
    std::size_t live = 0, entries = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        live = g_by_holder.size();
        entries = g_next;
    }
    host_log("gxvk buffers: %s; kept %llu immutable and %llu default buffers (%zu live, %zu entries); not kept: %s; retired: %s", mode,
             static_cast<ull>(g_made[0].load()), static_cast<ull>(g_made[1].load()), live, entries, nk.empty() ? "none" : nk.c_str(),
             rt.empty() ? "none" : rt.c_str());
    if (const std::uint64_t checks = g_watch_checks.load())
        host_log("gxvk buffers: copy watch checked on %llu copies: %llu near a watched range's granules, %llu meeting one; the granules "
                 "missed %llu (none may)",
                 static_cast<ull>(checks), static_cast<ull>(g_watch_near.load()), static_cast<ull>(g_watch_hits.load()),
                 static_cast<ull>(g_watch_missed.load()));
}

}  // namespace gxvk
