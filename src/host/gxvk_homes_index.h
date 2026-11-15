// Stage 4b.3c (host/gxvk_homes.h): the command processor's index of the live
// homes by their buffers' memory, with a page filter in front of it - the part
// of the homes that needs no GPU (tests/gxvk_homes_test.cpp).
//
// Every lookup by address of a range a home may hold - a dispatch's constant
// buffer, a copy's source or destination, a device address, the CPU copies
// refused over a home's stale guest bytes - asks it, and nearly all meet no
// home: the filter keeps a count per 4 KiB page (2^20 counters, by a hash of
// the page) of the indexed homes touching it, so a miss reads a counter a page
// and never the map. A collision only sends a lookup to the map; a count that
// saturates stays saturated (the map answers that page from then on).
//
// The map and the filter change in place, so only the thread that changes them
// reads them: the command processor, under the renderer's lock (and a thread
// holding that lock, or a run part's draw while the command processor holds it
// at the fork). The pool's threads (stage 4b.1b's list-mode draws,
// host/gxvk_lists.h) look homes up beside it without that lock, and read
// another view the index keeps for them (read_meets): the homes touching each
// 4 KiB page as an immutable list, published by an atomic pointer. An insert
// or an erase publishes a new list for each page it touches and retires the
// one it replaces, which is freed once no reader can hold it: each reader's
// slot holds the epoch it entered at (0 outside), and a list retired at an
// epoch is freed when no slot holds that epoch or an older one. The writer
// brackets a run of changes (Batch: the events the command processor drains
// at once) with a sequence count, odd while it runs, so a reader's answer is
// the index as one run left it - never the middle of one - or "unsure" (a run
// under way through all its spins, runs between its looks through all its
// tries, a range of 64 pages or more, no slot left), which the pool's draw
// takes as a reason to be drawn in order. Until
// 2026-11-13 the pool's threads read the map and the filter themselves while
// the command processor and the texture-ahead thread inserted into them: under
// ThreadSanitizer, through a fast-loading stress, a pool thread's lookup read
// the filter's bounds as an insert wrote them, and a std::map read while
// another thread rebalances or frees its nodes can follow a freed node.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <vector>

namespace gxvk_homes {

class HomeIndex {
public:
    struct Hit {
        std::uint32_t index = 0;
        std::uint64_t memory = 0;
        std::uint32_t bytes = 0;
    };

    HomeIndex() = default;
    HomeIndex(const HomeIndex&) = delete;
    HomeIndex& operator=(const HomeIndex&) = delete;
    ~HomeIndex() {
        for (std::atomic<Leaf*>& t : top_) {
            Leaf* const leaf = t.load(std::memory_order_relaxed);
            if (!leaf) continue;
            for (std::atomic<const List*>& p : leaf->page) std::free(const_cast<List*>(p.load(std::memory_order_relaxed)));
            delete leaf;
        }
        for (const Retired& r : retired_) std::free(const_cast<List*>(r.list));
    }

    // ---- The writer's (the command processor's, under the renderer's lock) ----

    // A run of inserts and erases a reader sees whole or not at all (nested:
    // the outermost one counts). Its end frees the retired lists no reader can
    // hold any more.
    class Batch {
    public:
        explicit Batch(HomeIndex& ix) : ix_(ix) {
            if (ix_.depth_++ == 0) {
                ix_.seq_.store(ix_.seq_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);  // odd: under way
                std::atomic_thread_fence(std::memory_order_release);
            }
        }
        ~Batch() {
            if (--ix_.depth_ == 0) {
                ix_.seq_.store(ix_.seq_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
                ix_.reclaim();
            }
        }
        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;

    private:
        HomeIndex& ix_;
    };

    // Home `index` holds [memory, memory + bytes): whatever the index held at
    // that memory before is the caller's to have ended first.
    void insert(std::uint32_t index, std::uint64_t memory, std::uint32_t bytes) {
        const Batch batch(*this);
        if (!filter_) filter_.reset(new std::uint16_t[kSlots]());
        auto [it, fresh] = map_.emplace(memory, Value{index, bytes});
        if (!fresh) {
            count(it->first, it->second.bytes, -1);
            pages_remove(it->second.index, it->first, it->second.bytes);
            it->second = Value{index, bytes};
        }
        count(memory, bytes, 1);
        pages_add(index, memory, bytes);
        lo_ = std::min(lo_, memory);
        hi_ = std::max(hi_, memory + bytes);
        ++version_;
    }
    // Home `index` out of the index, if it is there at `memory`.
    bool erase(std::uint32_t index, std::uint64_t memory) {
        const auto it = map_.find(memory);
        if (it == map_.end() || it->second.index != index) return false;
        const Batch batch(*this);
        count(memory, it->second.bytes, -1);
        pages_remove(index, memory, it->second.bytes);
        map_.erase(it);
        ++version_;
        return true;
    }
    // Whether [va, va + bytes) may meet a home: false only where none does.
    bool may_meet(std::uint64_t va, std::uint64_t bytes) const {
        if (!filter_ || !bytes || va >= hi_ || va + bytes <= lo_) return false;
        const std::uint64_t first = va >> 12, last = (va + bytes - 1) >> 12;
        if (last - first >= 64) return true;  // a long range: the map answers it
        for (std::uint64_t p = first; p <= last; ++p) {
            if (filter_[slot(p)]) return true;
        }
        return false;
    }
    // The home meeting [va, va + bytes): the one at or below va whose range
    // reaches past va, else the first starting inside the range.
    bool meets(std::uint64_t va, std::uint64_t bytes, Hit* out) const {
        if (!may_meet(va, bytes)) return false;
        auto it = map_.upper_bound(va);
        if (it != map_.begin()) {
            const auto p = std::prev(it);
            if (va < p->first + p->second.bytes) {
                if (out) *out = Hit{p->second.index, p->first, p->second.bytes};
                return true;
            }
        }
        if (it != map_.end() && it->first < va + bytes) {
            if (out) *out = Hit{it->second.index, it->first, it->second.bytes};
            return true;
        }
        return false;
    }
    // [va, va + bytes) inside one home.
    bool holding(std::uint64_t va, std::uint64_t bytes, Hit* out) const {
        Hit h;
        if (!meets(va, bytes, &h) || va < h.memory || va + bytes > h.memory + h.bytes) return false;
        if (out) *out = h;
        return true;
    }
    // [va, va + bytes) meets a home but is not inside one.
    bool straddles(std::uint64_t va, std::uint64_t bytes) const {
        Hit h;
        return meets(va, bytes, &h) && (va < h.memory || va + bytes > h.memory + h.bytes);
    }
    std::size_t size() const { return map_.size(); }
    // Moves at every insert and erase: an answer kept beside it is good while it stands.
    std::uint64_t version() const { return version_; }
    // The retired lists no reader can hold any more, freed (a batch's end does it; the writer may call it any time
    // outside a batch).
    void reclaim() {
        if (retired_.empty() || depth_) return;
        // Readers entering from here on read the lists that replaced these.
        epoch_.fetch_add(1, std::memory_order_seq_cst);
        std::uint64_t oldest = ~0ull;
        for (const Slot& s : slots_) {
            const std::uint64_t at = s.at.load(std::memory_order_seq_cst);
            if (at && at < oldest) oldest = at;
        }
        std::size_t keep = 0;
        for (const Retired& r : retired_) {
            if (r.epoch < oldest) std::free(const_cast<List*>(r.list));
            else retired_[keep++] = r;
        }
        retired_.resize(keep);
    }
    // Lists retired and not freed yet (tests).
    std::size_t retired() const { return retired_.size(); }

    // ---- A reader beside the writer (a pool thread, without the writer's lock) ----

    enum Read : int { kReadNone, kReadMeets, kReadUnsure };
    // As meets(), from the published lists: the index as a run of changes left
    // it, or kReadUnsure (a run under way through kReadSpins pauses - some
    // tens of microseconds of the pool thread's time, where drawing it in order
    // costs the command processor's - runs between its looks kReadTries
    // times, a range of kReadMostPages pages or more or past the pages' reach,
    // no reader's slot left).
    Read read_meets(std::uint64_t va, std::uint64_t bytes, Hit* out) const {
        if (!bytes) return kReadNone;
        if (va + bytes < va || va + bytes > kTopVa) return kReadUnsure;
        const std::uint64_t first = va >> kPageBits, last = (va + bytes - 1) >> kPageBits;
        if (last - first >= kReadMostPages) return kReadUnsure;
        const int s = reader_slot();
        if (s < 0) return kReadUnsure;
        for (int tries = 0, spins = 0; tries < kReadTries;) {
            const std::uint64_t s0 = seq_.load(std::memory_order_acquire);
            if (s0 & 1) {
                if (++spins > kReadSpins) break;
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
                continue;
            }
            enter(s);
            Hit h;
            const bool hit = pages_meet(va, bytes, first, last, h);
            leave(s);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq_.load(std::memory_order_relaxed) != s0) {  // a run of changes came between: again
                ++tries;
                continue;
            }
            if (hit && out) *out = h;
            return hit ? kReadMeets : kReadNone;
        }
        return kReadUnsure;
    }

    // A reader's section held open (tests: what the writer retires meanwhile stays until it leaves).
    class Hold {
    public:
        explicit Hold(const HomeIndex& ix) : ix_(ix), s_(ix.reader_slot()) {
            if (s_ >= 0) ix_.enter(s_);
        }
        ~Hold() {
            if (s_ >= 0) ix_.leave(s_);
        }
        Hold(const Hold&) = delete;
        Hold& operator=(const Hold&) = delete;

    private:
        const HomeIndex& ix_;
        int s_;
    };

    static constexpr int kReaders = 16;                // reader threads (the pool has at most four)
    static constexpr int kReadTries = 8, kReadSpins = 4096;
    static constexpr std::uint64_t kReadMostPages = 64;

private:
    static constexpr std::uint32_t kBits = 20, kSlots = 1u << kBits;
    struct Value {
        std::uint32_t index, bytes;
    };
    static std::uint32_t slot(std::uint64_t page) { return static_cast<std::uint32_t>((page * 0x9e3779b97f4a7c15ull) >> (64 - kBits)); }
    void count(std::uint64_t memory, std::uint32_t bytes, int d) {
        for (std::uint64_t p = memory >> 12; p <= (memory + std::max<std::uint32_t>(bytes, 1) - 1) >> 12; ++p) {
            std::uint16_t& c = filter_[slot(p)];
            if (c == 0xffff) continue;  // saturated: stays so
            c = static_cast<std::uint16_t>(d > 0 ? c + 1 : (c ? c - 1 : 0));
        }
    }

    // The published lists: a page's homes by memory, immutable once published.
    struct Entry {
        std::uint64_t memory;
        std::uint32_t bytes, index;
    };
    struct List {
        std::uint32_t n;
        Entry e[1];  // n of them
    };
    // Pages by address: a top of 1 GiB leaves over a 40-bit space (the guest's; nothing past it is published, and a
    // reader is unsure there).
    static constexpr unsigned kPageBits = 12, kLeafBits = 18, kTopBits = 10;
    static constexpr std::uint64_t kTopVa = 1ull << (kPageBits + kLeafBits + kTopBits);
    struct Leaf {
        std::atomic<const List*> page[1u << kLeafBits] = {};
    };
    struct Retired {
        const List* list;
        std::uint64_t epoch;  // retired at
    };
    struct alignas(64) Slot {
        std::atomic<std::uint64_t> at{0};  // the epoch its reader entered at, 0 outside
    };

    static List* make_list(std::uint32_t n) {
        List* l = static_cast<List*>(std::malloc(sizeof(List) + (n - 1) * sizeof(Entry)));
        if (!l) std::abort();
        l->n = n;
        return l;
    }
    // The writer's: a page's slot, its leaf made if need be.
    std::atomic<const List*>& page_slot(std::uint64_t page) {
        std::atomic<Leaf*>& t = top_[page >> kLeafBits];
        Leaf* leaf = t.load(std::memory_order_relaxed);
        if (!leaf) {
            leaf = new Leaf();
            t.store(leaf, std::memory_order_release);
        }
        return leaf->page[page & ((1u << kLeafBits) - 1)];
    }
    void retire(const List* l) { retired_.push_back({l, epoch_.load(std::memory_order_relaxed)}); }
    void pages_add(std::uint32_t index, std::uint64_t memory, std::uint32_t bytes) {
        const std::uint64_t end = memory + std::max<std::uint32_t>(bytes, 1);
        if (end < memory || end > kTopVa) return;
        for (std::uint64_t p = memory >> kPageBits; p <= (end - 1) >> kPageBits; ++p) {
            std::atomic<const List*>& at = page_slot(p);
            const List* const old = at.load(std::memory_order_relaxed);
            const std::uint32_t n = old ? old->n : 0;
            List* const l = make_list(n + 1);
            std::uint32_t k = 0, j = 0;
            while (k < n && old->e[k].memory < memory) l->e[j++] = old->e[k++];
            l->e[j++] = Entry{memory, bytes, index};
            while (k < n) l->e[j++] = old->e[k++];
            at.store(l, std::memory_order_seq_cst);
            if (old) retire(old);
        }
    }
    void pages_remove(std::uint32_t index, std::uint64_t memory, std::uint32_t bytes) {
        const std::uint64_t end = memory + std::max<std::uint32_t>(bytes, 1);
        if (end < memory || end > kTopVa) return;
        for (std::uint64_t p = memory >> kPageBits; p <= (end - 1) >> kPageBits; ++p) {
            std::atomic<const List*>& at = page_slot(p);
            const List* const old = at.load(std::memory_order_relaxed);
            if (!old) continue;
            std::uint32_t k = 0;
            while (k < old->n && !(old->e[k].memory == memory && old->e[k].index == index)) ++k;
            if (k == old->n) continue;
            List* l = nullptr;
            if (old->n > 1) {
                l = make_list(old->n - 1);
                for (std::uint32_t i = 0, j = 0; i < old->n; ++i) {
                    if (i != k) l->e[j++] = old->e[i];
                }
            }
            at.store(l, std::memory_order_seq_cst);
            retire(old);
        }
    }
    // A reader's: a page's list (null: no home touches it).
    const List* page_list(std::uint64_t page) const {
        const Leaf* const leaf = top_[page >> kLeafBits].load(std::memory_order_acquire);
        return leaf ? leaf->page[page & ((1u << kLeafBits) - 1)].load(std::memory_order_seq_cst) : nullptr;
    }
    // meets() over the lists. The home holding va touches va's page: the last there starting at or below va (homes
    // do not overlap). Else the first starting inside the range: page by page, a page's first home starting past va
    // starts in that page (one starting in an earlier page of the range is on that page's list too).
    bool pages_meet(std::uint64_t va, std::uint64_t bytes, std::uint64_t first, std::uint64_t last, Hit& out) const {
        if (const List* l = page_list(first)) {
            const Entry* below = nullptr;
            for (std::uint32_t k = 0; k < l->n && l->e[k].memory <= va; ++k) below = &l->e[k];
            if (below && va < below->memory + below->bytes) {
                out = Hit{below->index, below->memory, below->bytes};
                return true;
            }
        }
        for (std::uint64_t p = first; p <= last; ++p) {
            const List* l = page_list(p);
            if (!l) continue;
            for (std::uint32_t k = 0; k < l->n; ++k) {
                if (l->e[k].memory <= va) continue;
                if (l->e[k].memory >= va + bytes) return false;
                out = Hit{l->e[k].index, l->e[k].memory, l->e[k].bytes};
                return true;
            }
        }
        return false;
    }
    // A reader's section: the epoch it enters at, announced before it reads a list (the writer frees nothing retired
    // at that epoch or after while it stays), and its leaving.
    void enter(int s) const { slots_[s].at.store(epoch_.load(std::memory_order_seq_cst), std::memory_order_seq_cst); }
    void leave(int s) const { slots_[s].at.store(0, std::memory_order_release); }
    // This thread's reader slot (-1: none left).
    int reader_slot() const {
        thread_local const HomeIndex* t_owner = nullptr;
        thread_local int t_slot = -1;
        if (t_owner != this) {
            t_owner = this;
            const int s = next_slot_.fetch_add(1, std::memory_order_relaxed);
            t_slot = s < kReaders ? s : -1;
        }
        return t_slot;
    }

    std::map<std::uint64_t, Value> map_;
    std::unique_ptr<std::uint16_t[]> filter_;
    std::uint64_t lo_ = ~0ull, hi_ = 0;  // every indexed home lies inside
    std::uint64_t version_ = 0;
    // The published side.
    std::atomic<Leaf*> top_[1u << kTopBits] = {};
    std::atomic<std::uint64_t> seq_{0};    // odd while a run of changes is under way
    std::atomic<std::uint64_t> epoch_{1};  // moves at each reclaim
    int depth_ = 0;                        // the writer's batches open
    std::vector<Retired> retired_;
    mutable Slot slots_[kReaders];
    mutable std::atomic<int> next_slot_{0};
};

}  // namespace gxvk_homes
