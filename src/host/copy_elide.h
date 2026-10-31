// Copies left unmade (BBHOST_COPY_ELIDE): the rule, with no Vulkan in it.
//
// The game copies whole images into images of the same shape (a render
// target into a texture its later passes sample: GX's CopyResource reaching
// the command processor as a copy token). The copy's destination holds what
// the source held at the copy. As long as nothing has changed the source
// since, a pass that samples the destination can sample the source instead
// and see the same texels - and the copy itself needs to be made only when
// that stops being so: just before anything changes the source, or writes
// the destination in part, or reads the destination some other way than as
// a sampled image (a copy out of it, a readback, the display). If the
// destination is overwritten whole first (cleared whole, or the next copy
// into it), or retired, the copy is never made.
//
// A Ledger keeps those pending copies, by the images' bases. The renderer
// asks it, before each such event, which pending copies must be made now
// (`materialize`, a callback that records the copy) or forgotten, and at each
// sampled binding of a destination, which image to bind instead (`redirect`).
// The invariant, for every pending copy: no write has reached its source
// since the copy, so the source's image holds what the copy would have
// written - every source write, attachment bind, clear, fill, copy into it or
// retire goes through `touch` first. A pending copy's source is never itself
// a pending copy's destination (that one is made first), so redirection is
// one step.
//
// tests/copy_elide_test.cpp drives the ledger against a model of the images'
// contents in which every copy is made at once, and checks that every read
// sees what that model sees.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace copy_elide {

// What is about to happen to an image.
enum Touch : int {
    kWrite,      // written in part, or bound as a pass's attachment (a draw, a partial clear, a fill, a storage write)
    kOverwrite,  // written whole, nothing of what it held kept (a clear of every layer and aspect, a whole copy into it)
    kReadOther,  // read as something other than a sampled binding (a copy out of it, a readback, the display, a dump)
    kRetire,     // its image destroyed or replaced
    kTouches
};

struct Pending {
    std::uint64_t src = 0, dst = 0;  // the images' bases
    std::uint64_t serial = 0;        // when it was left unmade (the ledger's count of deferrals)
    std::uint32_t reads = 0;         // the destination's bindings redirected to the source
    std::uint32_t tag = 0;           // the caller's (the copy's kind, for its counts)
};

// Why a pending copy was made after all, or forgotten.
enum End : int {
    kEndSrcWrite,     // its source about to change (written, bound, cleared, copied into)
    kEndSrcRetire,    // its source's image about to go
    kEndDstWrite,     // its destination about to be written in part
    kEndDstRead,      // its destination about to be read another way
    kEndSrcPending,   // a copy out of it (its destination) about to be left unmade: made first, so no chain forms
    kEndLimit,        // the most pending at once (the oldest made)
    kEndFlush,        // the caller's (a submission's end, a dump)
    kEndDropped,      // forgotten: its destination overwritten whole or retired, the copy never made
    kEndSwapped,      // forgotten: its source about to be overwritten whole, the two images exchanged (the caller's take)
    kEnds
};

class Ledger {
public:
    explicit Ledger(std::size_t most = 16) : most_(most) {}

    bool live() const { return !pending_.empty(); }
    std::size_t size() const { return pending_.size(); }
    const std::vector<Pending>& pending() const { return pending_; }

    // A sampled binding of `base`: the base of the image to bind instead (a
    // pending copy's source), or 0 to bind it as it is.
    std::uint64_t redirect(std::uint64_t base) {
        for (Pending& p : pending_) {
            if (p.dst == base) {
                ++p.reads;
                return p.src;
            }
        }
        return 0;
    }
    // Whether `base` is a pending copy's destination (no read counted).
    bool is_dst(std::uint64_t base) const {
        for (const Pending& p : pending_) {
            if (p.dst == base) return true;
        }
        return false;
    }
    bool is_src(std::uint64_t base) const {
        for (const Pending& p : pending_) {
            if (p.src == base) return true;
        }
        return false;
    }
    // How many pending copies `base` is the source of, and the first.
    std::size_t sources(std::uint64_t base) const {
        std::size_t n = 0;
        for (const Pending& p : pending_) n += p.src == base;
        return n;
    }
    const Pending* find_src(std::uint64_t base) const {
        for (const Pending& p : pending_) {
            if (p.src == base) return &p;
        }
        return nullptr;
    }
    // The pending copy out of `src` let go without being made: the caller
    // exchanges the two images instead, just before the source is
    // overwritten whole - the destination then holds what the copy would
    // have written, and the overwrite lands on the image the destination
    // held, which nothing reads again before it (sources(src) must be 1).
    bool take(std::uint64_t src) {
        for (std::size_t i = 0; i < pending_.size(); ++i) {
            if (pending_[i].src != src) continue;
            pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(i));
            ++ends_[kEndSwapped];
            return true;
        }
        return false;
    }

    // Before `t` happens to the image at `base`: the pending copies it ends,
    // each made first through `make(const Pending&, End)` or forgotten through
    // `drop(const Pending&)`, in the order they were left unmade. The ended
    // copies leave the ledger before the first callback, so a callback sees
    // the ledger as it will be (and must not call back into it).
    template <class Make, class Drop>
    void touch(std::uint64_t base, Touch t, Make&& make, Drop&& drop) {
        if (pending_.empty()) return;
        ended_.clear();
        for (std::size_t i = 0; i < pending_.size();) {
            const Pending& p = pending_[i];
            int end = -1;
            if (p.src == base) {
                // Reading the source some other way leaves it as it was.
                if (t != kReadOther) end = t == kRetire ? kEndSrcRetire : kEndSrcWrite;
            } else if (p.dst == base) {
                if (t == kWrite) end = kEndDstWrite;
                else if (t == kReadOther) end = kEndDstRead;
                else end = kEndDropped;  // overwritten whole or retired: what the copy wrote is never read
            }
            if (end < 0) {
                ++i;
                continue;
            }
            ended_.push_back({p, end});
            pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(i));
        }
        finish(make, drop);
    }

    // A whole copy of `src` into `dst` (the caller has checked that the two
    // are images of one shape and format, and that neither is an attachment
    // of an open pass): left unmade, after what it ends first - the
    // destination is overwritten whole (its own pending copy forgotten, the
    // copies out of it made), a source that is itself a pending copy's
    // destination has that copy made, so no chain forms, and past the most
    // pending at once the oldest is made.
    template <class Make, class Drop>
    void defer(std::uint64_t src, std::uint64_t dst, std::uint32_t tag, Make&& make, Drop&& drop) {
        touch(dst, kOverwrite, make, drop);
        ended_.clear();
        for (std::size_t i = 0; i < pending_.size();) {
            if (pending_[i].dst != src) {
                ++i;
                continue;
            }
            ended_.push_back({pending_[i], kEndSrcPending});
            pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(i));
        }
        if (pending_.size() >= most_) {
            ended_.push_back({pending_.front(), kEndLimit});
            pending_.erase(pending_.begin());
        }
        Pending p;
        p.src = src;
        p.dst = dst;
        p.serial = ++deferred_;
        p.tag = tag;
        pending_.push_back(p);
        finish(make, drop);
    }

    // Every pending copy made now (`why`: the caller's reason).
    template <class Make>
    void flush(Make&& make, End why = kEndFlush) {
        while (!pending_.empty()) {
            const Pending p = pending_.front();
            pending_.erase(pending_.begin());
            ++ends_[why];
            make(p, why);
        }
    }

    std::uint64_t deferred() const { return deferred_; }
    std::uint64_t ends(int e) const { return ends_[e]; }

private:
    struct Ended {
        Pending p;
        int end;
    };
    template <class Make, class Drop>
    void finish(Make&& make, Drop&& drop) {
        if (ended_.empty()) return;
        std::vector<Ended> now;
        now.swap(ended_);
        for (const Ended& e : now) {
            ++ends_[e.end];
            if (e.end == kEndDropped) {
                drop(e.p);
            } else {
                make(e.p, static_cast<End>(e.end));
            }
        }
        now.clear();
        if (ended_.empty()) ended_.swap(now);  // keep the storage
    }
    std::vector<Ended> ended_;
    std::vector<Pending> pending_;  // oldest first
    std::size_t most_;
    std::uint64_t deferred_ = 0;
    std::uint64_t ends_[kEnds] = {};
};

}  // namespace copy_elide
