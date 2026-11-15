#include "engine/font_handle_set.h"

#include <cstring>

namespace {

constexpr std::int64_t kEmpty = -2, kEnd = -1;
constexpr std::uint64_t kMaxMask = 0xffff;  // far past any movie's handles: a larger mask is not this table

struct Entry {
    std::int64_t next;
    std::uint64_t natural;
    std::uint64_t handle;
    std::uint8_t byte;
    std::uint8_t pad[7];
};
static_assert(sizeof(Entry) == 32, "an entry is 32 bytes");

Entry* entry_at(std::uint8_t* table, std::uint64_t i) { return reinterpret_cast<Entry*>(table + 16 + i * 32); }

}  // namespace

bool font_set_remove_handle(std::uint8_t* table, std::uint64_t handle) {
    if (!table || !handle) return false;
    std::uint64_t count = 0, mask = 0;
    std::memcpy(&count, table, 8);
    std::memcpy(&mask, table + 8, 8);
    if (!count || mask > kMaxMask || ((mask + 1) & mask) != 0) return false;  // a size of 2^n
    // The entry holding the handle: every slot looked at, no hash needed.
    std::uint64_t at = mask + 1;
    for (std::uint64_t i = 0; i <= mask; ++i) {
        const Entry* e = entry_at(table, i);
        if (e->next != kEmpty && e->handle == handle) {
            at = i;
            break;
        }
    }
    if (at > mask) return false;
    Entry* e = entry_at(table, at);
    const std::uint64_t head = e->natural;
    if (head > mask) return false;
    if (head == at) {
        // The chain's head: its next entry moves into the head's slot.
        if (e->next == kEnd) {
            e->next = kEmpty;
        } else {
            if (e->next < 0 || static_cast<std::uint64_t>(e->next) > mask) return false;
            Entry* n = entry_at(table, static_cast<std::uint64_t>(e->next));
            if (n->next == kEmpty) return false;
            e->next = n->next;
            e->natural = n->natural;
            e->handle = n->handle;
            e->byte = n->byte;
            n->next = kEmpty;
        }
    } else {
        // Any other entry: the one before it in the chain, from the head on.
        std::uint64_t prev = head;
        for (std::uint64_t steps = 0;; ++steps) {
            const Entry* p = entry_at(table, prev);
            if (steps > mask || p->next == kEmpty || p->next == kEnd || static_cast<std::uint64_t>(p->next) > mask) return false;
            if (static_cast<std::uint64_t>(p->next) == at) break;
            prev = static_cast<std::uint64_t>(p->next);
        }
        entry_at(table, prev)->next = e->next;
        e->next = kEmpty;
    }
    --count;
    std::memcpy(table, &count, 8);
    return true;
}
