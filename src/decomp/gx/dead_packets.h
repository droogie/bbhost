// The Gnm packet writers whose packets nothing on the host reads, as our
// source (decomp/gx/dead_packets.cpp, docs/decomp.md): their table, for the
// census hooks in hle/gx_trace.cpp, and the mode the command processor's
// setup gives each of them.
#pragma once

#include <cstddef>
#include <cstdint>

struct GnmDeadWriter {
    const char* name;           // on the decomp list (BBHOST_DECOMP_OFF), the symbol file's where it has one
    const char* what;           // what its packet holds, for the logs
    std::uint64_t bn;           // the entry, as Binary Ninja addresses it
    std::uint8_t n;             // the prologue below: whole instructions, at least 14 bytes (a census hook's jump)
    std::uint8_t bytes[20];
    std::uint8_t dwords;        // the packet's dwords, for the reports
    bool context;               // context registers: dead only while the register file is not kept
    bool dead;                  // written as nothing by default; false: the game's packet, kept for its side effects
};

inline constexpr std::size_t kGnmDeadWriterCount = 21;
const GnmDeadWriter& gnm_dead_writer(std::size_t i);

// What ours writes: the game's packet exactly (the default until configured),
// nothing - the write pointer back, as the address no caller reads - or
// nothing and an address no caller could follow (0xdead000000000000, a
// check: a caller that used it would fault on the spot).
enum class GnmDeadMode : std::uint8_t { kWrite = 0, kNothing = 1, kPoison = 2 };

// BBHOST_CP_DEAD_KEEP=<i,j,...> (a bisection aid): the writers at those
// places in the table write the game's packets. A bit a place; parsing stops
// at the first item that is not a number.
std::uint64_t gnm_dead_keep_mask(const char* list);

// hle_gx_trace_arm, once the command processor's mode is known (before any of
// the game's code runs): each writer the table marks dead takes the
// context-register writers' mode or the others', unless `keep` has its bit;
// the others write the game's packets.
void gnm_dead_writers_configure(GnmDeadMode others, GnmDeadMode context, std::uint64_t keep = 0);

// The mode a writer was given (kWrite until configured).
GnmDeadMode gnm_dead_writer_mode(std::size_t i);

// Whether that writer is ours (on the decomp list and placed), after
// decomp_install.
bool gnm_dead_writer_placed(std::size_t i);
