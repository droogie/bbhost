// A save written whole or not at all (hle/system.cpp's save mounts).
//
// A save is a directory, <data>/saves/<dir>. A mount opened to write gets a
// stage instead - a copy of that directory beside it, <data>/saves/.<dir>.stage
// - and the game's writes go there. The unmount makes the stage the save:
// every file in it flushed, a marker written and flushed beside it
// (.<dir>.commit), then the directories swapped, the marker and the old copy
// removed. A process killed at any point leaves either the old save or the new
// one, whole: recover() (before any mount of the save) finishes a swap whose
// marker was written and drops a stage whose marker was not.
//
// The swap is one renameat2(RENAME_EXCHANGE) where the file system has it
// (Linux: ext4, btrfs, xfs, tmpfs), the marker naming the stage's inode so a
// recovery can tell which side holds what; elsewhere (Windows, and exFAT/vfat
// or NFS on Linux, which refuse the exchange) two renames - the save to
// .<dir>.old, the stage to the save - where the marker and which of the three
// directories exist say how far it got. Windows' renames are tried again for
// a while when another process (an antivirus scan, OneDrive) holds a file in
// them; one that never goes through leaves the stage and its marker, for the
// next recovery.
#pragma once

#include <string>

namespace save_commit {

struct Paths {
    std::string live, stage, old, marker;
};
// The stage, the old copy and the marker beside a save's directory `live`.
Paths paths_of(const std::string& live);

// What an interrupted commit or session left, settled before the save is
// mounted: a stage whose marker was written becomes the save; one without
// is dropped; an old copy is put back when the save itself is missing, else
// removed. Returns the directory that holds the newest complete save - `live`,
// or the stage when a swap could not be finished (a rename refused) - and a
// note of what it did in `*did` ("" when nothing was there to settle).
std::string recover(const std::string& live, std::string* did = nullptr);

// Opens a session that writes: the save's directory (none for a new save) is
// copied into the stage. Returns the stage's path, or "" when it could not be
// made - the caller then writes in place, as before.
std::string begin(const std::string& live);

// Makes the stage the save (see above). False when a step failed: what is
// there is left for recover(), and `*why` says which step.
bool commit(const std::string& live, std::string* why = nullptr);

// Tests. A hook called at each step with its name ("copy", "flushed",
// "marker", "exchanged", "live-moved", "stage-moved", "marker-removed",
// "old-removed"), where a test may end the process; the exchange not tried
// (the renames under their own marker); and the exchange tried but refused,
// as by a file system without it (exFAT, vfat, NFS: the renames under the
// exchange's marker).
using Hook = void (*)(const char* step);
void set_test_hook(Hook hook);
void set_no_exchange(bool on);
void set_exchange_refused(bool on);

}  // namespace save_commit
