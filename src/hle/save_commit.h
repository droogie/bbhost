// A save written whole or not at all (hle/system.cpp's save mounts).
//
// A save is a directory, <data>/saves/<dir>. A mount opened to write gets a
// stage instead - a copy of that directory beside it, <data>/saves/.<dir>.stage
// - and the game's writes go there. The unmount makes the stage the save:
// every file in it flushed, a marker written and flushed beside it
// (.<dir>.commit, put in place whole by a rename), then
//
// - where the file system has it (Linux: ext4, btrfs, xfs, tmpfs), the two
//   directories swapped in one renameat2(RENAME_EXCHANGE), the marker naming
//   the stage's inode so a recovery can tell which side holds what; then the
//   old copy and the marker removed;
// - elsewhere (Windows, and exFAT/vfat or NFS on Linux, which refuse the
//   exchange) file by file, the save's directory itself never renamed or
//   removed - a File Explorer window open in it stays there: the marker lists
//   every file and folder in the stage, each listed file is moved from the
//   stage over the save's own (a rename, which replaces it in one step), the
//   save's files and folders the list lacks (the game deleted them) are
//   removed, then the marker and the stage.
//
// A process killed at any point leaves either the old save or the new one,
// whole: recover() (before any mount of the save) finishes a commit whose
// marker was written and drops a stage whose marker was not. Windows' moves
// are tried again for a while when another process (an antivirus scan,
// OneDrive) holds the file; one that never goes through leaves the marker and
// the stage for the next recovery, which until it can finish copies the files
// already moved back into the stage and reads the save from there.
//
// The recovery also finishes what v0.2.32-v0.2.34 left: their commit without
// the exchange renamed the directories (the save to .<dir>.old, the stage to
// the save) under a marker "rename", or the exchange's when it was refused.
#pragma once

#include <string>

namespace save_commit {

struct Paths {
    std::string live, stage, old, marker, marker_new;
};
// The stage, the old copy (v0.2.32-v0.2.34), the marker and the marker being
// written beside a save's directory `live`.
Paths paths_of(const std::string& live);

// What an interrupted commit or session left, settled before the save is
// mounted: a stage whose marker was written becomes the save; one without
// is dropped; an old copy is put back when the save itself is missing, else
// removed. Returns the directory that holds the newest complete save - `live`,
// or the stage when a commit could not be finished (a move or a rename
// refused) - or "" when neither does (the stage could not be made whole
// again), and a note of what it did in `*did` ("" when nothing was there to
// settle).
std::string recover(const std::string& live, std::string* did = nullptr);

// Opens a session that writes: the save's directory (none for a new save) is
// copied into the stage. Returns the stage's path, or "" when it could not be
// made - the caller then writes in place, as before.
std::string begin(const std::string& live);

// Makes the stage the save (see above). False when a step failed: what is
// there is left for recover(), and `*why` says which step.
bool commit(const std::string& live, std::string* why = nullptr);

// Tests. A hook called at each step with its name, where a test may end the
// process: "copy" (begin), "flushed", "marker-written" (the marker whole
// under its own name, not in place yet), "marker"; the exchange's
// "exchanged", "old-removed", "marker-removed"; file by file "moved" (each
// file), "extras-removed", "marker-removed", "stage-removed", and a
// recovery's "copied-back" (each file); v0.2.32-v0.2.34's renames
// "live-moved", "stage-moved", "marker-removed", "old-removed".
// The exchange not tried; the exchange tried but refused, as by a file
// system without it (exFAT, vfat, NFS); commits as v0.2.32-v0.2.34 made
// them, to test the recovery of what they leave; and the move of one of the
// stage's files (its path in the save, "" for none) refused, as when another
// process holds the file.
using Hook = void (*)(const char* step);
void set_test_hook(Hook hook);
void set_no_exchange(bool on);
void set_exchange_refused(bool on);
void set_old_renames(bool on);
void set_refused_move(const std::string& rel);

}  // namespace save_commit
