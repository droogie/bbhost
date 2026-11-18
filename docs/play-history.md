# Play history

bbhost records each character's NG/NG+ playtime, attempts, and kills for 17 base-game bosses, five DLC bosses, and chalice bosses. Campaign and chalice records have separate lists and totals.

Open **Play history** in setup or the in-game F10 menu. Select a character and a cycle, or enable **Show totals across all cycles**. Switch between **Base game and DLC** and **Chalice dungeons**.

## Counting fights

An attempt starts when the fight's start event enables its AI and health bar. Standing outside the fog does not count. Deaths, retreats, interrupted fights, and the winning fight count as attempts: two losses followed by a win give three attempts and one kill. Resuming an active fight does not count it twice. A kill takes precedence if the player dies at the same time.

Recording pauses during boss rush/practice and visits to another player's world. Practice results remain in the boss-rush plugin's own records. Plugin API 30 appends `campaign_tracking_pause(name, paused)` after the API 29 fields; practice plugins can hold and release their own pause on the main thread. Older API field offsets are unchanged.

## Time and incomplete records

Time follows the save's play clock. A rising NG+ counter ends the previous cycle. At the game's clock limit, 999:59:59, measurement continues using elapsed time in the world, excluding title screens and loading. Time while bbhost is closed is never added.

When first observing an NG save, its saved total time belongs to cycle one. For a save already in NG+, earlier cycle boundaries cannot be reconstructed: the current cycle starts with a partial time record. Unobserved cycles are not invented. Missing a cycle boundary, for example after playing in another build, also makes the adjacent times partial.

Previously defeated campaign bosses can be marked defeated, but earlier attempts cannot be recovered. **Unknown**, **+**, and **partial record** identify missing information. New cycles are recorded separately.

## Storage

`play-history.txt` in the data folder persists across sessions. Records are associated with save slots; renaming a character updates its display name. A decrease in cycle number, a fresh NG character, or a clock rollback greater than five minutes creates a separate record while retaining the previous one. Its totals do not mix with the old record. Smaller rollbacks, such as the gap since the last autosave, retain attempts and resume timing.

The record is saved on attempts, kills, cycle changes, loading, and normal exit, and roughly every ten seconds during play. A forced exit may lose time since the last write. Writes use a temporary file and replacement. Unreadable, malformed, or future-version records are reported without being overwritten.

Back up history files together with the game save. Tracking does not write game saves or event flags, or upload records.

## Chalice dungeons

Fixed and root chalice encounters are identified by dungeon, altar, layer, and boss. Encounters with the same boss in different dungeons keep separate active/cleared state. The multi-enemy Watchers encounter counts as one boss fight.

Chalice progress survives NG+. Attempts and kills belong to the cycle in which they were observed; an existing clear is not counted again in the next cycle. Replaying a cleared dungeon starts a separate encounter and retains the prior clear. Totals combine matching boss types only within the selected character and cycles.

An encounter already cleared when recording begins is shown as a prior clear, without assigning its kill to the current cycle. Initial attempts are lower bounds because earlier attempts are unknown.

Chalice records live in `chalice-history.txt`; the campaign file's format is unchanged. Keep the save and both history files together when backing up or moving to another PC so character identifiers remain associated.
