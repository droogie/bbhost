#pragma once

// The character id the online service issues (/basic_utils/sync_chara_id).
// The game asks for one at its network login only while the character's
// saved id (PlayerGameData+0x690) is 0, and a server that answered with an
// empty list left 0x8000000000000000 there - saved with the character, so it
// never asked again and every such character sent the same CharaId. A hook on
// that login step (0x2043ee0) treats the empty value as none, so the game
// asks once more and saves the id the server issues.

struct ElfImage;

void chara_id_install(ElfImage* image);
