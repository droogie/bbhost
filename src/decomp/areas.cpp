// Every engine area's functions on the decomp list, in one place
// (docs/decomp.md, "Adding one"). Each area's file adds its functions in a
// decomp_<area>_add(); hle/runtime.cpp calls decomp_add_areas() once,
// before decomp_install(), and the list places them.
#include "decomp/decomp.h"

void decomp_flag_store_add();          // events/flag_store.cpp: SprjEventFlagMan's four flag functions
void decomp_event_interpreter_add();   // events/interpreter.cpp: the EMEVD interpreter
void decomp_player_data_add();         // player/player_data.cpp: echoes, the level price, item discovery
void decomp_item_lots_add();           // items/item_lots.cpp: the item-lot roll
void decomp_talk_evaluator_add();      // talk/evaluator.cpp: the EzState expression evaluator
void decomp_chalice_ritual_add();      // chalice/ritual.cpp: a chalice ritual's roll, map uid and pairs
void decomp_follow_camera_add();       // camera/follow_camera.cpp: ChrExFollowCam::Update
void decomp_gx_flush_wait_add();       // gx/flush_wait.cpp: the render path's flush wait
void decomp_gx_block_reclaim_add();    // gx/block_reclaim.cpp: the resource tables' block reclaim
void decomp_gx_drain_token_add();      // gx/drain_token.cpp: the pipeline drain as a barrier token
void decomp_gx_query_token_add();      // gx/query_token.cpp: an occlusion query's packets as one token
void decomp_gx_dead_packets_add();     // gx/dead_packets.cpp: the Gnm packet writers nothing reads
void decomp_game_memcpy_add();         // memory/game_memcpy.cpp: the eboot's memcpy
void decomp_ez_copy_add();             // memory/ez_copy.cpp: the streamed data's parallel copy
void decomp_sfx_ribbons_add();         // sfx/ribbons.cpp: the effect ribbons' writers
void decomp_adaptive_mutex_add();      // sync/adaptive_mutex.cpp: DLAdaptiveMutex::Lock

void decomp_add_areas() {
    decomp_flag_store_add();
    decomp_event_interpreter_add();
    decomp_player_data_add();
    decomp_item_lots_add();
    decomp_talk_evaluator_add();
    decomp_chalice_ritual_add();
    decomp_follow_camera_add();
    decomp_gx_flush_wait_add();
    decomp_gx_block_reclaim_add();
    decomp_gx_drain_token_add();
    decomp_gx_query_token_add();
    decomp_gx_dead_packets_add();
    decomp_game_memcpy_add();
    decomp_ez_copy_add();
    decomp_sfx_ribbons_add();
    decomp_adaptive_mutex_add();
}
