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
void decomp_gx_labels_add();           // gx/labels.cpp: the Gnm label, wait, inline-data and predication writers
void decomp_gx_dma_add();              // gx/dma.cpp: the Gnm DMA_DATA writer
void decomp_gx_constant_engine_add();  // gx/constant_engine.cpp: the Gnm constant-engine writers
void decomp_gx_draw_add();             // gx/draw.cpp: GX's draw, dispatch and commit methods, the post-draw, the buffer's flush
void decomp_gx_command_list_add();     // gx/command_list.cpp: GX's command lists finished and executed, a segment to the GPU
void decomp_gx_present_add();          // gx/present.cpp: the swap chain's present - the frame's end and its pacing
void decomp_gnmx_cue_add();            // gnmx/cue.cpp: Gnmx's constant update engine around the flip pass's draw
void decomp_yebis_gnm_add();           // yebis/gnm.cpp: YEBIS's Gnm draw and dispatch wrappers
void decomp_yebis_context_add();       // yebis/context.cpp: YEBIS's D3D11-shaped context: targets, draws, flushes, waits
void decomp_yebis_ppfx_add();          // yebis/ppfx.cpp: PPFX's entry points - the post-processing chain as one call
void decomp_yebis_ppfx_setters_add();  // yebis/ppfx_setters.cpp: PPFX's parameter setters - each frame's post-processing parameters
void decomp_yebis_clear_add();         // yebis/clear.cpp: ClearRenderTargetView and its compute clear
void decomp_game_memcpy_add();         // memory/game_memcpy.cpp: the eboot's memcpy
void decomp_ez_copy_add();             // memory/ez_copy.cpp: the streamed data's parallel copy
void decomp_sfx_ribbons_add();         // sfx/ribbons.cpp: the effect ribbons' writers
void decomp_adaptive_mutex_add();      // sync/adaptive_mutex.cpp: DLAdaptiveMutex::Lock
void decomp_ez_work_add();             // sync/ez_work.cpp: CSEzWork::ProcessOne - the work pools' job pop, and their node caches

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
    decomp_gx_labels_add();
    decomp_gx_dma_add();
    decomp_gx_constant_engine_add();
    decomp_gx_draw_add();
    decomp_gx_command_list_add();
    decomp_gx_present_add();
    decomp_gnmx_cue_add();
    decomp_yebis_gnm_add();
    decomp_yebis_context_add();
    decomp_yebis_ppfx_add();
    decomp_yebis_ppfx_setters_add();
    decomp_yebis_clear_add();
    decomp_game_memcpy_add();
    decomp_ez_copy_add();
    decomp_sfx_ribbons_add();
    decomp_adaptive_mutex_add();
    decomp_ez_work_add();
}
