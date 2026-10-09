#pragma once

#include "guest_abi.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <functional>
#include <utility>
#include <vector>

void hle_register_libc();
void hle_register_pthread();
void hle_register_kernel();
void hle_register_fs();
void hle_register_fios();
void hle_register_dialog();
void hle_register_avplayer();
void hle_register_video();
void hle_register_gnm();
void hle_register_audio();
void hle_register_net();
void hle_register_http();
void hle_register_ajm();
void hle_register_system();

struct HostEqueue;
void hle_video_detach_equeue(HostEqueue* eq);
void hle_gnm_detach_equeue(HostEqueue* eq);
void hle_kernel_memory_report();
// True when [va, va+len) lies inside guest-visible host mappings (dmem, alias, flexible).
bool hle_kernel_va_mapped(std::uint64_t va, std::size_t len);
// hle_kernel_va_mapped, and the mapped range [*lo, *hi) that holds the range.
bool hle_kernel_va_mapped_in(std::uint64_t va, std::size_t len, std::uint64_t* lo, std::uint64_t* hi);
void hle_guest_pool_report();
int hle_guest_pool_free();
// Who still points at these, with a count each and the first pointer that is
// not beside the value itself. Returns how many pointers were found.
int hle_kernel_find_pointers(const std::uint64_t* values, int n, int cap, int* counts = nullptr,
                             std::uint64_t* first_outside = nullptr);  // entries free in that pool, -1 when it cannot be read  // runtime.cpp: the pool the frame loop spins on when it freezes
bool hle_kernel_va_readable(std::uint64_t va, std::size_t len);  // mapped with CPU read access
bool hle_kernel_va_readable_in(std::uint64_t va, std::size_t len, std::uint64_t* lo, std::uint64_t* hi);  // and the readable range
// Direct memory backing: the memfd and its size (the GPU imports it).
int hle_kernel_dmem_fd();
std::uint64_t hle_kernel_dmem_size();
// Before the direct memory is first mapped (the GPU's mirror, the game's
// first allocation): its size, rounded to 16 MiB, between 6 GiB and the
// window reserved for it (7.75 GiB). False once it exists at another size.
bool hle_kernel_set_dmem_size(std::uint64_t bytes);
// A second, complete mapping of the direct memory backing (the GPU imports
// it in chunks): mmap of the memfd on Linux, a view of the section on Windows.
void* hle_kernel_dmem_mirror();
// The image slide (0 before the eboot is bound), for diagnostics that print
// ELF addresses.
std::uint64_t hle_image_slide();
// Windows: reserves the guest's fixed address windows as placeholders; call
// first thing in main, before anything else can take them. Linux: no-op.
bool hle_kernel_reserve_guest_windows();
std::uint64_t hle_kernel_dmem_high();  // highest allocated physical end
// Live guest mappings, for the GPU page table.
struct GuestMapInfo {
    std::uint64_t va;
    std::uint64_t len;
    std::int64_t phys;  // -1: anonymous (flexible) memory
    bool dmem;
    int prot;           // SCE_KERNEL_PROT_* bits
};
void hle_kernel_snapshot_maps(std::vector<GuestMapInfo>& out);
std::uint64_t hle_kernel_maps_generation();  // changes whenever the map list does
std::uint64_t hle_kernel_map_host(std::uint64_t len);  // host read/write memory in the GPU page table; 0 on failure
std::uint64_t hle_kernel_gpu_alias(std::uint64_t va);  // direct memory's GPU alias of a CPU address; 0 when none
std::uint64_t hle_kernel_mappings_of(std::uint64_t va);
// Write-protects [va, va+len) for the texture cache when it is direct memory
// mapped CPU read/write (core/write_watch.h); false when it is not.
struct WriteWatch;
bool hle_kernel_write_watch(std::uint64_t va, std::size_t len, WriteWatch& w);  // logs every mapping of va's physical memory; returns another CPU address of it, or 0
// The same over the range's GPU alias, when it has one (writes through it do
// not touch the CPU view's pages): the alias address, or 0.
std::uint64_t hle_kernel_write_watch_alias(std::uint64_t va, std::size_t len, WriteWatch& w);
// Perform the CPU-visible writes of a PM4 stream (fences, labels, WRITE_DATA, DMA).
void hle_gnm_execute(const void* cb, std::size_t bytes);
// Queue command buffers for the command-processor thread; `done` runs after they execute.
// queue_id 0 = graphics ring; compute rings use their sceGnmMapComputeQueue id.
// `ce_cbs` are the constant-engine buffers (the CCBs) paired with `cbs` by
// index; the command processor interleaves them the way the hardware does.
void hle_gnm_submit_async(int queue_id, const std::vector<std::pair<const void*, std::size_t>>& cbs,
                          std::function<void()> done,
                          const std::vector<std::pair<const void*, std::size_t>>& ce_cbs = {});
// BBHOST_CE_HISTORY=1 (diagnostic, CP thread only): log which DumpConstRam
// produced the memory at `va` and the WriteConstRam packets that filled it.
void hle_gnm_explain_table(std::uint64_t va, std::size_t bytes);
void hle_gnm_cp_drain();
std::uint64_t hle_gnm_cp_backlog();
// Command buffers submitted by the game and retired by the command processor,
// summed over the queues: work submitted before `submitted` is done once
// `retired` reaches it.
std::uint64_t hle_gnm_submitted_total();
std::uint64_t hle_gnm_retired_total();
std::string hle_gnm_flush_reasons();
// BBHOST_ARENA_MONITOR=1: per-flip scan of the game's command-arena markers
// (logs `arena: LOW`). The report carries the arena-wait counts whenever the
// wait has run, and the per-arena states under the monitor ("" otherwise).
void hle_gnm_arena_flip();
std::string hle_gnm_arena_report();
// Replaces the game's command-arena acquire (guest 0x2ad3b80, runtime.cpp):
// the same scan, but a failed one waits for the GPU to free a chunk.
GUEST_ABI int hle_gx_arena_acquire(std::uint64_t* allocator, std::uint64_t* out);
// Whether [va, va + bytes) lies in a command arena the host acquire has seen;
// `chunk_base` (optional) gets the chunk holding va.
bool hle_gx_arena_overlap(std::uint64_t va, std::uint64_t bytes, std::uint64_t* chunk_base);
bool hle_gx_arena_marker(std::uint64_t va);  // the last word of a chunk in a known arena
bool hle_gx_arena_covers_marker(std::uint64_t va, std::uint64_t bytes);  // a range covering any chunk's marker
bool hle_gnm_arena_monitor();  // BBHOST_ARENA_MONITOR=1
// The buttons the last scePadRead handed the game (system.cpp).
std::uint32_t hle_pad_delivered_buttons();
std::string hle_av_stats();     // " movie-frames=N movie-audio-chunks=N"
std::string hle_audio_stats();  // " audio-port<h>=<writes>" per open port  // " op:count" per CP packet that forced a flush
void hle_gnm_exec_stats(std::uint64_t* packets, std::uint64_t* writes);
std::uint64_t hle_gnm_draw_count();  // draws the command processor has walked
// The command processor's own write into guest memory: shadowed for the
// renderer and deferred when it lands on a command buffer, the way a packet's
// write is. A native component that takes a copy over from the guest's
// constant engine has to make it the same way.
void hle_gnm_cp_write(std::uint64_t va, const void* src, std::size_t n);
void hle_gnm_exec_histogram();
// PM4 packets the command processors executed since the last call, for the
// 300-flip report: the total, native-draw tokens, and each opcode, most first.
std::string hle_gnm_op_window();
// BBHOST_GNM_SOURCES=1: command-buffer dwords by the source of the draw or dispatch that ends them, since the last call.
std::string hle_gx_source_window();
// The last CP label writes (watchdog diagnostics).
void hle_gnm_label_report();
// Addresses whose waits timed out, against how often we wrote them.
void hle_gnm_wait_report();
// Diagnostic: warn when a GPU-side write lands on a recently submitted command
// buffer (which would erase packets the command processor has not read yet).
void hle_gnm_warn_if_command_buffer(const char* what, std::uint64_t va, std::size_t bytes);
void hle_gnm_dump_recent_writes(std::uint64_t near, unsigned count);
// Crash diagnostics: the recorded CP writes over [va, va + bytes) or of `value`.
void hle_gnm_find_writes(std::uint64_t va, std::uint64_t bytes, std::uint64_t value);
// `recorded`: called by the command processor once the frame's commands are
// recorded and submitted (the frame may then be shown at once); false from a
// game thread's flip request, which may be ahead of them.
void hle_video_finish_flip(int handle, int buffer, std::int64_t arg, bool recorded = false);
std::uint64_t hle_video_flip_count();
// Since the last call, for the 300-flip report: how late the vblank clock's
// ticks woke, and how long flips took from queued to shown and to completed.
std::string hle_video_pacing_window();
int hle_save_writable_mounts();  // saves mounted for writing now (system.cpp)
// The picture inside the display buffers: the top-left w x h of each is what
// the window shows (a live resolution change renders into buffers allocated
// for the largest size). 0, 0 shows the whole buffer.
void hle_video_set_picture(unsigned w, unsigned h);
// A pad press the game reads as the player's: `button` (the pad's bits) from
// `delay_ms` on, held `hold_ms`. Any thread.
void hle_pad_tap(std::uint32_t button, int delay_ms, int hold_ms);
// A pad button by the name BBHOST_AUTOPRESS gives it (cross, circle, l1, ...),
// 0 for none. The stick names (lup, rright, ...) are bits above the pad's,
// which only the autopress script turns into a deflection.
std::uint32_t hle_pad_button_named(const std::string& name);
void hle_video_set_fps_cap(int fps);  // video.fps_cap (upper bound on presented fps)
// The game's pace once engine/frame_rate.cpp chose it (30, 60, 90, 0 uncapped):
// 30 and 60 cap the flips at that rate; 90 and uncapped complete them at once.
// A frame rate set after that applies on the next run.
void hle_video_set_game_pace(int fps);
void hle_video_set_loading_uncapped(bool on);  // flips complete at once while a loading screen is up (engine/loading.cpp)
std::uint64_t hle_video_loading_unshown();      // loading flips completed without being shown, so far (video.cpp)
void hle_dialog_set_default_name(const char* name, bool type_in_window);  // IME text (player.name) and whether to type it
void hle_np_set_online_id(const char* id);          // sceNpGetOnlineId (online.online_id)

// Debug soft-watchpoint (BBHOST_WATCH_FILL): record the CPU RIP that writes
// into [lo,hi). Used to find which guest GFx code writes the collapsed
// fill-mesh vertex corners. No-op cost when disarmed.
void hle_watch_arm(std::uint64_t lo, std::uint64_t hi);
void hle_watch_arm_reads(std::uint64_t lo, std::uint64_t hi);
void hle_watch_report();

// Debug int3 breakpoint (BBHOST_BP_TESS): capture arg3 (RDX) at the Scaleform
// shape tessellator sub_4b71b0 to read the mesh packing Matrix2x4.
void hle_bp_arm(std::uint64_t va);
void hle_bp_report();

// BBHOST_GX_TRACE=<first>,<last>: hook the GX immediate draw
// and dispatch methods and the Gnm packet emitters, match executed draw and
// dispatch packets back to them, and log a coverage summary after flip <last>.
// Diagnostic only.
struct GuestMemory;
void hle_gx_trace_arm(GuestMemory* mem);
// Command-processor side, for every executed draw / dispatch packet (no-op unless armed).
// BBHOST_GX_BACKEND=1/2: returns the draw's inputs built from GX state (valid
// until the next call on this thread) when the packet came from a GX draw the
// backend takes; `render` says whether to draw from them or only compare.
struct GpuDrawInputs;
const GpuDrawInputs* hle_gx_trace_draw(std::uint32_t op, std::uint64_t packet_va, std::uint32_t count,
                                       std::uint32_t instances, std::uint64_t index_base, std::uint32_t index_type,
                                       std::uint32_t prim, const std::uint32_t* sh, const std::uint32_t* ctx,
                                       bool* render);
void hle_gx_trace_dispatch(std::uint32_t op, std::uint64_t packet_va, std::uint32_t x, std::uint32_t cs_lo);
// BBHOST_GX_NATIVE: the CP walked a host-draw token, a NOP whose
// payload is kGxNativeMagic and a 64-bit id; sh / ctx are its register file.
// With BBHOST_GX_NATIVE=2 returns the draw's inputs (its GX objects through
// hle_gx_trace_objects) for the CP to draw; null otherwise.
constexpr std::uint32_t kGxNativeMagic = 0x42424e44u;  // "DNBB"
const GpuDrawInputs* hle_gx_native_token(std::uint64_t id, const std::uint32_t* sh, const std::uint32_t* ctx);
// The token's draw is read in place, from the slot the call
// left it in; the command processor gives the slot back once it has drawn it.
void hle_gx_native_release();
// The command processor's lookahead: a draw token further on in the stream.
void hle_gx_native_prefetch(std::uint64_t id, bool next);
// BBHOST_GX_NATIVE_COPIES: a GX buffer-copy pass's copy token, a type-3 NOP
// with payload {kGxCopyMagic, id low, id high}; runs the call's copies.
constexpr std::uint32_t kGxCopyMagic = 0x50434242u;  // "BBCP"
// GX's fill primitive (0x2ab5380) and the per-slice colour clear
// (0x2ab4ef0) write a fill token, {kGxFillMagic, id low, id high}, in place of
// their fill dispatch; the command processor runs the fills there.
constexpr std::uint32_t kGxFillMagic = 0x4c464242u;  // "BBFL"
void hle_gx_fill_token(std::uint64_t id, const std::uint32_t* ctx);
// The texture upload (0x2ab35e0) writes an upload token in place
// of its dispatches; the command processor copies the rows into the image.
constexpr std::uint32_t kGxUploadMagic = 0x50554242u;  // "BBUP"
void hle_gx_upload_token(std::uint64_t id);
// The texture copy (0x2ab2f40) writes a copy-image token in
// place of its dispatch; the command processor copies image to image.
constexpr std::uint32_t kGxImageCopyMagic = 0x43494242u;  // "BBIC"
void hle_gx_image_copy_token(std::uint64_t id);
// A Scaleform HAL draw (0x73e670, 0x73e6b0, 0x73e640)
// writes a draw token built from the HAL's objects. BBHOST_GX_NATIVE_SCALEFORM=2
// returns the draw's inputs (its objects through hle_gx_trace_objects) for the
// CP to draw; =1 (shadow) keeps them for the draw packet that follows, which
// hle_gx_scaleform_check compares against the register file.
constexpr std::uint32_t kGxScaleformMagic = 0x46534242u;  // "BBSF"
const GpuDrawInputs* hle_gx_scaleform_token(std::uint64_t id);
bool hle_gx_scaleform_pending();
void hle_gx_scaleform_report();  // exit report (also on SIGTERM)
std::string hle_gx_scaleform_last();  // the last Scaleform token this thread drew: its tables and slots (diagnostics)
// The register file: true when a shadow or packet-path mode is
// on, which needs the command processor to keep the context registers and
// draw from draw packets; otherwise neither is done (BBHOST_CP_REGISTER_FILE=1
// and BBHOST_CP_DRAW_PACKETS=1 keep them anyway).
bool hle_gx_shadow_wanted();
// The state fields of a token completed from the wrapper's context state,
// against the register file (BBHOST_GX_YEBIS_FULL=1).
void hle_gx_wrapper_compare(const GpuDrawInputs& in, std::uint32_t count, std::uint64_t index_va, std::uint32_t index_type,
                            std::uint32_t instances, std::uint32_t prim, const std::uint32_t* sh, const std::uint32_t* ctx);
void hle_gnm_cp_report();  // the command processor's register-file and draw-packet counters, at exit
// A compute dispatch emitted through the wrapper writes a
// dispatch token, {kGxDispatchMagic, id low, id high}, built from the
// wrapper's context state (the compute program and its user data).
// BBHOST_GX_NATIVE_DISPATCH=2 returns the dispatch for the CP to run; =1
// (shadow) keeps it for the dispatch packet that follows, which
// hle_gx_dispatch_check compares against the register file. The default, 0,
// leaves dispatches to their packets (gx_trace.cpp says why).
constexpr std::uint32_t kGxDispatchMagic = 0x53444242u;  // "BBDS"
struct GpuDispatch;
// True when dispatches run from their tokens, so the command processor must
// not also run the dispatch packet.
bool hle_gx_dispatch_native();
const GpuDispatch* hle_gx_dispatch_token(std::uint64_t id);
bool hle_gx_dispatch_pending();
// writers: for each of the 16 compute user-data slots, the header of the
// SET_SH_REG packet that last wrote it and the NOP marker before it (checks).
void hle_gx_dispatch_check(const GpuDispatch& regs, const std::uint32_t* writers);
void hle_gx_scaleform_check(std::uint32_t count, std::uint64_t index_va, std::uint32_t index_type, std::uint32_t instances,
                            std::uint32_t prim, const std::uint32_t* sh, const std::uint32_t* ctx);
void hle_gx_copy_token(std::uint64_t id);
// BBHOST_GX_NATIVE_YEBIS: YEBIS draws through its own D3D11-shaped
// context (0x15fbc40), whose commit (0x15f9720) sets every stage's shader and
// resources on each draw - 392 dwords a draw, 21% of the ring from 3% of the
// draws, the largest single source of PM4 left. A type-3 NOP with payload
// {kGxYebisMagic, id low, id high} marks one, written where its commit and
// draw packets begin. Shadow first: the method still runs, and the CP checks
// that the draw packet after the token is the one the call asked for.
constexpr std::uint32_t kGxYebisMagic = 0x42595242u;  // "BRYB"
struct GxYebisDraw {
    std::uint32_t index_count;
    std::uint64_t index_va;
    // The rest of the draw's state was built from the wrapper's
    // context state at the call (the index type with it).
    bool state_built = false;
    std::uint32_t index_type = 0;
};
bool hle_gx_yebis_token(std::uint64_t id, GxYebisDraw* out);
void hle_gx_yebis_drew(std::uint32_t index_count, std::uint64_t index_va, const GxYebisDraw& want);
// With BBHOST_GX_NATIVE_YEBIS=4: checks the inputs built from YEBIS's own state
// against the register file its commit wrote for the same draw.
void hle_gx_yebis_check(const std::uint32_t* sh);
// With BBHOST_GX_NATIVE_YEBIS=5 the method wrote no draw packet, so the CP
// draws at the token from these inputs; null when the flip is off.
const GpuDrawInputs* hle_gx_yebis_token_draw(std::uint64_t id, GxYebisDraw* out);
// With BBHOST_GX_BACKEND: the GX objects of the draw hle_gx_trace_draw last
// matched on this thread (null when it did not come through a GX method).
struct GxDrawObjects;
const GxDrawObjects* hle_gx_trace_objects();
