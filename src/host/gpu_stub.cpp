// No Vulkan at build time: GPU work is skipped.
#include "host/gpu.h"

bool host_gpu_init(const char* const*, std::uint32_t, bool) { return false; }
bool host_gpu_available() { return false; }
bool host_gpu_memory_tight() { return false; }
bool host_gpu_small_vram() { return false; }
bool host_gpu_integrated() { return false; }
bool host_gpu_integrated_with_room() { return false; }
void host_gpu_queue_lock() {}
void host_gpu_queue_unlock() {}
std::uint64_t host_gpu_submit_presenter(void*, void*, std::uint32_t, void*, void*) { return 0; }
void host_gpu_wait_submitted(std::uint64_t) {}
GpuBusy host_gpu_busy() { return {}; }
void host_gpu_busy_present_begin(void*) {}
void host_gpu_busy_present_end(void*) {}
void host_gpu_busy_present_done() {}
std::string host_gpu_busy_report() { return {}; }
void host_gpu_queue_lock_only() {}
bool host_gpu_submit_for_flip(std::uint64_t) { return false; }
bool host_gpu_display_image(std::uint64_t, void**, std::uint32_t*, std::uint32_t*, std::uint32_t*) { return false; }
bool host_gpu_dispatch(const GpuDispatch&) { return false; }
GpuDispatchClass host_gpu_last_dispatch_class(const char** program) {
    if (program) *program = "-";
    return kDcNone;
}
bool host_gpu_draw(const GpuDraw&) { return false; }
int host_gpu_run_commit(const GpuDraw*, const std::uint64_t*, std::uint32_t, std::uint32_t&) { return 0; }
void host_gpu_flush() {}
void host_gpu_report() {}
void host_gpu_submit() {}
void host_gpu_save_pipeline_cache() {}
void host_gpu_save_pipeline_cache_at_exit() {}
std::string host_gpu_compile_report() { return {}; }
std::string host_gpu_profile_report() { return ""; }
bool host_gpu_mem_write(std::uint64_t, const void*, std::size_t) { return false; }
void host_gpu_set_write_tag(std::uint64_t, const std::uint32_t*) {}
void host_render_worker_jobs(std::uint64_t out[4]) { out[0] = out[1] = out[2] = out[3] = 0; }
bool host_gpu_mem_fill(std::uint64_t, std::uint32_t, std::size_t) { return false; }
bool host_gpu_mem_copy(std::uint64_t, std::uint64_t, std::size_t) { return false; }
bool host_gpu_gds_counters() { return false; }
bool host_gpu_gds_subgroup() { return true; }
bool host_gpu_counter_copy(std::uint64_t, std::uint64_t, std::size_t) { return false; }
bool host_gpu_counter_fill(std::uint64_t, std::uint32_t, std::size_t) { return false; }
void host_gpu_import_prefetch(std::uint64_t, std::uint64_t, std::uint8_t, std::uint8_t) {}
bool host_gpu_memory_glance(GpuMemoryGlance&) { return false; }
void host_gpu_memory_census_refresh() {}
void host_gpu_import_waits(std::uint64_t* n, std::uint64_t* us) {
    if (n) *n = 0;
    if (us) *us = 0;
}
void host_gpu_gx_buffer_made(std::uint64_t, std::uint64_t, std::uint8_t, std::uint8_t, std::uint64_t) {}
void host_gpu_gx_buffer_mapped_own(std::uint64_t, std::uint64_t, std::uint64_t) {}
void host_gpu_gx_buffer_gone(std::uint64_t, std::uint64_t) {}
bool host_gpu_copy_guest(std::uint64_t, std::uint64_t, std::size_t, std::uint64_t, std::uint64_t) { return false; }
bool host_gpu_copy_back(std::uint64_t, std::uint64_t, std::size_t) { return false; }
std::size_t host_gpu_copy_backs(const std::uint64_t*, const std::uint64_t*, const std::uint64_t*, std::size_t) { return 0; }
bool host_gpu_draw_window(std::uint64_t, const void*, std::uint32_t) { return false; }
std::uint64_t host_gpu_guest_copy_targets() { return 0; }
bool host_gpu_clear_target(std::uint64_t, std::uint32_t, std::size_t) { return false; }
bool host_gpu_clear_view(const GpuClearView&, bool, GpuClearViewResult*) { return false; }
bool host_gpu_clear_view_resolvable(const GpuClearView&, const char** why) {
    *why = "no GPU";
    return false;
}
void host_gpu_clear_view_check_begin(const GpuClearView&, const GpuClearViewResult&) {}
void host_gpu_clear_view_check_end() {}
std::string host_gpu_clear_view_report() { return {}; }
GpuStats host_gpu_stats() { return {}; }
bool host_gpu_recorder_ns(std::uint64_t*, std::uint64_t*) { return false; }
void host_gpu_phase_add(int, std::uint64_t) {}
GpuHandles host_gpu_handles() { return {}; }
void host_gpu_lock() {}
void host_gpu_unlock() {}
bool host_gpu_blit_display(void*, std::uint64_t, void*, std::int32_t, std::int32_t, std::uint32_t, std::uint32_t, std::uint32_t,
                           std::uint32_t, std::uint32_t, std::uint32_t, void*) { return false; }
bool host_gpu_dump_display(std::uint64_t, const char*, bool) { return false; }
void host_gpu_request_dump() {}
bool host_gpu_take_dump_request(std::string*) { return false; }
std::string host_gpu_capture_dir() { return {}; }
void host_gpu_capture_arm(int, const char*) {}
const char* host_gpu_draw_fail_name(int) { return nullptr; }
const char* host_gpu_capture_ext() { return ".ppm"; }
void host_gpu_glitch_watch(std::uint64_t, std::uint64_t) {}
std::uint64_t host_gpu_draw_mark() { return 0; }
std::uint64_t host_gpu_work_needs() { return 0; }
std::uint64_t host_gpu_submissions_completed() { return ~0ull; }
bool host_gpu_wait_completed(std::uint64_t, std::uint64_t) { return false; }
void* host_gpu_timeline() { return nullptr; }
void* host_gpu_present_timeline() { return nullptr; }
void host_gpu_present_reads_locked(std::uint64_t) {}
void host_gpu_note_shader_created(int, const std::uint8_t*, std::size_t, std::uint64_t) {}

void host_gpu_shadow_cp_write(std::uint64_t, std::size_t) {}
std::string host_gpu_shadow_report() { return {}; }
std::string host_gpu_recorder_report() { return {}; }
bool host_gpu_zpass_dump(std::uint64_t) { return false; }
void host_gpu_set_predication(std::uint64_t, unsigned, bool, bool) {}
std::string host_gpu_occlusion_report() { return {}; }
std::string host_gpu_image_heap_report() { return {}; }
void host_gpu_resources_report() {}
void host_gpu_texture_written(std::uint64_t, std::size_t) {}
void host_gpu_shadow_gx_write(std::uint64_t, std::size_t) {}
void host_gpu_shadow_gx_created(std::uint64_t, std::size_t) {}
void host_gpu_shadow_gx_handed(std::uint64_t, std::size_t, int) {}
bool host_gpu_shadow_hands_out() { return false; }
void host_gpu_shadow_gx_settled(std::uint64_t, std::size_t) {}
std::string host_gpu_target_copy_report() { return {}; }
std::string host_gpu_ps_wave_report() { return {}; }
std::string host_gpu_memory_budget_report() { return {}; }
std::string host_gpu_video_memory_report(bool) { return {}; }
void host_gpu_log_video_memory(bool) {}
void host_gpu_set_loading(bool) {}
void host_gpu_world_reached() {}
std::size_t host_gpu_shader_backlog() { return 0; }
void host_gpu_set_compile_session(bool) {}
HostShaderCompileProgress host_gpu_compile_progress() { return {}; }
void host_gpu_compile_cancel() {}
std::string host_gpu_device_identity() { return {}; }
int host_gpu_stream_selftest(int) { return 2; }  // no Vulkan: nothing to test
