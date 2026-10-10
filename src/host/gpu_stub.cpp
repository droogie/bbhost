// No Vulkan at build time: GPU work is skipped.
#include "host/gpu.h"

bool host_gpu_init(const char* const*, std::uint32_t, bool) { return false; }
bool host_gpu_available() { return false; }
bool host_gpu_memory_tight() { return false; }
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
bool host_gpu_draw(const GpuDraw&) { return false; }
void host_gpu_flush() {}
void host_gpu_report() {}
void host_gpu_submit() {}
void host_gpu_save_pipeline_cache() {}
void host_gpu_save_pipeline_cache_at_exit() {}
std::string host_gpu_compile_report() { return {}; }
std::string host_gpu_profile_report() { return ""; }
bool host_gpu_mem_write(std::uint64_t, const void*, std::size_t) { return false; }
bool host_gpu_mem_fill(std::uint64_t, std::uint32_t, std::size_t) { return false; }
bool host_gpu_mem_copy(std::uint64_t, std::uint64_t, std::size_t) { return false; }
bool host_gpu_copy_guest(std::uint64_t, std::uint64_t, std::size_t) { return false; }
bool host_gpu_copy_back(std::uint64_t, std::uint64_t, std::size_t) { return false; }
bool host_gpu_draw_window(std::uint64_t, const void*, std::uint32_t) { return false; }
std::uint64_t host_gpu_guest_copy_targets() { return 0; }
bool host_gpu_clear_target(std::uint64_t, std::uint32_t, std::size_t) { return false; }
GpuStats host_gpu_stats() { return {}; }
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
const char* host_gpu_capture_ext() { return ".ppm"; }
void host_gpu_glitch_watch(std::uint64_t, std::uint64_t) {}
std::uint64_t host_gpu_draw_mark() { return 0; }
std::uint64_t host_gpu_work_needs() { return 0; }
std::uint64_t host_gpu_submissions_completed() { return ~0ull; }
void host_gpu_note_shader_created(int, const std::uint8_t*, std::size_t, std::uint64_t) {}

void host_gpu_shadow_cp_write(std::uint64_t, std::size_t) {}
std::string host_gpu_shadow_report() { return {}; }
std::string host_gpu_recorder_report() { return {}; }
bool host_gpu_zpass_dump(std::uint64_t) { return false; }
void host_gpu_set_predication(std::uint64_t, unsigned, bool, bool) {}
std::string host_gpu_occlusion_report() { return {}; }
std::string host_gpu_image_heap_report() { return {}; }
std::string host_gpu_ps_wave_report() { return {}; }
std::string host_gpu_memory_budget_report() { return {}; }
void host_gpu_set_loading(bool) {}
void host_gpu_world_reached() {}
std::size_t host_gpu_shader_backlog() { return 0; }
int host_gpu_stream_selftest(int) { return 2; }  // no Vulkan: nothing to test
