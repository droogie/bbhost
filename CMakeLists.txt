cmake_minimum_required(VERSION 3.20)
project(bbhost LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32 AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "bbhost must be built with Clang on Windows so guest SysV functions can be implemented with __attribute__((sysv_abi)).")
endif()

# LibAtrac9 (MIT, https://github.com/Thealexbarney/LibAtrac9): ATRAC9 decoding for sceAjm.
file(GLOB LIBATRAC9_SOURCES third_party/libatrac9/C/src/*.c)
add_library(atrac9 STATIC ${LIBATRAC9_SOURCES})
target_include_directories(atrac9 PUBLIC third_party/libatrac9/C/src)
set_target_properties(atrac9 PROPERTIES C_STANDARD 99 POSITION_INDEPENDENT_CODE ON)
target_compile_options(atrac9 PRIVATE -w)

# Monocypher (BSD-2-Clause or CC0, https://monocypher.org): Ed25519 signature
# checks for the official plugins (host/plugins.cpp, host/updater.cpp).
add_library(monocypher STATIC third_party/monocypher/monocypher.c third_party/monocypher/monocypher-ed25519.c)
target_include_directories(monocypher PUBLIC third_party/monocypher)
set_target_properties(monocypher PROPERTIES C_STANDARD 99 POSITION_INDEPENDENT_CODE ON)

# Sea Islands shader decoder and the offline tools built on it.
find_package(ZLIB REQUIRED)
add_library(gcn STATIC src/gcn/decode.cpp src/gcn/container.cpp src/gcn/spirv.cpp src/gcn/half.cpp src/gcn/translate.cpp src/gcn/lift.cpp src/gcn/image_dims.cpp src/gcn/wave.cpp src/gcn/names.cpp src/gcn/hlsl.cpp src/gcn/gs_vertex.cpp)
target_include_directories(gcn PUBLIC src)
target_link_libraries(gcn PUBLIC ZLIB::ZLIB)
add_executable(gcndis tools/gcndis.cpp)
target_link_libraries(gcndis PRIVATE gcn)
find_library(SPIRV_TOOLS_LIB SPIRV-Tools)
# The headers: the system's on Linux, the dependency prefix's in the Windows
# cross build (tools/win_deps.sh), where the toolchain searches only the
# sysroot and the prefix.
find_path(SPIRV_TOOLS_INCLUDE_DIR spirv-tools/libspirv.hpp)
if(SPIRV_TOOLS_LIB AND NOT SPIRV_TOOLS_INCLUDE_DIR)
  set(SPIRV_TOOLS_LIB "")
endif()
if(SPIRV_TOOLS_LIB)
  include_directories(SYSTEM ${SPIRV_TOOLS_INCLUDE_DIR})
  add_executable(gcn2spv tools/gcn2spv.cpp)
  target_link_libraries(gcn2spv PRIVATE gcn ${SPIRV_TOOLS_LIB})
  # The lifted shaders as HLSL, compiled with DXC (loaded at run time) and checked.
  find_package(Threads REQUIRED)
  add_executable(gcn2hlsl tools/gcn2hlsl.cpp src/host/dxc.cpp)
  target_link_libraries(gcn2hlsl PRIVATE gcn ${SPIRV_TOOLS_LIB} ${CMAKE_DL_LIBS} Threads::Threads)
endif()


add_executable(bbhost
  src/main.cpp
  src/core/memory.cpp
  src/core/write_watch.cpp
  src/core/elf.cpp
  src/core/imports.cpp
  src/core/thunk.cpp
  src/core/portable.cpp
  src/core/paced_file.cpp
  src/core/host_clock.cpp
  src/core/futex.cpp
  src/core/win_vm.cpp
  src/core/win_watch.cpp
  src/host/win_crash.cpp
  src/host/plugins.cpp
  src/core/tls_rewrite.cpp
  src/core/config.cpp
  src/engine/image_text.cpp
  src/core/sha256.cpp
  src/core/sfo.cpp
  src/core/image_file.cpp
  src/host/tess_lds.cpp
  src/host/gs_emit_gate.cpp
  src/engine/guest.cpp
  src/engine/patch_manifest.cpp
  src/engine/gx_resources.cpp
  src/engine/gx_state.cpp
  src/engine/params.cpp
  src/engine/paramdef.cpp
  src/engine/lua_events.cpp
  src/engine/event_flags.cpp
  src/engine/player_data.cpp
  src/engine/world_chr.cpp
  src/engine/playlog.cpp
  src/engine/sos_signs.cpp
  src/engine/change_appearance.cpp
  src/engine/esd.cpp
  src/engine/menu_steps.cpp
  src/engine/rebirth.cpp
  src/engine/rebirth_script.cpp
  src/engine/dream_mirror_layout.cpp
  src/engine/chara_id.cpp
  src/engine/five_players.cpp
  src/engine/maiden_events.cpp
  src/engine/option_menu.cpp
  src/engine/menu_assets.cpp
  src/engine/menu_pointer.cpp
  src/engine/key_prompts.cpp
  src/engine/yebis.cpp
  src/engine/yebis_bind.cpp
  src/engine/graphics_patch.cpp
  src/engine/live_resolution.cpp
  src/engine/loading.cpp
  src/engine/camera.cpp
  src/engine/mouse_camera.cpp
  src/engine/mouse_camera_stubs.cpp
  src/engine/fmod_probe.cpp
  src/engine/fmod_ring.cpp
  src/engine/sf_text_layout.cpp
  src/engine/menu_memory.cpp
  src/engine/frame_rate.cpp
  src/engine/frame_tick.cpp
  src/engine/frame_pool.cpp
  src/engine/sf_heap_probe.cpp
  src/engine/debug_menu.cpp
  src/engine/summon_invite.cpp
  src/engine/np_test.cpp
  src/net/account.cpp
  src/net/http.cpp
  src/net/session.cpp
  src/net/stun.cpp
  src/hle/np_matching2.cpp
  src/hle/np_signaling.cpp
  src/host/window.cpp
  src/host/text_entry.cpp
  src/host/overlay.cpp
  src/host/overlay_font.cpp
  src/host/japanese_font.cpp
  src/host/present_pass.cpp
  src/host/ingame_menu.cpp
  src/host/options.cpp
  src/host/updater.cpp
  src/host/update_verify.cpp
  src/host/bindings.cpp
  src/host/audio.cpp
  src/host/frame_stats.cpp
  src/host/draw_digest.cpp
  src/host/perf_meter.cpp
  src/host/main_wait.cpp
  src/host/sampler.cpp
  src/host/foreign_hooks.cpp
  src/hle/runtime.cpp
  src/hle/libc.cpp
  src/hle/guest_console.cpp
  src/hle/sysv_va.cpp
  src/hle/pthread.cpp
  src/hle/sync.cpp
  src/hle/kernel.cpp
  src/hle/fs.cpp
  src/hle/fios.cpp
  src/hle/dialog.cpp
  src/hle/avplayer.cpp
  src/hle/video.cpp
  src/hle/gnm.cpp
  src/hle/gnm_exec.cpp
  src/host/host_lists.cpp
  src/host/cb_proof.cpp
  src/host/gxvk_census.cpp
  src/host/gxvk_buffers.cpp
  src/host/gxvk_arena.cpp
  src/host/gxvk_heapsnap.cpp
  src/hle/gx_trace.cpp
  src/hle/audio.cpp
  src/hle/net.cpp
  src/hle/http.cpp
  src/hle/ajm.cpp
  src/hle/system.cpp
  src/host/label_landing.cpp
)

target_include_directories(bbhost PRIVATE src include ${CMAKE_BINARY_DIR}/gen)
# The port's menu strings, built in (engine/menu_assets.cpp makes the message
# bundle from them at start): tools/pc_option_messages.tsv as a string literal.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tools/pc_option_messages.tsv)
file(READ ${CMAKE_SOURCE_DIR}/tools/pc_option_messages.tsv BBHOST_PC_MESSAGES)
file(WRITE ${CMAKE_BINARY_DIR}/gen/pc_option_messages.inc "R\"tsv(${BBHOST_PC_MESSAGES})tsv\"\n")
# The Japanese font is compiled in; both UI renderers share the same bytes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/third_party/mplus/mplus-1m-regular.ttf)
file(READ ${CMAKE_SOURCE_DIR}/third_party/mplus/mplus-1m-regular.ttf BB_JP_FONT_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," BB_JP_FONT_BYTES "${BB_JP_FONT_HEX}")
file(WRITE ${CMAKE_BINARY_DIR}/gen/japanese_font.inc.new "${BB_JP_FONT_BYTES}\n")
configure_file(${CMAKE_BINARY_DIR}/gen/japanese_font.inc.new ${CMAKE_BINARY_DIR}/gen/japanese_font.inc COPYONLY)
target_link_libraries(bbhost PRIVATE ZLIB::ZLIB)

# The git revision the binary logs on its first line (build/gen/bbhost_version.h,
# rewritten by every build only when it changes): a crash report from a
# machine names the build to symbolize it against.
add_custom_target(bbhost_version
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR} -DOUT=${CMAKE_BINARY_DIR}/gen/bbhost_version.h
          -P ${CMAKE_SOURCE_DIR}/cmake/version.cmake
  BYPRODUCTS ${CMAKE_BINARY_DIR}/gen/bbhost_version.h)
add_dependencies(bbhost bbhost_version)
# The translator's fingerprint (build/gen/translator_fingerprint.h): the
# translation cache's key for "the same translator" - its sources and the
# cache's own key and entry code.
add_custom_target(bbhost_translator_fingerprint
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR} -DOUT=${CMAKE_BINARY_DIR}/gen/translator_fingerprint.h
          -P ${CMAKE_SOURCE_DIR}/cmake/translator_fingerprint.cmake
  BYPRODUCTS ${CMAKE_BINARY_DIR}/gen/translator_fingerprint.h)
add_dependencies(bbhost bbhost_translator_fingerprint)

# The Windows exe's icon and version details.
if(WIN32)
  enable_language(RC)
  target_sources(bbhost PRIVATE res/bbhost.rc)
  set_source_files_properties(res/bbhost.rc PROPERTIES
    OBJECT_DEPENDS "${CMAKE_SOURCE_DIR}/res/bbhost.ico;${CMAKE_BINARY_DIR}/gen/bbhost_version.h")
endif()

# The example plugin (include/bbhost_plugin.h, plugins/hello): build/plugins/hello.so
# or .dll, which the host next to it loads.
add_library(bb_plugin_hello MODULE plugins/hello/hello.c)
target_include_directories(bb_plugin_hello PRIVATE include)
set_target_properties(bb_plugin_hello PROPERTIES OUTPUT_NAME hello PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins)

# Official plugins (docs/plugins.md): C++ against
# include/bbhost (sdk.hpp), built to build/plugins next to the example. On
# Windows each carries its own C++ runtime, so a DLL needs nothing beside it.
function(bb_plugin name)
  add_library(bb_plugin_${name} MODULE ${ARGN})
  target_include_directories(bb_plugin_${name} PRIVATE include)
  target_compile_features(bb_plugin_${name} PRIVATE cxx_std_17)
  set_target_properties(bb_plugin_${name} PROPERTIES OUTPUT_NAME ${name} PREFIX ""
    CXX_VISIBILITY_PRESET hidden LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins)
  if(WIN32)
    target_link_options(bb_plugin_${name} PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
endfunction()
add_library(bb_plugin_region_switch MODULE plugins/region_switch/region_switch.c)
set_target_properties(bb_plugin_region_switch PROPERTIES OUTPUT_NAME region_switch PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins)

bb_plugin(randomizer plugins/randomizer/randomizer.cpp)
bb_plugin(boss_rush plugins/boss_rush/boss_rush.cpp)
bb_plugin(mutators plugins/mutators/mutators.cpp)
# The Debug Menu plugin carries its glyphs (plugins/debug_menu/font14.bin,
# tools/make_debug_font.py) as bytes: build/gen/debug_font14.inc, rewritten
# only when the file changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/plugins/debug_menu/font14.bin)
file(READ ${CMAKE_SOURCE_DIR}/plugins/debug_menu/font14.bin BB_FONT14_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," BB_FONT14_BYTES "${BB_FONT14_HEX}")
file(WRITE ${CMAKE_BINARY_DIR}/gen/debug_font14.inc.new "${BB_FONT14_BYTES}\n")
configure_file(${CMAKE_BINARY_DIR}/gen/debug_font14.inc.new ${CMAKE_BINARY_DIR}/gen/debug_font14.inc COPYONLY)
bb_plugin(debug_menu plugins/debug_menu/debug_menu.cpp)
target_include_directories(bb_plugin_debug_menu PRIVATE ${CMAKE_BINARY_DIR}/gen)

target_link_libraries(bbhost PRIVATE atrac9 monocypher)
# A release build (the release workflow) looks for updates by default; one
# built from source only when update.check says so.
option(BBHOST_RELEASE "build a release: updates are checked for by default" OFF)
if(BBHOST_RELEASE)
  target_compile_definitions(bbhost PRIVATE BBHOST_RELEASE_BUILD=1)
endif()
if(UNIX)
  target_link_libraries(bbhost PRIVATE pthread dl m)
elseif(WIN32)
  # hle/net.cpp, net/session.cpp; timeBeginPeriod; WaitOnAddress (core/futex.cpp)
  target_link_libraries(bbhost PRIVATE ws2_32 winmm synchronization)
  # The SysV convention our guest-facing functions follow lets a leaf keep
  # data in the 128 bytes below rsp, and on Windows an exception - a
  # write-watch fault in a decomp leaf's write, say - builds its frame right
  # there (decomp/sfx/ribbons.cpp). Clang leaves it out on this target today;
  # the flag makes sure no compiler does.
  target_compile_options(bbhost PRIVATE -mno-red-zone)
  # The image must stay out of the guest's fixed windows (4-10 GiB CPU dmem,
  # 0xff00000000 GPU alias): a fixed base above them, no ASLR.
  target_link_options(bbhost PRIVATE -Wl,--image-base=0x600000000 -Wl,--disable-dynamicbase)
endif()

find_package(CURL QUIET)
if(CURL_FOUND)
  message(STATUS "bbhost: libcurl ${CURL_VERSION_STRING} found; sceHttp is real")
  target_compile_definitions(bbhost PRIVATE BBHOST_HAVE_CURL=1)
  target_link_libraries(bbhost PRIVATE CURL::libcurl)
endif()

find_package(SDL3 CONFIG QUIET)
# Movies (sceAvPlayer stage 2): ffmpeg as an optional LGPL dynamic dependency.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(FFMPEG IMPORTED_TARGET libavcodec libavformat libavutil libswscale libswresample)
endif()
if(FFMPEG_FOUND)
  message(STATUS "bbhost: ffmpeg found; movies decode")
  target_compile_definitions(bbhost PRIVATE BBHOST_HAVE_FFMPEG=1)
  target_link_libraries(bbhost PRIVATE PkgConfig::FFMPEG)
else()
  message(STATUS "bbhost: ffmpeg not found; movies are skipped")
endif()

find_package(Vulkan QUIET)
if(Vulkan_FOUND)
  # Draw capture manifests and raw-texel comparison.
  add_library(replay_common STATIC src/replay/json.cpp src/replay/compare.cpp)
  target_include_directories(replay_common PUBLIC src)
  target_link_libraries(replay_common PUBLIC Vulkan::Headers)
  target_sources(bbhost PRIVATE src/host/gpu.cpp src/host/gpu_busy.cpp src/host/gpu_memtrack.cpp src/host/gpu_mutex.cpp src/host/buffer_shadow.cpp src/host/gxvk_vk.cpp src/host/recorder.cpp src/host/bindless.cpp src/host/render.cpp src/host/textures.cpp src/host/gpu_untile.cpp src/host/depth_copy.cpp src/host/indirect_clamp.cpp src/host/fsr.cpp src/host/glitch.cpp src/host/occlusion.cpp src/host/draw_capture.cpp src/host/stream_selftest.cpp
                                src/host/shader_patch.cpp src/host/translation_cache.cpp src/host/dxc.cpp src/host/shader_bundles.cpp)
  target_link_libraries(bbhost PRIVATE Vulkan::Vulkan gcn replay_common)
  # The SPIR-V validator: the HLSL path validates the modules DXC compiled
  # without its own validator (a sample's run-time offset; host/translation_cache.cpp).
  # Without it those keep the lift.
  if(SPIRV_TOOLS_LIB)
    target_compile_definitions(bbhost PRIVATE BBHOST_HAVE_SPIRV_TOOLS=1)
    target_link_libraries(bbhost PRIVATE ${SPIRV_TOOLS_LIB})
  endif()
  # Which user-data tables a no-fallback shader still reads, from the SPIRV-Tools
  # optimizer; without it every table is built. Not in the Windows build yet,
  # though its prefix has the optimizer: it has not been run there.
  target_sources(bbhost PRIVATE src/host/shader_liveness.cpp)
  find_library(SPIRV_TOOLS_OPT_LIB SPIRV-Tools-opt)
  if(SPIRV_TOOLS_LIB AND SPIRV_TOOLS_OPT_LIB AND NOT WIN32)
    target_compile_definitions(bbhost PRIVATE BBHOST_HAVE_SPIRV_OPT=1)
    target_link_libraries(bbhost PRIVATE ${SPIRV_TOOLS_OPT_LIB} ${SPIRV_TOOLS_LIB})
  endif()
  # gcn2hlsl --sweep: each module built into the renderer's pipelines on a
  # device, for the driver's verdict (tools/pipeline_probe.h).
  if(TARGET gcn2hlsl)
    target_sources(gcn2hlsl PRIVATE tools/pipeline_probe.cpp)
    target_compile_definitions(gcn2hlsl PRIVATE GCN2HLSL_HAVE_VULKAN=1)
    target_link_libraries(gcn2hlsl PRIVATE Vulkan::Vulkan)
  endif()
  # The HLSL path against the lift on random inputs (run it on lavapipe).
  add_executable(hlsl_equiv tools/hlsl_equiv.cpp src/host/dxc.cpp)
  target_include_directories(hlsl_equiv PRIVATE src)
  target_link_libraries(hlsl_equiv PRIVATE gcn Vulkan::Vulkan ${CMAKE_DL_LIBS})
  # Offline replay of BBHOST_CAPTURE_DRAW captures.
  add_executable(drawreplay tools/drawreplay.cpp src/core/sha256.cpp)
  target_include_directories(drawreplay PRIVATE src)
  target_link_libraries(drawreplay PRIVATE replay_common Vulkan::Vulkan)
  if(SPIRV_TOOLS_LIB)
    target_compile_definitions(drawreplay PRIVATE DRAWREPLAY_HAVE_SPIRV_TOOLS=1)
    target_link_libraries(drawreplay PRIVATE ${SPIRV_TOOLS_LIB})
    # Lift a captured draw's pixel shader.
    add_executable(gcnlift tools/gcnlift.cpp)
    target_link_libraries(gcnlift PRIVATE gcn replay_common ${SPIRV_TOOLS_LIB})
  endif()
else()
  target_sources(bbhost PRIVATE src/host/gpu_stub.cpp)
endif()
if(SDL3_FOUND AND Vulkan_FOUND)
  message(STATUS "bbhost: SDL3 ${SDL3_VERSION} + Vulkan found; building the window shell")
  target_compile_definitions(bbhost PRIVATE BBHOST_HAVE_SDL3=1)
  target_link_libraries(bbhost PRIVATE SDL3::SDL3 Vulkan::Vulkan)
  # The setup window (host/launcher.h): Dear ImGui (MIT,
  # https://github.com/ocornut/imgui, third_party/imgui/VERSION) on SDL3's 2D renderer.
  add_library(imgui STATIC third_party/imgui/imgui.cpp third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_sdl3.cpp third_party/imgui/backends/imgui_impl_sdlrenderer3.cpp
    third_party/imgui/backends/imgui_impl_vulkan.cpp)
  target_include_directories(imgui PUBLIC third_party/imgui)
  target_link_libraries(imgui PUBLIC SDL3::SDL3 Vulkan::Vulkan)
  target_compile_options(imgui PRIVATE -w)
  target_sources(bbhost PRIVATE src/host/launcher.cpp src/host/plugin_ui.cpp src/host/shader_compile.cpp)
  target_link_libraries(bbhost PRIVATE imgui)
  set_source_files_properties(src/host/window.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
  set_source_files_properties(src/host/overlay.cpp src/host/present_pass.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
  set_source_files_properties(src/host/ingame_menu.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
  set_source_files_properties(src/host/buffer_shadow.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
  set_source_files_properties(src/host/gpu_untile.cpp src/host/depth_copy.cpp src/host/indirect_clamp.cpp src/host/fsr.cpp src/host/glitch.cpp src/host/occlusion.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
else()
  message(STATUS "bbhost: SDL3/Vulkan not found; headless only")
endif()

# The game's functions as ours (src/decomp, docs/decomp.md), one folder per
# engine area. A leaf runs on the game's thread state, where FS may be the
# game's, so no decomp file reads the stack protector's canary through it; and
# ours reproduces the game's floating point bit for bit, so nothing is fused
# into a multiply-add.
set(BBHOST_DECOMP_SOURCES
  src/decomp/decomp.cpp
  src/decomp/insn.cpp
  src/decomp/areas.cpp
  src/decomp/events/flag_store.cpp
  src/decomp/events/interpreter.cpp
  src/decomp/player/player_data.cpp
  src/decomp/items/item_lots.cpp
  src/decomp/talk/evaluator.cpp
  src/decomp/chalice/ritual.cpp
  src/decomp/camera/follow_camera.cpp
  src/decomp/gx/flush_wait.cpp
  src/decomp/gx/block_reclaim.cpp
  src/decomp/gx/drain_token.cpp
  src/decomp/gx/query_token.cpp
  src/decomp/gx/dead_packets.cpp
  src/decomp/gx/labels.cpp
  src/decomp/gx/dma.cpp
  src/decomp/gx/constant_engine.cpp
  src/decomp/gx/draw.cpp
  src/decomp/gx/command_list.cpp
  src/decomp/gx/present.cpp
  src/decomp/gx/resources.cpp
  src/decomp/gx/water_gate.cpp
  src/decomp/gx/blink.cpp
  src/decomp/gx/upload_heap.cpp
  src/decomp/gx/render_queue.cpp
  src/decomp/gnmx/cue.cpp
  src/decomp/yebis/gnm.cpp
  src/decomp/yebis/context.cpp
  src/decomp/yebis/ppfx.cpp
  src/decomp/yebis/ppfx_setters.cpp
  src/decomp/yebis/clear.cpp
  src/decomp/memory/game_memcpy.cpp
  src/decomp/memory/ez_copy.cpp
  src/decomp/sfx/ribbons.cpp
  src/decomp/sfx/step_gate.cpp
  src/decomp/sfx/node_update.cpp
  src/decomp/sfx/node_emit.cpp
  src/decomp/sync/adaptive_mutex.cpp
  src/decomp/sync/ez_work.cpp
)
target_sources(bbhost PRIVATE ${BBHOST_DECOMP_SOURCES})
set_source_files_properties(${BBHOST_DECOMP_SOURCES} PROPERTIES COMPILE_OPTIONS "-fno-stack-protector;-ffp-contract=off")
# The host lists record from decomp leaves, on the game's threads with the game's FS.
set_source_files_properties(src/host/host_lists.cpp PROPERTIES COMPILE_OPTIONS "-fno-stack-protector")
# The follow camera's step is the game's own AVX, intrinsic for intrinsic
# (every CPU that runs the eboot has AVX).
set_property(SOURCE src/decomp/camera/follow_camera.cpp APPEND PROPERTY COMPILE_OPTIONS -mavx)

if(MSVC)
  target_compile_options(bbhost PRIVATE /W4)
else()
  target_compile_options(bbhost PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

enable_testing()
add_executable(libc_abi_test tests/libc_abi_test.cpp)
target_include_directories(libc_abi_test PRIVATE src)
target_compile_features(libc_abi_test PRIVATE cxx_std_17)
if(WIN32 AND CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
  set_property(TARGET libc_abi_test PROPERTY CROSSCOMPILING_EMULATOR wine)
endif()
add_test(NAME libc_abi_test COMMAND libc_abi_test)
add_executable(libc_stdio_test tests/libc_stdio_test.cpp src/hle/sysv_va.cpp)
target_include_directories(libc_stdio_test PRIVATE src)
target_compile_features(libc_stdio_test PRIVATE cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(libc_stdio_test PRIVATE Threads::Threads)
if(WIN32 AND CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
  set_property(TARGET libc_stdio_test PROPERTY CROSSCOMPILING_EMULATOR wine)
endif()
add_test(NAME libc_stdio_test COMMAND libc_stdio_test)
add_executable(libc_errno_test tests/libc_errno_test.cpp)
target_include_directories(libc_errno_test PRIVATE src)
target_compile_features(libc_errno_test PRIVATE cxx_std_17)
if(WIN32 AND CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
  set_property(TARGET libc_errno_test PROPERTY CROSSCOMPILING_EMULATOR wine)
endif()
add_test(NAME libc_errno_test COMMAND libc_errno_test)
add_executable(guest_console_test tests/guest_console_test.cpp src/hle/guest_console.cpp)
target_include_directories(guest_console_test PRIVATE src)
target_compile_features(guest_console_test PRIVATE cxx_std_17)
if(WIN32 AND CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
  set_property(TARGET guest_console_test PROPERTY CROSSCOMPILING_EMULATOR wine)
endif()
add_test(NAME guest_console_test COMMAND guest_console_test)
add_executable(libc_wide_test tests/libc_wide_test.cpp src/hle/sysv_va.cpp)
target_include_directories(libc_wide_test PRIVATE src)
if(WIN32 AND CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
  set_property(TARGET libc_wide_test PROPERTY CROSSCOMPILING_EMULATOR wine)
endif()
add_test(NAME libc_wide_test COMMAND libc_wide_test)

if(UNIX)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME world_baseline_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/world_baseline_test.py)
    add_test(NAME area_check_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/area_check_test.py)
    add_test(NAME lift_verify_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/lift_verify_test.py)
    add_test(NAME pipeline_list_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/pipeline_list_test.py)
    add_test(NAME full_tour_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/full_tour_test.py)
    # The Debug Menu plugin's English against the 1.09 eboot: skipped without
    # one (BBHOST_EBOOT, or eboot-109-decrypted.bin in the checkout).
    add_test(NAME debug_menu_strings_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/debug_menu_strings.py check)
    set_tests_properties(debug_menu_strings_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # The map-validation probe's tour steps (tools/area_check.py's) on a pretend
  # host. Built beside the tests, never into plugins/, where bbhost would load it.
  add_library(mapval_probe MODULE tools/mapval_plugin.c)
  target_include_directories(mapval_probe PRIVATE include)
  target_link_libraries(mapval_probe PRIVATE m)
  set_target_properties(mapval_probe PROPERTIES PREFIX "")
  add_executable(mapval_probe_test tests/mapval_probe_test.cpp)
  target_include_directories(mapval_probe_test PRIVATE include)
  target_link_libraries(mapval_probe_test PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(mapval_probe_test mapval_probe)
  add_test(NAME mapval_probe_test COMMAND mapval_probe_test $<TARGET_FILE:mapval_probe>)
  set_tests_properties(mapval_probe_test PROPERTIES SKIP_RETURN_CODE 77)
  # The boss rush plugin's travel and fog warp on a pretend host whose loads
  # are slow (plugins/boss_rush): nothing moves the player inside a loading
  # screen, and only a travel that truly fails stops the run.
  add_executable(boss_rush_test tests/boss_rush_test.cpp)
  target_include_directories(boss_rush_test PRIVATE include)
  target_compile_features(boss_rush_test PRIVATE cxx_std_17)
  target_link_libraries(boss_rush_test PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(boss_rush_test bb_plugin_boss_rush)
  add_test(NAME boss_rush_test COMMAND boss_rush_test $<TARGET_FILE:bb_plugin_boss_rush>)
  add_executable(thunk_args tests/thunk_args.cpp src/core/thunk.cpp src/core/tls_rewrite.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  target_include_directories(thunk_args PRIVATE src)
  target_link_libraries(thunk_args PRIVATE pthread)
  add_test(NAME thunk_args COMMAND thunk_args)
  add_executable(updater_test tests/updater_test.cpp src/host/update_verify.cpp src/replay/json.cpp)
  target_include_directories(updater_test PRIVATE src)
  target_link_libraries(updater_test PRIVATE monocypher)
  add_test(NAME updater_test COMMAND updater_test)
  add_executable(sdk_headers_test tests/sdk_headers_test.cpp)
  target_include_directories(sdk_headers_test PRIVATE include)
  target_compile_features(sdk_headers_test PRIVATE cxx_std_17)
  add_test(NAME sdk_headers_test COMMAND sdk_headers_test)
  add_executable(libc_native_test tests/libc_native_test.cpp src/hle/sysv_va.cpp)
  target_include_directories(libc_native_test PRIVATE src)
  target_compile_features(libc_native_test PRIVATE cxx_std_17)
  add_test(NAME libc_native_test COMMAND libc_native_test)
  set_tests_properties(libc_native_test PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(libc_ctype_test tests/libc_ctype_test.cpp)
  target_include_directories(libc_ctype_test PRIVATE src)
  target_compile_features(libc_ctype_test PRIVATE cxx_std_17)
  add_test(NAME libc_ctype_test COMMAND libc_ctype_test)
  add_executable(libc_random_test tests/libc_random_test.cpp)
  target_include_directories(libc_random_test PRIVATE src)
  target_compile_features(libc_random_test PRIVATE cxx_std_17)
  add_test(NAME libc_random_test COMMAND libc_random_test)
  add_executable(sysv_va_test tests/sysv_va_test.cpp src/hle/sysv_va.cpp)
  target_include_directories(sysv_va_test PRIVATE src)
  add_test(NAME sysv_va_test COMMAND sysv_va_test)
  add_executable(stun_test tests/stun_test.cpp src/net/stun.cpp)
  target_include_directories(stun_test PRIVATE src)
  add_test(NAME stun_test COMMAND stun_test)
  # The event poller's long polls and generations (net/poll_pacing.h).
  add_executable(poll_pacing_test tests/poll_pacing_test.cpp)
  target_include_directories(poll_pacing_test PRIVATE src)
  target_link_libraries(poll_pacing_test PRIVATE pthread)
  add_test(NAME poll_pacing_test COMMAND poll_pacing_test)
  set_tests_properties(poll_pacing_test PROPERTIES TIMEOUT 60)  # a turn never handed back hangs its threads
  add_executable(uncapped_limit_test tests/uncapped_limit_test.cpp)
  target_include_directories(uncapped_limit_test PRIVATE src)
  add_test(NAME uncapped_limit_test COMMAND uncapped_limit_test)
  add_executable(sfx_schedule_test tests/sfx_schedule_test.cpp)
  target_include_directories(sfx_schedule_test PRIVATE src)
  add_test(NAME sfx_schedule_test COMMAND sfx_schedule_test)
  add_executable(wind_pace_test tests/wind_pace_test.cpp)
  target_include_directories(wind_pace_test PRIVATE src)
  add_test(NAME wind_pace_test COMMAND wind_pace_test)
  # WAIT_REG_MEM's compare and its can-never-pass rule on synthetic packets
  # (tests/wait_reg_mem_test.cpp).
  add_executable(wait_reg_mem_test tests/wait_reg_mem_test.cpp)
  target_include_directories(wait_reg_mem_test PRIVATE src)
  add_test(NAME wait_reg_mem_test COMMAND wait_reg_mem_test)
  # FMOD's sound-group ring guard without the game (tests/fmod_ring_test.cpp).
  add_executable(fmod_ring_test tests/fmod_ring_test.cpp)
  target_include_directories(fmod_ring_test PRIVATE src)
  add_test(NAME fmod_ring_test COMMAND fmod_ring_test)
  # The guard for a Scaleform text shown before its layout exists, without the
  # game (tests/sf_text_layout_test.cpp).
  add_executable(sf_text_layout_test tests/sf_text_layout_test.cpp src/hle/sync.cpp src/host/main_wait.cpp src/core/futex.cpp)
  target_include_directories(sf_text_layout_test PRIVATE src)
  target_link_libraries(sf_text_layout_test PRIVATE pthread)
  add_test(NAME sf_text_layout_test COMMAND sf_text_layout_test)
  set_tests_properties(sf_text_layout_test PROPERTIES TIMEOUT 60)  # a lock left held hangs its threads
  add_executable(tess_lds_test tests/tess_lds_test.cpp src/host/tess_lds.cpp)
  target_include_directories(tess_lds_test PRIVATE src)
  add_test(NAME tess_lds_test COMMAND tess_lds_test)
  add_executable(debug_font_test tests/debug_font_test.cpp)
  target_include_directories(debug_font_test PRIVATE plugins/debug_menu)
  target_compile_definitions(debug_font_test PRIVATE BB_FONT14_PATH="${CMAKE_SOURCE_DIR}/plugins/debug_menu/font14.bin")
  add_test(NAME debug_font_test COMMAND debug_font_test)
  add_executable(image_file_test tests/image_file_test.cpp src/core/image_file.cpp)
  target_include_directories(image_file_test PRIVATE src)
  target_link_libraries(image_file_test PRIVATE ZLIB::ZLIB)
  add_test(NAME image_file_test COMMAND image_file_test)
  add_executable(config_test tests/config_test.cpp src/core/config.cpp src/core/sfo.cpp)
  target_include_directories(config_test PRIVATE src ${CMAKE_BINARY_DIR}/gen)
  target_compile_definitions(config_test PRIVATE CONFIG_MUTANTS)  # the play-log migration's mutants
  add_dependencies(config_test bbhost_version)
  add_test(NAME config_test COMMAND config_test)
  # The session's online rules: a gameplay patch manifest makes them say modded.
  add_executable(ruleset_test tests/ruleset_test.cpp src/core/config.cpp src/core/sfo.cpp)
  target_include_directories(ruleset_test PRIVATE src ${CMAKE_BINARY_DIR}/gen)
  target_compile_definitions(ruleset_test PRIVATE BB_PATCHES_DIR="${CMAKE_SOURCE_DIR}/patches")
  add_dependencies(ruleset_test bbhost_version)
  add_test(NAME ruleset_test COMMAND ruleset_test)
  # The rings in video memory the CPU writes only within the heap's budget.
  add_executable(ring_budget_test tests/ring_budget_test.cpp)
  target_include_directories(ring_budget_test PRIVATE src)
  add_test(NAME ring_budget_test COMMAND ring_budget_test)
  # The flip keeper replayed against recorded flip timelines (hle/flip_pacing.h).
  # The bottleneck meter's verdict (host/bottleneck.h).
  add_executable(bottleneck_test tests/bottleneck_test.cpp)
  target_include_directories(bottleneck_test PRIVATE src)
  add_test(NAME bottleneck_test COMMAND bottleneck_test)
  add_executable(flip_pacing_test tests/flip_pacing_test.cpp)
  target_include_directories(flip_pacing_test PRIVATE src)
  add_test(NAME flip_pacing_test COMMAND flip_pacing_test)
  # The vertex clamp's lookup, only where its result is used (host/vertex_clamp.h).
  add_executable(vertex_clamp_test tests/vertex_clamp_test.cpp)
  target_include_directories(vertex_clamp_test PRIVATE src)
  add_test(NAME vertex_clamp_test COMMAND vertex_clamp_test)
  # The copy watch: its granules, then its ranges searched in halves (host/gxvk_watch.h).
  add_executable(gxvk_arena_test tests/gxvk_arena_test.cpp src/host/gxvk_arena.cpp)
  target_include_directories(gxvk_arena_test PRIVATE src)
  target_compile_definitions(gxvk_arena_test PRIVATE GXVK_ARENA_TESTING GXVK_ARENA_MUTANTS)
  add_test(NAME gxvk_arena_test COMMAND gxvk_arena_test)
  # Stage 1d.1: the heap's constant buffers with the lookups' answers known - the choice and the writers' guards, with mutants.
  add_executable(gxvk_heapsnap_test tests/gxvk_heapsnap_test.cpp src/host/gxvk_heapsnap.cpp)
  target_include_directories(gxvk_heapsnap_test PRIVATE src)
  target_compile_definitions(gxvk_heapsnap_test PRIVATE GXVK_HEAPSNAP_TESTING GXVK_HEAPSNAP_MUTANTS)
  add_test(NAME gxvk_heapsnap_test COMMAND gxvk_heapsnap_test)
  add_executable(gxvk_watch_test tests/gxvk_watch_test.cpp)
  target_include_directories(gxvk_watch_test PRIVATE src)
  target_link_libraries(gxvk_watch_test PRIVATE Threads::Threads)
  add_test(NAME gxvk_watch_test COMMAND gxvk_watch_test)
  # BBHOST_GXVK_BUFFERS's modes, unset and given, by the device's kind.
  add_executable(gxvk_buffers_mode_test tests/gxvk_buffers_mode_test.cpp)
  target_include_directories(gxvk_buffers_mode_test PRIVATE src)
  add_test(NAME gxvk_buffers_mode_test COMMAND gxvk_buffers_mode_test)
  # The buffer shadow's page copies grow a run only where both ends meet.
  add_executable(copy_runs_test tests/copy_runs_test.cpp)
  target_include_directories(copy_runs_test PRIVATE src)
  add_test(NAME copy_runs_test COMMAND copy_runs_test)
  # The buffer shadow's choices for pages an integrated GPU reads in place.
  add_executable(shadow_place_test tests/shadow_place_test.cpp)
  target_include_directories(shadow_place_test PRIVATE src)
  add_test(NAME shadow_place_test COMMAND shadow_place_test)
  # Which GX buffers the import thread imports as GX makes them (window imports).
  add_executable(import_prefetch_test tests/import_prefetch_test.cpp)
  target_include_directories(import_prefetch_test PRIVATE src)
  add_test(NAME import_prefetch_test COMMAND import_prefetch_test)
  # Whole spans or windows: the environment, the config, then the device and the platform.
  add_executable(import_mode_test tests/import_mode_test.cpp)
  target_include_directories(import_mode_test PRIVATE src)
  add_test(NAME import_mode_test COMMAND import_mode_test)
  # What an integrated GPU gives back: the switches, the caches' budget, the drain's thresholds, the spare blocks.
  add_executable(apu_memory_test tests/apu_memory_test.cpp)
  target_include_directories(apu_memory_test PRIVATE src)
  add_test(NAME apu_memory_test COMMAND apu_memory_test)
  # What serves a draw's range in no guest-memory import: the mirrors, a copy, or nothing and why.
  add_executable(import_fallback_test tests/import_fallback_test.cpp)
  target_include_directories(import_fallback_test PRIVATE src)
  add_test(NAME import_fallback_test COMMAND import_fallback_test)
  # The image heap's placement in its blocks and its compaction's policy: the block chosen, the room kept, the pace, the stop.
  add_executable(heap_compact_test tests/heap_compact_test.cpp)
  target_include_directories(heap_compact_test PRIVATE src)
  add_test(NAME heap_compact_test COMMAND heap_compact_test)
  # Where a plugin's read() reads: the image by Binary Ninja address, the rest at runtime - the slid Windows layout too.
  add_executable(plugin_address_test tests/plugin_address_test.cpp)
  target_include_directories(plugin_address_test PRIVATE src)
  add_test(NAME plugin_address_test COMMAND plugin_address_test)
  add_executable(text_entry_test tests/text_entry_test.cpp src/host/text_entry.cpp)
  target_include_directories(text_entry_test PRIVATE src)
  add_test(NAME text_entry_test COMMAND text_entry_test)
  # The settings' publication and two of its listeners: applied on a change,
  # never asked for per frame (host/settings_store.h).
  add_executable(settings_store_test tests/settings_store_test.cpp src/host/shader_patch.cpp)
  target_include_directories(settings_store_test PRIVATE src)
  target_link_libraries(settings_store_test PRIVATE Threads::Threads)
  add_test(NAME settings_store_test COMMAND settings_store_test)
  add_executable(camera_settings_test tests/camera_settings_test.cpp src/engine/camera.cpp)
  target_include_directories(camera_settings_test PRIVATE src)
  target_link_libraries(camera_settings_test PRIVATE Threads::Threads)
  add_test(NAME camera_settings_test COMMAND camera_settings_test)
  set_tests_properties(settings_store_test camera_settings_test PROPERTIES TIMEOUT 30)
  add_executable(engine_layout tests/engine_layout.cpp)
  target_include_directories(engine_layout PRIVATE src)
  add_test(NAME engine_layout COMMAND engine_layout)
  add_executable(write_watch_test tests/write_watch_test.cpp src/core/write_watch.cpp)
  target_include_directories(write_watch_test PRIVATE src)
  target_link_libraries(write_watch_test PRIVATE pthread)
  add_test(NAME write_watch_test COMMAND write_watch_test)
  add_executable(paced_file_test tests/paced_file_test.cpp src/core/paced_file.cpp)
  target_include_directories(paced_file_test PRIVATE src)
  add_test(NAME paced_file_test COMMAND paced_file_test)
  # The decomp's tests (tests/decomp, docs/decomp.md "How one is checked"):
  # the eboot kit loads the 1.09 eboot with bbhost's own loader, every import a
  # trap, and a test runs the game's functions beside ours. They skip (77)
  # without the eboot.
  add_library(eboot_kit STATIC tests/decomp/eboot_kit.cpp src/core/elf.cpp src/core/memory.cpp src/core/sha256.cpp
              src/core/tls_rewrite.cpp)
  target_include_directories(eboot_kit PUBLIC src include tests tests/decomp)
  target_compile_definitions(eboot_kit PRIVATE BBHOST_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  # decomp_<name>: tests/decomp/<name>_test.cpp, with SOURCES built beside it,
  # LIBS linked, and ARGS for its run.
  function(bbhost_decomp_test name)
    cmake_parse_arguments(T "" "" "SOURCES;LIBS;ARGS" ${ARGN})
    add_executable(decomp_${name} tests/decomp/${name}_test.cpp ${T_SOURCES})
    target_compile_options(decomp_${name} PRIVATE -fno-stack-protector -ffp-contract=off)
    target_link_libraries(decomp_${name} PRIVATE eboot_kit pthread ${T_LIBS})
    add_test(NAME decomp_${name} COMMAND decomp_${name} ${T_ARGS})
    set_tests_properties(decomp_${name} PROPERTIES SKIP_RETURN_CODE 77)
  endfunction()
  bbhost_decomp_test(insn SOURCES src/decomp/insn.cpp)
  bbhost_decomp_test(insn_eboot SOURCES src/decomp/insn.cpp)
  bbhost_decomp_test(event_interpreter SOURCES src/decomp/events/flag_store.cpp)
  bbhost_decomp_test(player_data)
  bbhost_decomp_test(item_lots SOURCES src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(talk_evaluator SOURCES src/engine/esd.cpp LIBS gcn)
  bbhost_decomp_test(chalice_ritual)
  bbhost_decomp_test(follow_camera SOURCES src/decomp/camera/follow_camera.cpp src/engine/mouse_camera_stubs.cpp ARGS 2000)
  bbhost_decomp_test(sfx_ribbons SOURCES src/core/write_watch.cpp)
  bbhost_decomp_test(sfx_step_gate SOURCES src/decomp/sfx/node_update.cpp src/decomp/sfx/node_emit.cpp)
  bbhost_decomp_test(sfx_node_update SOURCES src/decomp/sfx/node_emit.cpp)
  bbhost_decomp_test(sfx_node_emit)
  bbhost_decomp_test(sfx_emission_clock SOURCES src/decomp/sfx/node_emit.cpp)
  bbhost_decomp_test(sfx_classes)
  bbhost_decomp_test(adaptive_mutex)
  bbhost_decomp_test(ez_work)
  bbhost_decomp_test(ez_node_cache)
  bbhost_decomp_test(gx_drain_token SOURCES src/host/host_lists.cpp src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(gx_dead_packets SOURCES src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(gx_labels SOURCES src/host/host_lists.cpp src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(gx_dma SOURCES src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(gx_constant_engine SOURCES src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(gx_draw SOURCES src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(gx_command_list)
  bbhost_decomp_test(gx_present)
  bbhost_decomp_test(gx_resources SOURCES src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(gx_upload_heap)
  bbhost_decomp_test(gx_water_gate)
  bbhost_decomp_test(gx_blink)
  bbhost_decomp_test(gx_render_queue SOURCES src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(yebis_gnm SOURCES src/decomp/gx/draw.cpp src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(yebis_context SOURCES src/host/host_lists.cpp src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(yebis_ppfx)
  bbhost_decomp_test(yebis_ppfx_setters)
  bbhost_decomp_test(gnmx_cue)
  bbhost_decomp_test(gnmx_cue_ccb SOURCES src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  bbhost_decomp_test(yebis_clear SOURCES src/host/host_lists.cpp src/core/thunk.cpp src/core/portable.cpp src/hle/sysv_va.cpp src/host/main_wait.cpp)
  # The host command lists without the game (tests/host_lists_test.cpp), in shadow mode.
  add_executable(host_lists_test tests/host_lists_test.cpp src/core/portable.cpp)
  target_include_directories(host_lists_test PRIVATE src include)
  target_compile_options(host_lists_test PRIVATE -fno-stack-protector)
  target_link_libraries(host_lists_test PRIVATE pthread)
  add_test(NAME host_lists_test COMMAND host_lists_test)
  set_tests_properties(host_lists_test PROPERTIES ENVIRONMENT "BBHOST_HOST_LISTS=2")
  # The decomp's install report in parts a log line holds (tests/decomp_lines_test.cpp).
  add_executable(decomp_lines_test tests/decomp_lines_test.cpp)
  target_include_directories(decomp_lines_test PRIVATE src)
  add_test(NAME decomp_lines_test COMMAND decomp_lines_test)
  # The blend state of a colour target a draw writes no channel of (tests/masked_blend_test.cpp).
  add_executable(masked_blend_test tests/masked_blend_test.cpp)
  target_include_directories(masked_blend_test PRIVATE src)
  add_test(NAME masked_blend_test COMMAND masked_blend_test)
  # Which draws start a command buffer of their own on lavapipe (tests/lvp_split_test.cpp).
  add_executable(lvp_split_test tests/lvp_split_test.cpp)
  target_include_directories(lvp_split_test PRIVATE src)
  add_test(NAME lvp_split_test COMMAND lvp_split_test)
  # PA_CL_CLIP_CNTL as depth clamp and rasterizer discard (tests/clip_control_test.cpp).
  add_executable(clip_control_test tests/clip_control_test.cpp)
  target_include_directories(clip_control_test PRIVATE src)
  add_test(NAME clip_control_test COMMAND clip_control_test)
  # The constant-buffer proof's copies and compares without the game (tests/cb_proof_test.cpp).
  add_executable(cb_proof_test tests/cb_proof_test.cpp)
  target_include_directories(cb_proof_test PRIVATE src include)
  target_link_libraries(cb_proof_test PRIVATE pthread)
  add_test(NAME cb_proof_test COMMAND cb_proof_test)
  set_tests_properties(cb_proof_test PROPERTIES ENVIRONMENT "BBHOST_CB_RECORD_PROOF=1")
  # The twins' miss filter against the map walks it stands in for, without the GPU (tests/twin_filter_test.cpp;
  # twin_filter_test --bench times the miss path both ways).
  add_executable(twin_filter_test tests/twin_filter_test.cpp)
  target_include_directories(twin_filter_test PRIVATE src)
  add_test(NAME twin_filter_test COMMAND twin_filter_test)
  # A pipeline recipe recorded where draws pack their vertex elements, replayed where they do not
  # (host/vertex_plan.h; tests/vertex_plan_test.cpp).
  if(TARGET Vulkan::Headers)
    add_executable(vertex_plan_test tests/vertex_plan_test.cpp)
    target_include_directories(vertex_plan_test PRIVATE src)
    target_link_libraries(vertex_plan_test PRIVATE Vulkan::Headers)
    add_test(NAME vertex_plan_test COMMAND vertex_plan_test)
    # The device memory tracker's owners and heap classes, with the driver's
    # allocation calls faked (host/gpu_memtrack.h; tests/gpu_memtrack_test.cpp).
    add_executable(gpu_memtrack_test tests/gpu_memtrack_test.cpp src/host/gpu_memtrack.cpp)
    target_include_directories(gpu_memtrack_test PRIVATE src)
    target_link_libraries(gpu_memtrack_test PRIVATE Vulkan::Headers)
    add_test(NAME gpu_memtrack_test COMMAND gpu_memtrack_test)
  endif()
  # The engine's classes by the names they register (tools/class_catalog.cpp).
  add_executable(class_catalog tools/class_catalog.cpp src/decomp/insn.cpp)
  target_link_libraries(class_catalog PRIVATE eboot_kit)
  add_executable(sync_test tests/sync_test.cpp src/hle/sync.cpp src/host/main_wait.cpp src/core/futex.cpp)
  target_include_directories(sync_test PRIVATE src)
  target_link_libraries(sync_test PRIVATE pthread)
  add_test(NAME sync_test COMMAND sync_test)
  add_executable(at9_test tests/at9_test.cpp src/hle/ajm.cpp src/host/main_wait.cpp)
  target_include_directories(at9_test PRIVATE src third_party/libatrac9/C/src)
  target_link_libraries(at9_test PRIVATE atrac9 pthread)
  add_test(NAME at9_test COMMAND at9_test)
  set_tests_properties(at9_test PROPERTIES SKIP_RETURN_CODE 77)
  # Labels written by the host once their submission has run, a counter for the GPU (tests/label_landing_test.cpp).
  add_executable(label_landing_test tests/label_landing_test.cpp src/core/portable.cpp)
  target_include_directories(label_landing_test PRIVATE src include)
  target_link_libraries(label_landing_test PRIVATE pthread)
  add_test(NAME label_landing_test COMMAND label_landing_test)
  set_tests_properties(label_landing_test PROPERTIES ENVIRONMENT "BBHOST_LABEL_LANDING=1")
  if(SDL3_FOUND)
    # Audio output lifetimes under concurrent writes, on SDL's dummy driver.
    add_executable(host_audio_test tests/host_audio_test.cpp src/host/audio.cpp src/core/portable.cpp)
    target_include_directories(host_audio_test PRIVATE src)
    target_compile_definitions(host_audio_test PRIVATE BBHOST_HAVE_SDL3=1)
    target_link_libraries(host_audio_test PRIVATE SDL3::SDL3 pthread)
    add_test(NAME host_audio_test COMMAND host_audio_test)
    set_tests_properties(host_audio_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  endif()
  if(Vulkan_FOUND)
    add_executable(replay_compare_test tests/replay_compare_test.cpp)
    target_link_libraries(replay_compare_test PRIVATE replay_common)
    add_test(NAME replay_compare_test COMMAND replay_compare_test)
  endif()
  add_executable(dream_mirror_layout_test tests/dream_mirror_layout_test.cpp src/engine/dream_mirror_layout.cpp src/core/sha256.cpp)
  target_include_directories(dream_mirror_layout_test PRIVATE src)
  target_compile_features(dream_mirror_layout_test PRIVATE cxx_std_17)
  target_link_libraries(dream_mirror_layout_test PRIVATE gcn)
  add_test(NAME dream_mirror_layout_test COMMAND dream_mirror_layout_test)
  set_tests_properties(dream_mirror_layout_test PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(esd_test tests/esd_test.cpp src/engine/esd.cpp)
  target_include_directories(esd_test PRIVATE src)
  target_compile_features(esd_test PRIVATE cxx_std_17)
  target_link_libraries(esd_test PRIVATE gcn)
  add_test(NAME esd_test COMMAND esd_test)
  set_tests_properties(esd_test PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(rebirth_script_test tests/rebirth_script_test.cpp src/engine/rebirth_script.cpp src/engine/esd.cpp src/core/sha256.cpp)
  target_include_directories(rebirth_script_test PRIVATE src)
  target_compile_features(rebirth_script_test PRIVATE cxx_std_17)
  target_link_libraries(rebirth_script_test PRIVATE gcn)
  add_test(NAME rebirth_script_test COMMAND rebirth_script_test)
  set_tests_properties(rebirth_script_test PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(maiden_events_test tests/maiden_events_test.cpp src/engine/maiden_events.cpp src/core/sha256.cpp)
  target_include_directories(maiden_events_test PRIVATE src)
  target_compile_features(maiden_events_test PRIVATE cxx_std_17)
  target_link_libraries(maiden_events_test PRIVATE gcn)
  add_test(NAME maiden_events_test COMMAND maiden_events_test)
  set_tests_properties(maiden_events_test PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(randomizer_files_test tests/randomizer_files_test.cpp)
  target_include_directories(randomizer_files_test PRIVATE include src)
  target_compile_features(randomizer_files_test PRIVATE cxx_std_17)
  target_link_libraries(randomizer_files_test PRIVATE gcn)
  add_test(NAME randomizer_files_test COMMAND randomizer_files_test)
  set_tests_properties(randomizer_files_test PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(gcn_decode_test tests/gcn_decode_test.cpp)
  target_link_libraries(gcn_decode_test PRIVATE gcn)
  add_test(NAME gcn_decode_test COMMAND gcn_decode_test)
  set_tests_properties(gcn_decode_test PROPERTIES SKIP_RETURN_CODE 77)
  # The translation cache (src/host/translation_cache.cpp, which it includes)
  # against the shipped shader bundles.
  add_executable(translation_cache_test tests/translation_cache_test.cpp src/core/paced_file.cpp src/host/dxc.cpp)
  target_include_directories(translation_cache_test PRIVATE src ${CMAKE_BINARY_DIR}/gen)
  target_link_libraries(translation_cache_test PRIVATE gcn ZLIB::ZLIB ${CMAKE_DL_LIBS})
  add_dependencies(translation_cache_test bbhost_translator_fingerprint)
  add_test(NAME translation_cache_test COMMAND translation_cache_test)
  set_tests_properties(translation_cache_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  # The HLSL default (translation_cache.cpp, which it includes): compiles
  # behind the game, their lanes, the dump and edited HLSL; synthetic shaders,
  # skipped without DXC.
  add_executable(hlsl_default_test tests/hlsl_default_test.cpp src/core/paced_file.cpp src/host/dxc.cpp)
  target_include_directories(hlsl_default_test PRIVATE src ${CMAKE_BINARY_DIR}/gen)
  target_link_libraries(hlsl_default_test PRIVATE gcn ZLIB::ZLIB ${CMAKE_DL_LIBS})
  add_dependencies(hlsl_default_test bbhost_translator_fingerprint)
  add_test(NAME hlsl_default_test COMMAND hlsl_default_test)
  set_tests_properties(hlsl_default_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  # The PC options' generated menu assets (engine/menu_assets.cpp, which it
  # includes) against the game's option movie and message bundles.
  add_executable(menu_assets_test tests/menu_assets_test.cpp)
  target_include_directories(menu_assets_test PRIVATE src ${CMAKE_BINARY_DIR}/gen)
  target_link_libraries(menu_assets_test PRIVATE ZLIB::ZLIB)
  add_test(NAME menu_assets_test COMMAND menu_assets_test)
  set_tests_properties(menu_assets_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  # The setup window's "Compile shaders now" recommendation (host/shader_compile.h).
  add_executable(shader_compile_advice_test tests/shader_compile_advice_test.cpp)
  target_include_directories(shader_compile_advice_test PRIVATE src)
  add_test(NAME shader_compile_advice_test COMMAND shader_compile_advice_test)
  add_executable(gcn_wave_test tests/gcn_wave_test.cpp)
  target_link_libraries(gcn_wave_test PRIVATE gcn)
  add_test(NAME gcn_wave_test COMMAND gcn_wave_test)
  add_executable(gcn_bindings_test tests/gcn_bindings_test.cpp)
  target_link_libraries(gcn_bindings_test PRIVATE gcn)
  add_test(NAME gcn_bindings_test COMMAND gcn_bindings_test)
  set_tests_properties(gcn_bindings_test PROPERTIES SKIP_RETURN_CODE 77)
  if(SPIRV_TOOLS_LIB)
    add_executable(gcn_translate_test tests/gcn_translate_test.cpp)
    target_link_libraries(gcn_translate_test PRIVATE gcn ${SPIRV_TOOLS_LIB})
    add_test(NAME gcn_translate_test COMMAND gcn_translate_test)
  set_tests_properties(gcn_translate_test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(gcn_lds_window_test tests/gcn_lds_window_test.cpp)
    target_link_libraries(gcn_lds_window_test PRIVATE gcn ${SPIRV_TOOLS_LIB})
    add_test(NAME gcn_lds_window_test COMMAND gcn_lds_window_test)
    add_executable(gcn_lift_test tests/gcn_lift_test.cpp)
    target_link_libraries(gcn_lift_test PRIVATE gcn ${SPIRV_TOOLS_LIB})
    add_test(NAME gcn_lift_test COMMAND gcn_lift_test)
  set_tests_properties(gcn_lift_test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(gcn_tess_lift_test tests/gcn_tess_lift_test.cpp)
    target_link_libraries(gcn_tess_lift_test PRIVATE gcn ${SPIRV_TOOLS_LIB})
    add_test(NAME gcn_tess_lift_test COMMAND gcn_tess_lift_test)
    set_tests_properties(gcn_tess_lift_test PROPERTIES SKIP_RETURN_CODE 77)
    # The geometry-shader lifts (ES and GS) on the effects bundle's particles, through HLSL and DXC where installed.
    add_executable(gcn_gs_lift_test tests/gcn_gs_lift_test.cpp src/host/dxc.cpp src/host/gs_emit_gate.cpp)
    target_link_libraries(gcn_gs_lift_test PRIVATE gcn ${SPIRV_TOOLS_LIB} ${CMAKE_DL_LIBS})
    add_test(NAME gcn_gs_lift_test COMMAND gcn_gs_lift_test)
    set_tests_properties(gcn_gs_lift_test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(gcn_names_test tests/gcn_names_test.cpp)
    target_link_libraries(gcn_names_test PRIVATE gcn ${SPIRV_TOOLS_LIB})
    add_test(NAME gcn_names_test COMMAND gcn_names_test)
    set_tests_properties(gcn_names_test PROPERTIES SKIP_RETURN_CODE 77)
    # The HLSL printer on synthetic lifts; compiled by DXC where it is installed.
    add_executable(gcn_hlsl_test tests/gcn_hlsl_test.cpp src/host/dxc.cpp)
    target_link_libraries(gcn_hlsl_test PRIVATE gcn ${SPIRV_TOOLS_LIB} ${CMAKE_DL_LIBS})
    add_test(NAME gcn_hlsl_test COMMAND gcn_hlsl_test)
    if(Vulkan_FOUND)
      add_executable(gcn_sampling_test tests/gcn_sampling_test.cpp)
      target_link_libraries(gcn_sampling_test PRIVATE gcn ${SPIRV_TOOLS_LIB} Vulkan::Vulkan)
      add_test(NAME gcn_sampling_test COMMAND gcn_sampling_test)
      set_tests_properties(gcn_sampling_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
      add_executable(indirect_clamp_test tests/indirect_clamp_test.cpp)
      target_include_directories(indirect_clamp_test PRIVATE src)
      target_link_libraries(indirect_clamp_test PRIVATE Vulkan::Vulkan)
      add_test(NAME indirect_clamp_test COMMAND indirect_clamp_test)
      set_tests_properties(indirect_clamp_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    endif()
  endif()
endif()

# `cmake --build build --target smoke`: 60 s run must time out (124) without a core dump.
# For the Windows cross build (`cmake --build build-win --target smoke`) the
# run is bbhost.exe under wine, ended by BBHOST_EXIT_FLIP (tools/win_smoke.sh).
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
  add_custom_target(smoke
    COMMAND ${CMAKE_SOURCE_DIR}/tools/win_smoke.sh $<TARGET_FILE:bbhost> ${CMAKE_BINARY_DIR}/smoke-run.log ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS bbhost
    USES_TERMINAL)
  # `cmake --build build-win --target package-win`: the zip a Windows machine
  # runs from (tools/win_package.sh; docs/building.md).
  add_custom_target(package-win
    COMMAND ${CMAKE_SOURCE_DIR}/tools/win_package.sh $<TARGET_FILE:bbhost> ${CMAKE_SOURCE_DIR}/build/win
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    # The official plugins too: the package carries them, and a CI build makes
    # nothing it is not asked for (v0.2.0's Windows zip shipped without them).
    DEPENDS bbhost bb_plugin_region_switch bb_plugin_randomizer bb_plugin_boss_rush bb_plugin_mutators bb_plugin_debug_menu
    USES_TERMINAL)
else()
  add_custom_target(smoke
    COMMAND ${CMAKE_SOURCE_DIR}/tools/smoke.sh $<TARGET_FILE:bbhost> ${CMAKE_BINARY_DIR}/smoke-run.log ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS bbhost
    USES_TERMINAL)
endif()

# Japanese fork: glyph rendering and JP/EU request policy, without game data.
add_executable(japanese_font_test tests/japanese_font_test.cpp src/host/overlay_font.cpp src/host/japanese_font.cpp)
target_include_directories(japanese_font_test PRIVATE src ${CMAKE_BINARY_DIR}/gen)
add_test(NAME japanese_font_test COMMAND japanese_font_test)
add_executable(jp_eu_http_test tests/jp_eu_http_test.cpp src/core/futex.cpp src/core/portable.cpp)
target_include_directories(jp_eu_http_test PRIVATE src include ${CMAKE_BINARY_DIR}/gen)
add_dependencies(jp_eu_http_test bbhost_version)
# Only the request construction/policy code is exercised; discard the networking
# entry points so this test cannot accidentally contact a real server.
target_compile_options(jp_eu_http_test PRIVATE -ffunction-sections -fdata-sections)
target_link_options(jp_eu_http_test PRIVATE -Wl,--gc-sections)
if(WIN32)
  target_link_libraries(jp_eu_http_test PRIVATE ws2_32 winmm synchronization)
endif()
add_test(NAME jp_eu_http_test COMMAND jp_eu_http_test)
if(WIN32 OR UNIX)
  add_executable(jp_region_test tests/jp_region_test.cpp)
  target_include_directories(jp_region_test PRIVATE src include)
  target_link_libraries(jp_region_test PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(jp_region_test bb_plugin_region_switch)
  foreach(case success off wrong-title wrong-region wrong-eboot)
    add_test(NAME jp_region_${case} COMMAND jp_region_test ${case})
    set_tests_properties(jp_region_${case} PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/plugins)
  endforeach()
endif()
