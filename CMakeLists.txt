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
add_library(gcn STATIC src/gcn/decode.cpp src/gcn/container.cpp src/gcn/spirv.cpp src/gcn/half.cpp src/gcn/translate.cpp src/gcn/lift.cpp src/gcn/image_dims.cpp src/gcn/wave.cpp)
target_include_directories(gcn PUBLIC src)
target_link_libraries(gcn PUBLIC ZLIB::ZLIB)
add_executable(gcndis tools/gcndis.cpp)
target_link_libraries(gcndis PRIVATE gcn)
find_library(SPIRV_TOOLS_LIB SPIRV-Tools)
if(SPIRV_TOOLS_LIB)
  add_executable(gcn2spv tools/gcn2spv.cpp)
  target_link_libraries(gcn2spv PRIVATE gcn ${SPIRV_TOOLS_LIB})
endif()


add_executable(bbhost
  src/main.cpp
  src/core/memory.cpp
  src/core/write_watch.cpp
  src/core/elf.cpp
  src/core/imports.cpp
  src/core/thunk.cpp
  src/core/portable.cpp
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
  src/engine/menu_memory.cpp
  src/engine/frame_rate.cpp
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
  src/host/present_pass.cpp
  src/host/ingame_menu.cpp
  src/host/options.cpp
  src/host/updater.cpp
  src/host/update_verify.cpp
  src/host/bindings.cpp
  src/host/audio.cpp
  src/host/frame_stats.cpp
  src/host/main_wait.cpp
  src/host/sampler.cpp
  src/host/foreign_hooks.cpp
  src/hle/runtime.cpp
  src/hle/libc.cpp
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
  src/hle/gx_trace.cpp
  src/hle/audio.cpp
  src/hle/net.cpp
  src/hle/http.cpp
  src/hle/ajm.cpp
  src/hle/system.cpp
)

target_include_directories(bbhost PRIVATE src include ${CMAKE_BINARY_DIR}/gen)
# The port's menu strings, built in (engine/menu_assets.cpp makes the message
# bundle from them at start): tools/pc_option_messages.tsv as a string literal.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tools/pc_option_messages.tsv)
file(READ ${CMAKE_SOURCE_DIR}/tools/pc_option_messages.tsv BBHOST_PC_MESSAGES)
file(WRITE ${CMAKE_BINARY_DIR}/gen/pc_option_messages.inc "R\"tsv(${BBHOST_PC_MESSAGES})tsv\"\n")
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
# ... and its English (plugins/debug_menu/strings_en.tsv) as a string.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/plugins/debug_menu/strings_en.tsv)
file(READ ${CMAKE_SOURCE_DIR}/plugins/debug_menu/strings_en.tsv BB_DEBUG_MENU_EN)
file(WRITE ${CMAKE_BINARY_DIR}/gen/debug_menu_strings_en.inc.new "R\"tsv(${BB_DEBUG_MENU_EN})tsv\"\n")
configure_file(${CMAKE_BINARY_DIR}/gen/debug_menu_strings_en.inc.new ${CMAKE_BINARY_DIR}/gen/debug_menu_strings_en.inc COPYONLY)
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
  target_sources(bbhost PRIVATE src/host/gpu.cpp src/host/gpu_busy.cpp src/host/gpu_memtrack.cpp src/host/gpu_mutex.cpp src/host/buffer_shadow.cpp src/host/recorder.cpp src/host/bindless.cpp src/host/render.cpp src/host/textures.cpp src/host/gpu_untile.cpp src/host/depth_copy.cpp src/host/fsr.cpp src/host/glitch.cpp src/host/occlusion.cpp src/host/draw_capture.cpp src/host/stream_selftest.cpp
                                src/host/shader_patch.cpp src/host/translation_cache.cpp)
  target_link_libraries(bbhost PRIVATE Vulkan::Vulkan gcn replay_common)
  # Which user-data tables a no-fallback shader still reads, from the SPIRV-Tools
  # optimizer; without it every table is built.
  target_sources(bbhost PRIVATE src/host/shader_liveness.cpp)
  find_library(SPIRV_TOOLS_OPT_LIB SPIRV-Tools-opt)
  if(SPIRV_TOOLS_LIB AND SPIRV_TOOLS_OPT_LIB)
    target_compile_definitions(bbhost PRIVATE BBHOST_HAVE_SPIRV_OPT=1)
    target_link_libraries(bbhost PRIVATE ${SPIRV_TOOLS_OPT_LIB} ${SPIRV_TOOLS_LIB})
  endif()
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
  target_sources(bbhost PRIVATE src/host/launcher.cpp src/host/plugin_ui.cpp)
  target_link_libraries(bbhost PRIVATE imgui)
  set_source_files_properties(src/host/window.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
  set_source_files_properties(src/host/overlay.cpp src/host/present_pass.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
  set_source_files_properties(src/host/ingame_menu.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
  set_source_files_properties(src/host/buffer_shadow.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
  set_source_files_properties(src/host/gpu_untile.cpp src/host/depth_copy.cpp src/host/fsr.cpp src/host/glitch.cpp src/host/occlusion.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
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
  src/decomp/memory/game_memcpy.cpp
  src/decomp/memory/ez_copy.cpp
  src/decomp/sfx/ribbons.cpp
)
target_sources(bbhost PRIVATE ${BBHOST_DECOMP_SOURCES})
set_source_files_properties(${BBHOST_DECOMP_SOURCES} PROPERTIES COMPILE_OPTIONS "-fno-stack-protector;-ffp-contract=off")
# The follow camera's step is the game's own AVX, intrinsic for intrinsic
# (every CPU that runs the eboot has AVX).
set_property(SOURCE src/decomp/camera/follow_camera.cpp APPEND PROPERTY COMPILE_OPTIONS -mavx)

if(MSVC)
  target_compile_options(bbhost PRIVATE /W4)
else()
  target_compile_options(bbhost PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(UNIX)
  enable_testing()
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME world_baseline_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/world_baseline_test.py)
    add_test(NAME area_check_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/area_check_test.py)
    add_test(NAME lift_verify_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/lift_verify_test.py)
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
  add_dependencies(config_test bbhost_version)
  add_test(NAME config_test COMMAND config_test)
  add_executable(text_entry_test tests/text_entry_test.cpp src/host/text_entry.cpp)
  target_include_directories(text_entry_test PRIVATE src)
  add_test(NAME text_entry_test COMMAND text_entry_test)
  add_executable(engine_layout tests/engine_layout.cpp)
  target_include_directories(engine_layout PRIVATE src)
  add_test(NAME engine_layout COMMAND engine_layout)
  add_executable(write_watch_test tests/write_watch_test.cpp src/core/write_watch.cpp)
  target_include_directories(write_watch_test PRIVATE src)
  target_link_libraries(write_watch_test PRIVATE pthread)
  add_test(NAME write_watch_test COMMAND write_watch_test)
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
  add_executable(translation_cache_test tests/translation_cache_test.cpp)
  target_include_directories(translation_cache_test PRIVATE src ${CMAKE_BINARY_DIR}/gen)
  target_link_libraries(translation_cache_test PRIVATE gcn ZLIB::ZLIB)
  add_dependencies(translation_cache_test bbhost_translator_fingerprint)
  add_test(NAME translation_cache_test COMMAND translation_cache_test)
  set_tests_properties(translation_cache_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  add_executable(gcn_wave_test tests/gcn_wave_test.cpp)
  target_link_libraries(gcn_wave_test PRIVATE gcn)
  add_test(NAME gcn_wave_test COMMAND gcn_wave_test)
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
    if(Vulkan_FOUND)
      add_executable(gcn_sampling_test tests/gcn_sampling_test.cpp)
      target_link_libraries(gcn_sampling_test PRIVATE gcn ${SPIRV_TOOLS_LIB} Vulkan::Vulkan)
      add_test(NAME gcn_sampling_test COMMAND gcn_sampling_test)
      set_tests_properties(gcn_sampling_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
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
    DEPENDS bbhost bb_plugin_randomizer bb_plugin_boss_rush bb_plugin_mutators bb_plugin_debug_menu
    USES_TERMINAL)
else()
  add_custom_target(smoke
    COMMAND ${CMAKE_SOURCE_DIR}/tools/smoke.sh $<TARGET_FILE:bbhost> ${CMAKE_BINARY_DIR}/smoke-run.log ${CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS bbhost
    USES_TERMINAL)
endif()
