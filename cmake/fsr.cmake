include(FetchContent)

# Location of FSR-Vulkan provider: by default fetch pinned revision from GitHub,
# with BBHOST_FSR_SOURCE_DIR cache variable override for local/offline builds.
set(BBHOST_FSR_SOURCE_DIR "" CACHE PATH "Path to local clone of FSR-Vulkan repository (overrides FetchContent)")

if(BBHOST_FSR_SOURCE_DIR AND EXISTS "${BBHOST_FSR_SOURCE_DIR}")
  message(STATUS "bbhost: using local FSR-Vulkan provider from ${BBHOST_FSR_SOURCE_DIR}")
  FetchContent_Declare(
    fsr_vulkan
    SOURCE_DIR "${BBHOST_FSR_SOURCE_DIR}"
  )
else()
  FetchContent_Declare(
    fsr_vulkan
    GIT_REPOSITORY https://github.com/FireBurn/FSR-Vulkan.git
    GIT_TAG        c64f093404125e960813f95f1996c14780f0dd69
  )
endif()

# Configure options before making it available
set(FFX_VK_PORTABLE_BUILD_FSR3_HOST OFF CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_BUILD_FSR3_VK_BACKEND OFF CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_BUILD_FSR3_3_1_5_SCAFFOLD ON CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_BUILD_FSR4_V07_VULKAN ON CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_BUILD_PROBE OFF CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(fsr_vulkan)

# The pinned bridge elides same-layout transitions. A reset can clear an image
# immediately after its initial clear, so those transfer writes still need an
# execution dependency. Keep this narrow SDK fix reproducible for fetched and
# local providers, and reject unexpected source revisions rather than guessing.
find_package(Git REQUIRED)
set(fsr_clear_patch "${CMAKE_CURRENT_LIST_DIR}/fsr-clear-order.patch")
execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${fsr_clear_patch}"
  WORKING_DIRECTORY "${fsr_vulkan_SOURCE_DIR}" RESULT_VARIABLE fsr_patch_present
  OUTPUT_QUIET ERROR_QUIET)
if(NOT fsr_patch_present EQUAL 0)
  execute_process(COMMAND "${GIT_EXECUTABLE}" apply --check "${fsr_clear_patch}"
    WORKING_DIRECTORY "${fsr_vulkan_SOURCE_DIR}" RESULT_VARIABLE fsr_patch_check
    OUTPUT_QUIET ERROR_QUIET)
  if(NOT fsr_patch_check EQUAL 0)
    message(FATAL_ERROR "FSR provider does not match the pinned clear-order patch")
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" apply "${fsr_clear_patch}"
    WORKING_DIRECTORY "${fsr_vulkan_SOURCE_DIR}" RESULT_VARIABLE fsr_patch_result)
  if(NOT fsr_patch_result EQUAL 0)
    message(FATAL_ERROR "Could not apply FSR provider clear-order patch")
  endif()
endif()

add_library(bbhost_fsr_provider OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/host/fsr_provider.cpp)
target_include_directories(bbhost_fsr_provider PRIVATE
  "${fsr_vulkan_SOURCE_DIR}/upstream/ffx-2.3.0/Kits/FidelityFX/upscalers/fsr3/internal")
target_link_libraries(bbhost_fsr_provider PRIVATE ffx-vulkan::fsr3-vk-backend-3.1.5)

if(TARGET ffx_vulkan_fsr3_host_3_1_5_scaffold)
  target_compile_options(ffx_vulkan_fsr3_host_3_1_5_scaffold PRIVATE -Wno-missing-field-initializers -Wno-missing-braces)
endif()
if(TARGET ffx_vulkan_fsr3_sdk_core_3_1_5_scaffold)
  target_compile_options(ffx_vulkan_fsr3_sdk_core_3_1_5_scaffold PRIVATE -Wno-missing-field-initializers -Wno-missing-braces)
endif()
if(TARGET ffx_vulkan_fsr3_framegen_3_1_6_scaffold)
  target_compile_options(ffx_vulkan_fsr3_framegen_3_1_6_scaffold PRIVATE -Wno-missing-field-initializers -Wno-missing-braces)
endif()
if(TARGET ffx_vulkan_fsr3_vk_backend_3_1_5)
  target_compile_options(ffx_vulkan_fsr3_vk_backend_3_1_5 PRIVATE -Wno-missing-field-initializers -Wno-missing-braces)
endif()
if(TARGET ffx_vulkan_fsr3_vk_framegeneration_3_1_6)
  target_compile_options(ffx_vulkan_fsr3_vk_framegeneration_3_1_6 PRIVATE -Wno-missing-field-initializers -Wno-missing-braces)
endif()
if(TARGET ffx_vulkan_fsr4_v07_vulkan)
  target_compile_options(ffx_vulkan_fsr4_v07_vulkan PRIVATE -Wno-missing-field-initializers -Wno-missing-braces)
endif()
if(TARGET ffx_vulkan_fsr4_v07_assets)
  target_compile_options(ffx_vulkan_fsr4_v07_assets PRIVATE -Wno-missing-field-initializers -Wno-missing-braces)
endif()
if(TARGET ffx_vulkan_portable)
  target_compile_options(ffx_vulkan_portable PRIVATE -Wno-missing-field-initializers -Wno-missing-braces)
endif()
