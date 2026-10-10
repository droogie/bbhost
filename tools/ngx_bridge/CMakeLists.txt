cmake_minimum_required(VERSION 3.20)
project(ngx_bridge CXX)

# ngx_bridge.dll: NVIDIA DLSS behind a plain C API for bbhost's DLAA
# (src/host/dlaa.cpp, docs/dlaa.md). A separate MSVC build:
# nvsdk_ngx_s.lib is an MSVC static library, and bbhost is Clang + mingw-w64.
#
#   cmake -S tools/ngx_bridge -B build/ngx_bridge -G "Visual Studio 17 2022" -A x64
#         -DDLSS_SDK_DIR=<NVIDIA DLSS SDK> -DVULKAN_HEADERS_DIR=<Vulkan-Headers>
#   cmake --build build/ngx_bridge --config Release
#
# The SDK (https://github.com/NVIDIA/DLSS) is not in this repository: its
# licence is NVIDIA's own.
if(NOT MSVC)
    message(FATAL_ERROR "ngx_bridge links NVIDIA's MSVC static library: build it with MSVC")
endif()
set(DLSS_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dlss" CACHE PATH "NVIDIA DLSS SDK (include/, lib/Windows_x86_64/...)")
set(VULKAN_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan-headers" CACHE PATH "Khronos Vulkan-Headers (include/)")
find_path(VULKAN_INCLUDE_PATH vulkan/vulkan.h
          PATHS "${VULKAN_HEADERS_DIR}/include" "${VULKAN_HEADERS_DIR}"
          NO_DEFAULT_PATH REQUIRED)
find_file(NGX_STATIC_LIB nvsdk_ngx_s.lib
          PATHS "${DLSS_SDK_DIR}/lib" "${DLSS_SDK_DIR}/lib/Windows_x86_64/x64" "${DLSS_SDK_DIR}/lib/Windows_x86_64"
          NO_DEFAULT_PATH REQUIRED)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_library(ngx_bridge SHARED ngx_bridge.cpp ngx_bridge.h)
target_compile_features(ngx_bridge PRIVATE cxx_std_17)
target_include_directories(ngx_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DLSS_SDK_DIR}/include
    ${VULKAN_INCLUDE_PATH})
# The SDK resolves Vulkan entry points itself (vkGetInstanceProcAddr is passed
# in), so no vulkan-1.lib; VK_NO_PROTOTYPES keeps the headers from asking.
target_compile_definitions(ngx_bridge PRIVATE VK_NO_PROTOTYPES NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(ngx_bridge PRIVATE ${NGX_STATIC_LIB})

add_executable(ngx_probe ngx_probe.cpp ngx_gpu_test.cpp)
target_compile_features(ngx_probe PRIVATE cxx_std_17)
target_include_directories(ngx_probe PRIVATE ${VULKAN_INCLUDE_PATH})
target_compile_definitions(ngx_probe PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(ngx_probe PRIVATE ngx_bridge)
