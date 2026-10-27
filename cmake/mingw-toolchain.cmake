# Cross build for the Windows target from Linux: Clang with the
# x86_64-w64-mingw32 target and lld, the mingw-w64 sysroot and its GCC
# libstdc++ headers, static winpthreads and C++ runtime so the result is
# one .exe. The same file builds the dependencies (tools/win_deps.sh) and
# the host:
#   cmake -S . -B build-win -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-toolchain.cmake \
#         -DCMAKE_PREFIX_PATH=$PWD/build/win-deps/prefix
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

set(MINGW_TRIPLE x86_64-w64-mingw32)
find_program(MINGW_GCC ${MINGW_TRIPLE}-gcc REQUIRED)
execute_process(COMMAND ${MINGW_GCC} -print-libgcc-file-name
  OUTPUT_VARIABLE MINGW_LIBGCC OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
get_filename_component(MINGW_GCC_DIR "${MINGW_LIBGCC}" DIRECTORY)
execute_process(COMMAND ${MINGW_GCC} -print-sysroot
  OUTPUT_VARIABLE MINGW_DETECTED_SYSROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT MINGW_DETECTED_SYSROOT)
  set(MINGW_DETECTED_SYSROOT /usr/${MINGW_TRIPLE})
endif()
set(MINGW_SYSROOT ${MINGW_DETECTED_SYSROOT} CACHE PATH "mingw-w64 sysroot")
get_filename_component(MINGW_TARGET_ROOT "${MINGW_GCC_DIR}/../../../../${MINGW_TRIPLE}" ABSOLUTE)
get_filename_component(MINGW_GCC_VERSION ${MINGW_GCC_DIR} NAME)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET ${MINGW_TRIPLE})
set(CMAKE_CXX_COMPILER_TARGET ${MINGW_TRIPLE})
set(CMAKE_RC_COMPILER ${MINGW_TRIPLE}-windres)
set(CMAKE_RC_COMPILER_INIT ${MINGW_TRIPLE}-windres)
set(CMAKE_SYSROOT ${MINGW_SYSROOT})

set(MINGW_CXX_INCLUDES
  "-isystem ${MINGW_TARGET_ROOT}/include/c++/${MINGW_GCC_VERSION} -isystem ${MINGW_TARGET_ROOT}/include/c++/${MINGW_GCC_VERSION}/${MINGW_TRIPLE}")
# -Wno-invalid-constexpr: a libstdc++ header artifact under Clang.
set(CMAKE_CXX_FLAGS_INIT "${MINGW_CXX_INCLUDES} -Wno-invalid-constexpr")
# Some static-only GCC distributions keep the SEH unwinder in libgcc.a.
set(MINGW_UNWIND_FLAG "")
if(NOT EXISTS "${MINGW_GCC_DIR}/libgcc_eh.a")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/mingw-runtime")
  configure_file("${MINGW_LIBGCC}" "${CMAKE_BINARY_DIR}/mingw-runtime/libgcc_eh.a" COPYONLY)
  set(MINGW_UNWIND_FLAG "-L${CMAKE_BINARY_DIR}/mingw-runtime")
endif()
set(CMAKE_EXE_LINKER_FLAGS_INIT
  "${MINGW_UNWIND_FLAG} -fuse-ld=lld -L${MINGW_GCC_DIR} -L${MINGW_TARGET_ROOT}/lib -static -static-libstdc++ -static-libgcc")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${MINGW_UNWIND_FLAG} -fuse-ld=lld -L${MINGW_GCC_DIR} -L${MINGW_TARGET_ROOT}/lib -static-libstdc++ -static-libgcc")
# Plugins (add_library MODULE) link like shared libraries. The _INIT form
# only seeds a fresh cache, so this is the cache entry itself.
set(CMAKE_MODULE_LINKER_FLAGS "${MINGW_UNWIND_FLAG} -fuse-ld=lld -L${MINGW_GCC_DIR} -L${MINGW_TARGET_ROOT}/lib -static-libstdc++ -static-libgcc" CACHE STRING "Windows module link flags")

# libstdc++ here is built on winpthreads; every link gets it last. CMake's
# Windows-Clang platform module overrides the _INIT form, so this is the
# cache entry itself (the platform's list plus winpthread).
set(MINGW_STD_LIBS "-lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -lwinpthread")
set(CMAKE_CXX_STANDARD_LIBRARIES "${MINGW_STD_LIBS}" CACHE STRING "Windows link libraries")
set(CMAKE_C_STANDARD_LIBRARIES "${MINGW_STD_LIBS}" CACHE STRING "Windows link libraries")

# pkg-config must not hand out the Linux host's ffmpeg or curl: only what the
# sysroot and the prefix carry.
set(MINGW_PKG_DIRS "${MINGW_SYSROOT}/lib/pkgconfig")
foreach(p ${CMAKE_PREFIX_PATH})
  set(MINGW_PKG_DIRS "${MINGW_PKG_DIRS}:${p}/lib/pkgconfig")
endforeach()
set(ENV{PKG_CONFIG_LIBDIR} "${MINGW_PKG_DIRS}")
set(ENV{PKG_CONFIG_PATH} "")

# Searches are re-rooted under these, so the dependency prefix must be one.
set(CMAKE_FIND_ROOT_PATH ${MINGW_TARGET_ROOT} ${MINGW_SYSROOT} ${CMAKE_PREFIX_PATH})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
