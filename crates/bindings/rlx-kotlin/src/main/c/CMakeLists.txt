# RLX — versatile ML compiler + runtime. MIT OR Apache-2.0.
#
# librlx_jni: the generated JNI glue over librlx_capi. Android: point the app
# module's externalNativeBuild at this file and RLX_CAPI_LIB_DIR at the
# librlx_capi.so built for each ABI (cargo ndk -t arm64-v8a … build -p rlx-capi).
# Desktop JVM: cmake -S . -B build -DRLX_CAPI_LIB_DIR=<dir> && cmake --build build
cmake_minimum_required(VERSION 3.18)
project(rlx_jni C)

# A packaged release bundles librlx_capi under native/<os>-<arch>/; a checkout
# uses the workspace build. RLX_CAPI_LIB_DIR overrides either.
if(APPLE)
  set(_rlx_os macos)
elseif(WIN32)
  set(_rlx_os windows)
else()
  set(_rlx_os linux)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
  set(_rlx_arch arm64)
else()
  set(_rlx_arch x64)
endif()
set(_rlx_bundled "${CMAKE_CURRENT_SOURCE_DIR}/../../../native/${_rlx_os}-${_rlx_arch}")
if(EXISTS "${_rlx_bundled}")
  set(_rlx_default_lib_dir "${_rlx_bundled}")
else()
  set(_rlx_default_lib_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../target/debug")
endif()
set(RLX_CAPI_LIB_DIR "${_rlx_default_lib_dir}" CACHE PATH "directory holding librlx_capi")
find_library(RLX_CAPI rlx_capi PATHS ${RLX_CAPI_LIB_DIR} NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)

add_library(rlx_jni SHARED rlx_jni.c rlx_jni_support.c)
target_include_directories(rlx_jni PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../rlx-capi/include)
if(NOT ANDROID) # the NDK sysroot has jni.h
    find_package(JNI REQUIRED)
    target_include_directories(rlx_jni PRIVATE ${JNI_INCLUDE_DIRS})
endif()
target_link_libraries(rlx_jni PRIVATE ${RLX_CAPI})
