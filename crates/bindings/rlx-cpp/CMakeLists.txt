# RLX — versatile ML compiler + runtime. MIT OR Apache-2.0.
#
# The header-only `rlx::rlx` target over librlx_capi, its tests and examples.
#   cmake -B build -DRLX_CAPI_LIB_DIR=<dir with librlx_capi> && cmake --build build && ctest --test-dir build
cmake_minimum_required(VERSION 3.20)
project(rlx_cpp VERSION 0.2.19 LANGUAGES CXX)

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
set(_rlx_bundled "${CMAKE_CURRENT_SOURCE_DIR}/native/${_rlx_os}-${_rlx_arch}")
if(EXISTS "${_rlx_bundled}")
  set(_rlx_default_lib_dir "${_rlx_bundled}")
else()
  set(_rlx_default_lib_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../target/debug")
endif()
set(RLX_CAPI_LIB_DIR "${_rlx_default_lib_dir}" CACHE PATH "directory holding librlx_capi")
find_library(RLX_CAPI rlx_capi PATHS ${RLX_CAPI_LIB_DIR} NO_DEFAULT_PATH REQUIRED)

add_library(rlx INTERFACE)
add_library(rlx::rlx ALIAS rlx)
target_compile_features(rlx INTERFACE cxx_std_20)
target_include_directories(rlx INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../rlx-capi/include)
target_link_libraries(rlx INTERFACE ${RLX_CAPI})

enable_testing()
add_executable(test_rlx tests/test_rlx.cpp)
target_link_libraries(test_rlx rlx::rlx)
add_test(NAME test_rlx COMMAND test_rlx)
foreach(example inference training loading distributed backends models)
    add_executable(${example} examples/${example}.cpp)
    target_link_libraries(${example} rlx::rlx)
endforeach()
