cmake_minimum_required(VERSION 3.18)

project(lz4)

add_library(lz4 STATIC
    src/lz4.c
    src/lz4frame.c
    src/lz4hc.c
    src/xxhash.c
)

target_include_directories(lz4 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
