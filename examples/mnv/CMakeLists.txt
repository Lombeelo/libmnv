cmake_minimum_required(VERSION 3.14)
project(mnv-gen-example LANGUAGES CXX)

set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(mnv-gen-example)
target_sources(mnv-gen-example PRIVATE ${sources})
target_link_libraries(mnv-gen-example PRIVATE mnv::mnv)
