cmake_minimum_required(VERSION 3.14)
project(mnv
    VERSION 1.0.0
    DESCRIPTION "Template for C++ library built with CMake"
    LANGUAGES CXX)

#----------------------------------------------------------------------------------------------------------------------
# general settings and options
#----------------------------------------------------------------------------------------------------------------------

include(cmake/utils.cmake)
include(GNUInstallDirs)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

# MNV_SHARED_LIBS option (undefined by default) can be used to force shared/static build
option(MNV_BUILD_TESTS "Build mnv tests" OFF)
option(MNV_BUILD_EXAMPLES "Build mnv examples" OFF)
option(MNV_BUILD_DOCS "Build mnv documentation" OFF)


if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set (CMAKE_CXX_STANDARD 17)

set_if_undefined(CMAKE_CXX_VISIBILITY_PRESET hidden)
set_if_undefined(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(mnv) # initialized below
add_library(mnv::mnv ALIAS mnv)

#----------------------------------------------------------------------------------------------------------------------
# mnv sources
#----------------------------------------------------------------------------------------------------------------------

include(GenerateExportHeader)

set(export_file_name "export_static.h")

generate_export_header(mnv EXPORT_FILE_NAME include/mnv/${export_file_name})

set(sources
    include/mnv/mnv.hpp
    include/mnv/mnv-impl.hpp
    src/mnv.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

#----------------------------------------------------------------------------------------------------------------------
# mnv target
#----------------------------------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

target_sources(mnv PRIVATE ${sources})

target_include_directories(mnv
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

set_target_properties(mnv PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION})

#----------------------------------------------------------------------------------------------------------------------
# other targets
#----------------------------------------------------------------------------------------------------------------------

if(MNV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MNV_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(MNV_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    doxygen_add_docs(docs include/mnv ./docs)
endif()