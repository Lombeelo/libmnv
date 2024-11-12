cmake_minimum_required(VERSION 3.13)
project(mnv
    VERSION 1.0.0
    DESCRIPTION "Library for generating multivariate normal values"
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
option(MNV_ERRORS_INCLUDE_MESSAGES "The returned error types contain the error message, not only an enum" OFF)




if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set (CMAKE_CXX_STANDARD 17)

set_if_undefined(CMAKE_CXX_VISIBILITY_PRESET hidden)
set_if_undefined(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(mnv INTERFACE) # initialized below
add_library(mnv::mnv ALIAS mnv)

#----------------------------------------------------------------------------------------------------------------------
# mnv sources
#----------------------------------------------------------------------------------------------------------------------

set(sources
    include/mnv/mnv.hpp
    include/mnv/mnv-impl.hpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

#----------------------------------------------------------------------------------------------------------------------
# mnv target
#----------------------------------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

target_sources(mnv INTERFACE ${sources})

target_include_directories(mnv
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

  if(MNV_ERRORS_INCLUDE_MESSAGES)
        target_compile_definitions(mnv INTERFACE MNV_ERRORS_INCLUDE_MESSAGES)
  endif()
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
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
endif()