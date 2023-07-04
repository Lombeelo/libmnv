cmake_minimum_required(VERSION 3.14)
project(mnv-tests)

#----------------------------------------------------------------------------------------------------------------------
# general settings and options
#----------------------------------------------------------------------------------------------------------------------

include("../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    enable_testing()
endif()

#----------------------------------------------------------------------------------------------------------------------
# testing framework
#----------------------------------------------------------------------------------------------------------------------
option(MNV_USE_LOCAL_DEPS "Use locally installed dependencies (otherwise FetchContent is used)" ON)

if(NOT MNV_USE_LOCAL_DEPS)

    include(FetchContent)
    FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz)
    # For simplicity, always build googletest as static library.
    set(BUILD_SHARED_LIBS OFF)
    set(INSTALL_GTEST OFF)
    FetchContent_MakeAvailable(googletest)

else()

    set(INSTALL_GTEST OFF)
    set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCED)
    add_subdirectory(../deps/googletest-1.13.0 ../deps/build/googletest-1.13.0)
    set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_OLD} CACHE BOOL "" FORCED)

endif(NOT MNV_USE_LOCAL_DEPS)




#----------------------------------------------------------------------------------------------------------------------
# tests dependencies
#----------------------------------------------------------------------------------------------------------------------

if(is_top_level)
    find_package(mnv REQUIRED)
endif()

#----------------------------------------------------------------------------------------------------------------------
# tests sources
#----------------------------------------------------------------------------------------------------------------------

set(sources
    add_test.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

#----------------------------------------------------------------------------------------------------------------------
# tests target
#----------------------------------------------------------------------------------------------------------------------

add_executable(mnv-tests)
target_sources(mnv-tests PRIVATE ${sources})

target_link_libraries(mnv-tests
    PRIVATE
        mnv::mnv
        gtest_main)


include(GoogleTest)
gtest_discover_tests(mnv-tests)