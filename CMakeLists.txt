cmake_minimum_required(VERSION 3.20)
project(MOM CXX)

# C++ standard --------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options --------------------------------------------------------------
option(MOM_BUILD_SHARED "Build libMOM as a shared library (.so/.dylib)" OFF)
option(MOM_BUILD_TESTS "Build the standalone test executables" ON)
option(MOM_INSTALL "Generate install rules" ON)
option(MOM_USE_DICTIONARY "Enable dictionary support" OFF)

# Eigen ----------------------------------------------------------------------
if(DEFINED EIGEN3_DIR)
    message(STATUS "Using Eigen3 from: ${EIGEN3_DIR}")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_DIR})
else()
    find_package(Eigen3 3.4 QUIET NO_MODULE)
    if(NOT Eigen3_FOUND)
        message(STATUS "Eigen3 not found via find_package, fetching from gitlab")
        include(FetchContent)
        FetchContent_Declare(
            Eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(Eigen3)
    endif()
endif()

# std::expected (C++23) is used only in ParseConfig<> template bodies, which
# live at the bottom of each variant's .tpp and are compiled only when
# MOM_USE_DICTIONARY=ON.  The core library targets C++20; the dictionary
# build upgrades to C++23 below.

# Helper: apply common settings to a MOM target ------------------------------
function(mom_configure_target tgt access)
    target_include_directories(${tgt} ${access}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(${tgt} ${access} Eigen3::Eigen)
    target_compile_features(${tgt} ${access} cxx_std_20)
endfunction()

# MOM_headers: header-only INTERFACE target ---------------------------------
# Use this when you need to instantiate MOM templates with custom Thermo types.
add_library(MOM_headers INTERFACE)
mom_configure_target(MOM_headers INTERFACE)

# MOM: compiled library (static or shared) ----------------------------------
#
# Core sources — compiled unconditionally; no dependency on OpenSMOKE++.
set(_MOM_SOURCES
    src/ThreeEquations/ThreeEquations_BasicThermoData.cpp
    src/HMOM/HMOM_BasicThermoData.cpp
    src/HMOM6/HMOM6_BasicThermoData.cpp
    src/BrookesMoss/BrookesMoss_BasicThermoData.cpp
    src/MetalOxide/MetalOxide_BasicThermoData.cpp
    src/Utilities/OutputFileColumns.cpp
)

# Dictionary support ---------------------------------------------------------
#
# Grammar sources define DefineRules() for each variant's OpenSMOKE++ grammar.
# They include Grammar headers which pull in OpenSMOKE++ headers, so they must
# only be compiled when MOM_USE_DICTIONARY=ON and DICTIONARY_DIR is available.
if(MOM_USE_DICTIONARY)
    if(NOT DEFINED DICTIONARY_DIR)
        message(FATAL_ERROR
            "MOM_USE_DICTIONARY=ON requires DICTIONARY_DIR to be set.\n"
            "Example: -DDICTIONARY_DIR=/path/to/dictionary")
    endif()

    list(APPEND _MOM_SOURCES
        src/ThreeEquations/ThreeEquations_Grammar.cpp
        src/HMOM/HMOM_Grammar.cpp
        src/HMOM6/HMOM6_Grammar.cpp
        src/BrookesMoss/BrookesMoss_Grammar.cpp
        src/MetalOxide/MetalOxide_Grammar.cpp
    )

    message(STATUS "MOM: Dictionary support enabled")
    message(STATUS "     Dictionary path : ${DICTIONARY_DIR}")
endif()

# Choose library type
set(_MOM_LIB_TYPE STATIC)
if(MOM_BUILD_SHARED)
    set(_MOM_LIB_TYPE SHARED)
endif()

add_library(MOM ${_MOM_LIB_TYPE} ${_MOM_SOURCES})
mom_configure_target(MOM PUBLIC)

# This definition tells the headers that a pre-compiled version exists
target_compile_definitions(MOM INTERFACE MOM_COMPILED_LIBRARY)

# Dictionary linking and compile definition
if(MOM_USE_DICTIONARY)
    target_include_directories(MOM PUBLIC ${DICTIONARY_DIR}/include)
    target_link_libraries(MOM PUBLIC "${DICTIONARY_DIR}/build/libdictionary_parser.a")

    # Expose MOM_USE_DICTIONARY so variant headers activate ParseConfig<> and Grammar includes.
    target_compile_definitions(MOM PUBLIC MOM_USE_DICTIONARY)

    # ParseConfig<> uses std::expected — upgrade to C++23 for dictionary builds.
    target_compile_features(MOM PUBLIC cxx_std_23)
endif()

# Compiler warnings (only for our own sources)
target_compile_options(MOM PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)

# Tests ----------------------------------------------------------------------
if(MOM_BUILD_TESTS)
    enable_testing()

    add_executable(test_loop_benchmark tests/test_loop_benchmark.cpp)
    target_link_libraries(test_loop_benchmark PRIVATE MOM)
    target_compile_options(test_loop_benchmark PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)
    add_test(NAME LoopBenchmark COMMAND test_loop_benchmark)

    add_executable(test_verify_all_variants tests/test_verify_all_variants.cpp)
    target_link_libraries(test_verify_all_variants PRIVATE MOM)
    target_compile_options(test_verify_all_variants PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)
    add_test(NAME VerifyAllVariants COMMAND test_verify_all_variants)

    add_executable(reporter_usage_example tests/reporter_usage_example.cpp)
    target_link_libraries(reporter_usage_example PRIVATE MOM)
    target_compile_options(reporter_usage_example PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)
    add_test(NAME ReporterUsageExample COMMAND reporter_usage_example)

    if(MOM_USE_DICTIONARY)
        add_executable(test_dictionary_setup tests/test_dictionary_setup.cpp)
        target_link_libraries(test_dictionary_setup PRIVATE MOM)
        target_compile_options(test_dictionary_setup PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)
        add_test(NAME DictionarySetup COMMAND test_dictionary_setup)
    endif()

endif()

# Install rules --------------------------------------------------------------
if(MOM_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS MOM MOM_headers
        EXPORT MOMTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.tpp"
            PATTERN "*.h"
    )

    install(EXPORT MOMTargets
        FILE MOMTargets.cmake
        NAMESPACE MOM::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MOM
    )

    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/MOMConfigVersion.cmake"
        VERSION 0.1.0
        COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MOMConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/MOMConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MOM
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/MOMConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/MOMConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MOM
    )
endif()

# Configuration summary ------------------------------------------------------
message(STATUS "")
message(STATUS "MOM library configuration:")
message(STATUS "  Library type     : ${_MOM_LIB_TYPE}")
message(STATUS "  C++ standard     : C++20 (C++23 when MOM_USE_DICTIONARY=ON)")
message(STATUS "  Dictionary       : ${MOM_USE_DICTIONARY}")
if(MOM_USE_DICTIONARY)
    message(STATUS "    Dictionary dir : ${DICTIONARY_DIR}")
endif()
message(STATUS "  Build tests      : ${MOM_BUILD_TESTS}")
message(STATUS "  Install rules    : ${MOM_INSTALL}")
message(STATUS "")
