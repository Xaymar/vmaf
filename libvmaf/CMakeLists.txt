# CMake build for libvmaf, an alternative to the Meson build (meson.build). Both are kept in sync;
# the options below mirror meson_options.txt with a VMAF_ prefix.
#
#   cmake -S libvmaf -B build -G Ninja -DCMAKE_BUILD_TYPE=Release [-DVMAF_ENABLE_VULKAN=ON]
#   cmake --build build
#   ctest --test-dir build
#
# On x86 the assembly (src/x86/cpuid.asm) needs NASM (>= 2.13.02, AVX-512 needs >= 2.14); on Windows
# install it from https://www.nasm.us or `winget install nasm`, or pass -DVMAF_ENABLE_ASM=OFF.
cmake_minimum_required(VERSION 3.20)

project(libvmaf VERSION 3.2.0 LANGUAGES C CXX)

set(VMAF_SOVERSION ${PROJECT_VERSION_MAJOR})

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_vmaf_toplevel ON)
else()
    set(_vmaf_toplevel OFF)
endif()

option(VMAF_ENABLE_TESTS "Build libvmaf tests" ${_vmaf_toplevel})
option(VMAF_ENABLE_TOOLS "Build the vmaf command line tool" ${_vmaf_toplevel})
option(VMAF_ENABLE_ASM "Build asm (NASM) and SIMD files, if available" ON)
option(VMAF_ENABLE_AVX512 "Build AVX-512 files, requires nasm 2.14" ON)
option(VMAF_BUILT_IN_MODELS "Compile default vmaf models into the library" ON)
option(VMAF_ENABLE_FLOAT "Compile floating-point feature extractors into the library" OFF)
option(VMAF_ENABLE_CUDA "Enable CUDA support (not implemented in the CMake build, use Meson)" OFF)
option(VMAF_ENABLE_VULKAN "Enable the Vulkan compute feature extractors (needs the Vulkan headers and glslc or glslangValidator; the Vulkan loader is loaded at run time)" OFF)
option(VMAF_ENABLE_ZIMG "Enable zimg-based picture colorspace conversion" OFF)
option(VMAF_BUNDLED_PTHREADS "Use the bundled pthread-win32 (MSVC only), OFF = an external pthreads library" ON)
option(VMAF_BUILD_SHARED "Build the shared library" ON)
option(VMAF_BUILD_STATIC "Build the static library" ON)
option(VMAF_INSTALL "Generate install rules" ${_vmaf_toplevel})

if(VMAF_ENABLE_CUDA)
    message(FATAL_ERROR "VMAF_ENABLE_CUDA is not supported by the CMake build yet; use the Meson build for CUDA.")
endif()
if(NOT VMAF_BUILD_SHARED AND NOT VMAF_BUILD_STATIC)
    message(FATAL_ERROR "Enable at least one of VMAF_BUILD_SHARED and VMAF_BUILD_STATIC.")
endif()
if(VMAF_ENABLE_TESTS AND NOT VMAF_BUILD_STATIC)
    message(STATUS "libvmaf: tests need the static library, building it as well")
    set(VMAF_BUILD_STATIC ON)
endif()
if(VMAF_ENABLE_TOOLS AND NOT VMAF_BUILD_STATIC AND MSVC)
    message(STATUS "libvmaf: the tool needs the static library on MSVC, building it as well")
    set(VMAF_BUILD_STATIC ON)
endif()

if(_vmaf_toplevel AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckCSourceCompiles)
include(GNUInstallDirs)

set(VMAF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(VMAF_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model")

# ----------------------------------------------------------------------------------------------
# Platform definitions (mirrors meson.build)
# ----------------------------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME MATCHES "Linux|Windows|CYGWIN|MSYS")
    add_compile_definitions(_GNU_SOURCE)
    set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
elseif(APPLE)
    add_compile_definitions(_DARWIN_C_SOURCE)
    set(CMAKE_REQUIRED_DEFINITIONS -D_DARWIN_C_SOURCE)
endif()
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()
add_compile_definitions(OC_NEW_STYLE_INCLUDES)
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:/wd4028> # parameter different from declaration
        $<$<COMPILE_LANGUAGE:C,CXX>:/wd4996> # use of POSIX functions
    )
endif()

if(VMAF_ENABLE_VULKAN)
    # HAVE_VULKAN changes VmafFeatureExtractor, so it must be the same in every translation unit.
    add_compile_definitions(HAVE_VULKAN=1)
endif()

check_include_file(malloc.h HAVE_MALLOC_H)
check_include_file(alloca.h HAVE_ALLOCA_H)
if(HAVE_MALLOC_H)
    add_compile_definitions(HAVE_MALLOC_H)
endif()
if(HAVE_ALLOCA_H)
    add_compile_definitions(HAVE_ALLOCA_H)
endif()

# stdatomic.h: the system one, else our replacements
check_include_file(stdatomic.h VMAF_HAVE_STDATOMIC_H)
if(NOT VMAF_HAVE_STDATOMIC_H)
    if(MSVC)
        include_directories(BEFORE "${VMAF_ROOT}/src/compat/msvc")
    else()
        check_c_source_compiles("int main(void) { int v = 0; return __atomic_fetch_add(&v, 1, __ATOMIC_SEQ_CST); }"
                                VMAF_HAVE_GCC_ATOMICS)
        if(NOT VMAF_HAVE_GCC_ATOMICS)
            message(FATAL_ERROR "Atomics not supported")
        endif()
        include_directories(BEFORE "${VMAF_ROOT}/src/compat/gcc")
    endif()
endif()
if(MSVC)
    # __builtin_clz replacement
    include_directories(BEFORE "${VMAF_ROOT}/src/compat/msvc")
    add_compile_definitions(HAVE_BUILTIN_CLZ_COMPAT)
endif()

# getopt (tools)
check_symbol_exists(getopt_long getopt.h VMAF_HAVE_GETOPT_LONG)
check_symbol_exists(strsep string.h VMAF_HAVE_STRSEP)

# threads
if(MSVC)
    add_library(vmaf_threads INTERFACE)
    set(_ptw32_static ON)
    set(_pthread_lib "")
    if(VMAF_BUNDLED_PTHREADS)
        if(NOT EXISTS "${VMAF_ROOT}/subprojects/pthread-win32/CMakeLists.txt")
            message(FATAL_ERROR "The bundled pthread-win32 is missing: run `git submodule update --init` "
                                "(libvmaf/subprojects/pthread-win32), or use -DVMAF_BUNDLED_PTHREADS=OFF "
                                "with an external pthreads library.")
        endif()
        set(BUILD_SHARED_LIBS_SAVED ${BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS OFF)
        add_subdirectory(subprojects/pthread-win32 EXCLUDE_FROM_ALL)
        set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVED})
        if(TARGET libpthreadVC3)
            set(_pthread_lib libpthreadVC3)
            set(_ptw32_static ON)
        elseif(TARGET pthreadVC3)
            set(_pthread_lib pthreadVC3)
            set(_ptw32_static OFF)
        else()
            message(FATAL_ERROR "pthread-win32 subproject: neither libpthreadVC3 nor pthreadVC3 target found.")
        endif()
        target_include_directories(vmaf_threads INTERFACE $<BUILD_INTERFACE:${VMAF_ROOT}/subprojects/pthread-win32>)
    else()
        set(_names)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND _names libpthreadVC3d pthreadVC3d libpthreadVCE3d pthreadVCE3d libpthreadVSE3d pthreadVSE3d)
        endif()
        list(APPEND _names libpthreadVC3 pthreadVC3 libpthreadVC2 pthreadVC2 libpthreadVC pthreadVC
                           libpthreadVCE3 pthreadVCE3 libpthreadVSE3 pthreadVSE3 pthread libpthread pthreads)
        find_library(VMAF_PTHREAD_LIBRARY NAMES ${_names})
        find_path(VMAF_PTHREAD_INCLUDE_DIR pthread.h)
        if(NOT VMAF_PTHREAD_LIBRARY OR NOT VMAF_PTHREAD_INCLUDE_DIR)
            message(FATAL_ERROR "VMAF_BUNDLED_PTHREADS=OFF, but no external pthreads library (and pthread.h) "
                                "was found. Install one or point CMAKE_PREFIX_PATH at it, or use the bundled one.")
        endif()
        set(_pthread_lib "${VMAF_PTHREAD_LIBRARY}")
        get_filename_component(_pt_name "${VMAF_PTHREAD_LIBRARY}" NAME)
        if(_pt_name MATCHES "^libpthread")
            set(_ptw32_static ON)
        else()
            set(_ptw32_static OFF)
        endif()
        target_include_directories(vmaf_threads INTERFACE $<BUILD_INTERFACE:${VMAF_PTHREAD_INCLUDE_DIR}>)
    endif()
    target_compile_definitions(vmaf_threads INTERFACE __PTW32_CLEANUP_C)
    if(_ptw32_static)
        target_compile_definitions(vmaf_threads INTERFACE __PTW32_STATIC_LIB)
    endif()
    target_link_libraries(vmaf_threads INTERFACE $<BUILD_INTERFACE:${_pthread_lib}>)
    set(VMAF_PC_LIBS_PRIVATE "")
else()
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    add_library(vmaf_threads INTERFACE)
    target_link_libraries(vmaf_threads INTERFACE Threads::Threads)
    set(VMAF_PC_LIBS_PRIVATE "-lm -pthread")
endif()
if(NOT WIN32)
    find_library(VMAF_MATH_LIBRARY m)
    if(VMAF_MATH_LIBRARY)
        target_link_libraries(vmaf_threads INTERFACE ${VMAF_MATH_LIBRARY})
    endif()
endif()

# ----------------------------------------------------------------------------------------------
# Architecture, assembly and SIMD
# ----------------------------------------------------------------------------------------------
if(MSVC)
    string(TOLOWER "${CMAKE_C_COMPILER_ARCHITECTURE_ID}" _arch)
else()
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _arch)
endif()
set(VMAF_X86 OFF)
set(VMAF_X86_64 OFF)
set(VMAF_ARM64 OFF)
if(_arch MATCHES "^(x86_64|amd64|x64)$")
    set(VMAF_X86 ON)
    set(VMAF_X86_64 ON)
elseif(_arch MATCHES "^(i[3-6]86|x86)$")
    set(VMAF_X86 ON)
elseif(_arch MATCHES "^(aarch64|arm64|arm64e)$")
    set(VMAF_ARM64 ON)
endif()

set(VMAF_HAVE_ASM OFF)
set(VMAF_HAVE_AVX512 OFF)
if(VMAF_ENABLE_ASM)
    if(VMAF_ARM64)
        set(VMAF_HAVE_ASM ON)
    elseif(VMAF_X86)
        set(VMAF_HAVE_ASM ON)
        find_program(NASM_EXECUTABLE NAMES nasm)
        if(NOT NASM_EXECUTABLE)
            message(FATAL_ERROR
                "NASM was not found, but it is needed to build the x86 assembly.\n"
                "  Windows: install it from https://www.nasm.us (or `winget install nasm`) and make sure\n"
                "           nasm.exe is on PATH, or set -DNASM_EXECUTABLE=<path to nasm.exe>.\n"
                "  Linux:   install the nasm package.\n"
                "Or configure with -DVMAF_ENABLE_ASM=OFF to build without assembly and SIMD.")
        endif()
        execute_process(COMMAND "${NASM_EXECUTABLE}" -v OUTPUT_VARIABLE _nasm_out RESULT_VARIABLE _nasm_res
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT _nasm_res EQUAL 0 OR NOT _nasm_out MATCHES "[Vv]ersion ([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
            message(FATAL_ERROR "could not obtain the nasm version from '${NASM_EXECUTABLE}': ${_nasm_out}")
        endif()
        set(VMAF_NASM_VERSION "${CMAKE_MATCH_1}")
        if(VMAF_NASM_VERSION VERSION_LESS "2.13.02")
            message(FATAL_ERROR "nasm 2.13.02 or later is required, found nasm ${VMAF_NASM_VERSION}")
        endif()
        if(VMAF_ENABLE_AVX512)
            if(VMAF_NASM_VERSION VERSION_LESS "2.14")
                message(WARNING "nasm 2.14 or later is required for AVX-512 asm. Disabling AVX-512.")
            else()
                set(VMAF_HAVE_AVX512 ON)
            endif()
        endif()
        set(CMAKE_ASM_NASM_COMPILER "${NASM_EXECUTABLE}")
        enable_language(ASM_NASM)
    endif()
endif()

set(VMAF_ASM_PREFIX_DEFINE "")
if(VMAF_X86 AND VMAF_HAVE_ASM)
    # symbols have an underscore prefix on 32-bit Windows and on macOS
    if(APPLE OR (WIN32 AND NOT VMAF_X86_64))
        set(VMAF_ASM_PREFIX_DEFINE "%define PREFIX 1")
    endif()
endif()

# config.h / config.asm
set(_cfg "#pragma once\n\n")
if(VMAF_ASM_PREFIX_DEFINE)
    string(APPEND _cfg "#define PREFIX 1\n")
endif()
if(VMAF_HAVE_ASM AND VMAF_ARM64)
    string(APPEND _cfg "#define HAVE_ASM 1\n#define ARCH_AARCH64 1\n#define ARCH_ARM 0\n")
endif()
if(VMAF_HAVE_ASM AND VMAF_X86)
    string(APPEND _cfg "#define HAVE_ASM 1\n#define ARCH_X86 1\n")
    if(VMAF_X86_64)
        string(APPEND _cfg "#define ARCH_X86_64 1\n#define ARCH_X86_32 0\n")
    else()
        string(APPEND _cfg "#define ARCH_X86_64 0\n#define ARCH_X86_32 1\n")
    endif()
    if(VMAF_HAVE_AVX512)
        string(APPEND _cfg "#define HAVE_AVX512 1\n")
    else()
        string(APPEND _cfg "#define HAVE_AVX512 0\n")
    endif()
endif()
if(VMAF_ENABLE_FLOAT)
    string(APPEND _cfg "#define VMAF_FLOAT_FEATURES 1\n")
else()
    string(APPEND _cfg "#define VMAF_FLOAT_FEATURES 0\n")
endif()
if(VMAF_ENABLE_ZIMG)
    string(APPEND _cfg "#define HAVE_ZIMG 1\n")
endif()

if(VMAF_X86 AND VMAF_HAVE_ASM)
    if(VMAF_X86_64)
        set(ARCH_X86_64 1)
        set(ARCH_X86_32 0)
    else()
        set(ARCH_X86_64 0)
        set(ARCH_X86_32 1)
    endif()
    if(VMAF_HAVE_AVX512)
        set(HAVE_AVX512 1)
    else()
        set(HAVE_AVX512 0)
    endif()
    set(ASM_PREFIX_DEFINE "${VMAF_ASM_PREFIX_DEFINE}")
    configure_file(cmake/config.asm.in "${CMAKE_CURRENT_BINARY_DIR}/config.asm" @ONLY)
endif()

# Built-in models: embedded as C arrays (symbols src_<file name>, like `xxd -i -n`).
set(VMAF_MODEL_C_SOURCES)
if(VMAF_BUILT_IN_MODELS)
    set(_models
        vmaf_v0.6.1.json vmaf_b_v0.6.3.json vmaf_v0.6.1neg.json vmaf_4k_v0.6.1.json vmaf_4k_v0.6.1neg.json)
    if(VMAF_ENABLE_FLOAT)
        list(APPEND _models vmaf_float_v0.6.1neg.json vmaf_float_v0.6.1.json vmaf_float_b_v0.6.3.json
                            vmaf_float_4k_v0.6.1.json)
    endif()
    foreach(f vmaf_v1.0.16_3d0h.json vmaf_v1.0.16_3d0h_2160.json vmaf_v1.0.16_5d0h.json vmaf_v1.0.16_1d5h_2160.json)
        list(APPEND _models vmaf_v1.0.16/${f})
    endforeach()
    foreach(f vmaf_v1.0.16_hfr_3d0h.json vmaf_v1.0.16_hfr_3d0h_2160.json vmaf_v1.0.16_hfr_5d0h.json
              vmaf_v1.0.16_hfr_1d5h_2160.json)
        list(APPEND _models vmaf_v1.0.16_hfr/${f})
    endforeach()
    foreach(f hdr_vmaf_v1.1.8_3d0h.json hdr_vmaf_v1.1.8_3d0h_2160.json hdr_vmaf_v1.1.8_5d0h.json
              hdr_vmaf_v1.1.8_1d5h_2160.json)
        list(APPEND _models hdr_vmaf_v1.1.8/${f})
    endforeach()
    foreach(f hdr_vmaf_v1.1.8_hfr_3d0h.json hdr_vmaf_v1.1.8_hfr_3d0h_2160.json hdr_vmaf_v1.1.8_hfr_5d0h.json
              hdr_vmaf_v1.1.8_hfr_1d5h_2160.json)
        list(APPEND _models hdr_vmaf_v1.1.8_hfr/${f})
    endforeach()
    set(_model_out "${CMAKE_CURRENT_BINARY_DIR}/models")
    file(MAKE_DIRECTORY "${_model_out}")
    foreach(m ${_models})
        get_filename_component(_name "${m}" NAME)
        string(REGEX REPLACE "[^A-Za-z0-9]" "_" _sym "src_${_name}")
        set(_out "${_model_out}/${_name}.c")
        add_custom_command(
            OUTPUT "${_out}"
            COMMAND ${CMAKE_COMMAND} -DINPUT=${VMAF_MODEL_DIR}/${m} -DOUTPUT=${_out} -DSYMBOL=${_sym}
                    -P "${VMAF_ROOT}/cmake/embed_file.cmake"
            DEPENDS "${VMAF_MODEL_DIR}/${m}" "${VMAF_ROOT}/cmake/embed_file.cmake"
            COMMENT "Embedding model ${_name}"
            VERBATIM)
        list(APPEND VMAF_MODEL_C_SOURCES "${_out}")
    endforeach()
    string(APPEND _cfg "#define VMAF_BUILT_IN_MODELS 1\n")
endif()

file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/config.h" CONTENT "${_cfg}")

# installed version.h and vcs_version.h
set(VMAF_API_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(VMAF_API_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(VMAF_API_VERSION_PATCH ${PROJECT_VERSION_PATCH})
configure_file(include/libvmaf/version.h.in "${CMAKE_CURRENT_BINARY_DIR}/include/libvmaf/version.h" @ONLY)

set(VCS_TAG "${PROJECT_VERSION}")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${VMAF_ROOT}/../.git")
    execute_process(COMMAND "${GIT_EXECUTABLE}" --git-dir "${VMAF_ROOT}/../.git" describe --tags --long
                            --match "?.*.*" --always
                    OUTPUT_VARIABLE _git_desc RESULT_VARIABLE _git_res ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_git_res EQUAL 0 AND _git_desc)
        set(VCS_TAG "${_git_desc}")
    endif()
endif()
configure_file(include/vcs_version.h.in "${CMAKE_CURRENT_BINARY_DIR}/vcs_version.h" @ONLY)

# The headers this project builds with. Generated headers come first so `libvmaf/version.h` resolves.
include_directories(
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${VMAF_ROOT}/include"
    "${VMAF_ROOT}/src"
    "${VMAF_ROOT}/src/feature"
    "${VMAF_ROOT}/src/feature/common"
)

# ----------------------------------------------------------------------------------------------
# Vulkan: headers (the loader is dlopen'ed), shaders compiled to SPIR-V and embedded
# ----------------------------------------------------------------------------------------------
set(VMAF_VULKAN_SOURCES)
add_library(vmaf_vulkan_deps INTERFACE)
if(VMAF_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(TARGET Vulkan::Headers)
        target_link_libraries(vmaf_vulkan_deps INTERFACE $<BUILD_INTERFACE:Vulkan::Headers>)
    else()
        find_path(VMAF_VULKAN_INCLUDE_DIR vulkan/vulkan.h HINTS ENV VULKAN_SDK PATH_SUFFIXES include)
        if(NOT VMAF_VULKAN_INCLUDE_DIR)
            message(FATAL_ERROR "VMAF_ENABLE_VULKAN needs the Vulkan headers (vulkan/vulkan.h): install the "
                                "Vulkan SDK / libvulkan-dev, or set VULKAN_SDK.")
        endif()
        target_include_directories(vmaf_vulkan_deps INTERFACE $<BUILD_INTERFACE:${VMAF_VULKAN_INCLUDE_DIR}>)
    endif()
    if(NOT WIN32)
        target_link_libraries(vmaf_vulkan_deps INTERFACE ${CMAKE_DL_LIBS})
    endif()

    find_program(VMAF_GLSLC NAMES glslc HINTS ${Vulkan_GLSLC_EXECUTABLE} ENV VULKAN_SDK PATH_SUFFIXES bin Bin)
    set(VMAF_GLSLANG)
    if(NOT VMAF_GLSLC)
        find_program(VMAF_GLSLANG NAMES glslangValidator glslang HINTS ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
                     ENV VULKAN_SDK PATH_SUFFIXES bin Bin)
        if(NOT VMAF_GLSLANG)
            message(FATAL_ERROR "VMAF_ENABLE_VULKAN needs glslc (shaderc) or glslangValidator to compile the "
                                "shaders (both ship with the Vulkan SDK).")
        endif()
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(_shader_dir "${VMAF_ROOT}/src/vulkan/shaders")
    set(_shader_names
        adm_cmsum adm_dcm adm_dcm_multi adm_den adm_dwt
        cambi_cvalue cambi_cvalue_cols cambi_cvalue_p32 cambi_decimate cambi_dither cambi_dmode cambi_front
        cambi_mask cambi_mode cambi_pre
        motion_x motion_xy_wide motion_y
        speed_blocks speed_blur_h speed_blur_v speed_center speed_cov speed_dec speed_mean speed_score
        speed_small
        vif_stat_h vif_stat_v vif_sub_h vif_sub_v)
    file(GLOB _shader_deps CONFIGURE_DEPENDS "${_shader_dir}/*")
    set(_spv_files)
    foreach(s ${_shader_names})
        set(_spv "${CMAKE_CURRENT_BINARY_DIR}/spv/${s}.spv")
        if(VMAF_GLSLC)
            set(_cmd "${VMAF_GLSLC}" --target-env=vulkan1.2 -O -I "${_shader_dir}" -o "${_spv}"
                     "${_shader_dir}/${s}.comp")
        else()
            set(_cmd "${VMAF_GLSLANG}" -V --target-env vulkan1.2 "-I${_shader_dir}" -o "${_spv}"
                     "${_shader_dir}/${s}.comp")
        endif()
        add_custom_command(OUTPUT "${_spv}" COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/spv"
                           COMMAND ${_cmd}
                           DEPENDS ${_shader_deps} COMMENT "Compiling shader ${s}" VERBATIM)
        list(APPEND _spv_files "${_spv}")
    endforeach()
    set(_shaders_c "${CMAKE_CURRENT_BINARY_DIR}/vulkan_shaders.c")
    add_custom_command(OUTPUT "${_shaders_c}"
                       COMMAND Python3::Interpreter "${VMAF_ROOT}/src/vulkan/embed_spirv.py" "${_shaders_c}"
                               ${_spv_files}
                       DEPENDS ${_spv_files} "${VMAF_ROOT}/src/vulkan/embed_spirv.py"
                       COMMENT "Embedding SPIR-V shaders" VERBATIM)
    set(VMAF_VULKAN_SOURCES
        src/vulkan/common.c
        src/vulkan/engine.c
        "${_shaders_c}"
        src/feature/vulkan/integer_motion_vulkan.c
        src/feature/vulkan/integer_adm_vulkan.c
        src/feature/vulkan/integer_vif_vulkan.c
        src/feature/vulkan/cambi_vulkan.c
        src/feature/vulkan/speed_chroma_vulkan.c)
endif()

# zimg
if(VMAF_ENABLE_ZIMG)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ZIMG REQUIRED IMPORTED_TARGET zimg>=2.7)
endif()

# ----------------------------------------------------------------------------------------------
# Library
# ----------------------------------------------------------------------------------------------
# Per-target wiring that every compiled object needs
function(vmaf_target_setup target)
    target_link_libraries(${target} PRIVATE vmaf_threads vmaf_vulkan_deps)
    if(VMAF_ENABLE_ZIMG)
        target_link_libraries(${target} PRIVATE PkgConfig::ZIMG)
    endif()
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# CPU detection (+ the NASM cpuid helper)
set(_cpu_sources src/cpu.c)
if(VMAF_HAVE_ASM AND VMAF_X86)
    list(APPEND _cpu_sources src/x86/cpuid.asm src/x86/cpu.c)
elseif(VMAF_HAVE_ASM AND VMAF_ARM64)
    list(APPEND _cpu_sources src/arm/cpu.c)
endif()
add_library(vmaf_cpu OBJECT ${_cpu_sources})
vmaf_target_setup(vmaf_cpu)
if(VMAF_HAVE_ASM AND VMAF_X86)
    # NASM include paths need a trailing slash
    set_source_files_properties(src/x86/cpuid.asm PROPERTIES
        COMPILE_OPTIONS "-I${VMAF_ROOT}/src/;-I${CMAKE_CURRENT_BINARY_DIR}/")
endif()

# SIMD
set(_simd_objects)
if(VMAF_HAVE_ASM AND VMAF_X86)
    add_library(vmaf_avx2 OBJECT
        src/feature/common/convolution_avx.c
        src/feature/x86/motion_avx2.c
        src/feature/x86/vif_avx2.c
        src/feature/x86/adm_avx2.c
        src/feature/x86/cambi_avx2.c
        src/feature/x86/speed_avx2.c)
    vmaf_target_setup(vmaf_avx2)
    if(MSVC)
        target_compile_options(vmaf_avx2 PRIVATE /arch:AVX2)
    else()
        target_compile_options(vmaf_avx2 PRIVATE -mavx2 -mavx -mfma)
    endif()
    list(APPEND _simd_objects $<TARGET_OBJECTS:vmaf_avx2>)
    if(VMAF_HAVE_AVX512)
        add_library(vmaf_avx512 OBJECT
            src/feature/x86/motion_avx512.c
            src/feature/x86/vif_avx512.c
            src/feature/x86/adm_avx512.c
            src/feature/x86/speed_avx512.c)
        vmaf_target_setup(vmaf_avx512)
        if(MSVC)
            target_compile_options(vmaf_avx512 PRIVATE /arch:AVX512)
        else()
            target_compile_options(vmaf_avx512 PRIVATE -mavx512f -mavx512dq -mavx512bw -mavx512cd -mavx512vl)
        endif()
        list(APPEND _simd_objects $<TARGET_OBJECTS:vmaf_avx512>)
    endif()
elseif(VMAF_HAVE_ASM AND VMAF_ARM64)
    add_library(vmaf_neon OBJECT
        src/feature/arm64/vif_neon.c
        src/feature/arm64/adm_neon.c
        src/feature/arm64/motion_neon.c
        src/feature/arm64/speed_neon.c)
    vmaf_target_setup(vmaf_neon)
    target_compile_definitions(vmaf_neon PRIVATE ARCH_AARCH64)
    list(APPEND _simd_objects $<TARGET_OBJECTS:vmaf_neon>)
endif()

# feature extractors
set(_feature_sources
    src/feature/picture_copy.c
    src/feature/integer_psnr.c
    src/feature/third_party/xiph/psnr_hvs.c
    src/feature/feature_extractor.c
    src/feature/feature_name.c
    src/feature/alias.c
    src/feature/integer_adm.c
    src/feature/feature_collector.c
    src/feature/integer_motion.c
    src/feature/integer_vif.c
    src/feature/ciede.c
    src/feature/common/alignment.c
    src/feature/mkdirp.c
    src/feature/float_ssim.c
    src/feature/float_ms_ssim.c
    src/feature/ssim.c
    src/feature/ms_ssim.c
    src/feature/iqa/decimate.c
    src/feature/iqa/ssim_tools.c
    src/feature/iqa/math_utils.c
    src/feature/iqa/convolve.c
    src/feature/cambi.c
    src/feature/luminance_tools.c
    src/feature/null.c
    src/feature/common/convolution.c
    src/feature/vif_tools.c
    src/feature/speed.c)
if(VMAF_ENABLE_FLOAT)
    list(APPEND _feature_sources
        src/feature/float_adm.c src/feature/float_psnr.c src/feature/float_ansnr.c src/feature/float_motion.c
        src/feature/float_vif.c src/feature/float_moment.c
        src/feature/offset.c src/feature/adm.c src/feature/adm_tools.c src/feature/ansnr.c
        src/feature/ansnr_tools.c src/feature/vif.c src/feature/motion.c src/feature/psnr.c
        src/feature/psnr_tools.c src/feature/moment.c src/feature/common/blur_array.c)
endif()
list(APPEND _feature_sources ${VMAF_VULKAN_SOURCES})
add_library(vmaf_feature OBJECT ${_feature_sources})
vmaf_target_setup(vmaf_feature)

add_library(vmaf_svm OBJECT src/svm.cpp)
vmaf_target_setup(vmaf_svm)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(vmaf_svm PRIVATE -U_LIBCPP_ENABLE_ASSERTIONS)
endif()

add_library(vmaf_core OBJECT
    src/libvmaf.c src/predict.c src/model.c src/picture.c src/mem.c src/output.c src/fex_ctx_vector.c
    src/thread_pool.c src/dict.c src/opt.c src/ref.c src/read_json_model.c src/pdjson.c src/log.c
    src/framesync.c src/metadata_handler.c src/picture_pool.c src/conversion_policy.c
    ${VMAF_MODEL_C_SOURCES})
vmaf_target_setup(vmaf_core)
set_source_files_properties(${VMAF_MODEL_C_SOURCES} PROPERTIES GENERATED TRUE)

set(VMAF_ALL_OBJECTS
    $<TARGET_OBJECTS:vmaf_core>
    $<TARGET_OBJECTS:vmaf_feature>
    $<TARGET_OBJECTS:vmaf_cpu>
    $<TARGET_OBJECTS:vmaf_svm>
    ${_simd_objects})

function(vmaf_library_setup target)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${VMAF_ROOT}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(${target} PRIVATE vmaf_threads vmaf_vulkan_deps)
    if(VMAF_ENABLE_ZIMG)
        target_link_libraries(${target} PRIVATE PkgConfig::ZIMG)
    endif()
endfunction()

set(VMAF_LIB_TARGETS)
if(VMAF_BUILD_SHARED)
    add_library(vmaf SHARED ${VMAF_ALL_OBJECTS})
    vmaf_library_setup(vmaf)
    set_target_properties(vmaf PROPERTIES
        OUTPUT_NAME vmaf
        VERSION ${PROJECT_VERSION}
        SOVERSION ${VMAF_SOVERSION}
        # the Meson build exports nothing explicitly either, so every symbol is exported on Windows
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MSVC)
        set_target_properties(vmaf PROPERTIES PREFIX "")
    endif()
    list(APPEND VMAF_LIB_TARGETS vmaf)
    add_library(libvmaf::vmaf ALIAS vmaf)
endif()
if(VMAF_BUILD_STATIC)
    add_library(vmaf-static STATIC ${VMAF_ALL_OBJECTS})
    vmaf_library_setup(vmaf-static)
    if(MSVC AND VMAF_BUILD_SHARED)
        set_target_properties(vmaf-static PROPERTIES OUTPUT_NAME vmaf-static PREFIX "")
    elseif(MSVC)
        set_target_properties(vmaf-static PROPERTIES OUTPUT_NAME vmaf PREFIX "")
    else()
        set_target_properties(vmaf-static PROPERTIES OUTPUT_NAME vmaf)
    endif()
    list(APPEND VMAF_LIB_TARGETS vmaf-static)
    if(NOT VMAF_BUILD_SHARED)
        add_library(libvmaf::vmaf ALIAS vmaf-static)
    endif()
endif()
if(VMAF_BUILD_STATIC)
    set(VMAF_LINK_LIB vmaf-static)
else()
    set(VMAF_LINK_LIB vmaf)
endif()

# ----------------------------------------------------------------------------------------------
# Tools and tests
# ----------------------------------------------------------------------------------------------
if(VMAF_ENABLE_TOOLS)
    add_subdirectory(tools)
endif()
if(VMAF_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# ----------------------------------------------------------------------------------------------
# Install
# ----------------------------------------------------------------------------------------------
if(VMAF_INSTALL)
    set(_headers include/libvmaf/libvmaf.h include/libvmaf/feature.h include/libvmaf/model.h
                 include/libvmaf/picture.h "${CMAKE_CURRENT_BINARY_DIR}/include/libvmaf/version.h")
    if(VMAF_ENABLE_VULKAN)
        list(APPEND _headers include/libvmaf/libvmaf_vulkan.h)
    endif()
    install(FILES ${_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libvmaf)
    install(TARGETS vmaf_threads vmaf_vulkan_deps EXPORT vmaf-targets)
    install(TARGETS ${VMAF_LIB_TARGETS} EXPORT vmaf-targets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    if(VMAF_ENABLE_TOOLS)
        install(TARGETS vmaf_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(VMAF_PC_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
    else()
        set(VMAF_PC_LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
    endif()
    if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
        set(VMAF_PC_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
    else()
        set(VMAF_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    endif()
    if(NOT IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        # relocatable: the .pc file finds the prefix from its own location
        file(RELATIVE_PATH _pc_to_prefix "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/pkgconfig"
             "${CMAKE_INSTALL_PREFIX}")
        set(VMAF_PC_PREFIX "\${pcfiledir}/${_pc_to_prefix}")
    else()
        set(VMAF_PC_PREFIX "${CMAKE_INSTALL_PREFIX}")
    endif()
    configure_file(cmake/libvmaf.pc.in "${CMAKE_CURRENT_BINARY_DIR}/libvmaf.pc" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libvmaf.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

    include(CMakePackageConfigHelpers)
    configure_package_config_file(cmake/vmaf-config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/vmaf-config.cmake"
                                  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vmaf)
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/vmaf-config-version.cmake"
                                     VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
    install(EXPORT vmaf-targets NAMESPACE libvmaf:: FILE vmaf-targets.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vmaf)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vmaf-config.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/vmaf-config-version.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vmaf)
endif()

message(STATUS "libvmaf ${PROJECT_VERSION}: asm=${VMAF_HAVE_ASM} avx512=${VMAF_HAVE_AVX512} "
               "vulkan=${VMAF_ENABLE_VULKAN} float=${VMAF_ENABLE_FLOAT} models=${VMAF_BUILT_IN_MODELS} "
               "tools=${VMAF_ENABLE_TOOLS} tests=${VMAF_ENABLE_TESTS}")
