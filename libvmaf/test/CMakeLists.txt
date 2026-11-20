# Mirrors test/meson.build. Tests link the static library (they reach into internals).
set(_src "${VMAF_ROOT}/src")
set(VMAF_JSON_MODEL_PATH "${VMAF_MODEL_DIR}/")

# vmaf_test(name [SOURCES ...] [LINK_LIB] [OBJECTS ...] [DEFS ...] [LIBS ...] [INCLUDES ...])
function(vmaf_test name)
    cmake_parse_arguments(T "LINK_LIB;NO_THREADS" "" "SOURCES;OBJECTS;DEFS;LIBS;INCLUDES" ${ARGN})
    add_executable(${name} test.c ${name}.c ${T_SOURCES} ${T_OBJECTS})
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" ${T_INCLUDES})
    target_compile_definitions(${name} PRIVATE ${T_DEFS})
    if(T_LINK_LIB)
        target_link_libraries(${name} PRIVATE ${VMAF_LINK_LIB})
    endif()
    target_link_libraries(${name} PRIVATE vmaf_threads vmaf_vulkan_deps ${T_LIBS})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

vmaf_test(test_context LINK_LIB)
vmaf_test(test_picture SOURCES ${_src}/picture.c ${_src}/mem.c ${_src}/ref.c ${_src}/log.c ${_src}/thread_pool.c)
vmaf_test(test_propagate_metadata SOURCES ${_src}/metadata_handler.c)
vmaf_test(test_feature_collector LINK_LIB
    SOURCES ${_src}/log.c ${_src}/predict.c ${_src}/metadata_handler.c)
vmaf_test(test_log SOURCES ${_src}/log.c)
vmaf_test(test_thread_pool SOURCES ${_src}/thread_pool.c)
vmaf_test(test_model LINK_LIB
    SOURCES ${_src}/dict.c ${_src}/pdjson.c ${_src}/read_json_model.c ${_src}/log.c ${VMAF_MODEL_C_SOURCES}
    DEFS "JSON_MODEL_PATH=\"${VMAF_JSON_MODEL_PATH}\"")
vmaf_test(test_predict LINK_LIB
    SOURCES ${_src}/dict.c ${_src}/metadata_handler.c ${_src}/feature/feature_collector.c
            ${_src}/feature/alias.c ${_src}/model.c ${_src}/log.c ${_src}/read_json_model.c ${_src}/pdjson.c
            ${VMAF_MODEL_C_SOURCES} ${_src}/feature/feature_name.c ${_src}/feature/feature_extractor.c)
vmaf_test(test_feature_extractor LINK_LIB
    SOURCES ${_src}/mem.c ${_src}/picture.c ${_src}/ref.c ${_src}/dict.c ${_src}/opt.c ${_src}/log.c
            ${_src}/predict.c ${_src}/metadata_handler.c)
vmaf_test(test_dict)
vmaf_test(test_cpu OBJECTS $<TARGET_OBJECTS:vmaf_cpu>)
vmaf_test(test_ref SOURCES ${_src}/ref.c)
vmaf_test(test_feature SOURCES ${_src}/feature/alias.c ${_src}/dict.c)
vmaf_test(test_ciede LINK_LIB)
vmaf_test(test_cambi LINK_LIB SOURCES ${_src}/picture.c ${_src}/mem.c ${_src}/ref.c)
vmaf_test(test_luminance_tools LINK_LIB)
vmaf_test(test_cli_parse LINK_LIB
    SOURCES ${VMAF_ROOT}/tools/cli_parse.c ${VMAF_GETOPT_SOURCES}
    INCLUDES ${_src} ${VMAF_ROOT}/tools ${VMAF_GETOPT_INCLUDES}
    DEFS HAVE_GETOPT_H $<$<BOOL:${VMAF_HAVE_STRSEP}>:HAVE_STRSEP>)
vmaf_test(test_psnr LINK_LIB SOURCES ${_src}/picture.c)
vmaf_test(test_framesync LINK_LIB)
vmaf_test(test_adm_csf LINK_LIB)
vmaf_test(test_barten_csf LINK_LIB)
vmaf_test(test_conversion_policy SOURCES ${_src}/conversion_policy.c ${_src}/log.c)
vmaf_test(test_motion_blend LINK_LIB)
vmaf_test(test_speed_filter LINK_LIB)
vmaf_test(test_pic_preallocation LINK_LIB)
vmaf_test(test_picture_wrap_integration LINK_LIB)
vmaf_test(test_read_pictures_convert LINK_LIB
    DEFS "JSON_MODEL_PATH=\"${VMAF_JSON_MODEL_PATH}\"" $<$<BOOL:${VMAF_ENABLE_ZIMG}>:HAVE_ZIMG=1>)

if(VMAF_ENABLE_FLOAT)
    vmaf_test(test_vif_tools LINK_LIB)
    vmaf_test(test_speed_chroma LINK_LIB SOURCES ${_src}/picture.c ${_src}/mem.c ${_src}/ref.c)
endif()

if(VMAF_HAVE_ASM AND VMAF_ARM64)
    vmaf_test(test_motion_neon LINK_LIB
        SOURCES ${_src}/mem.c ${_src}/picture.c ${_src}/ref.c ${_src}/dict.c ${_src}/opt.c ${_src}/log.c
                ${_src}/predict.c ${_src}/metadata_handler.c)
endif()

if(VMAF_ENABLE_ZIMG)
    vmaf_test(test_colorspace SOURCES ${_src}/picture.c ${_src}/mem.c ${_src}/ref.c ${_src}/log.c
              LIBS PkgConfig::ZIMG DEFS HAVE_ZIMG=1)
endif()
