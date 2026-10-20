# ESP-IDF project file for the PlatformIO build (platformio.ini, repo root). The idf.py build in esp32/firmware has its own.
cmake_minimum_required(VERSION 3.16)
# append, don't set: PlatformIO passes the src_dir component (esp32/firmware/main) in EXTRA_COMPONENT_DIRS
list(APPEND EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/esp32/components")
add_compile_definitions(TASR_PROFILE)
if(NOT TASR_2C)  # -DTASR_2C=1 (env oido_s3_bench_2c): split the work across both cores (experimental, see README)
  add_compile_definitions(TASR_SINGLE_CORE)
endif()
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(oido)
