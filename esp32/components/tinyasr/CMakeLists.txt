idf_component_register(SRCS "tinyasr.c" "kernels.c" "tinyasr_lm.c" "tasr_nemo.c" "tasr_seg.c"
                       INCLUDE_DIRS "include" "."
                       PRIV_INCLUDE_DIRS ".")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -Wno-unused-function)

if(DEFINED ENV{TASR_DEBUG_SUMS})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE TASR_DEBUG_SUMS)
endif()
if(DEFINED ENV{TASR_DEBUG_DUMP})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE TASR_DEBUG_DUMP)
endif()
if(CONFIG_TASR_O3)
    target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -funroll-loops)  # after -O2: the last -O wins
endif()

if(CONFIG_TASR_P4_PIE)
    # esp.* vector instructions for the ESP32-P4 (IDF's default -march may not include them)
    set_source_files_properties(kernels.c PROPERTIES COMPILE_OPTIONS "-march=rv32imafc_zicsr_zifencei_xesppie")
endif()
