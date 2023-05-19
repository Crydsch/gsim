cmake_minimum_required(VERSION 3.20)

# This function adds commands to compile <infile> to <infile.spv>
#  and <infile.spv> to <outfile>
vulkan_compile_shader(INFILE fall.comp
                      OUTFILE fall.hpp
                      NAMESPACE "sim"
                      RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake")

vulkan_compile_shader(INFILE random_move.comp
                      OUTFILE random_move.hpp
                      NAMESPACE "sim"
                      RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake")

if(MOVEMENT_SIMULATOR_SHADER_INTO_HEADER)
    add_library(sim_shader
        "${CMAKE_CURRENT_BINARY_DIR}/fall.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/random_move.hpp"
    )

    set_target_properties(sim_shader PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(sim_shader PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
else()
    # We still depend need to depend on the shader files in order for cmake to build them
    add_custom_target(sim_shader
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/fall.comp.spv"
                "${CMAKE_CURRENT_BINARY_DIR}/random_move.comp.spv"
    )
endif()

add_subdirectory(utils)
