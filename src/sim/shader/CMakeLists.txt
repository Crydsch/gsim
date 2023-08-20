cmake_minimum_required(VERSION 3.20)

# This function adds commands to compile <infile> to <infile.spv>
#  and <infile.spv> to <outfile>

vulkan_compile_shader(INFILE standalone.comp
                      OUTFILE standalone.hpp
                      NAMESPACE "sim"
                      RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake")

vulkan_compile_shader(INFILE accelerator.comp
                      OUTFILE accelerator.hpp
                      NAMESPACE "sim"
                      RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake")

if(MOVEMENT_SIMULATOR_SHADER_INTO_HEADER)
    add_library(sim_shader
        "${CMAKE_CURRENT_BINARY_DIR}/standalone.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/accelerator.hpp"
    )

    set_target_properties(sim_shader PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(sim_shader PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
else()
    add_custom_command(
        OUTPUT
            "${CMAKE_SOURCE_DIR}/assets/shader/vulkan/standalone.comp.spv"
            "${CMAKE_SOURCE_DIR}/assets/shader/vulkan/accelerator.comp.spv"
        COMMAND "${CMAKE_COMMAND}" -E copy 
            "${CMAKE_CURRENT_BINARY_DIR}/standalone.comp.spv"
            "${CMAKE_SOURCE_DIR}/assets/shader/vulkan/standalone.comp.spv"
        COMMAND "${CMAKE_COMMAND}" -E copy 
            "${CMAKE_CURRENT_BINARY_DIR}/accelerator.comp.spv"
            "${CMAKE_SOURCE_DIR}/assets/shader/vulkan/accelerator.comp.spv"
        DEPENDS
            "${CMAKE_CURRENT_BINARY_DIR}/standalone.comp.spv"
            "${CMAKE_CURRENT_BINARY_DIR}/accelerator.comp.spv"
        COMMENT "Copying vulkan shaders into assets directory"
    )

    # We still depend need to depend on the shader files in order for cmake to build them
    add_custom_target(sim_shader
        DEPENDS "${CMAKE_SOURCE_DIR}/assets/shader/vulkan/standalone.comp.spv"
                "${CMAKE_SOURCE_DIR}/assets/shader/vulkan/standalone.comp.spv"
    )
endif()

add_subdirectory(utils)
