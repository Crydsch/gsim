cmake_minimum_required(VERSION 3.16)

add_subdirectory(shader)

add_library(sim 
    Simulator.cpp
    Entity.cpp
    Map.cpp
    GpuQuadTree.cpp
    Config.cpp
    Metadata.cpp
    PipeConnector.cpp
)

target_include_directories(sim SYSTEM PUBLIC ${GSIM_INCLUDE_DIRS_SYSTEM})
target_include_directories(sim PRIVATE ${GSIM_INCLUDE_DIRS})

target_link_libraries(sim PRIVATE 
    logger
    utils
    kompute::kompute
    nlohmann_json::nlohmann_json
    ${CMAKE_DL_LIBS}
)

if(MOVEMENT_SIMULATOR_SHADER_INTO_HEADER)
    # library providing shader headerfiles
    target_link_libraries(sim PRIVATE sim_shader)
else()
    # library providing shader load function
    target_link_libraries(sim PRIVATE sim_shaders_utils)
    # We still depend need to depend on the shader files in order for cmake to build them
    add_dependencies(sim sim_shader)
endif()
