cmake_minimum_required(VERSION 3.16)

add_subdirectory(shader)

include_directories(${CMAKE_CURRENT_LIST_DIR})

add_library(sim 
    Simulator.cpp
    Simulator.hpp
    Entity.cpp
    Entity.hpp
    Map.cpp
    Map.hpp
    PushConsts.cpp
    PushConsts.hpp
    GpuQuadTree.cpp
    GpuQuadTree.hpp
)

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
