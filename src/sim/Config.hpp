#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace sim
{

class Config
{
 public:
    // Note: Here are only definitions,
    //       for explanations and default values see 'Config.cpp'

    static std::vector<std::string> args;
    static bool run_headless;
    static int64_t max_ticks;
    static std::size_t num_entities;
    static std::size_t waypoint_buffer_size;
    static std::size_t waypoint_buffer_threshold;
    static std::size_t max_interface_collisions;
    static std::size_t max_link_events;
    static float map_width;
    static float map_height;
    static std::filesystem::path map_filepath;
    static float collision_radius;
    static std::filesystem::path pipe_in_filepath;
    static std::filesystem::path pipe_out_filepath;
    static float max_render_resolution_x;
    static float max_render_resolution_y;
    static size_t quad_tree_max_depth;
    static size_t quad_tree_entity_node_cap;

    static void parse_args();
    static void find_correct_working_directory();
    static std::filesystem::path working_directory();

    // Purely static class
    Config() = delete;
};

}  // namespace sim
