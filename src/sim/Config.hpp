#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace sim {

class Config {
 public:
    // Note: Here are only definitions,
    //       for explanations and default values see 'Config.cpp'

    static bool run_headless;
    static int64_t max_ticks;
    static std::size_t max_entities;
    static std::size_t max_link_events;
    static std::filesystem::path map_filepath;
    static float collision_radius;
    static int argc;
    static char** argv;
    static std::vector<std::string> args;

    Config() = delete;

};

}  // namespace sim
