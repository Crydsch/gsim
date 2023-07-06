#include "Config.hpp"

namespace sim {

    // Whether to run with GUI
    bool Config::run_headless = true;

    // Number of ticks to run for then shutdown
    // A value of '-1' runs the simulation indefinitely
    int64_t Config::max_ticks = 999;

    // Number of entities to simulate
    std::size_t Config::max_entities = 1000000;

    // Maximum number of link events that may be generated
    // Note: max_entities*11 is just a heuristic for CPU based link contact detection
    std::size_t Config::max_link_events = 1000000 * 11;

    // File path to the to be used map
    // std::filesystem::path Config::map_filepath = "/home/crydsch/msim/map/test_map.json";
    // std::filesystem::path Config::map_filepath = "/home/crydsch/msim/map/eck.json";
    // std::filesystem::path Config::map_filepath = "/home/crydsch/msim/map/obo.json";
    std::filesystem::path Config::map_filepath = "/home/crydsch/msim/map/munich.json";

    // Collision radius of each entity in meters
    float Config::collision_radius = 10;

    // Commandline arguments
    int Config::argc;
    char** Config::argv;
    std::vector<std::string> Config::args;

    // Simulation hints
    //  Can be used to improve performance
    
    // Enables asynchronous entity updates on every tick
    //  (Enable this if entities data is retrieved on every tick)
    bool Config::hintSyncEntitiesEveryTick = false;


    // TODO add config file parsing (json?)
    // TODO add commandline parsing

    // bool should_run_headless(const std::vector<std::string>& args) {
    //     for (std::string arg : args)
    //     {
    //         if (arg == "--headless")
    //         {
    //             return true;
    //         }
    //     }
    //     return false;
    // }

    // int64_t parse_max_ticks(const std::vector<std::string>& args) {
    //     for (std::size_t i = 0; i < args.size(); ++i)
    //     {
    //         std::string arg = args[i];
    //         if (arg == "--max-ticks")
    //         {
    //             i++;
    //             return std::stol(args[i]);
    //         }
    //     }
    //     return -1;
    // }

}  // namespace sim
