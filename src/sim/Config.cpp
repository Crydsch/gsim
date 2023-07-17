#include "Config.hpp"

namespace sim {

    // Whether to run with GUI
    bool Config::run_headless = false;

    // Number of ticks to run for then shutdown
    // A value of '-1' runs the simulation indefinitely
    int64_t Config::max_ticks = -1;

    // Number of entities to simulate
    std::size_t Config::max_entities = 1000000;

    // Maximum number of interface collisions that may be generated
    // Note: This is just an empirical heuristic
    std::size_t Config::max_interface_collisions = 1000000 * 11;

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
    std::vector<std::string> Config::args;

    /* Simulation hints */
    //  Can be used to improve performance
    
    // Enables asynchronous entity updates on every tick
    //  (Enable this if entities data is retrieved on every tick)
    bool Config::hintSyncEntitiesEveryTick = false;


    // TODO add config file parsing (json?)

    void Config::parse_args()
    {
        for (std::size_t i = 0; i < args.size(); i++)
        {
            std::string arg = args[i];

            if (arg == "--headless")
            {
                Config::run_headless = true;
            }
            else if (arg == "--max-ticks")
            {
                i++;
                Config::max_ticks = std::stol(args[i]);
            }
        }
    }

    }  // namespace sim
