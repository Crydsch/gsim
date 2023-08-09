#include "Config.hpp"
#include <sstream>

namespace sim {

    // Commandline arguments
    //  Format: ["--option=value","--option2=value2","--boolopt=false","--boolopt"]
    std::vector<std::string> Config::args{};

    // Whether to run with GUI
    bool Config::run_headless = true;
    constexpr std::string_view headless_option = "--headless";
    constexpr std::string_view gui_option = "--gui";

    // Number of ticks to run for then shutdown
    // A value of '-1' runs the simulation indefinitely
    int64_t Config::max_ticks = -1;
    constexpr std::string_view max_ticks_option = "--max-ticks";

    // Number of entities to simulate
    std::size_t Config::num_entities = 1000000;
    constexpr std::string_view num_entities_option = "--num-entities";

    // Maximum number of interface collisions that may be generated
    // Note: This is just an empirical heuristic
    std::size_t Config::max_interface_collisions = 1000000 * 11;
    constexpr std::string_view max_interface_collisions_option = "--max-interface-collisions";

    // Maximum number of link events that may be generated
    // Note: max_entities*11 is just a heuristic for CPU based link contact detection
    std::size_t Config::max_link_events = 1000000 * 11;
    constexpr std::string_view max_link_events_option = "--max-link-events";

    // Map size
    float Config::map_width = 0.0f;
    constexpr std::string_view map_width_option = "--map-width";
    float Config::map_height = 0.0f;
    constexpr std::string_view map_height_option = "--map-height";

    // File path to the map
    // std::filesystem::path Config::map_filepath = "/home/crydsch/msim/map/test_map.json";
    // std::filesystem::path Config::map_filepath = "/home/crydsch/msim/map/eck.json";
    // std::filesystem::path Config::map_filepath = "/home/crydsch/msim/map/obo.json";
    // std::filesystem::path Config::map_filepath = "/home/crydsch/msim/map/munich.json";
    std::filesystem::path Config::map_filepath = "";
    constexpr std::string_view map_filepath_option = "--map";

    // Collision radius of each entity in meters
    float Config::collision_radius = 10;
    constexpr std::string_view collision_radius_option = "--collision-radius";

    // File path to communication pipes
    //  May be deduced from commandline argument
    //  '--pipes=name'  =>  'pipe_in=name.in' & 'pipe_out=name.out'
    std::filesystem::path Config::pipe_in_filepath{};
    std::filesystem::path Config::pipe_out_filepath{};
    constexpr std::string_view pipes_option = "--pipes";
    constexpr std::string_view pipe_in_option = "--pipe-in";
    constexpr std::string_view pipe_out_option = "--pipe-out";

    /* Simulation hints */
    //  Can be used to improve performance
    
    // Enables asynchronous entity updates on every tick
    //  (Enable this if entity data is retrieved on every tick)
    bool Config::hint_sync_entities_every_tick = false;
    constexpr std::string_view hint_sync_entities_every_tick_option = "--hint-sync-entities-every-tick";


    void Config::parse_args()
    {
        for (std::string arg : args)
        {
            if (arg.starts_with(headless_option))
            {
                if (arg.size() == headless_option.size())
                {
                    // No value specified for boolean option => Equivalent to true
                    Config::run_headless = true;
                    continue;
                } // else

                std::string value = arg.substr(headless_option.size() + 1);
                std::istringstream(value) >> std::boolalpha >> Config::run_headless;
            }
            else if (arg.starts_with(gui_option))
            {
                if (arg.size() == gui_option.size())
                {
                    // No value specified for boolean option => Equivalent to true
                    Config::run_headless = false;
                    continue;
                } // else

                std::string value = arg.substr(gui_option.size() + 1);
                std::istringstream(value) >> std::boolalpha >> Config::run_headless;
                Config::run_headless = !Config::run_headless;
            }
            else if (arg.starts_with(max_ticks_option))
            {
                std::string value = arg.substr(max_ticks_option.size() + 1);
                Config::max_ticks = std::stol(value);
            }
            else if (arg.starts_with(num_entities_option))
            {
                std::string value = arg.substr(num_entities_option.size() + 1);
                Config::num_entities = std::stol(value);
            }
            else if (arg.starts_with(max_interface_collisions_option))
            {
                std::string value = arg.substr(max_interface_collisions_option.size() + 1);
                Config::max_interface_collisions = std::stol(value);
            }
            else if (arg.starts_with(max_link_events_option))
            {
                std::string value = arg.substr(max_link_events_option.size() + 1);
                Config::max_link_events = std::stol(value);
            }
            else if (arg.starts_with(map_width_option))
            {
                std::string value = arg.substr(map_width_option.size() + 1);
                Config::map_width = std::stof(value);
            }
            else if (arg.starts_with(map_height_option))
            {
                std::string value = arg.substr(map_height_option.size() + 1);
                Config::map_height = std::stof(value);
            }
            else if (arg.starts_with(map_filepath_option))
            {
                Config::map_filepath = arg.substr(map_filepath_option.size() + 1);
            }
            else if (arg.starts_with(collision_radius_option))
            {
                std::string value = arg.substr(collision_radius_option.size() + 1);
                Config::collision_radius = std::stof(value);
            }
            else if (arg.starts_with(pipes_option))
            {
                std::string value = arg.substr(pipes_option.size() + 1);
                Config::pipe_in_filepath = std::filesystem::path(value + ".in");
                Config::pipe_out_filepath = std::filesystem::path(value + ".out");
            }
            else if (arg.starts_with(pipe_in_option))
            {
                Config::pipe_in_filepath = arg.substr(pipe_in_option.size() + 1);
            }
            else if (arg.starts_with(pipe_out_option))
            {
                Config::pipe_out_filepath = arg.substr(pipe_out_option.size() + 1);
            }
            else if (arg.starts_with(hint_sync_entities_every_tick_option))
            {
                if (arg.size() == hint_sync_entities_every_tick_option.size())
                {
                    // No value specified for boolean option => Equivalent to true
                    Config::hint_sync_entities_every_tick = true;
                    continue;
                } // else

                std::string value = arg.substr(hint_sync_entities_every_tick_option.size() + 1);
                std::istringstream(value) >> std::boolalpha >> Config::hint_sync_entities_every_tick;
            }
        }

        // Check validity
        if (!Config::pipe_in_filepath.empty() || !Config::pipe_out_filepath.empty())
        {
            if (Config::pipe_in_filepath.empty() || Config::pipe_out_filepath.empty())
            {
                throw std::runtime_error("Invalid configuration: Both, pipe-in AND pipe-out must be specified.");
            }

            if (Config::pipe_in_filepath == Config::pipe_out_filepath)
            {
                throw std::runtime_error("Invalid configuration: pipe-in and pipe-out must be different.");
            }

            if (!std::filesystem::exists(pipe_in_filepath) || !std::filesystem::is_fifo(pipe_in_filepath))
            {
                throw std::runtime_error("Invalid configuration: pipe-in (" + pipe_in_filepath.string() + ") does not exist or is not a named pipe.");
            }

            if (!std::filesystem::exists(pipe_out_filepath) || !std::filesystem::is_fifo(pipe_out_filepath))
            {
                throw std::runtime_error("Invalid configuration: pipe-out (" + pipe_out_filepath.string() + ") does not exist or is not a named pipe.");
            }
        }

        if (Config::map_width < 0.0f || Config::map_height < 0.0f)
        {
            throw std::runtime_error("Invalid configuration: Map size must be positive.");
        }
        if ((Config::map_width == 0.0f && Config::map_height > 0.0f) ||
            (Config::map_width > 0.0f && Config::map_height == 0.0f))
        {
            throw std::runtime_error("Invalid configuration: If map sizes are specified, BOTH need to be specified.");
        }
    }

    // Returns wether msim is running in standalone mode (or accelerator mode)
    bool Config::standalone_mode()
    { // Note: It is enough to check only one pipe, since either both or none are set.
        return Config::pipe_in_filepath.empty();
    }

    }  // namespace sim
