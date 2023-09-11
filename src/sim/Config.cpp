#include "Config.hpp"
#include "spdlog/spdlog.h"
#include <sstream>

namespace sim
{

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

// Number of buffered waypoints per entity
std::size_t Config::waypoint_buffer_size = 1;
constexpr std::string_view waypoint_buffer_size_option = "--waypoint-buffer-size";

// If an entities waypoint buffer falls below this value, it generates a waypoint request
//  Entities try to keep at least this many waypoints in their buffer 
//  A value of == Config::waypoint_buffer_size causes a request after every consumed waypoint
//  A value of == 0 effectively disables threshold based waypoint requests
//   A request will then only be issued when the buffer is empty
std::size_t Config::waypoint_buffer_threshold = 0;
constexpr std::string_view waypoint_buffer_threshold_option = "--waypoint-buffer-threshold";

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

// Rendering resolution
float Config::max_render_resolution_x = 8192;  // Note: Larger values result in errors when creating frame buffers
float Config::max_render_resolution_y = 8192;
constexpr std::string_view max_render_resolution_x_option = "--render-res-x";
constexpr std::string_view max_render_resolution_y_option = "--render-res-y";

// Quadtree tuning parameters
size_t Config::quad_tree_max_depth = 8;
size_t Config::quad_tree_entity_node_cap = 10;
constexpr std::string_view quad_tree_max_depth_option = "--quadtree-depth";
constexpr std::string_view quad_tree_entity_node_cap_option = "--quadtree-nodes";

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
            }  // else

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
            }  // else

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
        else if (arg.starts_with(waypoint_buffer_size_option))
        {
            std::string value = arg.substr(waypoint_buffer_size_option.size() + 1);
            Config::waypoint_buffer_size = std::stol(value);
        }
        else if (arg.starts_with(waypoint_buffer_threshold_option))
        {
            std::string value = arg.substr(waypoint_buffer_threshold_option.size() + 1);
            Config::waypoint_buffer_threshold = std::stol(value);
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
        else if (arg.starts_with(max_render_resolution_x_option))
        {
            std::string value = arg.substr(max_render_resolution_x_option.size() + 1);
            Config::max_render_resolution_x = std::stof(value);
        }
        else if (arg.starts_with(max_render_resolution_y_option))
        {
            std::string value = arg.substr(max_render_resolution_y_option.size() + 1);
            Config::max_render_resolution_y = std::stof(value);
        }
        else if (arg.starts_with(quad_tree_max_depth_option))
        {
            std::string value = arg.substr(quad_tree_max_depth_option.size() + 1);
            Config::quad_tree_max_depth = std::stol(value);
        }
        else if (arg.starts_with(quad_tree_entity_node_cap_option))
        {
            std::string value = arg.substr(quad_tree_entity_node_cap_option.size() + 1);
            Config::quad_tree_entity_node_cap = std::stol(value);
        }

    }

    // Check validity
#if not STANDALONE_MODE
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

    if (waypoint_buffer_size <= 0)
    {
        throw std::runtime_error("Invalid configuration: waypoint_buffer_size must be given and >0.");
    }

    if (waypoint_buffer_threshold > waypoint_buffer_size)
    {
        throw std::runtime_error("Invalid configuration: waypoint_buffer_threshold must be <= waypoint_buffer_size.");
    }
#endif

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

void Config::find_correct_working_directory()
{
    // When debugging we want to adjust our current working directory
    // to the repo root to simulate the deployment environment

    auto path = std::filesystem::current_path();

    // Note: We check for the correct path, by searching for the shaders directory
    if (std::filesystem::exists(path / "assets"))
    {  // We're already set
        return;
    }

    // Search the directories upwards
    while (path != path.root_path())
    {
        path = path.parent_path();

        if (std::filesystem::exists(path / "assets"))
        {
            SPDLOG_INFO("Correcting working directory to {}", path.string().c_str());
            std::filesystem::current_path(path);
            return;
        }
    }

    throw std::runtime_error("Cannot find correct working directory. Cannot find directory 'assets'.");
}

std::filesystem::path Config::working_directory()
{
    return std::filesystem::current_path();
}

}  // namespace sim
