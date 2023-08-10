#include "Map.hpp"
#include "logger/Logger.hpp"
#include "sim/Config.hpp"
#include "sim/Entity.hpp"
#include "spdlog/spdlog.h"
#include "utils/RNG.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>

namespace sim
{
Coordinate::Coordinate(Vec2 pos, unsigned int connectedIndex, unsigned int connectedCount) : pos(pos),
                                                                                             connectedIndex(connectedIndex),
                                                                                             connectedCount(connectedCount) {}

Map::Map(std::vector<Road>&& roads, std::vector<RoadPiece>&& roadPieces, std::vector<unsigned int>&& connections) : roads(std::move(roads)),
                                                                                                                    roadPieces(std::move(roadPieces)),
                                                                                                                    connections(std::move(connections))
{
    assert(roadPieces.size() == roads.size() * 2);
}

std::shared_ptr<Map> Map::load_from_file(const std::filesystem::path& path)
{
    SPDLOG_INFO("Loading map from '{}'...", path.string());
    if (!std::filesystem::exists(path))
    {
        SPDLOG_ERROR("Failed to open map from '{}'. File does not exist.", path.string());
        return nullptr;
    }

    std::ifstream file(path);
    if (!file.is_open())
    {
        SPDLOG_ERROR("Failed to open map from '{}'.", path.string());
        return nullptr;
    }
    nlohmann::json json = nlohmann::json::parse(file);

    if (!json.contains("maxLat"))
    {
        throw std::runtime_error("Failed to parse map. 'maxLat' field missing.");
    }
    float width = 0;
    json.at("maxLat").get_to(width);
    assert(width > 0.0f);
    if (Config::map_width == 0.0f)
    {
        Config::map_width = width;
    }
    else if (Config::map_width != width)
    {
        throw std::runtime_error("Failed to parse map. Map width does not match: Configured: " + std::to_string(Config::map_width) +
                                 " - Loaded Map: " + std::to_string(width));
    }

    if (!json.contains("maxLong"))
    {
        throw std::runtime_error("Failed to parse map. 'maxLong' field missing.");
    }
    float height = 0;
    json.at("maxLong").get_to(height);
    assert(height > 0.0f);
    if (Config::map_height == 0.0f)
    {
        Config::map_height = height;
    }
    else if (Config::map_height != height)
    {
        throw std::runtime_error("Failed to parse map. Map height does not match: Configured: " + std::to_string(Config::map_height) +
                                 " - Loaded Map: " + std::to_string(height));
    }

    std::vector<Road> roads;
    std::vector<RoadPiece> roadPieces;

    if (!json.contains("roads"))
    {
        throw std::runtime_error("Failed to parse map. 'roads' field missing.");
    }
    nlohmann::json::array_t roadsArray;
    json.at("roads").get_to(roadsArray);
    for (const nlohmann::json& jRoad : roadsArray)
    {
        if (!jRoad.contains("connIndexStart"))
        {
            throw std::runtime_error("Failed to parse map. 'connIndexStart' field missing.");
        }
        unsigned int connIndexStart = 0;
        assert(jRoad.at("connIndexStart").is_number_unsigned());
        jRoad.at("connIndexStart").get_to(connIndexStart);

        if (!jRoad.contains("connCountStart"))
        {
            throw std::runtime_error("Failed to parse map. 'connCountStart' field missing.");
        }
        unsigned int connCountStart = 0;
        assert(jRoad.at("connCountStart").is_number_unsigned());
        jRoad.at("connCountStart").get_to(connCountStart);

        if (!jRoad.contains("start"))
        {
            throw std::runtime_error("Failed to parse map. 'start' field missing.");
        }
        nlohmann::json jStart = jRoad["start"];

        if (!jStart.contains("lat"))
        {
            throw std::runtime_error("Failed to parse map. 'lat' field missing.");
        }
        float latStart = 0;
        jStart.at("lat").get_to(latStart);

        if (!jStart.contains("long"))
        {
            throw std::runtime_error("Failed to parse map. 'long' field missing.");
        }
        float longStart = 0;
        jStart.at("long").get_to(longStart);

        if (!jRoad.contains("connIndexEnd"))
        {
            throw std::runtime_error("Failed to parse map. 'connIndexEnd' field missing.");
        }
        unsigned int connIndexEnd = 0;
        assert(jRoad.at("connIndexEnd").is_number_unsigned());
        jRoad.at("connIndexEnd").get_to(connIndexEnd);

        if (!jRoad.contains("connCountEnd"))
        {
            throw std::runtime_error("Failed to parse map. 'connCountEnd' field missing.");
        }
        unsigned int connCountEnd = 0;
        assert(jRoad.at("connCountEnd").is_number_unsigned());
        jRoad.at("connCountEnd").get_to(connCountEnd);

        if (!jRoad.contains("end"))
        {
            throw std::runtime_error("Failed to parse map. 'end' field missing.");
        }
        nlohmann::json jEnd = jRoad["end"];

        if (!jEnd.contains("lat"))
        {
            throw std::runtime_error("Failed to parse map. 'lat' field missing.");
        }
        float latEnd = 0;
        jEnd.at("lat").get_to(latEnd);

        if (!jEnd.contains("long"))
        {
            throw std::runtime_error("Failed to parse map. 'long' field missing.");
        }
        float longEnd = 0;
        jEnd.at("long").get_to(longEnd);

        Vec2 start{latStart, longStart};
        Vec2 end{latEnd, longEnd};

        // Ensure we don't have any zero long edges (aka just points):
        if (start.x == end.x && start.y == end.y)
        {
            SPDLOG_ERROR("Zero length road detected. => Aborting!");
            SPDLOG_ERROR("lat: {}   long: {}", start.x, start.y);
            std::abort();  // Note: Just skipping would break the connection indices!
        }

        roads.emplace_back(Road{Coordinate{start, connIndexStart, connCountStart}, Coordinate{end, connIndexEnd, connCountEnd}});
        roadPieces.emplace_back(RoadPiece{start, {}, sim::Rgba{1.0, 0.0, 0.0, 1.0}});  // Start
        roadPieces.emplace_back(RoadPiece{end, {}, sim::Rgba{1.0, 0.0, 0.0, 1.0}});  // End
    }

    std::vector<uint32_t> connections{};
    if (!json.contains("connections"))
    {
        throw std::runtime_error("Failed to parse map. 'connections' field missing.");
    }
    nlohmann::json::array_t connectionsArray;
    json.at("connections").get_to(connectionsArray);
    connections.reserve(connectionsArray.size());
    for (const nlohmann::json& jConnection : connectionsArray)
    {
        assert(jConnection.is_number_unsigned());
        connections.push_back(static_cast<uint32_t>(jConnection));
    }

    SPDLOG_INFO("Map loaded from '{}'. Found {} roads with {} connections.", path.string(), roads.size(), connections.size());
    assert(roads.size() * 2 == roadPieces.size());

    // Sanity checks
    for (uint32_t roadIndex = 0; roadIndex < roads.size(); ++roadIndex)
    {
        Road& road = roads[roadIndex];

        // Connected count must be >= 1
        assert(road.start.connectedCount >= 1);
        assert(road.end.connectedCount >= 1);
        // First connected index must be referencing the road itself
        assert(connections[road.start.connectedIndex] == roadIndex);
        assert(connections[road.end.connectedIndex] == roadIndex);

        // All connected indices must reference valid roads
        for (uint32_t i = 1; i < road.start.connectedCount; ++i)
        {
            [[maybe_unused]] uint32_t index = connections[road.start.connectedIndex + i];
            assert(index < roads.size());
        }
        for (uint32_t i = 0; i < road.end.connectedCount; ++i)
        {
            [[maybe_unused]] uint32_t index = connections[road.end.connectedIndex + i];
            assert(index < roads.size());
        }

        // Road coordinates must not lie exactly on the map border
        assert(road.start.pos.x > 0.0f);
        assert(road.start.pos.y > 0.0f);
        assert(road.start.pos.x < width);
        assert(road.start.pos.y < height);
        assert(road.end.pos.x > 0.0f);
        assert(road.end.pos.y > 0.0f);
        assert(road.end.pos.x < width);
        assert(road.end.pos.y < height);
    }

    return std::make_shared<Map>(std::move(roads), std::move(roadPieces), std::move(connections));
}

uint32_t Map::get_random_road_index() const
{
    static std::uniform_int_distribution<uint32_t> distr(0, roads.size() - 1);  // min & max inclusive
    return distr(utils::RNG::generator());
}

void Map::select_road(size_t roadIndex)
{
    assert(roadIndex < roads.size());

    const sim::Rgba UNSELECTED_COLOR{1.0, 0.0, 0.0, 1.0};
    const sim::Rgba SELECTED_COLOR{0.0, 1.0, 0.0, 1.0};

    // Unselect selected road:
    if (selectedRoad != std::nullopt)
    {
        roadPieces[(*selectedRoad) * 2].color = UNSELECTED_COLOR;
        roadPieces[((*selectedRoad) * 2) + 1].color = UNSELECTED_COLOR;
    }

    // Select new road:
    selectedRoad = roadIndex;
    roadPieces[(*selectedRoad) * 2].color = SELECTED_COLOR;
    roadPieces[((*selectedRoad) * 2) + 1].color = SELECTED_COLOR;
}
}  // namespace sim