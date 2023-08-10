#pragma once

#include "Entity.hpp"
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>
#include <sys/types.h>

namespace sim
{

struct Coordinate
{
    Vec2 pos{};
    unsigned int connectedIndex{};
    unsigned int connectedCount{};

    Coordinate(Vec2 pos, unsigned int connectedIndex, unsigned int connectedCount);
    Coordinate() = default;
} __attribute__((aligned(16))) __attribute__((__packed__));
constexpr std::size_t coordinateSize = sizeof(Coordinate);

struct Road
{
    Coordinate start;
    Coordinate end;

} __attribute__((aligned(32))) __attribute__((__packed__));
constexpr std::size_t roadSize = sizeof(Road);

struct RoadPiece
{
    Vec2 pos;
    Vec2 padding;
    sim::Rgba color;
} __attribute__((aligned(32))) __attribute__((__packed__));
constexpr std::size_t roadPieceSize = sizeof(RoadPiece);

class Map
{
 public:
    std::vector<Road> roads;
    std::vector<RoadPiece> roadPieces;
    std::vector<uint32_t> connections;
    std::optional<size_t> selectedRoad{std::nullopt};

    Map(std::vector<Road>&& roads, std::vector<RoadPiece>&& roadPieces, std::vector<uint32_t>&& connections);

    static std::shared_ptr<Map> load_from_file(const std::filesystem::path& path);

    [[nodiscard]] uint32_t get_random_road_index() const;
    void select_road(size_t roadIndex);
};

}  // namespace sim
