#pragma once

#include <cstdint>
#include <cmath>

namespace sim
{

struct Vec2
{
    float x{0};
    float y{0};

    [[nodiscard]] double dist(const Vec2& other) const
    {
        return std::sqrt(std::pow(other.x - this->x, 2) + std::pow(other.y - this->y, 2));
    };
} __attribute__((aligned(8))) __attribute__((__packed__));
constexpr std::size_t vec2Size = sizeof(Vec2);

struct Vec4U
{
    unsigned int x{0};
    unsigned int y{0};
    unsigned int z{0};
    unsigned int w{0};
} __attribute__((aligned(16))) __attribute__((__packed__));
constexpr std::size_t vec4uSize = sizeof(Vec4U);

struct Rgba
{
    float r{0};
    float g{0};
    float b{0};
    float a{0};
} __attribute__((aligned(16))) __attribute__((__packed__));
constexpr std::size_t rgbaSize = sizeof(Rgba);

#if STANDALONE_MODE

struct Entity
{
    Rgba color{1.0, 0.0, 0.0, 1.0};
    Vec2 pos{};
    Vec2 target{};
    Vec4U randomState{};
    unsigned int roadIndex{0};
    uint32_t PADDING[3]{0};

 public:
    Entity() = default;
    Entity(Rgba&& _color, Vec2&& _pos, Vec2&& _target, Vec4U&& _randomState, unsigned int _roadIndex) : color(_color), pos(_pos), target(_target), randomState(_randomState), roadIndex(_roadIndex){};
} __attribute__((aligned(64))) __attribute__((__packed__));
constexpr std::size_t entitySize = sizeof(Entity);

#else // acceleration mode

struct Entity
{
    Rgba color{1.0, 0.0, 0.0, 1.0};
    Vec2 pos{};
    uint32_t targetWaypointIndex{};
    uint32_t PADDING;

 public:
    Entity() = default;
    Entity(Rgba _color, Vec2 _pos, uint32_t _targetWaypointIndex) : color(_color), pos(_pos), targetWaypointIndex(_targetWaypointIndex) {}
} __attribute__((aligned(16))) __attribute__((__packed__));
constexpr std::size_t entitySize = sizeof(Entity);

#endif // STANDALONE_MODE

struct Waypoint
{
    Vec2 pos;
    float speed;
    float PADDING;
} __attribute__((aligned(8))) __attribute__((__packed__));
constexpr std::size_t WaypointSize = sizeof(Waypoint);

}  // namespace sim
