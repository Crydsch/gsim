#pragma once

#include "Config.hpp"
#include <cstdint>
#include <functional>

namespace sim
{

struct InterfaceCollisionBlock {
    uint32_t offset{0}; // points to next free slot in 'colls' or to next block
    uint32_t colls[Config::InterfaceCollisionBlockSize]{0};
} __attribute__((aligned(64))) __attribute__((__packed__));
constexpr std::size_t InterfaceCollisionBlockSize = sizeof(InterfaceCollisionBlock);

// Can represent a WaypointRequest, a Collision, a LinkUpEvent
struct IDPair
{
    uint32_t ID0{0};
    uint32_t ID1{0};

    bool operator==(const IDPair& other) const
    {
        return ID0 == other.ID0 &&
            ID1 == other.ID1;
    }
} __attribute__((aligned(8))) __attribute__((__packed__));
constexpr std::size_t IDPairSize = sizeof(IDPair);

using WaypointRequest = IDPair; // ID0: Entity ID / ID1: Number of requested waypoints
using InterfaceCollision = IDPair;
using LinkUpEvent = IDPair;

}  // namespace sim

// Ref.: https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
// Ref.: https://www.boost.org/doc/libs/1_55_0/doc/html/hash/combine.html
template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Ref.: https://en.cppreference.com/w/cpp/utility/hash
template <>
struct std::hash<sim::IDPair>
{
    std::size_t operator()(sim::IDPair const& pair) const noexcept
    {
        std::size_t seed = 0;
        hash_combine(seed, pair.ID0);
        hash_combine(seed, pair.ID1);
        return seed;
    }
};
