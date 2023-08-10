#pragma once

#include <cstdint>
#include <functional>

namespace sim
{

// TODO better place for this struct?

// Can represent a Collision, a LinkUpEvent or a LinkDownEvent
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

using InterfaceCollision = IDPair;
using LinkUpEvent = IDPair;
using LinkDownEvent = IDPair;

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
