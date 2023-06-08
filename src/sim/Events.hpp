#pragma once

#include <cstdint>
#include <functional>

namespace sim {

struct EventMetadata {
    uint32_t linkUpEventsCount{0};
    uint32_t linkDownEventsCount{0};
} __attribute__((aligned(8))) __attribute__((__packed__));
constexpr std::size_t eventMetadataSize = sizeof(EventMetadata);

// Can represent a LinkUpEvent or a LinkDownEvent
struct LinkStateEvent {
    uint32_t interfaceID0{0};
    uint32_t interfaceID1{0};

    bool operator==(const LinkStateEvent& other) const {
        return interfaceID0 == other.interfaceID0 && 
               interfaceID1 == other.interfaceID1;
    }
} __attribute__((aligned(8))) __attribute__((__packed__));
constexpr std::size_t linkUpEventSize = sizeof(LinkStateEvent);

}  // namespace sim


// Ref.: https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
// Ref.: https://www.boost.org/doc/libs/1_55_0/doc/html/hash/combine.html
template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// Ref.: https://en.cppreference.com/w/cpp/utility/hash
template<>
struct std::hash<sim::LinkStateEvent>
{
    std::size_t operator()(sim::LinkStateEvent const& lse) const noexcept
    {
        std::size_t seed = 0;
        hash_combine(seed, lse.interfaceID0);
        hash_combine(seed, lse.interfaceID1);
        return seed;
    }
};
