#pragma once

#include <cstdint>

namespace sim
{

struct Metadata
{
    uint32_t waypointRequestCount{0};
    uint32_t interfaceCollisionCount{0};
    uint32_t linkUpEventCount{0};
    uint32_t debug{0}; // may be used as simple debug counter
} __attribute__((aligned(8))) __attribute__((__packed__));
constexpr std::size_t metadataSize = sizeof(Metadata);

}  // namespace sim
