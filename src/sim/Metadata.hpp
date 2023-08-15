#pragma once

#include <cstdint>

namespace sim
{

struct Metadata
{
    uint32_t waypointRequestCount{0};
    uint32_t maxWaypointRequestCount{0};
    uint32_t interfaceCollisionCount{0};
    uint32_t maxInterfaceCollisionCount{0};
    uint32_t linkUpEventCount{0};
    uint32_t maxLinkUpEventCount{0};
    uint32_t linkDownEventCount{0};
    uint32_t maxLinkDownEventCount{0};
} __attribute__((aligned(8))) __attribute__((__packed__));
constexpr std::size_t metadataSize = sizeof(Metadata);

}  // namespace sim
