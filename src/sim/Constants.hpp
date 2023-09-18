#pragma once

#include "Map.hpp"
#include <array>
#include <cstdint>

namespace sim
{

struct Constants
{
    float worldSizeX{0};
    float worldSizeY{0};

    uint32_t nodeCount{0};
    uint32_t maxDepth{0};
    uint32_t entityNodeCap{0};

    float interfaceRange{0};

    uint32_t waypointBufferSize{0};
    uint32_t waypointBufferThreshold{0};

    uint32_t maxWaypointRequestCount{0};
    uint32_t maxInterfaceCollisionListCount{0};
    uint32_t maxInterfaceCollisionSetCount{0};
    uint32_t maxLinkEventCount{0};
} __attribute__((aligned(4))) __attribute__((packed));
constexpr std::size_t constantsSize = sizeof(Constants);

}  // namespace sim
