#pragma once

#include "Map.hpp"
#include <array>
#include <cstdint>

namespace sim
{

struct PushConsts
{
    float worldSizeX{0};
    float worldSizeY{0};

    uint32_t nodeCount{0};
    uint32_t maxDepth{0};
    uint32_t entityNodeCap{0};

    float collisionRadius{0};

    uint32_t pass{0};

    uint32_t PADDING;
} __attribute__((aligned(32))) __attribute__((packed));
constexpr std::size_t pushConstsSize = sizeof(PushConsts);

}  // namespace sim
