#pragma once

#include <cstdint>
#include <cstddef>

namespace sim
{

struct Metadata
{
    uint32_t waypointRequestCount{0};
    uint32_t interfaceCollisionListCount{0};
    uint32_t interfaceCollisionSetCount{0};
    uint32_t interfaceLinkUpListCount{0};
    uint32_t interfaceLinkDownListCount{0};
    uint32_t debug{0}; // may be used as simple debug counter
} __attribute__((aligned(64))) __attribute__((__packed__));
constexpr size_t metadataSize = sizeof(Metadata);

}  // namespace sim
