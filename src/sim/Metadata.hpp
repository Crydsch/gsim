#pragma once

#include <cstdint>

namespace sim
{

struct Metadata
{
    uint32_t interfaceCollisionCount{0};
    uint32_t maxInterfaceCollisionCount{0};  // TODO implement checking(shader and on retrieval) | maybe move to push constants | check with config
    uint32_t linkUpEventCount{0};
    uint32_t maxLinkUpEventCount{0};  // TODO
    uint32_t linkDownEventCount{0};
    uint32_t maxLinkDownEventCount{0};  // TODO
    uint32_t PADDING{0};
    uint32_t debugData{0};
} __attribute__((aligned(8))) __attribute__((__packed__));
constexpr std::size_t metadataSize = sizeof(Metadata);

}  // namespace sim
