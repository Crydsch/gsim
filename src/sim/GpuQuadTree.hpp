#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sim::gpu_quad_tree
{

struct Node
{
    uint32_t next{0}; // the next node for tree traversal (sibling or parent)

    float offsetX{0};
    float offsetY{0};
    float width{0};
    float height{0};

    uint32_t entityCount{0};
    uint32_t first{0}; // the first entity in the linked list on this node

    uint32_t parent{0};

    uint32_t nextTL{0}; // the top left child
    // Note: nextTL+1=nextTR, nextTL+2=nextBL, nextTL+3=nextBR

    uint32_t padding[7]{0};
} __attribute__((packed)) __attribute__((aligned(64)));

// NOLINTNEXTLINE (altera-struct-pack-align) Ignore alignment since we need a compact layout.
struct Entity
{
    uint32_t node{0}; // the entities "home node"
    uint32_t next{0}; // the next entity in the linked list on this "home node"
} __attribute__((packed)) __attribute__((aligned(8)));

void init_node_zero(Node& node, float worldSizeX, float worldSizeY);

size_t calc_node_count(size_t maxDepth);
}  // namespace sim::gpu_quad_tree