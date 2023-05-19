#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace sim {
std::vector<uint32_t> load_shader(const std::filesystem::path& path);
}  // namespace sim