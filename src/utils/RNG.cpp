#include "RNG.hpp"

namespace utils {

// Static variables
std::mt19937 RNG::gen;
bool RNG::initialized{false};

}  // namespace utils