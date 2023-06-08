#pragma once
#include <random>

namespace utils {

// Note: If SEED == 0 a device random seed will be used.
//       Set SEED != 0 to get reproducible pseudo random numbers.
constexpr uint32_t SEED = 1337;

// TODO all random functions (randomColor, randomVec4) should be here

class RNG {
 private:
    static std::mt19937 gen;
    static bool initialized;

 public:
    static std::mt19937& generator() {
        if (!initialized) {
            if (SEED == 0) {
                std::random_device device;
                gen = std::mt19937(device());
            } else {
                gen = std::mt19937(SEED);
            }
            initialized = true;
        }
        return gen;
    }
};

}  // namespace utils
