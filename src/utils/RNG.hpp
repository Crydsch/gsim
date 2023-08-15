#pragma once
#include <random>
#include "sim/Entity.hpp"

namespace utils
{

// Note: If SEED == 0 a device random seed will be used.
//       Set SEED != 0 to get reproducible pseudo random numbers.
constexpr uint32_t SEED = 1337;

class RNG
{
 private:
    static std::mt19937 gen;

 public:
    static void init()
    {
            if (SEED == 0)
            {
                std::random_device device;
                gen = std::mt19937(device());
            }
            else
            {
                gen = std::mt19937(SEED);
            }
    }

    static int random_int()
    {
        std::uniform_int_distribution<int> distr(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
        return distr(gen);
    }

    // Returns uniformly distributed values in [min,max]
    static uint32_t random_uint32_t(uint32_t _min, uint32_t _max)
    {
        std::uniform_int_distribution<uint32_t> distr(_min, _max);
        return distr(gen);
    }

    // Returns uniformly distributed values in [min,max[
    static sim::Vec2 random_vec2(float x_min, float x_max, float y_min, float y_max)
    {
        std::uniform_real_distribution<float> distr_x(x_min, x_max);
        std::uniform_real_distribution<float> distr_y(y_min, y_max);
        return {distr_x(gen), distr_y(gen)};
    }

    static sim::Vec4U random_vec4u()
    {
        static std::uniform_int_distribution<unsigned int> distr(std::numeric_limits<unsigned int>::min(), std::numeric_limits<unsigned int>::max());
        return {
            distr(gen),
            distr(gen),
            distr(gen),
            distr(gen)};
    }

    static sim::Rgba random_color()
    {
        static std::uniform_real_distribution<float> distr(0, 1.0);
        return {
            distr(gen),
            distr(gen),
            distr(gen),
            1.0};
    }
};

}  // namespace utils
