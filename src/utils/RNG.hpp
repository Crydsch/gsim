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
    static bool initialized;

 public:
    static std::mt19937& generator()
    {
        if (!initialized)
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
            initialized = true;
        }
        return gen;
    }

    static int random_int()
    {
        static std::uniform_int_distribution<int> distr(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
        auto gen = utils::RNG::generator();
        return distr(gen);
    }

    static sim::Vec2 random_vec2(float x_min, float x_max, float y_min, float y_max)
    {
        static std::uniform_real_distribution<float> distr_x(0, x_max);
        static std::uniform_real_distribution<float> distr_y(0, y_max);
        auto gen = utils::RNG::generator();

        // Reuse existing distributions:
        if (distr_x.a() != x_min || distr_x.b() != x_max)
        {
            distr_x = std::uniform_real_distribution<float>(x_min, x_max);
        }
        if (distr_y.a() != y_min || distr_y.b() != y_max)
        {
            distr_y = std::uniform_real_distribution<float>(y_min, y_max);
        }

        return {distr_x(gen), distr_y(gen)};
    }

    static sim::Vec4U random_vec4u()
    {
        static std::uniform_int_distribution<unsigned int> distr(std::numeric_limits<unsigned int>::min(), std::numeric_limits<unsigned int>::max());
        auto gen = utils::RNG::generator();
        return {
            distr(gen),
            distr(gen),
            distr(gen),
            distr(gen)};
    }

    static sim::Rgba random_color()
    {
        static std::uniform_real_distribution<float> distr(0, 1.0);
        auto gen = utils::RNG::generator();
        return {
            distr(gen),
            distr(gen),
            distr(gen),
            1.0};
    }
};

}  // namespace utils
