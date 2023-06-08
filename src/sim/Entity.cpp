#include "Entity.hpp"
#include "utils/RNG.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace sim {
Entity::Entity(Rgba&& color, Vec4U&& randomState, Vec2&& pos, Vec2&& target, unsigned int roadIndex) : color(color),
                                                                                                                                           randomState(randomState),
                                                                                                                                           pos(pos),
                                                                                                                                           target(target),
                                                                                                                                           roadIndex(roadIndex) {}

int Entity::random_int() {
    static std::uniform_int_distribution<int> distr(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    auto gen = utils::RNG::generator();
    return distr(gen);
}

double Vec2::dist(const Vec2& other) const {
    return std::sqrt(std::pow(other.x - this->x, 2) + std::pow(other.y - this->y, 2));
}

Vec2 Vec2::random_vec(float x_min, float x_max, float y_min, float y_max) {
    static std::uniform_real_distribution<float> distr_x(0, x_max);
    static std::uniform_real_distribution<float> distr_y(0, y_max);
    auto gen = utils::RNG::generator();

    // Reuse existing distributions:
    if (distr_x.a() != x_min || distr_x.b() != x_max) {
        distr_x = std::uniform_real_distribution<float>(x_min, x_max);
    }
    if (distr_y.a() != y_min || distr_y.b() != y_max) {
        distr_y = std::uniform_real_distribution<float>(y_min, y_max);
    }

    return Vec2{distr_x(gen), distr_y(gen)};
}

Vec4U Vec4U::random_vec() {
    static std::uniform_int_distribution<unsigned int> distr(std::numeric_limits<unsigned int>::min(), std::numeric_limits<unsigned int>::max());
    auto gen = utils::RNG::generator();
    return {
        distr(gen),
        distr(gen),
        distr(gen),
        distr(gen)};
}

Rgba Rgba::random_color() {
    static std::uniform_real_distribution<float> distr(0, 1.0);
    auto gen = utils::RNG::generator();
    return {
        distr(gen),
        distr(gen),
        distr(gen),
        1.0};
}
}  // namespace sim
