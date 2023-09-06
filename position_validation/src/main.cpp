#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>

template <class T>
struct vec2T
{
    T x;
    T y;

    explicit vec2T(T val) : x(val), y(val) {}
    vec2T(T x, T y) : x(x), y(y) {}

    bool operator==(const vec2T<T>& other) const
    {
        return other.x == x && other.y == y;
    }

    vec2T<T> operator-(const vec2T<T>& other) const
    {
        return {x - other.x, y - other.y};
    }

    vec2T<T>& operator-=(const vec2T<T>& other)
    {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    vec2T<T> operator+(const vec2T<T>& other) const
    {
        return {x + other.x, y + other.y};
    }

    vec2T<T>& operator+=(const vec2T<T>& other)
    {
        x += other.x;
        y += other.y;
        return *this;
    }

    vec2T<T>& operator*=(const vec2T<T>& other)
    {
        x *= other.x;
        y *= other.y;
        return *this;
    }

    vec2T<T> operator*(const vec2T<T>& other) const
    {
        return {x * other.x, y * other.y};
    }

    vec2T<T>& operator/=(const vec2T<T>& other)
    {
        x /= other.x;
        y /= other.y;
        return *this;
    }

    vec2T<T> operator/(const vec2T<T>& other) const
    {
        return {x / other.x, y / other.y};
    }

    vec2T<T>& operator*=(const T& val)
    {
        x *= val;
        y *= val;
        return *this;
    }

    vec2T<T> operator*(const T& val) const
    {
        return {x * val, y * val};
    }
};

using vec2d = vec2T<double>;

// Reads two double values (one per line newline) from _in into _out_vec2d
// Returns true on success, false otherwise
bool read_vec2d(std::ifstream& _in, vec2d& _out_vec2d)
{
    std::string line;
    if (!std::getline(_in, line))
    {
        return false;
    }
    _out_vec2d.x = strtod(line.c_str(), nullptr);
    if (!std::getline(_in, line))
    {
        return false;
    }
    _out_vec2d.y = strtod(line.c_str(), nullptr);
    return true;
}

// Reads _count vec2d values from _in into _out_tick
// Returns true on success, false otherwise
bool read_tick(std::ifstream& _in, size_t _count, std::vector<vec2d>& _out_tick)
{
    _out_tick.clear();
    for (size_t i = 0; i < _count; i++)
    {
        vec2d vec(0.0);
        if (!read_vec2d(_in, vec))
        {
            return false;
        }
        _out_tick.push_back(vec);
    }
    return true;
}

double dist(const vec2d v0, const vec2d v1)
{
    vec2d v = v0 - v1;
    v *= v;
    return std::sqrt(v.x + v.y);
}

// Calculates the mean absolute error betweens two ticks
double mean_absolute_error_ticks(const std::vector<vec2d>& _tick0, const std::vector<vec2d>& _tick1)
{
    double acc = 0.0;
    for (size_t i = 0; i < _tick0.size(); i++) {
        acc += dist(_tick0[i], _tick1[i]);
    }
    return acc / _tick0.size();
}

int main()
{
    std::ifstream msim_in("/home/crydsch/msim/logs/debug/pos_default.txt");
    std::ifstream one_in("/home/crydsch/msim/logs/debug/pos_msim.txt");
    size_t num_hosts = 100;

    // First results:
    //  0: default_one_pos.txt
    //  1: msim+grid_msim_pos.txt == msim+grid_one_pos.txt == msim+opt_msim_pos.txt == msim+opt_one_pos.txt
    //  0<->1: increasing error? maybe only on start?

    // float <=> double differences cause drift from tick to tick
    //  this causes entities to arrive at their destination slightly faster
    //  => further drift
    // TODO check by logging entity arrival ticks (only 1 entity first)

    std::vector<vec2d> tick0;
    std::vector<vec2d> tick1;

    double total_mean_err = 0.0;
    size_t num_ticks = 0;
    while (true) {
        if (!read_tick(msim_in, num_hosts, tick0))
        { // No more values
            break;
        }
        if(!read_tick(one_in, num_hosts, tick1))
        {
            printf("[Error] Number of ticks between the two files not equal\n");
            return 1;
        }
        
        double mae = mean_absolute_error_ticks(tick0, tick1);
        printf("%lf\n", mae);

        total_mean_err += mae;
        num_ticks++;
    }
    printf("==> %lf\n", total_mean_err / num_ticks);
}
