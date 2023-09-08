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

struct ePos
{
    size_t tick;
    uint32_t eID;
    double x;
    double y;

    ePos() {}

    // Expected format: "tick,eID,x,y"
    ePos(const std::string& _s)
    {
        char* s;

        tick = strtol(_s.c_str(), &s, 10);
        assert(*s == ',');
        s++;

        eID = (uint32_t) strtol(s, &s, 10);
        assert(*s == ',');
        s++;

        x = strtod(s, &s);
        assert(*s == ',');
        s++;

        y = strtod(s, &s);
        assert(*s == '\0');
    }

    double dist(const ePos& _other) const
    {
        const double xdiff = x - _other.x;
        const double ydiff = y - _other.y;
        return std::sqrt(xdiff * xdiff + ydiff * ydiff);
    }

    std::string to_string() const
    {
        return std::to_string(tick) + "," + std::to_string(eID) + "," +
            std::to_string(x) + "," + std::to_string(y);
    }
};

int main()
{
    // Results:
    //  error increases over time
    // float <=> double differences tiny cause drift from tick to tick
    //  this causes entities to arrive eventually at their destination slightly faster than in the other simulation
    //  => further drift

    std::ifstream in0("/home/crydsch/msim/logs/debug/pos_one_msimme_100Kt_10Ke.txt");
    if (in0.bad() || in0.fail()) std::exit(1);
    std::ifstream in1("/home/crydsch/msim/logs/debug/pos_one_defme_100Kt_10Ke.txt");
    if (in1.bad() || in1.fail()) std::exit(1);
    size_t num_ticks = 10000;
    size_t num_entities = 10000;
    size_t output_interval = 1000;

    std::vector<double> error_ticks(num_ticks+1);       // accumulated error per tick
    std::vector<double> error_entities(num_entities);   // accumulated error per entity

    std::string line;
    for (size_t tick = 1; tick <= num_ticks; tick++)
    {
        for (size_t entity = 0; entity < num_entities; entity++)
        {
            std::getline(in0, line);
            assert(!in0.bad() && !in0.fail());
            const ePos pos0 = ePos(line);
            assert(pos0.tick == tick);
            assert(pos0.eID == entity);

            std::getline(in1, line);
            assert(!in1.bad() && !in1.fail());
            const ePos pos1 = ePos(line);
            assert(pos1.tick == tick);
            assert(pos1.eID == entity);

            assert(pos0.tick == pos1.tick);
            assert(pos0.eID == pos1.eID);
            double error = pos0.dist(pos1);
            error_ticks[tick] += error;
            error_entities[entity] += error;
        }

        if (tick % output_interval == 0)
        {
            // Output mean entity error up to this tick
            double mean = 0.0;
            for (size_t entity = 0; entity < num_entities; entity++)
            {
                mean += (error_entities[entity] / tick);
            }

            mean /= num_entities;
            // printf("%ld: %lf\n", tick, mean);
        }

        // Output current tick error
        // printf("%ld: %lf\n", tick, error_ticks[tick] / num_entities);
    }

    // Output mean tick error (over entire simulation)
    double mean_tick_error = 0.0;
    for (size_t tick = 1; tick <= num_ticks; tick++)
    {
        error_ticks[tick] /= num_entities; // calc tick mean
        mean_tick_error += error_ticks[tick];
    }
    // printf("%lf\n", mean_tick_error / num_ticks);

    // Output error difference between ticks
    double mean_diff_error = 0.0;
    for (size_t tick = 2; tick <= num_ticks; tick++)
    {
        double diff = std::abs(error_ticks[tick] - error_ticks[tick-1]);
        mean_diff_error += diff;
        // printf("%lf\n", diff);
    }
    printf("%lf\n", mean_diff_error / (num_ticks-1));
}
