#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cfloat>
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

    ePos() : tick(0), eID(0), x(0.0), y(0.0) {}

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

    std::ifstream in0("/gsim/logs/debug/pos_gsim_validation");
    if (in0.bad() || in0.fail()) std::exit(1);
    std::ifstream in1("/gsim/logs/debug/pos_one_validation");
    if (in1.bad() || in1.fail()) std::exit(1);
    size_t num_ticks = 1000;
    size_t num_entities = 100000;
    // size_t output_interval = 100;

    std::vector<double> error_ticks(num_ticks + 1);  // accumulated error per tick
    std::vector<double> error_entities(num_entities);  // accumulated error per entity

    std::vector<ePos> pos0(num_entities);
    std::vector<ePos> pos1(num_entities);
    std::vector<ePos> old_pos0(num_entities);
    std::vector<ePos> old_pos1(num_entities);

    double min_error_entity = DBL_MAX;
    double max_error_entity = DBL_MIN;
    double min_error_tick = DBL_MAX;
    double max_error_tick = DBL_MIN;

    std::string line;
    for (size_t tick = 1; tick <= num_ticks; tick++)
    {
        for (size_t entity = 0; entity < num_entities; entity++)
        {
            std::getline(in0, line);
            assert(!in0.bad() && !in0.fail());
            old_pos0[entity] = pos0[entity];
            pos0[entity] = ePos(line);
            assert(pos0[entity].tick == tick);
            assert(pos0[entity].eID == entity);

            std::getline(in1, line);
            assert(!in1.bad() && !in1.fail());
            old_pos1[entity] = pos1[entity];
            pos1[entity] = ePos(line);
            assert(pos1[entity].tick == tick);
            assert(pos1[entity].eID == entity);

            assert(pos0[entity].tick == pos1[entity].tick);
            assert(pos0[entity].eID == pos1[entity].eID);

            double error = pos0[entity].dist(pos1[entity]);

            min_error_entity = std::min(min_error_entity, error);
            max_error_entity = std::max(max_error_entity, error);

            error_ticks[tick] += error;
            // error_entities[entity] += error;

            // Output the first three entities with positions and error
            // printf("%ld: [%ld] old(%.8lf, %.8lf) new(%.8lf, %.8lf) %.8lf\n", tick, entity, pos0[entity].x, pos0[entity].y, pos1[entity].x, pos1[entity].y, error);
            // if (entity == 3) std::exit(0);
        }
        error_ticks[tick] /= (double) num_entities;

        min_error_tick = std::min(min_error_tick, error_ticks[tick]);
        max_error_tick = std::max(max_error_tick, error_ticks[tick]);

        // if (tick % output_interval == 0)
        // {
        //     // Output mean entity error up to this tick
        //     double mean = 0.0;
        //     for (size_t entity = 0; entity < num_entities; entity++)
        //     {
        //         mean += (error_entities[entity] / tick);
        //     }

        //     mean /= num_entities;
        //     // printf("%ld: %lf\n", tick, mean);
        // }

        // Output current tick error
        printf("%ld: %lf  (+%lf)\n", tick, error_ticks[tick], error_ticks[tick] - error_ticks[tick - 1]);
        // printf("%ld,%lf\n", tick, error_ticks[tick]);
    }

    // Output min and max errors
    printf("errors entity: [%.8lf; %.8lf]\n", min_error_entity, max_error_entity);
    printf("errors tick: [%.8lf; %.8lf]\n", min_error_tick, max_error_tick);

    // // Output mean tick error (over entire simulation)
    // double mean_tick_error = 0.0;
    // for (size_t tick = 1; tick <= num_ticks; tick++)
    // {
    //     error_ticks[tick] /= num_entities; // calc tick mean
    //     mean_tick_error += error_ticks[tick];
    // }
    // // printf("%lf\n", mean_tick_error / num_ticks);

    // // Output error difference between ticks
    // double mean_diff_error = 0.0;
    // for (size_t tick = 2; tick <= num_ticks; tick++)
    // {
    //     double diff = std::abs(error_ticks[tick] - error_ticks[tick-1]);
    //     mean_diff_error += diff;
    //     // printf("%lf\n", diff);
    // }
    // printf("%lf\n", mean_diff_error / (num_ticks-1));
}
