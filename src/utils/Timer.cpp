#include "Timer.hpp"

#include <cstdlib>  // for abort
#include <iostream>  // for operator<<, basic_ostream, endl, cout, ostream
#include <ratio>  // for ratio
#include <utility>  // for pair

namespace utils
{

size_t Timer::num_expected_samples = 2048;

Timer& Timer::Instance()
{
    static Timer INSTANCE;
    return INSTANCE;
}

void Timer::Start(const string& id)
{
    Timing& t = _timings[id];
    if (t.results.capacity() < num_expected_samples) {
        t.results.reserve(num_expected_samples);
    }
    if (t.active)
    {
        std::cout << "TIMER STARTED TWICE: " << id << std::endl;
        std::abort();
    }
    t.active = true;
    t.start = Clock::now();
};

void Timer::Stop(const string& id)
{
    Timing& t = _timings[id];
    if (!t.active)
    {
        std::cout << "TIMER NOT STARTED: " << id << std::endl;
        std::abort();
    }
    t.active = false;
    t.results.push_back(Clock::now() - t.start);
};

std::string Timer::GetResultWithSamples(const string& id)
{
    string result;

    Timing& t = _timings[id];
    result += id + "\n";
    result += "  " + std::to_string(t.results.size()) + " samples (in ms):\n  ";

    Duration sum = Duration::zero();
    uint64_t zero_count = 0;
    for (auto d : t.results)
    {
        sum += d;

        int64_t millis = DurationTo<Millis>(d);
        if (millis == 0)
        {
            ++zero_count;
        }
        else
        {
            result += std::to_string(millis) + " ";
        }
    }
    if (zero_count > 0)
    {
        result += "0(x" + std::to_string(zero_count) + ")";
    }

    result += "\n";

    if (t.results.size() > 1)
    {
        result += "  mean = " + std::to_string(DurationTo<Millis>(sum / t.results.size())) + " ms\n";
    }

    return result;
}


}  // namespace utils
