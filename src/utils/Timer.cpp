#include "Timer.hpp"

#include <cstdlib>  // for abort
#include <iostream>  // for operator<<, basic_ostream, endl, cout, ostream
#include <ratio>  // for ratio
#include <utility>  // for pair

namespace utils
{

Timer& Timer::Instance()
{
    static Timer INSTANCE;
    return INSTANCE;
}

void Timer::Start(const string& id)
{
    Timing& t = _timings[id];
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

std::string Timer::GetResult(const string& id)
{
    string result;

    Timing& t = _timings[id];
    result += id + "\n";
    result += "  " + std::to_string(t.results.size()) + " samples\n";

    Duration sum = Duration::zero();
    for (auto d : t.results)
    {
        sum += d;
    }

    result += "  mean = " + std::to_string(DurationToMillis(sum / t.results.size())) + " ms\n";

    return result;
}

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

        int64_t millis = DurationToMillis(d);
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
        result += "  mean = " + std::to_string(DurationToMillis(sum / t.results.size())) + " ms\n";
    }

    return result;
}

std::string Timer::GetResults()
{
    string result;
    for (auto const& pair : _timings)
    {
        // cppcheck-suppress useStlAlgorithm
        result += GetResult(pair.first);
    }
    return result;
};

}  // namespace utils
