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

void Timer::Start(const std::string& id)
{
    Timing& t = _timings[id];
    if (t.results.capacity() < num_expected_samples)
    {
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

void Timer::Stop(const std::string& id)
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

Timer::Result Timer::GetResult(const std::string& id)
{
    Timing& t = _timings[id];
    Duration mean = GetMean(t);

    std::string info;
    info += std::format("{}\n", id);
    info += std::format("  {} samples\n", t.results.size());

    // No samples?
    if (t.results.size() == 0) return {t.results.size(), mean, info};

    int64_t mean_millis = DurationTo<Millis>(mean);
    if (mean_millis > 0)
    {
        info += std::format("  mean = {} ms\n", mean_millis);
        return {t.results.size(), mean, info};
    }  // Else resolution was too low => try again

    int64_t mean_micros = DurationTo<Micros>(mean);
    if (mean_micros > 0)
    {
        info += std::format("  mean = {} us\n", mean_micros);
        return {t.results.size(), mean, info};
    }  // Else resolution was too low => try again

    int64_t mean_nanos = DurationTo<Nanos>(mean);
    if (mean_nanos > 0)
    {
        info += std::format("  mean = {} ns\n", mean_nanos);
        return {t.results.size(), mean, info};
    }

    // Something is strange => print 0 ns
    info += std::format("  mean = 0 ns\n");
    return {t.results.size(), mean, info};
}

Timer::Duration Timer::GetMean(Timing& t)
{
    Duration mean = Duration::zero();
    if (t.results.size() == 0) return mean;
    for (auto d : t.results)
    {
        mean += d;
    }
    return mean / t.results.size();
}

std::string Timer::GetResults()
{
    std::vector<Result> results;

    for (auto const& pair : _timings)
    {
        results.push_back(GetResult(pair.first));
    }

    std::sort(results.begin(), results.end(), Result::compare);

    std::string all_infos;
    for (Result r : results)
    {
        all_infos.append(r.info);
    };

    return all_infos;
};

}  // namespace utils
