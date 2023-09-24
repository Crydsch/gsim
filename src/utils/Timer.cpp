#include "Timer.hpp"

#include <cstdlib>  // for abort
#include <iostream>  // for operator<<, basic_ostream, endl, cout, ostream
#include <ratio>  // for ratio
#include <utility>  // for pair
#include <cmath>
#include <fmt/core.h>

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

Timer::Duration Timer::CalcMean(Timing& t)
{
    Duration mean = Duration::zero();
    if (t.results.size() == 0) return mean;
    for (auto d : t.results)
    {
        mean += d;
    }
    return mean / t.results.size();
}

Timer::Result Timer::CalcResult(const std::string& _id, const float _scaling)
{
    Timing& t = _timings[_id];

    Result result;
    result.id = _id;
    result.sample_count = t.results.size();

    if (result.sample_count == 0) return result;
    
    result.mean = CalcMean(t);
    // cost per tick relative to entire runtime
    result.normalized_mean = std::chrono::duration_cast<Duration>(result.mean * (result.sample_count / _scaling));

    std::sort(t.results.begin(), t.results.end());
    auto percentile = [t](uint8_t _p) -> Duration {
        // Ref.: https://www.calculatorsoup.com/calculators/statistics/percentile-calculator.php
        float rankf = (_p / 100.0f) * (t.results.size() - 1);
        float ranki;
        rankf = std::modf(rankf, &ranki);
        return std::chrono::duration_cast<Duration>(t.results[ranki] + rankf * (t.results[ranki+1] - t.results[ranki]));
    };

    result.min = t.results.front();
    result.max = t.results.back();
    result.median = percentile(50);
    result.lower_quartile = percentile(25);
    result.upper_quartile = percentile(75);
    // cost per tick relative to entire runtime
    result.normalized_median = std::chrono::duration_cast<Duration>(result.median * (result.sample_count / _scaling));

    return result;
}

std::string Timer::DurationToString(const Timer::Duration& _duration)
{
    int64_t dur = DurationTo<Millis>(_duration);
    if (dur > 0)
    {
        return fmt::format("{} ms", dur);
    }  // Else resolution too low => try again

    dur = DurationTo<Micros>(_duration);
    if (dur > 0)
    {
        return fmt::format("{} us", dur);
    }  // Else resolution too low => try again

    dur = DurationTo<Nanos>(_duration);
    if (dur > 0)
    {
        return fmt::format("{} ns", dur);
    }  // Else resolution too low => try again

    // Something is strange => print 0 ns
    return fmt::format("0 ns");
}

std::string Timer::GetResultSummary1(const Result& _result) {
    std::string s;
    s += fmt::format("{}\n", _result.id);
    s += fmt::format("  {} samples\n", _result.sample_count);

    if (_result.sample_count == 0) return s;

    s += fmt::format("  mean: {}   norm_mean: {}\n", DurationToString(_result.mean), DurationToString(_result.normalized_mean));

    return s;
}

std::string Timer::GetResultSummary2(const Result& _result) {
    std::string s;
    s += fmt::format("{}\n", _result.id);
    s += fmt::format("  {} samples\n", _result.sample_count);

    if (_result.sample_count == 0) return s;

    s += fmt::format("  mean: {}   norm_mean: {}\n", DurationToString(_result.mean), DurationToString(_result.normalized_mean));

    s += fmt::format("  {}  [{}, {}, {}]  {}\n", 
            DurationToString(_result.min),
            DurationToString(_result.lower_quartile),
            DurationToString(_result.median),
            DurationToString(_result.upper_quartile),
            DurationToString(_result.max));

    return s;
}

std::string Timer::GetSummary1(const float _scaling)
{
    std::vector<Result> results;

    for (auto const& pair : _timings)
    {
        results.push_back(CalcResult(pair.first, _scaling));
    }

    std::sort(results.begin(), results.end(), Result::compare_norm_mean);

    std::string all_infos;
    for (Result r : results)
    {
        all_infos.append(GetResultSummary1(r));
    }

    return all_infos;
};

std::string Timer::GetSummary2(const float _scaling)
{
    std::vector<Result> results;

    for (auto const& pair : _timings)
    {
        Result r = CalcResult(pair.first, _scaling);
        if (r.sample_count > 1)
        { // Exclude initialization timings
            results.push_back(r);
        }
    }

    std::sort(results.begin(), results.end(), Result::compare_norm_mean);

    std::string all_infos;
    for (Result r : results)
    {
        all_infos.append(GetResultSummary2(r));
    }

    return all_infos;
};

}  // namespace utils
