#pragma once

#include <chrono>  // for milliseconds, duration_cast, __enable_if_is...
#include <cstdint>  // for int64_t
#include <string>  // for basic_string, hash, string
#include <unordered_map>  // for unordered_map
#include <vector>  // for vector

// By using these defines, we can remove all timing from the source code when no benchmark is wanted.
// Use '#define BENCHMARK 1' to enable it.

// clang-format off
#if BENCHMARK
#define TIMER_START(id) utils::Timer::Instance().Start(#id)
#define TIMER_STOP(id) utils::Timer::Instance().Stop(#id)
#define TIMER_START_STR(s) utils::Timer::Instance().Start(s)
#define TIMER_STOP_STR(s) utils::Timer::Instance().Stop(s)
#else
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TIMER_START(id) {}
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TIMER_STOP(id) {}
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TIMER_START_STR(s) {}
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TIMER_STOP_STR(s) {}
#endif
// clang-format on

namespace utils
{


// This class manages multiple concurrent timers for benchmarking purposes.
class Timer
{
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = Clock::duration;
    using Millis = std::chrono::milliseconds;
    using Micros = std::chrono::microseconds;
    using Nanos = std::chrono::nanoseconds;

 private:
    struct Timing
    {
        bool active = false;
        TimePoint start;
        std::vector<Duration> results;
    };

    struct Result {
        std::string id;
        size_t sample_count;
        Duration mean;
        Duration min;
        Duration max;
        Duration median;
        Duration lower_quartile;
        Duration upper_quartile;
        Duration normalized_mean;   // scaled by number of ticks
        Duration normalized_median; // scaled by number of ticks

        static bool compare_norm_mean(const Result &a, const Result &b)
        {
            return a.normalized_mean < b.normalized_mean;
        }
        static bool compare_norm_median(const Result &a, const Result &b)
        {
            return a.normalized_median < b.normalized_median;
        }
    };

    std::unordered_map<std::string, Timing> _timings;

    Timer() = default;
    ~Timer() = default;

    template<typename T>
    static inline int64_t DurationTo(Duration _duration)
    {
        return std::chrono::duration_cast<T>(_duration).count();
    };
    std::string DurationToString(const Duration& _duration);

    // Returns the mean of all samples in this timing
    Duration CalcMean(Timing& t);
    // Returns a set of calculated stats
    Result CalcResult(const std::string& _id, const float _scaling);
    // Returns a summary in formatX
    std::string GetResultSummary1(const Result& _result);
    std::string GetResultSummary2(const Result& _result);

 public:
    // Can be set to mitigate internal re-allocation (an thus influence on timings)
    static size_t num_expected_samples;

    static Timer& Instance();

    // Preserve singleton characteristics
    Timer(const Timer&) = delete;
    Timer(Timer&&) = delete;
    Timer& operator=(const Timer&) = delete;
    Timer& operator=(Timer&&) = delete;

    // Start the timer identified by <id>
    // Note: A timer with <id> must be stopped, before being started again.
    void Start(const std::string& _id);

    // Stop the timer identified by <id>
    // Note: Start() must precede this Stop()!
    void Stop(const std::string& _id);

    // Returns all results in formatX
    std::string GetSummary1(const float _scaling = 1);
    std::string GetSummary2(const float _scaling = 1);
};

/**
 *  Format1: includes: id,sample_count,mean
 *           sorted by normalized mean
 *  Format2: includes: id,sample_count,mean,(0,25,50,75,100 percentiles)
 *           sorted by normalized mean
 */

}  // namespace utils
