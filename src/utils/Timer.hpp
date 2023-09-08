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
    using string = std::string;
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

    std::unordered_map<string, Timing> _timings;

    Timer() = default;
    ~Timer() = default;

    template<typename T>
    static inline int64_t DurationTo(Duration duration)
    {
        return std::chrono::duration_cast<T>(duration).count();
    };

 public:

    // Can be set to mitigate timer internal reallocation (an thus influence on timings)
    static size_t num_expected_samples;

    static Timer& Instance();

    // Preserve singleton characteristics
    Timer(const Timer&) = delete;
    Timer(Timer&&) = delete;
    Timer& operator=(const Timer&) = delete;
    Timer& operator=(Timer&&) = delete;

    // Start the timer identified by <id>
    // Note: A timer with <id> must be stopped, before being started again.
    void Start(const string& id);

    // Stop the timer identified by <id>
    // Note: Start() must precede this Stop()!
    void Stop(const string& id);

    // Returns all timer results as pretty print.
    //  Averages timers started multiple times.
    std::string GetResult(const string& id)
    {
        Timing& t = _timings[id];

        string result;
        result += std::format("{}\n", id);
        result += std::format("  {} samples\n", t.results.size());
        
        if (t.results.size() == 0) return result;

        Duration sum = Duration::zero();
        for (auto d : t.results)
        {
            sum += d;
        }

        int64_t mean = DurationTo<Millis>(sum / t.results.size());
        if (mean > 0)
        {
            result += std::format("  mean = {} ms\n", mean);
            return result;
        } // Else resolution was too low => try again

        mean = DurationTo<Micros>(sum / t.results.size());
        if (mean > 0)
        {
            result += std::format("  mean = {} us\n", mean);
            return result;
        } // Else resolution was too low => try again

        mean = DurationTo<Nanos>(sum / t.results.size());
        if (mean > 0)
        {
            result += std::format("  mean = {} ns\n", mean);
            return result;
        }

        // Something is strange => print 0 ns
        result += std::format("  mean = {} ns\n", mean);
        return result;
    }

    std::string GetResultWithSamples(const string& id);
    string GetResults()
    {
        string result;
        for (auto const& pair : _timings)
        {
            // cppcheck-suppress useStlAlgorithm
            result += GetResult(pair.first);
        }
        return result;
    };
};

}  // namespace utils
