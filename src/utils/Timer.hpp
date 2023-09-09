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
        Duration mean;
        std::string info;

        static bool compare(const Result &a, const Result &b)
        {
            return a.mean < b.mean;
        }
    };

    std::unordered_map<std::string, Timing> _timings;

    Timer() = default;
    ~Timer() = default;

    template<typename T>
    static inline int64_t DurationTo(Duration duration)
    {
        return std::chrono::duration_cast<T>(duration).count();
    };
    
    // Returns the mean and info as pretty print
    Result GetResult(const std::string& id);

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
    void Start(const std::string& id);

    // Stop the timer identified by <id>
    // Note: Start() must precede this Stop()!
    void Stop(const std::string& id);

    // Returns the mean of all samples in this timing
    Duration GetMean(Timing& t);

    // Returns all results by mean sorted as pretty print
    std::string GetResults();
};

}  // namespace utils
