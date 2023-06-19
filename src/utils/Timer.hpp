#pragma once

#include <chrono>  // for milliseconds, duration_cast, __enable_if_is...
#include <cstdint>  // for int64_t
#include <string>  // for basic_string, hash, string
#include <unordered_map>  // for unordered_map
#include <vector>  // for vector

// By using these defines, we can remove all timing from the source code when no benchmark is wanted.
// Use '#define BENCHMARK 1' to enable it.
#if BENCHMARK
#define TIMER_START(id) \
    utils::Timer::Instance().Start(#id)
#define TIMER_STOP(id) utils::Timer::Instance().Stop(#id)
#else
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TIMER_START(id) {}
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define TIMER_STOP(id) {}
#endif

namespace utils {

// This class manages multiple concurrent timers for benchmarking purposes.
class Timer {
    using string = std::string;
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = Clock::duration;

 private:
    struct Timing {
        bool active = false;
        TimePoint start;
        std::vector<Duration> results;
    };

    std::unordered_map<string, Timing> _timings;

    Timer() = default;
    ~Timer() = default;

    static inline int64_t DurationToMillis(Duration duration) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    };

 public:
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
    //  Includes averages of timers started multiple times.
    string GetResult(const string& id);
    string GetResults();
};

}  // namespace GBS_TESTER
