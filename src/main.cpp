#include "utils/Timer.hpp"
#include "logger/Logger.hpp"
#include "sim/Simulator.hpp"
#include "ui/UiContext.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

// TODO combine these functions into 'struct config parse_arguments(args)'
bool should_run_headless(const std::vector<std::string>& args) {
    for (std::string arg : args)
    {
        if (arg == "--headless")
        {
            return true;
        }
    }
    return false;
}

int64_t parse_max_ticks(const std::vector<std::string>& args) {
    for (std::size_t i = 0; i < args.size(); ++i)
    {
        std::string arg = args[i];
        if (arg == "--max-ticks")
        {
            i++;
            return std::stol(args[i]);
        }
    }
    return -1;
}

int run_headless(int64_t _max_ticks) {
    SPDLOG_INFO("Launching Version {} {} in headless mode.", MOVEMENT_SIMULATOR_VERSION, MOVEMENT_SIMULATOR_VERSION_NAME);
    std::shared_ptr<sim::Simulator> simulator = sim::Simulator::get_instance(_max_ticks);
    simulator->start_worker();

    simulator->continue_simulation(); // aka un-pause worker thread

    while (simulator->get_state() == sim::SimulatorState::RUNNING) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    simulator->stop_worker();
    sim::Simulator::destroy_instance(); // Release internal shared pointer
    
    return EXIT_SUCCESS;
}

int run_ui(int _argc, char** _argv, int64_t _max_ticks) {
    SPDLOG_INFO("Launching Version {} {} in UI mode.", MOVEMENT_SIMULATOR_VERSION, MOVEMENT_SIMULATOR_VERSION_NAME);
    std::shared_ptr<sim::Simulator> simulator = sim::Simulator::get_instance(_max_ticks);
    simulator->start_worker();

    // The UI context manages everything that is UI related.
    // It will return once all windows have been terminated.
    ui::UiContext ui;
    int result = ui.run(_argc, _argv);

    simulator->stop_worker();
    return result;
}

int main(int argc, char** argv) {
    logger::setup_logger(spdlog::level::debug);
    SPDLOG_INFO(""); // Just some whitespace
    SPDLOG_INFO(""); // to have some distance
    SPDLOG_INFO(""); // from the previous run.

    std::vector<std::string> args(argv, argv + argc);

    bool headless = should_run_headless(args);
    int64_t max_ticks = parse_max_ticks(args);

    int exitCode = 0;
    if (headless)
    {
        exitCode = run_headless(config);
    }
    else
    {
        exitCode = run_ui(argc, argv, config);
    }

#if BENCHMARK
    std::string benchmarkResults = utils::Timer::Instance().GetResults();
    SPDLOG_INFO("Benchmarking Results:\n{}", benchmarkResults);
#endif

    return exitCode;
}
