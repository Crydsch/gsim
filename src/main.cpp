#include "utils/Timer.hpp"
#include "logger/Logger.hpp"
#include "sim/Simulator.hpp"
#include "sim/Config.hpp"
#include "utils/RNG.hpp"
#if ENABLE_GUI
#include "ui/UiContext.hpp"
#endif
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

std::shared_ptr<sim::Simulator> simulator{nullptr};

int run_headless() {
    SPDLOG_INFO("Launching Version {} {} in headless mode.", GSIM_VERSION, GSIM_VERSION_NAME);
    sim::Simulator::init_instance();
    simulator = sim::Simulator::get_instance();
    simulator->start_worker();

    simulator->continue_simulation(); // aka un-pause worker thread

    while (simulator->get_state() == sim::SimulatorState::RUNNING) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    simulator->stop_worker();
    sim::Simulator::destroy_instance(); // Release internal shared pointer

    return EXIT_SUCCESS;
}

int run_ui() {
#if ENABLE_GUI
    SPDLOG_INFO("Launching Version {} {} in UI mode.", GSIM_VERSION, GSIM_VERSION_NAME);
    sim::Simulator::init_instance();
    simulator = sim::Simulator::get_instance();
    simulator->start_worker();

    // The UI context manages everything that is UI related.
    // It will return once all windows have been terminated.
    ui::UiContext ui;
    int result = ui.run();

    simulator->stop_worker();
    sim::Simulator::destroy_instance(); // Release internal shared pointer

    return result;
#else
    SPDLOG_ERROR("Tried launching in UI mode, but UI support is not enabled in this build!");
    return 1;
#endif
}

int main(int argc, char** argv) {
#ifndef GSIM_SHADER_INTO_HEADER
    // Adjust working directory to find 'assets' (vulkan shader files)
    sim::Config::find_correct_working_directory();
#endif
    utils::RNG::init();

#if DEBUG
    logger::setup_logger(spdlog::level::trace);
#else
    logger::setup_logger(spdlog::level::info);
#endif
    SPDLOG_INFO(""); // Just some whitespace to
    SPDLOG_INFO(""); // have some distance from
    SPDLOG_INFO(""); // the previous run in logs.

    // Init Config
    sim::Config::args = std::vector<std::string>(argv, argv + argc);
    sim::Config::parse_args();

    int exitCode = 0;
    if (sim::Config::run_headless)
    {
        exitCode = run_headless();
    }
    else
    {
        exitCode = run_ui();
    }

#if BENCHMARK
    int64_t ticks = simulator->get_current_tick();
    std::string benchmarkResults = utils::Timer::Instance().GetSummary2(ticks);
    std::string benchmarkResultsCSV = utils::Timer::Instance().GetSummaryCSV();
    SPDLOG_INFO("Benchmarking Results:\n{}\n{}", benchmarkResults, benchmarkResultsCSV);
#endif

    return exitCode;
}
