#include "logger/Logger.hpp"
#include "sim/Simulator.hpp"
#include "ui/UiContext.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

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

int run_headless() {
    SPDLOG_INFO("Launching Version {} {} in headless mode.", MOVEMENT_SIMULATOR_VERSION, MOVEMENT_SIMULATOR_VERSION_NAME);
    std::shared_ptr<sim::Simulator> simulator = sim::Simulator::get_instance();
    simulator->start_worker();

    simulator->continue_simulation();
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    simulator->stop_worker();
    return EXIT_SUCCESS;
}

int run_ui(int argc, char** argv) {
    SPDLOG_INFO("Launching Version {} {} in UI mode.", MOVEMENT_SIMULATOR_VERSION, MOVEMENT_SIMULATOR_VERSION_NAME);
    std::shared_ptr<sim::Simulator> simulator = sim::Simulator::get_instance();
    simulator->start_worker();

    // The UI context manages everything that is UI related.
    // It will return once all windows have been terminated.
    ui::UiContext ui;
    int result = ui.run(argc, argv);

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

    if (headless) {
        return run_headless();
    }
    return run_ui(argc, argv);
}