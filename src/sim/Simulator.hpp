#pragma once

#include "GpuQuadTree.hpp"
#include "PushConsts.hpp"
#include "Events.hpp"
#include "sim/Entity.hpp"
#include "utils/TickDurationHistory.hpp"
#include "utils/TickRate.hpp"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <kompute/Manager.hpp>
#include <memory>
#include <mutex>
#include <sim/Map.hpp>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
#include <renderdoc_app.h>
#endif

namespace sim {

enum class SimulatorState {
    STOPPED,
    RUNNING,
    JOINING
};

// Note: An extra namespace is necessary to work around C++20 enums
//  We want to access the enum only via a safe namespace (ex. MyEnum::Value)
//  But also be able to cast the enum value to an integer (ex. int v = MyEnum::Value)
namespace simulator_pass_ns {
	enum simulator_pass {
		Initialization = 0,
		Movement = 1,
        CollisionDetection = 2
	};
}
typedef simulator_pass_ns::simulator_pass SimulatorPass;

constexpr size_t MAX_ENTITIES = 1000;
constexpr float MAX_RENDER_RESOLUTION_X = 8192;  // Larger values result in errors when creating frame buffers
constexpr float MAX_RENDER_RESOLUTION_Y = 8192;

constexpr size_t QUAD_TREE_MAX_DEPTH = 8;
constexpr size_t QUAD_TREE_ENTITY_NODE_CAP = 10;

/**
 * Specifies the collision radius in meters.
 **/
constexpr float COLLISION_RADIUS = 10;

class Simulator {
 private:
    static std::shared_ptr<Simulator> instance;

    std::unique_ptr<std::ofstream> logFile{nullptr};

    std::unique_ptr<std::thread> simThread{nullptr};
    SimulatorState state{SimulatorState::STOPPED};

    std::mutex waitMutex{};
    std::condition_variable waitCondVar{};
    bool simulating{false};

    int64_t current_tick{};
    int64_t max_ticks{};
    utils::TickDurationHistory tpsHistory{};
    utils::TickRate tps{};

    utils::TickDurationHistory updateTickHistory{};
    utils::TickDurationHistory collisionDetectionTickHistory{};

    std::shared_ptr<kp::Manager> mgr{nullptr};
    std::vector<uint32_t> shader{};
    std::shared_ptr<kp::Algorithm> algo{nullptr};
    std::vector<std::shared_ptr<kp::Tensor>> params{};

    std::vector<PushConsts> pushConsts{};

    std::shared_ptr<std::vector<Entity>> entities{std::make_shared<std::vector<Entity>>()};
    std::shared_ptr<kp::Tensor> tensorEntities{nullptr};
    std::shared_ptr<kp::Tensor> tensorConnections{nullptr};
    std::shared_ptr<kp::Tensor> tensorRoads{nullptr};
    std::shared_ptr<kp::Tensor> tensorDebugData{nullptr};

    std::shared_ptr<Map> map{nullptr};

    // -----------------QuadTree-----------------
    std::vector<gpu_quad_tree::Entity> quadTreeEntities;
    std::shared_ptr<std::vector<gpu_quad_tree::Node>> quadTreeNodes{std::make_shared<std::vector<gpu_quad_tree::Node>>()};
    std::vector<uint32_t> quadTreeNodeUsedStatus;

    std::shared_ptr<kp::Tensor> tensorQuadTreeEntities{nullptr};
    std::shared_ptr<kp::Tensor> tensorQuadTreeNodes{nullptr};
    std::shared_ptr<kp::Tensor> tensorQuadTreeNodeUsedStatus{nullptr};
    // ------------------------------------------

    // ------------------Events------------------
    std::vector<EventMetadata> eventMetadata;
    std::vector<LinkStateEvent> linkUpEvents;
    std::vector<LinkStateEvent> linkDownEvents;

    std::shared_ptr<kp::Tensor> tensorEventMetadata{nullptr};
    std::shared_ptr<kp::Tensor> tensorLinkUpEvents{nullptr};
    std::shared_ptr<kp::Tensor> tensorLinkDownEvents{nullptr};
    // ------------------------------------------

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
    RENDERDOC_API_1_5_0* rdocApi{nullptr};
#endif

 public:
    Simulator();
    ~Simulator();

    Simulator(Simulator&&) = delete;
    Simulator(const Simulator&) = delete;
    Simulator& operator=(Simulator&&) = delete;
    Simulator& operator=(const Simulator&) = delete;

    static std::shared_ptr<Simulator>& get_instance(int64_t _max_ticks = -1);
    // This method must be called in order to release all internal shared pointer
    static void destroy_instance();
    [[nodiscard]] SimulatorState get_state() const;
    void start_worker();
    void stop_worker();

    void continue_simulation();
    void pause_simulation();
    [[nodiscard]] bool is_simulating() const;
    [[nodiscard]] const utils::TickRate& get_tps() const;
    [[nodiscard]] const utils::TickDurationHistory& get_tps_history() const;
    [[nodiscard]] const utils::TickDurationHistory& get_update_tick_history() const;
    [[nodiscard]] const utils::TickDurationHistory& get_collision_detection_tick_history() const;
    std::shared_ptr<std::vector<Entity>> get_entities();
    std::shared_ptr<std::vector<gpu_quad_tree::Node>> get_quad_tree_nodes();
    [[nodiscard]] const std::shared_ptr<Map> get_map() const;

 private:
    void init(int64_t _max_ticks);
    void sim_worker();
    void sim_tick(std::shared_ptr<kp::Sequence>& calcSeq, 
        std::shared_ptr<kp::Sequence>& retrieveEntitiesSeq, 
        std::shared_ptr<kp::Sequence>& retrieveQuadTreeNodesSeq, 
        std::shared_ptr<kp::Sequence>& retrieveEventsSeq,
        std::shared_ptr<kp::Sequence>& pushEventMetadataSeq,
        std::shared_ptr<kp::Sequence>& retrieveMiscSeq);
    void add_entities();
    void check_device_queues();
    static const std::filesystem::path& get_log_csv_path();
    void prepare_log_csv_file();
    void write_log_csv_file(int64_t tick, std::chrono::nanoseconds durationUpdate, std::chrono::nanoseconds durationCollision, std::chrono::nanoseconds durationAll);
    static std::string get_time_stamp();

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
    void
    init_renderdoc();
    void start_frame_capture();
    void end_frame_capture();
#endif
};
}  // namespace sim
