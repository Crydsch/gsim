#pragma once

#include "Events.hpp"
#include "GpuQuadTree.hpp"
#include "Metadata.hpp"
#include "PipeConnector.hpp"
#include "Constants.hpp"
#include "Entity.hpp"
#include "GpuBuffer.hpp"
#include "utils/TickDurationHistory.hpp"
#include "utils/TickRate.hpp"
#include "GpuAlgorithm.hpp"
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
#include <stdint.h>

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
#include <renderdoc_app.h>
#endif

namespace sim
{

enum class SimulatorState
{
    STOPPED,
    RUNNING,
    JOINING
};


class Simulator
{
 private:
    static std::shared_ptr<Simulator> instance;

    PipeConnector* connector{nullptr};

    std::unique_ptr<std::ofstream> logFile{nullptr};

    std::unique_ptr<std::thread> simThread{nullptr};
    SimulatorState state{SimulatorState::STOPPED};

    std::mutex waitMutex{};
    std::condition_variable waitCondVar{};
    bool simulating{false};

    int64_t current_tick{};
    utils::TickDurationHistory tpsHistory{};
    utils::TickRate tps{};
    std::chrono::high_resolution_clock::time_point tickStart{};

    utils::TickDurationHistory updateTickHistory{};
    utils::TickDurationHistory collisionDetectionTickHistory{};

    std::shared_ptr<kp::Manager> mgr{nullptr};
    std::shared_ptr<GpuAlgorithm> algo{nullptr};

    // ----------------Constants-----------------
    std::shared_ptr<GpuBuffer<Constants>> bufConstants{nullptr};
    // ------------------------------------------

    // -------------------Map--------------------
    std::shared_ptr<Map> map{nullptr};
    std::shared_ptr<GpuBuffer<Road>> bufMapRoads{nullptr};
    std::shared_ptr<GpuBuffer<uint32_t>> bufMapConnections{nullptr};
    // ------------------------------------------

    // -----------------Entities-----------------
    std::shared_ptr<GpuBuffer<Entity>> bufEntities{nullptr};
    bool entitiesUpdateRequested{true};
    // ------------------------------------------

    // -----------------Waypoints----------------
    std::shared_ptr<GpuBuffer<Waypoint>> bufWaypoints{nullptr};
    std::shared_ptr<GpuBuffer<WaypointRequest>> bufWaypointRequests{nullptr};
    // ------------------------------------------

    // -----------------QuadTree-----------------
    std::shared_ptr<GpuBuffer<gpu_quad_tree::Node>> bufQuadTreeNodes{nullptr};
    bool quadTreeNodesUpdateRequested{false};
    std::shared_ptr<GpuBuffer<gpu_quad_tree::Entity>> bufQuadTreeEntities{nullptr};
    // ------------------------------------------

    // -----------------Metadata-----------------
    std::shared_ptr<GpuBuffer<Metadata>> bufMetadata{nullptr};
    // ------------------------------------------

    // -----------Collision/Connectivity Detection------------
    std::shared_ptr<GpuBuffer<InterfaceCollisionBlock>> bufInterfaceCollisionsSet{nullptr};
    uint32_t bufInterfaceCollisionSetOldOffset{0};
    uint32_t bufInterfaceCollisionSetNewOffset{0};

    std::shared_ptr<GpuBuffer<LinkUpEvent>> bufLinkUpEventsList{nullptr};
    std::shared_ptr<GpuBuffer<LinkDownEvent>> bufLinkDownEventsList{nullptr};
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

    // Initializes the shared Simulator instance.
    // Note: Configuration should be set before calling this.
    static void init_instance();
    static std::shared_ptr<Simulator>& get_instance();
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

    void reset_metadata();
    void recv_entity_positions();
    void send_entity_positions();

    // Move entities in the world
    void run_movement_pass();

    // Returns the current entity data in <_out_entities>
    // Returns true if <_inout_entity_epoch> is different from the internal epoch
    //  aka the returned data is different/updated
    // Returns the current epoch in <_inout_entity_epoch>
    // Queues an update request, for the next epoch
    //  (To be retrieved by a subsequent call)
    bool get_entities(const Entity** _out_entities, size_t& _inout_entity_epoch);

    // Returns the current quad tree node data in <_out_quad_tree_nodes>
    // Returns true if <_inout_quad_tree_nodes_epoch> is different from the internal epoch
    //  aka the returned data is different/updated
    // Returns the current epoch in <_inout_quad_tree_nodes_epoch>
    // Queues a synchronization request, for the next epoch.
    //  (To be retrieved by a subsequent call)
    bool get_quad_tree_nodes(const gpu_quad_tree::Node** _out_quad_tree_nodes, size_t& _inout_quad_tree_nodes_epoch);

    // Detect connectivity
    void run_connectivity_detection_pass();

    void send_connectivity_events();

    [[nodiscard]] const std::shared_ptr<Map> get_map() const;
    [[nodiscard]] int64_t get_current_tick() const;

 private:
    void init();
    void sim_worker();
    void sim_tick();
    void check_device_queues();
    std::filesystem::path get_log_csv_path();
    void prepare_log_csv_file();
    void write_log_csv_file(int64_t tick, std::chrono::nanoseconds durationUpdate, std::chrono::nanoseconds durationCollision, std::chrono::nanoseconds durationAll);
    static std::string get_time_stamp();

    void debug_output_positions();
    std::vector<Entity> debug_output_destinations_entities{};
    void debug_output_destinations_before_move();
    void debug_output_destinations_after_move();
    void debug_output_quadtree();

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
    void
    init_renderdoc();
    void start_frame_capture();
    void end_frame_capture();
#endif
};
}  // namespace sim
