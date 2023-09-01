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

#if MSIM_DETECT_CONTACTS_CPU_STD
#include <unordered_set>
#elif MSIM_DETECT_CONTACTS_CPU_EMIL
#include "3rdparty/emilib/hash_set.hpp"
#endif

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

constexpr float MAX_RENDER_RESOLUTION_X = 8192;  // Larger values result in errors when creating frame buffers
constexpr float MAX_RENDER_RESOLUTION_Y = 8192;

constexpr size_t QUAD_TREE_MAX_DEPTH = 8;
constexpr size_t QUAD_TREE_ENTITY_NODE_CAP = 10;

class Simulator
{
 private:
    static std::shared_ptr<Simulator> instance;

    PipeConnector* connector{nullptr};

#if MSIM_DETECT_CONTACTS_CPU_STD
    std::unordered_set<InterfaceCollision> collisions[2];
    int currCollIndex{0};
#elif MSIM_DETECT_CONTACTS_CPU_EMIL
    emilib::HashSet<InterfaceCollision> collisions[2];
    int currCollIndex{0};
#endif

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
    // ------------------------------------------

    // -----------------QuadTree-----------------
    std::shared_ptr<GpuBuffer<gpu_quad_tree::Node>> bufQuadTreeNodes{nullptr};
    bool quadTreeNodesUpdateRequested{false};
    std::shared_ptr<GpuBuffer<uint32_t>> bufQuadTreeNodeUsedStatus{nullptr};
    std::shared_ptr<GpuBuffer<gpu_quad_tree::Entity>> bufQuadTreeEntities{nullptr};
    // ------------------------------------------

    // -----------------Metadata-----------------
    std::shared_ptr<GpuBuffer<Metadata>> bufMetadata{nullptr};
    // ------------------------------------------

    // -----------Collision Detection------------
    std::shared_ptr<GpuBuffer<InterfaceCollision>> bufInterfaceCollisions{nullptr};
    // ------------------------------------------

    // ------------------Events------------------
    std::shared_ptr<GpuBuffer<WaypointRequest>> bufWaypointRequests{nullptr};
    std::shared_ptr<GpuBuffer<LinkUpEvent>> bufLinkUpEvents{nullptr};
    std::shared_ptr<GpuBuffer<LinkDownEvent>> bufLinkDownEvents{nullptr};
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

    void run_collision_detection_pass();

    void run_interface_contacts_pass();

    void run_interface_contacts_pass_cpu();
    void run_interface_contacts_pass_gpu();

    [[nodiscard]] const std::shared_ptr<Map> get_map() const;

 private:
    void init();
    void sim_worker();
    void sim_tick();
    void check_device_queues();
    std::filesystem::path get_log_csv_path();
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
