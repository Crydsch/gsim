#pragma once

#include "Events.hpp"
#include "GpuQuadTree.hpp"
#include "Metadata.hpp"
#include "PipeConnector.hpp"
#include "PushConsts.hpp"
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

// Note: An extra namespace is necessary to work around C++20 enums
//  We want to access the enum only via a safe namespace (ex. MyEnum::Value)
//  But also be able to cast the enum value to an integer (ex. int v = MyEnum::Value)
namespace simulator_pass_ns
{
enum simulator_pass
{
    Initialization = 0,
    Movement = 1,
    CollisionDetection = 2
};
}
typedef simulator_pass_ns::simulator_pass SimulatorPass;

constexpr float MAX_RENDER_RESOLUTION_X = 8192;  // Larger values result in errors when creating frame buffers
constexpr float MAX_RENDER_RESOLUTION_Y = 8192;

constexpr size_t QUAD_TREE_MAX_DEPTH = 8;
constexpr size_t QUAD_TREE_ENTITY_NODE_CAP = 10;

class Simulator
{
 private:
    static std::shared_ptr<Simulator> instance;

    PipeConnector* pipeConnector{nullptr};

#if MSIM_DETECT_CONTACTS_CPU_STD
    std::unordered_set<InterfaceCollision> collisions[2];
#elif MSIM_DETECT_CONTACTS_CPU_EMIL
    emilib::HashSet<InterfaceCollision> collisions[2];
#endif
#if MSIM_DETECT_CONTACTS_CPU_STD | MSIM_DETECT_CONTACTS_CPU_EMIL
    int oldColls{1};  // TODO rem and calc every tick
    int newColls{0};  // TODO rename currCollIndex
#endif

    std::unique_ptr<std::ofstream> logFile{nullptr};

    std::unique_ptr<std::thread> simThread{nullptr};
    SimulatorState state{SimulatorState::STOPPED};

    std::shared_ptr<kp::Sequence> shaderSeq{nullptr};
    std::shared_ptr<kp::Sequence> retrieveEntitiesSeq{nullptr};
    std::shared_ptr<kp::Sequence> retrieveQuadTreeNodesSeq{nullptr};
    std::shared_ptr<kp::Sequence> retrieveMetadataSeq{nullptr};
    std::shared_ptr<kp::Sequence> pushMetadataSeq{nullptr};
    std::shared_ptr<kp::Sequence> retrieveLinkEventsSeq{nullptr};

    std::mutex waitMutex{};
    std::condition_variable waitCondVar{};
    bool simulating{false};

    int64_t current_tick{};
    utils::TickDurationHistory tpsHistory{};
    utils::TickRate tps{};

    utils::TickDurationHistory updateTickHistory{};
    utils::TickDurationHistory collisionDetectionTickHistory{};

    std::shared_ptr<kp::Manager> mgr{nullptr};
    std::vector<uint32_t> shader{};
    std::shared_ptr<kp::Algorithm> algo{nullptr};
    std::vector<std::shared_ptr<kp::Tensor>> allTensors{};

    std::vector<PushConsts> pushConsts{};

    std::shared_ptr<Map> map{nullptr};
    std::shared_ptr<kp::Tensor> tensorConnections{nullptr};
    std::shared_ptr<kp::Tensor> tensorRoads{nullptr};

    // -----------------Waypoints-----------------
    std::shared_ptr<std::vector<Entity>> entities{std::make_shared<std::vector<Entity>>()};
    size_t entities_epoch_gpu{0};
    size_t entities_epoch_cpu{0};  // if not equal to entities_epoch_gpu, then out of sync
    size_t entities_epoch_last_retrieved{0};  // if equal to entities_epoch_cpu then update local copy
    std::shared_ptr<kp::Tensor> tensorEntities{nullptr};
    // ------------------------------------------

    // -----------------Waypoints-----------------
    std::shared_ptr<kp::Tensor> tensorWaypoints{nullptr};
    Waypoint* waypoints{nullptr};  // Points to raw data of <tensorWaypoints>
    // ------------------------------------------

    // -----------------QuadTree-----------------
    std::vector<gpu_quad_tree::Entity> quadTreeEntities;
    std::shared_ptr<std::vector<gpu_quad_tree::Node>> quadTreeNodes{std::make_shared<std::vector<gpu_quad_tree::Node>>()};
    std::vector<uint32_t> quadTreeNodeUsedStatus{};

    std::shared_ptr<kp::Tensor> tensorQuadTreeEntities{nullptr};
    std::shared_ptr<kp::Tensor> tensorQuadTreeNodes{nullptr};
    size_t quad_tree_nodes_epoch_gpu{0};
    size_t quad_tree_nodes_epoch_cpu{0};  // if not equal to <quad_tree_nodes_epoch_gpu>, then out of sync
    size_t quad_tree_nodes_epoch_last_retrieved{0};  // if equal to <quad_tree_nodes_epoch_cpu> then update local copy
    std::shared_ptr<kp::Tensor> tensorQuadTreeNodeUsedStatus{nullptr};
    // ------------------------------------------

    // -----------------Metadata-----------------
    std::shared_ptr<kp::Tensor> tensorMetadata{nullptr};
    Metadata* metadata{nullptr};  // Points to raw data of <tensorMetadata>
    // ------------------------------------------

    // -----------Collision Detection------------
    std::shared_ptr<kp::Tensor> tensorInterfaceCollisions{nullptr};
    InterfaceCollision* interfaceCollisions{nullptr};  // Points to raw data of <tensorInterfaceCollisions>
    std::shared_ptr<kp::Sequence> retrieveInterfaceCollisionsSeq{nullptr};
    // ------------------------------------------

    // ------------------Events------------------
    // std::shared_ptr<kp::Tensor> tensorWaypointRequestEvents{nullptr};
    // uint32_t* waypointRequestEvents{nullptr};  // Points to raw data of <tensorLinkUpEvents>
    std::shared_ptr<kp::Tensor> tensorLinkUpEvents{nullptr};
    LinkUpEvent* linkUpEvents{nullptr};  // Points to raw data of <tensorLinkUpEvents>
    std::shared_ptr<kp::Tensor> tensorLinkDownEvents{nullptr};
    LinkUpEvent* linkDownEvents{nullptr};  // Points to raw data of <tensorLinkDownEvents>
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

    void run_movement_pass();
    // Synchronizes the entities tensor from local to device memory
    void sync_entities_device();
    // Synchronizes the entities tensor from device to local memory
    void sync_entities_local();
    // Returns the current entity vector in <_out_entities>
    // Returns true if <_inout_entity_epoch> is different from the internal epoch
    //  aka the returned vector is different/updated
    // Returns the current epoch in <_inout_entity_epoch>
    // Queues a synchronization request, for the next epoch.
    //  (To be retrieved by a subsequent call)
    bool get_entities(std::shared_ptr<std::vector<Entity>>& _out_entities, size_t& _inout_entity_epoch);

    // Synchronizes the quad tree node tensor from device to local memory
    void sync_quad_tree_nodes_local();
    // Returns the current quad tree nodes vector in <_out_quad_tree_nodes>
    // Returns true if <_inout_quad_tree_nodes_epoch> is different from the internal epoch
    //  aka the returned vector is different/updated
    // Returns the current epoch in <_inout_quad_tree_nodes_epoch>
    // Queues a synchronization request, for the next epoch.
    //  (To be retrieved by a subsequent call)
    bool get_quad_tree_nodes(std::shared_ptr<std::vector<gpu_quad_tree::Node>>& _out_quad_tree_nodes, size_t& _inout_quad_tree_nodes_epoch);

    void run_collision_detection_pass();
    // Synchronizes the interface collision tensor from device to local memory
    void sync_interface_collisions_local();

    // Synchronizes the metadata tensors from device to local memory
    void sync_metadata_local();
    // Synchronizes the metadata tensors from local memory to device
    void sync_metadata_device();

    void run_interface_contacts_pass_cpu();
    void run_interface_contacts_pass_gpu();
    // Synchronizes the metadata tensors from device to local memory
    void sync_link_events_local();

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
