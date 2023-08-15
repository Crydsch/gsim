#include "Simulator.hpp"
#include "Config.hpp"
#include "kompute/Manager.hpp"
#include "kompute/Tensor.hpp"
#include "logger/Logger.hpp"
#include "sim/Entity.hpp"
#include "sim/GpuQuadTree.hpp"
#include "sim/Map.hpp"
#include "sim/PushConsts.hpp"
#include "spdlog/spdlog.h"
#include "utils/RNG.hpp"
#include "vulkan/vulkan_enums.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <kompute/operations/OpTensorSyncDevice.hpp>
#include <kompute/operations/OpTensorSyncLocal.hpp>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <bits/chrono.h>

#ifdef MOVEMENT_SIMULATOR_SHADER_INTO_HEADER
#include "fall.hpp"
#include "random_move.hpp"
#else
#include "sim/shader/utils/Utils.hpp"
#endif

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
#include <renderdoc_app.h>
#endif

namespace sim
{

// Static variables
std::shared_ptr<Simulator> Simulator::instance{nullptr};

Simulator::Simulator()
{
    prepare_log_csv_file();
}

Simulator::~Simulator()
{
    assert(logFile);
    logFile->close();
    logFile = nullptr;

    delete pipeConnector;
    pipeConnector = nullptr;
}

void Simulator::init()
{
#if STANDALONE_MODE
    SPDLOG_INFO("Simulation thread initializing (Standalone mode).");

    // Load map
    if (Config::map_filepath.empty())
    {
        throw std::runtime_error("No map configured. A map is required in standalone mode.");
    }
    map = Map::load_from_file(Config::map_filepath);

#ifdef MOVEMENT_SIMULATOR_SHADER_INTO_HEADER
    // load shader from headerfile
    shader = std::vector(STANDALONE_COMP_SPV.begin(), STANDALONE_COMP_SPV.end());
#else
    // load shader from filesystem
    shader = load_shader(Config::working_directory() / "assets/shader/vulkan/standalone.comp.spv");
#endif  // MOVEMENT_SIMULATOR_SHADER_INTO_HEADER

#else  // accelerator mode
    SPDLOG_INFO("Simulation thread initializing (Accelerator mode).");

    pipeConnector = new PipeConnector();

    // Initialization
    Header header = pipeConnector->read_header();
    if (header != Header::Initialize)
    {
        throw std::runtime_error("Simulator::init(): First request in accelerator mode was not of type 'Initialize'.");
    }

    Config::args = pipeConnector->read_config_args();
    Config::parse_args();

    // We load a map only for rendering purposes
    if (!Config::run_headless)
    {
        if (Config::map_filepath.empty())
        {
            SPDLOG_WARN("Simulator running with GUI, but no map configured. (Only required for visualization)");
        }
        else
        {
            map = Map::load_from_file(Config::map_filepath);
        }
    }

#ifdef MOVEMENT_SIMULATOR_SHADER_INTO_HEADER
    // load shader from headerfile
    shader = std::vector(ACCELERATOR_COMP_SPV.begin(), ACCELERATOR_COMP_SPV.end());
#else
    // load shader from filesystem
    shader = load_shader(Config::working_directory() / "assets/shader/vulkan/accelerator.comp.spv");
#endif  // MOVEMENT_SIMULATOR_SHADER_INTO_HEADER
#endif  // STANDALONE_MODE

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
    // Init RenderDoc:
    init_renderdoc();
#endif

    // kompute
    mgr = std::make_shared<kp::Manager>();

    // Entities
    std::vector<Entity> entities_init;  // Only for initialization
    entities_init.resize(Config::num_entities);
    tensorEntities = mgr->tensor(entities_init.data(), entities_init.size(), sizeof(Entity), kp::Tensor::TensorDataTypes::eUnsignedInt);
    entities = tensorEntities->data<Entity>();  // We access the tensor data directly without extra copy
#if STANDALONE_MODE
    // Initialize Entities
    for (size_t i = 0; i < Config::num_entities; i++)
    {
        const unsigned int roadIndex = map->get_random_road_index();
        assert(roadIndex < map->roads.size());
        const Road road = map->roads[roadIndex];
        entities[i] = Entity(utils::RNG::random_color(),
                                      Vec2(road.start.pos),
                                      Vec2(road.end.pos),
                                      utils::RNG::random_vec4u(),
                                      roadIndex);
    }
#else // accelerator mode
    // Receive initial positions
    for (size_t i = 0; i < Config::num_entities; i++)
    {
        Vec2 pos = pipeConnector->read_vec2();
        assert(pos.x > 0.0f);
        assert(pos.y > 0.0f);
        assert(pos.x < Config::map_width);
        assert(pos.y < Config::map_height);
        entities[i].pos = pos;
    }
#endif // STANDALONE_MODE

#if STANDALONE_MODE
    // Map
    tensorRoads = mgr->tensor(map->roads.data(), map->roads.size(), sizeof(Road), kp::Tensor::TensorDataTypes::eUnsignedInt);
    tensorConnections = mgr->tensor(map->connections.data(), map->connections.size(), sizeof(uint32_t), kp::Tensor::TensorDataTypes::eUnsignedInt);
#else
    // Waypoints
    std::vector<Waypoint> waypoints_init;  // Only for initialization
    waypoints_init.resize(Config::waypoint_buffer_size * Config::num_entities);
    tensorWaypoints = mgr->tensor(waypoints_init.data(), waypoints_init.size(), sizeof(Waypoint), kp::Tensor::TensorDataTypes::eUnsignedInt);
    waypoints = tensorWaypoints->data<Waypoint>();  // We access the tensor data directly without extra copy
#endif  // STANDALONE_MODE

    // Quad Tree
    // Sanity checks
    static_assert(sizeof(gpu_quad_tree::Entity) == sizeof(uint32_t) * 5, "Quad Tree entity size does not match. Expected to be constructed out of 5 uint32_t.");
    assert(gpu_quad_tree::calc_node_count(1) == 1);
    assert(gpu_quad_tree::calc_node_count(2) == 5);
    assert(gpu_quad_tree::calc_node_count(3) == 21);
    assert(gpu_quad_tree::calc_node_count(4) == 85);
    assert(gpu_quad_tree::calc_node_count(8) == 21845);

    std::vector<gpu_quad_tree::Entity> quadTreeEntities_init;  // Only for initialization
    quadTreeEntities_init.resize(Config::num_entities);
    tensorQuadTreeEntities = mgr->tensor(quadTreeEntities_init.data(), quadTreeEntities_init.size(), sizeof(gpu_quad_tree::Entity), kp::Tensor::TensorDataTypes::eUnsignedInt);

    std::vector<gpu_quad_tree::Node> quadTreeNodes_init{};  // Only for initialization
    quadTreeNodes_init.resize(gpu_quad_tree::calc_node_count(QUAD_TREE_MAX_DEPTH));
    gpu_quad_tree::init_node_zero(quadTreeNodes_init[0], Config::map_width, Config::map_height);
    tensorQuadTreeNodes = mgr->tensor(quadTreeNodes_init.data(), quadTreeNodes_init.size(), sizeof(gpu_quad_tree::Node), kp::Tensor::TensorDataTypes::eUnsignedInt);

    std::vector<uint32_t> quadTreeNodeUsedStatus_init{};  // Only for initialization
    quadTreeNodeUsedStatus_init.resize(quadTreeNodes_init.size() + 2);  // +2 since one is used as lock and one as next pointer
    quadTreeNodeUsedStatus_init[1] = 2;  // Pointer to the first free node index;
    tensorQuadTreeNodeUsedStatus = mgr->tensor(quadTreeNodeUsedStatus_init.data(), quadTreeNodeUsedStatus_init.size(), sizeof(uint32_t), kp::Tensor::TensorDataTypes::eUnsignedInt);

    // Metadata
    Metadata metadata_init{};  // Only for initialization
    tensorMetadata = mgr->tensor(&metadata_init, 1, sizeof(Metadata), kp::Tensor::TensorDataTypes::eUnsignedInt);
    metadata = tensorMetadata->data<Metadata>();  // We access the tensor data directly without extra copy

    // Collision Detection
    std::vector<InterfaceCollision> collisions_init;  // Only for initialization
    collisions_init.resize(Config::max_interface_collisions);
    tensorInterfaceCollisions = mgr->tensor(collisions_init.data(), collisions_init.size(), sizeof(InterfaceCollision), kp::Tensor::TensorDataTypes::eUnsignedInt);
    interfaceCollisions = tensorInterfaceCollisions->data<InterfaceCollision>();  // We access the tensor data directly without extra copy

    // Events
#if not STANDALONE_MODE
    std::vector<WaypointRequest> waypointRequests_init;  // Only for initialization
    waypointRequests_init.resize(Config::num_entities);
    tensorWaypointRequests = mgr->tensor(waypointRequests_init.data(), waypointRequests_init.size(), sizeof(WaypointRequest), kp::Tensor::TensorDataTypes::eUnsignedInt);
    waypointRequests = tensorWaypointRequests->data<WaypointRequest>();  // We access the tensor data directly without extra copy
#endif

    std::vector<LinkUpEvent> linkUpEvents_init;  // Only for initialization
    linkUpEvents_init.resize(Config::max_link_events);
    tensorLinkUpEvents = mgr->tensor(linkUpEvents_init.data(), linkUpEvents_init.size(), sizeof(LinkUpEvent), kp::Tensor::TensorDataTypes::eUnsignedInt);
    linkUpEvents = tensorLinkUpEvents->data<LinkUpEvent>();  // We access the tensor data directly without extra copy

    std::vector<LinkDownEvent> linkDownEvents_init;  // Only for initialization
    linkDownEvents_init.resize(Config::max_link_events);
    tensorLinkDownEvents = mgr->tensor(linkDownEvents_init.data(), linkDownEvents_init.size(), sizeof(LinkDownEvent), kp::Tensor::TensorDataTypes::eUnsignedInt);
    linkDownEvents = tensorLinkDownEvents->data<LinkDownEvent>();  // We access the tensor data directly without extra copy

    // Attention: The order in which tensors are specified for the shader, MUST be equivalent to the order in the shader (layout binding order)
#if STANDALONE_MODE
    allTensors = {
        tensorEntities,
        tensorConnections,
        tensorRoads,
        tensorQuadTreeNodes,
        tensorQuadTreeEntities,
        tensorQuadTreeNodeUsedStatus,
        tensorMetadata,
        tensorInterfaceCollisions,
        tensorLinkUpEvents,
        tensorLinkDownEvents};
#else
    allTensors = {
        tensorEntities,
        tensorWaypoints,
        tensorQuadTreeNodes,
        tensorQuadTreeEntities,
        tensorQuadTreeNodeUsedStatus,
        tensorMetadata,
        tensorInterfaceCollisions,
        tensorWaypointRequests,
        tensorLinkUpEvents,
        tensorLinkDownEvents};
#endif  // STANDALONE_MODE

    // Push constants
    pushConsts.emplace_back();  // Note: The vector size must stay fixed after algo is created
    pushConsts[0].worldSizeX = Config::map_width;
    pushConsts[0].worldSizeY = Config::map_height;
    pushConsts[0].nodeCount = static_cast<uint32_t>(quadTreeNodes_init.size());
    pushConsts[0].maxDepth = QUAD_TREE_MAX_DEPTH;
    pushConsts[0].entityNodeCap = QUAD_TREE_ENTITY_NODE_CAP;
    pushConsts[0].collisionRadius = Config::collision_radius;
    pushConsts[0].pass = SimulatorPass::Initialization;
    pushConsts[0].waypointBufferSize = Config::waypoint_buffer_size;
    pushConsts[0].waypointBufferThreshold = Config::waypoint_buffer_threshold;

    algo = mgr->algorithm<float, PushConsts>(allTensors, shader, {}, {}, {pushConsts});

    // Check gpu capabilities
    check_device_queues();

#if not STANDALONE_MODE
    pipeConnector->write_header(Header::Ok);
    pipeConnector->flush_output();
#endif
}

void Simulator::init_instance()
{
    if (instance)
    {
        throw std::runtime_error("Simulator::init_instance called twice. Instance already exists.");
    }
    instance = std::make_shared<Simulator>();
    instance->init();
}

std::shared_ptr<Simulator>& Simulator::get_instance()
{
    if (!instance)
    {
        throw std::runtime_error("Simulator::get_instance called before Simulator::init_instance. Instance does not yet exist.");
    }
    return instance;
}

void Simulator::destroy_instance()
{
    if (instance)
    {
        instance.reset();
    }
}

SimulatorState Simulator::get_state() const
{
    return state;
}

bool Simulator::get_entities(std::vector<Entity>& _inout_entities, size_t& _inout_entity_epoch)
{
    entities_update_requested = true;

    if (entities_epoch_cpu == _inout_entity_epoch) {
        // already up-to-date
        return false;
    }
    // else entity data in tensor is newer

    // return copy
    _inout_entities = tensorEntities->vector<Entity>();
    _inout_entity_epoch = entities_epoch_cpu;

    return true;
}

void Simulator::sync_waypoints_device()
{
    if (!pushWaypointsSeq->isRunning())
    {
        pushWaypointsSeq->evalAsync();
    }

    pushWaypointsSeq->evalAwait();
}

void Simulator::sync_quad_tree_nodes_local()
{
    if (quad_tree_nodes_epoch_cpu == quad_tree_nodes_epoch_gpu)
    {
        return;  // already up-to-date
    }
    assert(quad_tree_nodes_epoch_gpu > quad_tree_nodes_epoch_cpu);

    if (!retrieveQuadTreeNodesSeq->isRunning())
    {
        retrieveQuadTreeNodesSeq->evalAsync();
    }

    retrieveQuadTreeNodesSeq->evalAwait();
    quad_tree_nodes_epoch_cpu = quad_tree_nodes_epoch_gpu;
    quad_tree_nodes_updates_requested = false;
}

bool Simulator::get_quad_tree_nodes(std::vector<gpu_quad_tree::Node>& _inout_quad_tree_nodes, size_t& _inout_quad_tree_nodes_epoch)
{
    quad_tree_nodes_updates_requested = true;

    if (quad_tree_nodes_epoch_cpu == _inout_quad_tree_nodes_epoch) {
        // already up-to-date
        return false;
    }
    // else entity data in tensor is newer

    // return copy
    _inout_quad_tree_nodes = tensorQuadTreeNodes->vector<gpu_quad_tree::Node>();
    _inout_quad_tree_nodes_epoch = quad_tree_nodes_epoch_cpu;

    return true;
}

void Simulator::run_collision_detection_pass()
{
    pushConsts[0].pass = SimulatorPass::CollisionDetection;
    SPDLOG_DEBUG("Tick {}: Collision detection pass started.", current_tick);
    std::chrono::high_resolution_clock::time_point collisionDetectionTickStart = std::chrono::high_resolution_clock::now();
    shaderSeq->evalAsync<kp::OpAlgoDispatch>(algo, pushConsts);
    shaderSeq->evalAwait();
    std::chrono::nanoseconds durationCollisionDetection = std::chrono::high_resolution_clock::now() - collisionDetectionTickStart;
    collisionDetectionTickHistory.add_time(durationCollisionDetection);
    SPDLOG_DEBUG("Tick {}: Collision detection pass ended.", current_tick);
}

void Simulator::sync_interface_collisions_local()
{
    if (!retrieveInterfaceCollisionsSeq->isRunning())
    {
        retrieveInterfaceCollisionsSeq->evalAsync();
    }

    retrieveInterfaceCollisionsSeq->evalAwait();
}

void Simulator::sync_metadata_local()
{
    if (!retrieveMetadataSeq->isRunning())
    {
        retrieveMetadataSeq->evalAsync();
    }

    retrieveMetadataSeq->evalAwait();
}

void Simulator::sync_metadata_device()
{
    if (!pushMetadataSeq->isRunning())
    {
        pushMetadataSeq->evalAsync();
    }

    pushMetadataSeq->evalAwait();
}

void Simulator::sync_link_events_local()
{
    if (!retrieveLinkEventsSeq->isRunning())
    {
        retrieveLinkEventsSeq->evalAsync();
    }

    retrieveLinkEventsSeq->evalAwait();
}

const std::shared_ptr<Map> Simulator::get_map() const
{
    return map;
}

void Simulator::run_movement_pass()
{
    pushConsts[0].pass = SimulatorPass::Movement;
    SPDLOG_DEBUG("Tick {}: Movement pass started.", current_tick);
    std::chrono::high_resolution_clock::time_point updateTickStart = std::chrono::high_resolution_clock::now();
    shaderSeq->evalAsync<kp::OpAlgoDispatch>(algo, pushConsts);
    shaderSeq->evalAwait();
    std::chrono::nanoseconds durationUpdate = std::chrono::high_resolution_clock::now() - updateTickStart;
    updateTickHistory.add_time(durationUpdate);
    SPDLOG_DEBUG("Tick {}: Movement pass ended.", current_tick);

    entities_epoch_gpu++;
    quad_tree_nodes_epoch_gpu++;
}

void Simulator::sync_entities_device()
{
    if (entities_epoch_cpu == entities_epoch_gpu)
    {
        return;  // tensor is up-to-date
    }
    assert(entities_epoch_cpu > entities_epoch_gpu);

    if (!pushEntitiesSeq->isRunning())
    {
        pushEntitiesSeq->evalAsync();
    }

    pushEntitiesSeq->evalAwait();
    entities_epoch_gpu = entities_epoch_cpu;
}

void Simulator::sync_entities_local()
{
    if (entities_epoch_cpu == entities_epoch_gpu)
    {
        return;  // tensor is up-to-date
    }
    assert(entities_epoch_gpu > entities_epoch_cpu);

    if (!retrieveEntitiesSeq->isRunning())
    {
        retrieveEntitiesSeq->evalAsync();
    }

    retrieveEntitiesSeq->evalAwait();
    entities_epoch_cpu = entities_epoch_gpu;
    entities_update_requested = false;
}

void Simulator::run_interface_contacts_pass_cpu()
{
    SPDLOG_DEBUG("Tick {}: Interface contacts pass started.", current_tick);
    assert(metadata[0].linkUpEventCount == 0);
    assert(metadata[0].linkDownEventCount == 0);

    for (std::size_t i = 0; i < metadata[0].interfaceCollisionCount; i++)
    {
        const InterfaceCollision& collision = interfaceCollisions[i];

        [[maybe_unused]] auto res = collisions[newColls].insert(collision);
        assert(res.second);  // No duplicates!

        if (!collisions[oldColls].erase(collision))
        {
            // The connection was not up, but is now up
            //  => link came up
            uint32_t slot = metadata[0].linkUpEventCount++;

            // add event to tensor
            linkUpEvents[slot].ID0 = collision.ID0;
            linkUpEvents[slot].ID1 = collision.ID1;
        }
        // else connection was up and is still up
        //  => nothing changed
    }

    // We removed all connections which stayed up
    // Any remaining ones must have gone down
    metadata[0].linkDownEventCount = collisions[oldColls].size();

    uint32_t slot = 0;
    for (auto collision : collisions[oldColls])
    {
        // add event to tensor
        linkDownEvents[slot].ID0 = collision.ID0;
        linkDownEvents[slot].ID1 = collision.ID1;
        slot++;
    }
    assert(slot == metadata[0].linkDownEventCount);

    collisions[oldColls].clear();

    // Debug info // TODO rem
    SPDLOG_DEBUG(">>> links up:   {}", metadata[0].linkUpEventCount);
    SPDLOG_DEBUG(">>> links down: {}", metadata[0].linkDownEventCount);

    oldColls ^= 0x1;  // Swap sets
    newColls ^= 0x1;

    SPDLOG_DEBUG("Tick {}: Interface contacts ended.", current_tick);
}

void Simulator::run_interface_contacts_pass_gpu()
{
    SPDLOG_DEBUG("Tick {}: Interface contacts pass started.", current_tick);

    // TODO implement gpu-side interface contact detection

    SPDLOG_DEBUG("Tick {}: Interface contacts ended.", current_tick);
}

void Simulator::start_worker()
{
    assert(state == SimulatorState::STOPPED);
    assert(!simThread);

    SPDLOG_INFO("Starting simulation thread...");
    state = SimulatorState::RUNNING;
    simThread = std::make_unique<std::thread>(&Simulator::sim_worker, this);
}

void Simulator::stop_worker()
{
    assert(state != SimulatorState::STOPPED);
    assert(simThread);

    SPDLOG_INFO("Stopping simulation thread...");
    state = SimulatorState::JOINING;
    waitCondVar.notify_all();
    if (simThread->joinable())
    {
        simThread->join();
    }
    simThread.reset();
    state = SimulatorState::STOPPED;
    SPDLOG_INFO("Simulation thread stopped.");
}

void Simulator::sim_worker()
{
    SPDLOG_INFO("Simulation thread started.");

    // Initialize all tensors on the GPU
    std::shared_ptr<kp::Sequence> initSeq = mgr->sequence()->record<kp::OpTensorSyncDevice>(allTensors);
    initSeq->eval();

    // Prepare sequences
    shaderSeq = mgr->sequence();
    pushEntitiesSeq = mgr->sequence()->record<kp::OpTensorSyncDevice>({tensorEntities});
    retrieveEntitiesSeq = mgr->sequence()->record<kp::OpTensorSyncLocal>({tensorEntities});
    pushWaypointsSeq = mgr->sequence()->record<kp::OpTensorSyncDevice>({tensorWaypoints});
    retrieveQuadTreeNodesSeq = mgr->sequence()->record<kp::OpTensorSyncLocal>({tensorQuadTreeNodes});
    retrieveMetadataSeq = mgr->sequence()->record<kp::OpTensorSyncLocal>({tensorMetadata});
    pushMetadataSeq = mgr->sequence()->record<kp::OpTensorSyncDevice>({tensorMetadata});
    retrieveInterfaceCollisionsSeq = mgr->sequence()->record<kp::OpTensorSyncLocal>({tensorInterfaceCollisions});
    retrieveLinkEventsSeq = mgr->sequence()->record<kp::OpTensorSyncLocal>({tensorLinkUpEvents, tensorLinkDownEvents});

    // Perform initialization pass (with initial pushConstants)
    //  This builds the quadtree
    SPDLOG_DEBUG("Tick 0: Initialization pass started.", current_tick);
    shaderSeq->eval<kp::OpAlgoDispatch>(algo, pushConsts);
    SPDLOG_DEBUG("Tick 0: Initialization pass ended.", current_tick);

    std::unique_lock<std::mutex> lk(waitMutex);
    while (state == SimulatorState::RUNNING)
    {
        if (!simulating)
        {
            waitCondVar.wait(lk);
        }
        if (!simulating)
        {
            continue;
        }
        sim_tick();

        if (current_tick == Config::max_ticks)
        {
            state = SimulatorState::JOINING;  // Signal shutdown
            break;
        }
        current_tick++;
    }
}

void Simulator::sim_tick()
{
    std::chrono::high_resolution_clock::time_point tickStart = std::chrono::high_resolution_clock::now();

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
    start_frame_capture();
#endif

    // Reset Metadata
    metadata[0].interfaceCollisionCount = 0;
    metadata[0].linkUpEventCount = 0;
    metadata[0].linkDownEventCount = 0;
    metadata[0].waypointRequestCount = 0;
    sync_metadata_device();  // TODO use async?

    // TODO rem
    for (int i = 0; i < Config::num_entities; i++) {
        // Note: Every entity gets only 1 waypoint (at its entity index)
        waypoints[i].pos = utils::RNG::random_vec2(1, 99, 1, 99);
        waypoints[i].speed = utils::RNG::random_vec2(1, 50, 0, 0).x;
        fprintf(stderr, "%2.3f,%2.3f,%2.3f\n", waypoints[i].pos.x, waypoints[i].pos.y, waypoints[i].speed);
    }
    sync_waypoints_device();

    run_movement_pass();

    run_collision_detection_pass();

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    end_frame_capture();
#endif

    if ((Config::hint_sync_entities_every_tick ||
         entities_update_requested) &&
        entities_epoch_cpu == entities_epoch_gpu)
    {  // update requested AND outdated
        retrieveEntitiesSeq->evalAsync();  // start async retrieval
    }

    if (quad_tree_nodes_updates_requested &&
        quad_tree_nodes_epoch_cpu == quad_tree_nodes_epoch_gpu)
    {  // update requested AND outdated
        retrieveQuadTreeNodesSeq->evalAsync();  // start async retrieval
    }

    // TODO HERE WAS metadata eval async

    // retrieveEventsSeq->evalAsync(); // TODO

    if (entities_update_requested)
    {  // update requested
        sync_entities_local();
    }

    if (quad_tree_nodes_updates_requested)
    {  // update requested
        sync_quad_tree_nodes_local();
    }

    sync_metadata_local();

    // sanity check
    if (metadata[0].interfaceCollisionCount >= Config::max_interface_collisions)
    {
        // Cannot recover; some collisions are already lost
        throw std::runtime_error("Too many interface collisions (consider increasing the buffer size)");
    }

#if MSIM_DETECT_CONTACTS_CPU_STD | MSIM_DETECT_CONTACTS_CPU_EMIL
    sync_interface_collisions_local();
    run_interface_contacts_pass_cpu();
#else  // Run on GPU
    run_interface_contacts_pass_gpu();
    sync_link_events_local();
#endif

    // sanity check
    if (metadata[0].linkUpEventCount >= tensorLinkUpEvents->size())
    {
        // Cannot recover; some events are already lost
        throw std::runtime_error("Too many link up events (consider increasing the buffer size)");
    }
    if (metadata[0].linkDownEventCount >= tensorLinkDownEvents->size())
    {
        // Cannot recover; some events are already lost
        throw std::runtime_error("Too many link down events (consider increasing the buffer size)");
    }

    tpsHistory.add_time(std::chrono::high_resolution_clock::now() - tickStart);
    tps.tick();
}

void Simulator::continue_simulation()
{
    if (simulating)
    {
        return;
    }
    simulating = true;
    waitCondVar.notify_all();
}

void Simulator::pause_simulation()
{
    if (!simulating)
    {
        return;
    }
    simulating = false;
    waitCondVar.notify_all();
}

bool Simulator::is_simulating() const
{
    return simulating;
}

const utils::TickRate& Simulator::get_tps() const
{
    return tps;
}

const utils::TickDurationHistory& Simulator::get_tps_history() const
{
    return tpsHistory;
}

const utils::TickDurationHistory& Simulator::get_update_tick_history() const
{
    return updateTickHistory;
}

const utils::TickDurationHistory& Simulator::get_collision_detection_tick_history() const
{
    return collisionDetectionTickHistory;
}

void Simulator::check_device_queues()
{
    SPDLOG_INFO("Available GPU devices:");
    for (const vk::PhysicalDevice& device : mgr->listDevices())
    {
        SPDLOG_INFO("  GPU#{}: {}", device.getProperties().deviceID, device.getProperties().deviceName);
        for (const vk::QueueFamilyProperties2& props : device.getQueueFamilyProperties2())
        {
            SPDLOG_INFO("    {} queues supporting: {}{}{}{}{}{}",
                        props.queueFamilyProperties.queueCount,
                        (props.queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics) ? "graphics " : "",
                        (props.queueFamilyProperties.queueFlags & vk::QueueFlagBits::eCompute) ? "compute " : "",
                        (props.queueFamilyProperties.queueFlags & vk::QueueFlagBits::eTransfer) ? "transfer " : "",
                        (props.queueFamilyProperties.queueFlags & vk::QueueFlagBits::eSparseBinding) ? "sparse-binding " : "",
                        (props.queueFamilyProperties.queueFlags & vk::QueueFlagBits::eProtected) ? "protected " : "",
                        (props.queueFamilyProperties.queueFlags & vk::QueueFlagBits::eOpticalFlowNV) ? "optical-flow-nv " : "");
        }
    }
    SPDLOG_INFO("Using GPU#{}", mgr->getDeviceProperties().deviceID);
}

std::filesystem::path Simulator::get_log_csv_path()
{
    return Config::working_directory() / "logs" / (std::to_string(Config::num_entities) + ".csv");
}

void Simulator::prepare_log_csv_file()
{
    assert(!logFile);
    logFile = std::make_unique<std::ofstream>(Simulator::get_log_csv_path(), std::ios::out | std::ios::app);
    assert(logFile->is_open());
    assert(logFile->good());
}

std::string Simulator::get_time_stamp()
{
    std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
    std::chrono::days d = std::chrono::duration_cast<std::chrono::days>(tp.time_since_epoch());
    tp -= d;
    std::chrono::hours h = std::chrono::duration_cast<std::chrono::hours>(tp.time_since_epoch());
    tp -= h;
    std::chrono::minutes m = std::chrono::duration_cast<std::chrono::minutes>(tp.time_since_epoch());
    tp -= m;
    std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch());
    tp -= s;
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    tp -= ms;

    std::string msStr = std::to_string(ms.count());
    while (msStr.size() < 3)
    {
        // NOLINTNEXTLINE (performance-inefficient-string-concatenation)
        msStr = "0" + msStr;
    }

    std::string secStr = std::to_string(s.count());
    while (secStr.size() < 2)
    {
        // NOLINTNEXTLINE (performance-inefficient-string-concatenation)
        secStr = "0" + secStr;
    }

    std::string minStr = std::to_string(m.count());
    while (msStr.size() < 2)
    {
        // NOLINTNEXTLINE (performance-inefficient-string-concatenation)
        minStr = "0" + minStr;
    }

    std::string hourStr = std::to_string(h.count());
    while (hourStr.size() < 2)
    {
        // NOLINTNEXTLINE (performance-inefficient-string-concatenation)
        hourStr = "0" + hourStr;
    }

    return hourStr + ":" + minStr + ":" + secStr + "." + msStr;
}

void Simulator::write_log_csv_file(int64_t tick, std::chrono::nanoseconds durationUpdate, std::chrono::nanoseconds durationCollision, std::chrono::nanoseconds durationAll)
{
    assert(logFile);
    assert(logFile->is_open());
    assert(logFile->good());
    double secUpdate = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(durationUpdate).count()) / 1000;
    double secCollision = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(durationCollision).count()) / 1000;
    double secAll = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(durationAll).count()) / 1000;

    (*logFile) << Simulator::get_time_stamp() << ";" << std::to_string(tick) << ";" << secUpdate << ";" << secCollision << ";" << secAll << "\n";
    // std::cerr << Simulator::get_time_stamp() << ";" << std::to_string(tick) << ";" << secUpdate << ";" << secCollision << ";" << secAll << "\n";
    logFile->flush();
}

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
void Simulator::init_renderdoc()
{
    SPDLOG_INFO("Initializing RenderDoc in application API...");
    void* mod = dlopen("/usr/lib64/renderdoc/librenderdoc.so", RTLD_NOW);
    if (!mod)
    {
        // NOLINTNEXTLINE (concurrency-mt-unsafe)
        const char* error = dlerror();
        if (error)
        {
            SPDLOG_ERROR("Failed to find librenderdoc.so with: {}", error);
        }
        else
        {
            SPDLOG_ERROR("Failed to find librenderdoc.so with: Unknown error");
        }
        assert(false);
    }

    // NOLINTNEXTLINE (google-readability-casting)
    pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI) dlsym(mod, "RENDERDOC_GetAPI");
    // NOLINTNEXTLINE (google-readability-casting)
    int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_5_0, (void**) &rdocApi);
    assert(ret == 1);
    SPDLOG_INFO("RenderDoc in application API initialized.");
}

void Simulator::start_frame_capture()
{
    assert(rdocApi);
    //NOLINTNEXTLINE (cppcoreguidelines-pro-type-union-access)
    // assert(rdocApi->IsTargetControlConnected() == 1);
    // NOLINTNEXTLINE (google-readability-casting)
    // rdocApi->StartFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(mgr->getVkInstance().get()), nullptr);
    rdocApi->StartFrameCapture(nullptr, nullptr);
    // assert(rdocApi->IsFrameCapturing());
    SPDLOG_INFO("Renderdoc frame capture started.");
}

void Simulator::end_frame_capture()
{
    assert(rdocApi);
    //NOLINTNEXTLINE (cppcoreguidelines-pro-type-union-access)
    // assert(rdocApi->IsTargetControlConnected() == 1);
    if (!rdocApi->IsFrameCapturing())
    {
        SPDLOG_WARN("Renderdoc no need to end frame capture, no capture running.");
        return;
    }
    // NOLINTNEXTLINE (google-readability-casting)
    // rdocApi->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(mgr->getVkInstance().get()), nullptr);
    rdocApi->EndFrameCapture(nullptr, nullptr);
    // assert(!rdocApi->IsFrameCapturing());
    SPDLOG_INFO("Renderdoc frame capture ended.");
}
#endif
}  // namespace sim