#include "Simulator.hpp"
#include "Config.hpp"
#include "kompute/Manager.hpp"
#include "kompute/Tensor.hpp"
#include "logger/Logger.hpp"
#include "sim/Entity.hpp"
#include "sim/GpuQuadTree.hpp"
#include "sim/Map.hpp"
#include "utils/RNG.hpp"
#include "utils/Timer.hpp"
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
#include <fmt/core.h>

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
    // prepare_log_csv_file();
}

Simulator::~Simulator()
{
    assert(logFile);
    logFile->close();
    logFile = nullptr;

    delete connector;
    connector = nullptr;
}

void Simulator::init()
{
    std::vector<uint32_t> shader{}; // spirv byte code
#if STANDALONE_MODE
    SPDLOG_INFO("Simulation thread initializing (Standalone mode).");

    // Load map
    if (Config::map_filepath.empty())
    {
        SPDLOG_ERROR("No map configured. A map is required in standalone mode.");
		std::exit(1);
    }
    map = Map::load_from_file(Config::map_filepath);

#ifdef MOVEMENT_SIMULATOR_SHADER_INTO_HEADER
    // load shader from headerfile
    shader = std::vector(STANDALONE_COMP_SPV.begin(), STANDALONE_COMP_SPV.end());
#else
    // load shader from filesystem
    shader = load_shader(Config::working_directory() / std::filesystem::path("assets/shader/vulkan/standalone.comp.spv"));
#endif  // MOVEMENT_SIMULATOR_SHADER_INTO_HEADER

#else  // accelerator mode
    SPDLOG_INFO("Simulation thread initializing (Accelerator mode).");

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
#if ENABLE_GUI
    shader = std::vector(ACCELERATOR_GUI_COMP_SPV.begin(), ACCELERATOR_GUI_COMP_SPV.end());
#else
    shader = std::vector(ACCELERATOR_COMP_SPV.begin(), ACCELERATOR_COMP_SPV.end());
#endif  // ENABLE_GUI
#else
    // load shader from filesystem
#if ENABLE_GUI
    shader = load_shader(Config::working_directory() / std::filesystem::path("assets/shader/vulkan/accelerator_gui.comp.spv"));
#else
    shader = load_shader(Config::working_directory() / std::filesystem::path("assets/shader/vulkan/accelerator.comp.spv"));
#endif  // ENABLE_GUI
#endif  // MOVEMENT_SIMULATOR_SHADER_INTO_HEADER
#endif  // STANDALONE_MODE

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
    // Init RenderDoc:
    init_renderdoc();
#endif

    // kompute
    mgr = std::make_shared<kp::Manager>();

    // Entities
    bufEntities = std::make_shared<GpuBuffer<Entity>>(mgr, Config::num_entities, "Entities");
#if STANDALONE_MODE
    // Initialize Entities
    for (size_t i = 0; i < Config::num_entities; i++)
    {
        const unsigned int roadIndex = map->get_random_road_index();
        assert(roadIndex < map->roads.size());
        const Road road = map->roads[roadIndex];
        Entity* entities = bufEntities->data();
        entities[i] = Entity(utils::RNG::random_color(),
                                      Vec2(road.start.pos),
                                      Vec2(road.end.pos),
                                      utils::RNG::random_vec4u(),
                                      roadIndex);
    }
#else // accelerator mode
    // Initialize Entities
    Entity* entities = bufEntities->data();
    for (size_t i = 0; i < Config::num_entities; i++)
    { // waypoint buffer is initially empty => offset should point beyond buffer
        entities[i].targetWaypointOffset = Config::waypoint_buffer_size;
    }
#endif // STANDALONE_MODE
    bufEntities->push_data();

#if STANDALONE_MODE
    // Map
    bufMapRoads = std::make_shared<GpuBuffer<Road>>(mgr, map->roads.size(), "MapRoads");
    std::memcpy(bufMapRoads->data(), map->roads.data(), map->roads.size() * sizeof(Road));
    bufMapRoads->push_data();
    bufMapConnections = std::make_shared<GpuBuffer<uint32_t>>(mgr, map->connections.size(), "MapConnections");
    std::memcpy(bufMapConnections->data(), map->connections.data(), map->connections.size() * sizeof(uint32_t));
    bufMapConnections->push_data();
#else
    // Waypoints
    bufWaypoints = std::make_shared<GpuBuffer<Waypoint>>(mgr, Config::waypoint_buffer_size * Config::num_entities, "Waypoints");
    bufWaypoints->data();
    bufWaypoints->push_data();
#endif  // STANDALONE_MODE

    // Quad Tree
    //  Sanity checks
    assert(gpu_quad_tree::calc_node_count(1) == 1);
    assert(gpu_quad_tree::calc_node_count(2) == 5);
    assert(gpu_quad_tree::calc_node_count(3) == 21);
    assert(gpu_quad_tree::calc_node_count(4) == 85);
    assert(gpu_quad_tree::calc_node_count(8) == 21845);

    bufQuadTreeEntities = std::make_shared<GpuBuffer<gpu_quad_tree::Entity>>(mgr, Config::num_entities, "QuadTreeEntities");
    bufQuadTreeEntities->data();
    bufQuadTreeEntities->push_data();

    bufQuadTreeNodes = std::make_shared<GpuBuffer<gpu_quad_tree::Node>>(mgr, gpu_quad_tree::calc_node_count(Config::quad_tree_max_depth), "QuadTreeNodes");
    gpu_quad_tree::Node* quadTreeNodes_init = bufQuadTreeNodes->data();
    gpu_quad_tree::init_node_zero(quadTreeNodes_init[0], Config::map_width, Config::map_height);
    quadTreeNodes_init[0].first = Config::num_entities;
    quadTreeNodes_init[0].entityCount = Config::num_entities; // There are always all entities in the root node (directly or in children)

    // Init entire quadtree
    uint32_t index = 0;
    for (size_t i = 0; i < gpu_quad_tree::calc_node_count(Config::quad_tree_max_depth - 1); i++) {
        float width = quadTreeNodes_init[i].width / 2.0f;
        float height = quadTreeNodes_init[i].height / 2.0f;

        // Node TopLeft
        index++;
        quadTreeNodes_init[i].nextTL = index;
        quadTreeNodes_init[index].parent = i;
        quadTreeNodes_init[index].width = width;
        quadTreeNodes_init[index].height = height;
        quadTreeNodes_init[index].offsetX = quadTreeNodes_init[i].offsetX;
        quadTreeNodes_init[index].offsetY = quadTreeNodes_init[i].offsetY + height;
        quadTreeNodes_init[index].first = Config::num_entities;

        // Node TopRight
        index++;
        quadTreeNodes_init[index].parent = i;
        quadTreeNodes_init[index].width = width;
        quadTreeNodes_init[index].height = height;
        quadTreeNodes_init[index].offsetX = quadTreeNodes_init[i].offsetX + width;
        quadTreeNodes_init[index].offsetY = quadTreeNodes_init[i].offsetY + height;
        quadTreeNodes_init[index].first = Config::num_entities;

        // Node BottomLeft
        index++;
        quadTreeNodes_init[index].parent = i;
        quadTreeNodes_init[index].width = width;
        quadTreeNodes_init[index].height = height;
        quadTreeNodes_init[index].offsetX = quadTreeNodes_init[i].offsetX;
        quadTreeNodes_init[index].offsetY = quadTreeNodes_init[i].offsetY;
        quadTreeNodes_init[index].first = Config::num_entities;

        // Node BottomRight
        index++;
        quadTreeNodes_init[index].parent = i;
        quadTreeNodes_init[index].width = width;
        quadTreeNodes_init[index].height = height;
        quadTreeNodes_init[index].offsetX = quadTreeNodes_init[i].offsetX + width;
        quadTreeNodes_init[index].offsetY = quadTreeNodes_init[i].offsetY;
        quadTreeNodes_init[index].first = Config::num_entities;
    }

    bufQuadTreeNodes->push_data();

    // Constants
    bufConstants = std::make_shared<GpuBuffer<Constants>>(mgr, 1, "Constants");
    Constants* constants = bufConstants->data();
    constants[0].worldSizeX = Config::map_width;
    constants[0].worldSizeY = Config::map_height;
    constants[0].nodeCount = static_cast<uint32_t>(bufQuadTreeNodes->size());
    constants[0].entityCount = Config::num_entities;
    constants[0].maxDepth = Config::quad_tree_max_depth;
    constants[0].entityNodeCap = Config::quad_tree_entity_node_cap;
    constants[0].interfaceRange = Config::interface_range;
    constants[0].waypointBufferSize = Config::waypoint_buffer_size;
    constants[0].waypointBufferThreshold = Config::waypoint_buffer_threshold;
    constants[0].maxInterfaceCollisionSetCount = Config::interface_collisions_set_size;
    constants[0].maxLinkEventCount = Config::interface_link_events_list_size;
    bufConstants->push_data();

    // Metadata
    bufMetadata = std::make_shared<GpuBuffer<Metadata>>(mgr, 1, "Metadata");
    bufMetadata->data();
    bufMetadata->push_data();

    // Collision/Connectivity Detection
    //  Note: We actually allocate 2 sets in one buffer
    bufInterfaceCollisionsSet = std::make_shared<GpuBuffer<InterfaceCollisionBlock>>(mgr, 2 * Config::interface_collisions_set_size, "InterfaceCollisionBlock");
    bufInterfaceCollisionsSet->data();
    bufInterfaceCollisionsSet->push_data();
    bufInterfaceCollisionSetOldOffset = Config::interface_collisions_set_size;
    bufInterfaceCollisionSetNewOffset = 0;

#if not STANDALONE_MODE
    // Waypoint Requests
    bufWaypointRequests = std::make_shared<GpuBuffer<WaypointRequest>>(mgr, Config::num_entities, "WaypointRequests");
    bufWaypointRequests->data();
    bufWaypointRequests->push_data();
#endif

    // Connectivity
    bufLinkUpEventsList = std::make_shared<GpuBuffer<LinkUpEvent>>(mgr, Config::interface_link_events_list_size, "LinkUpEvent");
    bufLinkUpEventsList->data();
    bufLinkUpEventsList->push_data();
    bufLinkDownEventsList = std::make_shared<GpuBuffer<LinkDownEvent>>(mgr, Config::interface_link_events_list_size, "LinkDownEvent");
    bufLinkDownEventsList->data();
    bufLinkDownEventsList->push_data();

#if STANDALONE_MODE
    std::vector<std::shared_ptr<IGpuBuffer>> buffer = 
        {
            bufEntities, // Note: this bufEntities.size is the default number of shader invocations
            bufConstants,
            bufMapConnections,
            bufMapRoads,
            bufQuadTreeNodes,
            bufQuadTreeEntities,
            bufQuadTreeNodeUsedStatus,
            bufMetadata,
            bufWaypointRequests,
            bufInterfaceCollisionsSet,
            bufLinkUpEventsList,
            bufLinkDownEventsList
        };
#else
    std::vector<std::shared_ptr<IGpuBuffer>> buffer = 
        {
            bufEntities, // Note: bufEntities.size() is the default number of shader invocations
            bufConstants,
            bufWaypoints,
            bufQuadTreeNodes,
            bufQuadTreeEntities,
            bufMetadata,
            bufWaypointRequests,
            bufInterfaceCollisionsSet,
            bufLinkUpEventsList,
            bufLinkDownEventsList
        };
#endif  // STANDALONE_MODE
    algo = std::make_shared<GpuAlgorithm>(mgr, shader, buffer);

    // Check gpu capabilities
    check_device_queues();

#if not STANDALONE_MODE
    // Start communication
    connector = new PipeConnector();
#endif
}

void Simulator::init_instance()
{
    if (instance)
    {
        SPDLOG_ERROR("Simulator::init_instance called twice. Instance already exists.");
		std::exit(1);
    }
    instance = std::make_shared<Simulator>();
    instance->init();
}

std::shared_ptr<Simulator>& Simulator::get_instance()
{
    if (!instance)
    {
        SPDLOG_ERROR("Simulator::get_instance called before Simulator::init_instance. Instance does not yet exist.");
		std::exit(1);
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

bool Simulator::get_entities(const Entity** _out_entities, size_t& _inout_entity_epoch)
{
    entitiesUpdateRequested = true; // pull newest data from gpu when available

    bool update_available = bufEntities->epoch_cpu() != _inout_entity_epoch;

    *_out_entities = bufEntities->const_data();
    _inout_entity_epoch = bufEntities->epoch_cpu();

    return update_available;
}

bool Simulator::get_quad_tree_nodes(const gpu_quad_tree::Node** _out_quad_tree_nodes, size_t& _inout_quad_tree_nodes_epoch)
{
    quadTreeNodesUpdateRequested = true; // pull newest data from gpu when available

    bool update_available = bufQuadTreeNodes->epoch_cpu() != _inout_quad_tree_nodes_epoch;

    *_out_quad_tree_nodes = bufQuadTreeNodes->const_data();
    _inout_quad_tree_nodes_epoch = bufQuadTreeNodes->epoch_cpu();

    return update_available;
}

const std::shared_ptr<Map> Simulator::get_map() const
{
    return map;
}

int64_t Simulator::get_current_tick() const
{
    return current_tick;
}

void Simulator::reset_metadata()
{
    Metadata* metadata = bufMetadata->data();
    metadata[0].waypointRequestCount = 0;
    metadata[0].interfaceCollisionSetCount = 0; // Set before each pass
    metadata[0].interfaceLinkUpListCount = 0;
    metadata[0].interfaceLinkDownListCount = 0;
    bufMetadata->push_data();
}

void Simulator::recv_entity_positions()
{
    // TIMER_START(recv_entity_positions);
    bufEntities->pull_data();
    // Receive (initial) positions
    Entity* entities = bufEntities->data();
    for (size_t i = 0; i < Config::num_entities; i++)
    {
        Vec2 pos = connector->read_vec2();
        assert(pos.x > 0.0f);
        assert(pos.y > 0.0f);
        assert(pos.x < Config::map_width);
        assert(pos.y < Config::map_height);
        entities[i].pos = pos;
    }
    bufEntities->push_data();
    // TIMER_STOP(recv_entity_positions);
}

void Simulator::send_entity_positions()
{
    // TIMER_START(send_entity_positions);
    bufEntities->pull_data();
    const Entity* entities = bufEntities->const_data();
    for (size_t i = 0; i < Config::num_entities; i++)
    {
        connector->write_vec2(entities[i].pos);
    }
    connector->flush_output();
    // TIMER_STOP(send_entity_positions);
}

void Simulator::run_movement_pass()
{
    TIMER_START(move);
#if not STANDALONE_MODE

    float timeIncrement = connector->read_float();

    /* Receive waypoint updates */
    uint32_t numWaypointUpdates = connector->read_uint32();
    // if (numWaypointUpdates > 0) SPDLOG_TRACE("1>>> Received {} waypoint updates", numWaypointUpdates);
    assert(numWaypointUpdates <= Config::num_entities * Config::waypoint_buffer_size);

    if (numWaypointUpdates > 0) {
        // TIMER_START(recv_waypoint_updates);
        bufEntities->pull_data();
        Entity* entities = bufEntities->data();
        Waypoint* waypoints = bufWaypoints->data();

        for (uint32_t i = 0; i < numWaypointUpdates; i++) {
            uint32_t entityID = connector->read_uint32();
            uint16_t numWaypoints = connector->read_uint16();
            // SPDLOG_TRACE("1>>>  {} got {}", entityID, numWaypoints);
            assert(numWaypoints <= Config::waypoint_buffer_size);

            uint32_t remainingWaypoints = Config::waypoint_buffer_size - entities[entityID].targetWaypointOffset;
            assert(remainingWaypoints + numWaypoints <= Config::waypoint_buffer_size);

            uint32_t baseIndex = entityID * Config::waypoint_buffer_size;
            uint32_t oldIndex = baseIndex + entities[entityID].targetWaypointOffset;
            entities[entityID].targetWaypointOffset -= numWaypoints;
            uint32_t newIndex = baseIndex + entities[entityID].targetWaypointOffset;

            // Shift remaining waypoints
            for (uint32_t o = 0; o < remainingWaypoints; o++) {
                waypoints[newIndex + o] = waypoints[oldIndex + o];
            }

            newIndex += remainingWaypoints; // Index for new waypoints
            for (uint16_t o = 0; o < numWaypoints; o++) {
                waypoints[newIndex + o].pos = connector->read_vec2();
                waypoints[newIndex + o].speed = connector->read_float();
                assert(waypoints[newIndex + o].speed > 0.0f);
            }
        }

        bufEntities->push_data();
        bufWaypoints->push_data();
        // TIMER_STOP(recv_waypoint_updates);
    }
#else
    float timeIncrement = 0.5f;
#endif

    /* Perform movement */
    // debug_output_destinations_before_move();
    std::chrono::high_resolution_clock::time_point updateTickStart = std::chrono::high_resolution_clock::now();
    algo->run_pass<ShaderPass::Movement>(timeIncrement);
    std::chrono::nanoseconds durationUpdate = std::chrono::high_resolution_clock::now() - updateTickStart;
    updateTickHistory.add_time(durationUpdate);
    bufEntities->mark_gpu_data_modified();
    bufQuadTreeNodes->mark_gpu_data_modified();
    // debug_output_destinations_after_move();

    // After the movement pass entity positions and therefore the quadtree has changed
    // => Pull both buffers if an update is requested
    if (entitiesUpdateRequested) {
        bufEntities->pull_data();
        entitiesUpdateRequested = false;
    }

    if (quadTreeNodesUpdateRequested) {
        bufQuadTreeNodes->pull_data();
        quadTreeNodesUpdateRequested = false;
    }

#if not STANDALONE_MODE
    /* Send waypoint requests */
    bufMetadata->mark_gpu_data_modified(); // waypointRequestCount
    bufWaypointRequests->mark_gpu_data_modified();

    bufMetadata->pull_data();

    // sanity check
    const Metadata* metadata = bufMetadata->const_data();
    if (metadata[0].waypointRequestCount > bufConstants->const_data()->entityCount)
    { // Cannot recover; some requests are already lost
        SPDLOG_ERROR(fmt::format("Too many waypoint requests ({}). Consider increasing the buffer size.", metadata[0].waypointRequestCount));
		std::exit(1);
    }

    // Send waypoint requests
    connector->write_uint32(metadata[0].waypointRequestCount);
    if (metadata[0].waypointRequestCount > 0) {
        // TIMER_START(send_waypoint_requests);
        bufWaypointRequests->pull_data();
        const WaypointRequest* waypointRequests = bufWaypointRequests->const_data();
        for (uint32_t i = 0; i < metadata[0].waypointRequestCount; i++) {
            connector->write_uint32(waypointRequests[i].ID0); // Entity ID
            connector->write_uint16(waypointRequests[i].ID1); // Number of requested waypoints
        }
        // TIMER_STOP(send_waypoint_requests);
    }
    connector->flush_output();
#endif
    TIMER_STOP(move);
}

void Simulator::run_connectivity_detection_pass()
{
    TIMER_START(detect_connectivity);
    // Run connectivity detection pass
    Metadata* metadata = bufMetadata->data();
    // First available free block is after all implicit entity blocks
    metadata[0].interfaceCollisionSetCount = Config::num_entities;
    bufMetadata->push_data();

    std::chrono::high_resolution_clock::time_point collisionDetectionTickStart = std::chrono::high_resolution_clock::now();
    algo->run_pass<ShaderPass::BuildQuadTreeInit>();
    for (size_t i = 0; i < Config::quad_tree_max_depth - 1; i++) {
        algo->run_pass<ShaderPass::BuildQuadTreeStep>();
    }
    algo->run_pass<ShaderPass::BuildQuadTreeFini>();
    algo->run_pass<ShaderPass::ConnectivityDetection>(bufInterfaceCollisionSetOldOffset, bufInterfaceCollisionSetNewOffset);
    std::chrono::nanoseconds durationCollisionDetection = std::chrono::high_resolution_clock::now() - collisionDetectionTickStart;
    collisionDetectionTickHistory.add_time(durationCollisionDetection);
    bufMetadata->mark_gpu_data_modified(); // interfaceCollisionSetCount
    bufInterfaceCollisionsSet->mark_gpu_data_modified();
    bufLinkUpEventsList->mark_gpu_data_modified();
    bufLinkDownEventsList->mark_gpu_data_modified();

    // sanity check
    bufMetadata->pull_data();
    if (metadata[0].interfaceCollisionSetCount > Config::interface_collisions_set_size)
    {  // Cannot recover; some collisions are already lost
        SPDLOG_ERROR(fmt::format("Too many interface collisions ({} blocks required). Consider increasing the size or number of blocks.", metadata[0].interfaceCollisionSetCount));
		std::exit(1);
    }

    // Swap "buffers" for next tick
    std::swap(bufInterfaceCollisionSetOldOffset, bufInterfaceCollisionSetNewOffset);

    // sanity check
    bufMetadata->pull_data();
    if (metadata[0].interfaceLinkUpListCount >= Config::interface_link_events_list_size)
    { // Cannot recover; some events are already lost
        SPDLOG_ERROR(fmt::format("Too many link up events ({}). Consider increasing the buffer size.", metadata[0].interfaceLinkUpListCount));
		std::exit(1);
    }
    if (metadata[0].interfaceLinkDownListCount >= Config::interface_link_events_list_size)
    { // Cannot recover; some events are already lost
        SPDLOG_ERROR(fmt::format("Too many link down events ({}). Consider increasing the buffer size.", metadata[0].interfaceLinkDownListCount));
		std::exit(1);
    }
    TIMER_STOP(detect_connectivity);
}

void Simulator::send_connectivity_events()
{
    // TIMER_START(send_connectivity_events);
    bufMetadata->pull_data();

    const Metadata* metadata = bufMetadata->const_data();
    const LinkUpEvent* linkDownEvents = bufLinkDownEventsList->const_data();
    const LinkUpEvent* linkUpEvents = bufLinkUpEventsList->const_data();

    
    // Send link down events
    connector->write_uint32(metadata[0].interfaceLinkDownListCount);
    if (metadata[0].interfaceLinkDownListCount > 0) {
        bufLinkDownEventsList->pull_data_region(0, metadata[0].interfaceLinkDownListCount);
        for (uint32_t i = 0; i < metadata[0].interfaceLinkDownListCount; i++) {
            connector->write_uint32(linkDownEvents[i].ID0);
            connector->write_uint32(linkDownEvents[i].ID1);
            if (i % 512 == 0) {
                connector->flush_output();
            }
        }
    }
    connector->flush_output();

    // Send link up events
    connector->write_uint32(metadata[0].interfaceLinkUpListCount);
    if (metadata[0].interfaceLinkUpListCount > 0) {
        bufLinkUpEventsList->pull_data_region(0, metadata[0].interfaceLinkUpListCount);
        for (uint32_t i = 0; i < metadata[0].interfaceLinkUpListCount; i++) {
            connector->write_uint32(linkUpEvents[i].ID0);
            connector->write_uint32(linkUpEvents[i].ID1);
            if (i % 512 == 0) {
                connector->flush_output();
            }
        }
    }
    connector->flush_output();

    // TIMER_STOP(send_connectivity_events);
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
    }
}

void Simulator::sim_tick()
{
#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
    start_frame_capture();
#endif

#if STANDALONE_MODE
    current_tick++;
    TIMER_START(sim_tick); // Start next tick
    tickStart = std::chrono::high_resolution_clock::now();

    reset_metadata();
    run_movement_pass();
    run_collision_detection_pass(); // Detect all entity collisions
    run_connectivity_detection_pass(); // Detect Link up events

    TIMER_STOP(sim_tick); // Stop previous tick
    tpsHistory.add_time(std::chrono::high_resolution_clock::now() - tickStart);
    tps.tick();
#else
    Header header = connector->read_header();
    switch (header)
    {
    case Header::TestDataExchange :
        connector->testDataExchange();
        break;

    case Header::Shutdown :
        SPDLOG_DEBUG("Shutdown initiated.");
        state = SimulatorState::JOINING;  // Initiate shutdown
        return;

    case Header::Move :
        if (current_tick != 0) {
            tpsHistory.add_time(std::chrono::high_resolution_clock::now() - tickStart);
            tps.tick();
        }
        current_tick++;
        SPDLOG_DEBUG("Tick {}: Running movement pass", current_tick);

        tickStart = std::chrono::high_resolution_clock::now();
        reset_metadata();
        run_movement_pass();
        // debug_output_positions();
        // debug_output_quadtree();
        break;

    case Header::SetPositions :
        SPDLOG_DEBUG("Tick {}: Receiving entity positions", current_tick);
        recv_entity_positions();
        break;

    case Header::GetPositions :
        SPDLOG_DEBUG("Tick {}: Sending entity positions", current_tick);
        send_entity_positions();
        break;

    case Header::ConnectivityDetection :
        SPDLOG_DEBUG("Tick {}: Running connectivity detection pass", current_tick);
        run_connectivity_detection_pass();
        // debug_output_collisions_list();
        // debug_output_collisions_list_counted();
        SPDLOG_DEBUG("Tick {}: Sending link events", current_tick);
        send_connectivity_events();
        break;
    
    default:
        SPDLOG_ERROR("Simulator::sim_tick(): Received invalid header");
		std::exit(1);
        break;
    }
#endif

#ifdef MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    end_frame_capture();
#endif

    // simulating = false; // [Debug] Pause after each tick.
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
    return Config::working_directory() / std::filesystem::path("logs") / std::filesystem::path((std::to_string(Config::num_entities) + ".csv"));
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

void Simulator::debug_output_positions() {
    FILE* file = fopen("/gsim/logs/debug/pos_m", "a+");

    bufEntities->pull_data();
    const Entity* entities = bufEntities->const_data();
    for (size_t i = 0; i < bufEntities->size(); i++) {
        // fprintf(file, "%03ld,%04ld,%f,%f\n", current_tick, i, entities[i].pos.x, entities[i].pos.y);
        fprintf(file, "%03ld,%04ld,%a,%a\n", current_tick, i, entities[i].pos.x, entities[i].pos.y);
    }
    fclose(file);
}

void Simulator::debug_output_destinations_before_move()
{
    bufEntities->pull_data();

    // Save copy
    debug_output_destinations_entities = bufEntities->tensor_raw()->vector<Entity>();
}

void Simulator::debug_output_destinations_after_move() {
#if not STANDALONE_MODE
    // Compare offsets to previous tick, to identify the number reached destinations
    bufEntities->pull_data();
    const Entity* entities = bufEntities->const_data();
    const Waypoint* waypoints = bufWaypoints->const_data();

    FILE* file = fopen("/gsim/logs/debug/dests_gsim.txt", "a+");

    for (size_t i = 0; i < bufEntities->size(); i++) {
        size_t reached_destinations = entities[i].targetWaypointOffset - debug_output_destinations_entities[i].targetWaypointOffset;
        for (size_t o = 0; o < reached_destinations; o++) {
            size_t waypoint_offset = i * Config::waypoint_buffer_size + debug_output_destinations_entities[i].targetWaypointOffset + o;
            fprintf(file, "%ld,%ld,%f,%f\n", current_tick, i, waypoints[waypoint_offset].pos.x, waypoints[waypoint_offset].pos.y);
        }
    }

    fclose(file);
#endif
}

void Simulator::debug_output_quadtree() {
    bufQuadTreeNodes->mark_gpu_data_modified();
    bufQuadTreeNodes->pull_data();
    const gpu_quad_tree::Node* nodes = bufQuadTreeNodes->const_data();
    bufQuadTreeEntities->mark_gpu_data_modified();
    bufQuadTreeEntities->pull_data();
    const gpu_quad_tree::Entity* entities = bufQuadTreeEntities->const_data();

    FILE* file = fopen("/gsim/logs/debug/tree_m", "a+");

    // Ref.: https://stackoverflow.com/questions/2067988/recursive-lambda-functions-in-c11
    auto print_node = [nodes, entities, file](const gpu_quad_tree::Node* node) {
        auto print_node_impl = [nodes, entities, file](const gpu_quad_tree::Node* node, auto& print_node_ref) mutable {
            if (node->first < Config::num_entities) {
                // it is a leaf node
                uint32_t id = node->first;

                for (uint32_t i = 0; i < node->entityCount; i++) {
                    fprintf(file, "%d - %d\n", node->entityCount, id);
                    id = entities[id].next;
                }

                return;
            }

            // else it is an internal node
            print_node_ref(&nodes[node->nextTL], print_node_ref);
            print_node_ref(&nodes[node->nextTL+1], print_node_ref);
            print_node_ref(&nodes[node->nextTL+2], print_node_ref);
            print_node_ref(&nodes[node->nextTL+3], print_node_ref);
        };
        print_node_impl(node, print_node_impl);
    };

    print_node(&nodes[0]);

    fclose(file);
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