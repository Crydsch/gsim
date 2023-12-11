#pragma once

#include <kompute/Manager.hpp>
#include <kompute/Sequence.hpp>
#include <memory>
#include <vector>

#include "GpuBuffer.hpp"
#include "Constants.hpp"

namespace sim
{

// Note: An extra namespace is necessary to work around C++20 enums
//  We want to access the enum only via a safe namespace (ex. MyEnum::Value)
//  But also be able to cast the enum value to an integer (ex. int v = MyEnum::Value)
namespace shader_pass_ns
{

enum shader_pass
{
    Movement = 0,
    BuildQuadTreeInit = 1,
    BuildQuadTreeStep = 2,
    BuildQuadTreeFini = 3,
    ConnectivityDetection = 4,
};
}
typedef shader_pass_ns::shader_pass ShaderPass;


struct Parameter
{
    uint32_t pass{0};
    float timeIncrement{0};
    uint32_t interfaceCollisionSetOldOffset{0}; // Points at start of previous tick collisions
    uint32_t interfaceCollisionSetNewOffset{0}; // Points at start of current tick collisions
} __attribute__((aligned(4))) __attribute__((packed));
constexpr std::size_t parameterSize = sizeof(Parameter);


class GpuAlgorithm
{
 private:
    std::shared_ptr<kp::Algorithm> algo{nullptr};
    std::shared_ptr<kp::Sequence> generalShaderPass{nullptr};
    std::shared_ptr<kp::Sequence> pqcPass{nullptr};
    std::vector<sim::Parameter> parameter{};

 public:
    // Attention: The order in which buffers are specified, MUST be equivalent to the order in the shader (layout binding order)
    GpuAlgorithm(std::shared_ptr<kp::Manager> _mgr, std::vector<uint32_t> _shader, std::vector<std::shared_ptr<IGpuBuffer>> _buffer) {
        std::vector<std::shared_ptr<kp::Tensor>> tensors;
        tensors.reserve(_buffer.size());
        for (std::shared_ptr<IGpuBuffer> b : _buffer) {
            tensors.emplace_back(b->tensor_raw());
        }

        parameter.emplace_back(); // Note: The vector size must stay fixed after algorithm is created

        // Note: The max warp size of nvidia cards is 32
        const uint32_t local_group_size = 32; // Attention: This value must match in the shader!
        const uint32_t global_group_size = (uint32_t)ceil((float)(Config::num_entities) / (float)local_group_size);
        algo = _mgr->algorithm<float, Parameter>(tensors, _shader, {global_group_size, 1, 1}, {}, {parameter});
        generalShaderPass = _mgr->sequence();
        pqcPass = _mgr->sequence();

        // The pqc algorithm does need any runtime parameters
        //  so we prepare a dedicated sequence for faster execution
        parameter[0].pass = ShaderPass::BuildQuadTreeInit;
        pqcPass->record<kp::OpAlgoDispatch>(algo, parameter);
        for (size_t i = 0; i < Config::quad_tree_max_depth - 1; i++) {
            parameter[0].pass = ShaderPass::BuildQuadTreeStep;
            pqcPass->record<kp::OpAlgoDispatch>(algo, parameter);
        }
        parameter[0].pass = ShaderPass::BuildQuadTreeFini;
        pqcPass->record<kp::OpAlgoDispatch>(algo, parameter);
    }

    void run_pass(ShaderPass _pass) {
        [[maybe_unused]] std::string id = "gpu_algorithm_pass_";
        id.append(to_string(_pass));
        TIMER_START_STR(id);

        parameter[0].pass = _pass;
        generalShaderPass->evalAsync<kp::OpAlgoDispatch>(algo, parameter);
        generalShaderPass->evalAwait();

        TIMER_STOP_STR(id);
    }

    void run_movement_pass(float _timeIncrement) {
        parameter[0].timeIncrement = _timeIncrement;
        run_pass(ShaderPass::Movement);
    }

    void run_connectivity_detection_pass(uint _oldOffset, uint _newOffset) {
        // Build quadtree
        pqcPass->evalAsync();
        pqcPass->evalAwait();
        // Detect connectivity
        parameter[0].interfaceCollisionSetOldOffset = _oldOffset;
        parameter[0].interfaceCollisionSetNewOffset = _newOffset;
        run_pass(ShaderPass::ConnectivityDetection);
    }

    // Utility function for logging purposes
    static const char * to_string(ShaderPass _pass) {
        switch (_pass)
        {
        case ShaderPass::Movement :
            return "Movement";
        case ShaderPass::BuildQuadTreeInit :
            return "BuildQuadTreeInit";
        case ShaderPass::BuildQuadTreeStep :
            return "BuildQuadTreeStep";
        case ShaderPass::BuildQuadTreeFini :
            return "BuildQuadTreeFini";
        case ShaderPass::ConnectivityDetection :
            return "ConnectivityDetection";
        default:
            return "unknown";
        }
    }
};

}  // namespace sim
