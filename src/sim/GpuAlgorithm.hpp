#pragma once

#include <kompute/Manager.hpp>
#include <kompute/Sequence.hpp>
#include <memory>
#include <vector>

#include "GpuBuffer.hpp"
#include "PushConsts.hpp"

namespace sim
{

// Note: An extra namespace is necessary to work around C++20 enums
//  We want to access the enum only via a safe namespace (ex. MyEnum::Value)
//  But also be able to cast the enum value to an integer (ex. int v = MyEnum::Value)
namespace shader_pass_ns
{
enum shader_pass
{
    Initialization = 0,
    Movement = 1,
    CollisionDetection = 2
};
}
typedef shader_pass_ns::shader_pass ShaderPass;


class GpuAlgorithm
{
 private:
    std::shared_ptr<kp::Algorithm> algo{nullptr};
    std::shared_ptr<kp::Sequence> shaderSeq{nullptr};
    std::vector<sim::PushConsts> parameter{};

 public:
    // Attention: The order in which buffers are specified, MUST be equivalent to the order in the shader (layout binding order)
    GpuAlgorithm(std::shared_ptr<kp::Manager> _mgr, std::vector<uint32_t> _shader, std::vector<std::shared_ptr<IGpuBuffer>> _buffer, std::vector<sim::PushConsts> _parameter) {
        parameter = _parameter;

        std::vector<std::shared_ptr<kp::Tensor>> tensors;
        tensors.reserve(_buffer.size());
        for (std::shared_ptr<IGpuBuffer> b : _buffer) {
            tensors.emplace_back(b->tensor_raw());
        }

        algo = _mgr->algorithm<float, PushConsts>(tensors, _shader, {}, {}, {_parameter});
        shaderSeq = _mgr->sequence();
    }

    template<ShaderPass _pass>
    void run_pass() {
        parameter[0].pass = _pass;
        shaderSeq->evalAsync<kp::OpAlgoDispatch>(algo, parameter);
        shaderSeq->evalAwait();
    }

    template<ShaderPass _pass>
    std::enable_if_t<_pass == ShaderPass::Movement, void>
    run_pass(float _timeIncrement) {
        parameter[0].timeIncrement = _timeIncrement;
        run_pass<ShaderPass::Movement>();
    }
};

}  // namespace sim
