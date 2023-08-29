#pragma once

#include <cstddef>
#include <memory>
#include <kompute/Manager.hpp>
#include <kompute/Sequence.hpp>
#include <kompute/Tensor.hpp>
#include <kompute/operations/OpTensorSyncDevice.hpp>
#include <kompute/operations/OpTensorSyncLocal.hpp>

#include "logger/Logger.hpp"

namespace sim
{

// Manages access to a GPU data buffer and its CPU-side staging buffer
template <typename T>
class GpuBuffer
{
 private:
    std::shared_ptr<kp::Tensor> tensor{nullptr};
    std::shared_ptr<kp::Sequence> pushSeq{nullptr};
    std::shared_ptr<kp::Sequence> pullSeq{nullptr};
    size_t epochCPU{0};
    size_t epochGPU{0};

 public:
    GpuBuffer(std::shared_ptr<kp::Manager> _mgr, size_t _size)
    {
        std::vector<T> init;  // Only for initialization
        init.resize(_size);
        tensor = _mgr->tensor(init.data(), init.size(), sizeof(T), kp::Tensor::TensorDataTypes::eUnsignedInt);
        pushSeq = _mgr->sequence()->record<kp::OpTensorSyncDevice>({tensor});
        pullSeq = _mgr->sequence()->record<kp::OpTensorSyncLocal>({tensor});
    }

    GpuBuffer(GpuBuffer&&) = delete;
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(GpuBuffer&&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // Pushes data from cpu to gpu, if newer
    // Does nothing if already in sync
    void push_data()
    {
        if (epochCPU == epochGPU)
        {  // already in sync
            return;
        }
        if (epochCPU < epochGPU)
        {
            SPDLOG_ERROR("GpuBuffer::push_data() overwriting newer gpu data");
        }

        // TIMER_START(fun_sync_entities_device); TODO
        if (!pushSeq->isRunning())
        {
            pushSeq->evalAsync();
        }

        pushSeq->evalAwait();
        epochGPU = epochCPU;
        // TIMER_STOP(fun_sync_entities_device); TODO
    }
    // Pulls data from gpu to cpu, if newer
    // Does nothing if already in sync
    void pull_data()
    {
        if (epochCPU == epochGPU)
        {  // already in sync
            return;
        }
        if (epochCPU > epochGPU)
        {
            SPDLOG_ERROR("GpuBuffer::pull_data() overwriting newer cpu data");
        }

        // TIMER_START(fun_sync_entities_local); TODO
        if (!pullSeq->isRunning())
        {
            pullSeq->evalAsync();
        }

        pullSeq->evalAwait();
        epochCPU = epochGPU;
        // TIMER_STOP(fun_sync_entities_local); TODO
    }
    // Starts asynchronous push of data from cpu to gpu, if newer
    // Does nothing if already in sync
    // Transfer MUST be finalized with a call to 'push_data()'
    //  void push_data_async(); TODO
    // Starts asynchronous pull of data from gpu to cpu, if newer
    // Does nothing if already in sync
    // Transfer MUST be finalized with a call to 'pull_data()'
    //  void pull_data_async(); TODO
    // TODO region variants
    // void push_data_region(size_t _offset, size_t _size);
    // void pull_data_region(size_t _offset, size_t _size);
    // void push_data_region_async(size_t _offset, size_t _size);
    // void pull_data_region_async(size_t _offset, size_t _size);

    // Provides read-write access to cpu data
    // Marks cpu data as modified
    T* data()
    {
        epochCPU++;
        return tensor->data<T>();
    }
    // Provides read-only access to cpu buffer data
    // Does NOT mark cpu data as modified
    const T* const_data()
    {
        return tensor->data<T>();
    }
    // Returns the internal kompute tensor
    const std::shared_ptr<kp::Tensor> tensor_raw()
    {
        return tensor;
    }

    // Marks gpu data as modified
    void mark_gpu_data_modified()
    {
        epochGPU++;
    }

    // Returns the current CPU data epoch
    size_t epoch_cpu()
    {
        return epochCPU;
    }
    // Returns the current GPU data epoch
    size_t epoch_gpu()
    {
        return epochGPU;
    }

    // Returns the total number of elements in this buffer
    size_t size()
    {
        return tensor->size();
    }
};

}  // namespace sim
