#pragma once

#include <cstddef>
#include <kompute/Manager.hpp>
#include <kompute/Sequence.hpp>
#include <kompute/Tensor.hpp>
#include <kompute/operations/OpTensorSyncDevice.hpp>
#include <kompute/operations/OpTensorSyncLocal.hpp>
#include <kompute/operations/OpTensorSyncRegionDevice.hpp>
#include <kompute/operations/OpTensorSyncRegionLocal.hpp>
#include <memory>

#include "logger/Logger.hpp"
#include "utils/Timer.hpp"

namespace sim
{

// Interface to GpuBuffer template class
class IGpuBuffer
{
 protected:
    std::shared_ptr<kp::Tensor> tensor{nullptr};
    size_t epochCPU{0};
    size_t epochGPU{0};

 public:
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

// Manages access to a GPU data buffer and its CPU-side staging buffer
template <typename T>
class GpuBuffer : public IGpuBuffer
{
 private:
    std::shared_ptr<kp::Manager> mgr{nullptr};
    std::shared_ptr<kp::Sequence> pushSeq{nullptr};
    std::shared_ptr<kp::Sequence> pullSeq{nullptr};
    const char* name;
    
 public:
    GpuBuffer(std::shared_ptr<kp::Manager> _mgr, size_t _size, const char* _name) : mgr(_mgr), name(_name)
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

    // Pushes data from cpu to gpu, if newer (copies the entire buffer)
    // Does nothing if already in sync
    void push_data()
    {
        if (epochCPU == epochGPU)
        {  // already in sync
            return;
        }
        if (epochCPU < epochGPU)
        {
            SPDLOG_ERROR("GpuBuffer<{}>::push_data() overwriting newer gpu data", name);
        }

        [[maybe_unused]] const std::string id = std::format("gpu_buffer_push_{}", name);
        TIMER_START_STR(id);

        if (!pushSeq->isRunning())
        {
            pushSeq->evalAsync();
        }

        pushSeq->evalAwait();
        epochGPU = epochCPU;

        TIMER_STOP_STR(id);
    }
    // Pulls data from gpu to cpu, if newer (copies the entire buffer)
    // Does nothing if already in sync
    void pull_data()
    {
        if (epochCPU == epochGPU)
        {  // already in sync
            return;
        }
        if (epochCPU > epochGPU)
        {
            SPDLOG_ERROR("GpuBuffer<{}>::pull_data() overwriting newer cpu data", name);
        }

        [[maybe_unused]] const std::string id = std::format("gpu_buffer_pull_{}", name);
        TIMER_START_STR(id);

        if (!pullSeq->isRunning())
        {
            pullSeq->evalAsync();
        }

        pullSeq->evalAwait();
        epochCPU = epochGPU;

        TIMER_STOP_STR(id);
    }
    // Starts asynchronous push of data from cpu to gpu, if newer
    // Does nothing if already in sync
    // Transfer MUST be finalized with a call to 'push_data()'
    //  void push_data_async(); TODO
    // Starts asynchronous pull of data from gpu to cpu, if newer
    // Does nothing if already in sync
    // Transfer MUST be finalized with a call to 'pull_data()'
    //  void pull_data_async(); TODO
    
    // Pushes data from cpu to gpu, if newer (copies only the region)
    // Does nothing if already in sync
    void push_data_region(uint32_t _offset, uint32_t _size)
    {
#if KOMPUTE_COPY_REGIONS
        if (epochCPU == epochGPU)
        {  // already in sync
            return;
        }
        if (epochCPU < epochGPU)
        {
            SPDLOG_ERROR("GpuBuffer<{}>::push_data_region() overwriting newer gpu data", name);
        }

        [[maybe_unused]] const std::string id = std::format("gpu_buffer_push_region_{}", name);
        TIMER_START_STR(id);

        mgr->sequence()
            ->evalAsync<kp::OpTensorSyncRegionDevice>({{ tensor, _offset, _offset, _size }})
            ->evalAwait();

        epochGPU = epochCPU;

        TIMER_STOP_STR(id);
#else
     push_data(); // Fallback to full buffer copy
#endif
    }
    // Pulls data from gpu to cpu, if newer (copies only the region)
    // Does nothing if already in sync
    void pull_data_region(uint32_t _offset, uint32_t _size)
    {
#if KOMPUTE_COPY_REGIONS
        if (epochCPU == epochGPU)
        {  // already in sync
            return;
        }
        if (epochCPU > epochGPU)
        {
            SPDLOG_ERROR("GpuBuffer<{}>::pull_data_region() overwriting newer cpu data", name);
        }

        [[maybe_unused]] const std::string id = std::format("gpu_buffer_pull_region_{}", name);
        TIMER_START_STR(id);

        mgr->sequence()
            ->evalAsync<kp::OpTensorSyncRegionLocal>({{ tensor, _offset, _offset, _size }})
            ->evalAwait();

        epochCPU = epochGPU;

        TIMER_STOP_STR(id);
#else
    pull_data(); // Fallback to full buffer copy
#endif
    }
    // void push_data_region_async(size_t _offset, size_t _size); TODO
    // void pull_data_region_async(size_t _offset, size_t _size); TODO

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
};

}  // namespace sim
