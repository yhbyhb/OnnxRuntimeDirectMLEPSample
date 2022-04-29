#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <iostream>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <functional>
#include <utility>
#include <string_view>
#include <span>
#include <optional>
#include <memory>
#include <assert.h>

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h> // Use the good old helper functions, not the huge WinRT entanglement.

#include "dml_provider_factory.h"
#include "onnxruntime_cxx_api.h"

////////////////////////////////////////////////////////////////////////////////
// Configuration

#define USE_DML_EXECUTION_PROVIDER true
#define PASS_TENSORS_AS_D3D_RESOURCES true
#define EXPORT_ORT_FILE false

static_assert(USE_DML_EXECUTION_PROVIDER == true || PASS_TENSORS_AS_D3D_RESOURCES == false); // CPU can't accept D3D resources.

////////////////////////////////////////////////////////////////////////////////
// Common helpers

#define THROW_IF_FAILED(hr) {HRESULT localHr = (hr); if (FAILED(localHr)) throw localHr;}
#define RETURN_IF_FAILED(hr) {HRESULT localHr = (hr); if (FAILED(localHr)) return localHr;}
#define THROW_IF_NOT_OK(status) {auto localStatus = (status); if (localStatus) throw E_FAIL;}
#define RETURN_HR_IF_NOT_OK(status) {auto localStatus = (status); if (localStatus) return E_FAIL;}


template <typename T>
using BaseType =
    std::remove_cv_t<
        std::remove_reference_t<
            std::remove_pointer_t<
                std::remove_all_extents_t<T>
            >
        >
    >;


template <typename T>
using deleting_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;


template <typename C, typename T = BaseType<decltype(*std::declval<C>().data())>>
T GetElementCount(C const& range)
{
    return std::accumulate(range.begin(), range.end(), static_cast<T>(1), std::multiplies<T>());
};


// Used to work-around std::vector bool specialization.
template <typename T>
class WrapperClass
{
public:
    WrapperClass() {}
    WrapperClass(T const& value) : value_(value) {}
    T value_;
};


template <typename T>
struct ComPtr : public Microsoft::WRL::ComPtr<T>
{
    // Having to call .Get() dozens of times for every function call that takes a T* is ridiculous.
    operator T* () { return this->Get(); }
};


size_t ByteSizeOfOnnxTensorElementDataType(ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:   return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:        return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:       return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:        return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:      return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:       return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:     return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:       return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:       return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:       return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   return 8; // 32*2
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  return 16;
    default: return 1;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Forward declarations for helpers

Ort::Value CreateTensorValueUsingD3DResource(
    ID3D12Device* d3dDevice,
    OrtDmlApi const& ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    std::span<const int64_t> dimensions,
    ONNXTensorElementDataType elementDataType,
    size_t elementByteSize,
    /*out opt*/ ID3D12Resource** d3dResource,
    /*out*/ void** dmlEpResourceWrapper
);

void UploadTensorData(
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocator,
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* destinationResource,
    std::span<const std::byte> sourceData
);

void DownloadTensorData(
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocatar,
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* sourceResource,
    std::span<std::byte> destinationData
);


////////////////////////////////////////////////////////////////////////////////

int main()
{
#if 1
    const wchar_t* modelFilePath = L"Upsample4xOpset11.onnx";
    const char* modelInputTensorName = "input";
    const char* modelOutputTensorName = "output";
    const std::array<int64_t, 4> inputShape = { 1, 3, 100, 100 };
    const std::array<int64_t, 4> outputShape = { 1, 3, 400, 400 };
    ONNXTensorElementDataType inputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    ONNXTensorElementDataType outputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    using inputDataTypeT = float;
    using outputDataTypeT = float;
#elif 1
    // Squeezenet opset v7 https://github.com/onnx/models/blob/master/vision/classification/squeezenet/README.md
    const wchar_t* modelFilePath = L"squeezenet/SqueezeNet.onnx";
    const char* modelInputTensorName = "data_0";
    const char* modelOutputTensorName = "softmaxout_1";
    const std::array<int64_t, 4> inputShape = { 1, 3, 224, 224 };
    const std::array<int64_t, 4> outputShape = { 1, 1000, 1, 1 };
    ONNXTensorElementDataType inputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    ONNXTensorElementDataType outputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    using inputDataTypeT = float;
    using outputDataTypeT = float;
#elif 1
    const wchar_t* modelFilePath = L"OnnxBackendTestData/test_nonzero_example/model.onnx";
    const char* modelInputTensorName = "condition";
    const char* modelOutputTensorName = "result";
    const std::array<int64_t, 2> inputShape = { 2, 2 };
    const std::array<int64_t, 2> outputShape = { 2, 4 };
    ONNXTensorElementDataType inputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    ONNXTensorElementDataType outputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    using inputDataTypeT = WrapperClass<bool>;
    using outputDataTypeT = int64_t;
#elif 0
    const wchar_t* modelFilePath = L"OnnxBackendTestData/test_shape/model.onnx";
    const char* modelInputTensorName = "x";
    const char* modelOutputTensorName = "y";
    const std::array<int64_t, 3> inputShape = { 3, 4, 5 };
    const std::array<int64_t, 1> outputShape = { 3 };
    ONNXTensorElementDataType inputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    ONNXTensorElementDataType outputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    using inputDataTypeT = float;
    using outputDataTypeT = int64_t;
#endif

    LARGE_INTEGER startTime;
    LARGE_INTEGER d3dDeviceCreationTime;
    LARGE_INTEGER sessionCreationTime;
    LARGE_INTEGER tensorCreationTime;
    LARGE_INTEGER bindingTime;
    LARGE_INTEGER runStartTime;
    LARGE_INTEGER runTime;
    LARGE_INTEGER runEndTime;
    LARGE_INTEGER synchronizeOutputsTime;
    LARGE_INTEGER cpuFrequency;
    QueryPerformanceFrequency(&cpuFrequency);
    QueryPerformanceCounter(&startTime);

    try
    {
        // Note that if you want a specific GPU, you should call EnumAdapters.
        // Otherwise in a system with multiple GPU's (a fast discrete one and a slow
        // integrated one), you might get the slow one depending on the defaults.
        // todo: change the adapter from nullptr to an explicit EnumAdaptersByGpu call,
        // or EnumAdapters.
        printf("Creating Direct3D device.\n");

        // Yeah, D3D's interface just to upload some resource data is a bit ... verbose.
        ComPtr<ID3D12Device> d3d12Device;
        ComPtr<ID3D12CommandQueue> commandQueue;
        ComPtr<ID3D12CommandAllocator> commandAllocator;
        ComPtr<ID3D12GraphicsCommandList> commandList;

        D3D12_COMMAND_QUEUE_DESC const commandQueueDescription =
        {
            .Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
            .Priority = 0,
            .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
            .NodeMask = 0,
        };

        THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12Device)));
        QueryPerformanceCounter(&d3dDeviceCreationTime);

        THROW_IF_FAILED(d3d12Device->CreateCommandQueue(&commandQueueDescription, IID_PPV_ARGS(&commandQueue)));
        THROW_IF_FAILED(d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)));
        THROW_IF_FAILED(d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList)));

        OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
        const OrtDmlApi* ortDmlApi;
        THROW_IF_NOT_OK(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));

        // ONNX Runtime setup
        Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "DirectML_Direct3D_TensorAllocation_Test"); // Note ORT_LOGGING_LEVEL_VERBOSE is useful too.
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.DisableMemPattern();
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // Note ORT_ENABLE_BASIC is useful for debugging.
        ortApi.AddFreeDimensionOverrideByName(sessionOptions, "batch_size", 1);

        if (EXPORT_ORT_FILE)
        // Test export and reload of optimized model.
        {
            Ort::SessionOptions sessionOptions2(sessionOptions.Clone());
            //sessionOptions2.AddConfigEntry("ep.dml.disable_graph_fusion", "1");
            sessionOptions2.SetOptimizedModelFilePath(L"optimized.ort");
            if (USE_DML_EXECUTION_PROVIDER)
            {
                ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions2, 0);
            }
            Ort::Session session2 = Ort::Session(ortEnvironment, modelFilePath, sessionOptions);
            modelFilePath = L"optimized.ort";
        }

        if (USE_DML_EXECUTION_PROVIDER)
        {
            printf("Adding the DirectML execution provider.\n");
            ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, 0); // todo: change this to an explicit device id, not just 0 using adapter above.
        }

        printf("Loading model.\n");
        Ort::Session session = Ort::Session(ortEnvironment, modelFilePath, sessionOptions);
        QueryPerformanceCounter(&sessionCreationTime);

        Ort::IoBinding ioBinding = Ort::IoBinding::IoBinding(session);
        const char* memoryInformationName = PASS_TENSORS_AS_D3D_RESOURCES ? "DML" : "Cpu";
        Ort::MemoryInfo memoryInformation(memoryInformationName, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
        // Not needed: Ort::Allocator allocator(session, memoryInformation);

        // Create input tensor.
        Ort::Value inputTensor(nullptr);
        std::vector<inputDataTypeT> inputTensorValues(static_cast<size_t>(GetElementCount(inputShape)), inputDataTypeT(0));
        if constexpr (std::is_same_v<inputDataTypeT, WrapperClass<bool>>) // Why C++, why?... Just let iota work as expected with a wrapping sequence of {false, true, false, true...}.
        {
            std::fill(inputTensorValues.begin(), inputTensorValues.end(), inputDataTypeT(1));
        }
        else
        {
            std::iota(inputTensorValues.begin(), inputTensorValues.end(), inputDataTypeT(0));
        }
        ComPtr<IUnknown> inputTensorExecutionProviderWrapper;

        if (PASS_TENSORS_AS_D3D_RESOURCES)
        {
            ComPtr<ID3D12Resource> inputD3dResource;
            // Create empty D3D resource for input.
            inputTensor = CreateTensorValueUsingD3DResource(
                d3d12Device,
                *ortDmlApi,
                memoryInformation,
                inputShape,
                inputDataType,
                ByteSizeOfOnnxTensorElementDataType(inputDataType),
                /*out*/ &inputD3dResource,
                /*out*/ IID_PPV_ARGS_Helper(inputTensorExecutionProviderWrapper.GetAddressOf())
            );
            UploadTensorData(
                commandQueue,
                commandAllocator,
                commandList,
                inputD3dResource,
                std::as_bytes(std::span<inputDataTypeT>(inputTensorValues))
            );
        }
        else // CPU tensor
        {
            inputTensor = Ort::Value::CreateTensor<inputDataTypeT>(
                memoryInformation,
                reinterpret_cast<inputDataTypeT*>(inputTensorValues.data()),
                inputTensorValues.size(),
                inputShape.data(),
                inputShape.size()
            );
        }

        // Create output tensor on device memory.
        Ort::Value outputTensor(nullptr);
        std::vector<outputDataTypeT> outputTensorValues(static_cast<size_t>(GetElementCount(outputShape)), outputDataTypeT(0));
        ComPtr<IUnknown> outputTensorEpWrapper;
        ComPtr<ID3D12Resource> outputD3dResource;

        if (PASS_TENSORS_AS_D3D_RESOURCES)
        {
            outputTensor = CreateTensorValueUsingD3DResource(
                d3d12Device,
                *ortDmlApi,
                memoryInformation,
                outputShape,
                outputDataType,
                ByteSizeOfOnnxTensorElementDataType(outputDataType),
                /*out*/ &outputD3dResource,
                /*out*/ IID_PPV_ARGS_Helper(outputTensorEpWrapper.GetAddressOf())
            );
        }
        else // CPU tensor
        {
            outputTensor = Ort::Value::CreateTensor<outputDataTypeT>(
                memoryInformation,
                outputTensorValues.data(),
                outputTensorValues.size(),
                outputShape.data(),
                outputShape.size()
            );
        }

        QueryPerformanceCounter(&tensorCreationTime);

        ////////////////////////////////////////
        // Bind the tensor inputs to the model, and run it.
        ioBinding.BindInput(modelInputTensorName, inputTensor);
        ioBinding.BindOutput(modelOutputTensorName, outputTensor);
        ioBinding.SynchronizeInputs();
        QueryPerformanceCounter(&bindingTime);

        Ort::RunOptions runOptions;

        printf("Beginning execution.\n");
        QueryPerformanceCounter(&runStartTime);
        session.Run(runOptions, ioBinding);
        QueryPerformanceCounter(&runTime);
        ioBinding.SynchronizeOutputs();
        QueryPerformanceCounter(&synchronizeOutputsTime);
        runEndTime = synchronizeOutputsTime;
        printf("Finished execution.\n");

        if (PASS_TENSORS_AS_D3D_RESOURCES)
        {
            DownloadTensorData(
                commandQueue,
                commandAllocator,
                commandList,
                outputD3dResource,
                std::as_writable_bytes(std::span<outputDataTypeT>(outputTensorValues))
            );
        }

        auto printDuration = [=](char const* message, LARGE_INTEGER nextTime, LARGE_INTEGER previousTime = {}) mutable
        {
            if (previousTime.QuadPart == 0)
            {
                previousTime = startTime;
            }
            double durationMs = static_cast<double>(nextTime.QuadPart - previousTime.QuadPart);
            durationMs /= static_cast<double>(cpuFrequency.QuadPart);
            durationMs *= 1000.0;
            printf("%s % 12.6fms\n", message, durationMs);

            startTime = nextTime;
        };
        printDuration("D3D device creation time ....", d3dDeviceCreationTime);
        printDuration("session creation time .......", sessionCreationTime);
        printDuration("tensor creation time ........", tensorCreationTime);
        printDuration("binding time ................", bindingTime);
        printDuration("run time ....................", runTime);
        printDuration("synchronize outputs time ....", synchronizeOutputsTime);
        printDuration("run+synchronize time.........", runEndTime, runStartTime);
        printDuration("total time...................", synchronizeOutputsTime, startTime);

        ////////////////////////////////////////
        // Print the first 10 and top 10 results.
        printf("First 10 results:\n");
        for (int i = 0; i <= std::min(outputTensorValues.size(), size_t(10)); ++i)
        {
            printf("    output[%d] = %f\n", i, outputTensorValues[i]);
        }

        printf("Top 10 results:\n");
        size_t maxSortSize = std::min(size_t(10'000), outputTensorValues.size());
        std::vector<uint32_t> indices(maxSortSize, 0);
        std::iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) { return (outputTensorValues[a] > outputTensorValues[b]);});
        for (size_t i = 0, ci = std::min(indices.size(), size_t(10)); i <= ci; ++i)
        {
            printf("    output[%d] = %f\n", indices[i], outputTensorValues[indices[i]]);
        }
    }
    catch (Ort::Exception const& exception)
    {
        printf("Error running model inference: %s\n", exception.what());
        return EXIT_FAILURE;
    }
    catch (std::exception const& exception)
    {
        printf("Error running model inference: %s\n", exception.what());
        return EXIT_FAILURE;
    }

    return 0;
}


ComPtr<ID3D12Resource> CreateD3D12ResourceOfByteSize(
    ID3D12Device* d3dDevice,
    size_t resourceByteSize,
    D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT,
    D3D12_RESOURCE_STATES resourceState = D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
)
{
    resourceByteSize = std::max(resourceByteSize, size_t(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));

    // DML needs the resources' sizes to be a multiple of 4 bytes
    (resourceByteSize += 3) &= ~3;

    D3D12_HEAP_PROPERTIES const heapProperties = {
        .Type = heapType, // Default to D3D12_HEAP_TYPE_DEFAULT.
        .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
        .CreationNodeMask = 1,
        .VisibleNodeMask = 1
    };

    D3D12_RESOURCE_DESC const resourceDesc =
    {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = static_cast<uint64_t>(resourceByteSize),
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = {1, 0},
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = resourceFlags // Default to D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
    };

    ComPtr<ID3D12Resource> gpuResource;
    THROW_IF_FAILED(d3dDevice->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        resourceState, // Default to D3D12_RESOURCE_STATE_COMMON
        nullptr,
        __uuidof(ID3D12Resource),
        /*out*/ &gpuResource
    ));

    return gpuResource;
}


ComPtr<ID3D12Resource> CreateD3D12ResourceForTensor(
    ID3D12Device* d3dDevice,
    size_t elementByteSize,
    std::span<const int64_t> tensorDimensions
)
{
    // Try to allocate the backing memory for the caller
    auto bufferSize = GetElementCount(tensorDimensions);
    assert(bufferSize > 0);
    assert(elementByteSize > 0);
    size_t bufferByteSize = static_cast<size_t>(bufferSize * elementByteSize);

    return CreateD3D12ResourceOfByteSize(d3dDevice, bufferByteSize);
}


Ort::Value CreateTensorValueFromExistingD3DResource(
    OrtDmlApi const& ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    ID3D12Resource* d3dResource,
    std::span<const int64_t> tensorDimensions,
    ONNXTensorElementDataType elementDataType,
    /*out*/ void** dmlEpResourceWrapper // Must stay alive with Ort::Value.
    )
{
    *dmlEpResourceWrapper = nullptr;

    void* dmlAllocatorResource;
    THROW_IF_NOT_OK(ortDmlApi.CreateGPUAllocationFromD3DResource(d3dResource, &dmlAllocatorResource));
    auto deleter = [&](void*) {ortDmlApi.FreeGPUAllocation(dmlAllocatorResource); };
    deleting_unique_ptr<void> dmlAllocatorResourceCleanup(dmlAllocatorResource, deleter);

    size_t tensorByteSize = static_cast<size_t>(d3dResource->GetDesc().Width);
    Ort::Value newValue(
        Ort::Value::CreateTensor(
            memoryInformation,
            dmlAllocatorResource,
            tensorByteSize,
            tensorDimensions.data(),
            tensorDimensions.size(),
            elementDataType
        )
    );

    // Return values and the wrapped resource.
    // TODO: Is there some way to get Ort::Value to just own the D3DResource
    // directly so that it gets freed after execution or session destruction?
    *dmlEpResourceWrapper = dmlAllocatorResource;
    dmlAllocatorResourceCleanup.release();

    return newValue;
}


Ort::Value CreateTensorValueUsingD3DResource(
    ID3D12Device* d3dDevice,
    OrtDmlApi const& ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    std::span<const int64_t> tensorDimensions,
    ONNXTensorElementDataType elementDataType,
    size_t elementByteSize,
    /*out opt*/ ID3D12Resource** d3dResource,
    /*out*/ void** dmlEpResourceWrapper // Must stay alive with Ort::Value.
    )
{
    // Create empty resource (values don't matter because we won't read them back anyway).
    ComPtr<ID3D12Resource> localD3dResource = CreateD3D12ResourceForTensor(
        d3dDevice,
        elementByteSize,
        tensorDimensions
    );
    if (d3dResource)
    {
        localD3dResource->AddRef();
        *d3dResource = localD3dResource;
    }

    return CreateTensorValueFromExistingD3DResource(
        ortDmlApi,
        memoryInformation,
        localD3dResource,
        tensorDimensions,
        elementDataType,
        /*out*/ dmlEpResourceWrapper
    );
}


void WaitForQueueToComplete(ID3D12CommandQueue* queue)
{
    ComPtr<ID3D12Device> device;
    THROW_IF_FAILED(queue->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));
    ComPtr<ID3D12Fence> fence;
    THROW_IF_FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
    THROW_IF_FAILED(queue->Signal(fence, 1));
    THROW_IF_FAILED(fence->SetEventOnCompletion(1, nullptr));
}


void UploadTensorData(
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocator,
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* destinationResource,
    std::span<const std::byte> sourceData
    )
{
    // Get the size of the resource.
    ComPtr<ID3D12Device> d3d12Device;
    THROW_IF_FAILED(commandQueue->GetDevice(IID_PPV_ARGS(&d3d12Device)));
    D3D12_RESOURCE_DESC resourceDesc = destinationResource->GetDesc();
    assert(resourceDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
    const size_t dataSizeInBytes = static_cast<size_t>(resourceDesc.Width);

    // Create intermediate upload resource visible to both CPU and GPU.
    ComPtr<ID3D12Resource> uploadBuffer = CreateD3D12ResourceOfByteSize(d3d12Device, dataSizeInBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);

    // Copy CPU-side data to shared memory that is both CPU and GPU visible.
    size_t clampedDataByteSize = std::min(dataSizeInBytes, sourceData.size());
    std::byte* uploadBufferData = nullptr;
    THROW_IF_FAILED(uploadBuffer->Map(0, nullptr, reinterpret_cast<void**>(&uploadBufferData)));
    memcpy(uploadBufferData, sourceData.data(), clampedDataByteSize);
    uploadBuffer->Unmap(0, nullptr);

    D3D12_RESOURCE_BARRIER const resourceBarrier =
    {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
        .Transition = {
            .pResource = destinationResource,
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_COPY_DEST,
            .StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        },
    };

    // Issue deferred command to copy from the intermediate shared resource to the final GPU resource,
    // and then execute the commands.
    commandList->CopyResource(destinationResource, uploadBuffer);
    commandList->ResourceBarrier(1, &resourceBarrier);
    THROW_IF_FAILED(commandList->Close());
    ID3D12CommandList* commandLists[] = { commandList };
    commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
    WaitForQueueToComplete(commandQueue);

    THROW_IF_FAILED(commandAllocator->Reset());
    THROW_IF_FAILED(commandList->Reset(commandAllocator, nullptr));
}


void DownloadTensorData(
    ID3D12CommandQueue* commandQueue,
    ID3D12CommandAllocator* commandAllocatar,
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* sourceResource,
    std::span<std::byte> destinationData
    )
{
    // Get the size of the resource.
    ComPtr<ID3D12Device> d3d12Device;
    THROW_IF_FAILED(commandQueue->GetDevice(IID_PPV_ARGS(d3d12Device.GetAddressOf())));
    D3D12_RESOURCE_DESC resourceDesc = sourceResource->GetDesc();
    assert(resourceDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
    const size_t dataSizeInBytes = static_cast<size_t>(resourceDesc.Width);

    // Create intermediate upload resource visible to both CPU and GPU.
    ComPtr<ID3D12Resource> downloadBuffer = CreateD3D12ResourceOfByteSize(d3d12Device, dataSizeInBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);

    D3D12_RESOURCE_BARRIER const resourceBarrier =
    {
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
        .Transition = {
            .pResource = sourceResource,
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            .StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE,
        },
    };

    // Copy GPU data into the download buffer.
    commandList->ResourceBarrier(1, &resourceBarrier);
    commandList->CopyResource(downloadBuffer, sourceResource);
    THROW_IF_FAILED(commandList->Close());
    ID3D12CommandList* commandLists[] = { commandList };
    commandQueue->ExecuteCommandLists(static_cast<uint32_t>(std::size(commandLists)), commandLists);
    WaitForQueueToComplete(commandQueue);

    // Copy from shared GPU/CPU memory to ordinary system RAM.
    size_t clampedDataByteSize = std::min(dataSizeInBytes, destinationData.size());
    std::byte* sourceData = nullptr;
    D3D12_RANGE range = {0, clampedDataByteSize };
    THROW_IF_FAILED(downloadBuffer->Map(0, &range, reinterpret_cast<void**>(&sourceData)));
    memcpy(destinationData.data(), sourceData, clampedDataByteSize);
    downloadBuffer->Unmap(0, nullptr);
}
