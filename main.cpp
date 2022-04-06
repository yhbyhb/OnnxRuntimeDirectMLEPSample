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
#include <wrl/client.h>

#include "headers/dml_provider_factory.h"
#include "headers/onnxruntime_cxx_api.h"

////////////////////////////////////////////////////////////////////////////////

#define THROW_IF_FAILED(hr) {HRESULT localHr = (hr); if (FAILED(hr)) throw hr;}
#define RETURN_IF_FAILED(hr) {HRESULT localHr = (hr); if (FAILED(hr)) return hr;}
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

Ort::Value CreateTensorValueUsingD3DResource(
    ID3D12Device* d3dDevice,
    OrtDmlApi const& ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    std::span<const int64_t> dimensions,
    ONNXTensorElementDataType elementDataType,
    size_t elementByteSize,
    /*out*/ void** dmlEpResourceWrapper
);

////////////////////////////////////////////////////////////////////////////////

int main()
{
    #if 1
    const wchar_t* modelFilePath = L"Upsample4xOpset11.onnx";
    const char* modelInputTensorName = "input";
    const char* modelOutputTensorName = "output";
    const std::array<int64_t, 4> inputShape = { 1, 3, 1000, 1000 };
    const std::array<int64_t, 4> outputShape = { 1, 3, 4000, 4000 };
    ONNXTensorElementDataType inputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    ONNXTensorElementDataType outputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    using inputDataTypeT = float;
    using outputDataTypeT = float;
    #elif 0
    // Squeezenet opset v7 https://github.com/onnx/models/blob/master/vision/classification/squeezenet/README.md
    const wchar_t* modelFilePath = L"D:/ai/models/squeezenet/SqueezeNet.onnx";
    const char* modelInputTensorName = "data_0";
    const char* modelOutputTensorName = "softmaxout_1";
    const std::array<int64_t, 4> inputShape = { 1, 3, 224, 224 };
    const std::array<int64_t, 4> outputShape = { 1, 1000, 1, 1 };
    ONNXTensorElementDataType inputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    ONNXTensorElementDataType outputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    using inputDataTypeT = float;
    using outputDataTypeT = float;
    #elif 1
    const wchar_t* modelFilePath = L"S:/WindowsAI/build/x64-win-redist-debug/install/bin/OnnxBackendTestData/test_nonzero_example/model.onnx";
    const char* modelInputTensorName = "condition";
    const char* modelOutputTensorName = "result";
    const std::array<int64_t, 2> inputShape = { 2, 2 };
    const std::array<int64_t, 2> outputShape = { 2, 4 };
    ONNXTensorElementDataType inputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    ONNXTensorElementDataType outputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    using inputDataTypeT = bool;
    using outputDataTypeT = int64_t;
    #elif 0
    const wchar_t* modelFilePath = L"S:/WindowsAI/build/x64-win-redist-debug/install/bin/OnnxBackendTestData/test_shape/model.onnx";
    const char* modelInputTensorName = "x";
    const char* modelOutputTensorName = "y";
    const std::array<int64_t, 3> inputShape = { 3, 4, 5 };
    const std::array<int64_t, 1> outputShape = { 3 };
    ONNXTensorElementDataType inputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    ONNXTensorElementDataType outputDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    using inputDataTypeT = float;
    using outputDataTypeT = int64_t;
    #endif

    const bool passTensorsAsD3DResources = false;

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
        Microsoft::WRL::ComPtr<ID3D12Device> d3d12Device;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList;
        Microsoft::WRL::ComPtr<IDMLCommandRecorder> commandRecorder;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptorHeap;
        Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> downloadBuffer;

        THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12Device)));
        QueryPerformanceCounter(&d3dDeviceCreationTime);

        D3D12_COMMAND_QUEUE_DESC commandQueueDescription = {};
        commandQueueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        commandQueueDescription.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

        THROW_IF_FAILED(d3d12Device->CreateCommandQueue(
            &commandQueueDescription,
            IID_PPV_ARGS(&commandQueue)
        ));

        THROW_IF_FAILED(d3d12Device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&commandAllocator)
        ));

        THROW_IF_FAILED(d3d12Device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            commandAllocator.Get(),
            nullptr,
            IID_PPV_ARGS(&commandList)
        ));

        D3D12_HEAP_PROPERTIES uploadHeapProperties = {
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            D3D12_MEMORY_POOL_UNKNOWN,
            0,
            0
        };
        size_t inputBufferByteSize = ByteSizeOfOnnxTensorElementDataType(inputDataType) * GetElementCount(inputShape);
        D3D12_RESOURCE_DESC uploadResourceDesc = {
            D3D12_RESOURCE_DIMENSION_BUFFER,
            0,
            static_cast<uint64_t>(inputBufferByteSize),
            1,
            1,
            1,
            DXGI_FORMAT_UNKNOWN,
            {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        };

        THROW_IF_FAILED(d3d12Device->CreateCommittedResource(
            &uploadHeapProperties,
            D3D12_HEAP_FLAG_NONE,
            &uploadResourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&uploadBuffer)
        ));

        OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
        const OrtDmlApi* ortDmlApi;
        THROW_IF_NOT_OK(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));

        // ONNX Runtime setup
        Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "DirectML_Direct3D_TensorAllocation_Test");
        //Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_VERBOSE, "DirectML_Direct3D_TensorAllocation_Test");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.DisableMemPattern();
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ortApi.AddFreeDimensionOverrideByName(sessionOptions, "batch_size", 1);
        //ortApi.SetOptimizedModelFilePath(sessionOptions, L"optimized.ort");
        // todo: change this to an explicit device id, not just 0 using adapter above.
        ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, 0);

        printf("Loading model.\n");
        Ort::Session session = Ort::Session(ortEnvironment, modelFilePath, sessionOptions);

        #if 0
        /// hack:::
        // Test reload of optimized model.
        // If you comment out "*isDmlGraphNode = true;" in GraphPartitioner.cpp - GetRegistrationProperties(),
        // then this works.
        Ort::OrtRelease(session.release());
        ortApi.SetOptimizedModelFilePath(sessionOptions, L"");
        Ort::Session session = Ort::Session(ortEnvironment, L"O:\\out.ort", sessionOptions);
        /// :::hack
        #endif

        QueryPerformanceCounter(&sessionCreationTime);

        Ort::IoBinding ioBinding = Ort::IoBinding::IoBinding(session);
        const char* memoryInformationName = passTensorsAsD3DResources ? "DML" : "Cpu";
        Ort::MemoryInfo memoryInformation(memoryInformationName, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
        // Not needed: Ort::Allocator allocator(session, memoryInformation);

        // Create input tensor.
        Ort::Value inputTensor(nullptr);
        std::vector<WrapperClass<inputDataTypeT>> inputTensorValues(static_cast<size_t>(GetElementCount(inputShape)), inputDataTypeT(0));
        if constexpr (std::is_same_v<inputDataTypeT, bool>) // Why, C++, why?... Just let me have a sequence of {false, true, false, true...}.
        {
            std::fill(inputTensorValues.begin(), inputTensorValues.end(), inputDataTypeT(1));
        }
        else
        {
            std::iota(inputTensorValues.begin(), inputTensorValues.end(), inputDataTypeT(0));
        }
        Microsoft::WRL::ComPtr<IUnknown> inputTensorEpWrapper;

        if (passTensorsAsD3DResources)
        {
            // Create empty D3D resource for input.
            inputTensor = CreateTensorValueUsingD3DResource(
                d3d12Device.Get(),
                *ortDmlApi,
                memoryInformation,
                inputShape,
                inputDataType,
                ByteSizeOfOnnxTensorElementDataType(inputDataType),
                /*out*/ IID_PPV_ARGS_Helper(inputTensorEpWrapper.GetAddressOf())
            );
            #if 0
            UploadTensorData();//!!!
            #endif
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
        Microsoft::WRL::ComPtr<IUnknown> outputTensorEpWrapper;

        if (passTensorsAsD3DResources)
        {
            outputTensor = CreateTensorValueUsingD3DResource(
                d3d12Device.Get(),
                *ortDmlApi,
                memoryInformation,
                outputShape,
                outputDataType,
                ByteSizeOfOnnxTensorElementDataType(outputDataType),
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

        // TODO: Upload inputTensorValues to GPU inputTensor.

        printf("Beginning execution.\n");
        QueryPerformanceCounter(&runStartTime);
        session.Run(runOptions, ioBinding);
        QueryPerformanceCounter(&runTime);
        ioBinding.SynchronizeOutputs();
        QueryPerformanceCounter(&synchronizeOutputsTime);
        runEndTime = synchronizeOutputsTime;
        printf("Finished execution.\n");

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

        // TODO: Download outputTensorValues from GPU outputTensor.

        ////////////////////////////////////////
        // Print the top results if the output tensors were on the CPU.
        if (!passTensorsAsD3DResources)
        {
            #if 0
            #if 1 // Print first 10 values.
            for (int i = 0; i <= std::min(outputTensorValues.size(), size_t(10)); ++i)
            {
                printf("output[%d] = %f\n", i, outputTensorValues[i]);
            }
            #else // Print top 10.
            std::vector<uint32_t> indices(outputTensorValues.size(), 0);
            std::iota(indices.begin(), indices.end(), 0);
            sort(
                indices.begin(),
                indices.end(),
                [&](uint32_t a, uint32_t b)
                {
                    return (outputTensorValues[a] > outputTensorValues[b]);
                }
            );
            for (int i = 0; i <= std::min(indices.size(), size_t(10)); ++i)
            {
                printf("output[%d] = %f\n", indices[i], outputTensorValues[indices[i]]);
            }
            #endif
            #endif
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

Microsoft::WRL::ComPtr<ID3D12Resource> CreateD3D12ResourceForTensor(
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
    bufferByteSize = std::max(bufferByteSize, size_t(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));

    // DML needs the resources' sizes to be a multiple of 4 bytes
    (bufferByteSize += 3) &= ~3;

    D3D12_HEAP_PROPERTIES heapProperties = {
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN,
        0,
        0
    };
    D3D12_RESOURCE_DESC resourceDesc = {
        D3D12_RESOURCE_DIMENSION_BUFFER,
        0,
        static_cast<uint64_t>(bufferByteSize),
        1,
        1,
        1,
        DXGI_FORMAT_UNKNOWN,
        {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    };

    Microsoft::WRL::ComPtr<ID3D12Resource> gpuResource;
    THROW_IF_FAILED(d3dDevice->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        __uuidof(ID3D12Resource),
        /*out*/ &gpuResource
    ));

    return gpuResource;
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
    /*out*/ void** dmlEpResourceWrapper // Must stay alive with Ort::Value.
    )
{
    // Create empty resource (values don't matter because we won't read them back anyway).
    Microsoft::WRL::ComPtr<ID3D12Resource> d3dResource = CreateD3D12ResourceForTensor(
        d3dDevice,
        elementByteSize,
        tensorDimensions
    );

    return CreateTensorValueFromExistingD3DResource(
        ortDmlApi,
        memoryInformation,
        d3dResource.Get(),
        tensorDimensions,
        elementDataType,
        /*out*/ dmlEpResourceWrapper
    );
}

#if 0

https://github.com/microsoft/DirectML/blob/master/Samples/HelloDirectML/main.cpp
https://docs.microsoft.com/en-us/windows/win32/direct3d12/uploading-resources
https://github.com/microsoft/DirectML/blob/master/Python/src/device.cpp


void UploadTensorData(
    ID3D12Device* d3dDevice,
    ID3D12Resource* d3dResource,
    std::span<const std::byte> data
    )
{
    size_t dataByteSize = data.size();
    std::byte* uploadBufferData = nullptr;
    THROW_IF_FAILED(uploadBuffer->Map(0, nullptr, reinterpret_cast<void**>(&uploadBufferData)));

    memcpy(uploadBufferData, data.data(), data.size());
    uploadBuffer->Unmap(0, nullptr);

    // Record the copy from the upload heap into the inputs resource
    commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            d3dResource,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST
        )
    );
    commandList->CopyBufferRegion(d3dResource.Get(), 0, uploadBuffer.Get(), 0, dataByteSize);
    commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            d3dResource,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        )
    );

    m_commandList->SetDescriptorHeaps(1, m_descriptorHeap.GetAddressOf());
    m_commandRecorder->RecordDispatch(m_commandList.Get(), m_operatorInitializer.Get(), m_bindingTable.Get());
    
    THROW_IF_FAILED(m_commandList->Close());

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

    WaitForQueueToComplete(m_commandQueue);

    THROW_IF_FAILED(m_commandAllocator->Reset());
    THROW_IF_FAILED(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
}
#endif

#if 0

    std::array<FLOAT, tensorElementCount> inputTensorElementArray;
    {
        std::wcout << L"input tensor: ";
        for (auto & element : inputTensorElementArray)
        {
            element = 1.618f;
            std::wcout << element << L' ';
        };
        std::wcout << std::endl;

        D3D12_SUBRESOURCE_DATA tensorSubresourceData{};
        tensorSubresourceData.pData = inputTensorElementArray.data();
        tensorSubresourceData.RowPitch = static_cast<LONG_PTR>(tensorBufferSize);
        tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

        // Upload the input tensor to the GPU.
        ::UpdateSubresources(
            commandList.get(),
            inputBuffer.get(),
            uploadBuffer.get(),
            0,
            0,
            1,
            &tensorSubresourceData);

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                inputBuffer.get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            )
        );
    }
#endif


#if 0
void Device::RecordOutputReadBack(uint64_t outputsResourceSize)
{
    // Copy output to readback heap
    if (outputsResourceSize != 0)
    {
        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputsResource->GetResource(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE)
            );

        m_commandList->CopyBufferRegion(m_downloadBuffer->GetResource(), 0, m_outputsResource->GetResource(), 0, outputsResourceSize);

        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputsResource->GetResource(),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            );
    }
}
#endif

#if 0
std::vector<pydml::TensorData*> DownloadFromDownloadBuffer(
    uint64_t outputsResourceSize, 
    std::vector<dml::Expression*>& outputs,
    std::vector<DmlBufferBinding>& outputBindings
    )
{
    std::vector<pydml::TensorData*> outputData;

    if (outputsResourceSize != 0)
    {
        CD3DX12_RANGE readRange(0, static_cast<size_t>(outputsResourceSize));

        byte* downloadBufferData = nullptr;

        ThrowIfFailed(m_downloadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&downloadBufferData)));

        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto output = outputs[i];

            if (!output)
            {
                // This output tensor is optional (and null)
                continue;
            }

            dml::TensorDesc desc = output->GetOutputDesc();
            DmlBufferTensorDesc bufferDesc = *desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

            auto data = new TensorData(&desc);
            void* dest = data->Get();
            const void* src = downloadBufferData + outputBindings[i].offset;

            memcpy(dest, src, static_cast<size_t>(bufferDesc.totalTensorSizeInBytes));

            outputData.push_back(data);
        }

        m_downloadBuffer->Unmap(0, nullptr);
    }

    return outputData;
}
#endif

#if 0
void Device::ExecuteCommandListAndWait()
{
    ThrowIfFailed(m_commandList->Close());

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    if (m_residencyManager != nullptr)
    {
        gpgmm::d3d12::ResidencySet* residencySets[] = { &m_residencySet };
        m_residencyManager->ExecuteCommandLists(m_commandQueue.Get(), commandLists, residencySets, ARRAYSIZE(commandLists));
    }
    else
    {
        m_commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
    }

    WaitForQueueToComplete(m_commandQueue.Get());

    ThrowIfFailed(m_commandAllocator->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocator.Get(), nullptr));

    if (m_residencyManager != nullptr)
    {
        ThrowIfFailed(m_residencySet.Reset());
    }
}
#endif


#if 0
    com_ptr<ID3D12Resource> downloadBuffer;
    check_hresult(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        __uuidof(downloadBuffer),
        downloadBuffer.put_void()));

    commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            outputBuffer.get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE
        )
    );

    commandList->CopyResource(downloadBuffer.get(), outputBuffer.get());

    CloseExecuteResetWait(d3D12Device, commandQueue, commandAllocator, commandList);

    D3D12_RANGE tensorBufferRange{ 0, static_cast<SIZE_T>(tensorBufferSize) };
    FLOAT* outputBufferData{};
    check_hresult(downloadBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void**>(&outputBufferData)));
#endif
