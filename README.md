# ONNX Runtime DirectML Execution Provider Sample

Dwayne Robinson 2022-04-06

## What is it?
Just testing the DirectML execution provider in ONNX Runtime via D3D resources instead of CPU tensors (which incur synchronization costs).

## Usage
- **OS**: Windows 10+.
- **GPU**: DirectX 12 compute capable.
- **Running**: Command line app.
- **License**: [License.txt](License.txt) tldr: Do whatever you want with the code.

## Building
- Standard Visual Studio msbuild project.
- The Nuget dependencies [onnxruntime.dll 1.11](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/) and [directml.dll 1.8.0](https://www.nuget.org/packages/Microsoft.AI.DirectML/) should automatically be copied into your build folder (and after building, it all just works 🤞).

## Related
- https://github.com/microsoft/DirectML
- https://docs.microsoft.com/en-us/windows/ai/directml/dml-intro
- https://www.nuget.org/packages/Microsoft.AI.DirectML/1.8.0
- https://github.com/microsoft/onnxruntime/
- https://onnx.ai/
