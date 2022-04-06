# ONNX Runtime DirectML Execution Provider Sample

Dwayne Robinson 2022-04-06

## What is it?
Just testing the DirectML execution provider in ONNX Runtime.

## Usage
- **OS**: Windows 10+, Local copy of [DirectML 1.8](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.8.0).
- **GPU**: DirectX 12 compute capable.
- **Installation**: Portable app, and so just unzip the files into a folder where you want them - no bloated frameworks or dependencies needed.
- **Running**: Command line app.
- **License**: [License.txt](License.txt) tldr: Do whatever you want with the binary at no cost.

## Building
- Standard Visual Studio msbuild project.

## Features
- Uploads weights to GPU rather than starting from GPU tensors.

## Related
- https://github.com/microsoft/DirectML
- https://docs.microsoft.com/en-us/windows/ai/directml/dml-intro
- https://www.nuget.org/packages/Microsoft.AI.DirectML/1.8.0
- https://github.com/microsoft/onnxruntime/
- https://onnx.ai/
