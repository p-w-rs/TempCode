#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <chrono>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
} gLogger;

// Helper function to check CUDA errors
#define CHECK_CUDA(status)                                                     \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      std::cerr << "CUDA error: " << ret << std::endl;                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Helper to get size of a tensor
size_t getSizeByDim(const Dims &dims, DataType type) {
  size_t size = 1;
  for (int i = 0; i < dims.nbDims; i++) {
    size *= dims.d[i];
  }

  switch (type) {
  case DataType::kFLOAT:
    return size * 4;
  case DataType::kHALF:
    return size * 2;
  case DataType::kINT8:
    return size * 1;
  case DataType::kINT32:
    return size * 4;
  default:
    return size * 4;
  }
}

int main() {
  const int BATCH_SIZE = 252;
  const int INPUT_H = 160;
  const int INPUT_W = 160;
  const int INPUT_C = 6;
  const int OUTPUT_SIZE = 3;
  const int WARMUP_RUNS = 10;
  const int BENCHMARK_RUNS = 100;

  // Create builder
  IBuilder *builder = createInferBuilder(gLogger);
  if (!builder) {
    std::cerr << "Failed to create builder" << std::endl;
    return 1;
  }

  // Create network - using default flags (no strongly typed)
  INetworkDefinition *network = builder->createNetworkV2(0);
  if (!network) {
    std::cerr << "Failed to create network" << std::endl;
    return 1;
  }

  // Create ONNX parser
  nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
  if (!parser) {
    std::cerr << "Failed to create parser" << std::endl;
    return 1;
  }

  // Parse ONNX model
  std::cout << "Loading ONNX model..." << std::endl;
  if (!parser->parseFromFile(
          "refine_model.onnx",
          static_cast<int32_t>(ILogger::Severity::kWARNING))) {
    std::cerr << "Failed to parse ONNX model" << std::endl;
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      std::cerr << parser->getError(i)->desc() << std::endl;
    }
    return 1;
  }
  std::cout << "ONNX model loaded successfully" << std::endl;

  // Print network information
  std::cout << "\nNetwork Information:" << std::endl;
  for (int i = 0; i < network->getNbInputs(); i++) {
    ITensor *input = network->getInput(i);
    std::cout << "Input " << i << " (" << input->getName() << "): ";
    Dims dims = input->getDimensions();
    for (int j = 0; j < dims.nbDims; j++) {
      std::cout << dims.d[j] << " ";
    }
    std::cout << std::endl;
  }

  // Build configuration
  IBuilderConfig *config = builder->createBuilderConfig();
  if (!config) {
    std::cerr << "Failed to create builder config" << std::endl;
    return 1;
  }

  // Set workspace size (4GB)
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 4ULL << 30);

  // Enable FP16 if supported
  if (builder->platformHasFastFp16()) {
    config->setFlag(BuilderFlag::kFP16);
    std::cout << "FP16 support enabled" << std::endl;
  }

  // Create optimization profile for dynamic batch size
  IOptimizationProfile *profile = builder->createOptimizationProfile();

  // Set dimensions for inputs - NHWC format based on your model
  for (int i = 0; i < network->getNbInputs(); i++) {
    ITensor *input = network->getInput(i);
    const char *input_name = input->getName();

    // NHWC format: (batch, height, width, channels)
    Dims4 dims{BATCH_SIZE, INPUT_H, INPUT_W, INPUT_C};

    profile->setDimensions(input_name, OptProfileSelector::kMIN, dims);
    profile->setDimensions(input_name, OptProfileSelector::kOPT, dims);
    profile->setDimensions(input_name, OptProfileSelector::kMAX, dims);

    std::cout << "Set dimensions for " << input_name << ": " << BATCH_SIZE
              << "x" << INPUT_H << "x" << INPUT_W << "x" << INPUT_C
              << std::endl;
  }

  config->addOptimizationProfile(profile);

  // Build engine
  std::cout << "\nBuilding TensorRT engine (this may take a while)..."
            << std::endl;
  IHostMemory *serializedModel =
      builder->buildSerializedNetwork(*network, *config);
  if (!serializedModel) {
    std::cerr << "Failed to build serialized network" << std::endl;
    return 1;
  }

  // Clean up builder objects
  delete parser;
  delete network;
  delete config;
  delete builder;

  // Create runtime and deserialize engine
  IRuntime *runtime = createInferRuntime(gLogger);
  ICudaEngine *engine = runtime->deserializeCudaEngine(serializedModel->data(),
                                                       serializedModel->size());
  delete serializedModel;

  if (!engine) {
    std::cerr << "Failed to create engine" << std::endl;
    return 1;
  }
  std::cout << "Engine built successfully!" << std::endl;

  // Create execution context
  IExecutionContext *context = engine->createExecutionContext();
  if (!context) {
    std::cerr << "Failed to create execution context" << std::endl;
    return 1;
  }

  // Get tensor names and set input shapes
  int nbIOTensors = engine->getNbIOTensors();
  std::vector<const char *> inputNames, outputNames;

  std::cout << "\nConfiguring tensors:" << std::endl;
  for (int i = 0; i < nbIOTensors; i++) {
    const char *tensorName = engine->getIOTensorName(i);
    TensorIOMode ioMode = engine->getTensorIOMode(tensorName);

    if (ioMode == TensorIOMode::kINPUT) {
      inputNames.push_back(tensorName);
      // Set input shape for dynamic dimensions
      Dims4 dims{BATCH_SIZE, INPUT_H, INPUT_W, INPUT_C};
      context->setInputShape(tensorName, dims);
      std::cout << "Set input shape for " << tensorName << std::endl;
    } else {
      outputNames.push_back(tensorName);
    }
  }

  std::cout << "Found " << inputNames.size() << " inputs and "
            << outputNames.size() << " outputs" << std::endl;

  // Allocate device memory
  std::vector<void *> buffers;
  std::vector<size_t> bufferSizes;

  // Allocate input buffers
  for (const char *name : inputNames) {
    Dims dims = context->getTensorShape(name);
    DataType dtype = engine->getTensorDataType(name);
    size_t size = getSizeByDim(dims, dtype);

    void *deviceBuffer;
    CHECK_CUDA(cudaMalloc(&deviceBuffer, size));
    buffers.push_back(deviceBuffer);
    bufferSizes.push_back(size);

    context->setTensorAddress(name, deviceBuffer);
    std::cout << "Allocated " << size << " bytes for input: " << name
              << std::endl;
  }

  // Allocate output buffers
  for (const char *name : outputNames) {
    Dims dims = context->getTensorShape(name);
    DataType dtype = engine->getTensorDataType(name);
    size_t size = getSizeByDim(dims, dtype);

    void *deviceBuffer;
    CHECK_CUDA(cudaMalloc(&deviceBuffer, size));
    buffers.push_back(deviceBuffer);
    bufferSizes.push_back(size);

    context->setTensorAddress(name, deviceBuffer);
    std::cout << "Allocated " << size << " bytes for output: " << name
              << std::endl;
  }

  // Initialize input data with dummy values
  std::vector<float> hostInput(BATCH_SIZE * INPUT_H * INPUT_W * INPUT_C, 1.0f);

  // Copy input data to device
  for (size_t i = 0; i < inputNames.size(); i++) {
    CHECK_CUDA(cudaMemcpy(buffers[i], hostInput.data(), bufferSizes[i],
                          cudaMemcpyHostToDevice));
  }

  // Create CUDA stream
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Warmup
  std::cout << "\nWarming up..." << std::endl;
  for (int i = 0; i < WARMUP_RUNS; ++i) {
    if (!context->enqueueV3(stream)) {
      std::cerr << "Failed to enqueue" << std::endl;
      return 1;
    }
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Benchmark
  std::cout << "Running benchmark with " << BENCHMARK_RUNS << " iterations..."
            << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < BENCHMARK_RUNS; ++i) {
    if (!context->enqueueV3(stream)) {
      std::cerr << "Failed to enqueue" << std::endl;
      return 1;
    }
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  double avgTimeMs = duration.count() / 1000.0 / BENCHMARK_RUNS;
  double fps = 1000.0 / avgTimeMs;

  std::cout << "\n========== Benchmark Results ==========" << std::endl;
  std::cout << "Model: refine_model.onnx" << std::endl;
  std::cout << "Precision: FP16" << std::endl;
  std::cout << "Batch size: " << BATCH_SIZE << std::endl;
  std::cout << "Input shape: " << BATCH_SIZE << "x" << INPUT_H << "x" << INPUT_W
            << "x" << INPUT_C << std::endl;
  std::cout << "Average inference time: " << avgTimeMs << " ms" << std::endl;
  std::cout << "Throughput: " << fps << " FPS" << std::endl;
  std::cout << "Total samples/second: " << fps * BATCH_SIZE << std::endl;
  std::cout << "======================================" << std::endl;

  // Cleanup
  CHECK_CUDA(cudaStreamDestroy(stream));
  for (void *buffer : buffers) {
    CHECK_CUDA(cudaFree(buffer));
  }

  delete context;
  delete engine;
  delete runtime;

  return 0;
}
