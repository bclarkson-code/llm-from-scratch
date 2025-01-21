/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_common.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#define CUBLAS_LOWP CUDA_R_32F

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4
// is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
int cuda_arch_major = 0;
int cuda_arch_minor = 0;
int cuda_num_SMs =
    0;  // for persistent threads where we want 1 threadblock per SM
int cuda_threads_per_SM =
    0;  // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
    exit(EXIT_FAILURE);
  }
}
#define cublasCheck(status) \
  { cublasCheck((status), __FILE__, __LINE__); }

void setup_main() {
  srand(0);  // determinism

  // set up the device
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceIdx);
  cuda_num_SMs = deviceProp.multiProcessorCount;
  cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
  cuda_arch_major = deviceProp.major;
  cuda_arch_minor = deviceProp.minor;

  // setup cuBLAS and cuBLASLt
  cublasCheck(cublasCreate(&cublas_handle));
  cublasCheck(cublasLtCreate(&cublaslt_handle));
  cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

  // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
  int enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
  // TODO implement common CLI for all tests/benchmarks
  // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via
  // arg
  cublas_compute_type =
      enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublasMath_t cublas_math_mode =
      enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

#endif  // CUBLAS_COMMON_H
