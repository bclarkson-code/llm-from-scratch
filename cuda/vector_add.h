#pragma once
#include <cuda_runtime.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

EXPORT void launch_vector_add(const float* a, const float* b, float* c,
                            int n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
