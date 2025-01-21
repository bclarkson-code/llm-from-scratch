#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H
#include <cuda_runtime.h>

typedef float floatX;
#define WARP_SIZE 32U
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// CUDA error checking. Underscore added so this function can be called directly
// not just via macro
inline void cudaCheck_(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
template <class T>
inline void cudaFreeCheck_(T **ptr, const char *file, int line) {
  cudaError_t error = cudaFree(*ptr);
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
  *ptr = nullptr;
}
#define cudaFreeCheck(ptr) (cudaFreeCheck_(ptr, __FILE__, __LINE__))
#endif
