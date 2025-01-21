#include "vector_add.h"

__global__ void vector_add_kernel(const float *a, const float *b, float *c,
                                  int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    c[idx] = a[idx] + b[idx];
  }
}

extern "C"
{
  EXPORT void launch_vector_add(const float *a, const float *b, float *c, int n,
                                cudaStream_t stream)
  {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, b, c, n);
  }
}
