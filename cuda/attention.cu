#include <assert.h>
#include <cublasLt.h>
#include <cublas_common.h>
#include <cuda_common.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <matmul.cuh>

const float FLT_MAX =
    340282346638528859811704183484516925440.0f;  // to avoid including float.h
// ---------------------------------CPU-----------------------------------
void attention_forward_cpu_kernel(float *out, float *preatt, float *att,
                                  const float *inp, int B, int T, int C,
                                  int NH) {
  // input is (B, T, 3C) Q,K,V
  // preatt, att are (B, NH, T, T)
  // output is (B, T, C)
  int C3 = C * 3;
  int hs = C / NH;  // head size
  float scale = 1.0 / sqrtf(hs);

  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < NH; h++) {
        const float *query_t = inp + b * T * C3 + t * C3 + h * hs;
        float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;

        // pass 1: calculate query dot key and maxval
        float maxval = -FLT_MAX;
        for (int t2 = 0; t2 <= t; t2++) {
          const float *key_t2 =
              inp + b * T * C3 + t2 * C3 + h * hs + C;  // +C because it's key

          // (query_t) dot (key_t2)
          float val = 0.0f;
          for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
          }
          val *= scale;
          if (val > maxval) {
            maxval = val;
          }

          preatt_bth[t2] = val;
        }
        // pad with -INFINITY outside of autoregressive region for debugging
        // comparisons
        for (int t2 = t + 1; t2 < T; t2++) {
          preatt_bth[t2] = -INFINITY;
        }

        // pass 2: calculate the exp and keep track of sum
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
          float expv = expf(preatt_bth[t2] - maxval);
          expsum += expv;
          att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // pass 3: normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
          if (t2 <= t) {
            att_bth[t2] *= expsum_inv;
          } else {
            // causal attention mask. not strictly necessary to set to zero here
            // only doing this explicitly for debugging and checking to PyTorch
            att_bth[t2] = 0.0f;
          }
        }

        // pass 4: accumulate weighted values into the output of attention
        float *out_bth = out + b * T * C + t * C + h * hs;
        for (int i = 0; i < hs; i++) {
          out_bth[i] = 0.0f;
        }
        for (int t2 = 0; t2 <= t; t2++) {
          const float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs +
                                  C * 2;  // +C*2 because it's value
          float att_btht2 = att_bth[t2];
          for (int i = 0; i < hs; i++) {
            out_bth[i] += att_btht2 * value_t2[i];
          }
        }
      }
    }
  }
}

void attention_backward_cpu_kernel(float *dinp, float *dpreatt, float *datt,
                                   float *dout, float *inp, float *att, int B,
                                   int T, int C, int NH) {
  // inp/dinp are (B, T, 3C) Q,K,V
  // att/datt/dpreatt are (B, NH, T, T)
  // dout is (B, T, C)
  int C3 = C * 3;
  int hs = C / NH;  // head size
  float scale = 1.0 / sqrtf(hs);

  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < NH; h++) {
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;
        float *datt_bth = datt + b * NH * T * T + h * T * T + t * T;
        float *dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
        float *dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
        float *query_t = inp + b * T * C3 + t * C3 + h * hs;

        // backward pass 4, through the value accumulation
        float *dout_bth = dout + b * T * C + t * C + h * hs;
        for (int t2 = 0; t2 < T;
             t2++) {  // ADJUSTED! this was t2 <= t (see note on function)
          float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs +
                            C * 2;  // +C*2 because it's value
          float *dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;
          for (int i = 0; i < hs; i++) {
            // in the forward pass this was:
            // out_bth[i] += att_bth[t2] * value_t2[i];
            // so now we have:
            datt_bth[t2] += value_t2[i] * dout_bth[i];
            dvalue_t2[i] += att_bth[t2] * dout_bth[i];
          }
        }

        // backward pass 2 & 3, the softmax
        // note that softmax (like e.g. tanh) doesn't need the input (preatt) to
        // backward
        for (int t2 = 0; t2 <= t; t2++) {
          for (int t3 = 0; t3 <= t; t3++) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
            dpreatt_bth[t3] += scale * local_derivative * datt_bth[t2];
          }
        }

        // backward pass 1, the query @ key matmul
        for (int t2 = 0; t2 <= t; t2++) {
          float *key_t2 =
              inp + b * T * C3 + t2 * C3 + h * hs + C;  // +C because it's key
          float *dkey_t2 =
              dinp + b * T * C3 + t2 * C3 + h * hs + C;  // +C because it's key
          for (int i = 0; i < hs; i++) {
            // in the forward pass this was:
            // preatt_bth[t2] += query_t[i] * key_t2[i]
            // so now we have:
            dquery_t[i] += key_t2[i] * dpreatt_bth[t2];
            dkey_t2[i] += query_t[i] * dpreatt_bth[t2];
          }
        }
      }
    }
  }
}
// ---------------------------------GPU-----------------------------------
__device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

__device__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

// requires all 32 threads in the warp to be active, but should work for any
// block size uses non-dynamic shared memory so every call increases shared
// memory requirements by 128 bytes the fact it's unique shared memory allows us
// to avoid an extra __syncthreads() call at the end but if called inside a
// loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = float (*)(float);

template <reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync,
                                    float out_of_bounds) {
  // two reductions of up to 1024 threads:
  // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp
  // (shuffle)
  __shared__ float shared_val[WARP_SIZE];
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;

  float warp_val = warp_reduction(val);
  if (lane_id == 0) {
    shared_val[warp_id] = warp_val;
  }
  __syncthreads();
  warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
  float block_val = warp_reduction(warp_val);

  if (final_sync) {
    __syncthreads();  // only needed in loops when effectively reusing shared
                      // memory etc.
  }
  return block_val;
}

// Helper function to call blockReduce with default arguments
template <reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val) {
  return blockReduce<warp_reduction>(val, false, 0.0f);
}

// inputs floatX, outputs FP32 (for current FP32-only activation path for this
// WIP)
__global__ void permute_kernel(floatX *qkvr, const floatX *inp, int B, int N,
                               int NH, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * NH * N * d) {
    return;
  }

  // Calculate indices
  int b = idx / (NH * N * d);
  int rest = idx % (NH * N * d);
  int nh_ = rest / (N * d);
  rest = rest % (N * d);
  int n = rest / d;
  int d_ = rest % d;

  // Calculate input index
  int inp_idx =
      (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;

  // Write Q,K,V into contiguous sections of qkvr
  qkvr[idx] = __ldcs(&inp[inp_idx]);                            // Q section
  qkvr[idx + B * NH * N * d] = __ldcs(&inp[inp_idx + NH * d]);  // K section
  qkvr[idx + 2 * B * NH * N * d] =
      __ldcs(&inp[inp_idx + 2 * NH * d]);  // V section
}

__global__ void permute_kernel_backward(floatX *dinp, const floatX *dq,
                                        const floatX *dk, const floatX *dv,
                                        int B, int N, int NH, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * NH * N * d) {
    return;
  }

  int b = idx / (NH * N * d);
  int rest = idx % (NH * N * d);
  int nh_ = rest / (N * d);
  rest = rest % (N * d);
  int n = rest / d;
  int d_ = rest % d;

  int inp_idx =
      (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
  dinp[inp_idx] = dq[idx];
  dinp[inp_idx + NH * d] = dk[idx];
  dinp[inp_idx + 2 * (NH * d)] = dv[idx];
}

__global__ void unpermute_kernel(floatX *inp, floatX *out, int B, int N, int NH,
                                 int d) {
  // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)

  int idx = (blockIdx.x * blockDim.x + threadIdx.x);
  // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
  if (idx >= B * NH * N * d) {
    return;
  }

  int b = idx / (NH * N * d);
  int rest = idx % (NH * N * d);
  int nh_ = rest / (N * d);
  rest = rest % (N * d);
  int n = rest / d;
  int d_ = rest % d;
  int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
  out[other_idx] = __ldcs(&inp[idx]);
}

__global__ void unpermute_kernel_backward(floatX *dinp, const floatX *dout,
                                          int B, int N, int NH, int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * NH * N * d) {
    return;
  }

  int b = idx / (NH * N * d);
  int rest = idx % (NH * N * d);
  int nh_ = rest / (N * d);
  rest = rest % (N * d);
  int n = rest / d;
  int d_ = rest % d;
  int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
  dinp[idx] = (floatX)dout[other_idx];
}
__global__ void softmax_forward_kernel1(floatX *att, float scale,
                                        floatX *preatt, int BNH, int T) {
  // Thread and block indexing
  int batch_idx = blockIdx.x;
  int head_idx = threadIdx.x / T;
  int token_idx = threadIdx.x % T;

  // Validate index within bounds
  if (batch_idx >= BNH || head_idx >= T) return;

  // Shared memory for max and sum computation
  extern __shared__ floatX shared_mem[];

  floatX *max_vals = shared_mem;
  floatX *sum_vals = shared_mem + T;

  // Find maximum value for numerical stability
  floatX max_val = -FLT_MAX;
  for (int i = 0; i < T; i++) {
    max_val = fmaxf(max_val, preatt[batch_idx * T * T + head_idx * T + i]);
  }
  max_vals[token_idx] = max_val;

  // Synchronize threads in block
  __syncthreads();

  // Compute sum of exponentials after shifting by max_val
  floatX sum = 0.0f;
  for (int i = 0; i < T; i++) {
    floatX shifted_val =
        preatt[batch_idx * T * T + head_idx * T + i] - max_vals[token_idx];
    floatX exp_val = expf(shifted_val * scale);
    sum += exp_val;
    preatt[batch_idx * T * T + head_idx * T + i] = exp_val;  // Store exp result
  }
  sum_vals[token_idx] = sum;

  // Synchronize threads in block
  __syncthreads();

  // Normalize values to produce softmax probabilities
  for (int i = 0; i < T; i++) {
    preatt[batch_idx * T * T + head_idx * T + i] /= sum_vals[token_idx];
    att[batch_idx * T * T + head_idx * T + i] =
        preatt[batch_idx * T * T + head_idx * T + i];
  }
}

__global__ void softmax_forward_kernel5(floatX *out, float inv_temperature,
                                        const floatX *inp, int N, int T) {
  // inp, out shape: (N, T, T), where N = B * NH
  // fuses the multiplication by scale inside attention
  // directly autoregressive, so we only compute the lower triangular part
  // uses the online softmax algorithm
  assert(T % 4 == 0);
  int lane_id = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  int num_warps = blockDim.x / WARP_SIZE;

  // micro-optimization: we iterate backwards so that
  // after the softmax backward operation completes, the cache retains the
  // part of the matrix close to the upper left corner, which benefits the
  // matmul operation that immediately follows.
  // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); //
  // forward order
  int idx =
      (gridDim.x - blockIdx.x - 1) * num_warps + warp_id;  // backward order
  if (idx >= N * T) {
    return;
  }
  int own_pos = idx % T;
  int pos_by_4 = own_pos / 4;

  // one row of inp, i.e. inp[idx, :] of shape (T,)
  const floatX *x = inp + idx * T;

  // not INF, so we don't get NaNs accidentally when subtracting two values.
  const float flt_max =
      340282346638528859811704183484516925440.0f;  // to avoid including float.h
  float maxval = -flt_max;
  float sumval = 0.0f;

  const floatX *x_aligned =
      reinterpret_cast<const floatX *>(__builtin_assume_aligned(x, 16));
  for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
    float regarray[4];
    for (int k = 0; k < 4; ++k) {
      regarray[k] = (float)x_aligned[4 * i + k];
    }
    float old_maxval = maxval;
    for (int k = 0; k < 4; ++k) {
      maxval = fmaxf(maxval, regarray[k]);
    }
    sumval *= expf(inv_temperature * (old_maxval - maxval));
    for (int k = 0; k < 4; ++k) {
      sumval += expf(inv_temperature * (regarray[k] - maxval));
    }
  }

  if (4 * pos_by_4 + lane_id <= own_pos) {
    float old_maxval = maxval;
    maxval = fmaxf(maxval, (float)x[4 * pos_by_4 + lane_id]);
    sumval *= expf(inv_temperature * (old_maxval - maxval));
    sumval +=
        expf(inv_temperature * ((float)x[4 * pos_by_4 + lane_id] - maxval));
  }

  float global_maxval = warpReduceMax(maxval);
  sumval *= expf(inv_temperature * (maxval - global_maxval));

  float sum = warpReduceSum(sumval);
  float norm = 1.f / sum;

  // divide the whole row by the sum
  for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
    // recalculation is faster than doing the round-trip through memory.
    float ev = expf(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
    __stcs(out + idx * T + i, (floatX)(ev * norm));
  }
}

__global__ void softmax_autoregressive_backward_inplace_kernel(
    floatX *datt, const floatX *att, int B, int T, int C, float scale) {
  constexpr const int BlockSize = 256;
  constexpr int T_per_block = 4;

  // go through blocks in reverse order, so the slowest block starts first
  int t0 = T - 1 - T_per_block * blockIdx.x;
  int idx = blockIdx.y;

  att += idx * T * T;
  datt += idx * T * T;

  for (int to = 0; to < T_per_block; ++to) {
    int t = t0 - to;
    if (t < 0) return;
    const floatX *att_bth = att + t * T;
    const floatX *datt_bth = datt + t * T;
    floatX *dpreatt_bth = datt + t * T;

    float local_sum = 0;
    for (int t2 = threadIdx.x; t2 <= t; t2 += BlockSize) {
      local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
    }

    local_sum = blockReduce<warpReduceSum>(local_sum);

    for (int t3 = threadIdx.x; t3 < T; t3 += BlockSize) {
      // don't touch the cache. Some parts will still be here from the previous
      // loop, and we want to exploit those.
      if (t3 <= t) {
        float acc = (float)__ldcs(att_bth + t3) *
                    ((float)__ldcs(datt_bth + t3) - local_sum);
        __stcs(dpreatt_bth + t3, (floatX)(scale * acc));
      } else {
        // explicitly set non-causal elements to zero
        __stcs(dpreatt_bth + t3, (floatX)0.f);
      }
    }
  }
}
// --------------------------------KERNEL LAUNCHERS-----------------------------
extern "C" {
void initialize_cuda() {
  static bool initialized = false;
  if (!initialized) {
    setup_main();
    initialized = true;
  }
}

// cpu "kernel" launchers
// These don't actaully use block_size of stream but were keeping them here for
// a consistent interface
// void attention_forward_cpu(float *out, float *preatt, float *att,
//                            const float *inp, int B, int T, int C, int NH,
//                            cudaStream_t stream) {
//   attention_forward_cpu_kernel(out, preatt, att, inp, B, T, C, NH);
// }

// void attention_backward_cpu(floatX *dinp, floatX *dqkvr, floatX *datt,
//                             floatX *scratch, const floatX *dout,
//                             const floatX *qkvr, const floatX *att, int B, int
//                             T, int C, int NH, cudaStream_t stream) {
//   attention_backward_cpu_kernel(dinp, dpreatt, datt, dout, scratch, att, B,
//   T,
//                                 C, NH);
// }
void check_nan_inf(floatX *data, const char *name, size_t size) {
  floatX *host_data = (floatX *)malloc(size * sizeof(floatX));
  cudaMemcpy(host_data, data, size * sizeof(floatX), cudaMemcpyDeviceToHost);

  int nan_count = 0, inf_count = 0;
  for (size_t i = 0; i < size; i++) {
    if (isnan(host_data[i])) nan_count++;
    if (isinf(host_data[i])) inf_count++;
  }

  printf("%s: NaN count=%d, Inf count=%d\\n", name, nan_count, inf_count);
  free(host_data);
}

// gpu kernel launchers
void attention_forward_cuda_1(floatX *out, floatX *qkvr, floatX *att,
                              floatX *inp, int B, int T, int C, int NH,
                              cudaStream_t stream) {
  // Note: `inp` is not needed for backward pass, so we re-use it as a scratch
  // buffer. Its contents will be overwritten by this function.
  const int block_size = 256;

  // inp is (B, T, 3C) QKV
  // preatt, att are (B, NH, T, T)
  // output is (B, T, C)
  const int HS = C / NH;  // head size

  // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
  floatX *q, *k, *v;
  q = qkvr + 0 * B * T * C;
  k = qkvr + 1 * B * T * C;
  v = qkvr + 2 * B * T * C;
  int total_threads = B * NH * T * HS;
  int num_blocks = CEIL_DIV(total_threads, block_size);
  permute_kernel<<<num_blocks, block_size, 0, stream>>>(qkvr, inp, B, T, NH,
                                                        HS);

  floatX *preatt = inp;  // reuse inp as scratch buffer
  matmul_cublaslt(preatt, q, k, nullptr, T, T, HS, stream, false, true, B * NH,
                  T * HS, T * HS, T * T);

  // multiply all elements of preatt elementwise by scale
  float scale = 1.f / sqrtf(HS);
  int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
  softmax_forward_kernel1<<<grid_size, block_size, 0, stream>>>(
      att, scale, preatt, B * NH, T);

  // new approach: first cuBLAS another batched matmul
  floatX *vaccum = inp;
  // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
  matmul_cublaslt(vaccum, att, v, nullptr, HS, T, T, stream, false, false,
                  B * NH, T * T, T * HS, T * HS);
  // now unpermute
  // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head
  // outputs side by side
  num_blocks = CEIL_DIV(B * T * C, block_size);
  unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, T, NH,
                                                          HS);
  cudaCheck(cudaGetLastError());
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) ->
// vaccum (B,T,C) -> out (B,T,C)
void attention_backward_cuda_1(floatX *dinp, floatX *dqkvr, floatX *datt,
                               floatX *scratch, const floatX *dout,
                               const floatX *qkvr, const floatX *att, int B,
                               int T, int C, int NH, cudaStream_t stream) {
  const int block_size = 256;
  const int HS = C / NH;  // head size

  // unpack convenience pointers into q, k, v
  const floatX *q, *k, *v;
  q = qkvr + 0 * B * T * C;
  k = qkvr + 1 * B * T * C;
  v = qkvr + 2 * B * T * C;
  floatX *dq, *dk, *dv;
  dq = dqkvr + 0 * B * T * C;
  dk = dqkvr + 1 * B * T * C;
  dv = dqkvr + 2 * B * T * C;

  // backward through the unpermute operation
  int num_blocks = CEIL_DIV(B * T * C, block_size);
  printf("before unpermute\n");
  unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(
      scratch, dout, B, T, NH, HS);
  printf("before unpermute\n");
  // backward into datt

  matmul_cublaslt(datt, v, scratch, nullptr, T, T, HS, stream, true, false,
                  B * NH, T * HS, T * HS, T * T);
  // backward into dv
  matmul_cublaslt(dv, scratch, att, nullptr, HS, T, T, stream, false, true,
                  B * NH, T * HS, T * T, T * HS);
  const float scale = 1.0f / sqrtf((float)HS);
  // backward into preatt. this is an in-place operation; datt turns into
  // dpreatt here
  softmax_autoregressive_backward_inplace_kernel<<<dim3(T / 4, B * NH), 256>>>(
      datt, att, B, T, C, scale);
  const floatX *dpreatt = datt;
  // backward into q
  matmul_cublaslt(dq, k, dpreatt, nullptr, HS, T, T, stream, false, false,
                  B * NH, T * HS, T * T, T * HS);
  // backward into k
  matmul_cublaslt(dk, q, dpreatt, nullptr, HS, T, T, stream, false, true,
                  B * NH, T * HS, T * T, T * HS);
  // backward into inp
  num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
  permute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(
      dinp, dq, dk, dv, B, T, NH, HS);
  cudaCheck(cudaGetLastError());
}
}
