#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {
// Main Flash Attention launch function
EXPORT cudnnStatus_t launch_flash_attention(
    const void* Q,          // Query tensor (b, h_q, s_q, d_qk)
    const void* K,          // Key tensor (b, h_k, s_kv, d_qk)
    const void* V,          // Value tensor (b, h_v, s_kv, d_v)
    void* O,               // Output tensor (b, h_q, s_q, d_v)
    void* Stats,           // Stats tensor for training (can be nullptr for inference)
    const void* Bias,      // Optional attention bias (can be nullptr)
    const int32_t* seq_len_q,  // Optional sequence lengths for Q (can be nullptr)
    const int32_t* seq_len_kv, // Optional sequence lengths for K/V (can be nullptr)
    int64_t b,             // Batch size
    int64_t h_q,          // Number of heads for Q
    int64_t h_k,          // Number of heads for K
    int64_t h_v,          // Number of heads for V
    int64_t s_q,          // Sequence length for Q
    int64_t s_kv,         // Sequence length for K/V
    int64_t d_qk,         // Hidden dimension for Q/K
    int64_t d_v,          // Hidden dimension for V
    float attn_scale,     // Attention scale factor
    bool is_inference,    // Whether running in inference mode
    bool causal_mask,     // Whether to apply causal mask
    bool padding_mask,    // Whether to use padding mask
    cudaStream_t stream); // CUDA stream
}
