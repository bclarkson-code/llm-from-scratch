#include "flash_attention.h"
#include <cudnn_frontend.h>
#include <memory>
#include <unordered_map>

// Helper macro for CUDA error checking
#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      return CUDNN_STATUS_EXECUTION_FAILED;                                    \
    }                                                                          \
  } while (0)

// Helper macro for CUDNN error checking
#define CUDNN_CHECK(stmt)                                                      \
  do {                                                                         \
    cudnnStatus_t err = stmt;                                                  \
    if (err != CUDNN_STATUS_SUCCESS) {                                         \
      return err;                                                              \
    }                                                                          \
  } while (0)

extern "C" {
cudnnStatus_t launch_flash_attention(
    const void *Q, const void *K, const void *V, void *O, void *Stats,
    const void *Bias, const int32_t *seq_len_q, const int32_t *seq_len_kv,
    int64_t b, int64_t h_q, int64_t h_k, int64_t h_v, int64_t s_q, int64_t s_kv,
    int64_t d_qk, int64_t d_v, float attn_scale, bool is_inference,
    bool causal_mask, bool padding_mask, cudaStream_t stream) {

  // Create CUDNN handle and set stream
  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));
  CUDNN_CHECK(cudnnSetStream(handle, stream));

  // Create the graph
  auto graph = cudnn_frontend::graph::Graph();
  graph.set_io_data_type(cudnn_frontend::DataType_t::FLOAT16);
  graph.set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT);
  graph.set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  // Define tensors
  auto Q_tensor =
      graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                       .set_name("Q")
                       .set_uid(1)
                       .set_dim({b, h_q, s_q, d_qk})
                       .set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1}));

  auto K_tensor =
      graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                       .set_name("K")
                       .set_uid(2)
                       .set_dim({b, h_k, s_kv, d_qk})
                       .set_stride({h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1}));

  auto V_tensor =
      graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                       .set_name("V")
                       .set_uid(3)
                       .set_dim({b, h_v, s_kv, d_v})
                       .set_stride({h_v * s_kv * d_v, s_kv * d_v, d_v, 1}));

  // Configure SDPA options
  auto sdpa_options = cudnn_frontend::graph::SDPA_attributes()
                          .set_name("flash_attention")
                          .set_is_inference(is_inference)
                          .set_causal_mask(causal_mask)
                          .set_attn_scale(attn_scale);

  cudnn_frontend::graph::Tensor_attributes *bias_tensor = nullptr;
  // Add optional tensors based on flags
  if (Bias != nullptr) {
    bias_tensor = new cudnn_frontend::graph::Tensor_attributes();
    *bias_tensor = cudnn_frontend::graph::Tensor_attributes()
                       .set_name("bias")
                       .set_uid(6)
                       .set_dim({b, 1, s_q, s_kv})
                       .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1});
    sdpa_options.set_bias(*bias_tensor);
  }

  cudnn_frontend::graph::Tensor_attributes *seq_q_tensor = nullptr;
  cudnn_frontend::graph::Tensor_attributes *seq_kv_tensor = nullptr;
  if (padding_mask) {
    seq_q_tensor = new cudnn_frontend::graph::Tensor_attributes();
    *seq_q_tensor = cudnn_frontend::graph::Tensor_attributes()
                        .set_name("seq_q")
                        .set_uid(7)
                        .set_dim({b, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(cudnn_frontend::DataType_t::INT32);

    seq_kv_tensor = new cudnn_frontend::graph::Tensor_attributes();
    *seq_kv_tensor = cudnn_frontend::graph::Tensor_attributes()
                         .set_name("seq_kv")
                         .set_uid(8)
                         .set_dim({b, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_data_type(cudnn_frontend::DataType_t::INT32);

    sdpa_options.set_padding_mask(padding_mask)
        .set_seq_len_q(*seq_q_tensor)
        .set_seq_len_kv(*seq_kv_tensor);
  }

  // Create SDPA operation
  auto sdpa_result = graph.sdpa(Q_tensor, K_tensor, V_tensor, sdpa_options);
  auto output_tensor = std::get<0>(sdpa_result);
  auto stats_tensor = std::get<1>(sdpa_result);

  // Configure output tensors
  cudnn_frontend::graph::Tensor_attributes output_attrs;
  output_attrs.set_output(true)
      .set_dim({b, h_q, s_q, d_v})
      .set_stride({h_q * d_v, d_v, b * h_q * d_v, 1})
      .set_uid(4);

  graph.tensor(output_attrs);

  if (!is_inference && stats_tensor) {
    cudnn_frontend::graph::Tensor_attributes stats_attrs;
    stats_attrs.set_output(true)
        .set_data_type(cudnn_frontend::DataType_t::FLOAT)
        .set_uid(5);
    graph.tensor(stats_attrs);
  }

  // Build the graph
  auto build_status = graph.build(handle, {cudnn_frontend::HeurMode_t::A});
  if (!build_status.is_good()) {
    // Cleanup
    if (bias_tensor)
      delete bias_tensor;
    if (seq_q_tensor)
      delete seq_q_tensor;
    if (seq_kv_tensor)
      delete seq_kv_tensor;
    cudnnDestroy(handle);
    return CUDNN_STATUS_EXECUTION_FAILED;
  }

  // Set up variant pack for execution
  std::unordered_map<int64_t, void *> variant_pack = {
      {1, const_cast<void *>(Q)},
      {2, const_cast<void *>(K)},
      {3, const_cast<void *>(V)},
      {4, O}};

  if (!is_inference) {
    variant_pack[5] = Stats;
  }
  if (Bias != nullptr) {
    variant_pack[6] = const_cast<void *>(Bias);
  }
  if (padding_mask) {
    variant_pack[7] = const_cast<void *>(static_cast<const void *>(seq_len_q));
    variant_pack[8] = const_cast<void *>(static_cast<const void *>(seq_len_kv));
  }

  // Get and allocate workspace
  int64_t workspace_size;
  auto ws_status = graph.get_workspace_size(workspace_size);
  if (!ws_status.is_good()) {
    // Cleanup
    if (bias_tensor)
      delete bias_tensor;
    if (seq_q_tensor)
      delete seq_q_tensor;
    if (seq_kv_tensor)
      delete seq_kv_tensor;
    cudnnDestroy(handle);
    return CUDNN_STATUS_BAD_PARAM;
  }

  void *workspace;
  CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

  // Execute the graph
  auto exec_status = graph.execute(handle, variant_pack, workspace);

  // Cleanup
  if (bias_tensor)
    delete bias_tensor;
  if (seq_q_tensor)
    delete seq_q_tensor;
  if (seq_kv_tensor)
    delete seq_kv_tensor;
  CUDA_CHECK(cudaFree(workspace));
  cudnnDestroy(handle);

  return exec_status.is_good() ? CUDNN_STATUS_SUCCESS
                               : CUDNN_STATUS_EXECUTION_FAILED;
}
}
