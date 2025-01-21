#include <cublasLt.h>
#include <cublas_common.h>
#include <cuda_common.h>

void matmul_cublaslt(floatX* d, const floatX* a, const floatX* b,
                     const floatX* bias, int m, int n, int k,
                     cudaStream_t stream = 0, bool transA = true,
                     bool transB = false, int batch_count = 0,
                     size_t strideA = 0, size_t strideB = 0,
                     size_t strideOut = 0, bool accumulate = false,
                     floatX* pre_gelu = NULL, bool backward = false) {
  bool has_bias = (bias != NULL);
  bool has_gelu = (pre_gelu != NULL);

  // check alignment (some modes work unaligned but it always best to be aligned
  // for performance)
  if (((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 ||
      ((uintptr_t)d % 16) != 0 || (has_bias && ((uintptr_t)bias % 16) != 0)) {
    printf("a aligned: %d\n", ((uintptr_t)a % 16) == 0);
    printf("b aligned: %d\n", ((uintptr_t)b % 16) == 0);
    printf("d aligned: %d\n", ((uintptr_t)d % 16) == 0);
    if (has_bias) {
      printf("bias aligned: %d\n", ((uintptr_t)bias % 16) == 0);
    }
    printf("All cuBLASLt pointers must be aligned!\n");
    exit(EXIT_FAILURE);
  }

  // create the operation descriptor
  cublasLtMatmulDesc_t operationDesc;
  cublasCheck(
      cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

  int returnedResults = 0;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulHeuristicResult_t heuristic;

  cublasOperation_t opNoTranspose = CUBLAS_OP_N;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasCheck(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
      (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
  cublasCheck(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
      (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

  // define matrix layouts
  cublasLtMatrixLayout_t ALayout;
  cublasLtMatrixLayout_t BLayout;
  cublasLtMatrixLayout_t DLayout;
  cublasLtMatrixLayout_t CLayout;
  if (transA) {
    cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k));
  } else {
    cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m));
  }
  if (transB) {
    cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
  } else {
    cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
  }
  // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
  cublasCheck(cublasLtMatrixLayoutCreate(
      &CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP, m, n, m));
  cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

  // Strided Batched GEMM (used for non-flash attention, equivalent to
  // cublasGemmStridedBatchedEx)
  if (batch_count) {
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));

    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA,
        sizeof(strideA)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB,
        sizeof(strideB)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut,
        sizeof(strideOut)));
    cublasCheck(cublasLtMatrixLayoutSetAttribute(
        DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut,
        sizeof(strideOut)));
  }

  // create a preference handle with specified max workspace
  cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
  cublasCheck(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

  // setup epilogue and associated pointers for bias & gelu
  cublasLtEpilogue_t epilogue;
  if (has_gelu) {
    int64_t gelu_ld = m;  // todo - is this affected by anything else?
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld,
        sizeof(gelu_ld)));
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu,
        sizeof(pre_gelu)));
    if (backward) {
      // assert(!has_bias);  // we shouldn't have any backward matmuls that use
      //                     // both GELU and bias
      epilogue = CUBLASLT_EPILOGUE_DGELU;
    } else {
      epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS
                          : CUBLASLT_EPILOGUE_GELU_AUX;
    }
  } else if (has_bias) {
    epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
  } else {
    epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  }
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc,
                                             CUBLASLT_MATMUL_DESC_EPILOGUE,
                                             &epilogue, sizeof(epilogue)));

  if (has_bias) {
    // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
    cublasDataType_t bias_data_type =
        (sizeof(floatX) == 1) ? CUDA_R_16BF
                              : CUBLAS_LOWP;  // force BF16 bias for FP8 mode
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type,
        sizeof(bias_data_type)));
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  }

  // set scale type to FP32 (needs to be FP16 if and only if using
  // CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
  cublasDataType_t scale_type = CUDA_R_32F;
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc,
                                             CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                             &scale_type, sizeof(scale_type)));

  // find a suitable algorithm (cached internally so shouldn't take much CPU
  // time in practice)
  cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout,
                                 BLayout, CLayout, DLayout, preference, 1,
                                 &heuristic, &returnedResults);
  if (returnedResults == 0) {
    printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d, bias: %d\n", n, m, k,
           has_bias);
    exit(EXIT_FAILURE);
  }

  // set whether to accumulate (i.e. D += C) or not - note this isn't considered
  // in algorithm selection (?!)
  const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

  // call the matmul
  cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc, &alpha, a, ALayout,
                             b, BLayout, &beta, d, CLayout, d, DLayout,
                             &heuristic.algo, cublaslt_workspace,
                             cublaslt_workspace_size, stream));

  // cleanups
  cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
  cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
  cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
  cudaCheck(cudaGetLastError());
};
