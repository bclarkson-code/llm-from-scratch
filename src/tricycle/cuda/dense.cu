/*
Matrix Multiplication, with help from cuBLASLt
*/
#include <Python.h>
#include <assert.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <type_traits> // std::bool_constant

// llmc internal imports
#include "cublas_common.h"
#include "cuda_common.h"

#define PY_SSIZE_T_CLEAN

// Generic wrapper around cublasLtMatmul
void matmul_cublaslt(floatX *d, const floatX *a, const floatX *b,
                     const floatX *bias, int m, int n, int k,
                     cudaStream_t stream = 0, bool transA = true,
                     bool transB = false, int batch_count = 0,
                     size_t strideA = 0, size_t strideB = 0,
                     size_t strideOut = 0, bool accumulate = false) {
  LOG("Entering matmul_cublaslt");
  LOG("Dimensions: m=" << m << ", n=" << n << ", k=" << k);
  LOG("Pointers: d=" << d << ", a=" << a << ", b=" << b << ", bias=" << bias);
  LOG("Other params: transA=" << transA << ", transB=" << transB
                              << ", batch_count=" << batch_count
                              << ", accumulate=" << accumulate);

  NVTX_RANGE_FN();
  bool has_bias = (bias != NULL);
  LOG("has_bias: " << has_bias);

  // check alignment
  if (((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 ||
      ((uintptr_t)d % 16) != 0 || (has_bias && ((uintptr_t)bias % 16) != 0)) {
    throw std::runtime_error("All cuBLASLt pointers must be aligned!");
  }
  LOG("Alignment check passed");

  // create the operation descriptor
  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t ALayout = nullptr, BLayout = nullptr,
                         CLayout = nullptr, DLayout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  try {
    LOG("Creating matmul descriptor");
    CUBLAS_CHECK(
        cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));
    LOG("Matmul descriptor created");

    // Set matrix operation attributes
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    LOG("Setting matrix operation attributes");
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
        (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
        (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));
    LOG("Matrix operation attributes set");

    // define matrix layouts
    LOG("Creating matrix layouts");
    if (transA) {
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k));
    } else {
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m));
    }
    if (transB) {
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
    } else {
      CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
    }

    cudaDataType_t computeType =
        (sizeof(floatX) == sizeof(float)) ? CUDA_R_32F : CUDA_R_16F;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&CLayout, computeType, m, n, m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&DLayout, computeType, m, n, m));
    LOG("Matrix layouts created");

    // create a preference handle
    LOG("Creating preference handle");
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));
    LOG("Preference handle created");

    // find a suitable algorithm
    LOG("Finding suitable algorithm");
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
        preference, 1, &heuristic, &returnedResults));

    if (returnedResults == 0) {
      throw std::runtime_error("No suitable cuBLASLt algorithm found");
    }
    LOG("Suitable algorithm found");

    // set whether to accumulate or not
    LOG("Setting alpha and beta");
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    LOG("Alpha and beta set: alpha=" << alpha << ", beta=" << beta);

    // call the matmul
    LOG("Calling cublasLtMatmul");
    CUBLAS_CHECK(cublasLtMatmul(cublaslt_handle, operationDesc, &alpha, a,
                                ALayout, b, BLayout, &beta, d, CLayout, d,
                                DLayout, &heuristic.algo, cublaslt_workspace,
                                cublaslt_workspace_size, stream));

    LOG("cublasLtMatmul completed successfully");
  } catch (const std::exception &e) {
    std::cerr << "Error in matmul_cublaslt: " << e.what() << std::endl;
    // Clean up resources
    if (preference)
      cublasLtMatmulPreferenceDestroy(preference);
    if (operationDesc)
      cublasLtMatmulDescDestroy(operationDesc);
    if (ALayout)
      cublasLtMatrixLayoutDestroy(ALayout);
    if (BLayout)
      cublasLtMatrixLayoutDestroy(BLayout);
    if (CLayout)
      cublasLtMatrixLayoutDestroy(CLayout);
    if (DLayout)
      cublasLtMatrixLayoutDestroy(DLayout);
    throw; // re-throw the exception
  }

  // Clean up resources
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatmulDescDestroy(operationDesc);
  cublasLtMatrixLayoutDestroy(ALayout);
  cublasLtMatrixLayoutDestroy(BLayout);
  cublasLtMatrixLayoutDestroy(CLayout);
  cublasLtMatrixLayoutDestroy(DLayout);

  CUDA_CHECK(cudaGetLastError());
  LOG("Exiting matmul_cublaslt");
}

// ---------------------------------------------------------------------------
// Kernels for dense layer

extern "C" void matmul_forward(floatX *out, floatX *inp, floatX *weight, int B,
                               int T, int C, int OC, cudaStream_t stream = 0) {
  LOG("Entering matmul_forward_cublaslt");
  LOG("Dimensions: B=" << B << ", T=" << T << ", C=" << C << ", OC=" << OC);
  LOG("Pointers: out=" << out << ", inp=" << inp << ", weight=" << weight);

  try {
    if (cublaslt_handle == nullptr) {
      throw std::runtime_error("cublaslt_handle is not initialized");
    }
    if (cublaslt_workspace == nullptr) {
      throw std::runtime_error("cublaslt_workspace is not initialized");
    }
    matmul_cublaslt(out,     // d (output matrix)
                    weight,  // b (second input matrix)
                    inp,     // a (first input matrix)
                    nullptr, // bias
                    OC,      // m (number of rows in output/a)
                    B * T,   // n (number of columns in output/b)
                    C,       // k (number of columns in a / rows in b)
                    stream,  // stream
                    true,    // transA
                    false,   // transB
                    0,       // batch_count
                    0,       // strideA
                    0,       // strideB
                    0,       // strideOut
                    false    // accumulate
    );
  } catch (const std::exception &e) {
    std::cerr << "Exception in matmul_forward_cublaslt: " << e.what()
              << std::endl;
    throw;
  }

  LOG("Exiting matmul_forward_cublaslt");
}

extern "C" void matmul_weight_backward(floatX *out, floatX *inp, floatX *grad,
                                       int B, int T, int C, int OC,
                                       cudaStream_t stream = 0) {
  LOG("Entering matmul_forward_cublaslt");
  LOG("Dimensions: B=" << B << ", T=" << T << ", C=" << C << ", OC=" << OC);
  LOG("Pointers: out=" << out << ", inp=" << inp << ", grad=" << grad);

  try {
    if (cublaslt_handle == nullptr) {
      throw std::runtime_error("cublaslt_handle is not initialized");
    }
    if (cublaslt_workspace == nullptr) {
      throw std::runtime_error("cublaslt_workspace is not initialized");
    }
    matmul_cublaslt(
        out,     // d (output matrix, weight gradient)
        grad,    // b (second input matrix, output gradient)
        inp,     // a (first input matrix, input)
        nullptr, // bias (typically not used in weight gradient computation)
        C,  // m (number of rows in weight_grad/inp.T, which is input channels)
        OC, // n (number of columns in weight_grad/gradient, which is output
            // channels)
        B * T,  // k (number of columns in inp / rows in gradient, which is
                // batch_size * sequence_length)
        stream, // stream
        false,  // transA (transpose input to get inp.T)
        true,   // transB
        0,      // batch_count
        0,      // strideA
        0,      // strideB
        0,      // strideOut
        true    // we want to add the weights grad to any existing grads, not to
                // overwrite
    );
  } catch (const std::exception &e) {
    std::cerr << "Exception in matmul_forward_cublaslt: " << e.what()
              << std::endl;
    throw;
  }

  LOG("Exiting matmul_forward_cublaslt");
}

extern "C" void matmul_input_backward(floatX *out, floatX *weight, floatX *grad,
                                      int B, int T, int C, int OC,
                                      cudaStream_t stream = 0) {
  LOG("Entering matmul_forward_cublaslt");
  LOG("Dimensions: B=" << B << ", T=" << T << ", C=" << C << ", OC=" << OC);
  LOG("Pointers: out=" << out << ", grad=" << grad << ", weight=" << weight);

  try {
    if (cublaslt_handle == nullptr) {
      throw std::runtime_error("cublaslt_handle is not initialized");
    }
    if (cublaslt_workspace == nullptr) {
      throw std::runtime_error("cublaslt_workspace is not initialized");
    }
    matmul_cublaslt(out,     // d (output matrix, input gradient)
                    weight,  // a (first input matrix, output gradient)
                    grad,    // b (second input matrix, weight matrix)
                    nullptr, // bias (not used in input gradient computation)
                    C,       // m (number of rows in input_grad/grad, which is
                             // batch_size * n_tokens)
                    B * T, // n (number of columns in input_grad/weight.T, which
                           // is input channels)
                    OC, // k (number of columns in grad / rows in weight, which
                        // is output channels)
                    stream, // stream
                    false,  // transA (don't transpose grad)
                    false,  // transB (DO transpose weight to get weight.T)
                    0,      // batch_count
                    0,      // strideA
                    0,      // strideB
                    0,      // strideOut
                    false   // accumulate
    );
  } catch (const std::exception &e) {
    std::cerr << "Exception in matmul_forward_cublaslt: " << e.what()
              << std::endl;
    throw;
  }

  LOG("Exiting matmul_forward_cublaslt");
}

// ---------------------------------------------------------------------------
// Handle interface with python

// Wrapper for forward matmul
static PyObject *py_matmul_forward(PyObject *self, PyObject *args) {
  PyObject *output, *input, *weights;
  long B, T, C, out_shape;
  PyObject *stream_obj;

  // check that the passed arguments are the right type
  if (!PyArg_ParseTuple(args, "OOOllllO", &output, &input, &weights, &B, &T, &C,
                        &out_shape, &stream_obj)) {
    PyErr_Print();
    return NULL;
  }

  LOG("Parsed arguments in py_matmul_forward");
  LOG("B: " << B << ", T: " << T << ", C: " << C
            << ", out_shape: " << out_shape);

  floatX *output_ptr, *input_ptr, *weights_ptr;

  // Access the cupy arrays
  output_ptr = static_cast<floatX *>(get_array_pointer(output, "output"));
  if (!output_ptr)
    return NULL;

  input_ptr = static_cast<floatX *>(get_array_pointer(input, "input"));
  if (!input_ptr)
    return NULL;

  weights_ptr = static_cast<floatX *>(get_array_pointer(weights, "weights"));
  if (!weights_ptr)
    return NULL;

  // Handle CUDA stream
  cudaStream_t cuda_stream = nullptr;
  if (PyObject_HasAttrString(stream_obj, "ptr")) {
    PyObject *ptr_attr = PyObject_GetAttrString(stream_obj, "ptr");
    if (ptr_attr) {
      cuda_stream = reinterpret_cast<cudaStream_t>(PyLong_AsVoidPtr(ptr_attr));
      Py_DECREF(ptr_attr);
    } else {
      PyErr_Clear();
    }
  }

  if (cuda_stream == nullptr) {
    LOG("Using default stream (nullptr)");
  } else {
    LOG("Using provided stream: " << cuda_stream);
  }
  LOG("Extracted CUDA stream");

  try {
    matmul_forward(output_ptr, input_ptr, weights_ptr, B, T, C, out_shape,
                   cuda_stream);
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }

  LOG("Completed py_matmul_forward");

  Py_RETURN_NONE;
}

// Wrapper for  weight gradient calculation
static PyObject *py_matmul_weight_backward(PyObject *self, PyObject *args) {
  PyObject *output, *input, *grad;
  long B, T, C, out_shape;
  PyObject *stream_obj;

  // check that the passed arguments are the right type
  if (!PyArg_ParseTuple(args, "OOOllllO", &output, &input, &grad, &B, &T, &C,
                        &out_shape, &stream_obj)) {
    PyErr_Print();
    return NULL;
  }

  LOG("Parsed arguments in py_matmul_forward");
  LOG("B: " << B << ", T: " << T << ", C: " << C
            << ", out_shape: " << out_shape);

  floatX *output_ptr, *input_ptr, *grad_ptr;

  // Access the cupy arrays
  output_ptr = static_cast<floatX *>(get_array_pointer(output, "output"));
  if (!output_ptr)
    return NULL;

  input_ptr = static_cast<floatX *>(get_array_pointer(input, "input"));
  if (!input_ptr)
    return NULL;

  grad_ptr = static_cast<floatX *>(get_array_pointer(grad, "grad"));
  if (!grad_ptr)
    return NULL;

  // Handle CUDA stream
  cudaStream_t cuda_stream = nullptr;
  if (PyObject_HasAttrString(stream_obj, "ptr")) {
    PyObject *ptr_attr = PyObject_GetAttrString(stream_obj, "ptr");
    if (ptr_attr) {
      cuda_stream = reinterpret_cast<cudaStream_t>(PyLong_AsVoidPtr(ptr_attr));
      Py_DECREF(ptr_attr);
    } else {
      PyErr_Clear();
    }
  }

  if (cuda_stream == nullptr) {
    LOG("Using default stream (nullptr)");
  } else {
    LOG("Using provided stream: " << cuda_stream);
  }
  LOG("Extracted CUDA stream");

  try {
    matmul_weight_backward(output_ptr, input_ptr, grad_ptr, B, T, C, out_shape,
                           cuda_stream);
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }

  LOG("Completed py_matmul_forward");

  Py_RETURN_NONE;
}

// Wrapper for input gradient calculation
static PyObject *py_matmul_input_backward(PyObject *self, PyObject *args) {
  PyObject *output, *weight, *grad;
  long B, T, C, out_shape;
  PyObject *stream_obj;

  // check that the passed arguments are the right type
  if (!PyArg_ParseTuple(args, "OOOllllO", &output, &weight, &grad, &B, &T, &C,
                        &out_shape, &stream_obj)) {
    PyErr_Print();
    return NULL;
  }

  LOG("Parsed arguments in py_matmul_forward");
  LOG("B: " << B << ", T: " << T << ", C: " << C
            << ", out_shape: " << out_shape);

  floatX *output_ptr, *weight_ptr, *grad_ptr;

  // Access the cupy arrays
  output_ptr = static_cast<floatX *>(get_array_pointer(output, "output"));
  if (!output_ptr)
    return NULL;

  weight_ptr = static_cast<floatX *>(get_array_pointer(weight, "weight"));
  if (!weight_ptr)
    return NULL;

  grad_ptr = static_cast<floatX *>(get_array_pointer(grad, "grad"));
  if (!grad_ptr)
    return NULL;

  // Handle CUDA stream
  cudaStream_t cuda_stream = nullptr;
  if (PyObject_HasAttrString(stream_obj, "ptr")) {
    PyObject *ptr_attr = PyObject_GetAttrString(stream_obj, "ptr");
    if (ptr_attr) {
      cuda_stream = reinterpret_cast<cudaStream_t>(PyLong_AsVoidPtr(ptr_attr));
      Py_DECREF(ptr_attr);
    } else {
      PyErr_Clear();
    }
  }

  if (cuda_stream == nullptr) {
    LOG("Using default stream (nullptr)");
  } else {
    LOG("Using provided stream: " << cuda_stream);
  }
  LOG("Extracted CUDA stream");

  try {
    matmul_input_backward(output_ptr, weight_ptr, grad_ptr, B, T, C, out_shape,
                          cuda_stream);
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }

  LOG("Completed py_matmul_forward");

  Py_RETURN_NONE;
}
