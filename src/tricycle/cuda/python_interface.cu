#include "cublas_common.h"
#include "utils.cuh"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#define PY_SSIZE_T_CLEAN
#include "attention.cpp"
#include "dense.cu"
#include <Python.h>
#include <cublasLt.h>
#include <string>

// init and clean up cublas
static int init_cublas() {
  if (cublasLtCreate(&cublaslt_handle) != CUBLAS_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create cuBLASLt handle");
    return -1;
  }
  if (cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size) != cudaSuccess) {
    cublasLtDestroy(cublaslt_handle);
    cublaslt_handle = nullptr;
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to allocate cuBLASLt workspace");
    return -1;
  }
  return 0;
}

static void cleanup_cublas() {
  if (cublaslt_workspace) {
    cudaFree(cublaslt_workspace);
    cublaslt_workspace = nullptr;
  }
  if (cublaslt_handle) {
    cublasLtDestroy(cublaslt_handle);
    cublaslt_handle = nullptr;
  }
}

void create_cudnn() { cuDNNCheck(cudnnCreate(&cudnn_handle)); }

void destroy_cudnn() {
  if (cudnn_workspace != NULL) {
    cudaCheck(cudaFree(cudnn_workspace));
  }
  cuDNNCheck(cudnnDestroy(cudnn_handle));
}

// Module definition

// Module methods
static PyMethodDef LlmcMethods[] = {
    {"matmul_forward", py_matmul_forward, METH_VARARGS,
     "Forward op for a dense layer"},
    {"matmul_weight_backward", py_matmul_weight_backward, METH_VARARGS,
     "Gradient for weights after a matmul"},
    {"matmul_input_backward", py_matmul_input_backward, METH_VARARGS,
     "Gradient for input after a matmul"},
    {"attention_forward", py_attention_forward, METH_VARARGS,
     "Forward operation for attention"},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef llmcmodule = {PyModuleDef_HEAD_INIT, "llmc", NULL, -1,
                                        LlmcMethods};

static void llmc_free(void *unused) {
  cleanup_cublas();
  destroy_cudnn();
}

PyMODINIT_FUNC PyInit_llmc(void) {
  PyObject *m;

  m = PyModule_Create(&llmcmodule);
  if (m == NULL)
    return NULL;

  if (init_cublas() < 0) {
    Py_DECREF(m);
    return NULL;
  }
  create_cudnn();

  if (PyModule_AddFunctions(m, LlmcMethods) < 0) {
    Py_DECREF(m);
    return NULL;
  }

  if (PyModule_AddObject(
          m, "__cleanup__",
          PyCapsule_New((void *)llmc_free, "__cleanup__", NULL)) < 0) {
    Py_DECREF(m);
    return NULL;
  }

  return m;
}
