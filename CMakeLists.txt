cmake_minimum_required(VERSION 3.18)
project(cuda_vector_add CUDA CXX)

# Set paths based on pixi environment
set(PIXI_ENV_ROOT $ENV{PIXI_PROJECT_ROOT}/.pixi/envs/default)
set(PYTHON_SITE_PACKAGES ${PIXI_ENV_ROOT}/lib/python3.12/site-packages)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_path(CUDNN_FRONTEND_INCLUDE_DIR cudnn_frontend.h
    HINTS
    ${PYTHON_SITE_PACKAGES}/include
    ${PIXI_ENV_ROOT}/include
    PATH_SUFFIXES cuda cuda/include)

if(NOT CUDNN_FRONTEND_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find cudnn_frontend.h")
endif()

# Common settings for all CUDA targets
function(setup_cuda_target target)
    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDNN_FRONTEND_INCLUDE_DIR}
    )
    target_link_libraries(${target} PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        cudnn  # Ensure cuDNN is linked
    )
    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES "86-real;86-virtual"
        PREFIX ""
    )
endfunction()

# Create and setup the CUDA libraries
# foreach(target IN ITEMS vector_add attention)
#     add_library(${target} SHARED cuda/${target}.cu)
#     setup_cuda_target(${target})
# endforeach()
add_library(attention_cudnn SHARED cuda/attention_cudnn.cpp)
setup_cuda_target(attention_cudnn)

# Debug output
message(STATUS "CUDA Toolkit: ${CUDAToolkit_ROOT}")
message(STATUS "CUDNN Frontend: ${CUDNN_FRONTEND_INCLUDE_DIR}")
