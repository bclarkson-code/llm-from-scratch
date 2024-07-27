import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CUDAExtension(Extension):
    def __init__(self, *args, **kwargs):
        super(CUDAExtension, self).__init__(*args, **kwargs)


class custom_build_ext(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")

        def customize_compiler_for_nvcc(compiler):
            # Save original executables
            default_compiler_so = compiler.compiler_so
            default_compiler_cxx = compiler.compiler_cxx
            default_linker_so = compiler.linker_so

            # Function to filter flags
            def filter_flags(flags):
                return [
                    flag
                    for flag in flags
                    if not flag.startswith(("-W", "-f", "-m", "-O", "-pipe"))
                ]

            # Set NVCC as the compiler and linker
            compiler.set_executable("compiler_so", "nvcc")
            compiler.set_executable("compiler_cxx", "nvcc")
            compiler.set_executable("linker_so", "nvcc")

            # Filter flags for CUDA compilation
            compiler.compiler_so = ["nvcc"] + filter_flags(
                default_compiler_so[1:]
            )
            compiler.compiler_cxx = ["nvcc"] + filter_flags(
                default_compiler_cxx[1:]
            )
            compiler.linker_so = ["nvcc"] + filter_flags(default_linker_so[1:])

            # Add CUDA-specific flags
            cuda_flags = [
                "-O3",
                "-t=0",
                "--use_fast_math",
                "--restrict",
                "-std=c++17",
                "-UENABLE_FP32",
                "-DENABLE_FP16",
                "-UENABLE_BF16",
                "-UENABLE_LOGGING",
                "-DENABLE_CUDNN",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "-Xcompiler",
                "-fPIC",
            ]
            compiler.compiler_so += cuda_flags
            compiler.compiler_cxx += cuda_flags

            # Add rpath for CUDA libraries
            compiler.linker_so += [
                "-Xlinker",
                "-rpath",
                "-Xlinker",
                "/usr/local/cuda/lib64",
            ]

        customize_compiler_for_nvcc(self.compiler)

        build_ext.build_extensions(self)


CUDA_DIR = Path(__file__).parent / "src/tricycle/cuda"
CONDA_DIR = Path("/home/ben/mambaforge/envs/tricycle")
CUDNN_FRONTEND = Path(__file__).parent.parent / "cudnn-frontend/include"

ext = CUDAExtension(
    "llmc",
    sources=[str(CUDA_DIR / "python_interface.cu")],
    include_dirs=[
        str(CUDA_DIR.absolute()),
        "/usr/lib/gcc/x86_64-linux-gnu/11/include",
        "/usr/local/include",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include",
        str(
            (
                CONDA_DIR
                / "nsight-compute-2024.2.0/host/target-linux-x64/nvtx/include/"
            ).absolute()
        ),
        str(CONDA_DIR / "include"),
        str(CUDNN_FRONTEND.absolute()),
    ],
    library_dirs=["/usr/local/cuda/lib64"],
    libraries=["cublas", "cublasLt", "cudnn"],
    extra_compile_args=[],  # We'll add CUDA-specific flags in the custom_build_ext
    extra_link_args=[
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        "/usr/local/cuda/lib64",
    ],
)

setup(
    name="llmc",
    ext_modules=[ext],
    cmdclass={"build_ext": custom_build_ext},
)
