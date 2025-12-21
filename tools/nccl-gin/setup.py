from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

nccl_home = os.environ["NCCL_HOME"]

setup(
    name="gin_ext",
    ext_modules=[
        CUDAExtension(
            name="gin_ext",
            sources=[
                "gin_lsa.cpp",
                "cuda/gin_lsa.cu",
                "cuda/gin_all_gather.cu",
                "cuda/gin_reduce_scatter.cu",
            ],
            include_dirs=[os.path.join(nccl_home, "include")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
