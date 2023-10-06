from setuptools import setup
from torch.utils import cpp_extension
import os

"""
    引入扩张率
"""

src_root = '/file/msa_conv_cuda/core/op'
cpp_src = ['msa_conv.cpp', 'msa_conv_cuda.cu'] # cpp文件目录

if __name__ == '__main__':
    include_dirs = ['/file/msa_conv_cuda/'] # include 文件的目录
    cpp_path = [os.path.join(src_root, src) for src in cpp_src]

    setup(
        name='msa_conv1',
        ext_modules=[
            # cpp_extension.CppExtension(
            #     'msa_conv', cpp_path, include_dirs=include_dirs) # 编译CPU版本使用的
            cpp_extension.CUDAExtension(
                'msa_conv1', cpp_path, include_dirs=include_dirs) # 编译带有GPU版本使用的
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
        # cmdclass={'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)})