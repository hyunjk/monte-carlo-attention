from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='monte_carlo_attention',
    ext_modules=[
        CUDAExtension(name='monte_carlo_attention', sources=['torch_ext.cpp', 'attention.cu'],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
