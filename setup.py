from setuptools import setup, Extension
import numpy 

setup_args = dict(
    ext_modules=[Extension('graphlearning.cextensions', 
                            sources=['c_code/cextensions.cpp',
                                     'c_code/lp_iterate.cpp',
                                     'c_code/hjsolvers.cpp',
                                     'c_code/memory_allocation.cpp',
                                     'c_code/mnist_benchmark.cpp',
                                     'c_code/mbo_convolution.cpp',
                                     'c_code/tsne.cpp',
                                     'c_code/sptree.cpp'],
                            include_dirs=[numpy.get_include()],
                            #extra_compile_args = ['-Ofast','-std=c11'],
                            extra_compile_args = ['-Ofast'],
                            extra_link_args = ['-lm'])])

setup(**setup_args)
