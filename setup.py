from setuptools import setup, Extension
import numpy 

setup_args = dict(
    ext_modules=[Extension('graphlearning.cextensions', 
                            sources=['c_code/cextensions.c',
                                     'c_code/lp_iterate.c',
                                     'c_code/hjsolvers.c',
                                     'c_code/memory_allocation.c',
                                     'c_code/mnist_benchmark.c',
                                     'c_code/mbo_speedy_volume_preserving.c'],
                            include_dirs=[numpy.get_include()],
                            extra_compile_args = ['-Ofast','-std=gnu99'],
                            extra_link_args = ['-lm'])])

setup(**setup_args)
