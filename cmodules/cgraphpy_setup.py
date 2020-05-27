from distutils.core import setup, Extension
import numpy
import os

#Change into cmodules directory
if not os.path.isfile('cgraphpy.c'):
    os.chdir('cmodules')

# define the extension module
cgraphpy = Extension('cgraphpy', sources=['cgraphpy.c','lp_iterate.c','dijkstra.c','memory_allocation.c','mnist_benchmark.c','mbo_speedy_volume_preserving.c'],include_dirs=[numpy.get_include()],extra_compile_args = ['-Ofast'],extra_link_args = ['-lm'])

# run the setup
setup(ext_modules=[cgraphpy])
