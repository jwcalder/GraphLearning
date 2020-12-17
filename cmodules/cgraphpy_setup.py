from distutils.core import setup, Extension
import numpy
import os


def main():

    #Change into cmodules directory
    if not os.path.isfile('cgraphpy.c'):
        os.chdir('cmodules')

    setup(name="cgraphpy",
          version="1.0.0",
          description="C extension accelerations for graphlearning Python package",
          author="Jeff Calder",
          author_email="jwcalder@umn.edu",
          ext_modules=[Extension('cgraphpy', sources=['cgraphpy.c','lp_iterate.c','dijkstra.c','memory_allocation.c','mnist_benchmark.c','mbo_speedy_volume_preserving.c'],include_dirs=[numpy.get_include()],extra_compile_args = ['-Ofast'],extra_link_args = ['-lm'])])

if __name__ == "__main__":
    main()
