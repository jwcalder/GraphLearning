import setuptools
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graphlearning", 
    version="1.1.5",
    author="Jeff Calder",
    author_email="jwcalder@umn.edu",
    description="Python package for graph-based clustering and semi-supervised learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwcalder/GraphLearning",
    packages=['graphlearning'],
    ext_modules=[setuptools.Extension('graphlearning.cextensions', 
                    sources=[   'c_code/cextensions.c',
                                'c_code/lp_iterate.c',
                                'c_code/hjsolvers.c',
                                'c_code/memory_allocation.c',
                                'c_code/mnist_benchmark.c',
                                'c_code/mbo_speedy_volume_preserving.c'],
                    include_dirs=[numpy.get_include()],
                    extra_compile_args = ['-Ofast','-std=gnu99'],
                    extra_link_args = ['-lm'])],
    classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent"],
    install_requires=[  'numpy', 
                        'scipy', 
                        'sklearn', 
                        'matplotlib'],
    python_requires='>=3.6',
)


