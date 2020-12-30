import setuptools
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graphlearning", 
    version="0.0.1",
    author="Jeff Calder",
    author_email="jwcalder@umn.edu",
    description="Python package for graph-based clustering and semi-supervised learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwcalder/GraphLearning",
    packages=['graphlearning'],
    ext_modules=[setuptools.Extension('graphlearning.cextensions', 
                    sources=[   'src/cextensions.c',
                                'src/lp_iterate.c',
                                'src/dijkstra.c',
                                'src/memory_allocation.c',
                                'src/mnist_benchmark.c',
                                'src/mbo_speedy_volume_preserving.c'],
                    install_requires=[  'numpy', 
                                        'scipy', 
                                        'torch', 
                                        'annoy', 
                                        'sklearn', 
                                        'kymatio',
                                        'matplotlib'],
                    include_dirs=[numpy.get_include()],
                    extra_compile_args = ['-Ofast'],
                    extra_link_args = ['-lm'])],
                    classifiers=[
                                "Programming Language :: Python :: 3",
                                "License :: OSI Approved :: MIT License",
                                "Operating System :: OS Independent"],
                    python_requires='>=3.6',
)

