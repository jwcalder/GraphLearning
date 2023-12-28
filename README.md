## Graph-based Clustering and Semi-Supervised Learning

![Clustering](https://github.com/jwcalder/GraphLearning/raw/master/images/clustering.png)

This python package is devoted to efficient implementations of modern graph-based learning algorithms for semi-supervised learning, active learning, and clustering. The package implements many popular datasets (currently MNIST, FashionMNIST, and CIFAR-10) in a way that makes it simple for users to test out new algorithms and rapidly compare against existing methods. Full [documentation](https://jwcalder.github.io/GraphLearning/) is available, including detailed example scripts.

This package also reproduces experiments from the paper

J. Calder, B. Cook, M. Thorpe, D. Slepcev. [Poisson Learning: Graph Based Semi-Supervised Learning at Very Low Label Rates.](http://proceedings.mlr.press/v119/calder20a.html), Proceedings of the 37th International Conference on Machine Learning, PMLR 119:1306-1316, 2020.

## Installation

Install with
```sh
pip install graphlearning
```
Required packages will be installed automatically, and include numpy, scipy, sklearn, and matplotlib. Some features in the package rely on other packages, including [annoy](https://github.com/spotify/annoy) for approximate nearest neighbor searches, and [torch](https://github.com/pytorch/pytorch) for GPU acceleration. You will have to install these manually, if needed, with
```sh
pip install annoy torch
```
It can be difficult to install annoy, depending on your operating system. 

To install the most recent version of GraphLearning from the github source, which is updated more frequently, run
```sh
git clone https://github.com/jwcalder/GraphLearning
cd GraphLearning
pip install .
```
If you prefer to use ssh swap the first line with
```sh
git clone git@github.com:jwcalder/GraphLearning.git
```

## Documentation and Examples

Full documentation for the package is available [here](https://jwcalder.github.io/GraphLearning/). The documentation includes examples of how to use the package. All example scripts linked from the documentation can be found in the examples folder. 

## Older versions of GraphLearning

This repository hosts the current version of the package, which is numbered >=1.0.0. This version is not backwards compatible with earlier versions of the package. The old version is archived [here](https://github.com/jwcalder/GraphLearningOld) and can be installed with
```sh
pip install graphlearning==0.0.3
```
To make sure you will load the old version when running `import graphlearning`, it may be necessary to uninstall all existing versions `pip uninstall graphlearning` before running the installation command above.

## Citations

If you use this package in your research, please cite the package with the bibtex entry below.
```
@software{graphlearning,
  author       = {Jeff Calder},
  title        = {GraphLearning Python Package},
  month        = jan,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5850940},
  url          = {https://doi.org/10.5281/zenodo.5850940}
}
```

## Contact and questions

Email <jwcalder@umn.edu> with any questions or comments.

## Acknowledgments

Several people have contributed to the development of this software:

1. Mauricio Rios Flores (Machine Learning Researcher, Amazon)
2. Brendan Cook (PhD Candidate in Mathematics, University of Minnesota)
3. Matt Jacobs (Postdoc, UCLA)
4. Mahmood Ettehad (Postdoc, IMA)
5. Jason Setiadi
6. Kevin Miller (Postdoc, Oden Institute)

