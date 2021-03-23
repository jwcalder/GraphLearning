## Graph-based Clustering and Semi-Supervised Learning

![Clustering](https://github.com/jwcalder/GraphLearning/raw/master/images/clustering.png)

This python package is devoted to efficient implementations of modern graph-based learning algorithms for both semi-supervised learning and clustering. The package implements many popular datasets (currently MNIST, FashionMNIST, cifar-10, and WEBKB) in a way that makes it simple for users to test out new algorithms and rapidly compare against existing methods.

This package reproduces experiments from the paper

Calder, Cook, Thorpe, Slepcev. [Poisson Learning: Graph Based Semi-Supervised Learning at Very Low Label Rates.](http://proceedings.mlr.press/v119/calder20a.html), Proceedings of the 37th International Conference on Machine Learning, PMLR 119:1306-1316, 2020.

## Installation

Install with

```
pip install graphlearning
```

Required packages include numpy, scipy, sklearn, matplotlib, and torch. The packages annoy and kymatio are required for running nearest neighbor searches and the scattering transform, respectively, but the rest of the code will run fine without those packages.

To install from the github source, which is updated more frequently, run

```
git clone https://github.com/jwcalder/GraphLearning
cd GraphLearning
pip install -r requirements.txt
python setup.py install --user
```

## Getting started with basic experiments
Below we outline some basic ways the package can be used. The [examples](https://github.com/jwcalder/GraphLearning/tree/master/examples) page from our GitHub repository contains several detailed example scripts that are useful for getting started.

A basic experiment comparing Laplace learning/Label propagation to Poisson learning on MNIST can be run with

```
import graphlearning as gl
gl.ssl_trials(dataset='mnist',metric='vae',algorithm='laplace',k=10,t=10)
gl.ssl_trials(dataset='mnist',metric='vae',algorithm='poisson',k=10,t=10)
```

Supported datasets include MNIST, FashionMNIST, WEBKB, and cifar. The metric is used for constructing the graph, and can be 'raw' for all datasets, which is Euclidean distance between raw data, 'vae' for MNIST and FashionMNIST, which is the variational autoencoder weights as described in our paper, 'scatter', which uses the scattering transform, or 'aet' for cifar, which uses the AutoEncoding Transformations weights, also described in our paper. The 'k=10' specifies how many nearest neighbors to use in constructing the graph, and 't=10' specifies how many trials to run, randomly assigning training/testing data. There are many other optional arguments, and full documentation is coming soon.

Below is a list of currently supported algorithms with links to the corresponding papers.

**Semi-supervised learning:** [Laplace](https://www.aaai.org/Papers/ICML/2003/ICML03-118.pdf), [RandomWalk](https://link.springer.com/chapter/10.1007/978-3-540-28649-3_29), [Poisson](https://arxiv.org/abs/2006.11184), [PoissonMBO](https://arxiv.org/abs/2006.11184), [pLaplace](https://arxiv.org/abs/1901.05031), [WNLL](https://link.springer.com/article/10.1007/s10915-017-0421-z), [ProperlyWeighted](https://arxiv.org/abs/1810.04351), NearestNeighbor, [MBO](https://ieeexplore.ieee.org/abstract/document/6714564), [ModularityMBO](https://doi.org/10.1137/17M1138972), [VolumeMBO](https://link.springer.com/chapter/10.1007/978-3-319-58771-4_27), [DynamicLabelPropagation](https://www.sciencedirect.com/science/article/abs/pii/S0031320315003738), [SparseLabelPropagation](https://arxiv.org/abs/1612.01414), [CenteredKernel](https://romaincouillet.hebfree.org/docs/conf/SSL_ICML18.pdf)


**Clustering:** [INCRES](https://link.springer.com/chapter/10.1007/978-3-319-91274-5_9), [Spectral](https://link.springer.com/article/10.1007/s11222-007-9033-z), [SpectralShiMalik](https://ieeexplore.ieee.org/abstract/document/868688), [SpectralNgJordanWeiss](http://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm.pdf)

The algorithm names are case-insensitive in all scripts. NearestNeighbor chooses the label of the closest labeled node in the geodesic graph distance.

The accuracy scores are saved in the subdirectory Results/ using a separate .csv file for each experiment. These can be loaded to generate plots and tables (see the [example](https://github.com/jwcalder/GraphLearning/tree/master/examples) scripts). The directory ResultsFromPaper/ contains all results from our ICML paper.

The commands shown above are rather high level, and can be split into several important subroutines when needed. The code below shows how to generate a weight matrix on the MNIST dataset, choose training data randomly, run Laplace and Poisson learning, and compute accuracy scores.

```
import graphlearning as gl

#Load labels, knndata, an build 10-nearest neighbor weight matrix
labels = gl.load_labels('mnist')
I,J,D = gl.load_kNN_data('mnist',metric='vae')
W = gl.weight_matrix(I,J,D,10)

#Randomly chose training datapoints
num_train_per_class = 1 
train_ind = gl.randomize_labels(labels, num_train_per_class)
train_labels = labels[train_ind]

#Run Laplace and Poisson learning
labels_laplace = gl.graph_ssl(W,train_ind,train_labels,algorithm='laplace')
labels_poisson = gl.graph_ssl(W,train_ind,train_labels,algorithm='poisson')

#Compute and print accuracy
print('Laplace learning: %.2f%%'%gl.accuracy(labels,labels_laplace,len(train_ind)))
print('Poisson learning: %.2f%%'%gl.accuracy(labels,labels_poisson,len(train_ind)))
```


## Contact and questions


Email <jwcalder@umn.edu> with any questions or comments.

## Acknowledgments

Several people have contributed to the development of this software:

1. Mauricio Rios Flores (Machine Learning Researcher, Amazon)
2. Brendan Cook (PhD Candidate in Mathematics, University of Minnesota)
3. Matt Jacobs (Postdoc, UCLA)
