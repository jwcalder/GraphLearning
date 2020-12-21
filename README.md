# Graph-based Clustering and Semi-Supervised Learning

![Clustering](images/clustering.png)

This python package is devoted to efficient implementations of modern graph-based learning algorithms for both semi-supervised learning and clustering. The package implements many popular datasets (currently MNIST, FashionMNIST, cifar-10, and WEBKB) in a way that makes it simple for users to test out new algorithms and rapidly compare against existing methods.

Download the package locally with 

```
git clone https://github.com/jwcalder/GraphLearning
```

The main python file is graphlearning.py. The demo scripts semi-supervised_demo.py and clustering_demo.py give basic examples of how to use the package on synthetic data. The clustering script reproduces the figures above, which are the result of spectral clustering on toy examples. The file graphlearning.py contains a main subroutine that implements a user-friendly interface to run experiments comparing different datasets and algorithms over randomization of labeled and unlabeled data.

This package also reproduces experiments from our paper

Calder, Cook, Thorpe, Slepcev. [Poisson Learning: Graph Based Semi-Supervised Learning at Very Low Label Rates.](https://arxiv.org/abs/2006.11184) To appear in International Conference on Machine Learning (ICML) 2020. 

## Getting started with basic experiments

A basic experiment comparing Laplace learning/Label propagation to Poisson learning on MNIST can be run from a shell with the commands

```
python graphlearning.py -d MNIST -m vae -a Laplace -k 10 -t 10
python graphlearning.py -d MNIST -m vae -a Poisson -k 10 -t 10
```

or equivalently from a python script with the code

```
import graphlearning as gl
gl.main(dataset='mnist',metric='vae',algorithm='laplace',k=10,t=10)
gl.main(dataset='mnist',metric='vae',algorithm='laplace',k=10,t=10)
```

The flag -d specifies the dataset (MNIST, FashionMNIST, WEBKB, or cifar), -m specifies the metric for constructing the graph, -a is the choice of semi-supervised learning or clustering algorithm, -k is the number of nearest neighbors in the graph construction, and -t is the number of trials to run. The choices for metric are 'L2' for all datasets, which is Euclidean distance between raw data. MNIST and FashionMNIST have the option of 'vae', which is the variational autoencoder weights as described in our paper, as well as scatter, which uses the scattering transform. For cifar, the metric 'aet' is the AutoEncoding Transformations weights, as described in our paper. 

The accuracy scores are saved in the subdirectory Results/ using a separate .csv file for each experiment. These can be loaded to generate plots and tables (see plot.py and table.py). The directory ResultsFromPaper/ contains all results from our ICML paper.

All options for the graphlearning.py script can be displayed by running the code with the -h option

```
python graphlearning.py -h
```

The package also has implementations of some graph-based clustering algorithms. For example, run 

```
python graphlearning.py -d MNIST -m vae -a SpectralNgJordanWeiss -x 4
python graphlearning.py -d MNIST -m vae -a INCRES
```

from a shell, or equivalently in Python run 

```
import graphlearning as gl
gl.main(dataset='mnist',metric='vae',algorithm='spectralngjordanweiss',num_classes=10,extra_dim=4)
gl.main(dataset='mnist',metric='vae',algorithm='incres',num_classes=10)
```

to perform spectral clustering and INCRES clustering on MNIST. The package will detect whether to perform clustering or semi-supervised learning based on the choice of algorithm.

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
print('Laplace learning: %.2f%%'%gl.accuracy(labels,labels_laplace,num_train_per_class))
print('Poisson learning: %.2f%%'%gl.accuracy(labels,labels_poisson,num_train_per_class))
```

### List of currently supported algorithms

Below is a list of currently supported algorithms with links to the corresponding papers.

**Semi-supervised learning:** [Laplace](https://www.aaai.org/Papers/ICML/2003/ICML03-118.pdf), [RandomWalk](https://link.springer.com/chapter/10.1007/978-3-540-28649-3_29), [Poisson](https://arxiv.org/abs/2006.11184), [PoissonMBO](https://arxiv.org/abs/2006.11184), [pLaplace](https://arxiv.org/abs/1901.05031), [WNLL](https://link.springer.com/article/10.1007/s10915-017-0421-z), [ProperlyWeighted](https://arxiv.org/abs/1810.04351), NearestNeighbor, [MBO](https://ieeexplore.ieee.org/abstract/document/6714564), [VolumeMBO](https://link.springer.com/chapter/10.1007/978-3-319-58771-4_27), [DynamicLabelPropagation](https://www.sciencedirect.com/science/article/abs/pii/S0031320315003738), [SparseLabelPropagation](https://arxiv.org/abs/1612.01414), [CenteredKernel](https://romaincouillet.hebfree.org/docs/conf/SSL_ICML18.pdf)


**Clustering:** [INCRES](https://link.springer.com/chapter/10.1007/978-3-319-91274-5_9), [Spectral](https://link.springer.com/article/10.1007/s11222-007-9033-z), [SpectralShiMalik](https://ieeexplore.ieee.org/abstract/document/868688), [SpectralNgJordanWeiss](http://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm.pdf)

The algorithm names are case-insensitive in all scripts. NearestNeighbor chooses the label of the closest labeled node in the geodesic graph distance.


## Label Permutations

The randomization of labeled vs unlabeled data is controlled by label permutation files stored in the subdirectory LabelPermutations/, to ensure the randomization is the same among all algorithms being compared. A randomized label permutation file can be generated with the script CreateLabelPermutation.py as below:

```
python CreateLabelPermutation.py -d MNIST -m 1,2,3,4,5 -t 100
```

The flag -m controls the label rate; the command above uses 1,2,3,4, and 5, labels per class. The flag -t controls how many trials. So the command above will produce 500 separate experiments, 100 at 1 label per class, 100 at 2 labels per class, etc. There is also a flag -n to give the label permutation a different name from the default one. The label permutations provided in github were constructed as above.

## Nearest neighbor data

All datasets have been preprocessed with a feature transformation followed by a k-nearest neighbor search. The k-nearest neighbor information is stored in the subdirectory kNNData/, and this data is loaded by graphlearning.py to construct a weight matrix for the graph. This allows the user flexibility in the construction of the graph, and removes the need to constantly recompute the k-nearest neighbors, which is computationally expensive.

Therefore, the raw data (e.g., MNIST images) is rarely required to run experiments, and due to size restrictions is not provided in GitHub. To download and load the raw data, use the graphlearning.load_dataset function, as in

```
import graphlearing as gl
data = gl.load_dataset('mnist')
```

The script ComputeKNN.py can be used to perform the preprocessing steps of applying a feature transformation followed by a k-nearest neighbor search. To run this on MNIST with the scattering transform as the feature transformation, run 

```
python ComputeKNN.py -d MNIST -m scatter
```

The deep learning-based feature transformations (e.g., variational autoencoder (vae) or autoencoding transformations (aet) weights) are not provided as built-in subroutines. Normally it will only be necessary to run ComputeKNN.py when adding a new dataset or trying a new feature transformation on an existing dataset.


## Plotting and LaTeX table creation

The accuracy results for each trial are saved to .csv files in the subdirectory Results/. The package has built-in functions to easily create plots and LaTeX tables of accuracy scores. To run experiments comparing Laplace and Poisson learning and generate accuracy plots and tables, run

```
python graphlearning.py -d MNIST -m vae -a Laplace -t 10
python graphlearning.py -d MNIST -m vae -a Poisson -t 10
python plot.py
python table.py
```

## Python requirements and C code extensions

To install required non-standard packages:

```
pip install -r requirements.txt
```

Some parts of the package rely on C code acceleration that needs to be compiled. The package will attempt to automatically compile the C extensions when they are needed. If this does not work for you, the command to compile them manually is 

```
python cmodules/cgraphpy_setup.py build_ext --inplace
```

This requires a C code compiler in your path. Only the algorithms VolumeMBO, pLaplace, and NearestNeighbor use C code acceleration. If you do not plan to use these algorithms, you can skip compiling the C code.

### Contact and questions


Email <jwcalder@umn.edu> with any questions or comments.

### Acknowledgments

Several people have contributed to the development of this software:

1. Mauricio Rios Flores (Machine Learning Researcher, Amazon)
2. Brendan Cook (PhD Candidate in Mathematics, University of Minnesota)
3. Matt Jacobs (Postdoc, UCLA)
