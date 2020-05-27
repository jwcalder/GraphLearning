# Graph-based Clustering and Semi-Supervised Learning

This python package is devoted to efficient implementations of modern graph-based learning algorithms for both semi-supervised learning and clustering. The package implements many popular datasets (currently MNIST, FashionMNIST, cifar-10, and WEBKB) in a way that makes it simple for users to test out new algorithms and rapidly compare against existing methods.

The main python file is graphlearning.py, which contains all the important subroutines. The demo scripts semi-supervised_demo.py and clustering_demo.py give basic examples of how to use the package on synthetic data. The file main.py implements a user-friendly interface to run experiments comparing different datasets and algorithms over randomization of labeled and unlabeled data.

## Compiling C code extensions

Some parts of the package rely on C code acceleration that needs to be compiled. To compile the C code run

```
python cmodules/cgraphpy_setup.py build_ext --inplace
```

## Getting started with basic experiments

A basic experiment comparing Laplace learning/Label propagation to Poisson learning on MNIST can be run with the commands below.

```
python main.py -d MNIST -m vae -a Laplace -t 10
python main.py -d MNIST -m vae -a Poisson -t 10
```

The flag -d specifies the dataset (MNIST, FashionMNIST, WEBKB, or cifar), -m specifies the metric for constructing the graph (vae is the variational autoencoder weights as described in our paper; other options are L2 and scatter), -a is the choice of semi-supervised learning or clustering algorithm, and -t is the number of trials to run. All scripts have a help flag -h that shows a detailed list of options. For example, run

```
python main.py -h
```

to see the list of all options for the main graph-based learning script. The package also has implementations of some graph-based clustering algorithms. For example, run 

```
python main.py -d MNIST -m vae -a SpectralNgJordanWeiss -x 4
python main.py -d MNIST -m vae -a INCRES
```

to perform spectral clustering and INCRES clustering method on MNIST.

## Label Permutations

The randomization of labeled vs unlabeled data is controlled by label permutation files stored in the subdirectory LabelPermutations/, to ensure the randomization is the same among all algorithms being compared. A randomized label permutation file can be generated with the script CreateLabelPermutation.py as below:

```
python CreateLabelPermutation.py -d MNIST -m 1,2,3,4,5 -t 100
```

The flag -m controls the label rate; the command above uses 1,2,3,4, and 5, labels per class. The flag -t controls how many trials. So the command above will produce 500 separate experiments, 100 at 1 label per class, 100 at 2 labels per class, etc. There is also a flag -n to give the label permutation a different name from the default one. The label permutations provided in github were constructed as above.

## Nearest neighbor data

All datasets have been preprocessed with a feature transformation followed by a k-nearest neighbor search. The k-nearest neighbor information is stored in the subdirectory kNNData/, and this data is loaded by main.py to construct a weight matrix for the graph. This allows the user flexibility in the construction of the graph, but removes the need to constantly recompute the k-nearest neighbors, which is computationally expensive.

In particular, the raw data from which the kNN data is computed is not provided in github, due to file size restrictions. If needed, the user can download the raw data

```
python download_data.py
```

Then the script ComputeKNN.py can be used to perform the preprocessing steps of applying a feature transformation followed by a k-nearest neighbor search. To run this on MNIST with the scattering transform as the feature transformation, run 

```
python ComputeKNN.py -d MNIST -m scatter
```

The deep learning-based feature transformations (e.g., variational autoencoder (vae) or autoencoding transformations (aet) weights) are not provided as built-in subroutines. Normally it only be necessary to run ComputeKNN.py when adding a new dataset or trying a new feature transformation on an existing dataset.


## Plotting and LaTeX table creation

The accuracy results for each trial are saved to .csv files in the subdirectory Results/. The package has built-in functions to easily create plots and LaTeX tables of accuracy scores. After running the Laplace and Poisson experiments above, run the scripts

```
python plot.py
python table.py
```

for examples of how to use the plotting and table functions. 


## List of currently supported algorithms

A list of currently supported algorithms for semi-supervised learning is:

Laplace, Poisson, PoissonMBO, PoissonMBOBalanced, NearestNeighbor, MBO, VolumeMBO, DynamicLabelPropagation, SparseLabelPropagation, CenteredKernel

A list of currently supported clustering algorithms is

INCRES, Spectral, SpectralShiMalik, SpectralNgJordanWeiss

The algorithm names are case-insensitive in all scripts.


Email <jwcalder@umn.edu> with any questions or comments.

