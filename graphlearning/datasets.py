"""
Datasets
==========

This module allows for loading standard datasets (currently mnist, fashionmnist, cifar10, signmnist), and creating
and saving new datasets by name locally.
"""

import numpy as np
import ssl
import os
import matplotlib.pyplot as plt
from . import utils
from . import graph

#Directory for storing datasets
data_dir = os.path.abspath(os.path.join(os.getcwd(),'data'))

def two_skies(n,sigma=0.15,sep=0.64):
    """Two skies dataset
    ======

    Random sample from the two skies dataset.

    Returns
    -------
    n : int
        Number of data points (should be even).
    sigma : float (optional)
        Standard deviation of the skies.
    sep : float (optional)
        Separation between the two skies.
        

    Returns
    -------
    data : numpy array, float
        (n,2) numpy array of data.
    labels : numpy array, int
        Binary labels indicating two skies.

    """

    m = int(n/2)
    y1 = sigma*np.random.randn(m,1) + sep/2
    y2 = sigma*np.random.randn(m,1) - sep/2
    y = np.vstack((y1,y2))
    x = np.random.rand(2*m,1)
    labels = np.vstack((np.zeros(m),np.ones(m)))
    data = np.hstack((x,y))
    return data,labels


def save(data, labels, dataset, metric='raw', overwrite=False):
    """Save dataset
    ======

    Add a new dataset to graph learning by saving the data and labels.
   
    Parameters
    ----------
    data : (n,m) numpy array, float
        n data points in m dimensions.
    labels : Length n numpy array, int
        Integer values for labels. 
    dataset : string
        Name of dataset.
    metric : string (optional), default='raw'
        A modifier to add to the dataset name when saving, to distinguish 
        different types of knn data (not case-sensitive).
    overwrite : bool (optional), default=False
        Whether to overwrite if dataset already exists.
    """

    #Dataset filename
    dataFile = dataset.lower()+"_"+metric.lower()+".npz"
    labelsFile = dataset.lower()+"_labels.npz"

    #Full path to file
    dataFile_path = os.path.join(data_dir, dataFile)
    labelsFile_path = os.path.join(data_dir, labelsFile)

    #Check if Data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    #Save dataset and labels
    if os.path.isfile(dataFile_path) and not overwrite:
        print('Data file '+dataFile_path+' already exists. Not saving.')
    else:
        np.savez_compressed(dataFile_path,data=data)
        np.savez_compressed(labelsFile_path,labels=labels)


def load(dataset, metric='raw', labels_only=False):
    """Load dataset
    ======

    Load a dataset. Currently implemented datasets include

    1. [mnist](http://yann.lecun.com/exdb/mnist/): metrics are 'raw' and 'vae' (variational autoencoder)
    2. [fashionmnist](https://github.com/zalandoresearch/fashion-mnist): metrics are 'raw' and 'vae' 
    3. [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html): metrics are 'raw' and 'simclr'. Loads CIFAR-10.
    4. [yalefaces](https://paperswithcode.com/dataset/extended-yale-b-1): Only metric is 'raw'.
    5. [signmnist](https://www.kaggle.com/datasets/datamunge/sign-language-mnist): Sign language version of MNIST.
   
    Parameters
    ----------
    dataset : string, {'mnist', 'fashionmnist', 'cifar'}
        Name of dataset.
    metric : string (optional), default='raw'
        Indicates the embedding method used in the graph construction. For example, dataset='mnist' with
        metric='vae' loads the latent features from a variational autoencoder trained on MNIST.
    labels_only : bool (optional), default=False
        Whether to return only the labels. Useful if the dataset is very large and knndata is already 
        precomputed, so the raw features are not needed.
    
    Returns
    -------
    data : numpy array, float
        (n,d) numpy array of n datapoints in dimension d. Not returned if `labels_only=True`.
    labels : numpy array, int
        Integer-valued labels in range 0 through k-1, where k is the number of classes.
    """

    #Dataset filename
    dataFile = dataset.lower()+"_"+metric.lower()+".npz"
    labelsFile = dataset.lower()+"_labels.npz"

    #Full path to file
    dataFile_path = os.path.join(data_dir, dataFile)
    labelsFile_path = os.path.join(data_dir, labelsFile)

    #Check if Data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    #Download labels file if needed
    if not os.path.exists(labelsFile_path):
        urlpath = 'https://github.com/jwcalder/GraphLearning/raw/master/Data/'+labelsFile
        utils.download_file(urlpath, labelsFile_path)

    #Load labels from npz file
    labels = utils.numpy_load(labelsFile_path, 'labels')

    if labels_only:
        return labels
    else:
        #Download dataset file if needed
        if not os.path.exists(dataFile_path):
            urlpath = 'http://www-users.math.umn.edu/~jwcalder/Data/'+dataFile
            utils.download_file(urlpath, dataFile_path)

        data = utils.numpy_load(dataFile_path, 'data')
        return data, labels

def load_graph(name):
    """Load graph
    ======

    Load a graph. Currently implemented graphs include

    1. [karate](https://en.wikipedia.org/wiki/Zachary's_karate_club): Zachary's karate club [1]
    2. [cora](https://proceedings.mlr.press/v48/yanga16): Cora citation graph [2]
    3. [citeseer](https://proceedings.mlr.press/v48/yanga16): CiteSeer citation graph [2]
    4. [pubmed](https://proceedings.mlr.press/v48/yanga16): PubMed citation graph [2]
    5. [webkb_cornell](https://openreview.net/forum?id=S1e2agrFvS): WebKB Cornell graph [4]
    6. [webkb_texas](https://openreview.net/forum?id=S1e2agrFvS): WebKB Texas graph [4]
    7. [webkb_wisconsin](https://openreview.net/forum?id=S1e2agrFvS): WebKB Wisconsin graph [4]
    8. [nell](https://proceedings.mlr.press/v48/yanga16): The NELL knowledge graph [2,3]
    9. [wikics](https://arxiv.org/abs/2007.02901): The Wiki-CS graph [5]
    10. [airports_usa](https://arxiv.org/abs/1704.03165): The USA airports graph [6,7]
    11. [airports_brazil](https://arxiv.org/abs/1704.03165): The Brazil airports graph [6,7]
    12. [airports_europe](https://arxiv.org/abs/1704.03165): The Europe airports graph [6,7]
    13. [polbooks](https://www.pnas.org/doi/full/10.1073/pnas.0601602103): The Political books graph [8]

    [1] Zachary, W. W. (1977). "An Information Flow Model for Conflict and Fission in Small Groups". Journal of Anthropological Research. 33 (4): 452–473

    [2] Yang, Zhilin, William Cohen, and Ruslan Salakhudinov. "Revisiting semi-supervised learning with graph embeddings." International conference on machine learning. PMLR, 2016.

    [3] Carlson, Andrew, Justin Betteridge, Bryan Kisiel, Burr Settles, Estevam Hruschka, and Tom Mitchell. "Toward an architecture for never-ending language learning." In Proceedings of the AAAI conference on artificial intelligence, vol. 24, no. 1, pp. 1306-1313. 2010.

    [4] Pei, Hongbin, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, and Bo Yang. "Geom-GCN: Geometric Graph Convolutional Networks." In International Conference on Learning Representations. 2019.

    [5] Mernyei, Péter, and Cătălina Cangea. "Wiki-cs: A wikipedia-based benchmark for graph neural networks." arXiv preprint arXiv:2007.02901 (2020).

    [6] Figueiredo, D.R., Ribeiro, L.F.R. and Saverese, P.H., 2017. struc2vec: Learning node representations from structural identity. CoRR, vol. abs/1704.03165.

    [7] Jin, Y., Song, G. and Shi, C., 2020, April. GraLSP: Graph neural networks with local structural patterns. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 4361-4368).
  
    [8] Newman, M.E., 2006. Modularity and community structure in networks. Proceedings of the national academy of sciences, 103(23), pp.8577-8582.

    Parameters
    ----------
    name : string, {'karate','cora','citeseer','pubmed','webkb_cornell','webkb_texas','webkb_wisconsin','nell','wikics'}
        Name of dataset.
    
    Returns
    -------
    G : graphlearning graph object
        Graph object with weight matrix and labels/features if available
    """

    #Dataset filename
    dataFile = name.lower()+".pkl"

    #Full path to file
    dataFile_path = os.path.join(data_dir, dataFile)

    #Check if Data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    #Download dataset file if needed
    if not os.path.exists(dataFile_path):
        urlpath = 'http://www-users.math.umn.edu/~jwcalder/Data/'+dataFile
        utils.download_file(urlpath, dataFile_path)

    return graph.graph.load(dataFile_path[:-4])

def load_image(name):
    """Load image 
    ======

    Load an image. 

    Parameters
    ----------
    name : string
        Name of image, choices are {'cameraman', 'cow', 'house', 'jetplane', 'lake', 'mandril_color', 'mandril_gray', 'peppers_color', 'peppers_gray', 'pirate', 'walkbridge', 'chairtoy', 'chairtoy_highres','chairtoy_bw', 'chairtoy_highres_bw'}
    Returns
    -------
    image : numpy array, float
        (m,n) or (m,n,3) numpy array containing image.
    """


    #Dataset filename
    dataFile = name.lower()+'.png'

    #Full path to file
    dataFile_path = os.path.join(data_dir, dataFile)

    #Check if Data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    #Download image file if needed
    if not os.path.exists(dataFile_path):
        urlpath = 'http://www-users.math.umn.edu/~jwcalder/TestImages/'+dataFile
        utils.download_file(urlpath, dataFile_path)

    #Load image
    image = plt.imread(dataFile_path)

    return image




