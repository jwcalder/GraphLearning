"""
Datasets
==========

This module allows for loading standard datasets (currently mnist, fashionmnist, cifar), and creating
and saving new datasets by name locally.
"""

import numpy as np
import ssl
import os
from . import utils

#Directory for storing datasets
data_dir = os.path.abspath(os.path.join(os.getcwd(),'data'))

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
    3. [cifar](https://www.cs.toronto.edu/~kriz/cifar.html): metrics are 'raw' and 'aet' (autoencoding transformations). Loads CIFAR-10.
   
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


