"""
Trainsets
==========

This module allows for generating training sets randomly for graph-based semi-supervised learning. 
It also allows for loading of pre-saved training sets, and to create and save training sets for future use
and reproducibility of experiments. 
"""

import numpy as np
import os
import sys
from . import utils

trainset_dir = os.path.abspath(os.path.join(os.getcwd(),'trainsets'))

def load(dataset, trainset_name = ''):
    """Load training sets
    ======

    Add a new dataset to graph learning by saving the data and labels.
   
    Parameters
    ----------
    dataset : string
        Name of dataset.
    trainset_name : string (optional), default=''
        A modifier to uniquely identify different training sets for each dataset.
    """

    dataFile = dataset.lower() + trainset_name.lower() +"_permutations.npz"   #Change this eventually
    dataFile_path = os.path.join(trainset_dir, dataFile)

    #Check if Data directory exists
    if not os.path.exists(trainset_dir):
        os.makedirs(trainset_dir)

    #Download trainset if needed
    if not os.path.exists(dataFile_path):
        urlpath = 'https://github.com/jwcalder/GraphLearning/raw/master/LabelPermutations/'+dataFile
        utils.download_file(urlpath, dataFile_path)

    trainset = utils.numpy_load(dataFile_path, 'perm')

    return trainset

def generate(labels, rate=1, num_trials=1, mask=None, dataset=None, trainset_name='', overwrite=False, seed=None):
    """Generate training sets
    ======

    Generates training sets at different labeling rates over multiple trials,
    including features to store the training set indices to file for reproducibility.
   
    Parameters
    ----------
    labels : numpy array, int
        Labels for the dataset as nonnegative integers. 
    rate : int, float, or numpy array
        Controls the number of labels per class. Functionality depends on the data type.
        
        1. A single integer is interpreted as the number of labels per class.
        2. A single float in the range [0,1] is interpreted as the fraction of training data.
        3. A numpy array of size (m,C) indicating different label rates, as int or float, for 
           m different subtrials. If C=1, then the rate is extended to all classes, while if C=num classes,
           then the rates are interpreted on a per-class basis.
    num_trials : int (optional), default=1
        Number of training sets to generate.
    mask : numpy array (optional), bool, default=None
        If provided, then the generated training set will be selected only from points where mask=True.
    dataset : string (optional), default=None
        Name of dataset. If provided, the generated training set is saved to a file
        so it can be loaded later for reproducibility.
    trainset_name : string (optional), default=''
        A modifier to uniquely identify different training sets for each dataset.
    overwrite : bool (optional), default=False
        Whether to overwrite an exisiting training set file.
    seed : int (optional), default=None
        Option to seed the random number generator.

    Returns
    -------
    trainset : numpy array or list of numpy arrays
        If m=1 and num_trials=1 then a numpy array with indices of training points is returned. 
        Otherwise, a list of numpy arrays are returned, one for each trial.
    """

    if seed is not None:
        np.random.seed(seed)
    
    unique_labels = np.unique(labels)
    num_per_class = np.bincount(labels)
    num_classes = len(unique_labels)
    num_points = len(labels)

    #Generate (m,C) integer numpy array giving number of 
    #training points per class per trial
    if type(rate) == int:
        rate = (np.ones(num_classes)[None,:]*rate).astype(int)
    elif type(rate) == float:
        rate = (rate*num_per_class[None,:]).astype(int)
    elif type(rate) == np.ndarray:
        ratetype = rate.dtype
        if rate.ndim != 2:
            sys.exit('Must provide a 2-dimensional array for rate')
        if rate.shape[1] == 1:
            rate = rate@np.ones((1,num_classes))
        if np.issubdtype(ratetype,np.integer):
            rate = rate.astype(int) 
        elif np.issubdtype(ratetype,np.floating):
            rate = (rate*num_per_class).astype(int)
        else:
            sys.exit('Invalid numpy array type '+rate.dtype)
    else:
        sys.exit('Invalid rate type '+str(type(rate)))

    if mask is None:
        mask = np.ones(num_points,dtype=bool)

    #Draw training sets at random
    trainset = list()
    for k in range(num_trials):
        for i in range(rate.shape[0]):
            L = list()
            for j, l in enumerate(unique_labels):
                p = ((labels == l) & mask).astype(float)
                p = p/np.sum(p)
                L = L + np.random.choice(num_points,size=rate[i,j],p=p,replace=False).tolist()
            L = np.array(L)
            trainset.append(L)
    
    #Remove outer list if only one trial
    if len(trainset)==1:
        trainset = trainset[0]

    #If dataset name is provided, save permutations to file
    if not dataset is None:

        trainset = np.array(trainset,dtype=object)

        #data file name
        dataFile = dataset.lower() + trainset_name.lower() + '_permutations.npz'

        #Full path to file
        dataFile_path = os.path.join(trainset_dir, dataFile)

        #Check if Data directory exists
        if not os.path.exists(trainset_dir):
            os.makedirs(trainset_dir)

        #Save permutations to file
        if os.path.isfile(dataFile_path) and not overwrite:
            print('Training set file '+dataFile_path+' already exists. Not saving.')
        else:
            np.savez_compressed(dataFile_path,perm=trainset)
   
    return trainset


