"""
Weight Matrices
==========

This module implements functions that are useful for constructing sparse weight matrices, including 
efficient high dimensional nearest neighbor searches.
"""

import numpy as np
from scipy import spatial
from scipy import sparse
import os
from . import utils

#Directory to store knn data
knn_dir = os.path.abspath(os.path.join(os.getcwd(),'knn_data'))

def knn(data, k, kernel='gaussian', eta=None, symmetrize=True, metric='raw', knn_data=None):
    """knn weight matrix
    ======

    General function for constructing knn weight matrices.
   
    Parameters
    ----------
    data : (n,m) numpy array, or string 
        If numpy array, n data points, each of dimension m, if string, then 'mnist', 'fashionmnist', or 'cifar'
    k : int
        Number of nearest neighbors to use.
    kernel : string (optional), {'uniform','gaussian','singular','distance'}, default='gaussian'
        The choice of kernel in computing the weights between \\(x_i\\) and each of its k 
        nearest neighbors. We let \\(d_k(x_i)\\) denote the distance from \\(x_i\\) to its kth 
        nearest neighbor. The choice 'uniform' corresponds to \\(w_{i,j}=1\\) and constitutes
        an unweighted k nearest neighbor graph, 'gaussian' corresponds to
        \\[ w_{i,j} = \\exp\\left(\\frac{-4\\|x_i - x_j\\|^2}{d_k(x_i)^2} \\right), \\]
        'distance' corresponds to
        \\[ w_{i,j} = \\|x_i - x_j\\|, \\]
        and 'singular' corresponds to 
        \\[ w_{i,j} = \\frac{1}{\\|x_i - x_j\\|}, \\]
        when \\(i\\neq j\\) and \\(w_{i,i}=1\\).
    eta : python function handle (optional)
        If provided, this overrides the kernel option and instead uses the weights
        \\[ w_{i,j} = \\eta\\left(\\frac{\\|x_i - x_j\\|^2}{d_k(x_i)^2} \\right), \\]
        where \\(d_k(x_i)\\) is the distance from \\(x_i\\) to its kth nearest neighbor.
    symmetrize : bool (optional), default=True, except when kernel='singular'
        Whether or not to symmetrize the weight matrix before returning. Symmetrization is 
        performed by returning \\( (W + W^T)/2 \\), except for when kernel='distance, in 
        which case the symmetrized edge weights are the true distances. Default for symmetrization
        is True, unless the kernel is 'singular', in which case it is False.
    metric : string (optional), default='raw'
        Metric identifier if data is a string (i.e., a dataset).
    knn_data : tuple (optional), default=None
        If desired, the user can provide knn_data = (knn_ind, knn_dist), the output of a knnsearch,
        in order to bypass the knnsearch step, which can be slow for large datasets.

    Returns
    -------
    W : (n,n) scipy sparse matrix, float 
        Sparse weight matrix.
    """

    #If knn_data provided
    if knn_data is not None:
        knn_ind, knn_dist = knn_data

    #If data is a string, then load knn from a stored dataset
    elif type(data) is str:
        knn_ind, knn_dist = load_knn_data(data, metric=metric)

    #Else we have to run a knnsearch
    else:
        knn_ind, knn_dist = knnsearch(data, k)

    #Restrict to k nearest neighbors
    n = knn_ind.shape[0]
    k = np.minimum(knn_ind.shape[1],k)
    knn_ind = knn_ind[:,:k]
    knn_dist = knn_dist[:,:k]

    #If eta is None, use kernel keyword
    if eta is None:

        if kernel == 'uniform':
            weights = np.ones_like(knn_dist)
        elif kernel == 'gaussian':
            D = knn_dist*knn_dist
            eps = D[:,k-1]
            weights = np.exp(-4*D/eps[:,None])
        elif kernel == 'distance':
            weights = knn_dist
        elif kernel == 'singular':
            weights = knn_dist
            weights[knn_dist==0] = 1
            weights = 1/weights
            symmetrize = False
        else:
            sys.exit('Invalid choice of kernel: ' + kernel)

    #Else use user-defined eta
    else:
        D = knn_dist*knn_dist
        eps = D[:,k-1]
        weights = eta(D/eps)

    #Flatten knn data and weights
    knn_ind = knn_ind.flatten()
    weights = weights.flatten()

    #Self indices
    self_ind = np.ones((n,k))*np.arange(n)[:,None]
    self_ind = self_ind.flatten()

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((weights, (self_ind, knn_ind)),shape=(n,n)).tocsr()

    if symmetrize:
        if kernel in ['distance','uniform']:
            W = utils.sparse_max(W, W.transpose())
        else:
            W = (W + W.transpose())/2;

    return W

def epsilon_ball(data, epsilon, kernel='gaussian', eta=None):
    """Epsilon ball weight matrix
    ======

    General function for constructing a sparse epsilon-ball weight matrix, whose weights have the form
    \\[ w_{i,j} = \\eta\\left(\\frac{\\|x_i - x_j\\|^2}{\\varepsilon^2} \\right), \\]
    when \\(\\|x_i - x_j\\|\\leq \\varepsilon\\), and \\(w_{i,j}=0\\) otherwise.
    This type of weight matrix is only feasible in relatively low dimensions.
   
    Parameters
    ----------
    data : (n,m) numpy array
        n data points, each of dimension m
    epsilon : float
        Connectivity radius
    kernel : string (optional), {'uniform','gaussian','singular','distance'}, default='gaussian'
        The choice of kernel in computing the weights between \\(x_i\\) and \\(x_j\\) when
        \\(\\|x_i-x_j\\|\\leq \\varepsilon\\). The choice 'uniform' corresponds to \\(w_{i,j}=1\\) 
        and constitutes an unweighted graph, 'gaussian' corresponds to
        \\[ w_{i,j} = \\exp\\left(\\frac{-4\\|x_i - x_j\\|^2}{\\varepsilon^2} \\right), \\]
        'distance' corresponds to
        \\[ w_{i,j} = \\|x_i - x_j\\|, \\]
        and 'singular' corresponds to 
        \\[ w_{i,j} = \\frac{1}{\\|x_i - x_j\\|}, \\]
        when \\(i\\neq j\\) and \\(w_{i,i}=1\\).
    eta : python function handle (optional)
        If provided, this overrides the kernel option and instead uses the weights
        \\[ w_{i,j} = \\eta\\left(\\frac{\\|x_i - x_j\\|^2}{\\varepsilon^2} \\right). \\]

    Returns
    -------
    W : (n,n) scipy sparse matrix, float 
        Sparse weight matrix.
    """
    n = data.shape[0]  #Number of points

    #Rangesearch to find nearest neighbors
    Xtree = spatial.cKDTree(data)
    M = Xtree.query_pairs(epsilon)
    M = np.array(list(M))

    #Differences between points and neighbors
    V = data[M[:,0],:] - data[M[:,1],:]
    dists = np.sum(V*V,axis=1)

    #If eta is None, use kernel keyword
    if eta is None:

        if kernel == 'uniform':
            weights = np.ones_like(dists)
            fzero = 1
        elif kernel == 'gaussian':
            weights = np.exp(-4*dists/(epsilon*epsilon))
            fzero = 1
        elif kernel == 'distance':
            weights = np.sqrt(dists)
            fzero = 0
        elif kernel == 'singular':
            weights = np.sqrt(dists)
            weights[dists==0] = 1
            weights = 1/weights
            fzero = 1
        else:
            sys.exit('Invalid choice of kernel: ' + kernel)

    #Else use user-defined eta
    else:
        weights = eta(dists/(epsilon*epsilon))
        fzero = eta(0)

    #Weights

    #Symmetrize weights and add diagonal entries
    weights = np.concatenate((weights,weights,fzero*np.ones(n,)))
    M1 = np.concatenate((M[:,0],M[:,1],np.arange(0,n)))
    M2 = np.concatenate((M[:,1],M[:,0],np.arange(0,n)))

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((weights, (M1,M2)),shape=(n,n))

    return W.tocsr()


def knnsearch(X, k, method=None, similarity='euclidean', dataset=None, metric='raw'):
    """knn search
    ======

    General function for k-nearest neighbor searching, including efficient 
    implementations for high dimensional data, and support for saving
    k-nn data to files automatically, for reuse later.

   
    Parameters
    ----------
    X : (n,m) numpy array
        n data points, each of dimension m.
    k : int
        Number of nearest neighbors to find.
    method : {'kdtree','annoy','brute'} (optional), default: 'kdtree' for m <=5 and 'annoy' for m>5
        Algorithm for search. Annoy is an approximate nearest neighbor search and requires
        the [Annoy](https://github.com/spotify/annoy) package. 
    similarity : {'euclidean','angular','manhattan','hamming','dot'} (optional), default='euclidean'
        Smilarity for nearest neighbor search. Only 'euclidean' and 'angular' are available with
        'kdtree' and 'brute'.
    dataset : string (optional), default=None
        If provided, results of the search are saved to a file that can be loaded later.
    metric : string (optional), default='raw'
        A modifier to add to the dataset name when saving, to distinguish different types of knn data.

    Returns
    -------
    knn_ind : (n,k) numpy array, int
        Indices of nearest neighbors, including the self point.
    knn_dist : (n,k) numpy array, float
        Distances to all neighbors.
    """

    n = X.shape[0]
    m = X.shape[1]
    if method is None:
        if m <= 5:
            method = 'kdtree'
        else:
            method = 'annoy'

    if method in ['kdtree','brute']:

        if not similarity in ['angular','euclidean']:
            sys.exit('Invalid choice of similarity ' + similarity)

        if similarity == 'angular':
            X /= np.linalg.norm(X,axis=1)[:,None]

        if method == 'kdtree':

            Xtree = spatial.cKDTree(X)
            knn_dist, knn_ind = Xtree.query(X,k=k)

        else: #Brute force knn search

            knn_ind = np.array((n,k),dtype=int)
            knn_dist = np.array((n,k))
            for i in range(n):
                dist  = np.linalg.norm(X - X[i,:],axis=1) 
                knn_ind[i,:] = np.argsort(dist)[:k]
                knn_dist[i,:] = dist[knn_ind]

    elif method == 'annoy':

        if not similarity in ['euclidean','angular','manhattan','hamming','dot']:
            sys.exit('Invalid choice of similarity ' + similarity)

        from annoy import AnnoyIndex

        u = AnnoyIndex(m, similarity)  # Length of item vector that will be indexed
        for i in range(n):
            u.add_item(i, X[i,:])

        u.build(10)  #10 trees
        
        knn_dist = []
        knn_ind = []
        for i in range(n):
            A = u.get_nns_by_item(i, k, include_distances=True, search_k=-1)
            knn_ind.append(A[0])
            knn_dist.append(A[1])

        knn_ind = np.array(knn_ind)
        knn_dist = np.array(knn_dist)

    else:
        sys.exit('Invalid choice of knnsearch method ' + method)

 
    #If dataset name is provided, save permutations to file
    if not dataset is None:
        #data file name
        dataFile = dataset.lower() + '_' + metric.lower() + '.npz'

        #Full path to file
        dataFile_path = os.path.join(knn_dir, dataFile)

        #Check if knn_dir exists
        if not os.path.exists(knn_dir):
            os.makedirs(knn_dir)

        np.savez_compressed(dataFile_path, J=knn_ind, D=knn_dist)

    return knn_ind, knn_dist

def load_knn_data(dataset, metric='raw'):
    """Load saved knn data
    ======

    Loads the results of a saved knn search.   

    Parameters
    ----------
    dataset : string
        Name of dataset to load knn data for (not case-sensitive).
    metric : string (optional), default='raw'
        A modifier to add to the dataset name when saving, to distinguish different types of knn data (not case-sensitive).

    Returns
    -------
    knn_ind : (n,k) numpy array, int
        Indices of nearest neighbors, including the self point.
    knn_dist : (n,k) numpy array, float
        Distances to all neighbors.
    """

    #dataFile = dataset.lower() + "_" + metric.lower() + ".npz" #Fix this later
    dataFile = dataset.lower() + "_" + metric.lower() + ".npz" 
    dataFile_path = os.path.join(knn_dir, dataFile)

    #Check if knn_dir exists
    if not os.path.exists(knn_dir):
        os.makedirs(knn_dir)

    #Download kNN data if necessary
    if not os.path.exists(dataFile_path):
        urlpath = 'https://github.com/jwcalder/GraphLearning/raw/master/kNNData/'+dataFile
        utils.download_file(urlpath, dataFile_path)

    knn_ind = utils.numpy_load(dataFile_path, 'J')
    knn_dist = utils.numpy_load(dataFile_path, 'D')

    return knn_ind, knn_dist






