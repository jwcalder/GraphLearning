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
import sys
from . import utils

#Directory to store knn data
knn_dir = os.path.abspath(os.path.join(os.getcwd(),'knn_data'))

def grid_graph(n,m=None,return_xy=False):
    '''Grid graph
    ========

    Returns the adjacency matrix for a graph on a regular grid.

    Parameters
    ----------
    n : int
        Number of pixels wide. Or if n is an image, then (n,m) are taken as the shape
        of the first and second dimensions of the image.
    m : int (Optional)
        Number of pixels high. Does not need to be specified if n is an image.
    return_xy : bool (optional, default=False)
        Whether to return x,y coordinates as well

    Returns
    -------
    W : (m*n,m*n) scipy sparse matrix
        Weight matrix of nearest neighbor graph on (m,n) grid
    X : (m*n,2) numpy array, float
        Coordiantes of vertices of grid
    '''

    if m is None:
        s = n.shape
        m = s[1]
        n = s[0]
        
    xm, ym = np.meshgrid(np.arange(m),np.arange(n))
    c = (xm + m*ym).flatten()
    ne = (np.clip(xm + 1,0,m-1) + m*ym).flatten()
    nw = (np.clip(xm - 1,0,m-1) + m*ym).flatten()
    nn = (xm + m*np.clip(ym + 1,0,n-1)).flatten()
    ns = (xm + m*np.clip(ym - 1,0,n-1)).flatten()
    edges = np.vstack((c,ne)).T
    edges = np.vstack((edges,np.vstack((c,nw)).T))
    edges = np.vstack((edges,np.vstack((c,nn)).T))
    edges = np.vstack((edges,np.vstack((c,ns)).T))
    ind = edges[:,0] != edges[:,1]
    edges = edges[ind,:]
    W = sparse.coo_matrix((np.ones(len(edges)), (edges[:,0],edges[:,1])),shape=(m*n,m*n))

    if return_xy:
        X = np.vstack((xm.flatten(),ym.flatten())).T
        return W.tocsr(),X.astype(float)
    else:
        return W.tocsr()

def knn(data, k, kernel='gaussian', eta=None, symmetrize=True, metric='raw', similarity='euclidean', knn_data=None):
    """knn weight matrix
    ======

    General function for constructing knn weight matrices.
   
    Parameters
    ----------
    data : (n,m) numpy array, or string 
        If numpy array, n data points, each of dimension m, if string, then 'mnist', 'fashionmnist', or 'cifar'
    k : int
        Number of nearest neighbors to use.
    kernel : string (optional), {'uniform','gaussian','symgaussian','singular','distance'}, default='gaussian'
        The choice of kernel in computing the weights between \\(x_i\\) and each of its k 
        nearest neighbors. We let \\(d_k(x_i)\\) denote the distance from \\(x_i\\) to its kth 
        nearest neighbor. The choice 'uniform' corresponds to \\(w_{i,j}=1\\) and constitutes
        an unweighted k nearest neighbor graph, 'gaussian' corresponds to
        \\[ w_{i,j} = \\exp\\left(\\frac{-4\\|x_i - x_j\\|^2}{d_k(x_i)^2} \\right), \\]
        'symgaussian' corresponds to
        \\[ w_{i,j} = \\exp\\left(\\frac{-4\\|x_i - x_j\\|^2}{d_k(x_i)d_k(x_j)} \\right), \\]
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
        performed by returning \\( (W + W^T)/2 \\), except for when kernel='distance','singular', in 
        which case the symmetrized edge weights are the true distances (or inverses), kernel='uniform', 
        where the weights are all 0/1, or kernel='symgaussian', where the same formula 
        is used for symmetry. Default for symmetrization is True, unless the kernel is
        'singular', in which case it is False.
    metric : string (optional), default='raw'
        Metric identifier if data is a string (i.e., a dataset).
    similarity : {'euclidean','angular','manhattan','hamming','dot'} (optional), default='euclidean'
        Smilarity for nearest neighbor search.
    knn_data : tuple (optional), default=None
        If desired, the user can provide knn_data = (knn_ind, knn_dist), the output of a knnsearch,
        in order to bypass the knnsearch step, which can be slow for large datasets.

    Returns
    -------
    W : (n,n) scipy sparse matrix, float 
        Sparse weight matrix.
    """
    
    #Self is counted in knn data, so add one
    k += 1

    #If knn_data provided
    if knn_data is not None:
        knn_ind, knn_dist = knn_data

    #If data is a string, then load knn from a stored dataset
    elif type(data) is str:
        knn_ind, knn_dist = load_knn_data(data, metric=metric)

    #Else we have to run a knnsearch
    else:
        knn_ind, knn_dist = knnsearch(data, k, similarity=similarity)

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
        elif kernel == 'symgaussian':
            eps = knn_dist[:,k-1]
            weights = np.exp(-4 * knn_dist * knn_dist / eps[:,None] / eps[knn_ind])
        elif kernel == 'distance':
            weights = knn_dist
        elif kernel == 'singular':
            weights = knn_dist
            weights[knn_dist==0] = 1
            weights = 1/weights
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
        if kernel in ['distance','uniform','singular']:
            W = utils.sparse_max(W, W.transpose())
        elif kernel == 'symgaussian':
            W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)
        else:
            W = (W + W.transpose())/2;

    W.setdiag(0)
    return W

def epsilon_ball(data, epsilon, kernel='gaussian', features=None, epsilon_f=1, eta=None):
    """Epsilon ball weight matrix
    ======

    General function for constructing a sparse epsilon-ball weight matrix, whose weights have the form
    \\[ w_{i,j} = \\eta\\left(\\frac{\\|x_i - x_j\\|^2}{\\varepsilon^2} \\right), \\]
    when \\(\\|x_i - x_j\\|\\leq \\varepsilon\\), and \\(w_{i,j}=0\\) otherwise.
    This type of weight matrix is only feasible in relatively low dimensions.
    The diagonals are always zero.
   
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
    features : (n,k) numpy array (optional)
        If provided, then the weights are additionally multiplied by the similarity in features, so that
        \\[ w_{i,j} =  \\eta\\left(\\frac{\\|y_i - y_j\\|^2}{\\varepsilon_F^2} \\right)\\eta\\left(\\frac{\\|x_i - x_j\\|^2}{\\varepsilon^2} \\right), \\]
        when \\(\\|x_i - x_j\\|\\leq \\varepsilon\\), and \\(w_{i,j}=0\\) otherwise. The 
        vector \\(y_i\\) is the feature vector associated with datapoint i. The features
        are useful for building a similarity graph over an image for image segmentation, and 
        here the \\(y_i\\) are either the pixel values at pixel i, or some other image feature
        such as a texture indicator.
    epsilon_f : float (optional).
        Connectivity radius for features \\(\\varepsilon_F\\). Default is \\(\\varepsilon_F=1\\).
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

    if len(M) == 0:
        return sparse.csr_matrix((n,n)) 
    else:
        #Differences between points and neighbors
        V = data[M[:,0],:] - data[M[:,1],:]
        dists = np.sum(V*V,axis=1)
        weights, fzero = __weights__(dists,epsilon,kernel,eta)

        #Add differences in features
        if features is not None:
            VF = features[M[:,0],:] - features[M[:,1],:]
            Fdists = np.sum(VF*VF,axis=1)
            feature_weights, _ = __weights__(Fdists,epsilon_f,kernel,eta)
            weights = weights*feature_weights
            fzero = fzero**2

        weights = np.concatenate((weights,weights,fzero*np.ones(n,)))
        M1 = np.concatenate((M[:,0],M[:,1],np.arange(0,n)))
        M2 = np.concatenate((M[:,1],M[:,0],np.arange(0,n)))

        #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
        W = sparse.coo_matrix((weights, (M1,M2)),shape=(n,n))

        W.setdiag(0)
        return W.tocsr()

def __weights__(dists,epsilon,kernel,eta):

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

    return weights, fzero


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
            Y = X/np.linalg.norm(X,axis=1)[:,None]
        else:
            Y = X

        if method == 'kdtree':

            Xtree = spatial.cKDTree(Y)
            knn_dist, knn_ind = Xtree.query(Y,k=k)

        else: #Brute force knn search

            knn_ind = np.zeros((n,k),dtype=int)
            knn_dist = np.zeros((n,k))
            for i in range(n):
                dist  = np.linalg.norm(Y - Y[i,:],axis=1) 
                knn_ind[i,:] = np.argsort(dist)[:k]
                knn_dist[i,:] = dist[knn_ind[i,:]]

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
        eps = 1e-20
        for i in range(n):
            #Get extra neighbors, in case there are mistakes
            A = u.get_nns_by_item(i, min(2*k,n), include_distances=True)
            ind = np.array(A[0])
            #knn_dist.append(A[1]) #These distances are floating point (32-bit) precision
            #The code below computes them more accurately
            if similarity == 'euclidean':
                dist = np.linalg.norm(X[i,:] - X[ind,:],axis=1)
            elif similarity == 'angular':
                vi = X[i,:]/np.maximum(np.linalg.norm(X[i,:]),eps)
                vj = X[ind,:]/np.maximum(np.linalg.norm(X[ind,:],axis=1)[:,None],eps)
                dist = np.linalg.norm(vi-vj,axis=1)
            elif similarity == 'manhattan':
                dist = np.linalg.norm(X[i,:] - X[ind,:],axis=1,ord=1)
            elif similarity == 'hamming':
                dist = A[1] #hamming is integer-valued, so no need to compute in double precision
            elif similarity == 'dot':
                dist = np.sum(X[i,:]*X[ind,:],axis=1)
            else:
                dist = A[1]

            ind_sort = np.argsort(dist)[:k]
            ind = ind[ind_sort]
            dist = dist[ind_sort]
            #print(np.max(np.absolute(dist - np.array(A[1]))))
            knn_ind.append(ind)
            knn_dist.append(dist)


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

def vae(data, layer_widths=[400,20], no_cuda=False, batch_size=128, epochs=100, learning_rate=1e-3):
    """Variational Autoencoder Embedding
    ======

    Embeds a dataset via a two layer variational autoencoder (VAE) latent representation. The VAE
    is essentially the original one from [1].

    Parameters
    ----------
    data : numpy array
        (n,d) array of n datapoints in dimension d.
    layer_widths : list of int, length=2 (optional), default=[400,20]
        First element is the width of the hidden layer, while the second is the dimension
        of the latent space.
    no_cuda : bool (optional), default=False
        Turn off GPU acceleration.
    batch_size : int (optional), default=128
        Batch size for gradient descent.
    epochs : int (optional), default=100
        How many epochs (loops over whole dataset) to train over.
    learning_rate : float (optional), default=1e-3
        Learning rate for optimizer.

    Returns
    -------
    data_vae : numpy array
        Data encoded by the VAE.

    Example
    -------
    Using a VAE embedding to construct a similarity weight matrix on MNIST and running Poisson learning
    at 1 label per class: [vae_mnist.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/vae_mnist.py).
    ```py
    import graphlearning as gl

    data, labels = gl.datasets.load('mnist')
    data_vae = gl.weightmatrix.vae(data)

    W_raw = gl.weightmatrix.knn(data, 10)
    W_vae = gl.weightmatrix.knn(data_vae, 10)

    num_train_per_class = 1
    train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
    train_labels = labels[train_ind]

    pred_labels_raw = gl.ssl.poisson(W_raw).fit_predict(train_ind,train_labels)
    pred_labels_vae = gl.ssl.poisson(W_vae).fit_predict(train_ind,train_labels)

    accuracy_raw = gl.ssl.ssl_accuracy(labels,pred_labels_raw,train_ind)
    accuracy_vae = gl.ssl.ssl_accuracy(labels,pred_labels_vae,train_ind)

    print('Raw Accuracy: %.2f%%'%accuracy_raw)
    print('VAE Accuracy: %.2f%%'%accuracy_vae)
    ```

    References
    ----------
    [1] D.P. Kingma and M. Welling. [Auto-encoding variational bayes.](https://arxiv.org/abs/1312.6114) arXiv:1312.6114, 2014.
    """

    import torch
    import torch.utils.data
    from torch.utils.data import Dataset, DataLoader
    from torch import nn, optim
    from torch.nn import functional as F
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    class MyDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = data
            self.targets = targets
            self.transform = transform
            
        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            
            if self.transform:
                x = self.transform(x)
            
            return x, y
        
        def __len__(self):
            return len(self.data)

    class VAE(nn.Module):
        def __init__(self, layer_widths):
            super(VAE, self).__init__()
            
            self.lw = layer_widths
            self.fc1 = nn.Linear(self.lw[0], self.lw[1])
            self.fc21 = nn.Linear(self.lw[1], self.lw[2])
            self.fc22 = nn.Linear(self.lw[1], self.lw[2])
            self.fc3 = nn.Linear(self.lw[2], self.lw[1])
            self.fc4 = nn.Linear(self.lw[1], self.lw[0])

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std

        def decode(self, z):
            h3 = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))

        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, self.lw[0]))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, data.shape[1]), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(data_loader.dataset)))


    layer_widths = [data.shape[1]] + layer_widths
    log_interval = 10    #how many batches to wait before logging training status

    #GPU settings
    cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    #Convert to torch dataloaders
    data = data - data.min()
    data = data/data.max()
    data = torch.from_numpy(data).float()
    target = np.zeros((data.shape[0],)).astype(int)
    target = torch.from_numpy(target).long()
    dataset = MyDataset(data, target) 
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    #Put model on GPU and set up optimizer
    model = VAE(layer_widths).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Training epochs
    for epoch in range(1, epochs + 1):
        train(epoch)

    #Encode the dataset and save to npz file
    with torch.no_grad():
        mu, logvar = model.encode(data.to(device).view(-1, layer_widths[0]))
        data_vae = mu.cpu().numpy()

    return data_vae





















