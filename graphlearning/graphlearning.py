'''
GraphLearning:  This python package is devoted to efficient implementations of modern graph-based learning algorithms for both semi-supervised learning and clustering. The package implements many popular datasets (currently MNIST, FashionMNIST, cifar-10, and WEBKB) in a way that makes it simple for users to test out new algorithms and rapidly compare against existing methods.

See README.md file for usage on GitHub: https://github.com/jwcalder/GraphLearning

Author: Jeff Calder, 2020
'''
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib 
import scipy.spatial as spatial
import scipy.optimize as opt
import numpy.random as random
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import scipy.sparse.csgraph as csgraph
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
import sys, getopt, time, csv, torch, os, multiprocessing
from joblib import Parallel, delayed
import urllib.request
import importlib

clustering_algorithms = ['incres','spectral','spectralshimalik','spectralngjordanweiss']

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def standardize_dataset_name(dataset):

    if dataset.lower()=='mnist':
        dataset='MNIST'
    if dataset.lower()=='fashionmnist':
        dataset='FashionMNIST'
    if dataset.lower()=='cifar':
        dataset='cifar'

    return dataset

#Get working graphlearning.py directory
#Returns directory where the graphlearning.py script resides, if it is
#writable. Otherwise it returns the current working Python direcotry
def GL_dir(): 

    d = os.path.dirname(os.path.realpath(__file__))
    if not os.access(d,os.W_OK):
        d = os.getcwd()

    return d


def Data_dir(): return os.path.abspath(os.path.join(os.getcwd(),os.pardir, 'Data'))
def kNNData_dir(): return os.path.abspath(os.path.join(os.getcwd(),os.pardir,'kNNData'))
def LabelPermutations_dir(): return os.path.abspath(os.path.join(os.getcwd(),os.pardir,'LabelPermutations'))
def Eigen_dir(): return os.path.abspath(os.path.join(os.getcwd(),os.pardir,'EigenData'))
def Results_dir(): return os.path.join(os.getcwd(),'Results')

def load_eig(dataset,metric,k):

    #Standardize case of dataset
    dataset = standardize_dataset_name(dataset)

    #Filenames
    dataFile = dataset+"_"+metric+"_k%d"%k+"_spectrum.npz"
    dataFile_path = os.path.join(Eigen_dir(), dataFile)

    #Load eigenvector data if MBO selected
    try:
        M = np.load(dataFile_path,allow_pickle=True)
        vals = M['vals']
        vecs = M['vecs']
        vals_norm = M['vals_norm']
        vecs_norm = M['vecs_norm']
    except:
        #Load kNN data
        I,J,D = load_kNN_data(dataset,metric=metric)

        if k > I.shape[1]:
            sys.exit('kNNData only has %d'%I.shape[1]+'-nearest neighbor information. Aborting...')
        else:
            W = weight_matrix(I,J,D,k)

        #UnNormalized Laplacian
        L = graph_laplacian(W)
        vals, vecs = sparse.linalg.eigs(L,k=300,which='SM')

        #Normalized Laplacian
        L = graph_laplacian(W,norm="normalized")
        vals_norm, vecs_norm = sparse.linalg.eigs(L,k=300,which='SM')

        #Check if Data directory exists
        if not os.path.exists(Eigen_dir()):
            os.makedirs(Eigen_dir())

        #Save eigenvectors to file
        np.savez_compressed(dataFile_path,vals=vals,vecs=vecs,vals_norm=vals_norm,vecs_norm=vecs_norm)

    return vals,vecs,vals_norm,vecs_norm

def load_label_permutation(dataset,label_perm='',t='-1'):

    #Standardize case of dataset
    dataset = standardize_dataset_name(dataset)

    dataFile = dataset+label_perm+"_permutations.npz"
    dataFile_path = os.path.join(LabelPermutations_dir(), dataFile)

    #Check if Data directory exists
    if not os.path.exists(LabelPermutations_dir()):
        os.makedirs(LabelPermutations_dir())

    #Try to Load kNNdata and/or download it
    if not os.path.exists(dataFile_path):
        urlpath = 'https://github.com/jwcalder/GraphLearning/raw/master/LabelPermutations/'+dataFile
        try:
            print('Downloading '+urlpath+' to '+dataFile_path+'...')
            urllib.request.urlretrieve(urlpath, dataFile_path)
        except:
            sys.exit('Error: Cannot find '+dataFile+', and could not downoad '+urlpath+'.')
    try:
        M = np.load(dataFile_path,allow_pickle=True)
        perm = M['perm']
    except:
        sys.exit('Error: Cannot open '+dataFile+'.')

    #Restrict trials
    t = [int(e)  for e in t.split(',')]
    if t[0] > -1:
        if len(t) == 1:
            perm = perm[0:t[0]]
        else:
            perm = perm[(t[0]-1):t[1]]

    return perm

def save_labels(L,dataset,overwrite=False):

    #Standardize case of dataset
    dataset = standardize_dataset_name(dataset)

    #Dataset filename
    dataFile = dataset+"_labels.npz"

    #Full path to file
    dataFile_path = os.path.join(Data_dir(), dataFile)

    #Check if Data directory exists
    if not os.path.exists(Data_dir()):
        os.makedirs(Data_dir())
    
    #Save dataset
    if os.path.isfile(dataFile_path) and not overwrite:
        print('Labels file '+dataFile_path+' already exists. Not saving.')
    else:
        np.savez_compressed(dataFile_path,labels=L)


def save_dataset(X,dataset,metric='raw',overwrite=False):

    #Standardize case of dataset
    dataset = standardize_dataset_name(dataset)

    #Dataset filename
    dataFile = dataset+"_"+metric+".npz"

    #Full path to file
    dataFile_path = os.path.join(Data_dir(), dataFile)

    #Check if Data directory exists
    if not os.path.exists(Data_dir()):
        os.makedirs(Data_dir())

    #Save dataset
    if os.path.isfile(dataFile_path) and not overwrite:
        print('Data file '+dataFile_path+' already exists. Not saving.')
    else:
        np.savez_compressed(dataFile_path,data=X)

def load_dataset(dataset,metric='raw'):

    #Standardize case of dataset
    dataset = standardize_dataset_name(dataset)

    #Dataset filename
    dataFile = dataset+"_"+metric+".npz"

    #Full path to file
    dataFile_path = os.path.join(Data_dir(), dataFile)

    #Check if Data directory exists
    if not os.path.exists(Data_dir()):
        os.makedirs(Data_dir())

    #Try to Load data and/or download dataset
    if not os.path.exists(dataFile_path):
        urlpath = 'http://www-users.math.umn.edu/~jwcalder/Data/'+dataFile
        try:
            print('Downloading '+urlpath+' to '+dataFile_path+'...')
            urllib.request.urlretrieve(urlpath, dataFile_path)
        except:
            sys.exit('Error: Cannot find '+dataFile+', and could not downoad '+urlpath+'.')
    try:
        M = np.load(dataFile_path,allow_pickle=True)
        data = M['data']
    except:
        sys.exit('Error: Cannot open '+dataFile+'.')
    
    return data

def load_labels(dataset):

    #Standardize case of dataset
    dataset = standardize_dataset_name(dataset)

    dataFile = dataset+"_labels.npz"
    dataFile_path = os.path.join(Data_dir(), dataFile)

    #Check if Data directory exists
    if not os.path.exists(Data_dir()):
        os.makedirs(Data_dir())

    #Try to Load labels and/or download labels
    if not os.path.exists(dataFile_path):
        urlpath = 'https://github.com/jwcalder/GraphLearning/raw/master/Data/'+dataFile
        try:
            print('Downloading '+urlpath+' to '+dataFile_path+'...')
            urllib.request.urlretrieve(urlpath, dataFile_path)
        except:
            sys.exit('Error: Cannot find '+dataFile+', and could not downoad '+urlpath+'.')
    try:
        M = np.load(dataFile_path,allow_pickle=True)
        labels = M['labels']
    except:
        sys.exit('Error: Cannot open '+dataFile+'.')

    return labels

def load_kNN_data(dataset,metric='raw'):

    #Standardize case of dataset
    dataset = standardize_dataset_name(dataset)

    dataFile = dataset+"_"+metric+".npz"
    dataFile_path = os.path.join(kNNData_dir(), dataFile)

    #Check if Data directory exists
    if not os.path.exists(kNNData_dir()):
        os.makedirs(kNNData_dir())

    #Try to Load kNNdata and/or download it
    if not os.path.exists(dataFile_path):
        urlpath = 'https://github.com/jwcalder/GraphLearning/raw/master/kNNData/'+dataFile
        try:
            print('Downloading '+urlpath+' to '+dataFile_path+'...')
            urllib.request.urlretrieve(urlpath, dataFile_path)
        except:
            sys.exit('Error: Cannot find '+dataFile+', and could not downoad '+urlpath+'.')
    try:
        M = np.load(dataFile_path,allow_pickle=True)
        I = M['I']
        J = M['J']
        D = M['D']
    except:
        sys.exit('Error: Cannot open '+dataFile+'.')

    return I,J,D

#Compute sizes of each class
def label_proportions(labels):
    L = np.unique(labels)
    L = L[L>=0]    

    k = len(L)
    #n = len(labels)
    n = np.sum(labels>=0)
    beta = np.zeros((k,))
    for i in range(k):
        beta[i] = np.sum(labels==L[i])/n

    return beta

#Constructs a weight matrix for graph on mxn grid with NSEW neighbors
def grid_graph(m,n):

    X,Y = np.mgrid[:m,:n]

    return W


#Reweights the graph to use self-tuning weights
def self_tuning(W,D,alpha):
    
    if alpha != 0:
        n = D.shape[0]
        k = D.shape[1]
        d = D[:,k-1]
        d = sparse.spdiags(d**(-alpha),0,n,n)
        W = d*W*d

    return W

#Reweights the graph based on a clustering prior
def cluster_prior(W,cluster_labels):
    
    n = W.shape[0]

    I,J,V = sparse.find(W)
    K = cluster_labels[I] == cluster_labels[J]
    V[K] = V[K]*10
    V = V/np.max(V)

    W = sparse.coo_matrix((V, (I,J)),shape=(n,n)).tocsr()

    return W

#Computes scattering transform of depth 2 of I
#Bruna, Joan, and Stéphane Mallat. "Invariant scattering convolution networks." IEEE transactions on pattern analysis and machine intelligence 35.8 (2013): 1872-1886.
def scattering_transform(I,n,m,depth=2):

    from kymatio import Scattering2D

    num_pts = I.shape[0]
    K = torch.from_numpy(I.reshape((num_pts,n,m))).float().contiguous() 
    scattering = Scattering2D(J=depth, shape=(n,m))
    Z = scattering(K).numpy()
    l = Z.shape[1]*Z.shape[2]*Z.shape[3]

    return Z.reshape((num_pts,l))


#Label permutations
#labels = labels
#T = number of trials
#r = label rate in (0,1)
def create_label_permutations_rate(labels,T,R):

    perm = list()
    n = labels.shape[0]
    labelvals = np.unique(labels)
    labelvals = labelvals[labelvals>=0]    
    num_labels = len(labelvals)
    num = np.zeros((num_labels,))
    for i in range(num_labels):
        num[i] = np.sum(labels == labelvals[i])
    
    J = np.arange(n).astype(int)
    for k in range(T):
        for r in R:
            L = list()
            for i in range(num_labels):
                l = labelvals[i]
                I = labels==l
                K = J[I]
                m = round(num[i]*r/100)
                L = L + random.choice(K,size=m.astype(int),replace=False).tolist()
            L = np.array(L)
            perm.append(L)

    return perm


#Label permutations
#labels = labels
#T = number of trials
#m = vector of number of labels
def create_label_permutations(labels,T,m,multiplier=None,dataset=None,name=None,overwrite=False):
    

    #Find all unique labels >= 0
    #Negative numbers indicate unlabeled nodes
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels>=0]    

    perm = list()
    n = labels.shape[0]
    J = np.arange(n).astype(int)
    for k in range(T):
        for i in m:
            L = list()
            ind = 0
            for l in unique_labels:
                I = labels==l
                K = J[I]
                if multiplier is None:
                    L = L + random.choice(K,size=i,replace=False).tolist()
                else:
                    sze = int(np.round(i*multiplier[ind]))
                    L = L + random.choice(K,size=sze,replace=False).tolist()
                ind = ind + 1
            L = np.array(L)
            perm.append(L)
    

    #If dataset name is provided, save permutations to file
    if not dataset is None:

        perm = np.array(perm,dtype=object)

        dataset = standardize_dataset_name(dataset)

        #data file name
        dataFile = dataset
        if not name is None:
            dataFile = dataFile + name
        dataFile = dataFile + '_permutations.npz'

        #Full path to file
        dataFile_path = os.path.join(LabelPermutations_dir(), dataFile)

        #Check if Data directory exists
        if not os.path.exists(LabelPermutations_dir()):
            os.makedirs(LabelPermutations_dir())

        #Save permutations to file
        if os.path.isfile(dataFile_path) and not overwrite:
            print('Label Permutations file '+dataFile_path+' already exists. Not saving.')
        else:
            np.savez_compressed(dataFile_path,perm=perm)
   
    return perm

#Randomly choose m labels per class
def randomize_labels(L,m):

    perm = create_label_permutations(L,1,[m])

    return perm[0]

#Default function
def exp_weight(x):
    return np.exp(-x)

#Pointwise max of non-negative sparse matrices A and B
def sparse_max(A,B):

    I = (A + B) > 0
    IB = B>A
    IA = I - IB

    return A.multiply(IA) + B.multiply(IB)


#Compute degrees of weight matrix W
def degrees(W):

    return np.squeeze(np.array(np.sum(W,axis=1)))


#Multiply diagonal of matrix by degree
def diag_multiply(W,b):

    n = W.shape[0]  #Number of points

    D = sparse.spdiags(W.diagonal(),0,n,n)
    
    return W - (1-b)*D

#Compute degrees of weight matrix W
#Returns sparse matrix with degrees on diagonal
def degree_matrix(W,p=1):

    n = W.shape[0]  #Number of points

    #Construct sparse degree matrix
    d = degrees(W)
    D = sparse.spdiags(d**p,0,n,n)

    return D.tocsr()

#Construct robin boundary condition matrix
def robin_bc_matrix(X,nu,eps,gamma):

    n = X.shape[0]
    Xtree = spatial.cKDTree(X)
    _,nn_ind = Xtree.query(X + eps*nu)
    #nn_dist = np.linalg.norm(X - X[nn_ind,:],axis=1)
    nn_dist = eps*np.ones((n,))

    #Robin matrix
    A = sparse.spdiags(gamma + (1-gamma)/nn_dist,0,n,n)
    B = sparse.coo_matrix(((1-gamma)/nn_dist, (range(n),nn_ind)),shape=(n,n))
    R = (A - B).tocsr()

    return R


#Laplace matrix
#W = weight matrix
#norm = type of normalization
#   Options: none, randomwalk, normalized
def graph_laplacian(W,norm="none"):

    D = degree_matrix(W)

    if norm=="none":
        L = D - W
    elif norm=="randomwalk1":
        Dinv = degree_matrix(W,p=-1)
        L = Dinv*(D-W)
    elif norm=="randomwalk2":
        Dinv = degree_matrix(W,p=-1)
        L = (D-W)*Dinv
    elif norm=="normalized":
        Dinv2 = degree_matrix(W,p=-1/2)
        L = Dinv2*(D-W)*Dinv2
    else:
        print("Invalid option for graph Laplacian normalization. Returning unnormalized Laplacian.")
        L = D - W

    return L.tocsr()

#Graph infinity Laplacian
#W = sparse weight matrix
#u = function on graph
def graph_phi_laplacian(W,u,phi,I=None,J=None,V=None):

    n = W.shape[0]
    if I is None or J is None:
        I,J,V = sparse.find(W)

    w = u[J]-u[I]
    a = np.absolute(w)
    pa = phi(a)
    m = pa/(a+1e-13)
    M = sparse.coo_matrix((V*pa/(a+1e-13), (I,J)),shape=(n,n)).tocsr()
    m = degrees(M)

    M = sparse.coo_matrix((V*pa*np.sign(w), (I,J)),shape=(n,n)).tocsr()
    M = np.squeeze(np.array(np.sum(M,axis=1)))

    return M, m


#Graph infinity Laplacian
#W = sparse weight matrix
#u = function on graph
def graph_infinity_laplacian(W,u,I=None,J=None,V=None):

    n = W.shape[0]
    if I is None or J is None:
        I,J,V = sparse.find(W)
    M = sparse.coo_matrix((V*(u[J]-u[I]), (I,J)),shape=(n,n)).tocsr()
    M = M.min(axis=1) + M.max(axis=1)

    return M.toarray().flatten()


#Construct epsilon-graph sparse distance matrix
def eps_weight_matrix(X,eps,f=exp_weight):

    n = X.shape[0]  #Number of points

    #Rangesearch to find nearest neighbors
    Xtree = spatial.cKDTree(X)
    M = Xtree.query_pairs(eps)
    M = np.array(list(M))

    #Differences between points and neighbors
    V = X[M[:,0],:] - X[M[:,1],:]
    D = np.sum(V*V,axis=1)

    #Weights
    D = f(4*D/(eps*eps))

    #Symmetrize weights and add diagonal entries
    D = np.concatenate((D,D,f(0)*np.ones(n,)))
    M1 = np.concatenate((M[:,0],M[:,1],np.arange(0,n)))
    M2 = np.concatenate((M[:,1],M[:,0],np.arange(0,n)))

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (M1,M2)),shape=(n,n))

    return W.tocsr()

#Exact knnsearch
def knnsearch(X,k,dataset=None,metric='raw'):
    #KDtree to find nearest neighbors
    n = X.shape[0]
    Xtree = spatial.cKDTree(X)
    D, J = Xtree.query(X,k=k)
    I = np.ones((n,k),dtype=int)*J[:,0][:,None]

    #If dataset name is provided, save permutations to file
    if not dataset is None:
        #data file name
        dataFile = dataset + '_' + metric + '.npz'

        #Full path to file
        dataFile_path = os.path.join(kNNData_dir(), dataFile)

        #Check if Data directory exists
        if not os.path.exists(kNNData_dir()):
            os.makedirs(kNNData_dir())

        np.savez_compressed(dataFile_path,I=I,J=J,D=D)

    return I,J,D

#Perform approximate nearest neighbor search, returning indices I,J of neighbors, and distance D
# Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot".
def knnsearch_annoy(X,k, similarity='euclidean', dataset=None, metric='raw'):

    from annoy import AnnoyIndex

    n = X.shape[0]  #Number of points
    dim = X.shape[1]#Dimension

    print('kNN search with Annoy approximate nearest neighbor package...')
    printProgressBar(0, n, prefix = 'Progress:', suffix = 'Complete', length = 50)

    u = AnnoyIndex(dim, similarity)  # Length of item vector that will be indexed
    for i in range(n):
        u.add_item(i, X[i,:])

    u.build(10)  #10 trees
    
    D = []
    I = []
    J = []
    for i in range(n):
        printProgressBar(i+1, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
        A = u.get_nns_by_item(i, k,include_distances=True,search_k=-1)
        I.append([i]*k)
        J.append(A[0])
        D.append(A[1])

    I = np.array(I)
    J = np.array(J)
    D = np.array(D)

    #If dataset name is provided, save permutations to file
    if not dataset is None:
        #data file name
        dataFile = dataset + '_' + metric + '.npz'

        #Full path to file
        dataFile_path = os.path.join(kNNData_dir(), dataFile)

        #Check if Data directory exists
        if not os.path.exists(kNNData_dir()):
            os.makedirs(kNNData_dir())

        np.savez_compressed(dataFile_path,I=I,J=J,D=D)

    return I,J,D

#Compute weight matrix from nearest neighbor indices I,J and distances D
def weight_matrix_selftuning(I,J,D):

    n = I.shape[0]
    k = I.shape[1]

    #Distance to kth nearest neighbor as a matrix
    sigma = D[:,k-1]
    sigma = sparse.spdiags(1/sigma,0,n,n)
    sigma = sigma.tocsr()

    #Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    #Symmetrize and remove redundant entries
    M1 = np.vstack((I,J,D))
    M2 = np.vstack((J,I,D))
    M = np.concatenate((M1,M2),axis=1)
    M = np.unique(M,axis=1)

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    I = M[0,:]
    J = M[1,:]
    D = M[2,:]
    dist = sparse.coo_matrix((D,(I,J)),shape=(n,n)).tocsr()
    B = sparse.coo_matrix((np.ones(len(D),),(I,J)),shape=(n,n)).tocsr() #Ones in all entries

    #Self-tuning weights
    E = -4*sigma*(dist**2)*sigma
    W = E.expm1()
    W = W.multiply(B) + B

    return W

#Compute weight matrix from nearest neighbor indices I,J and distances D
#k = number of neighbors
#Chooses k neighbors at random from I.shape[1] nearset neighbors
def weight_matrix_homogenized(I,J,D,k,f=exp_weight):

    #I = I[:,:10]
    #J = J[:,:10]
    #D = D[:,:10]

    #Restrict I,J,D to k neighbors
    k = np.minimum(I.shape[1],k)
    n = I.shape[0]
    for i in range(n):
        ind = random.choice(I.shape[1],k,replace=False)
        I[i,:k] = I[i,ind]
        J[i,:k] = J[i,ind]
        D[i,:k] = 1

    n = I.shape[0]
    k = I.shape[1]

    D = D*D
    eps = D[:,k-1]/4
    D = f(D/eps[:,None])

    #Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (I,J)),shape=(n,n)).tocsr()

    return W

#Compute distance matrix from nearest neighbor indices I,J and distances D
#k = number of neighbors
def dist_matrix(I,J,D,k):

    #Restrict I,J,D to k neighbors
    k = np.minimum(I.shape[1],k)
    I = I[:,:k]
    J = J[:,:k]
    D = D[:,:k]

    n = I.shape[0]
    k = I.shape[1]

    #Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (I,J)),shape=(n,n)).tocsr()

    return W

#Adds weights to an adjacency matrix W using similarity in data X
def add_weights(W,X,labels):

    n = W.shape[0]

    #pca = PCA(n_components=20)
    #X = pca.fit_transform(X)
    #print(X.shape)

    I,J,V = sparse.find(W)
    
    #Dot products
    Y = X[I,:]-X[J,:]
    Y = np.sum(Y*Y,axis=1)

    W = sparse.coo_matrix((Y, (I,J)),shape=(n,n)).tocsr()
    max_dist = np.reshape(np.max(W,axis=1).todense().tolist(),(n,))
    D = sparse.spdiags((max_dist+1e-10)**(-1),0,n,n).tocsr()
    W = D*W

    I,J,V = sparse.find(W)
    V = np.exp(-2*V)
    W = sparse.coo_matrix((V, (I,J)),shape=(n,n)).tocsr()

    return W

#Finds largest connected component of the graph represented by adjacency matrix W
#Returns the weighted adjacency matrix, along with a boolean mask indicating the 
#vertices from the input matrix that were selected
def largest_conn_component(W):

    ncomp,labels = csgraph.connected_components(W,directed=False) 
    num_verts = np.zeros((ncomp,))
    for i in range(ncomp):
        num_verts[i] = np.sum(labels==i)
    
    i_max = np.argmax(num_verts)
    ind = labels==i_max

    A = W[ind,:]
    A = A[:,ind]

    print("Found %d"%ncomp+" connected components.")
    print("Returning component with %d"%num_verts[i_max]+" vertices out of %d"%W.shape[0]+" total vertices.")

    return A,ind

#Compute weight matrix from nearest neighbor indices I,J and distances D
#k = number of neighbors
def weight_matrix(I,J,D,k,f=exp_weight,symmetrize=True):

    #Restrict I,J,D to k neighbors
    k = np.minimum(I.shape[1],k)
    I = I[:,:k]
    J = J[:,:k]
    D = D[:,:k]

    n = I.shape[0]
    k = I.shape[1]

    D = D*D
    eps = D[:,k-1]/4
    D = f(D/eps[:,None])

    #Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (I,J)),shape=(n,n)).tocsr()

    if symmetrize:
        W = (W + W.transpose())/2;

    return W

#Compute boundary points
#k = number of neighbors to use
def boundary_points_new(X,k,I=None,J=None,D=None,ReturnNormals=False):

    if (I is None) or (J is None) or (D is None):
        n = X.shape[0]
        d = X.shape[1]
        if d <= 5:
            I,J,D = knnsearch(X,k)
        else:
            I,J,D = knnsearch_annoy(X,k)
    
    #Restrict I,J,D to k neighbors
    k = np.minimum(I.shape[1],k)
    n = X.shape[0]
    I = I[:,:k]
    J = J[:,:k]
    D = D[:,:k]

    W = weight_matrix(I,J,D,k,f=lambda x : np.ones_like(x),symmetrize=False)
    L = graph_laplacian(W)
    
    #Estimates of normal vectors
    nu = -L*X
    nu = np.transpose(nu)
    norms = np.sqrt(np.sum(nu*nu,axis=0))
    nu = nu/norms
    nu = np.transpose(nu)

    print(nu.shape)

    #Boundary test
    NN = X[J]
    NN = np.swapaxes(NN[:,1:,:],0,1) #This is kxnxd
    V = NN - X #This is x^i-x^0 kxnxd array
    NN_nu = nu[J]
    W = (np.swapaxes(NN_nu[:,1:,:],0,1) + nu)/2
    xd = np.sum(V*W,axis=2) #dist to boundary
    Y = np.max(-xd,axis=0)
    
    if ReturnNormals:
        return Y,nu
    else:
        return Y


#Compute boundary points
#k = number of neighbors to use
def boundary_points(X,k,I=None,J=None,D=None,ReturnNormals=False,R=np.inf):

    if (I is None) or (J is None) or (D is None):
        n = X.shape[0]
        d = X.shape[1]
        if d <= 5:
            I,J,D = knnsearch(X,k)
        else:
            I,J,D = knnsearch_annoy(X,k)
    
    #Restrict I,J,D to k neighbors
    k = np.minimum(I.shape[1],k)
    n = X.shape[0]
    I = I[:,:k]
    J = J[:,:k]
    D = D[:,:k]

    W = weight_matrix(I,J,D,k,f=lambda x : np.ones_like(x),symmetrize=False)
    L = graph_laplacian(W)
    
    #Estimates of normal vectors
    nu = -L*X
    nu = np.transpose(nu)
    norms = np.sqrt(np.sum(nu*nu,axis=0))
    nu = nu/norms
    nu = np.transpose(nu)

    #Boundary test
    NN = X[J]
    NN = np.swapaxes(NN[:,1:,:],0,1) #This is kxnxd
    V = NN - X #This is x^i-x^0 kxnxd array
    xd = np.sum(V*nu,axis=2) #xd coordinate (kxn)
    sqdist = np.sum(V*V,axis=2)
    Y = np.max((xd*xd - sqdist)/(2*R) - xd,axis=0)
    
    if ReturnNormals:
        return Y,nu
    else:
        return Y


#Construct k-nn sparse distance matrix
#Note: Matrix is not symmetric
def knn_weight_matrix(X,k,f=exp_weight):

    I,J,D = knnsearch_annoy(X,k)
    W = weight_matrix(I,J,D,k,f=f)
   
    return W

#Solves Lx=f subject to Rx=g at ind points
def gmres_bc_solve(L,f,R,g,ind):

    #Mix matrices based on boundary points
    A = L.copy()
    A = A.tolil()
    A[ind,:] = R[ind,:]
    A = A.tocsr()

    #Right hand side
    b = f.copy()
    b[ind] = g[ind]

    #Preconditioner
    m = A.shape[0]
    M = A.diagonal()
    M = sparse.spdiags(1/M,0,m,m).tocsr()

    #GMRES solver
    #start_time = time.time()
    u,info = sparse.linalg.gmres(A,b,M=M)
    #print("--- %s seconds ---" % (time.time() - start_time))

    #print('gmres_err = %f'%np.max(np.absolute(A*u-b)))

    return u

def conjgrad(A, b, x=None, T=1e5, tol=1e-10):
#A = nxn numpy array matrix or scipy sparse matrix
#b = nxm RHS
#x = nxm initial condition
#T = maximum number of iterations
#Outputs solution to Ax=b
#Using conjugate gradient method

    if x is None:
        x = np.zeros_like(b)

    r = b - A@x
    p = r
    rsold = np.sum(r**2,axis=0)
  
    err = 1 
    i = 0
    while (err > tol) and (i < T):
        i += 1
        Ap = A@p
        alpha = rsold / np.sum(p*Ap,axis=0)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.sum(r**2,axis=0)
        err = np.sqrt(np.sum(rsnew)) 
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x,i

#Poisson solve
#Solves Lu = f with preconditioned conjugate gradient
def pcg_solve(L,f,x0=None,tol=1e-10):

    #start_time = time.time()
    L = L.tocsr()

    #Conjugate gradient with Jacobi preconditioner
    m = L.shape[0]
    M = L.diagonal()
    M = sparse.spdiags(1/M,0,m,m).tocsr()
    if x0 is None:
        u,i = splinalg.cg(L,f,tol=tol,M=M)
    else:
        u,i = splinalg.cg(L,f,x0=x0,tol=tol,M=M)
    #print("--- %s seconds ---" % (time.time() - start_time))

    return u

#Finds k Dirichlet eigenvectors
#Solves Lu = lambda u subject to u(I)=0
def dirichlet_eigenvectors(L,I,k):

    L = L.tocsr()
    n = L.shape[0]

    #Locations of labels
    idx = np.full((n,), True, dtype=bool)
    idx[I] = False

    #Left hand side matrix
    A = L[idx,:]
    A = A[:,idx]
    
    #Eigenvector solver
    vals, vec = sparse.linalg.eigs(A,k=k,which='SM')
    vec = vec.real
    vals = vals.real
    
    #Add labels back into array
    u = np.zeros((n,k))
    u[idx,:] = vec

    if k == 1:
        u = u.flatten()

    return u,vals

#Constrained linear solve
#Solves Lu = f subject to u(I)=g
def constrained_solve(L,I,g,f=None,x0=None,tol=1e-10):

    L = L.tocsr()
    n = L.shape[0]

    #Locations of labels
    idx = np.full((n,), True, dtype=bool)
    idx[I] = False

    #Right hand side
    b = -L[:,I]*g
    b = b[idx]

    if f is not None:
        b = b + f[idx]

    #Left hand side matrix
    A = L[idx,:]
    A = A[:,idx]
    
    #start_time = time.time()

    #Conjugate gradient with Jacobi preconditioner
    m = A.shape[0]
    M = A.diagonal()
    M = sparse.spdiags(1/(M+1e-10),0,m,m).tocsr()

    if x0 is None:
        v,i = splinalg.cg(A,b,tol=tol,M=M)
    else:
        v,i = splinalg.cg(A,b,x0=x0[idx],tol=tol,M=M)
    #print("--- %s seconds ---" % (time.time() - start_time))

    #Add labels back into array
    u = np.ones((n,))
    u[idx] = v
    u[I] = g

    return u

#Returns n random points in R^d
def rand(n,d):
    return random.rand(n,d)

#Returns n random points in annulus (r1,r2)
def rand_annulus(n,d,r1,r2):

    N = 0
    X = np.zeros((1,d))
    while X.shape[0] <= n:

        Y = r2*(2*rand(n,d) - 1)
        dist2 = np.sum(Y*Y,axis=1) 
        I = (dist2 < r2*r2)&(dist2 > r1*r1)
        Y = Y[I,:]
        X = np.vstack((X,Y))


    X = X[1:(n+1)]
    return X


#Returns n random points in unit ball in R^d
def rand_ball(n,d):

    N = 0
    X = np.zeros((1,d))
    while X.shape[0] <= n:

        Y = 2*rand(n,d) - 1
        I = np.sum(Y*Y,axis=1) < 1
        Y = Y[I,:]
        X = np.vstack((X,Y))


    X = X[1:(n+1)]
    return X


def randn(n,d):
    X = np.zeros((n,d))
    for i in range(d):
        X[:,i] = np.random.normal(0,1,n) 

    return X

def bean_data(n,h):

    #n = number of points
    #h = height of bridge (h=0.2)

    a=-1
    b=1
    x = a + (b-a)*random.rand(3*n);
    c=-0.6
    d=0.6;
    y = c + (d-c)*random.rand(3*n);

    X=np.transpose(np.vstack((x,y)))

    dist_from_x_axis=0.4*np.sqrt(1-x**2)*(1+h-np.cos(3*x))
    in_bean = abs(y) <= dist_from_x_axis
    X = X[in_bean,:]
    if X.shape[0] < n:
        print('Not enough samples');
    else:
        X = X[:n,:]

    return X

    
def mesh(X):
    T = spatial.Delaunay(X[:,:2]);
    return T.simplices

def box_mesh(X,u=None):

    n = X.shape[0]
    d = X.shape[1]
    if d > 2:
        X = X[:,0:2]

    x1 = X[:,0].min()
    x2 = X[:,0].max()
    y1 = X[:,1].min()
    y2 = X[:,1].max()
    corners = np.array([[x1,y1],[x2,y2],[x1,y2],[x2,y1]])
    X = np.append(X,corners,axis=0)

    Tri = mesh(X)
    
    if u is not None:
        u = np.append(u,[0,0,0,0])
        for i in range(n,n+4):
            I = (Tri[:,0] == i) | (Tri[:,1] == i) | (Tri[:,2] == i)
            nn_tri = Tri[I,:].flatten()
            nn_tri = np.unique(nn_tri[nn_tri < n])
            u[i] = np.mean(u[nn_tri])
            #u[i] = np.max(u[nn_tri])

        return X,Tri,u
    else:
        return X,Tri

#Triangulation of domain
def improved_mesh(X):

    n = X.shape[0]
    d = X.shape[1]
    if d > 2:
        X = X[:,0:2]

    #Normalize data to unit box
    x1 = X[:,0].min()
    x2 = X[:,0].max()
    y1 = X[:,1].min()
    y2 = X[:,1].max()
    X = X - [x1,y1]
    X[:,0] = X[:,0]/(x2-x1)
    X[:,1] = X[:,1]/(y2-y1)

    #Add padding data around
    pad = 10/np.sqrt(n)
    m = int(pad*n)
    Y = rand(m,2)
    Y[:,0] = Y[:,0]*pad - pad
    Z = np.vstack((X,Y))
    Y = rand(m,2)
    Y[:,0] = Y[:,0]*pad + 1
    Z = np.vstack((Z,Y))
    Y = rand(m,2)
    Y[:,1] = Y[:,1]*pad - pad
    Z = np.vstack((Z,Y))
    Y = rand(m,2)
    Y[:,1] = Y[:,1]*pad + 1
    Z = np.vstack((Z,Y))

    #Delaunay triangulation
    T = spatial.Delaunay(Z);
    Tri = T.simplices
    J = np.sum(Tri >= n,axis=1)==0;
    Tri = Tri[J,:]

    return Tri

def plot(X,u):
    Tri = improved_mesh(X)

    import mayavi.mlab as mlab
    mlab.triangular_mesh(X[:,0],X[:,1],u,Tri)
    mlab.view(azimuth=-45,elevation=60)

#Reweights the weight matrix for WNLL
def wnll(W,I):

    n = W.shape[0]
    m = len(I)

    a = np.ones((n,))
    a[I] = n/m
    
    D = sparse.spdiags(a,0,n,n).tocsr()
    W = D*W + W*D

    return W

#Weighted nonlocal Laplacian
#Shi, Zuoqiang, Stanley Osher, and Wei Zhu. "Weighted nonlocal laplacian on interpolation from sparse data." Journal of Scientific Computing 73.2-3 (2017): 1164-1177.
def wnll_learning(W,I,g):
    return laplace_learning(wnll(W,I),I,g)

#Properly weighted Laplacian
#Calder, Jeff, and Dejan Slepcev. "Properly-weighted graph Laplacian for semi-supervised learning." arXiv preprint arXiv:1810.04351 (2018).
def properlyweighted_laplace_learning(W,I,g,X,alpha,zeta,r):

    n = W.shape[0]
    rzeta = r/(zeta-1)**(1/alpha)

    Xtree = spatial.cKDTree(X[I,:])
    D, J = Xtree.query(X)
    D[D < rzeta] = rzeta
    gamma = 1 + (r/D)**alpha

    D = sparse.spdiags(gamma,0,n,n).tocsr()

    return laplace_learning(D*W + W*D,I,g)

#Game theoretic p-Laplace learning
#Rios, Mauricio Flores, Jeff Calder, and Gilad Lerman. "Algorithms for $\ell_p$-based semi-supervised learning on graphs." arXiv preprint arXiv:1901.05031 (2019).
def plaplace_learning(W,I,g,p,sol_method="GradientDescentCcode",norm="none"):

    n = W.shape[0]
    k = len(np.unique(g))  #Number of labels
    u = np.zeros((k,n))
    i = 0
    for l in np.unique(g):
        u[i,:] = plaplace_solve(W,I,g==l,p,sol_method=sol_method,norm=norm)
        i+=1

    return u

#Game theoretic p-Laplace learning
#Rios, Mauricio Flores, Jeff Calder, and Gilad Lerman. "Algorithms for $\ell_p$-based semi-supervised learning on graphs." arXiv preprint arXiv:1901.05031 (2019).
def plaplace_solve(W,I,g,p,sol_method="GradientDescentCcode",norm="none"):

    #start_time = time.time()

    n = W.shape[0]
    W = W/W.max()
    
    if p == float("inf"):
        alpha = 0
        delta = 1
    else:
        alpha = 1/p
        delta = 1-2/p
    
    dx = degrees(W)
    theta = 1.2*(2*alpha + np.max(dx)*delta)

    if p == float("inf"):
        beta = 1
        gamma = 1/theta
    else:
        beta = (theta*p - 2)/(theta*p)
        gamma = (p-2)/(theta*p-2)

    if norm=="normalized":
        deg = dx[I]**(1/2) 
        g = g/deg

    L = graph_laplacian(W)
    u = constrained_solve(L,I,g)
    uu = np.max(g)*np.ones((n,))
    ul = np.min(g)*np.ones((n,))

    WI,WJ,WV = sparse.find(W)

    #Set labels
    u[I] = g
    uu[I] = g
    ul[I] = g

    #Time step for gradient descent
    dt = 0.9/(alpha + 2*delta)

    if sol_method=="GradientDescentCcode":
        try:
            #Import c extensions
            import graphlearning.cextensions as cext
        except:
            sys.exit("C extensions not found.")

        #Type casting and memory blocking
        uu = np.ascontiguousarray(uu,dtype=np.float64)
        ul = np.ascontiguousarray(ul,dtype=np.float64)
        WI = np.ascontiguousarray(WI,dtype=np.int32)
        WJ = np.ascontiguousarray(WJ,dtype=np.int32)
        WV = np.ascontiguousarray(WV,dtype=np.float64)
        I = np.ascontiguousarray(I,dtype=np.int32)
        g = np.ascontiguousarray(g,dtype=np.float64)

        cext.lp_iterate(uu,ul,WI,WJ,WV,I,g,p,1e6,1e-1,0.0)
        u = (uu+ul)/2

        #Check residual
        L2uu = -L*uu
        LIuu = graph_infinity_laplacian(W,uu,I=WI,J=WJ,V=WV)
        resu = alpha*L2uu/dx + delta*LIuu
        resu[I]=0

        L2ul = -L*ul
        LIul = graph_infinity_laplacian(W,ul,I=WI,J=WJ,V=WV)
        resl = alpha*L2ul/dx + delta*LIul
        resl[I]=0

        #print('Upper residual = %f' % np.max(np.absolute(resu)))
        #print('Lower residual = %f' % np.max(np.absolute(resl)))

    else:
        err = 1e6
        i = 0
        while err > 1e-1:

            i+=1
            
            #Graph laplacians
            L2u = -L*u
            LIu = graph_infinity_laplacian(W,u,I=WI,J=WJ,V=WV)

            #Residual error
            res = alpha*L2u/dx + delta*LIu
            res[I]=0
            #err = np.max(np.absolute(res))
            #print("Residual error = "+str(err))

            #Update
            if sol_method=="GradientDescent":
                L2uu = -L*uu
                LIuu = graph_infinity_laplacian(W,uu,I=WI,J=WJ,V=WV)
                res = alpha*L2uu/dx + delta*LIuu
                res[I]=0
                uu = uu + dt*res        
                err = np.max(np.absolute(res))
                #print("Upper residual = "+str(err))

                L2ul = -L*ul
                LIul = graph_infinity_laplacian(W,ul,I=WI,J=WJ,V=WV)
                res = alpha*L2ul/dx + delta*LIul
                res[I]=0
                ul = ul + dt*res        
                err = np.max(np.absolute(res))
                #print("Lower residual = "+str(err))
                err1 = np.max(uu-ul)
                err2 = np.min(uu-ul)

                #print("Residual error = "+str(err1)+","+str(err2))
                err = err1

                u = (uu + ul)/2
            elif sol_method=="SemiImplicit":
                rhs = beta*(2*gamma*dx*LIu - L2u)
                u = constrained_solve(L,I,g,f=rhs,x0=u,tol=err/100)
            else:
                print("Invalid p-Laplace solution method.")
                sys.exit()
            
    if norm=="normalized":
        deg = dx**(1/2) 
        u = u*deg

    #print("--- %s seconds ---" % (time.time() - start_time))
    return u

#Gradient of function on graph
#W = sparse weight matrix
#u = function on graph
def graph_gradient(W,u,I=None,J=None,V=None):

    n = W.shape[0]
    if I is None or J is None:
        I,J,V = sparse.find(W)

    G = sparse.coo_matrix((V*(u[J]-u[I]), (I,J)),shape=(n,n)).tocsr()

    return G

#Divergence of F, need not be skew symmetric
#F = sparse matrix representing function edges of the graph
def graph_divergence(F,W):
    
    F = F.multiply(W)
    return 2*np.squeeze(np.array(np.sum(F,axis=1)))


#Divergence of vector field F (F should be skew-symmetric)
#F = sparse matrix representing vector field
def div(F,W):
    
    F = F - F.transpose()
    F = F.multiply(W)
    return np.squeeze(np.array(np.sum(F,axis=1)))/2

#Gradient of function on graph (unweighted)
#W = sparse weight matrix
#u = function on graph
def grad(W,u,I=None,J=None,V=None):

    n = W.shape[0]
    if I is None or J is None:
        I,J,V = sparse.find(W)

    G = sparse.coo_matrix((u[J]-u[I], (I,J)),shape=(n,n)).tocsr()

    return G

#Random-walk SSL 
#Zhou, Dengyong, et al. "Learning with local and global consistency." Advances in neural information processing systems. 2004.
def randomwalk_learning(W,I,g,epsilon):
    
    n = W.shape[0]

    #Zero diagonals
    W = W - sparse.spdiags(W.diagonal(),0,n,n)

    #Construct Laplacian matrix
    Dinv2 = degree_matrix(W,p=-1/2)
    L = sparse.identity(n) - (1-epsilon)*Dinv2*W*Dinv2;

    #Preconditioner
    m = L.shape[0]
    M = L.diagonal()
    M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()
   
    #Conjugate gradient solver
    u,T = conjgrad(M*L*M, M*onehot_labels(I,g,n).T, tol=1e-6)
    u = M*u

    return u.T


#Computes accuracy of labeling
#m = number of labeled points used
def accuracy(L,L_true,m):   
    #Remove unlabeled nodes
    I = L_true >=0
    L = L[I]
    L_true = L_true[I]

    #Compute accuracy
    return 100*np.maximum(np.sum(L==L_true)-m,0)/(len(L)-m)

#Projects all columns of (kxn) matrix X onto k-simplex
def ProjectToSimplex(X):
   
    n = X.shape[1]
    k = X.shape[0]

    Xs = -np.sort(-X,axis=0)  #Sort descending
    A = np.tril(np.ones((k,k)))
    Sum = A@Xs
    Max = np.transpose((np.transpose(Sum) - 1)/(np.arange(k)+1))
    Xs[:-1,:] = Xs[1:,:]
    Xs[-1,:] = (Sum[k-1,:]-1)/k
    I = np.argmax(Max >= Xs,axis=0)
    X = np.maximum(X-Max[I,range(n)],0)
    return X

#Takes list of labels and converts to vertices of simplex format
def LabelsToVec(L):

    n = L.shape[0]

    labels = np.unique(L)
    k = len(labels)
    for i in range(k):
        L[L==labels[i]] = i

    L = L.astype(int)
    X = np.zeros((k,n))
    X[L,range(n)] = 1

    return X,labels

#Projects all rows of (nxk) matrix X to closest vertex of the simplex
#Assume X already lives in the simplex, e.g., is the output of ProjectToSimplex
def ClosestVertex(X):

    n = X.shape[1]
    k = X.shape[0]
    L = np.argmax(X,axis=0)
    X = np.zeros((k,n))
    X[L,range(n)] = 1
    return X

#Threshold with temperature to closest vertex
def ClosestVertexTemp(X,T=0.01):

    n = X.shape[1]
    k = X.shape[0]
    
    beta = 1/T
    Y = np.exp(beta*X)
    Ysum = np.sum(Y,axis=0)
    Y = Y/Ysum

    X[0,:] = Y[0,:]
    for i in range(1,k):
        X[i,:] = X[i-1,:] + Y[i,:]

    R = random.rand(n,1)
    L = np.sum(R.flatten() > X,axis=0)

    X = np.zeros((k,n))
    X[L,range(n)] = 1
    return X

#Volume MBO, initialized with Poisson
def poisson_volumeMBO(W,I,g,dataset,beta,T,volume_mult):

    #Set diagonal entries to zero
    W = diag_multiply(W,0)

    try:
        #Import c extensions
        import graphlearning.cextensions as cext
    except:
        sys.exit("C extensions not found.")

    #Solve Poisson problem and compute labels
    u,_ = poisson(W,I,g)
    max_locations = np.argmax(u,axis=0)
    u = (np.unique(g))[max_locations]

    n = W.shape[0]
    k = len(np.unique(g))
    WI,WJ,WV = sparse.find(W)

    #Class counts
    ClassCounts = (n*beta).astype(int)

    #Type casting and memory blocking
    u = np.ascontiguousarray(u,dtype=np.int32)
    WI = np.ascontiguousarray(WI,dtype=np.int32)
    WJ = np.ascontiguousarray(WJ,dtype=np.int32)
    WV = np.ascontiguousarray(WV,dtype=np.float32)
    I = np.ascontiguousarray(I,dtype=np.int32)
    g = np.ascontiguousarray(g,dtype=np.int32)
    ClassCounts = np.ascontiguousarray(ClassCounts,dtype=np.int32)

    cext.volume_mbo(u,WI,WJ,WV,I,g,ClassCounts,k,0.0,T,volume_mult)

    #Set given labels and convert to vector format
    u[I] = g
    u,_ = LabelsToVec(u)
    return u



#Volume MBO (Jacobs, et al.)
def volumeMBO(W,I,g,dataset,beta,T,volume_mult):

    #Set diagonal entries to zero
    W = diag_multiply(W,0)

    try:
        #Import c extensions
        import graphlearning.cextensions as cext
    except:
        sys.exit("C extensions not found.")

    n = W.shape[0]
    k = len(np.unique(g))
    u = np.zeros((n,))
    WI,WJ,WV = sparse.find(W)

    #Class counts
    ClassCounts = (n*beta).astype(int)

    #Type casting and memory blocking
    u = np.ascontiguousarray(u,dtype=np.int32)
    WI = np.ascontiguousarray(WI,dtype=np.int32)
    WJ = np.ascontiguousarray(WJ,dtype=np.int32)
    WV = np.ascontiguousarray(WV,dtype=np.float32)
    I = np.ascontiguousarray(I,dtype=np.int32)
    g = np.ascontiguousarray(g,dtype=np.int32)
    ClassCounts = np.ascontiguousarray(ClassCounts,dtype=np.int32)

    cext.volume_mbo(u,WI,WJ,WV,I,g,ClassCounts,k,1.0,T,volume_mult)

    #Set given labels and convert to vector format
    u[I] = g
    u,_ = LabelsToVec(u)
    return u


#Multiclass MBO
#Garcia-Cardona, Cristina, et al. "Multiclass data segmentation using diffuse interface methods on graphs." IEEE transactions on pattern analysis and machine intelligence 36.8 (2014): 1600-1613.
def multiclassMBO(W,I,g,eigvals,eigvecs,dataset,true_labels=None):

    n = W.shape[0]
    k = len(np.unique(g))

    Ns = 6
    if dataset=='MNIST' or dataset=='FashionMNIST' or dataset=='cifar':
        dt = 0.15
        mu = 50
    elif dataset=='WEBKB':
        dt = 1
        mu = 4
    else:
        dt = 0.1
        mu = 1

    #Load eigenvalues and eigenvectors
    X = eigvecs
    num_eig = len(eigvals)
    
    #Form matrices
    V = np.diag(1/(1 + (dt/Ns)*eigvals)) 
    Y = X@V
    Xt = np.transpose(X)

    #Random initial labeling
    u = random.rand(k,n)
    u = ProjectToSimplex(u)

    #Set initial known labels
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = Kg*J
    u = Kg + (1-J)*u
    
    #Maximum number of iterations
    T = 10
    for i in range(T):
        for s in range(Ns):
            Z = (u - (dt/Ns)*mu*J*(u - Kg))@Y
            u = Z@Xt
            
        #Projection step
        u = ProjectToSimplex(u)
        u = ClosestVertex(u)

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=0)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)

    return u


#Poisson MBO
def poissonMBO(W,I,g,dataset,beta,true_labels=None,temp=0,use_cuda=False,Ns=40,mu=1,T=50):

    n = W.shape[0]
    unique_labels = np.unique(g)
    k = len(unique_labels)

    num_labels = np.zeros((k,))
    for i in range(k):
        num_labels[i] = np.sum(g==unique_labels[i])

    W = diag_multiply(W,0)
    if dataset=='WEBKB':
        mu = 1000
        Ns = 8
    
    #Labels to vector and correct position
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = Kg*J
    
    #Poisson source term
    c = np.sum(Kg,axis=1)/len(I)
    b = np.transpose(Kg)
    b[I,:] = b[I,:]-c
    b = np.transpose(b)

    L = graph_laplacian(W,norm='none')

    #Initialize u via Poisson learning
    #u = np.zeros((k,n))
    #for j in range(k):
    #    u[j,:] = pcg_solve(L,b[j,:])
    #u = mu*u
    #u = np.transpose(np.transpose(u) - np.mean(u,axis=1))
    u,mix_time = poisson(W,I,g,use_cuda=use_cuda,beta=beta)
    #Ns = int(mix_time/4)
    u = ProjectToSimplex(u)
    u = ClosestVertex(u)

    #Time step for stability
    dt = 1/np.max(degrees(W))

    P = sparse.identity(n) - dt*L
    Db = mu*dt*b

    if use_cuda:
        Pt = torch_sparse(P).cuda()
        Dbt = torch.from_numpy(np.transpose(Db)).float().cuda()

    for i in range(T):

        if use_cuda:

            #Put on GPU and run heat equation
            ut = torch.from_numpy(np.transpose(u)).float().cuda()
            for s in range(Ns):
                #u = u*P + Db
                ut = torch.sparse.addmm(Dbt,Pt,ut)

            #Put back on CPU
            u = np.transpose(ut.cpu().numpy())
         
        else: #Use CPU 
            for s in range(Ns):
                #u = u + dt*(mu*b - u*L)
                u = u*P + Db

        #Projection step
        #u = np.diag(beta/num_labels)@u
        u = ProjectToSimplex(u)
        u = ClosestVertex(u)
        u = np.transpose(np.transpose(u) - np.mean(u,axis=1) + beta)

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=0)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)
    
    return u

def torch_sparse(A):

    A = A.tocoo()
    values = A.data
    indices = np.vstack((A.row, A.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

#Sparse Label Propagation
def SparseLabelPropagation(W,I,g,true_labels=None):

    n = W.shape[0]
    k = len(np.unique(g))

    WI,WJ,WV = sparse.find(W)
    B = sparse.coo_matrix((np.ones(len(WV),),(WI,WJ)),shape=(n,n)).tocsr() #Ones in all entries

    #Construct matrix 1/2W and 1/deg
    lam = 2*W - (1-1e-10)*B
    lam = -lam.log1p()
    lam = lam.expm1() + B
    Id = sparse.identity(n) 
    gamma = degree_matrix(W+1e-10*Id,p=-1)

    #Random initial labeling
    #u = random.rand(k,n)
    u = np.zeros((k,n))

    #Set initial known labels
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = Kg*J

    #Initialization
    Y = list()
    for j in range(k):
        Gu = graph_gradient(W,u[j,:],I=WI,J=WJ,V=WV)
        Y.append(Gu)

    #Main loop for sparse label propagation
    T = 100
    for i in range(T):

        u_prev = np.copy(u)
        #Compute div
        for j in range(k):
            div = graph_divergence(Y[j],W)
            u[j,:] = u_prev[j,:] - gamma*div
            u[j,I] = Kg[j,I]  #Set labels
            u_tilde = 2*u[j,:] - u_prev[j,:]

            Gu = -graph_gradient(W,u_tilde,I=WI,J=WJ,V=WV)
            Y[j] = Y[j] + Gu.multiply(lam)
            ind1 = B.multiply(abs(Y[j])>1)
            ind2 = B - ind1
            Y[j] = ind1.multiply(Y[j].sign()) + ind2.multiply(Y[j])

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=0)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)

    return u


#Dynamic Label Propagation
def DynamicLabelPropagation(W,I,g,alpha=0.05,lam=0.1,true_labels=None):

    n = W.shape[0]
    k = len(np.unique(g))

    W = diag_multiply(W,0)
    
    #Labels to vector and correct position
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    u,_ = LabelsToVec(K)
    u = u*J

    #Set initial known labels
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = np.transpose(Kg*J)
    u = np.copy(Kg)
  
    if n > 5000:
        print("Cannot use Dynamic Label Propagation on large datasets.")
    else:
        #Setup matrices
        Id = sparse.identity(n) 
        D = degree_matrix(W,p=-1)
        P = D*W
        P = np.array(P.todense())
        Pt = np.copy(P)

        T = 2
        for i in range(T):
            v = P@u
            u = Pt@u
            u[I,:] = Kg[I,:]
            Pt = P@Pt@np.transpose(P) + alpha*v@np.transpose(v) + lam*Id

            #Compute accuracy if all labels are provided
            if true_labels is not None:
                u = np.array(u)
                max_locations = np.argmax(u,axis=1)
                labels = (np.unique(g))[max_locations]
                labels[I] = g
                acc = accuracy(labels,true_labels,len(I))
                print('i:%d'%i+',Accuracy = %.2f'%acc)
        

        u = np.transpose(np.array(u))

    return u

#Centered and Iterated Centered Kernel of Mai/Coulliet 2018 
def CenteredKernel(W,I,g,true_labels=None):

    n = W.shape[0]
    k = len(np.unique(g))

    W = diag_multiply(W,0)

    #Labels to vector and correct position
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = np.transpose(Kg*J)
    
    #Center labels
    c = np.sum(Kg,axis=0)/len(I)
    Kg[I,:] = Kg[I,:]-c

    u = np.copy(Kg)
    v = np.ones((n,1))
    vt = np.ones((1,n))

    e = np.random.rand(n,1)
    for i in range(100):
        y = W*(e -  (1/n)*v@(vt@e))
        w = y - (1/n)*v@(vt@y) #=Ae
        l = abs(np.transpose(e)@w/(np.transpose(e)@e))
        e = w/np.linalg.norm(w)

    #Number of iterations
    #alpha = 5*l/4
    alpha = 105*l/100
    T = 1000
    err = 1
    while err > 1e-10:
        y = W*(u -  (1/n)*v@(vt@u))
        w = (1/alpha)*(y - (1/n)*v@(vt@y)) - u #Laplacian
        w[I,:] = 0
        err = np.max(np.absolute(w))
        u = u + w

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=1)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)
    
    return np.transpose(u)


def vec_acc(u,I,g,true_labels):

    max_locations = np.argmax(u,axis=0)
    labels = (np.unique(g))[max_locations]
    labels[I] = g
    acc = accuracy(labels,true_labels,len(I))

    return acc
#def volume_label_projection(u,beta,s=None):
#
#    k = u.shape[0]
#    n = u.shape[1]
#    if s is None:
#        s = np.ones((k,))
#    for i in range(100):
#        grad = beta - np.sum(ClosestVertex(np.diag(s)@u),axis=1)/n
#        err0 = np.max(np.absolute(grad))
#
#        dt = 1
#        snew = s + dt*grad
#        gradnew = beta - np.sum(ClosestVertex(np.diag(snew)@u),axis=1)/n
#        err = err0
#        newerr = np.max(np.absolute(gradnew))
#        while newerr < err:
#            print(dt)
#            dt = 2*dt
#            snew = s + dt*grad
#            gradnew = beta - np.sum(ClosestVertex(np.diag(snew)@u),axis=1)/n
#            err = newerr
#            newerr = np.max(np.absolute(gradnew))
#        dt = dt/2
#        snew = s + dt*grad
#        gradnew = beta - np.sum(ClosestVertex(np.diag(snew)@u),axis=1)/n
#        newerr = np.max(np.absolute(gradnew))
#        while newerr >= err:
#            print(dt)
#            dt = dt/2
#            snew = s + dt*grad
#            gradnew = beta - np.sum(ClosestVertex(np.diag(snew)@u),axis=1)/n
#            newerr = np.max(np.absolute(gradnew))
#        if dt < 1:
#            dt = dt/2
#       
#        s = s + dt*grad 
#
#        print(err)
#        if err == 0:
#            print(i)
#            break
#
#        #s = s + dt*(beta - beta_u)
#    
#    return ClosestVertex(np.diag(s)@u),s

def volume_label_projection(u,beta,s=None,dt=None):

    k = u.shape[0]
    n = u.shape[1]
    if s is None:
        s = np.ones((k,))
    if dt is None:
        dt = 10
    #print(np.around(100*beta,decimals=1))
    #print(np.around(100*np.sum(ClosestVertex(np.diag(s)@u),axis=1)/n,decimals=1))
    for i in range(100):
        class_size = np.sum(ClosestVertex(np.diag(s)@u),axis=1)/n
        grad = beta - class_size 
        #print(np.around(100*class_size,decimals=1))
        #err = np.max(np.absolute(grad))

        #if err == 0:
        #    break
        s = np.clip(s + dt*grad,0.5,2)
    
    #print(np.around(100*beta,decimals=1))
    #print(np.around(100*np.sum(ClosestVertex(np.diag(s)@u),axis=1)/n,decimals=1))
    #print(np.around(100*beta - 100*np.sum(ClosestVertex(np.diag(s)@u),axis=1)/n,decimals=4))
    return ClosestVertex(np.diag(s)@u),s

#Poisson MBO with volume constraints
def poissonMBO_volume(W,I,g,dataset,beta,true_labels=None,temp=0,use_cuda=False,Ns=40,mu=1,T=20):

    n = W.shape[0]
    k = len(np.unique(g))

    W = diag_multiply(W,0)
    if dataset=='WEBKB':
        mu = 1000
        Ns = 8
    
    #Labels to vector and correct position
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    Kg = Kg*J
    
    #Poisson source term
    c = np.sum(Kg,axis=1)/len(I)
    b = np.transpose(Kg)
    b[I,:] = b[I,:]-c
    b = np.transpose(b)

    D = degree_matrix(W)
    #L = graph_laplacian(W,norm='none')
    L = D - W.transpose()

    #Initialize u via Poisson learning
    u,_ = poisson(W,I,g,true_labels=true_labels,use_cuda=use_cuda, beta=beta)
    u = mu*u

    #Time step for stability
    dt = 1/np.max(degrees(W))

    P = sparse.identity(n) - dt*L
    Db = mu*dt*b

    if use_cuda:
        Pt = torch_sparse(P).cuda()
        Dbt = torch.from_numpy(np.transpose(Db)).float().cuda()

    for i in range(T):

        #Heat equation step
        if use_cuda:

            #Put on GPU and run heat equation
            ut = torch.from_numpy(np.transpose(u)).float().cuda()
            for j in range(Ns):
                ut = torch.sparse.addmm(Dbt,Pt,ut)

            #Put back on CPU
            u = np.transpose(ut.cpu().numpy())
         
        else: #Use CPU 
            for j in range(Ns):
                u = u*P + Db

        #Projection step
        u,s = volume_label_projection(u,beta)

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=0)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)
    
    return u


#Poisson Volume
def PoissonVolume(W,I,g,true_labels=None,use_cuda=False,training_balance=True,beta=None,min_iter=50):


    #Run Poisson learning
    u,_ = poisson(W,I,g,true_labels=true_labels,use_cuda=use_cuda, training_balance=training_balance,beta = beta)

    #Volume constraints
    _,s = volume_label_projection(u,beta)
    return np.diag(s)@u

def onehot_labels(I,g,n):
#Converts labels to one-hot vectos and places
#them in the correct positions in output array.

    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = LabelsToVec(K)
    return Kg*J

#Laplace learning
#Zhu, Xiaojin, Zoubin Ghahramani, and John D. Lafferty. "Semi-supervised learning using gaussian fields and harmonic functions." Proceedings of the 20th International conference on Machine learning (ICML-03). 2003.
def laplace_learning(W,I,g,norm='none'):

    n = W.shape[0]
    unique_labels = np.unique(g)
    k = len(unique_labels)
    
    #Graph Laplacian and one-hot labels
    L = graph_laplacian(W,norm=norm)
    F,_ = LabelsToVec(g)
    F = F.T

    #Locations of unlabeled points
    idx = np.full((n,), True, dtype=bool)
    idx[I] = False

    #Right hand side
    b = -L[:,I]*F
    b = b[idx,:]

    #Left hand side matrix
    A = L[idx,:]
    A = A[:,idx]

    #Preconditioner
    m = A.shape[0]
    M = A.diagonal()
    M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()
   
    #Conjugate gradient solver
    v,T = conjgrad(M*A*M, M*b, tol=1e-5)
    v = M*v

    #Add labels back into array
    u = np.ones((n,k))
    u[idx,:] = v
    u[I,:] = F

    return u.T

#Shift trick
#W = Weight matrix
#I = indices of labels
#g = +1/-1 values of labels
def mean_shifted_laplace_learning(W,I,g,norm='none'):

    #Laplace learning
    u = laplace_learning(W,I,g,norm=norm)

    #Mean shift
    c = np.mean(u,axis=1)
    u = u -  c[:,np.newaxis]

    return u

#Poisson learning, alternative version
def poisson2(W,I,g,true_labels=None,min_iter=50,solver='conjgrad'):

    n = W.shape[0]
    unique_labels = np.unique(g)
    k = len(unique_labels)
    
    #Zero out diagonal for faster convergence
    W = diag_multiply(W,0)

    #Poisson source term
    Kg = onehot_labels(I,g,n)
    b = Kg.T - np.mean(Kg,axis=1)

    #Check solver method (conjgrad or gradientdescent)
    if solver.lower() == "conjgrad":

        L = graph_laplacian(W,norm='none')

        #Preconditioner
        m = L.shape[0]
        M = L.diagonal()
        M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()
       
        #Conjugate gradient solver
        u,T = conjgrad(M*L*M, M*b, tol=1e-6)
        u = M*u

        #Centering
        d = degrees(W)
        c = np.sum(d[:,np.newaxis]*u,axis=0)/np.sum(d)
        u = u - c

    else:
        #Setup matrices
        D = degree_matrix(W,p=-1)
        P = D*W.transpose()
        Db = D*b

        v = np.max(Kg,axis=0)
        v = v/np.sum(v)
        vinf = degrees(W)/np.sum(degrees(W))
        RW = W.transpose()*D
        u = np.zeros((n,k))


        #start_time = time.time()
        #L = graph_laplacian(W,norm='none')
        #D2 = degree_matrix(W,p=-0.5)
        #cu,i = conjgrad(D2*L*D2, D2*b, tol=1e-5)
        #cu = D2*cu
        #print("ConjGrad: %d iter: %s seconds ---" % (i,time.time() - start_time))

        #Number of iterations
        T = 0
        #start_time = time.time()
        while (T < min_iter or np.max(np.absolute(v-vinf)) > 1/n) and (T < 1000):
            u = Db + P*u
            v = RW*v
            T = T + 1

            #Compute accuracy if all labels are provided
            if true_labels is not None:
                max_locations = np.argmax(u,axis=1)
                labels = (np.unique(g))[max_locations]
                labels[I] = g
                acc = accuracy(labels,true_labels,len(I))
                print('%d,Accuracy = %.2f'%(T,acc))
        
        #print("Grad Desc: %d iter: %s seconds ---" % (T,time.time() - start_time))

    return u.T,T



#Poisson learning
def poisson(W,I,g,true_labels=None,use_cuda=False,training_balance=True,beta=None,min_iter=50):

    n = W.shape[0]
    unique_labels = np.unique(g)
    k = len(unique_labels)
    
    #Zero out diagonal for faster convergence
    W = diag_multiply(W,0)

    #Poisson source term
    Kg = onehot_labels(I,g,n)
    c = np.sum(Kg,axis=1)/len(I)
    b = np.transpose(Kg)
    b[I,:] = b[I,:]-c

    #Setup matrices
    D = degree_matrix(W + 1e-10*sparse.identity(n),p=-1)
    #L = graph_laplacian(W,norm='none')
    #P = sparse.identity(n) - D*L #Line below is equivalent when W symmetric
    P = D*W.transpose()
    Db = D*b

    v = np.max(Kg,axis=0)
    v = v/np.sum(v)
    vinf = degrees(W)/np.sum(degrees(W))
    RW = W.transpose()*D
    u = np.zeros((n,k))

    #vals, vec = sparse.linalg.eigs(RW,k=1,which='LM')
    #vinf = np.absolute(vec.flatten())
    #vinf = vinf/np.sum(vinf)

    #Number of iterations
    T = 0
    if use_cuda:
        
        Pt = torch_sparse(P).cuda()
        ut = torch.from_numpy(u).float().cuda()
        Dbt = torch.from_numpy(Db).float().cuda()

        #start_time = time.time()
        while (T < min_iter or np.max(np.absolute(v-vinf)) > 1/n) and (T < 1000):
            ut = torch.sparse.addmm(Dbt,Pt,ut)
            v = RW*v
            T = T + 1
        #print("--- %s seconds ---" % (time.time() - start_time))

        #Transfer to CPU and convert to numpy
        u = ut.cpu().numpy()

    else: #Use CPU

        #start_time = time.time()
        while (T < min_iter or np.max(np.absolute(v-vinf)) > 1/n) and (T < 1000):
            u = Db + P*u
            v = RW*v
            T = T + 1

            #Compute accuracy if all labels are provided
            if true_labels is not None:
                max_locations = np.argmax(u,axis=1)
                labels = (np.unique(g))[max_locations]
                labels[I] = g
                acc = accuracy(labels,true_labels,len(I))
                print('%d,Accuracy = %.2f'%(T,acc))
        
        #print("--- %s seconds ---" % (time.time() - start_time))

    #Balancing for training data/class size discrepancy
    if training_balance:
        if beta is None:
            u = u@np.diag(1/c)
        else:
            u = u@np.diag(beta/c)

    return np.transpose(u),T



#Poisson L1 based on Split Bregman Method
#Does not work as well as PoissonMBO
def poissonL1(W,I,g,dataset,norm="none",lam=100,mu=1000,Nouter=30,Ninner=6,true_labels=None):

    n = W.shape[0]
    k = len(np.unique(g))

    #mu = mu*W.count_nonzero()/len(g)  #Normalize constants
    gamma = 1/lam

    WI,WJ,WV = sparse.find(W)
    B = sparse.coo_matrix((np.ones(len(WV),),(WI,WJ)),shape=(n,n)).tocsr() #Ones in all entries
    L = graph_laplacian(2*W.multiply(W),norm=norm)
    deg = degrees(W)
    dt = 1/np.max(deg)

    #Random initial labeling
    #u = random.rand(k,n)
    #u = ProjectToSimplex(u)
    u = np.zeros((k,n))

    #Set initial known labels
    Kg = onehot_labels(I,g,n)

    #Poisson parameters
    c = np.sum(Kg,axis=1)/len(I)
    b = np.transpose(Kg)
    b[I,:] = b[I,:]-c
    b = (mu/lam)*np.transpose(b)

    #Initialize u via Poisson learning
    u = np.zeros((k,n))
    L = graph_laplacian(W,norm='none')
    for j in range(k):
        u[j,:] = pcg_solve(L,b[j,:])
    u = np.transpose(np.transpose(u) - np.mean(u,axis=1))
   
    #Initialization
    V = list()
    R = list()
    gradu = list()
    for j in range(k):
        Gu = graph_gradient(W,u[j,:],I=WI,J=WJ,V=WV)
        gradu.append(Gu)
        V.append(Gu)
        R.append(Gu)

    #Main loop for Split Bregman iteration
    for i in range(Nouter):
        print('Outer:%d'%i)
        for s in range(Ninner):
            normV = 0*W
            for j in range(k):
                divVR = graph_divergence(R[j] - V[j],W)
                u[j,:] = pcg_solve(L,b[j,:] + divVR,x0=u[j,:],tol=1e-10)
                #for s in range(100):
                #    u[j,:] = u[j,:] + dt*(b[j,:] + divVR - u[j,:]*L)
                gradu[j] = graph_gradient(W,u[j,:],I=WI,J=WJ,V=WV)
                V[j] = gradu[j] + R[j]
                normV = normV + V[j].multiply(V[j]) 

            normV = normV.sqrt()

            #Shrinkage operation
            #normV^{-1} for nonzero entries (tricky to do in sparse format)
            #normV.eliminate_zeros(X)
            normVinv = normV - (1-1e-10)*B
            normVinv = -normVinv.log1p()
            normVinv = normVinv.expm1() + B
            
            C = normV.multiply(normVinv)
            #print(np.sum(C>0))
            #print(np.sum(C>0.9999))

            #Compute shrinkage factor
            #print(np.sum(normV>0))
            shrink = normV - gamma*B
            shrink = shrink.maximum(0)
            #print(np.sum(shrink>0))
            shrink = shrink.multiply(normVinv)
            
            #Apply shrinkage
            for j in range(k):
                V[j] = V[j].multiply(shrink)

        for j in range(k):
            R[j] = R[j] + gradu[j] - V[j]
         

        #Compute accuracy if all labels are provided
        if true_labels is not None:
            max_locations = np.argmax(u,axis=0)
            labels = (np.unique(g))[max_locations]
            labels[I] = g
            acc = accuracy(labels,true_labels,len(I))
            print('Accuracy = %.2f'%acc)

    return u

#Heap functions
#d = values in heap (indexed by graph vertex)
#h = heap (contains indices of graph elements in heap)
#p = pointers from graph back to heap (are updated with heap operations)
#s = number of elements in heap

#Sift up
#i = heap index of element to be sifted up
def SiftUp(d,h,s,p,i):

    pi = int(i/2)  #Parent index in heap
    while pi != 0:
        if d[h[pi]] > d[h[i]]:  #If parent larger, then swap
            #Swap in heap
            tmp = h[pi]
            h[pi] = h[i]
            h[i] = tmp

            #Update pointers to heap
            p[h[i]] = i     
            p[h[pi]] = pi

            #Update parent/child indices
            i = pi
            pi = int(i/2)
        else:
            pi = 0
            
            
#Sift down
#i = heap index of element to be sifted down
def SiftDown(d,h,s,p,i):

    ci = 2*i  #child index in heap
    while ci <= s:
        if d[h[ci+1]] < d[h[ci]] and ci+1 <= s:  #Choose smallest child
            ci = ci+1

        if d[h[ci]] < d[h[i]]:  #If child smaller, then swap
            #Swap in heap
            tmp = h[ci]
            h[ci] = h[i]
            h[i] = tmp

            #Update pointers to heap
            p[h[i]] = i     
            p[h[ci]] = ci

            #Update parent/child indices
            i = ci
            ci = 2*i
        else:
            ci = s+1

#Pop smallest off of heap
#Returns index of smallest and size of new heap
def PopHeap(d,h,s,p):
    
    #Index of smallest in heap
    i = h[1]

    #Put last element on top of heap
    h[1] = h[s]

    #Update pointer
    p[h[1]] = 1

    #Sift down the heap
    SiftDown(d,h,s-1,p,1)

    return i,s-1
     
#Push element onto heap
#i = Graph index to add to heap
def PushHeap(d,h,s,p,i):

    h[s+1] = i  #add to heap at end
    p[i] = s+1  #Update pointer to heap
    SiftUp(d,h,s+1,p,s+1)

    return s+1

def stencil_solver(ui,u,w=None):

    if w is None:
        w = np.ones((len(u),))

    m = len(u)

    #Sort neighbors
    I = np.argsort(u)
    u = u[I]
    w = w[I]

    f = np.zeros((m+1,))
    for i in range(m):
        f[i] = np.sum(np.maximum(u[i]-u,0)**2)

    f[m] = np.maximum(1,f[m-1])
    k = np.argmin(f < 1)

    b = np.sum(u[:k])
    c = np.sum(u[:k]**2)
    t = (b + np.sqrt(b*b - k*c + k))/k

    check = np.sum(np.maximum(t - u,0)**2)

    if(abs(check - 1) > 1e-5):
        print("Error")

    return t
    #return np.min(u) + 1

#C code version of dijkstra
def cDijkstra(W,I,g,WI=None,WJ=None,K=None):

    n = W.shape[0]
    k = len(I)
    u = np.ones((n,))*1e10          #HJ Solver
    l = -np.ones((n,),dtype=int)    #Index of closest label

    if (WI == None) or (WJ == None) or (K==None):
        #Reformat weight matrix W into form more useful for Dijkstra
        WI,WJ,WV = sparse.find(W)
        K = np.array((WJ[1:] - WJ[:-1]).nonzero()) + 1
        K = np.append(0,np.append(K,len(WJ)))

    try:
        #Import c extensions
        import graphlearning.cextensions as cext

        #Type casting and memory blocking
        u = np.ascontiguousarray(u,dtype=np.float64)
        l = np.ascontiguousarray(l,dtype=np.int32)
        WI = np.ascontiguousarray(WI,dtype=np.int32)
        WV = np.ascontiguousarray(WV,dtype=np.float64)
        K = np.ascontiguousarray(K,dtype=np.int32)
        I = np.ascontiguousarray(I,dtype=np.int32)
        g = np.ascontiguousarray(g,dtype=np.float64)

        cext.dijkstra(u,l,WI,K,WV,I,g,1.0)
    except:
        sys.exit("C extensions not found.")

    return u

#Solve a general HJ equation with fast marching
def HJsolver(W,I,g,WI=None,WJ=None,K=None,p=1):

    n = W.shape[0]
    k = len(I)
    u = np.ones((n,))*1e10          #HJ Solver
    l = -np.ones((n,),dtype=int)    #Index of closest label

    if (WI == None) or (WJ == None) or (K==None):
        #Reformat weight matrix W into form more useful for Dijkstra
        WI,WJ,WV = sparse.find(W)
        K = np.array((WJ[1:] - WJ[:-1]).nonzero()) + 1
        K = np.append(0,np.append(K,len(WJ)))

    try:  #Try to use fast C version, if compiled

        #Import c extensions
        import graphlearning.cextensions as cext

        #Type casting and memory blocking
        u = np.ascontiguousarray(u,dtype=np.float64)
        l = np.ascontiguousarray(l,dtype=np.int32)
        WI = np.ascontiguousarray(WI,dtype=np.int32)
        WV = np.ascontiguousarray(WV,dtype=np.float64)
        K = np.ascontiguousarray(K,dtype=np.int32)
        I = np.ascontiguousarray(I,dtype=np.int32)
        g = np.ascontiguousarray(g,dtype=np.int32)

        cext.HJsolver(u,l,WI,K,WV,I,g,1.0,p,1.0)

    except:

        #Initialization
        s = 0                           #Size of heap
        h = -np.ones((n+1,),dtype=int)  #Active points heap (indices of active points)
        A = np.zeros((n,),dtype=bool)   #Active flag
        p = -np.ones((n,),dtype=int)    #Pointer back to heap
        V = np.zeros((n,),dtype=bool)   #Finalized flag
        l = -np.ones((n,),dtype=int)    #Index of closest label

        #Build active points heap and set distance = 0 for initial points
        for i in range(k):
            s = PushHeap(u,h,s,p,I[i])
            u[I[i]] = g[i]      #Initialize distance to zero
            A[I[i]] = True   #Set active flag to true
            l[I[i]] = I[i]   #Set index of closest label
        
        #Dijkstra's algorithm 
        while s > 0:
            i,s = PopHeap(u,h,s,p) #Pop smallest element off of heap

            #Finalize this point
            V[i] = True  #Mark as finalized
            A[i] = False #Set active flag to false

            #Update neighbors (the code below is wrong: compare against C sometime)
            for j in WI[K[i]:K[i+1]]:
                if j != i and V[j] == False:
                    nn_ind = WI[K[j]:K[j+1]]
                    w_vals = WV[K[j]:K[j+1]]
                    u_vals = u[nn_ind]
                    u_tmp = stencil_solver(u[j],u_vals,w=w_vals)
                    if A[j]:  #If j is already active
                        if u_tmp < u[j]: #Need to update heap
                            u[j] = u_tmp
                            SiftUp(u,h,s,p,p[j])
                            l[j] = l[i]
                    else: #If j is not active
                        #Add to heap and initialize distance, active flag, and label index
                        s = PushHeap(u,h,s,p,j)
                        u[j] = u_tmp
                        A[j] = True  
                        l[j] = l[i]

    return u

#eikonal classifier
def eikonalSSL(W,I,g,p=2,beta=None):

    
    k = len(I) #Number of labels
    n = W.shape[0] #Number of datapoints
    d = np.zeros((n,))        #Distance function
    l = -np.ones((n,),dtype=int)    #Index of closest label

    #Reformat weight matrix W into form more useful for Dijkstra
    WI,WJ,WV = sparse.find(W)
    K = np.array((WJ[1:] - WJ[:-1]).nonzero()) + 1
    K = np.append(0,np.append(K,len(WJ)))

    c_code = False
    try:  #Try to use fast C version, if compiled

        #Import c extensions
        import graphlearning.cextensions as cext

        #Type casting and memory blocking
        d = np.ascontiguousarray(d,dtype=np.float64)
        l = np.ascontiguousarray(l,dtype=np.int32)
        WI = np.ascontiguousarray(WI,dtype=np.int32)
        WV = np.ascontiguousarray(WV,dtype=np.float64)
        K = np.ascontiguousarray(K,dtype=np.int32)
        I = np.ascontiguousarray(I,dtype=np.int32)

        c_code = True
    except:
        c_code = False


    labels = np.unique(g)
    numl = len(labels)

    u = np.zeros((numl,n))
    for i in range(numl):
        ind = I[g == labels[i]]
        lab = np.zeros((len(ind),))

        if c_code:
            ind = np.ascontiguousarray(ind,dtype=np.int32)
            lab = np.ascontiguousarray(lab,dtype=np.int32)
            cext.HJsolver(d,l,WI,K,WV,ind,lab,1.0,p,0.0)
            u[i,:] = -d
        else:
            u[i,:] = -HJsolver(W,ind,lab,WI=WI,WV=WV,K=K,p=p)
        

    if beta is not None:
        _,s = volume_label_projection(u,beta,dt=-0.5)
        u = np.diag(s)@u
    return u


#Nearest neighbor classifier (graph geodesic distance)
def nearestneighbor(W,I,g):

    
    k = len(I) #Number of labels
    n = W.shape[0] #Number of datapoints
    d = np.ones((n,))*1e10        #Distance function
    l = -np.ones((n,),dtype=int)    #Index of closest label

    #Reformat weight matrix W into form more useful for Dijkstra
    WI,WJ,WV = sparse.find(W)
    K = np.array((WJ[1:] - WJ[:-1]).nonzero()) + 1
    K = np.append(0,np.append(K,len(WJ)))

    try:  #Try to use fast C version of dijkstra

        #Import c extensions
        import graphlearning.cextensions as cext

        #Type casting and memory blocking
        d = np.ascontiguousarray(d,dtype=np.float64)
        l = np.ascontiguousarray(l,dtype=np.int32)
        WI = np.ascontiguousarray(WI,dtype=np.int32)
        WV = np.ascontiguousarray(WV,dtype=np.float64)
        K = np.ascontiguousarray(K,dtype=np.int32)
        I = np.ascontiguousarray(I,dtype=np.int32)
        init = np.ascontiguousarray(np.zeros_like(I),dtype=np.float64)

        cext.dijkstra(d,l,WI,K,WV,I,init,1.0)
        
    except: #Use python version, which is slower

        print('Could not find C extensions, defaulting to Python code.')
        #Initialization
        s = 0                           #Size of heap
        h = -np.ones((n+1,),dtype=int)  #Active points heap (indices of active points)
        A = np.zeros((n,),dtype=bool)   #Active flag
        p = -np.ones((n,),dtype=int)    #Pointer back to heap
        V = np.zeros((n,),dtype=bool)   #Finalized flag

        
        #Build active points heap and set distance = 0 for initial points
        for i in range(k):
            d[I[i]] = 0      #Initialize distance to zero
            A[I[i]] = True   #Set active flag to true
            l[I[i]] = I[i]   #Set index of closest label
            s = PushHeap(d,h,s,p,I[i])
        
        #Dijkstra's algorithm 
        while s > 0:
            i,s = PopHeap(d,h,s,p) #Pop smallest element off of heap

            #Finalize this point
            V[i] = True  #Mark as finalized
            A[i] = False #Set active flag to false

            #Update neighbors
            #for j in WI[K[i]:K[i+1]]:
            for jj in range(K[i],K[i+1]):
                j = WI[jj]
                if j != i and V[j] == False:
                    if A[j]:  #If j is already active
                        tmp_dist = d[i] + WV[jj]
                        if tmp_dist < d[j]: #Need to update heap
                            d[j] = tmp_dist
                            SiftUp(d,h,s,p,p[j])
                            l[j] = l[i]
                    else: #If j is not active
                        #Add to heap and initialize distance, active flag, and label index
                        d[j] = d[i] + WV[jj]
                        A[j] = True  
                        l[j] = l[i]
                        s = PushHeap(d,h,s,p,j)

    #Set labels based on nearest neighbor
    u = np.zeros((n,))
    u[I] = g
    u,_ = LabelsToVec(u[l])

    return u


#Computes accuracy of clustering
def clustering_accuracy(L,L_true):

    unique_classes = np.unique(L_true)
    num_classes = len(unique_classes)

    C = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            C[i][j] = np.sum((L == i) & (L_true != j))
    row_ind, col_ind = opt.linear_sum_assignment(C)

    return 100*(1-C[row_ind,col_ind].sum()/len(L))

#Spectral embedding
#Projects the graph to R^k via spectral projection 
#Method can be 'unnormalized', 'ShiMalik', or 'NgJordanWeiss'
def spectral_embedding(W,k,method='NgJordanWeiss'):

    n = W.shape[0]

    if method == 'unnormalized':
        L = graph_laplacian(W,norm='none') 
        vals, vec = sparse.linalg.eigs(L,k=k,which='SM')
        vec = vec.real
        vals = vals.real
    elif method == 'ShiMalik':
        D = degree_matrix(W)
        L = graph_laplacian(W,norm='none') 
        vals, vec = sparse.linalg.eigs(L,M=D,k=k,which='SM')
        vec = vec.real
        vals = vals.real
    elif method == 'NgJordanWeiss':
        L = graph_laplacian(W,norm='normalized') 
        vals, vec = sparse.linalg.eigs(L,k=k,which='SM')
        vec = vec.real
        vals = vals.real
        norms = np.sum(vec*vec,axis=1)
        T = sparse.spdiags(norms**(-1/2),0,n,n)
        vec = T@vec  #Normalize rows

    return vec

def kmeans(X,k):
    KM = cluster.KMeans(n_clusters=k).fit(X)
    return KM.labels_

#Spectral Clustering
def spectral_cluster(W,k,method='NgJordanWeiss',extra_dim=0):

    V = spectral_embedding(W,k+extra_dim,method=method)
    kmeans = cluster.KMeans(n_clusters=k).fit(V)
    #V = spectral_embedding(W,k,method=method)
    #kmeans = cluster.KMeans(n_clusters=k).fit(V)
    return kmeans.labels_


#INCRES clustering
#Bresson, Xavier, et al. "An incremental reseeding strategy for clustering." International Conference on Imaging, Vision and Learning based on Optimization and PDEs. Springer, Cham, 2016.
#W = weight matrix 
def incres_cluster(W,k,speed,T,labels):

    n = W.shape[0]

    #Increment
    Dm = np.maximum(int(speed*1e-4*n/k),1)
    
    #Random initial labeling
    u = random.randint(0,k,size=n)

    #Initialization
    F = np.zeros((n,k))
    J = np.arange(n).astype(int)

    #Random walk transition
    D = degree_matrix(W,p=-1)
    P = W*D

    m = int(1)
    for i in range(T):
        #Plant
        F.fill(0)
        for r in range(k):
            I = u == r
            ind = J[I]
            F[ind[random.choice(np.sum(I),m)],r] = 1
        
        #Grow
        while np.min(F) == 0:
            F = P*F

        #Harvest
        u = np.argmax(F,axis=1)

        #Increment
        m = m + Dm
            
        #Compute accuracy
        if labels is not None: 
            acc = clustering_accuracy(u,labels)
            print("Iteration "+str(i)+": Accuracy = %.2f" % acc+"%%, #seeds= %d" % m)

    return u

#Check if graph is connected
def isconnected(W):
    num_comp,comp = csgraph.connected_components(W)
    if num_comp == 1:
        return True
    else:
        return False

#Graph-based clustering
#W = sparse weight matrix describing graph
#algorithm = SSL method
#   Options: incres
def graph_clustering(W,k,true_labels=None,algorithm="incres",speed=5,T=100,extra_dim=0):

    n = W.shape[0]
    
    #Symmetrize W, if not already symmetric
    W = (W + W.transpose())/2

    #Check if connected
    if not isconnected(W):
        print('Warning: Graph is not connected!')
    
    #Clustering
    if algorithm=="incres":
        labels = incres_cluster(W,k,speed,T,true_labels)
    elif algorithm=="spectral":
        labels = spectral_cluster(W,k,method="unnormalized",extra_dim=extra_dim)
    elif algorithm=="spectralshimalik":
        labels = spectral_cluster(W,k,method="ShiMalik",extra_dim=extra_dim)
    elif algorithm=="spectralngjordanweiss":
        labels = spectral_cluster(W,k,method="NgJordanWeiss",extra_dim=extra_dim)
    else:
        print("Invalid choice of clustering method.")
        sys.exit()

    return labels


#Graph-based semi-supervised learning
#W = sparse weight matrix describing graph
#I = indices of labeled datapoints
#g = values of labels
#algorithm = SSL method
#   Options: laplace, poisson, poisson_nodeg, wnll, properlyweighted, plaplace, randomwalk
def graph_ssl(W,I,g,D=None,Ns=40,mu=1,numT=50,beta=None,algorithm="laplace",p=3,volume_mult=0.5,alpha=2,zeta=1e7,r=0.1,epsilon=0.05,X=None,plaplace_solver="GradientDescentCcode",norm="none",true_labels=None,vals=None,vecs=None,vals_norm=None,vecs_norm=None,dataset=None,T=0,use_cuda=False,return_vector=False,poisson_training_balance=True,symmetrize=True,poisson_solver="conjgrad",params=None):

    #Convert to scipy.sparse format
    W = sparse.csr_matrix(W)

    n = W.shape[0]
    algorithm = algorithm.lower()

    if beta is None:
        beta = np.ones((len(np.unique(g)),))

    #Symmetrize D,W, if not already symmetric
    if symmetrize:
        W = (W + W.transpose())/2
        if D is not None:
            D = sparse_max(D,D.transpose())

    if not isconnected(W):
        print('Warning: Graph is not connected!')
    
    if algorithm=="mbo":
        u = multiclassMBO(W,I,g,vals_norm,vecs_norm,dataset,true_labels=true_labels)
    elif algorithm=="laplace":
        u = laplace_learning(W,I,g,norm=norm)
    elif algorithm=="randomwalk":
        u = randomwalk_learning(W,I,g,epsilon)
    elif algorithm=="wnll":
        u = wnll_learning(W,I,g)
    elif algorithm=="properlyweighted":
        if X is None:
            print("Must supply raw data points for properly weighted Laplacian.")
            sys.exit()
        else:
            u = properlyweighted_laplace_learning(W,I,g,X,alpha,zeta,r)
    elif algorithm=="mean_shifted_laplace":
        u = mean_shifted_laplace_learning(W,I,g,norm=norm)
    elif algorithm=="plaplace":
        u = plaplace_learning(W,I,g,p,sol_method=plaplace_solver,norm=norm)
    elif algorithm=="volumembo":
        u = volumeMBO(W,I,g,dataset,beta,T,volume_mult)
    elif algorithm=="poissonvolumembo":
        u = poisson_volumeMBO(W,I,g,dataset,beta,T,volume_mult)
    elif algorithm=="poissonmbo_old":
        u = poissonMBO(W,I,g,dataset,np.ones_like(beta),true_labels=true_labels,temp=T,use_cuda=use_cuda,Ns=Ns,mu=mu,T=numT)
    elif algorithm=="poissonmbobalanced":
        u = poissonMBO(W,I,g,dataset,beta,true_labels=true_labels,temp=T,use_cuda=use_cuda,Ns=Ns,mu=mu,T=numT)
    elif algorithm=="poissonl1":
        u = poissonL1(W,I,g,dataset,true_labels=true_labels)
    elif algorithm=="poisson2":
        u,_ = poisson2(W,I,g,true_labels=true_labels,solver=poisson_solver)
    elif algorithm=="poisson":
        u,_ = poisson(W,I,g,true_labels=true_labels,use_cuda=use_cuda,training_balance=poisson_training_balance)
    elif algorithm=="poissonbalanced":
        u,_ = poisson(W,I,g,true_labels=true_labels,use_cuda=use_cuda,training_balance=poisson_training_balance,beta = beta)
    elif algorithm=="poissonvolume":
        u = PoissonVolume(W,I,g,true_labels=true_labels,use_cuda=use_cuda,training_balance=poisson_training_balance,beta = beta)
    elif algorithm=="poissonmbo":
        u = poissonMBO_volume(W,I,g,dataset,beta,true_labels=true_labels,temp=T,use_cuda=use_cuda,Ns=Ns,mu=mu)
    elif algorithm=="dynamiclabelpropagation":
        u = DynamicLabelPropagation(W,I,g,true_labels=true_labels)
    elif algorithm=="sparselabelpropagation":
        u = SparseLabelPropagation(W,I,g,true_labels=true_labels)
    elif algorithm=="centeredkernel":
        u = CenteredKernel(W,I,g,true_labels=true_labels)
    elif algorithm=="nearestneighbor":
        #Use distance matrix if provided, instead of weight matrix
        if D is None:
            u = nearestneighbor(W,I,g)
        else:
            u = nearestneighbor(D,I,g)
    elif algorithm=="eikonal":
        #Use distance matrix if provided, instead of weight matrix
        if D is None:
            u = eikonalSSL(W,I,g,p=p,beta=beta)
        else:
            u = eikonalSSL(W,I,g,p=p,beta=beta)
    else:
        #Try loading algorithm
        if os.path.exists(algorithm+'.py'):
            alg_module = importlib.import_module(algorithm)
            u = alg_module.ssl(W,I,g,params)
        else:
            sys.exit("Invalid choice of SSL algorithm.")
            i = i+1

    if return_vector:
        labels = np.transpose(u)
    else:
        #Select labels
        max_locations = np.argmax(u,axis=0)
        labels = (np.unique(g))[max_locations]

        #Make sure to set labels at labeled points
        labels[I] = g

    return labels

#Read numerical data from csv file
def csvread(filename):
    
    X = [] 
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        n = 0
        for row in csv_reader:
            if not row[0]=='Date/Time':
                X += [float(i) for i in row]
                m = len(row)
                n += 1

    return np.array(X).reshape((n,m))

#Compute average and standard deviation of accuracy over many trials
#Reads data from csv file filename
#Returns accuracy (acc), standard deviation (stddev) and number of labeled points (N)
def accuracy_statistics(filename):

    X = csvread(filename)
    N = np.unique(X[:,0])

    acc = []
    stddev = []
    quant = []
    for n in N:
        Y = X[X[:,0]==n,1]
        Y = np.sort(Y)
        acc += [np.mean(Y)]
        quant += [Y[int(3*len(Y)/4)]]
        stddev += [np.std(Y)]
        #print("%.1f (%.1f)"%(np.mean(Y),np.std(Y)), end="&")

    num_trials = len(X[:,0])/len(N) 
    return acc,stddev,N,quant,num_trials

#Makes an accuracy table to be included in LaTeX documents
#dataset = name of dataset
#ssl_methods = list of names of methods to compare
def accuracy_table_icml(table_list,legend_list,num_of_classes,testerror=False,savefile='tables.tex',title='',quantile=False,append=False,directory=Results_dir(),fontsize='small',small_caps=True,two_column=True):

    #Retrieve number of different label rates m
    accfile = os.path.join(directory,table_list[0]+'_accuracy.csv')
    acc,stddev,N,quant,num_trials = accuracy_statistics(accfile)
    m = len(N)

    #Determine best algorithm at each label rate
    best = [None]*m
    best_score = [0]*m
    i=0
    for table in table_list:
        accfile = os.path.join(directory,table+"_accuracy.csv")
        acc,stddev,N,quant,num_trials = accuracy_statistics(accfile)
        if quantile:
            acc = quant
        for j in range(m):
            if acc[j] > best_score[j]:
                best_score[j] = acc[j]
                best[j] = i
        i+=1
    
    if append:
        f = open(savefile,"r")
        lines = f.readlines()
        f.close()
        f = open(savefile,"w")
        f.writelines([item for item in lines[:-1]])
    else:
        f = open(savefile,"w")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[T1]{fontenc}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage[margin=1in]{geometry}\n")
        f.write("\\begin{document}\n")

    f.write("\n\n\n")
    if two_column:
        f.write("\\begin{table*}[t!]\n")
    else:
        f.write("\\begin{table}[t!]\n")
    f.write("\\vspace{-3mm}\n")
    f.write("\\caption{"+title+": Average (standard deviation) classification accuracy over %d trials.}\n"%num_trials)
    f.write("\\vspace{-3mm}\n")
    f.write("\\label{tab:"+title+"}\n")
    f.write("\\vskip 0.15in\n")
    f.write("\\begin{center}\n")
    f.write("\\begin{"+fontsize+"}\n")
    if small_caps:
        f.write("\\begin{sc}\n")
    f.write("\\begin{tabular}{l")
    for i in range(m):
        f.write("l")
    f.write("}\n")
    f.write("\\toprule\n")
    f.write("\\# Labels per class")
    for i in range(m):
        f.write("&\\textbf{%d}"%int(N[i]/num_of_classes))
    f.write("\\\\\n")
    f.write("\\midrule\n")
    i = 0

    for table in table_list:
        f.write(legend_list[i].ljust(15))
        accfile = os.path.join(directory,table+"_accuracy.csv")
        acc,stddev,N,quant,num_trials = accuracy_statistics(accfile)
        for j in range(m):
            if best[j] == i: 
                f.write("&{\\bf %.1f"%acc[j]+" (%.1f)}"%stddev[j])
                #f.write("&${\\bf %.1f"%acc[j]+"\\pm %.1f}$"%stddev[j])
            else:
                f.write("&%.1f"%acc[j]+" (%.1f)      "%stddev[j])
                #f.write("&$%.1f"%acc[j]+"\\pm %.1f$     "%stddev[j])
        f.write("\\\\\n")
        i+=1

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    if small_caps:
        f.write("\\end{sc}\n")
    f.write("\\end{"+fontsize+"}\n")
    f.write("\\end{center}\n")
    f.write("\\vskip -0.1in\n")
    if two_column:
        f.write("\\end{table*}")
    else:
        f.write("\\end{table}")
    f.write("\n\n\n")
    f.write("\\end{document}\n")
    f.close()

def plot_graph(X,W,l=None):
#Other colormaps, coolwarm, winter, Set3, tab20b, rainbow

    #plt.ion()
    colors = np.array([[1.0,0,0],[0,0.9,0]])
    plt.rcParams['figure.facecolor'] = 'navy'

    n = W.shape[0]
    I,J,V = sparse.find(W)

    for i in range(len(I)):
        xval = [X[I[i],0],X[J[i],0]]
        yval = [X[I[i],1],X[J[i],1]]
        #plt.plot(xval,yval, color='black', linewidth=0.15, markersize=0)
        plt.plot(xval,yval, color=[0.5,0.5,0.5], linewidth=0.5, markersize=0)

    if l is None:
        #plt.scatter(X[:,0],X[:,1], s=30, cmap='Paired')
        plt.scatter(X[:,0],X[:,1], s=8, zorder=3)
    else:
        #plt.scatter(X[:,0],X[:,1], s=30, c=l, cmap='Paired')
        plt.scatter(X[:,0],X[:,1], s=8, c=colors[l,:], zorder=3)

    plt.axis("off")

#plot average and standard deviation of accuracy over many trials
#dataset = name of dataset
#ssl_methods = list of names of methods to compare
def accuracy_plot(plot_list,legend_list,num_of_classes,title=None,errorbars=False,testerror=False,savefile=None,loglog=False):

    #plt.ion()
    plt.figure()
    if errorbars:
        matplotlib.rcParams.update({'errorbar.capsize': 5})
    matplotlib.rcParams.update({'font.size': 16})
    styles = ['^b-','or-','dg-','sk-','pm-','xc-','*y-']
    i = 0
    for plot in plot_list:
        accfile = os.path.join(Results_dir(),plot+"_accuracy.csv")
        acc,stddev,N,quant,num_trials = accuracy_statistics(accfile)
        if testerror:
            acc = 100-acc
            #z = np.polyfit(np.log(N),np.log(acc),1)
            #print(z[0])
        if errorbars:
            plt.errorbar(N/num_of_classes,acc,fmt=styles[i],yerr=stddev,label=legend_list[i])
        else:
            if loglog:
                plt.loglog(N/num_of_classes,acc,styles[i],label=legend_list[i])
            else:
                plt.plot(N/num_of_classes,acc,styles[i],label=legend_list[i])
        i+=1
    plt.xlabel('Number of labels per class')
    if testerror:
        plt.ylabel('Test error (%)')
        plt.legend(loc='upper right')
    else:
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='lower right')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()


#Select labels based on a ranking
#Prodces a label permutation with 1 trial with same variations of #labels per class as the label permutation perm provided as input
def SelectLabels(labels,permold,rank):

    perm = permold

    #Number of classes
    L = np.unique(labels)
    k = len(L)
    n = len(labels)

    m = len(permold)
    num = np.zeros((m,))
    for i in range(m):
        num[i] = len(permold[i])

    
    num,unique_perm = np.unique(num,return_index=True)

    perm = list()
    for i in unique_perm:
        p = permold[i]
        pl = labels[p]
        ind = []
        for l in L:
            numl = np.sum(pl == l)
            K = labels == l
            c = np.argsort(-rank[K])
            j = np.arange(0,n)[K]
            ind = ind + j[c[:numl]].tolist()
        ind = np.array(ind)
        perm.append(ind)


    return perm

#PageRank algorithm
def PageRank(W,alpha=0.85,v=None,tol=1e-10):

    n = W.shape[0]

    u = np.ones((n,))/n
    if v is None:
        v = np.ones((n,))/n

    D = degree_matrix(W,p=-1)
    P = W.T@D

    err = tol+1
    while err > tol:
        w = alpha*P@u + (1-alpha)*v
        err = np.max(np.absolute(w-u))
        u = w.copy()

    return u

#Displays a grid of images
def image_grid(X, n_rows=10, n_cols=10, padding=2, title=None, normalize=False, fontsize=None, transpose=True):
#X = (n,m) array of n grayscale images, flattened to length m arrays
#OR X = (n_rows,n_cols,m) array, in which case n_rows and n_cols are read dirctly from X
#n_rows: number of rows in grid (optional)
#n_cols: number of columns in grid (optional)
#padding: space between images in grid (optional)
#fontsize: Font size for title (optional)
#normalize: Whether to normalize pixel intensities for viewing
#transpose: Whether to transpose images
    
    #Basic dimensions
    if X.ndim == 3:
        n_rows = X.shape[0]
        n_cols = X.shape[1]
        m = X.shape[2]
        im_width = int(np.sqrt(m))
  
        #Reshape
        X = np.reshape(X,(n_rows*n_cols,im_width,im_width))
        n = X.shape[0]
    else:
        n = X.shape[0]
        m = X.shape[1]
        im_width = int(np.sqrt(m))
  
        #Reshape
        X = np.reshape(X,(n,im_width,im_width))
  
    if normalize:
        X = X - X.min()
        X = X/X.max()
  
    #Declare memory for large image that contains the whole grid
    I = np.ones(((n_rows-1)*padding+n_rows*im_width,(n_cols-1)*padding+n_cols*im_width))
  
    #Loop over the grid, placing each image in the correct position
    c = 0
    for j in range(n_rows):
        row_pos = j*(im_width+padding)
        for i in range(n_cols):
            col_pos = i*(im_width+padding)
            if c < n:
                im = X[c,:,:]
                if transpose:
                    im = im.T
                I[row_pos:row_pos+im_width,col_pos:col_pos+im_width] = im
                c += 1
  
    #Create a new window and plot the image
    plt.figure(figsize=(10,10))
    plt.imshow(I,cmap='gray')
    plt.axis('off')
    if title is not None:
        if fontsize is not None:
            plt.title(title,fontsize=fontsize)
        else:
            plt.title(title)

#Print help
def print_help():
    
    print('========================================================')
    print('GraphLearning: Python package for graph-based learning. ')
    print('========================================================')
    print('                                                        ')
    print('Options:')
    print('   -d (--dataset=): MNIST, FashionMNIST, WEBKB, cifar (default=MNIST)')
    print('   -m (--metric=):  Metric for computing similarities (default=L2)')
    print('          Choices:  vae, scatter, L2, aet')
    print('   -a (--algorithm=): Learning algorithm (default=Laplace)')
    print('   -k (--knn=): Number of nearest neighbors (default=10)')
    print('   -t (--num_trials=): Number of trial permutations to run (default=all)')
    print('   -l (--label_perm=): Choice of label permutation file (format=dataset<label_perm>_permutations.npz). (default is empty).')
    print('   -p (--p=): Value of p for plaplace method (default=3)')
    print('   -n (--normalization=): Laplacian normalization (default=none)')
    print('                 Choices: none, normalized')
    print('   -N (--num_classes): Number of clusters if choosing clustering algorithm (default=10)')
    print('   -s (--speed=): Speed in INCRES method (1--10) (default=2)')
    print('   -i (--num_iter=): Number of iterations for iterative methods (default=1000)')
    print('   -x (--extra_dim=): Number of extra dimensions in spectral clustering (default=0)')
    print('   -c (--cuda): Use GPU acceleration (when available)')
    print('   -T (--temperature): Temperature for volume constrained MBO (default=0)')
    print('   -v (--volume_constraint=): Volume constraint for MBO (default=0.5)')
    print('   -j (--num_cores=): Number of cores to use in parallel processing (default=1)')
    print('   -r (--results): Turns off automatic saving of results to .csv file')
    print('   -b (--verbose): Turns on verbose mode (displaying more intermediate steps).')

#Default settings
def default_dataset(): return 'MNIST'
def default_metric(): return 'raw'
def default_algorithm(): return 'laplace'
def default_k(): return 10
def default_t(): return '-1'
def default_label_perm(): return ''
def default_p(): return 3
def default_norm(): return "none"
def default_use_cuda(): return False
def default_T(): return 0
def default_num_cores(): return 1
def default_results(): return True
def default_num_classes(): return 10
def default_speed(): return 2
def default_num_iter(): return 1000
def default_extra_dim(): return 0
def default_volume_constraint(): return 0.5
def default_verbose(): return False
def default_poisson_training_balance(): return True
def default_directed_graph(): return False
def default_require_eigen_data(): return False

#Main subroutine for ssl trials 
def ssl_trials(dataset = default_dataset(), metric = default_metric(), algorithm = default_algorithm(), k = default_k(), t = default_t(), label_perm = default_label_perm(), p = default_p(), norm = default_norm(), use_cuda = default_use_cuda(), T = default_T(), num_cores = default_num_cores(), results = default_results(), num_classes = default_num_classes(), speed = default_speed(), num_iter = default_num_iter(), extra_dim = default_extra_dim(), volume_constraint = default_volume_constraint(), verbose = default_verbose(), poisson_training_balance = default_poisson_training_balance(), directed_graph = default_directed_graph(),params={},require_eigen_data=default_require_eigen_data()):

    #Standardize case of dataset
    dataset = standardize_dataset_name(dataset)

    #Load labels
    labels = load_labels(dataset)

    #Load raw data for properly weighted Laplacian
    #Otherwise data is not needed, since knn data is stored
    data = None
    if algorithm.startswith('properlyweighted'):
        data = load_dataset(dataset, metric=metric)

    #Load nearest neighbor data
    I,J,D = load_kNN_data(dataset,metric=metric)

    #Consturct weight matrix and distance matrix
    W = weight_matrix(I,J,D,k,symmetrize=False)
    Wdist = dist_matrix(I,J,D,k)

    #Load label permutation (including restrictions in t)
    if isinstance(t,int) or isinstance(t,float):
        t = str(int(t))
    perm = load_label_permutation(dataset,label_perm=label_perm,t=t)

    #Load eigenvector data if MBO selected
    if algorithm in ['mbo'] or require_eigen_data:
        vals,vecs,vals_norm,vecs_norm = load_eig(dataset,metric,k)
        params['vals']=vals
        params['vals_norm']=vals_norm
        params['vecs']=vecs
        params['vecs_norm']=vecs_norm
    else:
        vals = None
        vecs = None
        vals_norm = None
        vecs_norm = None

    #Output file
    outfile = dataset+label_perm+"_"+metric+"_k%d"%k
    if algorithm == 'plaplace':
        outfile = outfile+"_p%.1f"%p+algorithm[1:]+"_"+norm
    elif algorithm == 'eikonal':
        outfile = outfile+"_p%.1f"%p+algorithm
    else:
        outfile = outfile+"_"+algorithm

    if algorithm == 'volumembo' or algorithm == 'poissonvolumembo':
        outfile = outfile+"_T%.3f"%T
        outfile = outfile+"_V%.3f"%volume_constraint

    if algorithm == 'poisson' and poisson_training_balance == False:
        outfile = outfile+"_NoBal"

    basename = outfile
    outfile = outfile+"_accuracy.csv"
    outfile = os.path.join(Results_dir(),outfile)

    #Print basic info
    print('========================================================')
    print('GraphLearning: Python package for graph-based learning. ')
    print('========================================================')
    print('                                                        ')
    print('Dataset: '+dataset)
    print('Metric: '+metric)
    print('Number of neighbors: %d'%k)
    print('Learning algorithm: '+algorithm)
    print('Laplacian normalization: '+norm)
    if algorithm == 'plaplace' or algorithm == 'eikonal':
        print("p-Laplace/eikonal value p=%.2f" % p)
    if algorithm in clustering_algorithms:
        print('Number of clusters: %d'%num_classes)
        if algorithm == 'INCRES':
            print('INCRES speed: %.2f'%speed)
            print('Number of iterations: %d'%num_iter)
        if algorithm[:8] == 'Spectral':
            print('Number of extra dimensions: %d'%extra_dim)
    else:
        print('Number of trial permutations: %d'%len(perm))
        print('Permutations file: LabelPermutations/'+dataset+label_perm+'_permutations.npz')

        if algorithm == 'volumembo' or algorithm == 'poissonvolumembo':
            print("Using temperature=%.3f"%T)
            print("Volume constraints = [%.3f,%.3f]"%(volume_constraint,2-volume_constraint))

        #If output file selected
        if results:
            print('Output file: '+outfile)

    print('                                                        ')
    print('========================================================')
    print('                                                        ')

    true_labels = None
    if verbose:
        true_labels = labels

    #If output file selected
    if results:
        #Check if Results directory exists
        if not os.path.exists(Results_dir()):
            os.makedirs(Results_dir())

        now = datetime.datetime.now()
        
        #Add time stamp to output file
        f = open(outfile,"a+")
        f.write("Date/Time, "+now.strftime("%Y-%m-%d_%H:%M")+"\n")
        f.close()

    #Loop over label permutations
    print("Number of labels, Accuracy")
    def one_trial(label_ind):

        #Number of labels
        m = len(label_ind)

        #Label proportions (used by some algroithms)
        beta = label_proportions(labels)

        #start_time = time.time()
        #Graph-based semi-supervised learning
        u = graph_ssl(W,label_ind,labels[label_ind],D=Wdist,beta=beta,algorithm=algorithm,epsilon=0.3,p=p,norm=norm,vals=vals,vecs=vecs,vals_norm=vals_norm,vecs_norm=vecs_norm,dataset=dataset,T=T,use_cuda=use_cuda,volume_mult=volume_constraint,true_labels=true_labels,poisson_training_balance=poisson_training_balance,symmetrize = not directed_graph, X=data, params=params)
        #print("--- %s seconds ---" % (time.time() - start_time))

        #Compute accuracy
        acc = accuracy(u,labels,m)
        
        #Print to terminal
        print("%d" % m + ",%.2f" % acc)

        #Write to file
        if results:
            f = open(outfile,"a+")
            f.write("%d" % m + ",%.2f\n" % acc)
            f.close()

    #Number of cores for parallel processing
    num_cores = min(multiprocessing.cpu_count(),num_cores)
    Parallel(n_jobs=num_cores)(delayed(one_trial)(label_ind) for label_ind in perm)

    return basename

if __name__ == '__main__':

    #Default settings
    dataset = default_dataset()
    metric = default_metric()
    algorithm = default_algorithm()
    k = default_k()
    t = default_t()
    label_perm = default_label_perm()
    p = default_p()
    norm = default_norm()
    use_cuda = default_use_cuda()
    T = default_T()
    num_cores = default_num_cores()
    results = default_results()
    num_classes = default_num_classes()
    speed = default_speed()
    num_iter = default_num_iter()
    extra_dim = default_extra_dim()
    volume_constraint = default_volume_constraint()
    verbose = default_verbose()
    poisson_training_balance = default_poisson_training_balance()
    directed_graph = default_directed_graph()

    #Read command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hd:m:k:a:p:n:v:N:s:i:x:t:cl:T:j:rboz",["dataset=","metric=","knn=","algorithm=","p=","normalization=","volume_constraint=","num_classes=","speed=","num_iter=","extra_dim=","num_trials=","cuda","label_perm=","temperature=","num_cores=","results","verbose","poisson_training_balance","directed"])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-m", "--metric"):
            metric = arg
        elif opt in ("-k", "--knn"):
            k = int(arg)
        elif opt in ("-a", "--algorithm"):
            algorithm = arg.lower()
        elif opt in ("-p", "--p"):
            p = float(arg)
        elif opt in ("-n", "--normalization"):
            norm = arg
        elif opt in ("-v", "--volume_constraint"):
            volume_constraint = float(arg)
        elif opt in ("-N", "--num_classes"):
            num_classes = int(arg)
        elif opt in ("-s", "--speed"):
            speed = float(arg)
        elif opt in ("-i", "--num_iter"):
            num_iter = int(arg)
        elif opt in ("-x", "--extra_dim"):
            extra_dim = int(arg)
        elif opt in ("-t", "--num_trials"):
            t = arg
        elif opt in ("-c", "--cuda"):
            use_cuda = True
        elif opt in ("-l", "--label_perm"):
            label_perm = arg
        elif opt in ("-T", "--temperature"):
            T = float(arg)
        elif opt in ("-j", "--num_cores"):
            num_cores = int(arg)
        elif opt in ("-r", "--results"):
            results = False
        elif opt in ("-b", "--verbose"):
            verbose = True
        elif opt in ("-o", "--poisson_training_balance"):
            poisson_training_balance = False
        elif opt in ("-z", "--directed"):
            directed_graph = True

    #Call main subroutine
    main(dataset=dataset, metric=metric, algorithm=algorithm, k=k, t=t, label_perm=label_perm, p=p, norm=norm, use_cuda=use_cuda, T=T, num_cores=num_cores, results=results, num_classes=num_classes, speed=speed, num_iter=num_iter, extra_dim=extra_dim, volume_constraint=volume_constraint, verbose=verbose, poisson_training_balance=poisson_training_balance, directed_graph=directed_graph)




