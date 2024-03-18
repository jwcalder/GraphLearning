"""
Clustering
==========

This module implements many graph-based clustering algorithms in an objected-oriented
fashion, similar to [sklearn](https://scikit-learn.org/stable/).
"""

from abc import ABCMeta, abstractmethod
import scipy.optimize as opt
import sklearn.cluster as cluster
from scipy import sparse
from scipy import linalg
import numpy as np
import sys

from . import graph

class clustering:
    __metaclass__ = ABCMeta

    def __init__(self, W, num_clusters):
        if type(W) == graph.graph:
            self.graph = W
        else:
            self.graph = graph.graph(W)
        self.cluster_labels = None
        self.num_clusters = num_clusters
        self.fitted = False

    def predict(self):
        """Predict
        ========

        Makes label predictions based on clustering. 
        
        Returns
        -------
        pred_labels : (int) numpy array
            Predicted labels as integers for all datapoints in the graph.
        """

        if self.fitted == False:
            sys.exit('Model has not been fitted yet.')

        return self.cluster_labels

    def fit_predict(self, all_labels=None):
        """Fit and predict
        ======

        Calls fit() and predict() sequentially.

        Parameters
        ----------
        all_labels : numpy array, int (optional)
            True labels for all datapoints.

        Returns
        -------
        pred_labels : (int) numpy array
            Predicted labels as integers for all datapoints in the graph.
        """

        self.fit(all_labels=all_labels)
        return self.predict()

    def fit(self, all_labels=None):
        """Fit
        ======

        Solves clustering problem to perform clustering. 

        Parameters
        ----------
        all_labels : numpy array, int (optional)
            True labels for all datapoints.

        Returns
        -------
        all_labels : numpy array, int (optional)
            True labels for all datapoints.
        """

        pred_labels = self._fit(all_labels=all_labels)
        self.fitted = True
        self.cluster_labels = pred_labels

        return pred_labels


    @abstractmethod
    def _fit(self, all_labels=None):
        """Fit
        ======

        Solves clustering problem to perform clustering. 

        Parameters
        ----------
        all_labels : numpy array, int (optional)
            True labels for all datapoints.

        Returns
        -------
        all_labels : numpy array, int (optional)
            True labels for all datapoints.
        """

        raise NotImplementedError("Must override _fit")


class spectral(clustering):
    def __init__(self, W, num_clusters, method='NgJordanWeiss', extra_dim=0):
        """Spectral clustering
        ===================

        Implements several methods for spectral clustering, including Shi-Malik and Ng-Jordan-Weiss. See
        the tutorial paper [1] for details.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object
            Weight matrix representing the graph.
        num_clusters : int
            Number of desired clusters.
        method : {'combinatorial', 'ShiMalik', 'NgJordanWeiss'} (optional), default='NgJordanWeiss'
            Spectral clustering method.
        extra_dim : int (optional), default=0
            Extra dimensions to include in spectral embedding.
        
        Examples
        ----
        Spectral clustering on the two-moons dataset: [spectral_twomoons.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/spectral_twomoons.py).
        ```py
        import numpy as np
        import graphlearning as gl
        import matplotlib.pyplot as plt
        import sklearn.datasets as datasets

        X,labels = datasets.make_moons(n_samples=500,noise=0.1)
        W = gl.weightmatrix.knn(X,10)

        model = gl.clustering.spectral(W, num_clusters=2)
        pred_labels = model.fit_predict()

        accuracy = gl.clustering.clustering_accuracy(pred_labels, labels)
        print('Clustering Accuracy: %.2f%%'%accuracy)

        plt.scatter(X[:,0],X[:,1], c=pred_labels)
        plt.axis('off')
        plt.show()
        ```
        Spectral clustering on MNIST: [spectral_mnist.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/spectral_mnist.py).
        ```py
        import graphlearning as gl

        W = gl.weightmatrix.knn('mnist', 10, metric='vae')
        labels = gl.datasets.load('mnist', labels_only=True)

        model = gl.clustering.spectral(W, num_clusters=10, extra_dim=4)
        pred_labels = model.fit_predict(all_labels=labels)
        
        accuracy = gl.clustering.clustering_accuracy(pred_labels,labels)
        print('Clustering Accuracy: %.2f%%'%accuracy)
        ```

        Reference
        ---------
        [1] U. Von Luxburg.  [A tutorial on spectral clustering.](https://link.springer.com/content/pdf/10.1007/s11222-007-9033-z.pdf) Statistics and computing 17.4 (2007): 395-416.
        """
        super().__init__(W, num_clusters)
            
        self.method = method
        self.extra_dim = extra_dim

    def _fit(self, all_labels=None):

        n = self.graph.num_nodes
        num_clusters = self.num_clusters
        method = self.method
        extra_dim = self.extra_dim

        if method == 'combinatorial':
            vals, vec = self.graph.eigen_decomp(k=num_clusters+extra_dim)
        elif method == 'ShiMalik':
            vals, vec = self.graph.eigen_decomp(normalization='randomwalk', k=num_clusters+extra_dim)
        elif method == 'NgJordanWeiss':
            vals, vec = self.graph.eigen_decomp(normalization='normalized', k=num_clusters+extra_dim)
            norms = np.sum(vec*vec,axis=1)
            T = sparse.spdiags(norms**(-1/2),0,n,n)
            vec = T@vec  #Normalize rows
        else:
            sys.exit("Invalid spectral clustering method " + method)

        kmeans = cluster.KMeans(n_clusters=num_clusters).fit(vec)

        return kmeans.labels_

class fokker_planck(clustering):
    def __init__(self, W, num_clusters, beta=0.5, t=1, rho=None):
        """FokkerPlanck clustering
        ===================

        Implements the Fokker-Planck clustering algorithm from [1].

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object
            Weight matrix representing the graph.
        num_clusters : int
            Number of desired clusters.
        beta : float (optional), default=0.5
            Interpolation parameter between mean shift and diffusion.
        t : float (optional), default=1
            Time to run Fokker-Planck equation
        rho : numpy array (optional), default=None
            Density estimator for mean shift. Default is uniform density.

        Examples
        ----
        Fokker-Planck clustering on the two-skies dataset: [fokker_planck_clustering.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/fokker_planck_clustering.py).
        ```py
        import numpy as np
        import graphlearning as gl
        import matplotlib.pyplot as plt

        X,L = gl.datasets.two_skies(1000)
        W = gl.weightmatrix.knn(X,10)

        knn_ind,knn_dist = gl.weightmatrix.knnsearch(X,50)
        rho = 1/np.max(knn_dist,axis=1)

        model = gl.clustering.fokker_planck(W,num_clusters=2,t=1000,beta=0.5,rho=rho)
        labels = model.fit_predict()

        plt.scatter(X[:,0],X[:,1], c=labels)
        plt.show()
        ```

        Reference
        ---------
        [1] K. Craig, N.G. Trillos, & D. Slepčev. (2021). Clustering dynamics on graphs: from spectral clustering to mean shift through Fokker-Planck interpolation. arXiv:2108.08687.
        """
        super().__init__(W, num_clusters)
            
        self.beta = beta
        self.t = t
        if rho is None:
            self.rho = np.ones(W.shape[0])
        else:
            self.rho = rho

    def _fit(self, all_labels=None):

        beta = self.beta
        t = self.t
        rhoinv = 1/self.rho

        #Coifman/Lafon
        Q1 = -self.graph.laplacian(normalization='coifmanlafon')

        #Mean shift transition matrix
        Qms = self.graph.gradient(rhoinv, weighted=True).T
        Qms[Qms<0] = 0
        Qms = Qms - graph.graph(Qms).degree_matrix()

        #Interplation
        Q = beta*Qms + (1-beta)*Q1
        Q = Q.toarray()

        #Matrix exponential
        #expQt = sparse.linalg.expm(Q*t)
        #Y = expQt.toarray()
        expQt = linalg.expm(Q*t)

        #kmeans
        kmeans = cluster.KMeans(n_clusters=self.num_clusters).fit(expQt)

        return kmeans.labels_

class incres(clustering):
    def __init__(self, W, num_clusters, speed=5, T=200):
        """INCRES clustering
        ===================

        Implements the INCRES clustering algorithm from [1].

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object
            Weight matrix representing the graph.
        num_clusters : int
            Number of desired clusters.
        speed : float (optional), default=5
            Speed parameter.
        T : int (optional), default=100
            Number of iterations.

        Example
        ----
        INCRES clustering on MNIST: [incres_mnist.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/incres_mnist.py).
        ```py
        import graphlearning as gl

        W = gl.weightmatrix.knn('mnist', 10, metric='vae')
        labels = gl.datasets.load('mnist', labels_only=True)

        model = gl.clustering.incres(W, num_clusters=10)
        pred_labels = model.fit_predict(all_labels=labels)
        
        accuracy = gl.clustering.clustering_accuracy(pred_labels,labels)
        print('Clustering Accuracy: %.2f%%'%accuracy)
        ```

        Reference
        ---------
        [1] X. Bresson, H. Hu, T. Laurent, A. Szlam, and J. von Brecht. [An incremental reseeding strategy for clustering](https://arxiv.org/pdf/1406.3837.pdf). In International Conference on Imaging, Vision and Learning based on Optimization and PDEs (pp. 203-219), 2016.
        """
        super().__init__(W, num_clusters)
            
        self.speed = speed
        self.T = T

    def _fit(self, all_labels=None):

        #Short cuts
        n = self.graph.num_nodes
        speed = self.speed
        T = self.T
        k = self.num_clusters

        #Increment
        Dm = np.maximum(int(speed*1e-4*n/k),1)
        
        #Random initial labeling
        u = np.random.randint(0,k,size=n)

        #Initialization
        F = np.zeros((n,k))
        J = np.arange(n).astype(int)

        #Random walk transition
        D = self.graph.degree_matrix(p=-1)
        P = self.graph.weight_matrix*D

        m = int(1)
        for i in range(T):
            #Plant
            F.fill(0)
            for r in range(k):
                I = u == r
                ind = J[I]
                F[ind[np.random.choice(np.sum(I),m)],r] = 1
            
            #Grow
            while np.min(F) == 0:
                F = P*F

            #Harvest
            u = np.argmax(F,axis=1)

            #Increment
            m = m + Dm
                
            #Compute accuracy
            if all_labels is not None: 
                acc = clustering_accuracy(u,all_labels)
                print("Iteration "+str(i)+": Accuracy = %.2f" % acc+"%%, #seeds= %d" % m)

        return u

def withinss(x):
    """WithinSS
    ======

    Clustering of 1D data with WithinSS. Gives exact solution to the 2-means clustering problem

    Parameters
    ----------
    x : numpy array
        1D array of data to cluter.

    Returns
    -------
    w : float
        WithinSS value, essentially the 2-means energy.
    m : float
        Threshold that clusters the data array x optimally.
    """

    x = np.sort(x)
    n = x.shape[0]
    sigma = np.std(x)
    v = np.zeros(n-1,)

    #Initial values for m1,m2
    x1 = x[:1]
    x2 = x[1:]
    m1 = np.mean(x1)
    m2 = np.mean(x2)
    for i in range(n-1):
        v[i] = (i+1)*m1**2 + (n-i-1)*m2**2
        if i < n-2:
            m1 = ((i+1)*m1 + x[i+1])/(i+2)
            m2 = ((n-i-1)*m2 - x[i+1])/(n-i-2)
    ind = np.argmax(v)
    m = x[ind]
    w = (np.sum(x**2) - v[ind])/(n*sigma**2)
    return w,m

def RP1D(X,T=100):
    """Random Projection Clustering
    ======

    Binary clustering of 1D data with the Random Projection 1D (RP1D) clustering method from [1].

    Parameters
    ----------
    X : numpy array
        (n,d) dimensional array of n datapoints in dimension d.
    T : int (optional), default=100
        Number of random projections to try.

    Example
    -------
    RP1D clustering on MNIST: [RP1D_mnist.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/RP1D_mnist.py).
    ```py
    import graphlearning as gl

    data, labels = gl.datasets.load('mnist')

    x = data[labels <= 1] 
    y = labels[labels <= 1]
    y_pred = gl.clustering.RP1D(x,20)

    accuracy = gl.clustering.clustering_accuracy(y_pred, y)
    print('Clustering Accuracy: %.2f%%'%accuracy)
    ```

    Returns
    -------
    cluster_labels : int
        0/1 array indicating cluster membership

    References
    ----------
    [1] S. Han and M. Boutin. [The hidden structure of image datasets.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7350969&casa_token=UsN9y0textMAAAAA:-K9r-Sv4njFQ_txJUpkqCbavM-wTA2CmkgU3co7RjmjTKdcP3guTjahyHA7jZBs1WZTz-E2fETQ&tag=1) 2015 IEEE International Conference on Image Processing (ICIP). IEEE, 2015.
    """

    n = X.shape[0]
    d = X.shape[1]
    v = np.random.rand(T,d)
    wmin = np.inf
    imin = 0;
    for i in range(T):
        x = np.sum(v[i,:]*X,axis=1)
        w,m = withinss(x)
        if w < wmin:
            wmin = w
            imin = i
    x = np.sum(v[imin,:]*X,axis=1)
    w,m = withinss(x)

    cluster_labels = np.zeros(n,)
    cluster_labels[x>m] = 1

    return cluster_labels

def clustering_accuracy(pred_labels,true_labels):
    """Clustering accuracy
    ======

    Accuracy for clustering in graph learning. Uses a linear sum assignment
    to find the best permutation of cluster labels.

    Parameters
    ----------
    pred_labels : numpy array, int
        Predicted labels. Should be 0,1,...,k-1 if k classes/clusters
    true_labels : numpy array, int
        True labels. Can be any integers, will be converted to 0,1,...,k-1

    Returns
    -------
    accuracy : float
        Accuracy as a number in [0,100].
    """

    tl = true_labels.copy()
    unique_classes = np.unique(tl)
    num_classes = len(unique_classes)

    #Need to copy true labels

    ind = []
    for c in unique_classes:
        ind_c = tl == c
        ind.append(ind_c)

    for i in range(num_classes):
        tl[ind[i]] = i

    C = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            C[i][j] = np.sum((pred_labels == i) & (tl != j))
    row_ind, col_ind = opt.linear_sum_assignment(C)

    return 100*(1-C[row_ind,col_ind].sum()/len(pred_labels))


def purity(cluster_labels,true_labels):
    """Clustering purity
    ======

    Computes clustering purity, which is the fraction of nodes
    that belong to the largest class in each cluster.

    Parameters
    ----------
    cluster_labels : numpy array, int
        Cluster labels
    true_labels : numpy array, int
        True labels

    Returns
    -------
    purity : float
        Purity as a number in [0,100].
    """

    classes = np.unique(true_labels)
    num_classes = len(classes)
    clusters = np.unique(cluster_labels)
    num_clusters = len(clusters)

    purity = []
    size = []
    for c in clusters:
        labels = true_labels[cluster_labels == c]
        purity += [np.max(np.bincount(labels))]
        size += [len(labels)]
    purity = np.array(purity)
    size = np.array(size)

    return 100*np.sum(purity)/np.sum(size), purity/size



