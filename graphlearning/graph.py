"""
Graph Class
========================

This module contains the `graph` class, which implements many graph-based algorithms, including
spectral decompositions, distance functions (via Dijkstra and peikonal), PageRank, AMLE (Absolutely 
Minimal Lipschitz Extensions), p-Laplace equations, and basic calculus on graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import spatial
import scipy.sparse.linalg as splinalg
import scipy.sparse.csgraph as csgraph
import time
import sys
import pickle

from . import utils

class graph:

    def __init__(self, W, labels=None, features=None, label_names=None, node_names=None):
        """Graph class
        ========

        A class for graphs, including routines to compute Laplacians and their
        eigendecompositions, which are useful in graph learning.

        Parameters
        ----------
        W : (n,n) numpy array, matrix, or scipy sparse matrix
            Weight matrix representing the graph.
        labels : (n,) numpy array (optional)
            Node labels.
        features : (n,k) numpy array (optional)
            Node features.
        label_names : list (optional)
            Names corresponding to each label.
        node_names : list (optional)
            Names for each node in the graph.
        """

        self.weight_matrix = sparse.csr_matrix(W)
        self.labels = labels
        self.features = features
        self.num_nodes = W.shape[0]
        self.label_names = label_names
        self.node_names = node_names

        self.__ccode_init__()

        self.eigendata = {}
        normalizations = ['combinatorial','randomwalk','normalized']

        for norm in normalizations:
            self.eigendata[norm] = {}
            self.eigendata[norm]['eigenvectors'] = None
            self.eigendata[norm]['eigenvalues'] = None
            self.eigendata[norm]['method'] = None
            self.eigendata[norm]['k'] = None
            self.eigendata[norm]['c'] = None
            self.eigendata[norm]['gamma'] = None
            self.eigendata[norm]['tol'] = None
            self.eigendata[norm]['q'] = None

    def __ccode_init__(self):

        #Coordinates of sparse matrix for passing to C code
        I,J,V = sparse.find(self.weight_matrix)
        ind = np.argsort(I)
        self.I,self.J,self.V = I[ind], J[ind], V[ind]
        self.K = np.array((self.I[1:] - self.I[:-1]).nonzero()) + 1
        self.K = np.append(0,np.append(self.K,len(self.I)))
        self.Vinv = 1/self.V

        #For passing to C code
        self.I = np.ascontiguousarray(self.I, dtype=np.int32)
        self.J = np.ascontiguousarray(self.J, dtype=np.int32)
        self.V = np.ascontiguousarray(self.V, dtype=np.float64)
        self.Vinv = np.ascontiguousarray(self.Vinv, dtype=np.float64)
        self.K = np.ascontiguousarray(self.K, dtype=np.int32)

    def subgraph(self,ind):
        """Sub-Graph
        ======

        Returns the subgraph corresponding to the supplied indices.

        Parameters
        ----------
        ind : numpy array, int
            Indices for subgraph.

        Returns
        ----------
        G : graph object
            Subgraph corresponding to the indices contained in `ind`.

        """

        W = self.weight_matrix 
        return graph(W[ind,:][:,ind])


    def degree_vector(self):
        """Degree Vector
        ======

        Given a weight matrix \\(W\\), returns the diagonal degree vector
        \\[d_{i} = \\sum_{j=1}^n w_{ij}.\\]

        Returns
        -------
        d : numpy array, float
            Degree vector for weight matrix.
        """

        d = self.weight_matrix*np.ones(self.num_nodes)
        return d

    def neighbors(self, i, return_weights=False):
        """Neighbors
        ======

        Returns neighbors of node i.

        Parameters
        ----------
        i : int 
            Index of vertex to return neighbors of.
        return_weights : bool (optional), default=False
            Whether to return the weights of neighbors as well.

        Returns
        -------
        N : numpy array, int
            Array of nearest neighbor indices.
        W : numpy array, float
            Weights of edges to neighbors. 
        """
        
        N = self.weight_matrix[i,:].nonzero()[1]
        N = N[N != i]

        if return_weights:
            return N, self.weight_matrix[i,N].toarray().flatten()
        else:
            return N

    def fiedler_vector(self, return_value=False, tol=1e-8):
        """Fiedler Vector
        ======

        Computes the Fiedler vector for graph, which is the eigenvector 
        of the graph Laplacian correpsonding to the second smallest eigenvalue.

        Parameters
        ----------
        return_value : bool (optional), default=False
            Whether to return Fiedler value.
        tol : float (optional), default=0
            Tolerance for eigensolvers.

        Returns
        -------
        v : numpy array, float
            Fiedler vector
        l : float (optional)
            Fiedler value
        """
        
        #vals, vecs = self.eigen_decomp(k=2,method=method,tol=tol)
        #if return_value:
        #    return vecs[:,1], vals[1]
        #else:
        #    return vecs[:,1]

        L = self.laplacian()
        m = self.num_nodes
        v = np.random.rand(m,1)
        o = np.ones((m,1))/m
        v -= np.sum(v)*o
        d = self.degree_vector()
        lam = 2*np.max(d)
        M = lam*sparse.identity(m) - L
        fval_old = v.T@(L@v)
        err = 1
        while err > tol:
            x = M@v
            x -= np.sum(x)*o
            v = x/np.linalg.norm(x)
            fval = v.T@(L@v)
            err = abs(fval_old-fval)
            fval_old = fval

        v = v.flatten()
        #Fix consistent sign
        if v[0] > 0:
            v = -v
        if return_value:
            return v, fval
        else:
            return v



    def degree_matrix(self, p=1):
        """Degree Matrix
        ======

        Given a weight matrix \\(W\\), returns the diagonal degree matrix 
        in the form
        \\[D_{ii} = \\left(\\sum_{j=1}^n w_{ij}\\right)^p.\\]

        Parameters
        ----------
        p : float (optional), default=1
            Optional exponent to apply to the degree.

        Returns
        -------
        D : (n,n) scipy sparse matrix, float
            Sparse diagonal degree matrix.
        """

        #Construct sparse degree matrix
        d = self.degree_vector()
        D = sparse.spdiags(d**p, 0, self.num_nodes, self.num_nodes)

        return D.tocsr()


    def rand(self):
        """Uniform random matrix with same sparsity structure
        ======

        Given a weight matrix \\(W\\), returns a random matrix \\(A\\),
        where the entry \\(A_{ij}\\) is a uniform random variable on \\([0,1]\\)
        whenever \\(w_{ij}>0\\), and \\(A_{ij}=0\\) otherwise.

        Returns
        -------
        A : (n,n) scipy sparse matrix, float
            Sparse rand_like matrix.
        """

        n = self.num_nodes
        vals = np.random.rand(len(self.I),1).flatten()
        A = sparse.coo_matrix((vals,(self.I,self.J)),shape=(n,n)).tocsr() 
        return A

    def randn(self):
        """Gaussian random matrix with same sparsity structure
        ======

        Given a weight matrix \\(W\\), returns a random matrix \\(A\\),
        where the entry \\(A_{ij}\\) is a uniform random variable on \\([0,1]\\)
        whenever \\(w_{ij}>0\\), and \\(A_{ij}=0\\) otherwise.

        Returns
        -------
        A : (n,n) scipy sparse matrix, float
            Sparse rand_like matrix.
        """

        n = self.num_nodes
        vals = np.random.randn(len(self.I),1).flatten()
        A = sparse.coo_matrix((vals,(self.I,self.J)),shape=(n,n)).tocsr() 
        return A

    def adjacency(self):
        """Adjacency matrix
        ======

        Given a weight matrix \\(W\\), returns the adjacency matrix \\(A\\),
        which satisfies \\(A_{ij}=1\\) whenever \\(w_{ij}>0\\), and  \\(A_{ij}=0\\)
        otherwise.

        Returns
        -------
        A : (n,n) scipy sparse matrix, float
            Sparse adjacency matrix.
        """

        n = self.num_nodes
        A = sparse.coo_matrix((np.ones(len(self.V),),(self.I,self.J)),shape=(n,n)).tocsr() 
        return A

    def gradient(self, u, weighted=False, p=0.0):
        """Graph Gradient
        ======

        Computes the graph gradient \\(\\nabla u\\) of \\(u\\in \\mathbb{R}^n\\), which is
        the sparse matrix with the form
        \\[\\nabla u_{ij} = u_j - u_i,\\]
        whenever \\(w_{ij}>0\\), and \\(\\nabla u_{ij}=0\\) otherwise.
        If `weighted=True` is chosen, then the gradient is weighted by the graph weight 
        matrix as follows
        \\[\\nabla u_{ij} = w_{ij}^p(u_j - u_i).\\]

        Parameters
        ----------
        u : numpy array, float
            Vector (graph function) to take gradient of
        weighted : bool (optional), default=False,True
            Whether to weight the gradient by the graph weight matrix. Default is False when p=0 and True when \\(p\\neq 0\\).
        p : float (optional), default=0,1
            Power for weights on weighted gradient. Default is 0 when unweighted and 1 when weighted.

        Returns
        -------
        G : (n,n) scipy sparse matrix, float
            Sparse graph gradient matrix
        """

        n = self.num_nodes

        if p != 0.0:
            weighted = True

        if weighted == True and p==0.0:
            p = 1.0

        if weighted:
            G = sparse.coo_matrix(((self.V**p)*(u[self.J]-u[self.I]), (self.I,self.J)),shape=(n,n)).tocsr()
        else:
            G = sparse.coo_matrix((u[self.J]-u[self.I], (self.I,self.J)),shape=(n,n)).tocsr()

        return G

    def divergence(self, V, weighted=True):
        """Graph Divergence
        ======

        Computes the graph divergence \\(\\text{div} V\\) of a vector field \\(V\\in \\mathbb{R}^{n\\times n}\\), 
        which is the vector 
        \\[\\nabla u_{ij} = u_j - u_i,\\]
        If `weighted=True` is chosen, then the divergence is weighted by the graph weight 
        matrix as follows
        \\[\\nabla u_{ij} = w_{ij}(u_j - u_i).\\]

        Parameters
        ----------
        V : scipy sparse matrix, float
            Sparse matrix representing a vector field over the graph.
        weighted : bool (optional), default=True
            Whether to weight the divergence by the graph weight matrix.

        Returns
        -------
        divV : numpy array
            Divergence of V.
        """
    
        V = V - V.transpose()

        if weighted:
            V = V.multiply(self.weight_matrix)

        divV = V*np.ones(self.num_nodes)/2

        return divV

     
    def reweight(self, idx, method='poisson', normalization='combinatorial', tau=0, X=None, alpha=2, zeta=1e7, r=0.1):
        """Reweight a weight matrix
        ======

        Reweights the graph weight matrix more heavily near labeled nodes. Used in semi-supervised
        learning at very low label rates. [Need to describe all methods...]

        Parameters
        ----------
        idx : numpy array (int)
            Indices of points to reweight near (typically labeled points).
        method : {'poisson','wnll','properly'}, default='poisson'
            Reweighting method. 'poisson' is described in [1], 'wnll' is described in [2], and 'properly'
            is described in [3]. If 'properly' is selected, the user must supply the data features `X`.
        normalization : {'combinatorial','normalized'}, default='combinatorial'
            Type of normalization to apply for the graph Laplacian when method='poisson'.
        tau : float or numpy array (optional), default=0
            Zeroth order term in Laplace equation. Can be a scalar or vector.
        X : numpy array (optional)
            Data features, used to construct the graph. This is required for the `properly` weighted 
            graph Laplacian method.
        alpha : float (optional), default=2
            Parameter for `properly` reweighting.
        zeta : float (optional), default=1e7
            Parameter for `properly` reweighting.
        r : float (optional), default=0.1
            Radius for `properly` reweighting.

        Returns
        -------
        W : (n,n) scipy sparse matrix, float
            Reweighted weight matrix as sparse scipy matrix.

        References
        ----------
        [1] J. Calder, B. Cook, M. Thorpe, D. Slepcev. [Poisson Learning: Graph Based Semi-Supervised Learning at Very Low Label Rates.](http://proceedings.mlr.press/v119/calder20a.html), 
        Proceedings of the 37th International Conference on Machine Learning, PMLR 119:1306-1316, 2020.

        [2] Z. Shi, S. Osher, and W. Zhu. [Weighted nonlocal laplacian on interpolation from sparse data.](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/article/10.1007/s10915-017-0421-z&casa_token=33Z7gqJy3mMAAAAA:iMO0pGmpn_qf5PioVIGocSRq_p4CDm-KNOQhgIC1uvqG9pWlZ6t7I-IZtSJfocFDEHCdMpK8j7Fx1XbzDQ)
        Journal of Scientific Computing 73.2 (2017): 1164-1177.

        [3] J. Calder, D. Slepčev. [Properly-weighted graph Laplacian for semi-supervised learning.](https://link.springer.com/article/10.1007/s00245-019-09637-3) Applied mathematics & optimization (2019): 1-49.
        """

        if method == 'poisson':
            
            n = self.num_nodes
            f = np.zeros(n)
            f[idx] = 1

            if normalization == 'combinatorial':
                f -= np.mean(f)
                L = self.laplacian()
            elif normalization == 'normalized':
                d = self.degree_vector()**(0.5)
                c = np.sum(d*f)/np.sum(d)
                f -= c
                L = self.laplacian(normalization=normalization)
            else:
                sys.exit('Unsupported normalization '+normalization+' for graph.reweight.')

            w = utils.conjgrad(L, f, tol=1e-5)
            w -= np.min(w)
            w += 1e-5
            D = sparse.spdiags(w,0,n,n).tocsr()

            return D*self.weight_matrix*D

        elif method == 'wnll':

            n = self.num_nodes
            m = len(idx)

            a = np.ones((n,))
            a[idx] = n/m
            
            D = sparse.spdiags(a,0,n,n).tocsr()

            return D*self.weight_matrix + self.weight_matrix*D

        elif method == 'properly':

            if X is None:
                sys.exit('Must provide data features X for properly weighted graph Laplacian.')

            n = self.num_nodes
            m = len(idx)
            rzeta = r/(zeta-1)**(1/alpha)
            Xtree = spatial.cKDTree(X[idx,:])
            D, J = Xtree.query(X)
            D[D < rzeta] = rzeta
            gamma = 1 + (r/D)**alpha

            D = sparse.spdiags(gamma,0,n,n).tocsr()

            return D*self.weight_matrix + self.weight_matrix*D

        else:
            sys.exit('Invalid reweighting method ' + method + '.')


    def laplacian(self, normalization="combinatorial", alpha=1):
        """Graph Laplacian
        ======

        Computes various normalizations of the graph Laplacian for a 
        given weight matrix \\(W\\). The choices are
        \\[L_{\\rm combinatorial} = D - W,\\]
        \\[L_{\\rm randomwalk} = I - D^{-1}W,\\]
        and
        \\[L_{\\rm normalized} = I - D^{-1/2}WD^{-1/2},\\]
        where \\(D\\) is the diagonal degree matrix, which is defined as
        \\[D_{ii} = \\sum_{j=1}^n w_{ij}.\\]
        The Coifman-Lafon Laplacian is also supported. 

        Parameters
        ----------
        normalization : {'combinatorial','randomwalk','normalized','coifmanlafon'}, default='combinatorial'
            Type of normalization to apply.
        alpha : float (optional)
            Parameter for Coifman-Lafon Laplacian

        Returns
        -------
        L : (n,n) scipy sparse matrix, float
            Graph Laplacian as sparse scipy matrix.
        """

        I = sparse.identity(self.num_nodes)
        D = self.degree_matrix()

        if normalization == "combinatorial":
            L = D - self.weight_matrix
        elif normalization == "randomwalk":
            Dinv = self.degree_matrix(p=-1)
            L = I - Dinv*self.weight_matrix
        elif normalization == "normalized":
            Dinv2 = self.degree_matrix(p=-0.5)
            L = I - Dinv2*self.weight_matrix*Dinv2
        elif normalization == "coifmanlafon":
            D = self.degree_matrix(p=-alpha)
            L = graph(D*self.weight_matrix*D).laplacian(normalization='randomwalk')
        else:
            sys.exit("Invalid option for graph Laplacian normalization.")

        return L.tocsr()

    def infinity_laplacian(self,u):
        """Graph Infinity Laplacian
        ======

        Computes the graph infinity Laplacian of a vector \\(u\\), given by
        \\[L_\\infty u_i= \\min_j w_{ij}(u_j-u_i) + \\max_j w_{ij} (u_j-u_i).\\]
               
        Returns
        -------
        Lu : numpy array
            Graph infinity Laplacian.
        """

        n = self.num_nodes
        M = sparse.coo_matrix((self.V*(u[self.J]-u[self.I]), (self.I,self.J)),shape=(n,n)).tocsr()
        M = M.min(axis=1) + M.max(axis=1)
        Lu = M.toarray().flatten()

        return Lu

    def isconnected(self):
        """Is Connected
        ======

        Checks if the graph is connected.
               
        Returns
        -------
        connected : bool
            True or False, depending on connectivity.
        """

        num_comp,comp = csgraph.connected_components(self.weight_matrix)
        connected = False
        if num_comp == 1:
            connected = True
        return connected

    def largest_connected_component(self):
        """Largest connected component
        ======

        Finds the largest connected component of the graph. Returns the restricted 
        graph, as well as a boolean mask indicating the nodes belonging to 
        the component.
               
        Returns
        -------
        G : graph object
            Largest connected component graph.
        ind : numpy array (bool)
            Mask indicating which nodes from the original graph belong to the 
            largest component.
        """

        ncomp,labels = csgraph.connected_components(self.weight_matrix,directed=False) 
        num_verts = np.zeros((ncomp,))
        for i in range(ncomp):
            num_verts[i] = np.sum(labels==i)
        
        i_max = np.argmax(num_verts)
        ind = labels==i_max

        A = self.weight_matrix[ind,:]
        A = A[:,ind]
        G = graph(A)

        return G, ind


    def eigen_decomp(self, normalization='combinatorial', method='exact', k=10, c=None, gamma=0, tol=0, q=1):
        """Eigen Decomposition of Graph Laplacian
        ======

        Computes the the low-lying eigenvectors and eigenvalues of 
        various normalizations of the graph Laplacian. Computations can 
        be either exact, or use a fast low-rank approximation via 
        randomized SVD. 

        Parameters
        ----------
        normalization : {'combinatorial','randomwalk','normalized'}, default='combinatorial'
            Type of normalization of graph Laplacian to apply.
        method : {'exact','lowrank'}, default='exact'
            Method for computing eigenvectors. 'exact' uses scipy.sparse.linalg.svds, while
            'lowrank' uses a low rank approximation via randomized SVD. Lowrank is not 
            implemented for gamma > 0.
        k : int (optional), default=10
            Number of eigenvectors to compute.
        c : int (optional), default=2*k
            Cutoff for randomized SVD.
        gamma : float (optional), default=0
            Parameter for modularity (add more details)
        tol : float (optional), default=0
            tolerance for eigensolvers.
        q : int (optional), default=1
            Exponent to use in randomized svd.

        Returns
        -------
        vals : numpy array, float 
            eigenvalues in increasing order.
        vecs : (n,k) numpy array, float
            eigenvectors as columns.

        Example
        -------
        This example compares the exact and lowrank (ranomized svd) methods for computing the spectrum: [randomized_svd.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/randomized_svd.py).
        ```py
        import numpy as np
        import matplotlib.pyplot as plt
        import sklearn.datasets as datasets
        import graphlearning as gl

        X,L = datasets.make_moons(n_samples=500,noise=0.1)
        W = gl.weightmatrix.knn(X,10)
        G = gl.graph(W)

        num_eig = 7
        vals_exact, vecs_exact = G.eigen_decomp(normalization='normalized', k=num_eig, method='exact')
        vals_rsvd, vecs_rsvd = G.eigen_decomp(normalization='normalized', k=num_eig, method='lowrank', q=50, c=50)

        for i in range(1,num_eig):
            rsvd = vecs_rsvd[:,i]
            exact = vecs_exact[:,i]

            sign = np.sum(rsvd*exact)
            if sign < 0:
                rsvd *= -1

            err = np.max(np.absolute(rsvd - exact))/max(np.max(np.absolute(rsvd)),np.max(np.absolute(exact)))

            fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
            fig.suptitle('Eigenvector %d, err=%f'%(i,err))

            ax1.scatter(X[:,0],X[:,1], c=rsvd)
            ax1.set_title('Random SVD')

            ax2.scatter(X[:,0],X[:,1], c=exact)
            ax2.set_title('Exact')

        plt.show()
        ```
        """

        #Default choice for c
        if c is None:
            c = 2*k

        same_method = self.eigendata[normalization]['method'] == method
        same_k = self.eigendata[normalization]['k'] == k
        same_c = self.eigendata[normalization]['c'] == c
        same_gamma = self.eigendata[normalization]['gamma'] == gamma
        same_tol = self.eigendata[normalization]['tol'] == tol
        same_q = self.eigendata[normalization]['q'] == q

        #If already computed, then return eigenvectors
        if same_method and same_k and same_c and same_gamma and same_tol and same_q:
        
            return self.eigendata[normalization]['eigenvalues'], self.eigendata[normalization]['eigenvectors']
        
        #Else, we need to compute the eigenvectors
        else:
            self.eigendata[normalization]['method'] = method 
            self.eigendata[normalization]['k'] = k
            self.eigendata[normalization]['c'] = c
            self.eigendata[normalization]['gamma'] = gamma
            self.eigendata[normalization]['tol'] = tol
            self.eigendata[normalization]['q'] = q

            n = self.num_nodes

            #If not using modularity
            if gamma == 0:
                
                if normalization == 'randomwalk' or normalization == 'normalized':

                    D = self.degree_matrix(p=-0.5)
                    A = D*self.weight_matrix*D

                    if method == 'exact':
                        u,s,vt = splinalg.svds(A, k=k, tol=tol)
                    elif method == 'lowrank':
                        u,s,vt = utils.randomized_svd(A, k=k, c=c, q=q)
                    else:
                        sys.exit('Invalid eigensolver method '+method)

                    vals = 1 - s
                    ind = np.argsort(vals)
                    vals = vals[ind]
                    vecs = u[:,ind]

                    if normalization == 'randomwalk':
                        vecs = D@vecs

                elif normalization == 'combinatorial':

                    L = self.laplacian()
                    deg = self.degree_vector()
                    M = 2*np.max(deg)
                    A = M*sparse.identity(n) - L

                    if method == 'exact':
                        u,s,vt = splinalg.svds(A, k=k, tol=tol)
                    elif method == 'lowrank':
                        u,s,vt = utils.randomized_svd(A, k=k, c=c, q=q)
                    else:
                        sys.exit('Invalid eigensolver method '+method)
                    
                    vals = M - s
                    ind = np.argsort(vals)
                    vals = vals[ind]
                    vecs = u[:,ind]

                else:
                    sys.exit('Invalid choice of normalization')


            #Modularity
            else:

                if method == 'lowrank':
                    sys.exit('Low rank not implemented for modularity')

                if normalization == 'randomwalk':
                    lap = self.laplacian(normalization='normalized')
                    P = self.degree_matrix(p=-0.5)
                    p1,p2 = 1.5,0.5
                else:
                    lap = self.laplacian(normalization=normalization)
                    P = sparse.identity(n)
                    p1,p2 = 1,1

                #If using modularity
                deg = self.degree_vector()
                deg1 = deg**p1
                deg2 = deg**p2
                m = np.sum(deg)/2 
                def M(v):
                    v = v.flatten()
                    return (lap*v).flatten() + (gamma/m)*(deg2.T@v)*deg1

                L = sparse.linalg.LinearOperator((n,n), matvec=M)
                vals, vecs = sparse.linalg.eigsh(L, k=k, which='SM', tol=tol)

                #Correct for random walk Laplacian if chosen
                vecs = P@vecs


            #Store eigenvectors for resuse later
            self.eigendata[normalization]['eigenvalues'] = vals
            self.eigendata[normalization]['eigenvectors'] = vecs

            return vals, vecs

    def peikonal(self, bdy_set, bdy_val=0, f=1, p=1, nl_bdy=False, u0=None, solver='fmm',
                              max_num_it=1e5, tol=1e-3, num_bisection_it=30, prog=False,):
        """p-eikonal equation 
        =====================

        Sovles the graph p-eikonal equation 
        \\[ \\sum_{j=1}^n w_{ij} (u_i - u_j)_+^p = f_i\\]
        for \\(i\\not\\in \\Gamma\\), subject to \\(u_i=g_i\\) for \\(i\\in \\Gamma\\).

        Parameters
        ----------
        bdy_set : numpy array (int or bool) 
            Indices or boolean mask indicating the boundary nodes \\(\\Gamma\\).
        bdy_val : numpy array or single float (optional), default=0
            Boundary values \\(g\\) on \\(\\Gamma\\). A single float is
            interpreted as a constant over \\(\\Gamma\\).
        f : numpy array or single float (optional), default=1
            Right hand side of the p-eikonal equation, a single float
            is interpreted as a constant vector of the graph.
        p : float (optional), default=1
            Value of exponent p in the p-eikonal equation.
        nl_bdy : bool (optional), default = False
            Whether to extend the boundary conditions to non-local ones (to graph neighbors).
        solver : {'fmm', 'gauss-seidel'}, default='fmm'
            Solver for p-eikonal equation.
        u0 : numpy array (float, optional), default=None
            Initialization of solver. If not provided, then u0=0.
        max_num_it : int (optional), default=1e5
            Maximum number of iterations for 'gauss-seidel' solver.
        tol : float (optional), default=1e-3
            Tolerance with which to solve the equation for 'gauss-seidel' solver.
        num_bisection_it : int (optional), default=30
            Number of bisection iterations for solver for 'gauss-seidel' solver with \\(p>1\\).
        prog : bool (optional), default=False
            Toggles whether to print progress information.

        Returns
        -------
        u : numpy array (float)
            Solution of p-eikonal equation.

        Example
        -------
        This example uses the peikonal equation to compute a data depth: [peikonal.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/peikonal.py).
        ```py
        import graphlearning as gl
        import numpy as np
        import matplotlib.pyplot as plt

        X = np.random.rand(int(1e4),2)
        x,y = X[:,0],X[:,1]

        eps = 0.02
        W = gl.weightmatrix.epsilon_ball(X, eps)
        G = gl.graph(W)

        bdy_set = (x < eps) | (x > 1-eps) | (y < eps) | (y > 1-eps)
        u = G.peikonal(bdy_set)

        plt.scatter(x,y,c=u,s=0.25)
        plt.scatter(x[bdy_set],y[bdy_set],c='r',s=0.5)
        plt.show() 
        ```
        """

        #Import c extensions
        from . import cextensions
        
        n = self.num_nodes

        #Set initial data
        if u0 is None:
            u = np.zeros((n,))
        else:
            u = u0.copy()

        #Convert f to an array if scalar is given
        if type(f) != np.ndarray:
            f = np.ones((n,))*f

        #Convert boundary data to standard format
        bdy_set, bdy_val = utils._boundary_handling(bdy_set, bdy_val)
        
        #Extend boundary data if nl_bdy=True
        if nl_bdy:
            D = self.degree_matrix(p=-1)
            bdy_mask = np.zeros(n)
            bdy_mask[bdy_set] = 1
            bdy_dilate = (D*self.weight_matrix*bdy_mask) > 0
            bdy_set = bdy_set = np.where(bdy_dilate)[0]
            bdy_val_all = np.zeros(n)
            bdy_val_all[bdy_mask==1] = bdy_val
            bdy_val = D*self.weight_matrix*bdy_val_all
            bdy_val = bdy_val[bdy_set]

        #Type casting and memory blocking
        u = np.ascontiguousarray(u,dtype=np.float64)
        bdy_set = np.ascontiguousarray(bdy_set,dtype=np.int32)
        f = np.ascontiguousarray(f,dtype=np.float64)
        bdy_val = np.ascontiguousarray(bdy_val,dtype=np.float64)

        if solver == 'fmm':
            cextensions.peikonal_fmm(u,self.J,self.K,self.V,bdy_set,f,bdy_val,p,num_bisection_it)
        else:
            cextensions.peikonal(u,self.J,self.K,self.V,bdy_set,f,bdy_val,p,max_num_it,tol,num_bisection_it,prog)

        return u

    def dijkstra_hl(self, bdy_set, bdy_val=0, f=1, max_dist=np.inf, return_cp=False):
        """Dijkstra's algorithm (Hopf-Lax Version)
        ======

        Solves the graph Hamilton-Jacobi equation
        \\[ \\max_j w_{ji}^{-1} (u(x_i)^2 - u(x_j)^2) = u(x_i)f_i\\]
        subject to \\(u=g\\) on \\(\\Gamma\\).

        Parameters
        ----------
        bdy_set : numpy array (int) 
            Indices or boolean mask identifying the boundary nodes \\(\\Gamma\\).
        bdy_val : numpy array (float), optional
            Boundary values \\(g\\) on \\(\\Gamma\\). A single float is
            interpreted as a constant over \\(\\Gamma\\).
        f : numpy array or scalar float, default=1
            Right hand side of eikonal equation. If a scalar, it is extended to a vector 
            over the graph.
        max_dist : float or np.inf (optional), default = np.inf
            Distance at which to terminate Dijkstra's algorithm. Nodes with distance
            greater than `max_dist` will contain the value `np.inf`.
        return_cp : bool (optional), default=False
            Whether to return closest point. Nodes with distance greater than max_dist 
            contain `-1` for closest point index.

        Returns
        -------
        dist_func : numpy array, float 
            Distance function computed via Dijkstra's algorithm.
        cp : numpy array, int 
            Closest point indices. Only returned if `return_cp=True`

        Example
        -------
        This example uses Dijkstra's algorithm to compute the distance function to a single point,
        and compares the result to a cone: [dijkstra.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/dijkstra.py).
        ```py
        import graphlearning as gl
        import numpy as np

        for n in [int(10**i) for i in range(3,6)]:

            X = np.random.rand(n,2)
            X[0,:]=[0.5,0.5]
            W = gl.weightmatrix.knn(X,50,kernel='distance')
            G = gl.graph(W)
            u = G.dijkstra([0])

            u_true = np.linalg.norm(X - [0.5,0.5],axis=1)
            error = np.linalg.norm(u-u_true, ord=np.inf)
            print('n = %d, Error = %f'%(n,error))
        ```
        """

        #Import c extensions
        from . import cextensions

        #Convert boundary data to standard format
        bdy_set, bdy_val = utils._boundary_handling(bdy_set, bdy_val)

        #Variables
        n = self.num_nodes
        dist_func = np.ones((n,))*np.inf        
        cp = -np.ones((n,),dtype=int)

        #Right hand side
        if type(f) != np.ndarray:
            f = np.ones((n,))*f

        #Type casting and memory blocking
        dist_func = np.ascontiguousarray(dist_func,dtype=np.float64)
        cp = np.ascontiguousarray(cp,dtype=np.int32)
        bdy_set = np.ascontiguousarray(bdy_set,dtype=np.int32)
        bdy_val = np.ascontiguousarray(bdy_val,dtype=np.float64)
        f = np.ascontiguousarray(f,dtype=np.float64)

        cextensions.dijkstra_hl(dist_func,cp,self.J,self.K,self.V,bdy_set,bdy_val,f,1.0,max_dist)

        if return_cp:
            return dist_func, cp
        else:
            return dist_func

    def distance(self, i, j, return_path=False, return_distance_vector=False):
        """Graph distance
        ======
        
        Computes the shortest path distance between two points. Can also return the shortest path.
        Edges are weighted by the reciprocals of the edge weights \\(w_{ij}^{-1}\\).

        Parameters
        ----------
        i : int 
            First index
        j : int 
            Second index
        return_path : bool (optional), default = False
            Whether to return optimal path.
        return_distance_vector : bool (optional), default = False
            Whether to return distance vector to node i.

        Returns 
        -------
        d : float
            Distance
        path : numpy array, int (optional)
            Indices of optimal path
        v : numpy array, float (optional)
            Distance vector to node i
        """

        v = self.dijkstra([i],reciprocal_weights=True)
        d = v[j]
        if return_path:
            p = j
            path = [p]
            while p != i:
                nn, w = self.neighbors(p, return_weights=True)
                k = np.argmin(v[nn] + w**-1)
                p = nn[k]
                path += [p]
            path = np.array(path)
            if return_distance_vector:
                return d,path,v
            else:
                return d,path
        else:
            if return_distance_vector:
                return d,v
            else:
                return d

    def distance_matrix(self, centered=False):
        """Graph distance matrix
        ======
        
        Computes the shortest path distance between all pairs of points in the graph.
        Edges are weighted by the reciprocals of the edge weights \\(w_{ij}^{-1}\\).

        Parameters
        -------
        centered : bool (optional), default=False
            Whether to center the distance matrix, as in ISOMAP.

        Returns 
        -------
        T : numpy array, float
            Distance matrix
        """

        n = self.num_nodes
        T = np.zeros((n,n))
        for i in range(n):
            d,T[i,:] = self.distance(i,i,return_distance_vector=True)

        if centered:
            J = np.eye(n)  - (1/n)*np.ones((n,n))
            T = -0.5*J@T@J

        return T

    def dijkstra(self, bdy_set, bdy_val=0, f=1, max_dist=np.inf, return_cp=False, reciprocal_weights=False):
        """Dijkstra's algorithm
        ======

        Computes a graph distance function with Dijkstra's algorithm. The graph distance is
        \\[ d(x,y) = \\min_p \\sum_{i=1}^M w_{p_i,p_{i+1}}f_{p_{i+1}},\\]
        where the minimum is over paths \\(p\\) connecting \\(x\\) and \\(y\\), \\(w_{ij}\\) is 
        the weight from \\(i\\) to \\(j\\), and \\(f_i\\) is an additional per-vertex weights. 
        A path must satisfy \\(w_{p_i,p_{i+1}}>0\\) for all \\(i\\). Dijkstra's algorithm returns the
        distance function to a terminal set \\(\\Gamma\\), given by
        \\[u(x) = \\min_{i\\in \\Gamma} \\{g(x_i) + d(x,x_i)\\},\\]
        where \\(g\\) are boundary values.
        An optional feature also returns the closest point information
        \\[cp(x) = \\text{argmin}_{i\\in \\Gamma} \\{g(x_i) + d(x,x_i)\\}.\\]
        We note that the distance function \\(u\\) can also be interpreted as the solution of the
        graph eikonal equation
        \\[ \\max_j w_{ji}^{-1} (u(x_i) - u(x_j)) = f_i\\]
        subject to \\(u=g\\) on \\(\\Gamma\\).

        Parameters
        ----------
        bdy_set : numpy array (int) 
            Indices or boolean mask identifying the boundary nodes \\(\\Gamma\\).
        bdy_val : numpy array (float), optional
            Boundary values \\(g\\) on \\(\\Gamma\\). A single float is
            interpreted as a constant over \\(\\Gamma\\).
        f : numpy array or scalar float, default=1
            Right hand side of eikonal equation. If a scalar, it is extended to a vector 
            over the graph.
        max_dist : float or np.inf (optional), default = np.inf
            Distance at which to terminate Dijkstra's algorithm. Nodes with distance
            greater than `max_dist` will contain the value `np.inf`.
        return_cp : bool (optional), default=False
            Whether to return closest point. Nodes with distance greater than max_dist 
            contain `-1` for closest point index.
        reciprocal_weights : bool (optional), default=False
            Whether to use the reciprocals of the weights \\(w_{ij}^{-1}\\) in the definition of 
            graph distance. 

        Returns
        -------
        dist_func : numpy array, float 
            Distance function computed via Dijkstra's algorithm.
        cp : numpy array, int 
            Closest point indices. Only returned if `return_cp=True`

        Example
        -------
        This example uses Dijkstra's algorithm to compute the distance function to a single point,
        and compares the result to a cone: [dijkstra.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/dijkstra.py).
        ```py
        import graphlearning as gl
        import numpy as np

        for n in [int(10**i) for i in range(3,6)]:

            X = np.random.rand(n,2)
            X[0,:]=[0.5,0.5]
            W = gl.weightmatrix.knn(X,50,kernel='distance')
            G = gl.graph(W)
            u = G.dijkstra([0])

            u_true = np.linalg.norm(X - [0.5,0.5],axis=1)
            error = np.linalg.norm(u-u_true, ord=np.inf)
            print('n = %d, Error = %f'%(n,error))
        ```
        """

        #Import c extensions
        from . import cextensions

        #Convert boundary data to standard format
        bdy_set, bdy_val = utils._boundary_handling(bdy_set, bdy_val)

        #Variables
        n = self.num_nodes
        dist_func = np.ones((n,))*np.inf        
        cp = -np.ones((n,),dtype=int)

        #Right hand side
        if type(f) != np.ndarray:
            f = np.ones((n,))*f

        #Type casting and memory blocking
        dist_func = np.ascontiguousarray(dist_func,dtype=np.float64)
        cp = np.ascontiguousarray(cp,dtype=np.int32)
        bdy_set = np.ascontiguousarray(bdy_set,dtype=np.int32)
        bdy_val = np.ascontiguousarray(bdy_val,dtype=np.float64)
        f = np.ascontiguousarray(f,dtype=np.float64)

        if reciprocal_weights:
            cextensions.dijkstra(dist_func,cp,self.J,self.K,self.Vinv,bdy_set,bdy_val,f,1.0,max_dist)
        else:
            cextensions.dijkstra(dist_func,cp,self.J,self.K,self.V,bdy_set,bdy_val,f,1.0,max_dist)

        if return_cp:
            return dist_func, cp
        else:
            return dist_func

    def plaplace(self, bdy_set, bdy_val, p, tol=1e-1, max_num_it=1e6, prog=False, fast=True):
        """Game-theoretic p-Laplacian
        ======

        Computes the solution of the game-theoretic p-Laplace equation \\(L_p u_i=0\\) 
        for \\(i\\not\\in \\Gamma\\), subject to \\(u_i=g_i\\) for \\(i\\in \\Gamma\\).
        The game-theoretic p-Laplacian is given by
        \\[ L_p u = \\frac{1}{p}L_{\\rm randomwalk} + \\left(1-\\frac{2}{p}\\right)L_\\infty u,\\]
        where \\(L_{\\rm randomwalk}\\) is the random walk graph Laplacian and \\(L_\\infty\\) is the
        graph infinity-Laplace operator, given by
        \\[ L_\\infty u_i = \\min_j w_{ij}(u_i-u_j) + \\max_j w_{ij} (u_i-u_j).\\]

        Parameters
        ----------
        bdy_set : numpy array (int or bool) 
            Indices or boolean mask indicating the boundary nodes \\(\\Gamma\\).
        bdy_val : numpy array or single float (optional), default=0
            Boundary values \\(g\\) on \\(\\Gamma\\). A single float is
            interpreted as a constant over \\(\\Gamma\\).
        p : float
            Value of \\(p\\).
        tol : float (optional), default=1e-1
            Tolerance with which to solve the equation.
        max_num_it : int (optional), default=1e6
            Maximum number of iterations.
        prog : bool (optional), default=False
            Toggles whether to print progress information.
        fast : bool (optional), default=True
            Whether to use constant \\(w_{ij}=1\\) weights for the infinity-Laplacian
            which allows a faster algorithm to be used.

        Returns
        -------
        u : numpy array, float 
            Solution of graph p-Laplace equation.

        Example
        -------
        This example uses the p-Laplace equation to interpolate boundary values: [plaplace.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/plaplace.py).
        ```py
        import graphlearning as gl
        import numpy as np
        import matplotlib.pyplot as plt

        X = np.random.rand(int(1e4),2)
        x,y = X[:,0],X[:,1]

        eps = 0.02
        W = gl.weightmatrix.epsilon_ball(X, eps)
        G = gl.graph(W)

        bdy_set = (x < eps) | (x > 1-eps) | (y < eps) | (y > 1-eps)
        bdy_val = x**2 - y**2

        u = G.plaplace(bdy_set, bdy_val[bdy_set], p=10)

        plt.scatter(x,y,c=u,s=0.25)
        plt.scatter(x[bdy_set],y[bdy_set],c='r',s=0.5)
        plt.show()
        ```
        """
            
        #Import c extensions
        from . import cextensions
        
        n = self.num_nodes
        alpha = 1/(p-1)
        beta = 1-alpha

        #Convert boundary data to standard format
        bdy_set, bdy_val = utils._boundary_handling(bdy_set, bdy_val)

        #If fast solver
        if fast:

            u = np.zeros((n,))        #Initial condition

            #Type casting and memory blocking
            u = np.ascontiguousarray(u,dtype=np.float64)
            bdy_set = np.ascontiguousarray(bdy_set,dtype=np.int32)
            bdy_val = np.ascontiguousarray(bdy_val,dtype=np.float64)

            weighted = False
            tol = 1e-6
            cextensions.lip_iterate(u,self.J,self.I,self.V,bdy_set,bdy_val,max_num_it,tol,float(prog),float(weighted),float(alpha),float(beta))
        else:
            uu = np.max(bdy_val)*np.ones((n,))
            ul = np.min(bdy_val)*np.ones((n,))

            #Set labels
            uu[bdy_set] = bdy_val
            ul[bdy_set] = bdy_val

            #Type casting and memory blocking
            uu = np.ascontiguousarray(uu,dtype=np.float64)
            ul = np.ascontiguousarray(ul,dtype=np.float64)
            bdy_set = np.ascontiguousarray(bdy_set,dtype=np.int32)
            bdy_val = np.ascontiguousarray(bdy_val,dtype=np.float64)

            cextensions.lp_iterate(uu,ul,self.J,self.I,self.V,bdy_set,bdy_val,p,float(max_num_it),float(tol),float(prog))
            u = (uu+ul)/2

        return u

    def amle(self, bdy_set, bdy_val, tol=1e-5, max_num_it=1000, weighted=True, prog=False):
        """Absolutely Minimal Lipschitz Extension (AMLE)
        ======

        Computes the absolutely minimal Lipschitz extension (AMLE) of boundary values on a graph.
        The AMLE is the solution of the graph infinity Laplace equation
        \\[ \\min_j w_{ij}(u_i-u_j) + \\max_j w_{ij} (u_i-u_j) = 0\\]
        for \\(i\\not\\in \\Gamma\\), subject to \\(u_i=g_i\\) for \\(i\\in \\Gamma\\).

        Parameters
        ----------
        bdy_set : numpy array (int) 
            Indices of boundary nodes \\(\\Gamma\\).
        bdy_val : numpy array (float)
            Boundary values \\(g\\) on \\(\\Gamma\\).
        tol : float (optional), default=1e-5
            Tolerance with which to solve the equation.
        max_num_it : int (optional), default=1000
            Maximum number of iterations.
        weighted : bool (optional), default=True
            When set to False, the weights are converted to a 0/1 adjacency matrix,
            which allows for a much faster solver.
        prog : bool (optional), default=False
            Toggles whether to print progress information.

        Returns
        -------
        u : numpy array, float 
            Absolutely minimal Lipschitz extension.
        """

        #Import c extensions
        from . import cextensions

        #Variables
        n = self.num_nodes
        u = np.zeros((n,))        #Initial condition
        max_num_it = float(max_num_it)
        alpha = 0
        beta = 1

        #Convert boundary data to standard format
        bdy_set, bdy_val = utils._boundary_handling(bdy_set, bdy_val)

        #Type casting and memory blocking
        u = np.ascontiguousarray(u,dtype=np.float64)
        bdy_set = np.ascontiguousarray(bdy_set,dtype=np.int32)
        bdy_val = np.ascontiguousarray(bdy_val,dtype=np.float64)

        cextensions.lip_iterate(u,self.J,self.I,self.V,bdy_set,bdy_val,max_num_it,tol,float(prog),float(weighted),float(alpha),float(beta))

        return u


    def save(self, filename):
        """Save
        ======

        Saves the graph and all its attributes to a file.

        Parameters
        ----------
        filename : string
            File to save graph to, without any extension.
        """

        filename += '.pkl'
        with open(filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


    def load(filename):
        """Load
        ======

        Load a graph from a file.

        Parameters
        ----------
        filename : string
            File to load graph from, without any extension.
        """

        filename += '.pkl'
        with open(filename, 'rb') as inp:
            G = pickle.load(inp)
        G.__ccode_init__()
        return G


    def page_rank(self,alpha=0.85,v=None,tol=1e-10):
        """PageRank
        ======

        Solves for the PageRank vector, which is the solution of the PageRank equation
        \\[ (I - \\alpha P)u = (1-\\alpha) v, \\]
        where \\(P = W^T D^{-1}\\) is the probability transition matrix, with \\(D\\) the diagonal
        degree matrix, \\(v\\) is the teleportation distribution, and \\(\\alpha\\) is the 
        teleportation paramter. Solution is computed with the power iteration
        \\[ u_{k+1} = \\alpha P u_k + (1-\\alpha) v.\\]

        Parameters
        ----------
        alpha : float (optional), default=0.85
            Teleportation parameter.
        v : numpy array (optional), default=None
            Teleportation distribution. Default is the uniform distribution.
        tol : float (optional), default=1e-10
            Tolerance with which to solve the PageRank equation.

        Returns
        -------
        u : numpy array, float 
            PageRank vector.
        """

        n = self.num_nodes

        u = np.ones((n,))/n
        if v is None:
            v = np.ones((n,))/n

        D = self.degree_matrix(p=-1)
        P = self.weight_matrix.T@D

        err = tol+1
        while err > tol:
            w = alpha*P@u + (1-alpha)*v
            err = np.max(np.absolute(w-u))
            u = w.copy()

        return u

    def draw(self,X=None,c=None,cmap='viridis',markersize=None,linewidth=None,edges=True,linecolor='black'):
        """Draw Graph
        ======

        Draws a planar representation of a graph using metric MDS. 

        Parameters
        ----------
        X : (n,2) numpy array (optional)
            Coordinates of graph vertices to draw. If not provided, uses metric MDS.
        c : (n,) numpy array (optional)
            Colors of vertices. If not provided, vertices are colored black.
        cmap : string (optional)
            Colormap. Default is 'viridis'.
        markersize : float (optional)
            Markersize.
        linewidth : float (optional)
            Linewidth.
        edges : bool (optional)
            Whether to plot edges (default=True)
        linecolor : string (optional)
            Color for lines (default='black')

        Parameters
        ----------
        X : (n,2) numpy array
            Returns coordinates of points.

        """

        plt.figure()
        n = self.num_nodes

        #If points are not provided, we use metric MDS
        if X is None:
            #J = np.eye(n) - (1/n)*np.ones((n,n))
            #dist = np.zeros((n,n))
            #for i in range(n):
            #    dist[i,:] = self.dijkstra([i])
            #H = -(1/2)*J@dist@J
            H = self.distance_matrix(centered=True)

            #Need to sort eigenvalues, since H may not be positive semidef
            vals,V = sparse.linalg.eigsh(H,k=10,which='LM')
            ind = np.argsort(-vals)
            V = V[:,ind]
            vals = vals[ind]

            #Get top eigenvectors and square roots of positive parts of eigenvalues
            P = V[:,:2]
            S = np.maximum(vals[:2],0)**(1/2)

            #MDS embedding
            X = P@np.diag(S)

        #Plot points
        x,y = X[:,0],X[:,1]
        if c is None:
            if markersize is None:
                plt.scatter(x,y,zorder=2)
            else:
                plt.scatter(x,y,s=markersize,zorder=2)
        else:
            if markersize is None:
                plt.scatter(x,y,c=c,cmap=cmap,zorder=2)
            else:
                plt.scatter(x,y,c=c,cmap=cmap,s=markersize,zorder=2)

        #Draw edges
        if edges:
            for i in range(n):
                nn = self.weight_matrix[i,:].nonzero()[1]
                for j in nn:
                    if linewidth is None:
                        plt.plot([x[i],x[j]],[y[i],y[j]],color=linecolor,zorder=0)
                    else:
                        plt.plot([x[i],x[j]],[y[i],y[j]],color=linecolor,linewidth=linewidth,zorder=0)

        return X

    def ars(X, dim=2, perplexity=30, kappa=0.5, iters=1000, time_step=1, theta1=2,
            theta2=3, alpha=10, num_early=250, use_pca=True, init_dim=50, prog = False):
        """Attraction-Repulsion Swarming t-SNE
        ======

        Computes a low dimensional embedding (visualization) of a graph or data set using the Attraction-Repulsion Swarming method of [1]. Uses the Barnes-Hut approximation.

        Parameters
        ----------
        X : numpy array (float) 
            Data matrix, rows are data points.
        dim : int (optional, default=2)
            Dimension of embedding (usually 2 or 3).
        perplexity : float (optional, default=30.0)
            Perplexity for graph construction.
        kappa : float (optional, default = 0.5)
            Parameter for Barnes-Hut tree decomposition.
        iters : int (optional, default=1000)
            Number of iterations.
        time_step : float (optional, default=1.0)
            Time step for ARS iterations.
        theta1 : float (optional, default = 2.0)
            Attraction scaling exponent.
        theta2 : float (optional, default = 3.0)
            Repulsion scaling exponent. 
        alpha : float (optional, default = 10.0)
            Early exaggeration factor.
        num_early : int (optional, default = 250)
            Number of early exaggeration iterations.
        use_pca : bool (optional, default = true)
            Whether to use PCA to reduce the dimension to d=init_dim.
        init_dim : int (optional, default = 50)
            PCA dimension.
        prog : bool (optional, default = False)
            Whether to print out progress.


        Returns
        -------
        Y : numpy array, float 
            Matrix whose rows are the embedded points.

        Example
        -------
        This example uses ARS t-SNE to visualize the MNIST data set: [ars_tsne.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/ars_tsne.py).
        ```py
        import graphlearning as gl 
        import numpy as np
        import matplotlib.pyplot as plt

        #Load the MNIST data
        data,labels = gl.datasets.load('mnist')

        #In order to run the code more quickly, 
        #you may want to subsample MNIST. 
        size = 70000
        if size < data.shape[0]: #If less than 70000
            ind = np.random.choice(data.shape[0], size=size, replace=False)
            data = data[ind,:]
            labels = labels[ind]

        #Run ARS t-SNE and plot the result
        Y = gl.graph.ars(data, prog=True)
        plt.scatter(Y[:,0],Y[:,1],c=labels,s=1)
        plt.show()
        ```

        References
        ----------
        [1] J. Lu, J. Calder. [Attraction-Repulsion Swarming: A Generalized Framework of t-SNE via Force Normalization and Tunable Interactions](https://arxiv.org/abs), Submitted, 2024.

        """

        #Import c extensions
        from . import cextensions

        if use_pca and (X.shape[1] > init_dim):
            X = X - np.mean(X, axis=0)
            vals, Q = sparse.linalg.eigsh(X.T@X, k=init_dim, which='LM')
            X = X@Q

        #Type casting and memory blocking
        X = np.ascontiguousarray(X,dtype=np.float64)
        Y = np.zeros((X.shape[0],dim),dtype=float)
        Y = np.ascontiguousarray(Y,dtype=np.float64)

        cextensions.ars(X,Y,dim,perplexity,kappa,iters,time_step,theta1,theta2,alpha,num_early,prog)

        return Y





