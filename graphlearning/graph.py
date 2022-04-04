"""
Graph Class
========================

This module contains the `graph` class, which implements many graph-based algorithms, including
spectral decompositions, distance functions (via Dijkstra and peikonal), PageRank, AMLE (Absolutely 
Minimal Lipschitz Extensions), p-Laplace equations, and basic calculus on graphs.
"""

import numpy as np
from scipy import sparse
from scipy import spatial
import scipy.sparse.linalg as splinalg
import scipy.sparse.csgraph as csgraph
import time
import sys
import pickle

from . import utils

class graph:

    def __init__(self, W):
        """Graph class
        ========

        A class for graphs, including routines to compute Laplacians and their
        eigendecompositions, which are useful in graph learning.

        Parameters
        ----------
        W : (n,n) numpy array, matrix, or scipy sparse matrix
            Weight matrix representing the graph.
        """

        self.weight_matrix = sparse.csr_matrix(W)
        self.num_nodes = W.shape[0]

        #Coordinates of sparse matrix for passing to C code
        I,J,V = sparse.find(W)
        self.I = I
        self.J = J
        self.V = V
        K = np.array((J[1:] - J[:-1]).nonzero()) + 1
        self.K = np.append(0,np.append(K,len(J)))

        #For passing to C code
        self.I = np.ascontiguousarray(self.I, dtype=np.int32)
        self.J = np.ascontiguousarray(self.J, dtype=np.int32)
        self.V = np.ascontiguousarray(self.V, dtype=np.float64)
        self.K = np.ascontiguousarray(self.K, dtype=np.int32)

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

     
    def reweight(self, idx, method='poisson', X=None, alpha=2, zeta=1e7, r=0.1):
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
            f -= np.mean(f)

            L = self.laplacian()
            w = utils.conjgrad(L, f, tol=1e-5)
            w -= np.min(w)
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
            Method for computing eigenvectors. 'exact' uses scipy.sparse.linalg.eigs, while
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

    def fiedler_vector(self, method='exact', tol=0):
        """Fiedler vector
        ======

        Computes the Fiedler vector, which is the second eigenvector for the 
        combinatorial graph Laplacian \\(L = D-W\\).

        Parameters
        ----------
        method : {'exact','lowrank'}, default='exact'
            Method for computing eigenvectors. 'exact' uses scipy.sparse.linalg.eigs, while
            'lowrank' uses a low rank approximation via randomized SVD.
        tol : float (optional), default=0
            tolerance for eigensolvers.


        Returns
        -------
        fiedler_vector : numpy array, float 
            Contents of fiedler vector.
        """

        vals, vecs = self.eigen_decomp(normalization='combinatorial', method=method, k=2, tol=tol)
        fiedler_vector = vecs[:,1]

        return fiedler_vector

    def peikonal(self, bdy_set, bdy_val=0, f=1, p=1, u0=None, solver='fmm', max_num_it=1e5, 
                                                     tol=1e-3, num_bisection_it=30, prog=False,):
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

        #Type casting and memory blocking
        u = np.ascontiguousarray(u,dtype=np.float64)
        bdy_set = np.ascontiguousarray(bdy_set,dtype=np.int32)
        f = np.ascontiguousarray(f,dtype=np.float64)
        bdy_val = np.ascontiguousarray(bdy_val,dtype=np.float64)

        if solver == 'fmm':
            cextensions.peikonal_fmm(u,self.I,self.K,self.V,bdy_set,f,bdy_val,p,num_bisection_it)
        else:
            cextensions.peikonal(u,self.I,self.K,self.V,bdy_set,f,bdy_val,p,max_num_it,tol,num_bisection_it,prog)

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

        cextensions.dijkstra_hl(dist_func,cp,self.I,self.K,self.V,bdy_set,bdy_val,f,1.0,max_dist)

        if return_cp:
            return dist_func, cp
        else:
            return dist_func


    def dijkstra(self, bdy_set, bdy_val=0, f=1, max_dist=np.inf, return_cp=False):
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

        cextensions.dijkstra(dist_func,cp,self.I,self.K,self.V,bdy_set,bdy_val,f,1.0,max_dist)

        if return_cp:
            return dist_func, cp
        else:
            return dist_func

    def plaplace(self, bdy_set, bdy_val, p, tol=1e-1, max_num_it=1e6, prog=False):
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

        #Convert boundary data to standard format
        bdy_set, bdy_val = utils._boundary_handling(bdy_set, bdy_val)

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

        cextensions.lp_iterate(uu,ul,self.I,self.J,self.V,bdy_set,bdy_val,p,float(max_num_it),float(tol),float(prog))
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
        k = len(bdy_set)
        u = np.zeros((n,))        #Initial condition
        max_num_it = float(max_num_it)

        #Type casting and memory blocking
        u = np.ascontiguousarray(u,dtype=np.float64)
        bdy_set = np.ascontiguousarray(bdy_set,dtype=np.int32)
        bdy_val = np.ascontiguousarray(bdy_val,dtype=np.float64)

        cextensions.lip_iterate(u,self.I,self.J,self.V,bdy_set,bdy_val,max_num_it,tol,float(prog),float(weighted))

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



