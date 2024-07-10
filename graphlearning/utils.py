"""
Utilities
==========

This module implements several useful functions that are used throughout the package.
"""

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy import spatial
import matplotlib.pyplot as plt
import ssl, os, urllib.request, sys, re, csv

from . import weightmatrix
from . import graph

def boundary_statistic(X, r, knn=False, return_normals=False, second_order=True, cutoff=True, knn_data=None):
    """Boundary test statistic
    ===================

    Computes the boundary test statistics from [1] for identifying the boundary of a point cloud.

    Parameters
    ----------
    X : (n,d) numpy array (float)
        Point cloud in dimension d.
    r : float or int,
        Radius for test (or numgber of neighbors if knn=True)
    knn : bool (optional), default=False
        Whether to ues the k-nearest neighbor version of the test, or the radius search version.
    return_normals : bool (optional), default=False
        Wehther to return estimated normal vectors as well.
    second_order : bool (optional), default=True
        Whether to use the second order version of the test.
    cutoff : bool (optional), default=True
        Whether to use the cutoff for the second order test.
    knn_data : tuple (optional), default=None
        Output of `weightmatrix.knnsearch`, which can be provided to accelerate the computation.

    Returns
    -------
    T : numpy array
        Test statistic as a length n numpy array.
    nu : (n,d) numpy array
        Estimated normals, if `return_normals=True`.

    References
    ---------
    [1] J. Calder, S. Park, and D. Slepčev. [Boundary Estimation from Point Clouds: Algorithms, Guarantees and Applications.](https://arxiv.org/abs/2111.03217) arXiv:2111.03217, 2021.
    """

    #Estimation of normal vectors
    n = X.shape[0]
    d = X.shape[1]
    if knn:
        k = r
        #Run knnsearch only if knn_data is not provided
        if knn_data is None:
            J,D = weightmatrix.knnsearch(X,k)
        else:
            J,D = knn_data
        W = weightmatrix.knn(X, k, kernel='uniform', symmetrize=False, knn_data=(J,D))
    else:
        W = weightmatrix.epsilon_ball(X, r, kernel='uniform')
        
    deg = W*np.ones(n)
    if np.min(deg)==1:
        print('\nWarning: Some points have no neighbors!!!\n')

    #Estimation of normals
    if second_order:
        if knn:
            theta = graph.graph(W).degree_matrix(p=-1)
        else:
            W2 = weightmatrix.epsilon_ball(X, r/2, kernel='uniform')
            theta = graph.graph(W).degree_matrix(p=-1)
        nu = -graph.graph(W*theta).laplacian()*X
    else:
        nu = -graph.graph(W).laplacian()*X

    #Normalize to unit norm
    norms = np.sqrt(np.sum(nu*nu,axis=1))
    nu = nu/norms[:,np.newaxis]

    #Switch to knn if not selected
    if not knn:
        k = int(np.max(W*np.ones(W.shape[0]))) #Number of neighbors to use in knnsearch
        J,D = weightmatrix.knnsearch(X,k); J=J[:,1:]; D=D[:,1:] #knnsearch and remove self

    #Difference between center point and neighbors
    V = X[:,np.newaxis,:] - X[J] #(x^0-x^i), nxkxd array

    #Compute boundary statistic to all neighbors
    if second_order:
        nu2 = (nu[:,np.newaxis,:] + nu[J])/2
        if cutoff:
            nn_mask = np.sum(nu[:,np.newaxis,:]*nu[J],axis=2) > 0
            nn_mask = nn_mask[:,:,np.newaxis]
            nu2 = nn_mask*nu2 + (1-nn_mask)*nu[:,np.newaxis,:]
        xd = np.sum(V*nu2,axis=2) #xd coordinate (nxk)
    else: #First order boundary test 
        xd = np.sum(V*nu[:,np.newaxis,:],axis=2) #xd coordinate (nxk)

    #Return test statistic, masking out to B(x,r), and normals if return_normals=True
    if knn:
        T = np.max(xd,axis=1)
    else:
        T = np.max(xd*(D<=r),axis=1)

    if return_normals:
        return T,nu
    else:
        return T


def class_priors(labels):
    """Class priors
    ======

    Computes class priors (fraction of data in each class). Ignores labels that are negative.

    Parameters
    ----------
    labels : numpy array (int)
        Labels as integers \\(0,1,\\dots,k-1\\), where \\(k\\) is the number of classes.

    Returns
    -------
    class_priors : numpy array 
        Fraction of data in each class
    """
    L = np.unique(labels)
    L = L[L>=0]    

    k = len(L)
    n = np.sum(labels>=0)
    class_priors = np.zeros((k,))
    for i in range(k):
        class_priors[i] = np.sum(labels==L[i])/n

    return class_priors

def _boundary_handling(bdy_set,bdy_val):
    """Boundary value handling
    ======

    Converts boundary values from boolean or scalar to numpy arrays.

    Parameters
    ----------
    bdy_set : numpy array (int or bool), or list
        Indices of boundary nodes \\(\\Gamma\\) or boolean mask of boundary.
    bdy_val : numpy array or single float (optioanl), default=0
        Boundary values \\(g\\) on \\(\\Gamma\\). A single float is
        interpreted as a constant over \\(\\Gamma\\).

    Returns
    -------
    bdy_set : numpy array 
        Indices of boundary points.
    bdy_val : numpy array
        Array of boundary values.
    """
    if type(bdy_set) == list:
        bdy_set = np.array(bdy_set)
    if bdy_set.dtype == bool:  #If bdy_set is boolean
        bdy_set = np.where(bdy_set)[0]
    m = len(bdy_set)
    if type(bdy_val) != np.ndarray:
        bdy_val = np.ones((m,))*bdy_val

    return bdy_set, bdy_val


def csvread(filename):
    """CSV Read
    ======

    Reads numerical data from a csv file.

    Parameters
    ----------
    filename : String
        Name of csv file

    Returns
    -------
    X : numpy array 
        Contents of csv file
    """
    
    X = [] 
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        n = 0
        for row in csv_reader:
            #Skip if the row has letters
            if not row[0].lower().islower():
                X += [float(i) for i in row]
                m = len(row)
                n += 1

    X = np.array(X).reshape((n,m))

    return X


#This is to santize url names so they agree with the specific cases on github
def _sanitize_pathname(name):
    name = re.sub('mnist', 'MNIST', name, flags=re.IGNORECASE)
    name = re.sub('fashionmnist', 'FashionMNIST', name, flags=re.IGNORECASE)
    name = re.sub('cifar', 'cifar', name, flags=re.IGNORECASE)
    name = re.sub('webkb', 'WEBKB', name, flags=re.IGNORECASE)
    name = re.sub('mult', 'Mult', name, flags=re.IGNORECASE)
    name = re.sub('modrate', 'ModRate', name, flags=re.IGNORECASE)
    return name

def numpy_load(file, field):
    """Load an array from a numpy file
    ======

    Loads a numpy .npz file and returns a specific field.

    Parameters
    ----------
    file : string
        Namename of .npz file
    field : string
        Name of field to load
    """

    try:
        M = np.load(file,allow_pickle=True)
        d = M[field]
    except:
        sys.exit('Error: Cannot open '+file+'.')

    return d

def download_file(url, file):
    """Download a file from a url
    ======

    Attemps to download from a url. 

    Parameters
    ----------
    url : string 
        Web address of file to download.
    file : string
        Name of file to download to.
    """

    ssl._create_default_https_context = ssl._create_unverified_context
    url = _sanitize_pathname(url)
    try:
        print('Downloading '+url+' to '+file+'...')
        urllib.request.urlretrieve(url, file)
    except:
        sys.exit('Error: Cannot download '+url+'.')

def sparse_max(A,B):
    """Max of two sparse matrices
    ======

    Computes the elementwise max of two sparse matrices.
    Matrices should both be nonegative and square.

    Parameters
    ----------
    A : (n,n) scipy sparse matrix
        First matrix.
    B : (n,n) scipy sparse matrix
        Second matrix.

    Returns
    -------
    C : (n,n) scipy sparse matrix
        Sparse max of A and B
    """

    I = (A + B) > 0
    IB = B>A
    IA = I - IB
    return A.multiply(IA) + B.multiply(IB)

def torch_sparse(A):
    """Torch sparse matrix, from scipy sparse
    ======

    Converts a scipy sparse matrix into a torch sparse matrix.

    Parameters
    ----------
    A : (n,n) scipy sparse matrix
        Matrix to convert to torch sparse

    Returns
    -------
    A_torch : (n,n) torch.sparse.FloatTensor
        Sparse matrix in torch form.
    """

    import torch

    A = A.tocoo()
    values = A.data
    indices = np.vstack((A.row, A.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    A_torch = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return A_torch

#Constrained linear solve
#Solves Lu = f subject to u(I)=g
def constrained_solve(L,I,g,f=None,x0=None,tol=1e-10):
    """Constrained Solve
    ======

    Uses preconditioned [Conjugate Gradient Method](https://en.wikipedia.org/wiki/Conjugate_gradient_method) 
    to solve the equation \\(Lx=f\\) subject to \\(x=g\\) on a contraint set. \\(L\\) must be positive
    definite and symmetric.

    Parameters
    ----------
    L : (n,n) numpy array or scipy sparse matrix
        Left hand side of linear equation.
    I : numpy array (bool or int)
        Indices of contraint set.
    g : numpy array (float)
        Constrained values
    f : numpy array (optional), default=None
        Right hand side of linear equation. Default is interpreted as \\(f=0\\).
    x0 : numpy array (optional), default=None
        Initial condition. Default is zero.
    tol : float (optional), default = 1e-10
        Tolerance for the conjugate gradient method.

    Returns
    -------
    x : numpy array
        Solution of linear equation with constraints.
    """

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
    

    #Conjugate gradient with Jacobi preconditioner
    m = A.shape[0]
    M = A.diagonal()
    M = sparse.spdiags(1/(M+1e-10),0,m,m).tocsr()

    if x0 is None:
        v,i = sparse.linalg.cg(A,b,tol=tol,M=M)
    else:
        v,i = sparse.linalg.cg(A,b,x0=x0[idx],tol=tol,M=M)

    #Add labels back into array
    u = np.ones((n,))
    u[idx] = v
    u[I] = g

    return u

def dirichlet_eigenvectors(L,ind,k):
    """Dirichlet eigenvectors
    ======

    Finds the smallest magnitude Dirichlet eigenvectors/eigenvalues of a symmetric matrix \\(L\\), which satisfy 
    \\(x_i=0\\) for \\(i\\in \\Gamma\\) and \\(Lx_i=\\lambda x_i\\) for \\(i\\not\\in \\Gamma\\).

    Parameters
    ----------
    L : (n,n) numpy array or scipy sparse matrix
        Matrix to compute eigenvectors of.
    ind : numpy array (bool or int)
        Indices or boolean mask indicating contraint set \\(\\Gamma\\).
    k : int 
        Number of eigenvectors to find.

    Returns
    -------
    vals : numpy array
        Eigenvalues in increasing order.
    vecs : numpy array
        Corresponding eigenvectors as columns.
    """


    L = L.tocsr()
    n = L.shape[0]

    #Locations of labels
    idx = np.full((n,), True, dtype=bool)
    idx[ind] = False

    #Left hand side matrix
    A = L[idx,:]
    A = A[:,idx]
    
    #Eigenvector solver
    vals, vec = sparse.linalg.eigsh(A,k=k,which='SM')
    
    #Add labels back into array
    vecs = np.zeros((n,k))
    vecs[idx,:] = vec

    if k == 1:
        vecs = vecs.flatten()

    return vals, vecs


def constrained_solve_gmres(L,f,R,g,ind,tol=1e-5):
    """Constrained GMRES Solve
    ======

    Uses preconditioned [GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method) to solve
    the equation \\(Lx=f\\) subject to \\(Rx=g\\) on a contraint set.

    Parameters
    ----------
    L : (n,n) numpy array or scipy sparse matrix
        Left hand side of linear equation.
    f : (n,1) numpy array
        Right hand side of linear equation.
    R : (n,n) numpy array or scipy sparse matrix
        Constraint matrix.
    g : numpy array
        Length n numpy array for boundary constriants.
    ind : numpy array (bool or int)
        Indices or boolean mask indicating contraint set.
    tol : float (optional), default = 1e-5
        Tolerance for GMRES.

    Returns
    -------
    x : numpy array
        Solution of linear equation with constraints.
    """

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
    u,info = sparse.linalg.gmres(A, b, M=M, tol=tol)

    return u

def conjgrad(A, b, x0=None, max_iter=1e5, tol=1e-10):
    """Conjugate Gradient Method
    ======

    Conjugate gradient method for solving the linear equation
    \\[ Ax = b\\]
    where \\(A\\) is \\(n\\times n\\), and \\(x\\) and \\(b\\) are \\(n\\times m\\).

    Parameters
    ----------
    A : (n,n) numpy array or scipy sparse matrix
        Left hand side of linear equation.
    b : (n,k) numpy array
        Right hand side of linear equation.
    x0 : (n,k) numpy array (optional)
        Initial guess. If not provided, then x0=0.
    max_iter : int (optional), default = 1e5
        Maximum number of iterations.
    tol : float (optional), default = 1e-10
        Tolerance for stopping conjugate gradient iterations.

    Returns
    -------
    x : (n,k) numpy array
        Solution of \\(Ax=b\\) with conjugate gradient
    """

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    r = b - A@x
    p = r.copy()
    rsold = np.sum(r**2,axis=0)
  
    err = 1 
    i = 0
    while (err > tol) and (i < max_iter):
        i += 1
        Ap = A@p
        alpha = rsold / np.sum(p*Ap,axis=0)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.sum(r**2,axis=0)
        err = np.sqrt(np.sum(rsnew)) 
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x



def labels_to_onehot(labels, k, standardize=False):
    """Onehot labels
    ======

    Converts numerical labels to one hot vectors.

    Parameters
    ----------
    labels : numpy array, int
        Labels as integers.
    k : int
        Number of classes.
    standardize : bool (optional), default=False
        Whether to map labels to 0,1,...,k-1 first, before encoding.

    Returns
    -------
    onehot_labels : (n,k) numpy array, float
        One hot representation of labels.
    """

    n = labels.shape[0]
    k = max(int(np.max(labels))+1,k) #Make sure the given k is not too small

    if standardize:
        #First convert to standard 0,1,...,k-1
        unique_labels = np.unique(L)
        k = len(unique_labels)
        for i in range(k):
            labels[labels==unique_labels[i]] = i

    #Now convert to onehot
    labels = labels.astype(int)
    onehot_labels = np.zeros((n,k))
    onehot_labels[range(n),labels] = 1

    return onehot_labels



def randomized_svd(A, k=10, c=None, q=1):
    """Randomized SVD
    ======

    Approximates top k singular values and vectors of A with a randomized
    SVD algorithm.

   
    Parameters
    ----------
    A : numpy array or matrix, scipy sparse matrix, or sparse linear operator
        Matrix to compute SVD of.
    k : int (optional), default=10
        Number of eigenvectors to compute.
    q : int (optional), default=1
        Exponent to use in randomized svd.
    c : int (optional), default=2*k
        Cutoff for randomized SVD.

    Returns
    -------
    u : (n,k) numpy array, float 
        Unitary matrix having left singular vectors as columns. 
    s : numpy array, float
        The singular values.
    vt : (k,n) numpy array, float
        Unitary matrix having right singular vectors as rows.

    Reference
    ---------
    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. [Finding structure 
    with randomness: Probabilistic algorithms for constructing approximate matrix 
    decompositions.](https://arxiv.org/abs/0909.4061) SIAM review 53.2 (2011): 217-288.
    """


    if c is None:
        c = 2*k

    n = A.shape[1]

    #Random Gaussian projection
    Omega = np.random.randn(n,c)
    Y = A@Omega
    for i in range(q):
        Y = A@(A.T@Y)

    #QR Factorization
    Q,R = np.linalg.qr(Y)

    #SVD
    B = Q.T@A
    u,s,vt = linalg.svd(B, full_matrices=False)
    u = Q@u

    #Sort singular values from largest to smallest
    ind = np.argsort(-s)
    u = u[:,ind]
    s = s[ind]
    vt = vt[ind,:]

    #Truncate to k
    u = u[:,:k]
    s = s[:k]
    vt = vt[:k,:]

    return u,s,vt
    

def rand_annulus(n,d,r1,r2):
    """Random points in annulus
    ======

    Generates independent and uniformly distributed random variables in the annulus
    \\(B_{r_2} \\setminus B_{r_1}\\).
   
    Parameters
    ----------
    n : int 
        Number of points.
    d : int
        Dimension.
    r1 : float
        Inner radius.
    r2 : float
        Outer radius

    Returns
    -------
    X : (n,d) numpy array
        Random points in annulus.
    """

    N = 0
    X = np.zeros((1,d))
    while X.shape[0] <= n:

        Y = r2*(2*np.random.rand(n,d) - 1)
        dist2 = np.sum(Y*Y,axis=1) 
        I = (dist2 < r2*r2)&(dist2 > r1*r1)
        Y = Y[I,:]
        X = np.vstack((X,Y))


    X = X[1:(n+1)]
    return X


def rand_ball(n,d):
    """Random points in a ball
    ======

    Generates independent and uniformly distributed random variables in the unit ball.
   
    Parameters
    ----------
    n : int 
        Number of points.
    d : int
        Dimension.

    Returns
    -------
    X : (n,d) numpy array
        Random points in unit ball.
    """

    N = 0
    X = np.zeros((1,d))
    while X.shape[0] <= n:

        Y = 2*np.random.rand(n,d) - 1
        I = np.sum(Y*Y,axis=1) < 1
        Y = Y[I,:]
        X = np.vstack((X,Y))


    X = X[1:(n+1)]
    return X


def bean_data(n,h):
    """Random bean data
    ======

    Generates independent and uniformly distributed random variables in a bean shaped domain
    in two dimensions.
   
    Parameters
    ----------
    n : int 
        Number of points.
    h : float
        Height of bridge between the two sides of the bean.

    Returns
    -------
    X : (n,2) numpy array
        Random points in the bean.
    """

    a=-1
    b=1
    x = a + (b-a)*np.random.rand(3*n);
    c=-0.6
    d=0.6;
    y = c + (d-c)*np.random.rand(3*n);

    X=np.transpose(np.vstack((x,y)))

    dist_from_x_axis=0.4*np.sqrt(1-x**2)*(1+h-np.cos(3*x))
    in_bean = abs(y) <= dist_from_x_axis
    X = X[in_bean,:]
    if X.shape[0] < n:
        print('Not enough samples');
    else:
        X = X[:n,:]

    return X


def mesh(X, boundary_improvement=False):
    """Mesh
    ======

    Creates a Delaunay triangulation of a 2D point cloud. Useful for visualizations.
   
    Parameters
    ----------
    X : (n,d) numpy array
        Numpy array of \\(n\\) points in dimension \\(d\\). If \\(d\\geq 3\\), only
        first 2 coordintes are used.
    boundary_improvement : bool (optional), default=False
        Whether to use improved meshing near the boundary to ensure there are no 
        boundary triangles with very large side lengths.

    Returns
    -------
    T : (n,3) numpy array
        Triangulation.
    """

    if boundary_improvement:

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
        Y = np.random.rand(m,2)
        Y[:,0] = Y[:,0]*pad - pad
        Z = np.vstack((X,Y))
        Y = np.random.rand(m,2)
        Y[:,0] = Y[:,0]*pad + 1
        Z = np.vstack((Z,Y))
        Y = np.random.rand(m,2)
        Y[:,1] = Y[:,1]*pad - pad
        Z = np.vstack((Z,Y))
        Y = np.random.rand(m,2)
        Y[:,1] = Y[:,1]*pad + 1
        Z = np.vstack((Z,Y))

        #Delaunay triangulation
        T = spatial.Delaunay(Z);
        Tri = T.simplices
        J = np.sum(Tri >= n,axis=1)==0;
        T = Tri[J,:]

    else:

        T = spatial.Delaunay(X[:,:2])
        T = T.simplices

    return T


def image_grid(X, n_rows=10, n_cols=10, padding=2, title=None, normalize=False, 
                             fontsize=None, transpose=False, return_image=False):
    """Image Grid
    ======

    Displays (or returns) a grid of images.
   
    Parameters
    ----------
    X : numpy array
        (n,m) numpy array of n grayscale images, flattened to length m arrays.
        Alternatively, X can have shape (n_rows, n_cols, m), in which case the
        parameters n_rows and n_cols below are overridden.
    n_rows : int (optional), default=10
        Number of rows in image grid.
    n_cols : int (optional), default=10
        Number of columns in image grid.
    padding : int (optional), default=2
        Amount of padding between images in the grid.
    title : str (optional), default=None
        Optional title to add to image.
    normalize : bool (optional), default=False
        Whether to normalie pixel intensities for viewing.
    fontsize : int (optional), default=None
        Font size for title, if provided. None uses the default in matplotlib.
    transpose : bool (optional), default=False
        Whether to transpose the images or not.
    return_image : bool (optional), default=False
        Whether to return the image or display it to a matplotlib window.

    Returns
    -------
    I : numpy array
        Image grid as a grayscale image (if `return_image=True).
    """

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
  
    if return_image:
        return I
    else:
        #Create a new window and plot the image
        plt.figure(figsize=(10,10))
        plt.imshow(I,cmap='gray')
        plt.axis('off')
        if title is not None:
            if fontsize is not None:
                plt.title(title,fontsize=fontsize)
            else:
                plt.title(title)

def color_image_grid(X, n_rows=10, n_cols=10, padding=2, title=None, normalize=False,
                        fontsize=None, transpose=False, return_image=False):
    """Color Image Grid
    ======

    Displays (or returns) a color grid of images.
   
    Parameters
    ----------
    X : numpy array
        (n,m) numpy array of n color images in (RRRGGGBBB) format, flattened to length m arrays.
        Alternatively, X can have shape (n_rows, n_cols, m), in which case the
        parameters n_rows and n_cols below are overridden.
    n_rows : int (optional), default=10
        Number of rows in image grid.
    n_cols : int (optional), default=10
        Number of columns in image grid.
    padding : int (optional), default=2
        Amount of padding between images in the grid.
    title : str (optional), default=None
        Optional title to add to image.
    normalize : bool (optional), default=False
        Whether to normalie pixel intensities for viewing.
    fontsize : int (optional), default=None
        Font size for title, if provided. None uses the default in matplotlib.
    transpose : bool (optional), default=False
        Whether to transpose the images or not.
    return_image : bool (optional), default=False
        Whether to return the image or display it to a matplotlib window.

    Returns
    -------
    I : numpy array
        Image grid as a color image (if `return_image=True).
    """

    m = int(X.shape[1]/3)
    imgs = []
    for i in range(3):
        imgs += [image_grid(X[:,m*i:m*(i+1)], n_rows=n_rows, n_cols=n_cols, padding=padding, title=title, normalize=normalize, fontsize=fontsize, transpose=transpose, return_image=True)]

    I = np.stack((imgs[0],imgs[1],imgs[2]),axis=2)

    if return_image:
        return I
    else:
        #Create a new window and plot the image
        plt.figure(figsize=(10,10))
        plt.imshow(I)
        plt.axis('off')
        if title is not None:
            if fontsize is not None:
                plt.title(title,fontsize=fontsize)
            else:
                plt.title(title)


def image_to_patches(I,patch_size=(16,16)):
    """Image to Patches
    =======
    Converts an image into an array of patches.
    Supports color or grayscale images.

    Parameters
    ----------
    I : numpy array
        Image to convert into patches
    patch_size : tuple (optional)
        Size of patches to use

    Returns
    -------
    P : numpy array
        Numpy array of size (num_patches,num_pixels_per_patch).
    """

    if I.ndim == 2:
        return _image_to_patches(I,patch_size=patch_size)
    elif I.ndim == 3:
        imgs = []
        for i in range(I.shape[2]):
            imgs += [_image_to_patches(I[:,:,i],patch_size=patch_size)]
        img = imgs[0]
        for J in imgs[1:]:
            img = np.hstack((img,J))
        return img
    else: 
        print('Error: Number of dimensions not supported.')
        return 0

def _image_to_patches(I,patch_size=(16,16)):

    #Compute number of patches and enlarge image if necessary
    num_patches = (np.ceil(np.array(I.shape)/np.array(patch_size))).astype(int)
    image_size = num_patches*patch_size
    J = np.zeros(tuple(image_size.astype(int)))
    J[:I.shape[0],:I.shape[1]]=I

    patches = np.zeros((num_patches[0]*num_patches[1],patch_size[0]*patch_size[1]))
    p = 0
    for i in range(int(num_patches[0])):
        for j in range(int(num_patches[1])):
            patches[p,:] = J[patch_size[0]*i:patch_size[0]*(i+1),patch_size[1]*j:patch_size[1]*(j+1)].flatten()
            p+=1

    return patches

def patches_to_image(patches,image_shape,patch_size=(16,16)):
    """Patches to image
    =======
    Converts an array of patches back into an image.
    Supports color or grayscale images.

    Parameters
    ----------
    patches : numpy array
        Array containing patches along the rows.
    image_shape : tuple 
        Shape of output image.
    patch_size : tuple (optional)
        Size of patches.

    Returns
    -------
    I : numpy array
        Numpy array of reconstructed image.
    """

    m = patch_size[0]*patch_size[1]
    num_channels = int(patches.shape[1]/m)

    if num_channels == 1:
        return _patches_to_image(patches,image_shape,patch_size=patch_size)
    else:
        img = np.zeros((image_shape[0],image_shape[1],num_channels))
        for i in range(num_channels):
            img[:,:,i] = _patches_to_image(patches[:,i*m:(i+1)*m],image_shape,patch_size=patch_size)
        return img


def _patches_to_image(patches,image_shape,patch_size=(16,16)):

    #Compute number of patches and enlarge image if necessary
    num_patches = (np.ceil(np.array(image_shape)/np.array(patch_size))).astype(int)
    image_size = num_patches*np.array(patch_size)

    I = np.zeros(tuple(image_size.astype(int)))
    p = 0
    for i in range(num_patches[0]):
        for j in range(num_patches[1]):
            I[patch_size[0]*i:patch_size[0]*(i+1),patch_size[1]*j:patch_size[1]*(j+1)] = np.reshape(patches[p,:],patch_size)
            p+=1

    return I[:image_shape[0],:image_shape[1]]



