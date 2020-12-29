#AlgName.py
#
#This script shows how to code a new graph-based learning
#algorithm and incorporate it into ssl_trials to 
#compare to other SSL algorithms. 

import graphlearning as gl
import numpy as np
import scipy.sparse as sparse
import os

#Below we define a new ssl algorithm.  The name must be 'ssl'. The file name 'alg_name.py' can be anything you like,
#provided it is **lower case**. The basename 'alg_name' without the extension '.py' is the algorithm name for 
#gl.ssl_trials and gl.graph_ssl.
#
#The algorithm below is Laplace learning with a soft label fidelity
#min_u u^T L u + lam*(u-g)^2
def ssl(W,I,g,params):
#The inputs/outputs must match exactly:
#Inputs:
#  W = sparse weight matrix 
#  I = indices of labeled nodes in graph
#  g = integer labels corresponding to labels
#  params = a Python dictionary with any custom parameters for the algorithm
#Output:
#  u = kxn array of probability or one-hot vectors for each node
    
    #Size of graph
    n = W.shape[0]

    #Regularization parameter
    lam = params['lambda']
 
    unique_labels = np.unique(g)
    k = len(unique_labels)

    #One-hot encoding of labels
    F = gl.onehot_labels(I,g,n).T

    #Matrix
    A = gl.graph_laplacian(W,norm='none') + lam*gl.sparse.identity(n)

    #Preconditioner
    m = A.shape[0]
    M = A.diagonal()
    M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()
   
    #Conjugate gradient solver
    u,T = gl.conjgrad(M*A*M, M*lam*F, tol=1e-6)
    u = M*u

    return u.T



