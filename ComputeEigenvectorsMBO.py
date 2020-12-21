#Computes eigenvector decomposition used in MBO algorithm
import graphlearning as gl
import numpy as np
import scipy.sparse as sparse
import sys, getopt
import os

def print_help():
    
    print('=======================================================')
    print('GraphLearning: Python package for graph-based learning.')
    print('=======================================================')
    print('=======================================================')
    print('Compute Eigenvector Expansion for MBO')
    print('=======================================================')
    print('                                                       ')
    print('Options:')
    print('   -d (--dataset=): MNIST, FashionMNIST,...more soon (default=MNIST)')
    print('   -m (--metric=):  Metric for computing similarities (default=L2)')
    print('          Choices:  scatter, L2')
    print('   -k (--knn=): Number of nearest neighbors (default=10)')
    print('   -N (--NumEig=): Number of eigenvectors (default=300)')



#Default settings
dataset = 'MNIST'
metric = 'L2'
k = 10
N = 300

#Read command line arguments
try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:m:k:N:",["dataset=","method=","knn=","NumEig="])
except getopt.GetoptError:
    print_help()
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt in ("-d", "--dataset"):
        dataset = arg
    elif opt in ("-m", "--method"):
        metric = arg
    elif opt in ("-k", "--knn"):
        k = int(arg)
    elif opt in ("-N", "--NumEig"):
        N = int(arg)

outfile = "MBOdata/"+dataset+"_"+metric+"_k%d"%k+"_spectrum.npz"

#Print basic info
print('=======================================================')
print('GraphLearning: Python package for graph-based learning.')
print('=======================================================')
print('=======================================================')
print('Compute Eigenvector Expansion for MBO')
print('=======================================================')
print('                                                       ')
print('Dataset: '+dataset)
print('Metric: '+metric)
print('Number of neighbors: %d'%k)
print('Number of Eigenvectors: %d'%N)
print('Output file: '+outfile)
print('                                                       ')
print('=======================================================')
print('                                                       ')


#Load kNN data
I,J,D = gl.load_kNN_data(dataset,metric=metric)

if k > I.shape[1]:
    print('kNNData only has %d'%I.shape[1]+'-nearest neighbor information. Aborting...')
    sys.exit(2)
else:
    W = gl.weight_matrix(I,J,D,k)
    W = (W + W.transpose())/2

#Normalized Laplacian
L = gl.graph_laplacian(W,norm="normalized")
vals_norm, vecs_norm = sparse.linalg.eigs(L,k=N,which='SM')

#Check if MBOdata directory exists
if not os.path.exists('MBOdata'):
    os.makedirs('MBOdata')

#Save weight matrix to file
np.savez_compressed(outfile,eigenvalues=vals_norm,eigenvectors=vecs_norm)
