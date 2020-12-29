#clustering_synthetic.py
#
#Demo of spectral clustering on synthetic data

import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

#Draw data randomly
n = 500
X,L = datasets.make_moons(n_samples=n,noise=0.1)
#X,L = datasets.make_circles(n_samples=n,noise=0.075,factor=0.5)
#X,L = datasets.make_blobs(n_samples=n, cluster_std=[1,1.5,0.5])

#Build a knn graph
k = 10
I,J,D = gl.knnsearch(X,k)
W = gl.weight_matrix(I,J,D,k)

#Spectral clustering
l = gl.spectral_cluster(W,np.max(L)+1)

#Compute accuracy
acc = gl.clustering_accuracy(l,L)
print("Accuracy=%f"%acc)

#Plot result
plt.scatter(X[:,0],X[:,1], c=l)
plt.axis('off')
plt.show()
