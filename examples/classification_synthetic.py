#classification_synthetic.py
#
#Demo of graph-based semi-supervised learning on the
#two moons dataset.
import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

#Draw data randomly
n = 500
X,L = datasets.make_moons(n_samples=n,noise=0.1)

#Build a knn graph
k = 10
W = gl.knn_weight_matrix(k,data=X)

#Randomly choose labels
m = 5 #5 labels per class
ind = gl.randomize_labels(L,m)  #indices of labeled points

#Semi-supervised learning 
l = gl.graph_ssl(W,ind,L[ind],algorithm='poisson')

#Compute accuracy
acc = gl.accuracy(l,L,len(ind))   
print("Accuracy=%f"%acc)

#Plot result (red points are labels)
plt.scatter(X[:,0],X[:,1], c=l)
plt.scatter(X[ind,0],X[ind,1], c='r')
plt.show()

