#Demonstration of semi-supervised learning
import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

#data
n = 500
X,L = datasets.make_moons(n_samples=n,noise=0.1)

#Build graph
k = 10
I,J,D = gl.knnsearch(X,k)
W = gl.weight_matrix(I,J,D,k,symmetrize=False) #W is not not symmetric

#Randomly choose labels
m = 5 #5 labels per class
ind = gl.randomize_labels(L,m)  #indices of labeled points

#Semi-supervised learning 
l = gl.graph_ssl(W,ind,L[ind],method='poisson',symmetrize=False)

#Compute accuracy
acc = gl.accuracy(l,L,m)   
print("Accuracy=%f"%acc)

#Plot result (red points are labels)
plt.scatter(X[:,0],X[:,1], c=l)
plt.scatter(X[ind,0],X[ind,1], c='r')
plt.show()

