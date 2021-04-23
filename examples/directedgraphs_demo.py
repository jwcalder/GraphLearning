#directedgraphs_demo.py
#
#This script is the same as classification_synthetic.py, except
#that the graph is not symmetrized in this script, showing how
#Poisson learning can handle directed (nonsymmetric) graphs
#There are two places below where flags symmetrize=False 
#must be thrown to prevent symmetrization.
#Only Poisson learning works on nonsymmetric graphs.
import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

#Draw data randomly
n = 500
X,L = datasets.make_moons(n_samples=n,noise=0.1)

#Build a knn graph
k = 10
I,J,D = gl.knnsearch(X,k)
W = gl.weight_matrix(I,J,D,k,symmetrize=False) #W is not not symmetric


#Randomly choose labels
m = 5 #5 labels per class
ind = gl.randomize_labels(L,m)  #indices of labeled points

#Semi-supervised learning 
u,T = gl.poisson(W,ind,L[ind],solver='graddesc')  #Returns kxn matrix u
l = np.argmax(u,axis=0)

#Compute accuracy
acc = gl.accuracy(l,L,len(ind))   
print("Accuracy=%f"%acc)

#Plot result (red points are labels)
plt.scatter(X[:,0],X[:,1], c=l)
plt.scatter(X[ind,0],X[ind,1], c='r')
plt.show()

