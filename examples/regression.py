#Graph Laplacian Based Regression: This code shows how to use 
#the GraphLearning package to perform graph-Laplacian-based regression
#using the ordinary Laplacian and p-Laplacian.
import numpy as np
from scipy import sparse
import graphlearning as gl

n=1000  #Number of data points
m=40    #Number of features
v=0.75  #Train set size
lam=0.1 #Ridge regression parameter (larger values encourage more smoothness)
k=20    #Number of neighbors to use in graph
p=5     #p-Laplace value

#Data, regression function, and train indices/mask
X = np.random.rand(n,m)
y = np.sum(X,axis=1) #Regression function to predict
train_ind = np.random.choice(n,size=int(v*n),replace=False)
train_mask = np.zeros(n,dtype=bool)
train_mask[train_ind]=True
test_mask = ~train_mask

#Graph-Laplace based regression
#yhat = (B + \lambda L)^{-1}
#yhat = argmin_u \{ || B(u - y) ||^2 + \lambda u^TLu \}
#Diagonal matrix B indicates the locations of training labels.
B = sparse.spdiags(train_mask[None,:].astype(float),0)
W = gl.weightmatrix.knn(X,k)
G = gl.graph(W)
L = G.laplacian()
yhat = gl.utils.conjgrad(B + lam*L,B*y)

#The commented out code below implements p-Laplacian regression with p given above (p=5)
#yhat = G.plaplace(train_ind,y[train_ind],p)

#Compare mean squarad error on test set
rmse = np.sqrt(np.mean((yhat[test_mask] - y[test_mask])**2))
print('RMSE',rmse)
print('Relative RMSE: %.2f%%'%(100*rmse/np.sqrt(np.mean(y**2))))
