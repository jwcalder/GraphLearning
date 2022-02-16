import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt

X,L = gl.datasets.two_skies(1000)
W = gl.weightmatrix.knn(X,10)

knn_ind,knn_dist = gl.weightmatrix.knnsearch(X,50)
rho = 1/np.max(knn_dist,axis=1)

model = gl.clustering.fokker_planck(W,num_clusters=2,t=1000,beta=0.5,rho=rho)
labels = model.fit_predict()

plt.scatter(X[:,0],X[:,1], c=labels)
plt.show()

