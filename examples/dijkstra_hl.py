import graphlearning as gl
import matplotlib.pyplot as plt
import numpy as np


n = 10000
X = np.random.rand(n,2)
X[0,:]=[0.5,0.5]
W = gl.weightmatrix.knn(X,50,kernel='distance',symmetrize=True)
G = gl.graph(W)
u = G.dijkstra_hl([0])

#Check the residual
grad = G.gradient(u**2,p=-1)
H = grad.max(axis=0).toarray().flatten()
print('Residual=%f'%np.max(np.absolute(H-u)))

x,y = X[:,0],X[:,1]
plt.scatter(x,y,c=u)
plt.show()
