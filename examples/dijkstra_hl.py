import graphlearning as gl
import matplotlib.pyplot as plt
import numpy as np
#from mayavi import mlab


n = 10000
X = gl.utils.rand_ball(n,2)
X[0,:]=[0,0]
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
#Tri = gl.utils.mesh(X)
#mlab.figure(size=(1000,1000),bgcolor=(1,1,1))
#mlab.triangular_mesh(x,y,3*(np.max(u)-u),Tri)
#mlab.savefig('cone.png')
#mlab.show()
#plt.show()
