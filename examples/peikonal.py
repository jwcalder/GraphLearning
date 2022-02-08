import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt
import time


X = np.random.rand(int(10000),2)
x,y = X[:,0],X[:,1]

eps = 0.02
W = gl.weightmatrix.epsilon_ball(X, eps)
G = gl.graph(W)

bdy_set = (x < eps) | (x > 1-eps) | (y < eps) | (y > 1-eps)
u = G.peikonal(bdy_set)

plt.scatter(x,y,c=u,s=0.25)
plt.scatter(x[bdy_set],y[bdy_set],c='r',s=0.5)
plt.show()
