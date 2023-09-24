import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(int(1e4),2)
x,y = X[:,0],X[:,1]

eps = 0.02
W = gl.weightmatrix.epsilon_ball(X, eps)
G = gl.graph(W)

bdy_set = (x < eps) | (x > 1-eps) | (y < eps) | (y > 1-eps)
bdy_val = (x-0.5)**2 + (y-0.5)**2

u = G.plaplace(bdy_set, bdy_val[bdy_set], p=10)

plt.scatter(x,y,c=u,s=0.25)
plt.scatter(x[bdy_set],y[bdy_set],c='r',s=0.5)
plt.show()
