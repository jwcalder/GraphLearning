import graphlearning as gl
import numpy as np

for n in [int(10**i) for i in range(3,6)]:

    X = np.random.rand(n,2)
    X[0,:]=[0.5,0.5]
    W = gl.weightmatrix.knn(X,50,kernel='distance')
    G = gl.graph(W)
    u = G.dijkstra([0])

    u_true = np.linalg.norm(X - [0.5,0.5],axis=1)
    error = np.linalg.norm(u-u_true, ord=np.inf)
    print('n = %d, Error = %f'%(n,error))

