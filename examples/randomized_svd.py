import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import graphlearning as gl

X,L = datasets.make_moons(n_samples=500,noise=0.1)
W = gl.weightmatrix.knn(X,10)
G = gl.graph(W)

num_eig = 7
vals_exact, vecs_exact = G.eigen_decomp(normalization='normalized', k=num_eig, method='exact')
vals_rsvd, vecs_rsvd = G.eigen_decomp(normalization='normalized', k=num_eig, method='lowrank', q=50, c=50)

for i in range(1,num_eig):
    rsvd = vecs_rsvd[:,i]
    exact = vecs_exact[:,i]

    sign = np.sum(rsvd*exact)
    if sign < 0:
        rsvd *= -1

    err = np.max(np.absolute(rsvd - exact))/max(np.max(np.absolute(rsvd)),np.max(np.absolute(exact)))

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle('Eigenvector %d, err=%f'%(i,err))

    ax1.scatter(X[:,0],X[:,1], c=rsvd)
    ax1.set_title('Random SVD')

    ax2.scatter(X[:,0],X[:,1], c=exact)
    ax2.set_title('Exact')

plt.show()
