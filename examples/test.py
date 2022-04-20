import graphlearning as gl
import numpy as np
import time

labels = gl.datasets.load('mnist', labels_only=True)
W = gl.weightmatrix.knn('mnist', 10, metric='vae')

num_train_per_class = 1
np.random.seed(1)
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

knn_ind, knn_dist = gl.weightmatrix.load_knn_data('mnist',metric='vae')
d = np.max(knn_dist,axis=1)
d = d/np.max(d)
base_tau = 0.001

models = [gl.ssl.laplace(W), 
          gl.ssl.laplace(W,reweighting='poisson'),
          gl.ssl.laplace(W,reweighting='poisson',normalization=                  'combinatorial'),
          gl.ssl.laplace(W,reweighting='poisson',tau=base_tau,normalization=     'combinatorial'),
          gl.ssl.laplace(W,reweighting='poisson',tau=base_tau*d,normalization=   'combinatorial'),
          gl.ssl.laplace(W,reweighting='poisson',tau=base_tau*d**2,normalization='combinatorial'),
          gl.ssl.laplace(W,reweighting='poisson',tau=base_tau*d**3,normalization='combinatorial'),
          gl.ssl.laplace(W,reweighting='poisson',tau=base_tau*d**4,normalization='combinatorial'),
          gl.ssl.laplace(W,reweighting='poisson',tau=base_tau*d**5,normalization='combinatorial'),
          gl.ssl.laplace(W,reweighting='poisson',tau=base_tau*d**6,normalization='combinatorial'),
          gl.ssl.laplace(W,reweighting='poisson',tau=base_tau*d**7,normalization='combinatorial')]

for model in models:
    pred_labels = model.fit_predict(train_ind,train_labels)
    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
    print(model.name + ': %.2f%%'%accuracy)

