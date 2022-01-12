import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

X,labels = datasets.make_moons(n_samples=500,noise=0.1)
W = gl.weightmatrix.knn(X,10)

model = gl.clustering.spectral(W, num_clusters=2)
pred_labels = model.fit_predict()

accuracy = gl.clustering.clustering_accuracy(pred_labels, labels)
print('Clustering Accuracy: %.2f%%'%accuracy)

plt.scatter(X[:,0],X[:,1], c=pred_labels)
plt.axis('off')
plt.show()

