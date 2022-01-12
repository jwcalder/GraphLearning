import graphlearning as gl

W = gl.weightmatrix.knn('mnist', 10, metric='vae')
labels = gl.datasets.load('mnist', labels_only=True)

model = gl.clustering.spectral(W, num_clusters=10, extra_dim=4)
pred_labels = model.fit_predict(all_labels=labels)

accuracy = gl.clustering.clustering_accuracy(pred_labels,labels)
print('Clustering Accuracy: %.2f%%'%accuracy)

