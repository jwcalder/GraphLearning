import graphlearning as gl

data, labels = gl.datasets.load('mnist')

x = data[labels <= 1] 
y = labels[labels <= 1]
y_pred = gl.clustering.RP1D(x,20)

accuracy = gl.clustering.clustering_accuracy(y_pred, y)
print('Clustering Accuracy: %.2f%%'%accuracy)
