import graphlearning as gl
import time

labels = gl.datasets.load('mnist', labels_only=True)
W = gl.weightmatrix.knn('mnist', 10, metric='vae')

num_train_per_class = 1
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

models = [gl.ssl.laplace(W), gl.ssl.poisson(W)]

for model in models:
    pred_labels = model.fit_predict(train_ind,train_labels)
    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
    print(model.name + ': %.2f%%'%accuracy)

