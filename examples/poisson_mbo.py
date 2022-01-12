import graphlearning as gl

labels = gl.datasets.load('mnist', labels_only=True)
W = gl.weightmatrix.knn('mnist', 10, metric='vae')

num_train_per_class = 1
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

class_priors = gl.utils.class_priors(labels)
model = gl.ssl.poisson_mbo(W, class_priors)
pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)

accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
print(model.name + ': %.2f%%'%accuracy)
