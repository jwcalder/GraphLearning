import graphlearning as gl

labels = gl.datasets.load('mnist', labels_only=True)
W = gl.weightmatrix.knn('mnist', 10, metric='vae')

num_train_per_class = 1
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

class_priors = gl.utils.class_priors(labels)
model = gl.ssl.laplace(W, class_priors=class_priors)
model.fit(train_ind,train_labels)

pred_labels = model.predict(ignore_class_priors=True)
accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
print(model.name + ' without class priors: %.2f%%'%accuracy)

pred_labels = model.predict()
accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
print(model.name + ' with class priors: %.2f%%'%accuracy)


