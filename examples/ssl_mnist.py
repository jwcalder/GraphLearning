import graphlearning as gl

labels = gl.datasets.load('mnist', labels_only=True)

num_train_per_class = 1
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

W = gl.weightmatrix.knn('mnist', 10, metric='vae',kernel='gaussian')
models = [gl.ssl.laplace(W), gl.ssl.poisson(W),gl.ssl.plaplace(W,p=3),gl.ssl.amle(W),gl.ssl.volume_mbo(W,gl.utils.class_priors(labels))]

for model in models:
    pred_labels = model.fit_predict(train_ind,train_labels)
    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,train_ind)
    print(model.name + ': %.2f%%'%accuracy)

