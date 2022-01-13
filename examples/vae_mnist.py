import graphlearning as gl

data, labels = gl.datasets.load('mnist')
data_vae = gl.weightmatrix.vae(data)

W_raw = gl.weightmatrix.knn(data, 10)
W_vae = gl.weightmatrix.knn(data_vae, 10)

num_train_per_class = 1
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

pred_labels_raw = gl.ssl.poisson(W_raw).fit_predict(train_ind,train_labels)
pred_labels_vae = gl.ssl.poisson(W_vae).fit_predict(train_ind,train_labels)

accuracy_raw = gl.ssl.ssl_accuracy(labels,pred_labels_raw,len(train_ind))
accuracy_vae = gl.ssl.ssl_accuracy(labels,pred_labels_vae,len(train_ind))

print('Raw Accuracy: %.2f%%'%accuracy_raw)
print('VAE Accuracy: %.2f%%'%accuracy_vae)
