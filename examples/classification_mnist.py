#classification_mnist.py
#
#Semi-supervised classification example:
#This script shows how to construct a weight matrix for the whole
#MNIST dataset, using precomputed kNN data, randomly select some 
#training data, run Laplace and Poisson Learning, and compute accuracy.
import graphlearning as gl

#Load labels, knndata, and build 10-nearest neighbor weight matrix
labels = gl.load_labels('mnist')
W = gl.knn_weight_matrix(10,dataset='mnist',metric='vae')

#Equivalently, we can compute knndata from scratch
#X = gl.load_dataset('mnist',metric='vae')
#I,J,D = gl.knnsearch_annoy(X,10)
#W = gl.weight_matrix(I,J,D,10)

#Randomly chose training datapoints
num_train_per_class = 1 
train_ind = gl.randomize_labels(labels, num_train_per_class)
train_labels = labels[train_ind]

#Run Laplace and Poisson learning
labels_laplace = gl.graph_ssl(W,train_ind,train_labels,algorithm='laplace')
labels_poisson = gl.graph_ssl(W,train_ind,train_labels,algorithm='poisson')

#Compute and print accuracy
print('Laplace learning: %.2f%%'%gl.accuracy(labels,labels_laplace,len(train_ind)))
print('Poisson learning: %.2f%%'%gl.accuracy(labels,labels_poisson,len(train_ind)))
