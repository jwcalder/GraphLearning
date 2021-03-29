#clustering_mnist.py
#
#Clustering example:
#This script shows how to construct a weight matrix for the whole
#MNIST dataset, using precomputed kNN data, run spectral and incres
#clustering, and report clustering accuracy.
import graphlearning as gl

#Load labels, knndata, and build 10-nearest neighbor weight matrix
labels = gl.load_labels('mnist')
W = gl.knn_weight_matrix(10,dataset='mnist',metric='vae')

#Equivalently, we can compute knndata from scratch
#X = gl.load_dataset('mnist',metric='vae')
#labels = gl.load_labels('mnist')
#W = gl.knn_weight_matrix(10,dataset='mnist',metric='vae')

#Run Laplace and Poisson learning
labels_INCRES = gl.graph_clustering(W,10,algorithm='incres',true_labels=labels,T=200)
labels_SpectralClustering = gl.graph_clustering(W,10,algorithm='spectralngjordanweiss',extra_dim=4)

#Compute and print accuracy
print('INCRES Clustering: %.2f%%'%gl.clustering_accuracy(labels,labels_INCRES))
print('Spectral Clustering: %.2f%%'%gl.clustering_accuracy(labels,labels_SpectralClustering))
