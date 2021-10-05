#add_dataset.py
#
#This script shows how to add a new dataset into
#the graphlearning simulation framework.
import graphlearning as gl
import numpy as np

#Make up some mixture of Gaussian data with 3 classes
n = 500
separation = 0.8
X1 = np.random.randn(n,2)
L1 = np.zeros((n,),dtype=int)
X2 = np.random.randn(n,2) + separation*np.array([4,2])
L2 = np.ones((n,),dtype=int)
X3 = np.random.randn(n,2) + separation*np.array([0,4])
L3 = 2*np.ones((n,),dtype=int)

#Stack classes and labels together
n = 3*n
X = np.vstack((X1,X2,X3))
L = np.hstack((L1,L2,L3))

#Save dataset
gl.save_dataset(X,'blobs',overwrite=True)

#Save labels
gl.save_labels(L,'blobs',overwrite=True)

#To add a dataset to the simulation environment, we also need
#to save a label permutation, which is a number of random train/test splits
#and store some precomputed knn-data

#Create label permutation with 100 trials at 1,2,3,4,5 labels per class
#You can add any identifying string as name='...' if you need to create additional
#label permutations for a dataset.
gl.create_label_permutations(L,100,[1,2,3,4,5],dataset='blobs',name=None,overwrite=True)

#Run knn search and save info on 30 nearest neighbors
#Choose as many as you are likely to use in practice, the code will automatically subset if needed.
#This uses a kd-tree. For higher dimensional data use the annoy package, as below
#I,J,D = gl.knnsearch_annoy(X,30,dataset='blobs')
I,J,D = gl.knnsearch(X,30,dataset='blobs')

