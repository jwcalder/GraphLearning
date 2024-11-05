import graphlearning as gl 
import numpy as np
import matplotlib.pyplot as plt

#Load the MNIST data
data,labels = gl.datasets.load('mnist')

#In order to run the code more quickly, 
#you may want to subsample MNIST. 
size = 70000
if size < data.shape[0]: #If less than 70000
    ind = np.random.choice(data.shape[0], size=size, replace=False)
    data = data[ind,:]
    labels = labels[ind]

#Run ARS t-SNE and plot the result
Y = gl.graph.ars(data, prog=True)
plt.scatter(Y[:,0],Y[:,1],c=labels,s=1)
plt.show()

