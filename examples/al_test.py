import graphlearning.active_learning as al
import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

X,labels = datasets.make_moons(n_samples=500,noise=0.1)
W = gl.weightmatrix.knn(X,10)
train_ind = gl.trainsets.generate(labels, rate=5)
#plt.scatter(X[:,0],X[:,1], c=labels)
#plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
#plt.show()

model = gl.ssl.laplace(W)
acq = al.model_change_vopt()
act = al.active_learning(W, np.arange(labels.size), train_ind, labels[train_ind], 200)

for i in range(10):
    u = model.fit(act.current_labeled_set, act.current_labels) # perform classification with GSSL classifier
    query_points = act.select_query_points(u, acq, oracle=None) # return this iteration's newly chosen points
    query_labels = labels[query_points] # simulate the human in the loop process
    act.update_labeled_data(query_points, query_labels) # update the active_learning object's labeled set

    # plot
    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[act.current_labeled_set,0],X[act.current_labeled_set,1], c='r')
    plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
    plt.show()
    print(act.current_labeled_set)
    print(act.current_labels)

