"""
Active Learning
========================

This module implements many active learning algorithms in an objected-oriented
fashion, similar to [sklearn](https://scikit-learn.org/stable/). The usage is similar for all algorithms.
Below, we give some high-level examples of how to use this module. There are also examples for some
individual functions, given in the documentation below.

"""
import numpy as np
from scipy.special import softmax
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

from . import graph


class active_learning:
    def __init__(self, W, training_set, current_labeled_set, current_labels, eval_cutoff, gamma=0.1):
        if type(W) == graph.graph:
            self.graph = W
        else:
            self.graph = graph.graph(W)          
        self.current_labeled_set = current_labeled_set
        self.current_labels = current_labels
        self.initial_labeled_set = current_labeled_set
        self.training_set = training_set
        self.candidate_inds = np.setdiff1d(training_set, current_labeled_set)
        self.eval_cutoff = eval_cutoff
        self.gamma = gamma
        evals, self.evecs = self.graph.eigen_decomp(normalization='normalized', k=eval_cutoff)
        self.covariance_matrix = np.linalg.inv(np.diag(evals) + self.evecs[current_labeled_set,:].T @ self.evecs[current_labeled_set,:] / gamma**2.)

    def reset_labeled_set(self):
        """Reset Labeled Set
        ======

        Resets the current labeled set to the initial labeled set

        """
        self.current_labeled_set = self.initial_labeled_set

    def select_query_points(self, u, acquisition, oracle=None, batch_size=1, candidate_method='full', fraction_points=0.1):
        """Select query points
        ======

        Select "batch_size" number of points to be labeled by an active learning algorithm we specify

        Parameters
        ----------
        u : numpy array
            score matrix from GSSL classifier.
        acquisition : class object
            acquisition function object.
        oracle : numpy array, int, default=None
            true labels for all datapoints.
        batch_size : int (optional), default=1
            number of points want to be labeled in one iteration of active learning.
        candidate_method : str (optional), default='full'
            'full' for setting candidate indices to full unlabeled set.
            'rand' for setting candidate indices to a random fraction of the unlabeled set.
        fraction_points : int (optional), default=0.1
            fraction of unlabeled points we want to use as our candidate set.

        """

        if candidate_method == 'full':
            self.candidate_inds = np.setdiff1d(self.training_set, self.current_labeled_set)
        elif (candidate_method == 'rand') and (fraction_points>0 and fraction_points<1):
            unlabeled_inds = np.setdiff1d(self.training_set, self.current_labeled_set)
            self.candidate_inds = np.random.choice(unlabeled_inds, size=int(fraction_points * len(unlabeled_inds)), replace=False)
        else:
            raise ValueError("Wrong input for candidate_method or fraction_points")
        objective_values = acquisition.compute_acquisition_function_values(self, u, uncertainty_method='smallest_margin')
        query_inds = self.candidate_inds[(-objective_values).argsort()[:batch_size]]

        if oracle is not None:
            self.update_labeled_data(query_inds, oracle[query_inds])

        return query_inds

    def update_labeled_data(self, query_inds, query_labels):
        """Update labeled data
        ======

        Update the current labeled set, current labels, and covariance matrix of our active learning object

        Parameters
        ----------
        query_inds : numpy array
            new selected query points to label.
        query_labels : numpy array
            true labels of query_inds.

        """
        self.current_labeled_set = np.append(self.current_labeled_set, query_inds)
        self.current_labels = np.append(self.current_labels, query_labels)
        for k in query_inds:
            vk = self.evecs[k]
            Cavk = self.covariance_matrix @ vk
            ip = np.inner(vk, Cavk)
            self.covariance_matrix -= np.outer(Cavk, Cavk)/(self.gamma**2. + ip) 

class acquisition_function:
    @abstractmethod
    def compute_acquisition_function_values(self, active_learning, u, uncertainty_method):
        """Internal Compute Acquisition Function Values Function
        ======

        Internal function that any acquisition function object must override. 

        Parameters
        ----------
        active_learning : class object
            active learning object.
        u : numpy array
            score matrix from GSSL classifier.
        uncertainty_method : str
            if method requires "uncertainty calculation" this string specifies which type of uncertainty measure to apply

        Returns
        -------
        acquisition_values : numpy array, float
            acquisition function values
        """
        raise NotImplementedError("Must override compute_acquisition_function_values")

class uncertainty_sampling(acquisition_function):
    """Uncertainty Sampling
    ===================

    Active learning algorithm that selects points that the classifier is most uncertain of.

    Examples
    --------
    ```py
    import numpy as np
    import graphlearning as gl
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(oracle, rate=5)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    model = gl.ssl.laplace(W)
    acq = uncertainty_sampling()
    al = active_learning(W, np.arange(labels.size), train_ind, labels[train_ind], 200)

    for i in range(10):
        u = model.fit(al.current_labeled_set, al.current_labels) # perform classification with GSSL classifier
        query_points = al.select_query_points(u, acq, oracle=None) # return this iteration's newly chosen points
        query_labels = labels[query_points] # simulate the human in the loop process
        al.update_labeled_data(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=labels)
        plt.scatter(X[al.current_labeled_set,0],X[al.current_labeled_set,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
        plt.show()
    ```

    Reference
    ---------
    [1] Settles, B., [Active Learning], vol. 6, Morgan & Claypool Publishers LLC (June 2012).
    """
    def compute_acquisition_function_values(self, active_learning, u, uncertainty_method='smallest_margin'):
        if uncertainty_method == "norm":
            u_probs = softmax(u[active_learning.candidate_inds], axis=1)
            one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u[active_learning.candidate_inds], axis=1)]
            unc_terms = np.linalg.norm((u_probs - one_hot_predicted_labels), axis=1)
        elif uncertainty_method == "entropy":
            u_probs = softmax(u[active_learning.candidate_inds], axis=1)
            unc_terms = np.max(u_probs, axis=1) - np.sum(u_probs*np.log(u_probs +.00001), axis=1)
        elif uncertainty_method == "least_confidence":
            unc_terms = np.ones((u[active_learning.candidate_inds].shape[0],)) - np.max(u[active_learning.candidate_inds], axis=1)
        elif uncertainty_method == "smallest_margin":
            u_sort = np.sort(u[active_learning.candidate_inds])
            unc_terms = 1.-(u_sort[:,-1] - u_sort[:,-2])
        elif uncertainty_method == "largest_margin":
            u_sort = np.sort(u[active_learning.candidate_inds])
            unc_terms = 1.-(u_sort[:,-1] - u_sort[:,0])
        return unc_terms

class v_opt(acquisition_function):
    """Variance Optimization
    ===================

    Active learning algorithm that selects points that minimizes the variance of the distribution of unlabeled nodes

    Examples
    --------
    ```py
    import numpy as np
    import graphlearning as gl
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(oracle, rate=5)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    model = gl.ssl.laplace(W)
    acq = v_opt()
    al = active_learning(W, np.arange(labels.size), train_ind, labels[train_ind], 200)

    for i in range(10):
        u = model.fit(al.current_labeled_set, al.current_labels) # perform classification with GSSL classifier
        query_points = al.select_query_points(u, acq, oracle=None) # return this iteration's newly chosen points
        query_labels = labels[query_points] # simulate the human in the loop process
        al.update_labeled_data(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=labels)
        plt.scatter(X[al.current_labeled_set,0],X[al.current_labeled_set,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
        plt.show()
    ```

    Reference
    ---------
    [1] Ji, M. and Han, J., “A variance minimization criterion to active learning on graphs,” in [Artificial Intelligence
    and Statistics ], 556–564 (Mar. 2012).
    """
    def compute_acquisition_function_values(self, active_learning, u, uncertainty_method='smallest_margin'):
        Cavk = active_learning.covariance_matrix @ active_learning.evecs[active_learning.candidate_inds,:].T
        col_norms = np.linalg.norm(Cavk, axis=0)
        diag_terms = (active_learning.gamma**2. + np.array([np.inner(active_learning.evecs[k,:], Cavk[:, i]) for i,k in enumerate(active_learning.candidate_inds)]))
        return col_norms**2. / diag_terms

class sigma_opt(acquisition_function):
    """Sigma Optimization
    ===================

    Active learning algorithm that selects points that minimizes the sum of the associated entries in the covariance matrix

    Examples
    --------
    ```py
    import numpy as np
    import graphlearning as gl
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(oracle, rate=5)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    model = gl.ssl.laplace(W)
    acq = sigma_opt()
    al = active_learning(W, np.arange(labels.size), train_ind, labels[train_ind], 200)

    for i in range(10):
        u = model.fit(al.current_labeled_set, al.current_labels) # perform classification with GSSL classifier
        query_points = al.select_query_points(u, acq, oracle=None) # return this iteration's newly chosen points
        query_labels = labels[query_points] # simulate the human in the loop process
        al.update_labeled_data(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=labels)
        plt.scatter(X[al.current_labeled_set,0],X[al.current_labeled_set,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
        plt.show()
    ```

    Reference
    ---------
    [1] Ma, Y., Garnett, R., and Schneider, J., “Σ-optimality for active learning on Gaussian random fields,”
    in [Advances in Neural Information Processing Systems 26 ], Burges, C. J. C., Bottou, L., Welling, M.,
    Ghahramani, Z., and Weinberger, K. Q., eds., 2751–2759, Curran Associates, Inc. (2013).
    """
    def compute_acquisition_function_values(self, active_learning, u, uncertainty_method='smallest_margin'):
        Cavk = active_learning.covariance_matrix @ active_learning.evecs[active_learning.candidate_inds,:].T
        col_sums = np.sum(Cavk, axis=0)
        diag_terms = (active_learning.gamma**2. + np.array([np.inner(active_learning.evecs[k,:], Cavk[:, i]) for i,k in enumerate(active_learning.candidate_inds)]))
        return col_sums**2. / diag_terms

class model_change(acquisition_function):
    """Model Change
    ===================

    Active learning algorithm that selects points that will produce the greatest change in the model

    Examples
    --------
    ```py
    import numpy as np
    import graphlearning as gl
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(oracle, rate=5)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    model = gl.ssl.laplace(W)
    acq = model_change()
    al = active_learning(W, np.arange(labels.size), train_ind, labels[train_ind], 200)

    for i in range(10):
        u = model.fit(al.current_labeled_set, al.current_labels) # perform classification with GSSL classifier
        query_points = al.select_query_points(u, acq, oracle=None) # return this iteration's newly chosen points
        query_labels = labels[query_points] # simulate the human in the loop process
        al.update_labeled_data(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=labels)
        plt.scatter(X[al.current_labeled_set,0],X[al.current_labeled_set,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
        plt.show()
    ```

    Reference
    ---------
    [1] Miller, K. and Bertozzi, A. L., “Model-change active learning in graph-based semi-supervised learning,”
    (Oct. 2021). arXiv: 2110.07739.

    [2] Karzand, M. and Nowak, R. D., “Maximin active learning in overparameterized model classes,” IEEE
    Journal on Selected Areas in Information Theory 1, 167–177 (May 2020).
    """
    def compute_acquisition_function_values(self, active_learning, u, uncertainty_method='smallest_margin'):
        unc_terms = uncertainty_sampling().compute_acquisition_function_values(active_learning, u)
        Cavk = active_learning.covariance_matrix @ active_learning.evecs[active_learning.candidate_inds,:].T
        col_norms = np.linalg.norm(Cavk, axis=0)
        diag_terms = (active_learning.gamma**2. + np.array([np.inner(active_learning.evecs[k,:], Cavk[:, i]) for i,k in enumerate(active_learning.candidate_inds)]))
        return unc_terms * col_norms / diag_terms  

class model_change_vopt(acquisition_function):
    """Model Change Variance Optimization
    ===================

    Active learning algorithm that is a combination of Model Change and Variance Optimization

    Examples
    --------
    ```py
    import numpy as np
    import graphlearning as gl
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(oracle, rate=5)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    model = gl.ssl.laplace(W)
    acq = model_change_vopt()
    al = active_learning(W, np.arange(labels.size), train_ind, labels[train_ind], 200)

    for i in range(10):
        u = model.fit(al.current_labeled_set, al.current_labels) # perform classification with GSSL classifier
        query_points = al.select_query_points(u, acq, oracle=None) # return this iteration's newly chosen points
        query_labels = labels[query_points] # simulate the human in the loop process
        al.update_labeled_data(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=labels)
        plt.scatter(X[al.current_labeled_set,0],X[al.current_labeled_set,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
        plt.show()
    ```

    Reference
    ---------
    [1] Ji, M. and Han, J., “A variance minimization criterion to active learning on graphs,” in [Artificial Intelligence
    and Statistics ], 556–564 (Mar. 2012).

    [2] Miller, K. and Bertozzi, A. L., “Model-change active learning in graph-based semi-supervised learning,”
    (Oct. 2021). arXiv: 2110.07739.

    [3] Karzand, M. and Nowak, R. D., “Maximin active learning in overparameterized model classes,” IEEE
    Journal on Selected Areas in Information Theory 1, 167–177 (May 2020).
    """
    def compute_acquisition_function_values(self, active_learning, u, uncertainty_method='smallest_margin'):
        unc_terms = uncertainty_sampling().compute_acquisition_function_values(active_learning, u)
        Cavk = active_learning.covariance_matrix @ active_learning.evecs[active_learning.candidate_inds,:].T
        col_norms = np.linalg.norm(Cavk, axis=0)
        diag_terms = (active_learning.gamma**2. + np.array([np.inner(active_learning.evecs[k,:], Cavk[:, i]) for i,k in enumerate(active_learning.candidate_inds)]))
        return unc_terms * col_norms **2. / diag_terms
