"""
Active Learning
========================

This module implements many active learning algorithms in an objected-oriented
fashion, similar to [sklearn](https://scikit-learn.org/stable/) and [modAL](https://modal-python.readthedocs.io/en/latest/index.html). The usage is similar for all algorithms, and we give some high-level examples of how to use this module with each of the provided acquisition functions.

The common workflow, however, is as follows:
```py
import graphlearning as gl

# define ssl model and acquisition function
model = gl.ssl.laplace(W)                   # graph-based ssl classifier with a given graph
acq_func = gl.active_learning.unc_sampling  # acquisition function for prioritizing which points to query


# instantiate active learner object
AL = gl.active_learner(
     model=model,                                    
     acq_function=acq_func,   
     labeled_ind=..,                        # indices of initially labeled nodes 
     labels=,                               # (integer) labels for initially labeled nodes
     policy='max',                         # active learning policy (i.e., 'max', or 'prop')
     **kwargs=...                           # other keyword arguments for the specified acq_function
    )

# select next query points
query_point = AL.select_queries(
                batch_size=1               # number of query points to select at this iteration
               ) 

# acquire label for query points
query_labels = y[query_point] 


# update the labeled data of active_learner object (including the graph-based ssl ``model`` outputs)
AL.update(query_points, query_labels) 
```

Some clarification of terms:
- ``acquisition function``: a function that quantifies "how useful" it would be to label a currently unlabeled node. Oftentimes, this is reflected in the "uncertainty" of the current classifier's output for each node. 
- __NOTE:__ users can provide their own acquisition functions that inherit from the ``acquisition_function`` class, being sure to implement it so that __larger values__ of the acquisition function correspond to __more desirable__ nodes to be labeled.
- ``policy``: the active learning policy determines which node(s) will be selected as query points, given the set of acquisition function values evaluated on the unlabeled nodes. 
- The default value ``max`` indicates that query points will be the maximizers of the acquisition function on the unlabeled nodes. The policy ``prop`` selects the query points proportional to the ''softmax'' of the acquisition function values; namely, 
\\[\\mathbb{P}(X = x) \\propto e^{\\gamma \\mathcal{A}(x)}\\]
"""

import numpy as np
from scipy.special import softmax
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

from . import graph


class active_learner:
    def __init__(self, model, acq_function, labeled_ind, labels, policy='max', **kwargs):
        self.model = model
        self.labeled_ind = labeled_ind.copy()
        self.labels = labels.copy()
        self.acq_function = acq_function(**kwargs)
        self.acq_function.update(labeled_ind, labels)
        self.policy = policy
        self.u = self.model.fit(self.labeled_ind, self.labels) # initialize the ssl model on the initially labeled data
        self.n = self.model.graph.num_nodes
        self.all_inds = np.arange(self.n)
        self.unlabeled_ind = np.setdiff1d(self.all_inds, self.labeled_ind)
        self.printed_warning = False
        

    def select_queries(self, batch_size=1, policy=None, candidate_ind='full', rand_frac=0.1, return_acq_vals=False, prop_gamma=1.0, 
                       allow_repeat=False):
        if policy is None:
            policy = self.policy
        
        if isinstance(candidate_ind, np.ndarray):
            if (candidate_ind.min() < 0) or (candidate_ind.max() > self.n):
                raise ValueError(f"candidate_ind must have integer values between 0 and {self.n}")
        elif candidate_ind == 'full':
            if allow_repeat:
                candidate_ind = np.arange(self.all_inds)
            else:
                candidate_ind = np.setdiff1d(self.all_inds, self.labeled_ind)
        elif (candidate_ind == 'rand') and (rand_frac>0 and rand_frac<1):
            if allow_repeat:
                candidate_ind = np.random.choice(self.all_inds, size=int(rand_frac * self.n), replace=False)
            else:
                candidate_ind = np.random.choice(self.unlabeled_ind, size=int(rand_frac * len(self.unlabeled_ind)), replace=False)
        else:
            raise ValueError("Invalid input for candidate_ind")
        
        acq_vals = self.acq_function.compute(self.u, candidate_ind)
        
        if policy == 'max':
            query_ind = candidate_ind[(-acq_vals).argsort()[:batch_size]]
        elif policy == 'prop':
            probs = np.exp(prop_gamma*(acq_vals - acq_vals.max()))
            probs /= probs.sum()
            query_ind = np.random.choice(candidate_ind, batch_size, p=probs)
        else:
            query_ind = policy(candidate_ind, acq_vals, batch_size) # user-defined policy

        if return_acq_vals:
            return query_ind, acq_vals
        return query_ind
    

    def update(self, query_ind, query_labels):
        if np.intersect1d(query_ind, self.labeled_ind).size > 0 and not self.printed_warning:
            print("WARNING: Having multiple observations at a single node detected")
            self.printed_warning = True
        self.labeled_ind = np.append(self.labeled_ind, query_ind)
        self.labels = np.append(self.labels, query_labels)
        self.u = self.model.fit(self.labeled_ind, self.labels)
        self.unlabeled_ind = np.setdiff1d(self.all_inds, self.labeled_ind)
        self.acq_function.update(query_ind, query_labels)
        return

class acquisition_function:  
    """Acquisition Function
    ======
    
    Object that computes a measure of ''utility'' for labeling nodes that are currently unlabeled. Users can define their own acqusition functions to inherit from this class as follows:
    
    ```py
    import graphlearning as gl
    
    class new_acq_func(gl.acquisition_function):
        def __init__(self, arg1=None):
            self.arg1 = arg1           # any arguments that are passed into this acquisition function are given 
                                       # as kwargs in active_learner instantiation

        def compute(self, u, candidate_ind):
            vals = ...                 # compute the acquisition function so that larger values are more desired
            return vals  
    
    ```
    
    
    """
    @abstractmethod
    def compute(self, u, candidate_ind):
        """Internal Compute Acquisition Function Values Function
        ======

        Internal function that any acquisition function object must override. 

        Parameters
        ----------
        u : numpy array
            score matrix from GSSL classifier. 
        candidate_ind : numpy array (or list)
            (sub)set of indices on which to compute the acquisition function

        Returns
        -------
        acquisition_values : numpy array, float
            acquisition function values
        """
        raise NotImplementedError("Must override compute")

    @abstractmethod
    def update(self, query_ind, query_labels):
        return 



class unc_sampling(acquisition_function):
    """Uncertainty Sampling
    ===================

    Active learning algorithm that selects points that the classifier is most uncertain of.

    Examples
    --------
    ```py
    import graphlearning.active_learning as al
    import graphlearning as gl
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(labels, rate=5)
    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    
    model = gl.ssl.laplace(W)
    AL = gl.active_learning.active_learner(model, gl.active_learning.unc_sampling, train_ind, y[train_ind])

    for i in range(10):
        query_points = AL.select_queries() # return this iteration's newly chosen points
        query_labels = y[query_points] # simulate the human in the loop process
        AL.update(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=y)
        plt.scatter(X[AL.labeled_ind,0],X[AL.labeled_ind,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1)
        plt.show()
        print(AL.labeled_ind)
        print(AL.labels)
    ```

    Reference
    ---------
    [1] Settles, B., [Active Learning], vol. 6, Morgan & Claypool Publishers LLC (June 2012).
    """
    def __init__(self, unc_method='smallest_margin'):
        self.unc_method = unc_method

    def compute(self, u, candidate_ind):
        if self.unc_method == "norm":
            u_probs = softmax(u[candidate_ind], axis=1)
            one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u[candidate_ind], axis=1)]
            unc_terms = np.linalg.norm((u_probs - one_hot_predicted_labels), axis=1)
        elif self.unc_method == "entropy":
            u_probs = softmax(u[candidate_ind], axis=1)
            unc_terms = np.max(u_probs, axis=1) - np.sum(u_probs*np.log(u_probs +.00001), axis=1)
        elif self.unc_method == "least_confidence":
            unc_terms = np.ones((u[candidate_ind].shape[0],)) - np.max(u[candidate_ind], axis=1)
        elif self.unc_method == "smallest_margin":
            u_sort = np.sort(u[candidate_ind])
            unc_terms = 1.-(u_sort[:,-1] - u_sort[:,-2])
        elif self.unc_method == "largest_margin":
            u_sort = np.sort(u[candidate_ind])
            unc_terms = 1.-(u_sort[:,-1] - u_sort[:,0])
        elif self.unc_method == "unc_2norm":
            unc_terms = 1. - np.linalg.norm(u[candidate_ind], axis=1)
        return unc_terms



class var_opt(acquisition_function):
    """Variance Optimization
    ===================

    Active learning algorithm that selects points that minimizes the variance of the distribution of unlabeled nodes.

    Examples
    --------
    ```py
    import graphlearning.active_learning as al
    import graphlearning as gl
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(labels, rate=5)
    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    # compute initial, low-rank (spectral truncation) covariance matrix 
    evals, evecs = model.graph.eigen_decomp(normalization='normalized', k=50)
    C = np.diag(1. / (evals + 1e-11))
    AL = gl.active_learning.active_learner(model, gl.active_learning.var_opt, train_ind, y[train_ind], C=C.copy(), V=evecs.copy())

    for i in range(10):
        query_points = AL.select_queries() # return this iteration's newly chosen points
        query_labels = y[query_points] # simulate the human in the loop process
        AL.update(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=y)
        plt.scatter(X[AL.labeled_ind,0],X[AL.labeled_ind,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1)
        plt.show()
        print(AL.labeled_ind)
        print(AL.labels)
    ```

    Reference
    ---------
    [1] Ji, M. and Han, J., “A variance minimization criterion to active learning on graphs,” in [Artificial Intelligence
    and Statistics ], 556–564 (Mar. 2012).
    """

    ## this only handles the FULL C computation or the spectral truncation
    def __init__(self, C, V=None, gamma2=0.1**2.):
        assert (C.shape[0] == C.shape[1]) or (V is not None)
        self.C = C.copy()
        self.V = V
        self.gamma2 = gamma2
        if self.V is None:
            self.storage = 'full'
        else:
            self.storage = 'trunc'
        
        
    def compute(self, u, candidate_ind):
        if self.storage == 'full':
            col_norms = np.linalg.norm(self.C[:,candidate_ind], axis=0)**2.
            diag_terms = self.gamma2 + self.C.diagonal()[candidate_ind]
        else:
            Cavk = self.C @ self.V[candidate_ind,:].T
            col_norms = np.linalg.norm(Cavk, axis=0)**2.
            diag_terms = (self.gamma2 + np.array([np.inner(self.V[k,:], Cavk[:, i]) for i,k in enumerate(candidate_ind)]))
            
        return col_norms / diag_terms

    def update(self, query_ind, query_labels):
        for k in query_ind:
            if self.storage == 'full':
                self.C -= np.outer(self.C[:,k], self.C[:,k]) / (self.gamma2 + self.C[k,k])
            else:
                vk = self.V[k]
                Cavk = self.C @ vk
                ip = np.inner(vk, Cavk)
                self.C -= np.outer(Cavk, Cavk)/(self.gamma2 + ip) 

        return 




class sigma_opt(acquisition_function):
    """Sigma Optimization
    ===================

    Active learning algorithm that selects points that minimizes the sum of the associated entries in the covariance matrix.

    Examples
    --------
    ```py
    import graphlearning.active_learning as al
    import graphlearning as gl
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(labels, rate=5)
    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    # compute initial, low-rank (spectral truncation) covariance matrix 
    evals, evecs = model.graph.eigen_decomp(normalization='normalized', k=50)
    C = np.diag(1. / (evals + 1e-11))
    AL = gl.active_learning.active_learner(model, gl.active_learning.sigma_opt, train_ind, y[train_ind], C=C.copy(), V=evecs.copy())

    for i in range(10):
        query_points = AL.select_queries() # return this iteration's newly chosen points
        query_labels = y[query_points] # simulate the human in the loop process
        AL.update(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=y)
        plt.scatter(X[AL.labeled_ind,0],X[AL.labeled_ind,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1)
        plt.show()
        print(AL.labeled_ind)
        print(AL.labels)
    ```

    Reference
    ---------
    [1] Ma, Y., Garnett, R., and Schneider, J., “Σ-optimality for active learning on Gaussian random fields,”
    in [Advances in Neural Information Processing Systems 26 ], Burges, C. J. C., Bottou, L., Welling, M.,
    Ghahramani, Z., and Weinberger, K. Q., eds., 2751–2759, Curran Associates, Inc. (2013).
    """
    ## this only handles the FULL C computation or the spectral truncation
    def __init__(self, C, V=None, gamma2=0.1**2.):
        assert (C.shape[0] == C.shape[1]) or (V is not None)
        self.C = C.copy()
        self.V = V
        self.gamma2 = gamma2
        if self.V is None:
            self.storage = 'full'
        else:
            self.storage = 'trunc'
        
        
    def compute(self, u, candidate_ind):
        if self.storage == 'full':
            col_sums = np.sum(self.C[:, candidate_ind], axis=0)**2.
            diag_terms = self.gamma2 + self.C.diagonal()[candidate_ind]
        else:
            Cavk = self.C @ self.V[candidate_ind,:].T
            col_sums = np.sum(Cavk, axis=0)**2.
            diag_terms = (self.gamma2 + np.array([np.inner(self.V[k,:], Cavk[:, i]) for i,k in enumerate(candidate_ind)]))
            
        return col_sums/ diag_terms


    def update(self, query_ind, query_labels):
        for k in query_ind:
            if self.storage == 'full':
                self.C -= np.outer(self.C[:,k], self.C[:,k]) / (self.gamma2 + self.C[k,k])
            else:
                vk = self.V[k]
                Cavk = self.C @ vk
                ip = np.inner(vk, Cavk)
                self.C -= np.outer(Cavk, Cavk)/(self.gamma2 + ip) 

        return



class model_change(acquisition_function):
    """Model Change
    ===================

    Active learning algorithm that selects points that will produce the greatest change in the model.

    Examples
    --------
    ```py
    import graphlearning.active_learning as al
    import graphlearning as gl
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(labels, rate=5)
    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    # compute initial, low-rank (spectral truncation) covariance matrix 
    evals, evecs = model.graph.eigen_decomp(normalization='normalized', k=50)
    C = np.diag(1. / (evals + 1e-11))
    AL = gl.active_learning.active_learner(model, gl.active_learning.model_change, train_ind, y[train_ind], C=C.copy(), V=evecs.copy())

    for i in range(10):
        query_points = AL.select_queries() # return this iteration's newly chosen points
        query_labels = y[query_points] # simulate the human in the loop process
        AL.update(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=y)
        plt.scatter(X[AL.labeled_ind,0],X[AL.labeled_ind,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1)
        plt.show()
        print(AL.labeled_ind)
        print(AL.labels)
    ```

    Reference
    ---------
    [1] Miller, K. and Bertozzi, A. L., “Model-change active learning in graph-based semi-supervised learning,”
    (Oct. 2021). arXiv: 2110.07739.

    [2] Karzand, M. and Nowak, R. D., “Maximin active learning in overparameterized model classes,” IEEE
    Journal on Selected Areas in Information Theory 1, 167–177 (May 2020).
    """
    def __init__(self, C, V=None, gamma2=0.1**2., unc_method='smallest_margin'):
        assert (C.shape[0] == C.shape[1]) or (V is not None)
        self.C = C.copy()
        self.V = V
        self.gamma2 = gamma2
        self.unc_sampling = unc_sampling(unc_method=unc_method)
        if self.V is None:
            self.storage = 'full'
        else:
            self.storage = 'trunc'
        
    def compute(self, u, candidate_ind):
        unc_terms = self.unc_sampling.compute(u, candidate_ind)
        if self.storage == 'full':
            col_norms = np.linalg.norm(self.C, axis=0)
            diag_terms = self.gamma2 + self.C.diagonal()
        else:
            Cavk = self.C @ self.V[candidate_ind,:].T
            col_norms = np.linalg.norm(Cavk, axis=0)
            diag_terms = (self.gamma2 + np.array([np.inner(self.V[k,:], Cavk[:, i]) for i,k in enumerate(candidate_ind)]))
        return unc_terms * col_norms / diag_terms  


    def update(self, query_ind, query_labels):
        for k in query_ind:
            if self.storage == 'full':
                self.C -= np.outer(self.C[:,k], self.C[:,k]) / (self.gamma2 + self.C[k,k])
            else:
                vk = self.V[k]
                Cavk = self.C @ vk
                ip = np.inner(vk, Cavk)
                self.C -= np.outer(Cavk, Cavk)/(self.gamma2 + ip) 
        return
        

class model_change_var_opt(acquisition_function):
    """Model Change Variance Optimization
    ===================

    Active learning algorithm that is a combination of Model Change and Variance Optimization.

    Examples
    --------
    ```py
    import graphlearning.active_learning as al
    import graphlearning as gl
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(labels, rate=5)
    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    # compute initial, low-rank (spectral truncation) covariance matrix 
    evals, evecs = model.graph.eigen_decomp(normalization='normalized', k=50)
    C = np.diag(1. / (evals + 1e-11))
    AL = gl.active_learning.active_learner(model, gl.active_learning.model_change_var_opt, train_ind, y[train_ind], C=C.copy(), V=evecs.copy())

    for i in range(10):
        query_points = AL.select_queries() # return this iteration's newly chosen points
        query_labels = y[query_points] # simulate the human in the loop process
        AL.update(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        plt.scatter(X[:,0],X[:,1], c=y)
        plt.scatter(X[AL.labeled_ind,0],X[AL.labeled_ind,1], c='r')
        plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1)
        plt.show()
        print(AL.labeled_ind)
        print(AL.labels)
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
    def __init__(self, C, V=None, gamma2=0.1**2., unc_method='smallest_margin'):
        assert (C.shape[0] == C.shape[1]) or (V is not None)
        self.C = C.copy()
        self.V = V
        self.gamma2 = gamma2
        self.unc_sampling = unc_sampling(unc_method=unc_method)
        if self.V is None:
            self.storage = 'full'
        else:
            self.storage = 'trunc'
        
    def compute(self, u, candidate_ind):
        unc_terms = self.unc_sampling.compute(u, candidate_ind)
        if self.storage == 'full':
            col_norms = np.linalg.norm(self.C, axis=0)**2.
            diag_terms = self.gamma2 + self.C.diagonal()
        else:
            Cavk = self.C @ self.V[candidate_ind,:].T
            col_norms = np.linalg.norm(Cavk, axis=0)**2.
            diag_terms = (self.gamma2 + np.array([np.inner(self.V[k,:], Cavk[:, i]) for i,k in enumerate(candidate_ind)]))
        return unc_terms * col_norms / diag_terms  


    def update(self, query_ind, query_labels):
        for k in query_ind:
            if self.storage == 'full':
                self.C -= np.outer(self.C[:,k], self.C[:,k]) / (self.gamma2 + self.C[k,k])
            else:
                vk = self.V[k]
                Cavk = self.C @ vk
                ip = np.inner(vk, Cavk)
                self.C -= np.outer(Cavk, Cavk)/(self.gamma2 + ip) 
        return
