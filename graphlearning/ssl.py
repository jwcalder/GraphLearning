"""
Semi-Supervised Learning
========================

This module implements many graph-based semi-supervised learning algorithms in an objected-oriented
fashion, similar to [sklearn](https://scikit-learn.org/stable/). The usage is similar for all algorithms.
Below, we give some high-level examples of how to use this module. There are also examples for some
individual functions, given in the documentation below.

Two Moons Example
----
Semi-supervised learning on the two-moons dataset: [ssl_twomoons.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/ssl_twomoons.py).
```py
import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

X,labels = datasets.make_moons(n_samples=500,noise=0.1)
W = gl.weightmatrix.knn(X,10)

train_ind = gl.trainsets.generate(labels, rate=5)
train_labels = labels[train_ind]

model = gl.ssl.laplace(W)
pred_labels = model.fit_predict(train_ind, train_labels)

accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, train_ind)   
print("Accuracy: %.2f%%"%accuracy)

plt.scatter(X[:,0],X[:,1], c=pred_labels)
plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
plt.show()
```
Handwritten digit classification
-----
Laplace and Poisson learning on MNIST at 1 label per class: [ssl_mnist.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/ssl_mnist.py).
```py
import graphlearning as gl

labels = gl.datasets.load('mnist', labels_only=True)
W = gl.weightmatrix.knn('mnist', 10, metric='vae')

num_train_per_class = 1
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

models = [gl.ssl.laplace(W), gl.ssl.poisson(W)]

for model in models:
    pred_labels = model.fit_predict(train_ind,train_labels)
    accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, train_ind)
    print(model.name + ': %.2f%%'%accuracy)
```
Comparing different methods
-----------
The `ssl` module contains functions for running large scale experiments, with parallel processing,
to compare many different semi-supervised learning algorithms at different label rates
and with different, randomly chosen, training data. The package includes functions to automatically
create LaTeX tables and plots comparing accuracy or test error. This example compares several methods over 100 
randomized trials: [ssl_trials.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/ssl_trials.py).
```py
import graphlearning as gl

dataset = 'mnist'
metric = 'vae' 
k = 10 

W = gl.weightmatrix.knn(dataset, k, metric=metric)
D = gl.weightmatrix.knn(dataset, k, metric=metric, kernel='distance')

labels = gl.datasets.load(dataset, metric=metric, labels_only=True)
trainsets = gl.trainsets.load(dataset)

model_list = [gl.ssl.graph_nearest_neighbor(D),
              gl.ssl.laplace(W),
              gl.ssl.laplace(W, reweighting='wnll'), 
              gl.ssl.laplace(W, reweighting='poisson'),
              gl.ssl.poisson(W, solver='gradient_descent')]

tag = dataset + '_' + metric + '_k%d'%k
for model in model_list:
    model.ssl_trials(trainsets, labels, num_cores=20, tag=tag)

gl.ssl.accuracy_table(model_list, tag=tag, savefile='SSL_'+dataset+'.tex', title="SSL Comparison: "+dataset)
gl.ssl.accuracy_plot(model_list, tag=tag, title='SSL')
```
Class Priors
-----------
Prior information about relative class sizes can be used to improve performance: [ssl_classpriors.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/ssl_classpriors.py).
```py
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
accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,train_ind)
print(model.name + ' without class priors: %.2f%%'%accuracy)

pred_labels = model.predict()
accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,train_ind)
print(model.name + ' with class priors: %.2f%%'%accuracy)
```
"""
import numpy as np
from scipy import sparse
from abc import ABCMeta, abstractmethod
import multiprocessing
from joblib import Parallel, delayed
import sys, os, datetime, matplotlib
import matplotlib.pyplot as plt


from . import utils 
from . import graph



#Directories
results_dir = os.path.join(os.getcwd(),'results')

class ssl:
    __metaclass__ = ABCMeta

    def __init__(self, W, class_priors):
        if W is None:
            self.graph = None
        else:
            self.set_graph(W)
        self.prob = None
        self.fitted = False
        self.name = ''
        self.accuracy_filename = ''
        self.requires_eig = False
        self.onevsrest = False
        self.similarity = True
        self.class_priors = class_priors
        if self.class_priors is not None:
            self.class_priors = self.class_priors / np.sum(self.class_priors)
        self.weights = 1
        self.class_priors_error = 1

    def set_graph(self, W):
        """Set Graph
        ===================

        Sets the graph object for semi-supervised learning.

        Implements 3 different solvers, spectral, gradient_descent, and conjugate_gradient. 
        GPU acceleration is available for gradient descent. See [1] for details.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        """

        if type(W) == graph.graph:
            self.graph = W
        else:
            self.graph = graph.graph(W)

    def volume_label_projection(self):
        """Volume label projection
        ======
    
        Projects class probabilities to labels while enforcing a constraint on 
        class priors (i.e., class volumes). Does not return anything, just modifies `self.weights`.
        
        Returns
        -------
        labels : numpy array (int)
            Predicted labels after volume correction.
        """

        n = self.graph.num_nodes
        k = self.prob.shape[1]
        if type(self.weights) == int:
            self.weights = np.ones((k,))

        #Time step
        dt = 0.1
        if self.similarity:
            dt *= -1

        #np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        i = 0
        err = 1
        while i < 1e4 and err > 1e-3:
            i += 1
            class_size = np.mean(utils.labels_to_onehot(self.predict(),k),axis=0)
            #print(class_size-self.class_priors)
            grad = class_size - self.class_priors
            err = np.max(np.absolute(grad))
            self.weights += dt*grad
            self.weights = self.weights/self.weights[0]

        self.class_priors_error = err

        return self.predict()

    def get_accuracy_filename(self):
        """Get accuracy filename
        ========

        Returns name of the file that will store the accuracy results for `ssl_trials.py`.

        Returns
        -------
        fname : str
            Accuracy filename.
        """

        fname = self.accuracy_filename
        if self.class_priors is not None:
            fname += '_classpriors'
        fname += '_accuracy.csv'
        return fname


    def predict(self, ignore_class_priors=False):
        """Predict
        ========

        Makes label predictions based on the probabilities computed by `fit()`.
        Will use a volume label projection if `class_priors` were given, to ensure 
        the number of nodes predicted in each class is correct.
        
        Parameters
        ----------
        ignore_class_priors : bool (optional), default=False
            Used to disable the volume constrained label decision, when `class_priors` has been provided. 

        Returns
        -------
        pred_labels : (int) numpy array
            Predicted labels as integers for all datapoints in the graph.
        """

        if self.fitted == False:
            sys.exit('Model has not been fitted yet.')

        if ignore_class_priors:
            w = 1
        else:
            w = self.weights

        scores = self.prob - np.min(self.prob)
        scores = scores/np.max(scores)

        #Check if scores are similarity or distance
        if self.similarity:
            pred_labels = np.argmax(scores*w,axis=1)
        else: #Then distances
            pred_labels = np.argmin(scores*w,axis=1)

        return pred_labels 

    def fit_predict(self, train_ind, train_labels, all_labels=None):
        """Fit and predict
        ======

        Calls fit() and predict() sequentially.

        Parameters
        ----------
        train_ind : numpy array, int
            Indicies of training points.
        train_labels : numpy array, int
            Training labels as integers \\(0,1,\\dots,k-1\\) for \\(k\\) classes.
        all_labels : numpy array, int (optional)
            True labels for all datapoints.

        Returns
        -------
        pred_labels : (int) numpy array
            Predicted labels as integers for all datapoints in the graph.
        """

        self.fit(train_ind, train_labels, all_labels=all_labels)
        return self.predict()

    def ssl_trials(self, trainsets, labels, num_cores=1, tag='', save_results=True, overwrite=False, num_trials=-1):
        """Semi-supervised learning trials
        ===================

        Runs a semi-supervised learning algorithm on a list of training sets,
        recording the label rates and saves the results to a csv file in the
        local folder results/. The filename is controlled by the member function
        model.get_accuracy_filename(). The trial will abort if the accuracy result
        file already exists, unless `overwrite=True`.

        Parameters
        ----------
        trainsets : list of numpy arrays
            Collection of training sets to run semi-supervised learning on. This is the output
            of `graphlearning.trainsets.generate` or `graphlearning.trainsets.load`.
        labels : numpy array (int)
            Integer array of labels for entire dataset.
        num_cores : int
            Number of cores to use for parallel processing over trials
        tag : str (optional), default=''
            An extra identifying tag to add to the accuracy filename.
        save_results : bool (optional), default=True
            Whether to save results to csv file or just print to the screen.
        overwrite : bool (optional), default = False
            Whether to overwrite existing results file, if found. If `overwrite=False`, 
            `save_results=True`, and the results file is found, the trial is aborted.
        num_trials: int (optional), defualt = -1
            Number of trials. Any negative number runs all trials.
        """

        if num_trials > 0:
            trainsets = trainsets[:num_trials]

        #Print name
        print('\nModel: '+self.name)

        if save_results:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            outfile = os.path.join(results_dir, tag+self.get_accuracy_filename())
            if (not overwrite) and os.path.exists(outfile):
                print('Aborting: SSL trial ('+self.get_accuracy_filename()+') already completed , and overwrite is False.')
                return
            f = open(outfile,"w")
            #now = datetime.datetime.now()
            #f.write("Date/Time, "+now.strftime("%Y-%m-%d_%H:%M")+"\n")
            if self.class_priors is None:
                f.write('Number of labels,Accuracy\n')
            else:
                f.write('Number of labels,Accuracy,Accuracy with class priors,Class priors error\n')
            f.close()

        if save_results:
            print('Results File: '+outfile)

        if self.class_priors is None:
            print('\nNumber of labels,Accuracy')
        else:
            print('\nNumber of labels,Accuracy,Accuracy with class priors,Class priors error')

        #Trial run to trigger eigensolvers, when needed
        if self.requires_eig:
            pred_labels = self.fit_predict(trainsets[0], labels[trainsets[0]])
     
        def one_trial(train_ind):

            #Number of labels
            num_train = len(train_ind)
            train_labels = labels[train_ind]

            #Run semi-supervised learning
            pred_labels = self.fit_predict(train_ind, train_labels)

            #Compute accuracy
            accuracy = ssl_accuracy(pred_labels,labels,train_ind)

            #If class priors were provided, check accuracy without priors
            if self.class_priors is not None:
                pred_labels = self.predict(ignore_class_priors=True)
                accuracy_without_priors = ssl_accuracy(pred_labels,labels,train_ind)
            
            #Print to terminal
            if self.class_priors is None:
                print("%d" % num_train + ",%.2f" % accuracy)
            else:
                print("%d,%.2f,%.2f,%.5f" % (num_train,accuracy_without_priors,accuracy,self.class_priors_error))

            #Write to file
            if save_results:
                f = open(outfile,"a+")
                if self.class_priors is None:
                    f.write("%d" % num_train + ",%.2f\n" % accuracy)
                else:
                    f.write("%d,%.2f,%.2f,%.5f\n" % (num_train,accuracy_without_priors,accuracy,
                                                                            self.class_priors_error))
                f.close()

        num_cores = min(multiprocessing.cpu_count(),num_cores)

        if num_cores == 1:
            for train_ind in trainsets:
                one_trial(train_ind)
        else:
            Parallel(n_jobs=num_cores)(delayed(one_trial)(train_ind) for train_ind in trainsets)

    def trials_statistics(self, tag=''):
        """Trials statistics
        ===================

        Loads accuracy scores from each trial from csv files created by `ssl_trials`
        and returns summary statistics (mean and standard deviation of accuracy).

        Parameters
        ----------
        tag : str (optional), default=''
            An extra identifying tag to add to the accuracy filename.

        Returns
        -------
        num_train : numpy array
            Number of training examples in each label rate experiment.
        acc_mean : numpy array
            Mean accuracy over all trials in each experiment.
        acc_stddev : numpy array
            Standard deviation of accuracy over all trials in each experiment.
        num_trials : int
            Number of trials for each label rate.
        """

        accuracy_filename = os.path.join(results_dir, tag+self.get_accuracy_filename())
        X = utils.csvread(accuracy_filename)
        num_train = np.unique(X[:,0])

        acc_mean = []
        acc_stddev = []
        for n in num_train:
            Y = X[X[:,0]==n,1:]
            acc_mean += [np.mean(Y,axis=0)]
            acc_stddev += [np.std(Y,axis=0)]

        num_trials = int(len(X[:,0])/len(num_train))
        acc_mean = np.array(acc_mean)
        acc_stddev = np.array(acc_stddev)
        return num_train, acc_mean, acc_stddev, num_trials

   
    def fit(self, train_ind, train_labels, all_labels=None):
        """Fit
        ======

        Solves graph-based learning problem, computing the probability 
        that each node belongs to each class. If `all_labels` is provided, 
        then the solver operates in verbose mode, printing out accuracy
        at each iteration.

        Parameters
        ----------
        train_ind : numpy array, int
            Indicies of training points.
        train_labels : numpy array, int
            Training labels as integers \\(0,1,\\dots,k-1\\) for \\(k\\) classes.
        all_labels : numpy array, int (optional)
            True labels for all datapoints.

        Returns
        -------
        u : (n,k) numpy array, float
            Per-class scores computed by graph-based learning for each node and each class. For some methods these are probabilities (i.e., for Laplace learning), while for others they can take on negative values (e.g., Poisson learning). The class label prediction is either the argmax or argmin over the rows of u (most methods are argmax, except distance-function based methods like nearest neighbor and peikonal).
        """

        if self.graph is None:
            sys.exit('SSL object has no graph. Use graph.set_graph() to provide a graph for SSL.')

        self.fitted = True

        #If a one-vs-rest classifier
        if self.onevsrest:
            unique_labels = np.unique(train_labels)
            num_labels = len(unique_labels)
            self.prob = np.zeros((self.graph.num_nodes,num_labels))
            for i,l in enumerate(unique_labels):
                self.prob[:,i] = self._fit(train_ind, train_labels==l)
        else:
            self.prob = self._fit(train_ind, train_labels, all_labels=all_labels)

        if self.class_priors is not None:
            self.volume_label_projection()

        return self.prob

    @abstractmethod
    def _fit(self, train_ind, train_labels, all_labels=None):
        """Internal Fit Function
        ======

        Internal fit function that any ssl object must override. If `self.onevsrest=True` then
        `train_labels` are binary in the one-vs-rest framework, and `_fit` must return
        a scalar numpy array. Otherwise the labels are integer and `_fit` must return an (n,k)
        numpy array of probabilities, where `k` is the number of classes.

        Parameters
        ----------
        train_ind : numpy array, int
            Indicies of training points.
        train_labels : numpy array, int
            Training labels as integers \\(0,1,\\dots,k-1\\) for \\(k\\) classes, or binary
            if `self.onevsrest=True`.
        all_labels : numpy array, int (optional)
            True labels for all datapoints.

        Returns
        -------
        u : numpy array, float
            (n,k) array of probabilities computed by graph-based learning for each node and each class, unless
            `self.onevsrest=True`, in which case it is a length n numpy array of probablities for the current 
            class.
        """
        raise NotImplementedError("Must override _fit")


class poisson(ssl):
    def __init__(self, W=None, class_priors=None, solver='conjugate_gradient', p=1, use_cuda=False, min_iter=50, max_iter=1000, tol=1e-3, spectral_cutoff=10):
        """Poisson Learning
        ===================

        Semi-supervised learning via the solution of the Poisson equation 
        \\[L^p u = \\sum_{j=1}^m \\delta_j(y_j - \\overline{y})^T,\\]
        where \\(L=D-W\\) is the combinatorial graph Laplacian, 
        \\(y_j\\) are the label vectors, \\(\\overline{y} = \\frac{1}{m}\\sum_{i=1}^m y_j\\) 
        is the average label vector, \\(m\\) is the number of training points, and 
        \\(\\delta_j\\) are standard basis vectors. See the reference for more details.

        Implements 3 different solvers, spectral, gradient_descent, and conjugate_gradient. 
        GPU acceleration is available for gradient descent. See [1] for details.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        solver : {'spectral', 'conjugate_gradient', 'gradient_descent'} (optional), default='conjugate_gradient'
            Choice of solver for Poisson learning.
        p : int (optional), default=1
            Power for Laplacian, can be any positive real number. Solver will default to 'spectral' if p!=1.
        use_cuda : bool (optional), default=False
            Whether to use GPU acceleration for gradient descent solver.
        min_iter : int (optional), default=50
            Minimum number of iterations of gradient descent before checking stopping condition.
        max_iter : int (optional), default=1000
            Maximum number of iterations of gradient descent.
        tol : float (optional), default=1e-3
            Tolerance for conjugate gradient solver.
        spectral_cutoff : int (optional), default=10
            Number of eigenvectors to use for spectral solver.

        Examples
        --------
        Poisson learning works on directed (i.e., nonsymmetric) graphs with the gradient descent solver: [poisson_directed.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/poisson_directed.py).
        ```py
        import numpy as np
        import graphlearning as gl
        import matplotlib.pyplot as plt
        import sklearn.datasets as datasets

        X,labels = datasets.make_moons(n_samples=500,noise=0.1)
        W = gl.weightmatrix.knn(X,10,symmetrize=False)

        train_ind = gl.trainsets.generate(labels, rate=5)
        train_labels = labels[train_ind]

        model = gl.ssl.poisson(W, solver='gradient_descent')
        pred_labels = model.fit_predict(train_ind, train_labels)

        accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, train_ind)   
        print("Accuracy: %.2f%%"%accuracy)

        plt.scatter(X[:,0],X[:,1], c=pred_labels)
        plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
        plt.show()
        ```

        Reference
        ---------
        [1] J. Calder, B. Cook, M. Thorpe, D. Slepcev. [Poisson Learning: Graph Based Semi-Supervised
        Learning at Very Low Label Rates.](http://proceedings.mlr.press/v119/calder20a.html), 
        Proceedings of the 37th International Conference on Machine Learning, PMLR 119:1306-1316, 2020.
        """
        super().__init__(W, class_priors)
        if solver not in ['conjugate_gradient', 'spectral', 'gradient_descent']:
            sys.exit("Invalid Poisson solver")
        self.solver = solver
        self.p = p
        if p != 1:
            self.solver = 'spectral'
        self.use_cuda = use_cuda
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.tol = tol
        self.spectral_cutoff = spectral_cutoff

        #Setup accuracy filename
        fname = '_poisson' 
        if self.p != 1:
            fname += '_p%.2f'%p
        if self.solver == 'spectral':
            fname += '_N%d'%self.spectral_cutoff
            self.requries_eig = True
        self.accuracy_filename = fname

        #Setup Algorithm name
        self.name = 'Poisson Learning'

    def _fit(self, train_ind, train_labels, all_labels=None):

        n = self.graph.num_nodes
        unique_labels = np.unique(train_labels)
        k = len(unique_labels)
        
        #Zero out diagonal for faster convergence
        W = self.graph.weight_matrix
        W = W - sparse.spdiags(W.diagonal(),0,n,n)
        G = graph.graph(W)

        #Poisson source term
        onehot = utils.labels_to_onehot(train_labels,k)
        source = np.zeros((n, onehot.shape[1]))
        source[train_ind] = onehot - np.mean(onehot, axis=0)
        
        if self.solver == 'conjugate_gradient':  #Conjugate gradient solver

            L = G.laplacian(normalization='normalized')
            D = G.degree_matrix(p=-0.5)
            u = utils.conjgrad(L, D*source, tol=self.tol)
            u = D*u

        elif self.solver == "gradient_descent":

            #Setup matrices
            D = G.degree_matrix(p=-1)
            P = D*W.transpose()
            Db = D*source

            #Invariant distribution
            v = np.zeros(n)
            v[train_ind]=1
            v = v/np.sum(v)
            deg = G.degree_vector()
            vinf = deg/np.sum(deg)
            RW = W.transpose()*D
            u = np.zeros((n,k))

            #Number of iterations
            T = 0
            if self.use_cuda:

                import torch 

                Pt = utils.torch_sparse(P).cuda()
                ut = torch.from_numpy(u).float().cuda()
                Dbt = torch.from_numpy(Db).float().cuda()

                while (T < self.min_iter or np.max(np.absolute(v-vinf)) > 1/n) and (T < self.max_iter):
                    ut = torch.sparse.addmm(Dbt,Pt,ut)
                    v = RW*v
                    T = T + 1

                #Transfer to CPU and convert to numpy
                u = ut.cpu().numpy()

            else: #Use CPU

                while (T < self.min_iter or np.max(np.absolute(v-vinf)) > 1/n) and (T < self.max_iter):
                    u = Db + P*u
                    v = RW*v
                    T = T + 1

                    #Compute accuracy if all labels are provided
                    if all_labels is not None:
                        self.prob = u
                        labels = self.predict()
                        acc = ssl_accuracy(labels,all_labels,train_ind)
                        print('%d,Accuracy = %.2f'%(T,acc))
                
        #Use spectral solver
        elif self.solver == 'spectral':

            vals, vecs = G.eigen_decomp(normalization='randomwalk', k=self.spectral_cutoff+1)
            V = vecs[:,1:]
            vals = vals[1:]
            if self.p != 1:
                vals = vals**self.p
            L = sparse.spdiags(1/vals, 0, self.spectral_cutoff, self.spectral_cutoff)
            u = V@(L@(V.T@source))

        else:
            sys.exit("Invalid Poisson solver " + self.solver)

        return u

class poisson_mbo(ssl):
    def __init__(self, W=None, class_priors=None, solver='conjugate_gradient', use_cuda=False, min_iter=50, max_iter=1000, tol=1e-3, spectral_cutoff=10, Ns=40, mu=1, T=20):
        """PoissonMBO 
        ===================

        Semi-supervised learning via Poisson MBO method [1]. class_priors must be provided.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). 
        solver : {'spectral', 'conjugate_gradient', 'gradient_descent'} (optional), default='conjugate_gradient'
            Choice of solver for Poisson learning.
        use_cuda : bool (optional), default=False
            Whether to use GPU acceleration for gradient descent solver.
        min_iter : int (optional), default=50
            Minimum number of iterations of gradient descent before checking stopping condition.
        max_iter : int (optional), default=1000
            Maximum number of iterations of gradient descent.
        tol : float (optional), default=1e-3
            Tolerance for conjugate gradient solver.
        spectral_cutoff : int (optional), default=10
            Number of eigenvectors to use for spectral solver.
        Ns : int (optional), default=40
            Number of inner iterations in PoissonMBO.
        mu : float (optional), default=1
            Fidelity parameter.
        T : int (optional), default=20
            Number of MBO iterations.
        
        Example
        -------
        Running PoissonMBO on MNIST at 1 label per class: [poisson_mbo.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/poisson_mbo.py).
        ```py
        import graphlearning as gl

        labels = gl.datasets.load('mnist', labels_only=True)
        W = gl.weightmatrix.knn('mnist', 10, metric='vae')

        num_train_per_class = 1
        train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
        train_labels = labels[train_ind]

        class_priors = gl.utils.class_priors(labels)
        model = gl.ssl.poisson_mbo(W, class_priors)
        pred_labels = model.fit_predict(train_ind,train_labels,all_labels=labels)

        accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,train_ind)
        print(model.name + ': %.2f%%'%accuracy)
        ```
        Reference
        ---------
        [1] J. Calder, B. Cook, M. Thorpe, D. Slepcev. [Poisson Learning: Graph Based Semi-Supervised
        Learning at Very Low Label Rates.](http://proceedings.mlr.press/v119/calder20a.html), 
        Proceedings of the 37th International Conference on Machine Learning, PMLR 119:1306-1316, 2020.
        """
        super().__init__(W, class_priors)

        self.poisson_model = poisson(W, solver=solver, use_cuda=use_cuda, min_iter=min_iter, 
                                     max_iter=max_iter, tol=tol, spectral_cutoff=spectral_cutoff)

        self.Ns = Ns
        self.mu = mu
        self.T = T
        self.use_cuda = use_cuda

        #Setup accuracy filename
        fname = '_poisson_mbo' 
        if solver == 'spectral':
            fname += '_N%d'%spectral_cutoff
            self.requries_eig = True
        fname += '_Ns_%d_mu_%.2f_T_%d'%(Ns,mu,T)
        self.accuracy_filename = fname

        #Setup Algorithm name
        self.name = 'Poisson MBO' 

    def _fit(self, train_ind, train_labels, all_labels=None):

        #Short forms
        Ns = self.Ns
        mu = self.mu
        T = self.T
        use_cuda = self.use_cuda

        n = self.graph.num_nodes
        unique_labels = np.unique(train_labels)
        k = len(unique_labels)

        #Zero out diagonal for faster convergence
        W = self.graph.weight_matrix
        W = W - sparse.spdiags(W.diagonal(),0,n,n)
        G = graph.graph(W)
        
        #Poisson source term
        onehot = utils.labels_to_onehot(train_labels,k)
        source = np.zeros((n, onehot.shape[1]))
        source[train_ind] = onehot - np.mean(onehot, axis=0)

        #Initialize via Poisson learning
        labels = self.poisson_model.fit_predict(train_ind, train_labels, all_labels=all_labels)
        u = utils.labels_to_onehot(labels,k)

        #Time step for stability
        dt = 1/np.max(G.degree_vector())

        #Precompute some things
        P = sparse.identity(n) - dt*G.laplacian()
        Db = mu*dt*source

        if use_cuda:
            import torch
            Pt = utils.torch_sparse(P).cuda()
            Dbt = torch.from_numpy(Db).float().cuda()

        for i in range(T):

            #Heat equation step
            if use_cuda:

                #Put on GPU and run heat equation
                ut = torch.from_numpy(u).float().cuda()
                for j in range(Ns):
                    ut = torch.sparse.addmm(Dbt,Pt,ut)

                #Put back on CPU
                u = ut.cpu().numpy()
             
            else: #Use CPU 
                for j in range(Ns):
                    u = P*u + Db

            #Projection step
            self.prob = u
            labels = self.volume_label_projection()
            u = utils.labels_to_onehot(labels,k)

            #Compute accuracy if all labels are provided
            if all_labels is not None:
                acc = ssl_accuracy(labels,all_labels,train_ind)
                print('%d, Accuracy = %.2f'%(i,acc))

        return u

class volume_mbo(ssl):
    def __init__(self, W=None, class_priors=None, temperature=0.1, volume_constraint=0.5):
        """Volume MBO
        ===================

        Semi-supervised learning with the VolumeMBO method [1]. class_priors must be provided.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array
            Class priors (fraction of data belonging to each class). 
        temperature : float (optional), default=0.1
            Temperature for volume constrained MBO.
        volume_constraint : float (optional), default=0.5
            The number of points in each class is constrained to be a mulitple \\(\\lambda\\) of the true
            class size, where 
            \\[ \\text{volume_constraint} \\leq \\lambda \\leq 2-\\text{volume_constraint}.\\]
            Setting `volume_constraint=1` yields the tightest constraint.

        References
        ----------
        [1] M. Jacobs, E. Merkurjev, and S. Esedoḡlu. [Auction dynamics: A volume constrained MBO scheme.](https://www.sciencedirect.com/science/article/pii/S0021999117308033?casa_token=kNahPd2vu50AAAAA:uJQYQVnmMBV_oL0CG1UcOIulY4vhclMGTztm-jjAzy9Lns7rtoOnKs4iyvLOjKXaHU-D6qrQJT4) Journal of Computational Physics 354:288-310, 2018. 
        """
        super().__init__(W, None)

        if class_priors is None:
            sys.exit("Class priors must be provided for Volume MBO.")
        self.class_counts = (self.graph.num_nodes*class_priors).astype(int)
        self.temperature = temperature
        self.volume_constraint = volume_constraint

        #Setup accuracy filename and model name
        self.accuracy_filename = '_volume_mbo_temp_%.2f_vol_%.2f'%(temperature,volume_constraint)
        self.name = 'Volume MBO (T=%.2f, V=%.2f)'%(temperature,volume_constraint)


    def _fit(self, train_ind, train_labels, all_labels=None):

        #Import c extensions
        from . import cextensions

        n = self.graph.num_nodes

        #Set diagonal entries to zero
        W = self.graph.weight_matrix
        W = W - sparse.spdiags(W.diagonal(),0,n,n)
        G = graph.graph(W)

        #Set up graph for C-code
        k = len(np.unique(train_labels))
        u = np.zeros((n,))
        WI,WJ,WV = sparse.find(W)

        #Type casting and memory blocking
        u = np.ascontiguousarray(u,dtype=np.int32)
        WI = np.ascontiguousarray(WI,dtype=np.int32)
        WJ = np.ascontiguousarray(WJ,dtype=np.int32)
        WV = np.ascontiguousarray(WV,dtype=np.float32)
        train_ind = np.ascontiguousarray(train_ind,dtype=np.int32)
        train_labels = np.ascontiguousarray(train_labels,dtype=np.int32)
        ClassCounts = np.ascontiguousarray(self.class_counts,dtype=np.int32)

        cextensions.volume_mbo(u,WJ,WI,WV,train_ind,train_labels,ClassCounts,k,1.0,self.temperature,self.volume_constraint)

        #Set given labels and convert to vector format
        u[train_ind] = train_labels
        u = utils.labels_to_onehot(u,k)
        return u

class multiclass_mbo(ssl):
    def __init__(self, W=None, class_priors=None, Ns=6, T=10, dt=0.15, mu=50, num_eig=50):
        """Multiclass MBO
        ===================

        Semi-supervised learning via the Multiclass MBO method [1].

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        Ns : int (optional), default=6
            Number of inner iterations.
        T : int (optional), default=10
            Number of outer iterations.
        dt : float (optional), default=0.15
            Time step.
        mu : float (optional), default=50
            Fidelity penalty.
        num_eig : int (optional), default=300
            Number of eigenvectors.
        
        References
        ---------
        [1] C. Garcia-Cardona, E. Merkurjev, A.L. Bertozzi, A. Flenner, and A.G. Percus. [Multiclass data segmentation using diffuse interface methods on graphs.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.743.9516&rep=rep1&type=pdf) IEEE transactions on pattern analysis and machine intelligence, 36(8), 1600-1613, 2014.
        """
        super().__init__(W, class_priors)

        self.Ns = Ns
        self.T = T
        self.dt = dt
        self.mu = mu
        self.num_eig = num_eig
        self.requires_eig = True

        #Setup accuracy filename
        self.accuracy_filename = '_multiclass_mbo_Ns_%d_T_%d_dt_%.3f_mu_%.2f'%(Ns,T,dt,mu)
        self.name = 'Multiclass MBO'


    def _fit(self, train_ind, train_labels, all_labels=None):

        #Shorten names
        Ns = self.Ns
        T = self.T
        dt = self.dt
        mu = self.mu
        num_eig = self.num_eig

        #Basic parameters
        n = self.graph.num_nodes
        k = len(np.unique(train_labels))

        #Spectral decomposition
        eigvals, X = self.graph.eigen_decomp(normalization='normalized', k=self.num_eig)

        #Form matrices
        V = np.diag(1/(1 + (dt/Ns)*eigvals)) 
        Y = X@V
        Xt = np.transpose(X)

        #Random initial labeling
        u = np.random.rand(k,n)
        u = utils.labels_to_onehot(np.argmax(u,axis=0),k).T
        u[:,train_ind] = utils.labels_to_onehot(train_labels,k).T

        #Indicator of train_ind
        J = np.zeros(n,)
        K = np.zeros(n,)
        J[train_ind] = 1
        K[train_ind] = train_labels
        K = utils.labels_to_onehot(K,k).T

        for i in range(T):
            #Diffusion step
            for s in range(Ns):
                Z = (u - (dt/Ns)*mu*J*(u - K))@Y
                u = Z@Xt
                
            #Projection step
            u = utils.labels_to_onehot(np.argmax(u,axis=0),k).T

            #Compute accuracy if all labels are provided
            if all_labels is not None:
                self.prob = u.T
                labels = self.predict()
                acc = ssl_accuracy(labels,all_labels,train_ind)
                print('Accuracy = %.2f'%acc)

        return u.T

class modularity_mbo(ssl):
    def __init__(self, W=None, class_priors=None, gamma=0.5, epsilon=1, lamb=1, T=20, Ns=5):
        """Modularity MBO
        ===================

        Semi-supervised learning via the Modularity MBO method [1].

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        gamma : float (optional), default=0.5
            Parameter in algorithm.
        epsilon : float (optional), default=1
            Parameter in algorithm.
        lamb : float (optional), default=1
            Parameter in algorithm.
        T : int (optional), default=20
            Number of outer iterations.
        Ns : int (optional), default=5
            Number of inner iterations.
        
        References
        ---------
        [1] Z.M. Boyd, E. Bae, X.C. Tai, and A.L. Bertozzi. [Simplified energy landscape for modularity using total variation.](https://publications.ffi.no/nb/item/asset/dspace:4288/1619750.pdf) SIAM Journal on Applied Mathematics, 78(5), 2439-2464, 2018.
        """
        super().__init__(W, class_priors)

        self.gamma = gamma
        self.epsilon = epsilon
        self.lamb = lamb
        self.requires_eig = True
        self.T = T
        self.Ns = Ns

        #Setup accuracy filename
        self.accuracy_filename = '_modularity_mbo_gamma_%.2f_epsilon_%.2f_lamb_%.2f'%(gamma,epsilon,lamb)
        self.name = 'Modularity MBO'

    def _fit(self, train_ind, train_labels, all_labels=None):

        #Short form
        gamma = self.gamma
        eps = self.epsilon
        lamb = self.lamb
        T = self.T
        Ns = self.Ns

        #One-hot initialization
        n = self.graph.num_nodes
        num_classes = len(np.unique(train_labels))
        train_onehot = utils.labels_to_onehot(train_labels,k)
        u = np.zeros((n,num_classes))
        u[train_ind,:] = train_onehot

        #Spectral decomposition
        num_eig = 5*num_classes
        D, V = self.graph.eigen_decomp(normalization='combinatorial', k=num_eig, gamma=gamma)

        #Time step selection
        deg = self.graph.degree_vector()
        dtlow = 0.15/((gamma+1)*np.max(deg))
        dthigh = np.log(np.linalg.norm(u)/eps)/D[0]
        dt = np.sqrt(dtlow*dthigh)

        #Diffusion matrix
        P = sparse.spdiags(np.exp(-D*dt),0,num_eig,num_eig)@V.T

        #Main MBO iterations
        for i in range(T):

            #Diffusion 
            u = V@(P@u)

            #Training labels
            if lamb > 0:
                for _ in range(Ns):
                    u[train_ind,:] -= (dt/Ns)*lamb*(u[train_ind,:] - train_onehot)

            #Threshold to labels
            labels = np.argmax(u,axis=1)
            
            #Convert to 1-hot vectors
            u = utils.labels_to_onehot(labels,num_classes)

            #Compute accuracy if all labels are provided
            if all_labels is not None:
                self.prob = u
                labels = self.predict()
                acc = ssl_accuracy(labels,all_labels,train_ind)
                print('Accuracy = %.2f'%acc)

        return u


class laplace(ssl):
    def __init__(self, W=None, class_priors=None, X=None, reweighting='none', normalization='combinatorial', 
                 tau=0, order=1, mean_shift=False, tol=1e-5, alpha=2, zeta=1e7, r=0.1):
        """Laplace Learning
        ===================

        Semi-supervised learning via the solution of the Laplace equation
        \\[\\tau u_j + L^m u_j = 0, \\ \\ j \\geq m+1,\\]
        subject to \\(u_j = y_j\\) for \\(j=1,\\dots,m\\), where \\(L=D-W\\) is the 
        combinatorial graph Laplacian and \\(y_j\\) for \\(j=1,\\dots,m\\) are the 
        label vectors. Default order is m=1, and m > 1 corresponds to higher order Laplace Learning.

        The original method was introduced in [1]. This class also implements reweighting 
        schemes `poisson` proposed in [2], `wnll` proposed in [3], and `properly`, proposed in [4].
        If `properly` is selected, the user must additionally provide the data features `X`.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        X : numpy array (optional)
            Data features, used to construct the graph. This is required for the `properly` weighted 
            graph Laplacian method.
        normalization : {'combinatorial','randomwalk','normalized'} (optional), defualt='combinatorial'
            Normalization for the graph Laplacian.
        reweighting : {'none', 'wnll', 'poisson', 'properly'} (optional), default='none'
            Reweighting scheme for low label rate problems. If 'properly' is selected, the user
            must supply the data features `X`.
        tau : float or numpy array (optional), default=0
            Zeroth order term in Laplace equation. Can be a scalar or vector.
        order : integer (optional), default=1
            Power m for higher order Laplace learning. Currently only integers are allowed. 
        mean_shift : bool (optional), default=False
            Whether to shift output to mean zero.
        tol : float (optional), default=1e-5
            Tolerance for conjugate gradient solver.
        alpha : float (optional), default=2
            Parameter for `properly` reweighting.
        zeta : float (optional), default=1e7
            Parameter for `properly` reweighting.
        r : float (optional), default=0.1
            Radius for `properly` reweighting.

        References
        ---------
        [1] X. Zhu, Z. Ghahramani, and J. D. Lafferty. [Semi-supervised learning using gaussian fields 
        and harmonic functions.](https://www.aaai.org/Papers/ICML/2003/ICML03-118.pdf) Proceedings 
        of the 20th International Conference on Machine Learning (ICML-03), 2003.

        [2] J. Calder, B. Cook, M. Thorpe, D. Slepcev. [Poisson Learning: Graph Based Semi-Supervised
        Learning at Very Low Label Rates.](http://proceedings.mlr.press/v119/calder20a.html), 
        Proceedings of the 37th International Conference on Machine Learning, PMLR 119:1306-1316, 2020.

        [3] Z. Shi, S. Osher, and W. Zhu. [Weighted nonlocal laplacian on interpolation from sparse data.](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/article/10.1007/s10915-017-0421-z&casa_token=33Z7gqJy3mMAAAAA:iMO0pGmpn_qf5PioVIGocSRq_p4CDm-KNOQhgIC1uvqG9pWlZ6t7I-IZtSJfocFDEHCdMpK8j7Fx1XbzDQ)
        Journal of Scientific Computing 73.2 (2017): 1164-1177.

        [4] J. Calder, D. Slepčev. [Properly-weighted graph Laplacian for semi-supervised learning.](https://link.springer.com/article/10.1007/s00245-019-09637-3) Applied mathematics & optimization (2019): 1-49.
        """
        super().__init__(W, class_priors)

        self.reweighting = reweighting
        self.normalization = normalization
        self.mean_shift = mean_shift
        self.tol = tol
        self.order = order
        self.X = X

        #Set up tau
        if type(tau) in [float,int]:
            self.tau = np.ones(self.graph.num_nodes)*tau
        elif type(tau) is np.ndarray:
            self.tau = tau

        #Setup accuracy filename
        fname = '_laplace' 
        self.name = 'Laplace Learning'
        if self.reweighting != 'none':
            fname += '_' + self.reweighting
            self.name += ': ' + self.reweighting + ' reweighted'
        if self.normalization != 'combinatorial':
            fname += '_' + self.normalization
            self.name += ' ' + self.normalization
        if self.mean_shift:
            fname += '_meanshift'
            self.name += ' with meanshift'
        if self.order > 1:
            fname += '_order%d'%int(self.order)
            self.name += ' order %d'%int(self.order)
        if np.max(self.tau) > 0:
            fname += '_tau_%.3f'%np.max(self.tau)
            self.name += ' tau=%.3f'%np.max(self.tau)

        self.accuracy_filename = fname



    def _fit(self, train_ind, train_labels, all_labels=None):

        #Reweighting
        if self.reweighting == 'none':
            G = self.graph
        else:
            W = self.graph.reweight(train_ind, method=self.reweighting, normalization=self.normalization, X=self.X)
            #W = self.graph.reweight(train_ind, method=self.reweighting, X=self.X)
            G = graph.graph(W)

        #Get some attributes
        n = G.num_nodes
        unique_labels = np.unique(train_labels)
        k = len(unique_labels)
        
        #tau + Graph Laplacian and one-hot labels
        L = sparse.spdiags(self.tau, 0, G.num_nodes, G.num_nodes) + G.laplacian(normalization=self.normalization)
        if self.order > 1:
            Lpow = L*L
            if self.order > 2:
                for i in range(2,self.order):
                    Lpow = L*Lpow
            L = Lpow
        F = utils.labels_to_onehot(train_labels,k)

        #Locations of unlabeled points
        idx = np.full((n,), True, dtype=bool)
        idx[train_ind] = False

        #Right hand side
        b = -L[:,train_ind]*F
        b = b[idx,:]

        #Left hand side matrix
        A = L[idx,:]
        A = A[:,idx]

        #Preconditioner
        m = A.shape[0]
        M = A.diagonal()
        M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()
       
        #Conjugate gradient solver
        v = utils.conjgrad(M*A*M, M*b, tol=self.tol)
        v = M*v

        #Add labels back into array
        u = np.zeros((n,k))
        u[idx,:] = v
        u[train_ind,:] = F

        #Mean shift
        if self.mean_shift:
            u -= np.mean(u,axis=0)

        return u

class dynamic_label_propagation(ssl):
    def __init__(self, W=None, class_priors=None, alpha=0.05, lam=0.1, T=2):
        """Dynamic Label Propagation
        ===================

        Semi-supervised learning via dynamic label propagation [1].

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        alpha : float (optional), default=0.05
            Value of parameter \\(\\alpha\\).
        lam : float (optional), default=0.1
            Value of parameter \\(\\lambda\\).
        T : int (optional), default=2
            Number of iterations.

        References
        ---------
        [1] B. Wang, Z. Tu, and J.K. Tsotsos. [Dynamic label propagation for semi-supervised multi-class multi-label classification.](https://openaccess.thecvf.com/content_iccv_2013/html/Wang_Dynamic_Label_Propagation_2013_ICCV_paper.html) Proceedings of the IEEE international conference on computer vision. 2013.
        """
        super().__init__(W, class_priors)

        self.alpha = alpha
        self.lam = lam
        self.T = T

        self.accuracy_filename = '_dynamic_label_propagation'
        self.name = 'Dynamic Label Propagation'
 

    def _fit(self, train_ind, train_labels, all_labels=None):

        #Short forms
        alpha = self.alpha
        lam = self.lam
        T = self.T
        n = self.graph.num_nodes
        k = len(np.unique(train_labels))

        #Zero out diagonal 
        W = self.graph.weight_matrix
        W = W - sparse.spdiags(W.diagonal(),0,n,n)
        G = graph.graph(W)
        
        #Labels to vector and correct position
        K = utils.labels_to_onehot(train_labels,k)
        u = np.zeros((n,k))
        u[train_ind,:] = K 
      
        if n > 5000:
            print("Cannot use Dynamic Label Propagation on large datasets.")
        else:
            #Setup matrices
            Id = sparse.identity(n) 
            D = G.degree_matrix(p=-1)
            P = D*W
            P = np.array(P.todense())
            Pt = np.copy(P)

            for i in range(T):
                v = P@u
                u = Pt@u
                u[train_ind,:] = K
                Pt = P@Pt@np.transpose(P) + alpha*v@np.transpose(v) + lam*Id

                #Compute accuracy if all labels are provided
                if all_labels is not None:
                    self.prob = np.array(u)
                    labels = self.predict()
                    acc = ssl_accuracy(labels,all_labels,train_ind)
                    print('Accuracy = %.2f'%acc)

            u = np.array(u)

        return u


class centered_kernel(ssl):
    def __init__(self, W=None, class_priors=None, tol=1e-10, power_it=100, alpha=1.05):
        """Centered Kernel Method
        ===================

        Semi-supervised learning via the centered kernel method of [1].

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        tol : float (optional), default=1e-10
            Tolerance to solve equation.
        power_it : int (optional), default=100
            Number of power iterations to find largest eigenvalue.
        alpha : float (optional), default = 1.05
            Value of \\(\\alpha\\) as a fraction of largest eigenvalue.

        References
        ---------
        [1] X. Mai and R. Couillet. [Random matrix-inspired improved semi-supervised learning on graphs.](https://romaincouillet.hebfree.org/docs/conf/SSL_ICML18.pdf) International Conference on Machine Learning. 2018.
        """
        super().__init__(W, class_priors)

        self.tol = tol
        self.power_it = power_it
        self.alpha = alpha

        self.accuracy_filename = '_centered_kernel'
        self.name = 'Centered Kernel'
        
    def _fit(self, train_ind, train_labels, all_labels=None):

        n = self.graph.num_nodes
        k = len(np.unique(train_labels))

        #Zero diagonal
        W = self.graph.weight_matrix
        W = W - sparse.spdiags(W.diagonal(),0,n,n)

        #Indicator of train_ind
        K = np.zeros((n,k))
        K[train_ind] = utils.labels_to_onehot(train_labels,k)
        
        #Center labels
        K[train_ind,:] -= np.sum(K,axis=0)/len(train_ind)

        #Initialization
        u = np.copy(K)
        v = np.ones((n,1))
        vt = np.ones((1,n))

        e = np.random.rand(n,1)
        for i in range(self.power_it):
            y = W*(e -  (1/n)*v@(vt@e))
            w = y - (1/n)*v@(vt@y) #=Ae
            l = abs(np.transpose(e)@w/(np.transpose(e)@e))
            e = w/np.linalg.norm(w)

        #Number of iterations
        alpha = self.alpha * l
        err = 1
        while err > self.tol:
            y = W*(u -  (1/n)*v@(vt@u))
            w = (1/alpha)*(y - (1/n)*v@(vt@y)) - u #Laplacian
            w[train_ind,:] = 0
            err = np.max(np.absolute(w))
            u = u + w

            #Compute accuracy if all labels are provided
            if all_labels is not None:
                self.prob = u
                labels = self.predict()
                acc = ssl_accuracy(labels,all_labels,train_ind)
                print('Accuracy = %.2f'%acc)
            
        return u


class sparse_label_propagation(ssl):
    def __init__(self, W=None, class_priors=None, T=100):
        """Sparse Label Propagation
        ===================

        Semi-supervised learning via the sparse label propagation method of [1].

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        T : int (optional), default=100
            Number of iterations

        References
        ---------
        [1] A. Jung, A.O. Hero III, A. Mara, and S. Jahromi. [Semi-supervised learning via sparse label propagation.](https://arxiv.org/abs/1612.01414) arXiv:1612.01414, 2016.
        """
        super().__init__(W, class_priors)

        self.T = T
        self.accuracy_filename = '_sparse_label_propagation'
        self.name = 'Sparse LP'
        
    def _fit(self, train_ind, train_labels, all_labels=None):

        n = self.graph.num_nodes
        k = len(np.unique(train_labels))

        #Sparse matrix with ones in all entries
        B = self.graph.adjacency()

        #Construct matrix 1/2W and 1/deg
        lam = 2*self.graph.weight_matrix - (1-1e-10)*B
        lam = -lam.log1p()
        lam = lam.expm1() + B
        Id = sparse.identity(n) 
        gamma = self.graph.degree_matrix(p=-1)

        #Random initial labeling
        u = np.zeros((k,n))

        #Indicator of train_ind
        one_hot_labels = utils.labels_to_onehot(train_labels,k).T

        #Initialization
        Y = list()
        for j in range(k):
            Gu = self.graph.gradient(u[j,:], weighted=True)
            Y.append(Gu)

        #Main loop for sparse label propagation
        for i in range(self.T):

            u_prev = np.copy(u)
            #Compute div
            for j in range(k):
                div = 2*self.graph.divergence(Y[j])
                u[j,:] = u_prev[j,:] - gamma*div
                u[j,train_ind] = one_hot_labels[j,:]  #Set labels
                u_tilde = 2*u[j,:] - u_prev[j,:]

                Gu = -self.graph.gradient(u_tilde, weighted=True)
                Y[j] = Y[j] + Gu.multiply(lam)
                ind1 = B.multiply(abs(Y[j])>1)
                ind2 = B - ind1
                Y[j] = ind1.multiply(Y[j].sign()) + ind2.multiply(Y[j])

            #Compute accuracy if all labels are provided
            if all_labels is not None:
                self.prob = u.T
                labels = self.predict()
                acc = ssl_accuracy(labels,all_labels,train_ind)
                print('%d,Accuracy = %.2f'%(i,acc))
 
        return u.T


class graph_nearest_neighbor(ssl):
    def __init__(self, W=None, class_priors=None, D=None, alpha=1):
        """Graph nearest neighbor classifier
        ===================

        Semi-supervised learning by graph (geodesic) nearest neighbor classifier. The 
        graph geodesic distance is defined by
        \\[ d_{ij} = \\min_p \\sum_{k=1}^M w_{p_k,p_{k+1}},\\]
        where the minimum is over paths \\(p\\) connecting nodes \\(i\\) and \\(j\\). The label
        returned for each testing point is the label of the closest labeled point.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        D : numpy array or scipy sparse matrix (optional)
            (n,n) Distance matrix, giving distances between neighbors. If provided,
            this is used to construct a knn density estimator for reweighting the eikonal equation.
        alpha : float (optional), default=1
            Reweighting exponent.
        """
        super().__init__(W, class_priors)


        self.alpha = alpha

        if class_priors is not None:
            self.onevsrest = True
            self.similarity = False

        #Cannot reweight if distance matrix not provided
        if D is None:
            self.f = 1
        else:
            d = D.max(axis=1).toarray().flatten() #distance to furtherest neighbor
            self.f = (d/np.max(d))**alpha

        #Setup accuracy filename and model name
        self.accuracy_filename = '_graph_nearest_neighbor_alpha%.2f'%(self.alpha) 
        self.name = 'Graph NN (alpha=%.2f)'%(self.alpha)

    def _fit(self, train_ind, train_labels, all_labels=None):

        if self.onevsrest:
            u = self.graph.dijkstra(train_ind[train_labels], bdy_val=0, f=self.f)
        else: 
            _,l = self.graph.dijkstra(train_ind, bdy_val=np.zeros_like(train_ind), f=self.f, return_cp=True)
            u = np.zeros(l.shape)
            u[train_ind] = train_labels
            k = len(np.unique(train_labels))
            u = utils.labels_to_onehot(u[l],k)

        return u

class amle(ssl):
    def __init__(self, W=None, class_priors=None, tol=1e-3, max_num_it=1e5, weighted=False, prog=False):
        """AMLE learning
        ===================

        Semi-supervised learning by the Absolutely Minimal Lipschitz Extension (AMLE). This is the same as
        p-Laplace with \\(p=\\infty\\), except that AMLE has an option `weighted=False` that significantly 
        accelerates the solver.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            (n,n) Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        tol : float (optional), default=1e-3
            Tolerance with which to solve the equation.
        max_num_it : int (optional), default=1e5
            Maximum number of iterations
        weighted : bool (optional), default=False
            When False, the graph is converted to a 0/1 adjacency matrix, which affords a much faster solver.
        prog : bool (optional), default=False
            Whether to print progress information.
        """
        super().__init__(W, class_priors)

        self.tol = tol
        self.max_num_it = max_num_it
        self.weighted = weighted
        self.prog = prog
        self.onevsrest = True

        #Setup accuracy filename and model name
        self.accuracy_filename = '_amle'
        if not self.weighted:
            self.accuracy_filename += '_unweighted'
        self.name = 'AMLE'


    def _fit(self, train_ind, train_labels, all_labels=None):

        u = self.graph.amle(train_ind, train_labels, tol=self.tol, max_num_it=self.max_num_it, 
                                                     weighted=self.weighted, prog=self.prog)
        return u

class peikonal(ssl):
    def __init__(self, W=None, class_priors=None, D=None, p=1, alpha=1, max_num_it=1e5, tol=1e-3, num_bisection_it=30, eps_ball_graph=False):
        """Graph p-eikonal classifier
        ===================

        Semi-supervised learning by via the solution of the graph `graph.peikonal` equation.

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            (n,n) Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        D : numpy array or scipy sparse matrix (optional)
            (n,n) Distance matrix, giving distances between neighbors. If provided,
            this is used to construct a knn density estimator for reweighting the p-eikonal equation.
        p : float (optional), default=1
            Value of exponent p in the p-eikonal equation
        alpha : float (optional), default=1
            Reweighting exponent.
        max_num_it : int (optional), default=1e5
            Maximum number of iterations.
        tol : float (optional), default=1e-3
            Tolerance with which to solve the equation.
        num_bisection_it : int (optional), default=30
            Number of bisection iterations for solver.
        eps_ball_graph : bool (optional), default=False
            Whether the graph is an epsilon-ball graph or not. If it is, then the density reweighting
            will be done with the degree vector, which is a kernel density estimator in this case
        """
        super().__init__(W, class_priors)

        self.p = p
        self.alpha = alpha
        self.max_num_it = max_num_it
        self.tol = tol
        self.num_bisection_it = num_bisection_it
        self.onevsrest = True
        self.similarity = False

        #Cannot reweight if distance matrix not provided
        if D is None:
            if eps_ball_graph:
                d = self.graph.degree_vector()
                self.f = (d/np.max(d))**(-alpha)
            else:
                self.f = 1
        else:
            d = D.max(axis=1).toarray().flatten() #distance to furtherest neighbor
            self.f = (d/np.max(d))**alpha

        #Setup accuracy filename and model name
        self.accuracy_filename = '_peikonal_p%.2f_alpha%.2f'%(self.p,self.alpha) 
        self.name = 'p-eikonal (p=%.2f, alpha=%.2f)'%(self.p,self.alpha)


    def _fit(self, train_ind, train_labels, all_labels=None):

        u = self.graph.peikonal(train_ind[train_labels], bdy_val=0, f=self.f, p=self.p, max_num_it=self.max_num_it, 
                                                tol=self.tol, num_bisection_it=self.num_bisection_it, prog=False)
        return u


class plaplace(ssl):
    def __init__(self, W=None, class_priors=None, p=10, max_num_it=1e6, tol=1e-1, fast=True):
        """Graph p-laplace classifier
        ===================

        Semi-supervised learning by via the solution of the game-theoretic p-Laplace equation [1].

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            (n,n) Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        p : float (optional), default=10
            Value of p in the p-Laplace equation (\\( 2 \\leq p \\leq \\infty\\).
        max_num_it : int (optional), default=1e5
            Maximum number of iterations
        tol : float (optional)
            Tolerance with which to solve the equation. Default tol=0.1 for fast=False and tol=1e-5 otherwise.
        fast : bool (optional), default=True
            Whether to use constant \\(w_{ij}=1\\) weights for the infinity-Laplacian
            which allows a faster algorithm to be used.

        References
        ----------
        [1] M. Flores Rios, J. Calder, and G. Lerman. [Algorithms for \\( \\ell_p\\)-based semi-supervised learning on graphs.](https://arxiv.org/abs/1901.05031) arXiv:1901.05031, 2019.
        """
        super().__init__(W, class_priors)

        self.p = p
        self.max_num_it = max_num_it
        self.tol = tol
        self.onevsrest = True
        self.fast = fast

        if fast:
            self.tol = 1e-5
        #Setup accuracy filename and model name
        self.accuracy_filename = '_plaplace_p%.2f'%self.p 
        self.name = 'p-Laplace (p=%.2f)'%self.p


    def _fit(self, train_ind, train_labels, all_labels=None):
        u = self.graph.plaplace(train_ind, train_labels, self.p, max_num_it=self.max_num_it, tol=self.tol,fast=self.fast)
        return u



class randomwalk(ssl):
    def __init__(self, W=None, class_priors=None, alpha=0.95):
        """Lazy random walk classification
        ===================

        Add description.

        The original method was introduced in [1], and can be interpreted as a lazy random walk. 

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        alpha : float (optional), default=0.95
            Parameter in model.

        References
        ---------
        [1] D. Zhou and B. Schölkopf. [Learning from labeled and unlabeled data using random walks.](https://link.springer.com/chapter/10.1007/978-3-540-28649-3_29) Joint Pattern Recognition Symposium. 
        Springer, Berlin, Heidelberg, 2004.
        """
        super().__init__(W, class_priors)

        self.alpha = alpha

        #Setup accuracy filename and model name
        self.accuracy_filename = '_randomwalk' 
        self.name = 'Lazy Random Walks'


    def _fit(self, train_ind, train_labels, all_labels=None):
        
        alpha = self.alpha

        #Zero diagonals
        n = self.graph.num_nodes
        W = self.graph.weight_matrix
        W = W - sparse.spdiags(W.diagonal(),0,n,n)
        G = graph.graph(W)

        #Construct Laplacian matrix
        L = (1-alpha)*sparse.identity(n) + alpha*G.laplacian(normalization='normalized')

        #Preconditioner
        m = L.shape[0]
        M = L.diagonal()
        M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()
       
        #Right hand side
        k = len(np.unique(train_labels))
        onehot = utils.labels_to_onehot(train_labels,k)
        Y = np.zeros((n, onehot.shape[1]))
        Y[train_ind,:] = onehot

        #Conjugate gradient solver
        u = utils.conjgrad(M*L*M, M*Y, tol=1e-6)
        u = M*u

        return u

def ssl_accuracy(pred_labels, true_labels, train_ind):   
    """SSL Accuacy
    ======

    Accuracy for semi-supervised graph learning, taking care to remove training set.
    NOTE: Any true labels with negative values will be removed from 
    accuracy computation.

    Parameters
    ----------
    pred_labels : numpy array, int
        Predicted labels
    true_labels : numpy array, int
        True labels
    train_ind : numpy array, int 
        Indices of training points, which will be removed from the accuracy computation.

    Returns
    -------
    accuracy : float
        Accuracy as a number in [0,100].
    """

        #Remove labeled data
    mask = np.ones(len(pred_labels),dtype=bool)
    if type(train_ind) != np.ndarray:
        print("Warning: ssl_accuracy has been updated and now requires the user to provide the indices of the labeled points, and not just the number of labels. Accuracy computation will be incorrect unless you update your code accordingly. See https://jwcalder.github.io/GraphLearning/ssl.html#two-moons-example")
    else:
        mask[train_ind]=False

    pred_labels = pred_labels[mask]
    true_labels = true_labels[mask]

    #Remove unlabeled nodes
    I = true_labels >=0
    pred_labels = pred_labels[I]
    true_labels = true_labels[I]

    #Compute accuracy
    return 100*np.mean(pred_labels==true_labels)


def accuracy_plot(model_list, tag='', testerror=False, savefile=None, title=None, errorbars=False, 
                  loglog=False, ylim=None, fontsize=16, legend_fontsize=16, label_fontsize=16):
    """Accuracy Plot
    ======

    Creates a plot of accuracy scores for experiments run with `ssl_trials`.

    Parameters
    ----------
    model_list : list of ssl objects
        Models to include in table.
    tag : str (optional), default=''
        An extra identifying tag to add to the accuracy filename.
    testerror : bool (optional), default=False
        Show test error (instead of accuracy) in the table.
    savefile : str (optional), default=None
        If a savefile name is provided, then the plot is saved instead of displayed to the screen
    title : str (optional), default=None
        If title is provided, then it will be added to the plot.
    errorbars : bool (optional), default=False
        Whether to add error bars to plot.
    loglog : bool (optional), default=False
        Make the plot on a loglog scale.
    ylim : tuple (optional), default=None
        If provided, then y-limits are set with `plt.ylim(ylim)`
    fontsize : int (optional), default=16
        Font size text, other than legend and labels
    legend_fontsize : int (optional), default=16
        Font size for legend.
    label_fontsize : int (optional), default=16
        Font size for x and y labels.
    """

    fig = plt.figure()
    if errorbars:
        matplotlib.rcParams.update({'errorbar.capsize': 5})
    matplotlib.rcParams.update({'font.size': fontsize})
    styles = ['^b-','or-','dg-','sk-','pm-','xc-','*y-']
    i = -1
    for model in model_list:
        i = (i+1)%7
        num_train,acc_mean,acc_stddev,num_trials = model.trials_statistics(tag=tag)
        if testerror:
            acc_mean = 100-acc_mean

        if errorbars:
            plt.errorbar(num_train, acc_mean[:,0], fmt=styles[i], yerr=acc_stddev[:,0], label=model.name)
        else:
            if loglog:
                plt.loglog(num_train, acc_mean[:,0], styles[i], label=model.name)
            else:
                plt.plot(num_train, acc_mean[:,0], styles[i], label=model.name) #3rd argumnet=styles[i]

        #If class priors were run as well
        if acc_mean.shape[1] > 1:
            i = (i+1)%7
            if errorbars:
                plt.errorbar(num_train, acc_mean[:,1], fmt=styles[i], yerr=acc_stddev[:,1], label=model.name+'+CP')
            else:
                if loglog:
                    plt.loglog(num_train, acc_mean[:,1], styles[i], label=model.name+'+CP')
                else:
                    plt.plot(num_train, acc_mean[:,1], styles[i], label=model.name+'+CP') #3rd argumnet=styles[i]

     
    plt.xlabel('Number of labels',fontsize=label_fontsize)
    if testerror:
        plt.ylabel('Test error (%)',fontsize=label_fontsize)
        plt.legend(loc='upper right',fontsize=legend_fontsize)
    else:
        plt.ylabel('Accuracy (%)',fontsize=label_fontsize)
        plt.legend(loc='lower right',fontsize=legend_fontsize)
    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.grid(True)

    if ylim is not None:
        plt.ylim(ylim)

    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()

def accuracy_table(model_list, tag='', testerror=False, savefile='table.tex', title='', 
                               append=False, fontsize='small', small_caps=True, two_column=True):
    """Accuracy table
    ======

    Creates a table of accuracy scores for experiments run with `ssl_trials` to be
    included in a LaTeX document.

    Parameters
    ----------
    model_list : list of ssl objects
        Models to include in table.
    tag : str (optional), default=''
        An extra identifying tag to add to the accuracy filename.
    testerror : bool (optional), default=False
        Show test error (instead of accuracy) in the table.
    savefile : str (optional), default='table.tex'
        Filename to save tex code for table.
    title : str (optional), default=''
        Title for table.
    append : bool (optional), default=False
        Whether to create a new tex file or append table to existing file.
    fontsize : str (optional), default='small'
        LaTeX font size.
    small_caps : bool (optional), default=True
        Whether to use small caps in LaTeX for table.
    two_column : bool (optional), default=True
        Whether the table will be in a two-column LaTeX document.
    """


    num_train,acc_mean,acc_stddev,num_trials = model_list[0].trials_statistics(tag=tag)
    m = len(num_train)

    #Determine best algorithm at each label rate
    best = [None]*m
    class_priors_best = [False]*m
    best_score = [0]*m
    for i, model in enumerate(model_list):
        num_train,acc_mean,acc_stddev,num_trials = model.trials_statistics(tag=tag)
        for j in range(m):
            if acc_mean[j,0] > best_score[j]:
                best_score[j] = acc_mean[j,0]
                best[j] = i

        #Check if class priors accuracy was included as well
        if acc_mean.shape[1] > 1:
            for j in range(m):
                if acc_mean[j,1] > best_score[j]:
                    best_score[j] = acc_mean[j,1]
                    class_priors_best[j] = True
                    best[j] = i

    
    if append:
        f = open(savefile,"r")
        lines = f.readlines()
        f.close()
        f = open(savefile,"w")
        f.writelines([item for item in lines[:-1]])
    else:
        f = open(savefile,"w")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[T1]{fontenc}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage[margin=1in]{geometry}\n")
        f.write("\\begin{document}\n")

    f.write("\n\n\n")
    if two_column:
        f.write("\\begin{table*}[t!]\n")
    else:
        f.write("\\begin{table}[t!]\n")
    f.write("\\vspace{-3mm}\n")
    f.write("\\caption{"+title+": Average (standard deviation) classification accuracy over %d trials.}\n"%num_trials)
    f.write("\\vspace{-3mm}\n")
    f.write("\\label{tab:"+title+"}\n")
    f.write("\\vskip 0.15in\n")
    f.write("\\begin{center}\n")
    f.write("\\begin{"+fontsize+"}\n")
    if small_caps:
        f.write("\\begin{sc}\n")
    f.write("\\begin{tabular}{l")
    for i in range(m):
        f.write("l")
    f.write("}\n")
    f.write("\\toprule\n")
    f.write("\\# Labels")
    for i in range(m):
        f.write("&\\textbf{%d}"%int(num_train[i]))
    f.write("\\\\\n")
    f.write("\\midrule\n")
    i = 0

    for i, model in enumerate(model_list):
        num_train,acc_mean,acc_stddev,num_trials = model.trials_statistics(tag=tag)

        f.write(model.name.ljust(15))
        for j in range(m):
            if best[j] == i and not class_priors_best[j]: 
                f.write("&{\\bf %.1f"%acc_mean[j,0]+" (%.1f)}"%acc_stddev[j,0])
            else:
                f.write("&%.1f"%acc_mean[j,0]+" (%.1f)      "%acc_stddev[j,0])
        f.write("\\\\\n")
        
        #Check if class priors accuracy was included as well
        if acc_mean.shape[1] > 1:
            f.write((model.name+'+CP').ljust(15))
            for j in range(m):
                if best[j] == i and class_priors_best[j]: 
                    f.write("&{\\bf %.1f"%acc_mean[j,1]+" (%.1f)}"%acc_stddev[j,1])
                else:
                    f.write("&%.1f"%acc_mean[j,1]+" (%.1f)      "%acc_stddev[j,1])
            f.write("\\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    if small_caps:
        f.write("\\end{sc}\n")
    f.write("\\end{"+fontsize+"}\n")
    f.write("\\end{center}\n")
    f.write("\\vskip -0.1in\n")
    if two_column:
        f.write("\\end{table*}")
    else:
        f.write("\\end{table}")
    f.write("\n\n\n")
    f.write("\\end{document}\n")
    f.close()






