#ssl_trials.py
#
#This script shows how to use the ssl_trials
#function to run many trials of random test/train splits
#to compare various SSL algorithms. The code automatically
#generates LaTeX tables and plots (the files SSL_MNIST.png/pdf 
#that are included in the examples folder).
#
#The code also shows how to write a new SSL algorithm 'alg_name' and 
#plug it into the ssl_trials environment. The code will look for a file
#alg_name.py and import and run the function 'ssl' in this file. 
#See alg_name.py for details.
#
#ssl_trials supports parallel processing, via num_cores=

import graphlearning as gl
import os

dataset = 'mnist'
metric = 'vae' #Uses variational autoencoder to consruct graph
num_classes = 10
algorithm_list = ['laplace','poisson','nearestneighbor','volumembo','alg_name']
results_files = []

#Create a new label permutation, with 100 randomized trials at 1,2,4,8,16 labels per class
gl.create_label_permutations(gl.load_labels(dataset),100,[1,2,4,8,16],dataset='mnist',name='new',overwrite=True)

#Parameters specific to the new algorithm alg_name 
params = {'lambda':1} 

#Run experiments (we'll just do t=10 trials to save time)
for alg in algorithm_list:
    results = gl.ssl_trials(dataset=dataset,metric=metric,algorithm=alg,num_cores=2,t=10,label_perm='new',params=params)
    results_files.append(results)

#Generate plots
legend_list = ['Laplace learning','Poisson learning','Nearest Neighbor','VolumeMBO','alg name']
gl.accuracy_plot(results_files,legend_list,num_classes,title='SSL Comparison: MNIST',errorbars=False,testerror=False,loglog=False,savefile='SSL_MNIST.png')

#Generate a table showing accuracy scores
gl.accuracy_table_icml(results_files,legend_list,num_classes,savefile='SSL_MNIST.tex',title="SSL Comparison: MNIST",quantile=False,append=False)
os.system("pdflatex SSL_MNIST.tex")

