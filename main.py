#Main script for graph-based semi-supervised learning
import graphlearning as gl
import numpy as np
import datetime
import sys, getopt
import time
from joblib import Parallel, delayed
import multiprocessing
import torch
import os

clustering_algorithms = ['incres','spectral','spectralshimalik','spectralngjordanweiss']

def print_help():
    
    print('========================================================')
    print('GraphLearning: Python package for graph-based learning. ')
    print('========================================================')
    print('========================================================')
    print('Graph-based Clustering & Semi-Supervised Learning')
    print('========================================================')
    print('                                                        ')
    print('Options:')
    print('   -d (--dataset=): MNIST, FashionMNIST, WEBKB, cifar (default=MNIST)')
    print('   -m (--metric=):  Metric for computing similarities (default=L2)')
    print('          Choices:  vae, scatter, L2, aet')
    print('   -a (--algorithm=): Learning algorithm (default=Laplace)')
    print('   -k (--knn=): Number of nearest neighbors (default=10)')
    print('   -t (--num_trials=): Number of trial permutations to run (default=all)')
    print('   -l (--label_perm=): Choice of label permutation file (format=dataset<label_perm>_permutations.npz). (default is empty).')
    print('   -p (--p=): Value of p for plaplace method (default=3)')
    print('   -n (--normalization=): Laplacian normalization (default=none)')
    print('                 Choices: none, normalized')
    print('   -N (--num_classes): Number of clusters if choosing clustering algorithm (default=10)')
    print('   -s (--speed=): Speed in INCRES method (1--10) (default=2)')
    print('   -i (--num_iter=): Number of iterations for iterative methods (default=1000)')
    print('   -x (--extra_dim=): Number of extra dimensions in spectral clustering (default=0)')
    print('   -c (--cuda): Use GPU acceleration (when available)')
    print('   -T (--temperature): Temperature for volume constrained MBO (default=0)')
    print('   -v (--volume_constraint=): Volume constraint for MBO (default=0.5)')
    print('   -j (--num_cores=): Number of cores to use in parallel processing (default=1)')
    print('   -r (--results): Turns off automatic saving of results to .csv file')
    print('   -b (--verbose): Turns on verbose mode (displaying more intermediate steps).')

def print_info():

    #Print basic info
    print('========================================================')
    print('GraphLearning: Python package for graph-based learning. ')
    print('========================================================')
    print('========================================================')
    print('Graph-based Clustering & Semi-Supervised Learning')
    print('========================================================')
    print('                                                        ')
    print('Dataset: '+dataset)
    print('Metric: '+metric)
    print('Number of neighbors: %d'%k)
    print('Learning algorithm: '+algorithm)
    print('Laplacian normalization: '+norm)
    if algorithm == 'plaplace' or algorithm == 'eikonal':
        print("p-Laplace/eikonal value p=%.2f" % p)
    if algorithm in clustering_algorithms:
        print('Number of clusters: %d'%num_classes)
        if algorithm == 'INCRES':
            print('INCRES speed: %.2f'%speed)
            print('Number of iterations: %d'%num_iter)
        if algorithm[:8] == 'Spectral':
            print('Number of extra dimensions: %d'%extra_dim)
    else:
        print('Number of trial permutations: %d'%len(perm))
        print('Permutations file: LabelPermutations/'+dataset+label_perm+'_permutations.npz')

        if algorithm == 'volumembo' or algorithm == 'poissonvolumembo':
            print("Using temperature=%.3f"%T)
            print("Volume constraints = [%.3f,%.3f]"%(volume_constraint,2-volume_constraint))

        #If output file selected
        if results:
            print('Output file: '+outfile)

    print('                                                        ')
    print('========================================================')
    print('                                                        ')

#Default settings
dataset = 'MNIST'
metric = 'L2'
algorithm = 'laplace'
k = 10
t = '-1'
label_perm = ''
p = 3
norm = "none"
use_cuda = False
T = 0
num_cores = 1
results = True
num_classes = 10
speed = 2
num_iter = 1000
extra_dim = 0
volume_constraint = 0.5
verbose = False
poisson_training_balance = True

#Read command line arguments
try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:m:k:a:p:n:v:N:s:i:x:t:cl:T:j:rbo",["dataset=","metric=","knn=","algorithm=","p=","normalization=","volume_constraint=","num_classes=","speed=","num_iter=","extra_dim=","num_trials=","cuda","label_perm=","temperature=","--num_cores=","results","verbose","poisson_training_balance"])
except getopt.GetoptError:
    print_help()
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt in ("-d", "--dataset"):
        dataset = arg
    elif opt in ("-m", "--metric"):
        metric = arg
    elif opt in ("-k", "--knn"):
        k = int(arg)
    elif opt in ("-a", "--algorithm"):
        algorithm = arg.lower()
    elif opt in ("-p", "--p"):
        p = float(arg)
    elif opt in ("-n", "--normalization"):
        norm = arg
    elif opt in ("-v", "--volume_constraint"):
        volume_constraint = float(arg)
    elif opt in ("-N", "--num_classes"):
        num_classes = int(arg)
    elif opt in ("-s", "--speed"):
        speed = float(arg)
    elif opt in ("-i", "--num_iter"):
        num_iter = int(arg)
    elif opt in ("-x", "--extra_dim"):
        extra_dim = int(arg)
    elif opt in ("-t", "--num_trials"):
        t = arg
    elif opt in ("-c", "--cuda"):
        use_cuda = True
    elif opt in ("-l", "--label_perm"):
        label_perm = arg
    elif opt in ("-T", "--temperature"):
        T = float(arg)
    elif opt in ("-j", "--num_cores"):
        num_cores = int(arg)
    elif opt in ("-r", "--results"):
        results = False
    elif opt in ("-b", "--verbose"):
        verbose = True
    elif opt in ("-o", "--poisson_training_balance"):
        poisson_training_balance = False

#Load labels
try:
    M = np.load("Data/"+dataset+"_labels.npz",allow_pickle=True)
    labels = M['labels']
except:
    print('Cannot find dataset Data/'+dataset+'_labels.npz')
    sys.exit(2)


#Load kNN data and build weight matrix
try:
    M = np.load("kNNData/"+dataset+"_"+metric+".npz",allow_pickle=True)
    I = M['I']
    J = M['J']
    D = M['D']
    W = gl.weight_matrix(I,J,D,k)
    Wdist = gl.dist_matrix(I,J,D,k)
except:
    print('Cannot find kNNData/'+dataset+'_'+metric+'.npz')
    print('You need to run ComputeKNN.py.')
    sys.exit(2)


#Load label permutations
try:
    M = np.load("LabelPermutations/"+dataset+label_perm+"_permutations.npz",allow_pickle=True)
    perm = M['perm']
except:
    print('Cannot find LabelPermutations/'+dataset+label_perm+'_permutations.npz')
    print('You need to run CreateLabelPermutation.py first.')
    sys.exit(2)

#Restrict trials
t = [int(e)  for e in t.split(',')]
if t[0] > -1:
    if len(t) == 1:
        perm = perm[0:t[0]]
    else:
        perm = perm[(t[0]-1):t[1]]

#Load eigenvector data if MBO selected
eigvals = None
eigvecs = None
if algorithm == 'mbo':
    try:
        M = np.load("MBOdata/"+dataset+"_"+metric+"_k%d"%k+"_spectrum.npz")
        eigvals = M['eigenvalues']
        eigvecs = M['eigenvectors']
    except:
        print("Could not find MBOdata/"+dataset+"_"+metric+"_k%d"%k+"_spectrum.npz")
        print('You need to run ComputeEigenvectorsMBO.py first.')
        sys.exit(2)
#Output file
outfile = "Results/"+dataset+label_perm+"_"+metric+"_k%d"%k
if algorithm == 'plaplace':
    outfile = outfile+"_p%.1f"%p+algorithm[1:]+"_"+norm
elif algorithm == 'eikonal':
    outfile = outfile+"_p%.1f"%p+algorithm
else:
    outfile = outfile+"_"+algorithm

if algorithm == 'volumembo' or algorithm == 'poissonvolumembo':
    outfile = outfile+"_T%.3f"%T
    outfile = outfile+"_V%.3f"%volume_constraint

if algorithm == 'poisson' and poisson_training_balance == False:
    outfile = outfile+"_NoBal"

outfile = outfile+"_accuracy.csv"

#Print information 
print_info()

true_labels = None
if verbose:
    true_labels = labels

#If clustering algorithm was chosen
if algorithm in clustering_algorithms:
    #Clustering
    u = gl.graph_clustering(W,num_classes,labels,method=algorithm,T=num_iter,speed=speed,extra_dim=extra_dim)

    #Compute accuracy
    accuracy = gl.clustering_accuracy(u,labels)

    #Print to terminal
    print("Accuracy: %.2f" % accuracy+"%")

#If semi-supervised algorithms chosen
else:
    #If output file selected
    if results:
        #Check if Results directory exists
        if not os.path.exists('Results'):
            os.makedirs('Results')

        now = datetime.datetime.now()
        
        #Add time stamp to output file
        f = open(outfile,"a+")
        f.write("Date/Time, "+now.strftime("%Y-%m-%d_%H:%M")+"\n")
        f.close()



    #Loop over label permutations
    print("Number of labels, Accuracy")

    def one_trial(label_ind):

        #Number of labels
        m = len(label_ind)

        #Label proportions (used by some algroithms)
        beta = gl.label_proportions(labels)

        start_time = time.time()
        #Graph-based semi-supervised learning
        u = gl.graph_ssl(W,label_ind,labels[label_ind],D=Wdist,beta=beta,method=algorithm,epsilon=0.3,p=p,norm=norm,eigvals=eigvals,eigvecs=eigvecs,dataset=dataset,T=T,use_cuda=use_cuda,volume_mult=volume_constraint,true_labels=true_labels,poisson_training_balance=poisson_training_balance)
        print("--- %s seconds ---" % (time.time() - start_time))

        #Compute accuracy
        accuracy = gl.accuracy(u,labels,m)
        
        #Print to terminal
        print("%d" % m + ",%.2f" % accuracy)

        #Write to file
        if results:
            f = open(outfile,"a+")
            f.write("%d" % m + ",%.2f\n" % accuracy)
            f.close()

    #Number of cores for parallel processing
    num_cores = min(multiprocessing.cpu_count(),num_cores)
    Parallel(n_jobs=num_cores)(delayed(one_trial)(label_ind) for label_ind in perm)


