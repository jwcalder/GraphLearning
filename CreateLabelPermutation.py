import graphlearning as gl
import numpy as np
import sys, getopt
import os.path as path

def print_help():
    
    print('=======================================================')
    print('GraphLearning: Python package for graph-based learning.')
    print('=======================================================')
    print('=======================================================')
    print('Create Label Permutation')
    print('=======================================================')
    print('                                                       ')
    print('Options:')
    print('   -d (--dataset=): MNIST, FashionMNIST,...more soon (default=MNIST)')
    print('   -m (--NumLabels=): Number of labels per class for each trial (default=1,2,3,4,5)')
    print('   -t (--NumTrials=): Number of trials (default=100)')
    print('   -n (--name=):   Permutation name in form dataset<name>_permutations.npz (default is empty)')
    print('   -s (--multiplier=): List of multipliers for each class, to produce unbalanced experiments (default is balanced 1,1,1,1,1)')
    print('   -o (--overwrite=): Overwrite existing file.')


#Default settings
dataset = 'MNIST'
m = '1,2,3,4,5'
multiplier = None
t = 100
name = ''
overwrite = False

#Read command line arguments
try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:m:t:n:s:o",["dataset=","NumLabels=","NumTrials=","name=","multiplier=","overwrite"])
except getopt.GetoptError:
    print_help()
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt in ("-d", "--dataset"):
        dataset = arg
    elif opt in ("-m", "--NumLabels"):
        m = arg
    elif opt in ("-t", "--NumTrials"):
        t = int(arg)
    elif opt in ("-s", "--multiplier"):
        multiplier = arg
        multiplier = [float(e)  for e in multiplier.split(',')]
    elif opt in ("-n", "--name"):
        name = arg
    elif opt in ("-o", "--overwrite"):
        overwrite = True

outfile = "LabelPermutations/"+dataset+name+"_permutations.npz"

#Print basic info
print('=======================================================')
print('GraphLearning: Python package for graph-based learning.')
print('=======================================================')
print('=======================================================')
print('Compute Label Permutations')
print('=======================================================')
print('                                                       ')
print('Dataset: '+dataset)
print('Number of Labels per trial: '+m)
print('Number of Trials: %d'%t)
print('Output file: '+outfile)
print('                                                       ')
print('=======================================================')
print('                                                       ') 

#Load labels
try:
    M = np.load("Data/"+dataset+"_labels.npz",allow_pickle=True)
except:
    print('Cannot find dataset Data/'+dataset+'_labels.npz')
    sys.exit(2)

#Extract labels
labels = M['labels']

#Convert string to int list
m = [int(e)  for e in m.split(',')]

#Create label permutations
perm = gl.create_label_permutations(labels,t,m,multiplier)

#Save weight matrix to file
if path.isfile(outfile) and not overwrite:
    print('Output file: '+outfile+' already exists. Aborting...')
    sys.exit(2)
else:
    np.savez_compressed(outfile,perm=perm)
