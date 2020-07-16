import graphlearning as gl
import numpy as np
import sys, getopt
from sklearn.decomposition import PCA



def print_help():
    
    print('=======================================================')
    print('GraphLearning: Python package for graph-based learning.')
    print('=======================================================')
    print('=======================================================')
    print('Compute K-nearest neighbors')
    print('=======================================================')
    print('                                                       ')
    print('Options:')
    print('   -d (--dataset=): MNIST, FashionMNIST,...more soon (default=MNIST)')
    print('   -m (--metric=):  Metric for computing similarities (default=L2)')
    print('          Choices:  scatter, L2, scatter_pca')
    print('   -s (--pca=): Number dimensions for scatter_pca (default=100)')
    print('   -k (--knn=): Number of nearest neighbors (default=30)')
    print('   -n (--knn_method=): Method for computing neighbors (default=annoy)')
    print('              Choices: annoy, exact')

# Print basic info
def print_info(dataset, metric, k, knn_method, scatter_pca_dims):

    if metric == "scatter_pca":
        outfile = "kNNData/"+dataset+"_"+metric + "_" + str(scatter_pca_dims) + ".npz"
    else:
        outfile = "kNNData/"+dataset+"_"+metric+".npz"
    print('=======================================================')
    print('GraphLearning: Python package for graph-based learning.')
    print('=======================================================')
    print('=======================================================')
    print('Compute K-nearest neighbors')
    print('=======================================================')
    print('                                                       ')
    print('Dataset: ' + dataset)
    print('Metric: ' + metric)
    if metric=='scatter_pca':
        print('Number of scattering dimensions: %d' % scatter_pca_dims)
    print('Number of neighbors: %d' % k)
    print('kNN search method: ' + knn_method)
    print('Output file: ' + outfile)
    print('                                                       ')
    print('=======================================================')
    print('                                                       ')


def ComputeKNN(dataset, metric='L2', k=30, knn_method='annoy', scatter_pca_dims=100):

    if metric == "scatter_pca":
        outfile = "kNNData/"+dataset+"_"+metric + "_" + str(scatter_pca_dims) + ".npz"
    else:
        outfile = "kNNData/"+dataset+"_"+metric+".npz"

    #For variational autoencoder the vae data, e.g., Data/MNIST_vae.npz must exist.
    if metric[0:3]=='vae' or metric[0:3]=='aet':
        dataFile = "Data/"+dataset+"_"+metric+".npz"
    else:
        dataFile = "Data/"+dataset+"_raw.npz"

    #Try to Load data
    try:
        M = np.load(dataFile,allow_pickle=True)
    except:
        print('Cannot find '+dataFile+'.')
        sys.exit(2)

    data = M['data']

    #Apply transformations (just scatter now, but others could be included)
    if metric == 'scatter' or metric == 'scatter_pca':
        if metric == 'scatter_pca' and scatter_pca_dims <= 300: 
            #Changed to Data
            filePath = "Data/" + dataset + "_" + "scatter_pca" + ".npz"
            try:
                PCAfile = np.load(filePath)
                savedPCA = PCAfile['savedPCA']
            except:
                print("File not found: " + filePath)
                print("Recomputing " + filePath) 
                m = int(np.sqrt(data.shape[1]))  # number of pixels across image (assuming square)
                Y = gl.scattering_transform(data, m, m)
                print("Computing PCA...")
                pca = PCA(n_components=300)
                savedPCA = pca.fit_transform(Y)
                np.savez_compressed(filePath, savedPCA=savedPCA)
            pca = PCA(n_components=scatter_pca_dims)
            data = pca.fit_transform(savedPCA)
        else:
            print("Computing scattering transform...")
            m = int(np.sqrt(data.shape[1]))  
            data = gl.scattering_transform(data, m, m)

    #Perform kNN search
    if knn_method == 'annoy':
        if metric in ['angular', 'manhattan', 'hamming', 'dot']:
            similarity = metric
        else:
            similarity = 'euclidean'
        
        if metric[0:3] == 'aet':
            similarity = 'angular'

        # Similarity can be "angular", "euclidean", "manhattan", "hamming", or "dot".
        I,J,D = gl.knnsearch_annoy(data,k, similarity) 
    elif knn_method == 'exact':
        I,J,D = gl.knnsearch(data,k)
    else:
        print('Invalid kNN method.')
        return

    #Save kNN results to file
    np.savez_compressed(outfile,I=I,J=J,D=D)

        
if __name__ == "__main__":

    # Default settings
    dataset = 'MNIST'
    metric = 'L2'
    k=30
    scatter_pca_dims = 100
    knn_method='annoy'

    # Read command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:m:k:s:n:", ["dataset=", "method=", "knn=", "scatter_dims=","knn_method="])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-m", "--method"):
            metric = arg
        elif opt in ("-k", "--knn"):
            k = int(arg)
        elif opt in ("-s", "--scatter_dims"):
            scatter_pca_dims = int(arg)
        elif opt in ("-n", "--knn_method"):
            knn_method = arg

    print_info(dataset, metric, k, knn_method, scatter_pca_dims)
    ComputeKNN(dataset, metric, k, knn_method, scatter_pca_dims)





