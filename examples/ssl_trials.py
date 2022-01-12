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
