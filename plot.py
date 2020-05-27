import graphlearning as gl

#Example of how to generate accuracy plots 

ssl_method_list = ['laplace','poisson']
legend_list = ['Laplace learning','Poisson learning']

#Select dataset to plot
dataset = "MNIST_vae_k10"

num_classes = 10
gl.accuracy_plot(dataset,ssl_method_list,legend_list,num_classes,errorbars=False,testerror=False,loglog=False)


