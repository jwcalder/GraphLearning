import graphlearning as gl
import os

#Example of how to generate accuracy tables

ssl_method_list = ['laplace','poisson']
legend_list = ['Laplace learning','Poisson learning']

#Select dataset to plot
dataset = "MNIST_vae_k10"

num_classes = 10
gl.accuracy_table_icml(dataset,ssl_method_list,legend_list,num_classes,savefile='example_table.tex',title="Laplace vs Poisson Learning",quantile=False,append=False)

#Compile tex file
os.system("pdflatex example_table.tex")
os.system("pdflatex example_table.tex")

