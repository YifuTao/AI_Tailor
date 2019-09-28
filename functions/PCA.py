import numpy as np
import os
from scipy.io import loadmat
from sklearn.decomposition import PCA


# load data
X = loadmat('/home/yifu/workspace/Data/MPII Human Shape/caesar-norm-nh-fitted-meshes/csr4771a.mat')
# PCA
n_components = 10
pca = PCA(n_components=n_components)
pca.fit(X)
mean = pca.mean_
var = pca.explained_variance_
sigma = np.sqrt(var)
# sampling

n_samples = 1000
codes = np.expand_dims(sigma,1) * np.random.randn(n_components,n_samples)  # eigenvectors

# create mesh 
#  transpose some stuff to get the right dimension
#  n meshes = mean + codes dot pca components 

# Project
codes = meshes * pca.components_[:n_components,:]

# Reconstruct
samples = np.dot(np.transpose(codes), pca.components_[:n_components,:])
samples += mean
samples = np.reshape(samples,(n_samples,-1,3))