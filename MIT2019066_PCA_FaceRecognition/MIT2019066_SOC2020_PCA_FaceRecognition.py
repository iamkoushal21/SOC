#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: iamkoushal21
"""

import numpy as np
from sklearn.metrics import accuracy_score

N = 10
def data_load(filename):
    data = np.load(filename)
    N = 10
    data = data[0: N * 10].reshape(N * 10, -1) 
    
    return data

def train_test_split(data):
    arr = np.arange(N * 10)
    X_train_data = data[arr % 10 < 6]
    X_test_data = data[arr % 10 >= 6]
    X_train_data_mean = np.mean(X_train_data,axis=0)
    X_train_data = X_train_data - X_train_data_mean

    return X_train_data, X_test_data, X_train_data_mean

def eigen(cov):
    return np.linalg.eig(cov)
    
def feature_vector(X_train_data):
    m,n = X_train_data.shape
    cov = (X_train_data @ X_train_data.T) / n
    eigen_val, eigen_vec = eigen(cov)
    eig_dict = dict(zip(eigen_val, eigen_vec))
    components = 50
    feature_vec = np.zeros((components, m))
    for n,i in enumerate(sorted(eig_dict, reverse = True)):
        if (n < components):
            feature_vec[n] += eig_dict[i]
    
    return feature_vec

data = data_load('olivetti_faces.npy')
X_train_data, X_test_data, X_train_data_mean = train_test_split(data)
feature_vec = feature_vector(X_train_data)
eigen_faces = feature_vec @ X_train_data
sign_faces = eigen_faces @ X_train_data.T


X_test_data = np.column_stack((np.repeat(np.arange(N),4), X_test_data))
np.random.shuffle(X_test_data)

Y = X_test_data
X_test_data = X_test_data[:, 1:]
j, k = X_test_data.shape

pred_y = np.zeros(j)

for n,img in enumerate(X_test_data):
    proj = eigen_faces @ (img - X_train_data_mean).T
    dist = (sign_faces.T - proj)
    dist = np.linalg.norm(dist.T,axis = 0)
    index = np.argmin(dist)
    pred_y[n] = np.floor(index/6)


print("Accuracy for PCA Face-Recognition, Using Euclidean is = ", accuracy_score(Y.T[0],pred_y))
