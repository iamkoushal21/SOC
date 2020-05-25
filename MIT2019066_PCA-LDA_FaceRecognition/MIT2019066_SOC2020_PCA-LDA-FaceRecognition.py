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
    data = data[0:N*10].reshape(N*10,-1) 
    
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

img_class = {}
for a in range(10):
    img_class[a]=((sign_faces.T)[a * 6:(a+1) * 6,:])
mean_class = np.array([np.mean(img_class[i], axis = 0) for i in sorted(img_class)])
mean_proj =  np.mean(sign_faces.T, axis = 0) 


cov_within_class = {}
for a in sorted(img_class):
    temp1 = img_class[a] - mean_class[a]
    cov_within_class[a] = temp1.T @ temp1
j = cov_within_class[0].shape
SW = np.zeros(j)
for a in sorted(cov_within_class):
    SW += cov_within_class[a] 
SB = np.zeros(SW.shape)
for a in sorted(img_class):
    temp2 = (mean_class[a] - mean_proj)
    SB += (temp2.T @ temp2)
    
    
J = (np.linalg.inv(SW)) @ SB  
eig_val, eig_vec = eigen(J)
comp = 25

order = np.flip(np.argsort(eig_val))
eig_vec = eig_vec[order]
best_featc = np.real(eig_vec[:comp])

fisher_faces = best_featc @ sign_faces
temp = np.arange(N)

X_test_data = np.column_stack((np.repeat(temp,4),X_test_data))
np.random.shuffle(X_test_data)

Y = X_test_data
X_test_data = X_test_data[:,1:]
k, l = X_test_data.shape

pred_y = np.zeros(k)
for n,img in enumerate(X_test_data):
    proj = eigen_faces @ (img - X_train_data_mean).T
    fisher_proj = best_featc @ proj
    dist = (fisher_proj - fisher_faces.T)
    dist = np.linalg.norm(dist.T,axis = 0)
    index = np.argmin(dist)
    pred_y[n] = np.floor(index/6)

print("Accuracy for PCA - LDA Face-Recognition, Using Euclidean is = ", accuracy_score(Y.T[0],pred_y))
