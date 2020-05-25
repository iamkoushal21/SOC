#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: iamkoushal21
"""

#Header Files
import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


#Data Reading From File
def data_read(filename):
    dataframe = pd.read_csv(filename)
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    return dataframe

#Train and Testing Data Spliting
def train_test_split(X_data, y_data):
    rows, col = np.shape(X_data)
    train_ratio = 0.75
    test_ratio = 1 - train_ratio
    total_train_samples = int(train_ratio * rows)
    
    X_train_data = X_data[0:total_train_samples]
    X_test_data = X_data[total_train_samples : rows]
    Y_train_data = y_data[0:total_train_samples]
    Y_test_data = y_data[total_train_samples : rows]

    return X_train_data, X_test_data, Y_train_data, Y_test_data

#Box Muller 
def box_muller(X_data):
    X_col1 = X_data[:, 0]
    X_col2 = X_data[:, 1]
    
    z1 = np.sqrt(-2 * np.log(X_col1)) * np.cos(2 * np.pi * X_col2)
    z2 = np.sqrt(-2 * np.log(X_col1)) * np.sin(2 * np.pi * X_col2)

    return np.vstack((z1, z2)).T

def weightinitialisation(X_train_data, y_train_data):
    fc_data = X_train_data[y_train_data == 1]
    sc_data = X_train_data[y_train_data == 0]
    theta_fc = [np.mean(fc_data[0]), np.mean(fc_data[1])]
    theta_sc = [np.mean(sc_data[0]), np.mean(sc_data[1])]

    return theta_fc, theta_sc

def pred(X_test, y_test, inv_sigma, det_sigma, theta_fc, theta_sc):
    prediction = []
    for i in range(len(y_test)):
        fc = -0.5 * np.matmul(np.matmul((((X_test[i] - theta_fc).reshape(2,1)).T), inv_sigma), (((X_test[i] - theta_fc).reshape(2,1))))
        sc = -0.5 * np.matmul(np.matmul((((X_test[i] - theta_sc).reshape(2,1)).T), inv_sigma), (((X_test[i] - theta_sc).reshape(2,1))))
        prob_fc = 1/((2 * np.pi) ** (col/2) * math.sqrt(det_sigma)) * math.exp(fc)
        prob = np.sum(y_train_data) / len(y_train_data)
        prob_sc = 1/((2 * np.pi) ** (col/2) * math.sqrt(det_sigma)) * math.exp(sc)
        
        if (prob_fc * prob > prob_sc * (1 - prob)):
            prediction.append(1)
        else :
            prediction.append(0)

    return prediction



data = data_read('microchip.csv')
minmax_scaler = MinMaxScaler()
X_data = data.iloc[:,0:-1].values
X_data = minmax_scaler.fit_transform(X_data)
y_data = data.iloc[:,-1].values

#BOX MULLER
X_data = box_muller(X_data)

y_data = y_data[~np.isinf(X_data[:,0])]
X_data = X_data[~np.isinf(X_data[:,0])]


#spliting of data
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data)
rows, col = np.shape(X_train_data)
theta_fc, theta_sc = weightinitialisation(X_train_data, y_train_data)

sum = np.zeros((2,2))
for i in range(rows):
    if y_train_data[i] == 0:
        x = X_train_data[i] - theta_sc
    else:
        x = X_train_data[i] - theta_fc
        
    sum += np.dot(x.reshape(2,1), x.reshape(1,2))

sigma = sum / rows 
inv_sigma = np.linalg.inv(sigma)
det_sigma = np.linalg.det(sigma)
prediction = pred(X_data, y_data, inv_sigma, det_sigma, theta_fc, theta_sc)
print(accuracy_score(y_data, prediction))