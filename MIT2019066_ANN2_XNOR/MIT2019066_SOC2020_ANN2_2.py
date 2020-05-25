#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: iamkoushal21
"""

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def der_sigmoid(z):
    return z * (1-z)
 
neurons_input = 2 # number of neurons in input layer
neurons_hidden = 6 # number of neurons in hidden layer
neurons_output = 1 # number of neurons in output layer

w0 = np.random.random((neurons_input,neurons_hidden)) 
w1 = np.random.random((neurons_hidden,neurons_output))

#XNOR Gate INPUT
X = np.array([[0,0],[0,1],[1,0],[1,1]])

#XNOR Gate Output
y = np.array([[1],[0],[0],[1]]) 

ephocs = 500

for i in range(ephocs):
  l1 = sigmoid(np.matmul(X,w0))
  l2 = sigmoid(np.matmul(l1,w1)) 
  l2_err = y - l2
 
  dw2 = np.multiply(l2_err,der_sigmoid(l2))
  l1_err = np.dot(dw2,w1.T)
 
  dw1 = np.multiply(l1_err,der_sigmoid(l1))
  w1 += np.dot(l1.T,dw2)
  w0 += np.dot(X.T,dw1)

l1 = sigmoid(np.dot(X,w0))
l2 = sigmoid(np.dot(l1,w1))

prediction = np.round(l2)
print('Predicted Output Values for XNOR Gate using Multilayer Perceptron - ', prediction)
