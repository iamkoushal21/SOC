#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: iamkoushal21
"""

import numpy as np

def transformation(t):
    m,n = t.shape
    x = m * n
    c = t.copy()
    c = c.reshape(-1,1)
    b = np.zeros(t.shape).reshape(-1,1)
    for i in range(x):
        if c[i] >= 0:
            b[i] = 1
        else:
            b[i] = -1
   
    return b.reshape(m,n)


A = np.array([[1,1,1,1,1,1], 
              [-1,-1,-1,-1,-1,-1], 
              [1 ,-1, -1,1, 1, 1 ], 
              [1,1,-1,-1,-1,-1]])

B = np.array([[1,1,1],
              [-1,-1,-1],
              [-1,1,1],
              [1,-1,1]])

M =  4
#Weight Matric
W = np.dot(A.T,B)
V = W.T

W = W.astype('float64')
V = V.astype('float64')
l_r = 0.2

for i in range(M - 1):
    b_t = transformation(A @ W)
    a_t = transformation(B @ W.T)
    
    V1 = l_r * ((A - a_t).T @ (B + b_t)).T
    W1 = l_r *((B - b_t).T @(A + a_t)).T
    
    if (W1 == 0).all() and (V1 == 0).all():
        break
    W += W1
    V += V1

x = np.array([[1,1,1,1,1,-1],
              [-1,-1,-1,-1,-1,1],
              [1,-1,-1,1,1,-1],
              [1,1,-1,-1,-1,1]])
output_associative_x = transformation(x @ W)
print('output associative using Feed Forward with noise in last bit:- ')
print(output_associative_x)


y = np.array([[1,1,-1],
               [-1,-1,1],
               [-1,1,-1], 
               [1,-1,-1]])
input_associative_y = transformation(y @ V)
print('input associative using Feed Backward with noise in last bit:- ')
print(input_associative_y)