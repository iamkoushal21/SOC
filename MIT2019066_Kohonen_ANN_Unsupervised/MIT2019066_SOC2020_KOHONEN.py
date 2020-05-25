#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: iamkoushal21
"""

import numpy as np
import random

import matplotlib.pyplot as plt 

def random_generation(i,j):
    random_arr = np.ndarray((i,j))
    correct_decimal = 2
    for a in range(i):
        rand1 = random.uniform(-1,1)
        rand2 = random.uniform(-1,1)
        
        random_arr[a] = [round(rand1,correct_decimal), round(rand2,correct_decimal)]

    return random_arr

def random_generation_weights(i,j,k):
    random_arr = np.ndarray((i,j,k))
    correct_decimal = 2
    for a in range(i):
        for b in range(j):
            rand1 = random.uniform(-1,1)
            rand2 = random.uniform(-1,1)
        
            random_arr[a][b] = [round(rand1,correct_decimal), round(rand2,correct_decimal)]
                
    return random_arr
    
    
def weight_plotmap(w, title):
    plt.scatter(w[:,0], w[:,1])
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis')
    plt.title(title) 
    plt.show()

m = 1500
epochs = 1000
l_r = 0.1

#1500 input vectors two dimensional
X = random_generation(1500, 2)

#100 Neurons 
W = random_generation(100, 2)

weight_plotmap(W, "Kohonen Map - Weight Vector(Neurons) Before training")

#training
for i in range(epochs):
    for j in range(m):
        euclidian_distance = np.sqrt(np.sum((W - X[j]) ** 2, axis=1))
        mindistance_index = np.argmin(euclidian_distance)
        delta = l_r * (X[j] - W[mindistance_index])
        W[mindistance_index] += delta
        
weight_plotmap(W, "Kohonen Map - Weight Vector(Neurons) After training")

#testing
test_inputs = [[0.1, 0.8], [0.5, -0.2], [-0.8, -0.9], [-0.06, 0.9]]

for X_test in test_inputs:
    euclidian_distance_test = np.sqrt(np.sum((W - X_test) ** 2, axis=1))
    mindistance_index_test = np.argmin(euclidian_distance_test)
    weight = [round(W[mindistance_index_test][0], 2), round(W[mindistance_index_test][1], 2)]
    
    print('For Input Vector ', X_test, 'nearest weight(neuron) is - ', weight)
