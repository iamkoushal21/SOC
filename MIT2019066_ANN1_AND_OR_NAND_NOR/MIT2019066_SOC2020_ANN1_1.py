#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: iamkoushal21
"""
#Header Files
import numpy as np
import matplotlib.pyplot as plt


#Input for bias
X0 = np.array([1, 1, 1, 1])

#two inputs
X1 = np.array([0, 0, 1, 1])
X2 = np.array([0, 1, 0, 1])

#Output
out1 = np.array([0,0,0,1]) #AND Gate Output
out2 = np.array([0,1,1,1]) #OR Gate Output
out3 = np.array([1,1,1,0]) #NAND Gate Output
out4 = np.array([1,0,0,0]) #NOR Gate Output

X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
m,n = X.shape


#Random weight Initialisation for each Gate
def andgateweight():
    w0 = -0.1
    w1 = 0.2
    w2 = 0.2
    
    return w0,w1,w2

def orgateweight():
    w0 = -0.1
    w1 = 0.7
    w2 = 0.7
    
    return w0,w1,w2

def nandgateweight():
    w0 = 0.6
    w1 = -0.8
    w2 = -0.8
    
    return w0,w1,w2

def norgateweight():
    w0 = 0.5
    w1 = -0.7
    w2 = -0.7
    
    return w0,w1,w2

#Processing for false classification
def falseNegative(w0,w1,w2,x0,x1,x2):
    w0 = w0 + x0 
    w1 = w1 + x1
    w2 = w2 + x2
    
    return w0,w1,w2

def falsePositive(w0,w1,w2,x0,x1,x2):
    w0 = w0 - x0 
    w1 = w1 - x1
    w2 = w2 - x2
    
    return w0,w1,w2

def classification_graph(X1,X2,y,slope,c, gate):
    x = np.linspace(-.5,3,10)
    scatter = plt.scatter(X1,X2,c=y)
    plt.plot(x, slope * x + c, color = 'r')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(*scatter.legend_elements(),loc="upper left", title="Class")
    plt.title(gate + ' gate perceptron')
    plt.show()

def prediction(x0,x1,x2,w0,w1,w2,output,count,i):
    y_pre = 0
    y_pre = y_pre + (w0 * x0 + w1 * x1 + w2 * x2)
   
    #False_Classification
    if (output[i] == 1 and y_pre < 0):
      w0,w1,w2 = falseNegative(w0,w1,w2,x0,x1,x2)
    elif (output[i] == 0 and y_pre >= 0):
      w0,w1,w2 = falsePositive(w0,w1,w2,x0,x1,x2)   
    #True Classification
    else :
      count += 1
    
    return w0,w1,w2,count

def perceptron(W0,W1,W2,output,gate):
    w0 = W0
    w1 = W1
    w2 = W2
    while(True):
        count = 0
        for i in range(m):
            x0 = X[i][0]
            x1 = X[i][1]
            x2 = X[i][2]
            w0,w1,w2,count = prediction(x0,x1,x2,w0,w1,w2,output,count,i)
        if count == m:
            slope = -w1 / w2
            c = -w0 / w2
            break
    classification_graph(X1,X2,output,slope,c,gate)

#For AND Gate
w0,w1,w2 = andgateweight()
perceptron(w0,w1,w2,out1, 'AND')
  
#For OR Gate
w0,w1,w2 = orgateweight()
perceptron(w0,w1,w2,out2, 'OR')

#For NAND Gate
w0,w1,w2 = nandgateweight()
perceptron(w0,w1,w2,out3, 'NAND')

#For NOR Gate
w0,w1,w2 = norgateweight()
perceptron(w0,w1,w2,out4, 'NOR')  