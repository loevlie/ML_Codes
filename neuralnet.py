#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:51:56 2020

@author: Denny
"""

import sys
import numpy as np
from io import StringIO   # StringIO behaves like a file object

train_input = sys.argv[1]
test_input = sys.argv[2]
train_out = sys.argv[3]
test_out = sys.argv[4]
metric_out = sys.argv[5]
num_epoch = int(sys.argv[6])
hidden_units = abs(int(sys.argv[7]))
init_flag = int(sys.argv[8]) # 1 - RANDOM || 2 - ZERO
learning_rate = float(sys.argv[9])

with open(train_input,'r') as f:
    train_in = f.read()

with open(test_input,'r') as g:
    test_in = g.read()
    
training = np.genfromtxt(StringIO(train_in),delimiter='\t',dtype='str',comments=None)
testing = np.genfromtxt(StringIO(test_in),delimiter='\t',dtype='str',comments=None)

train = [x.split(',') for x in training]
test = [x.split(',') for x in testing]

x_train = [i[1:] for i in train]

y_train = np.zeros((len(train),10))
for i,y in enumerate(train):
    y_train[i][int(y[0])-1] = 1
    
x_test = [i[1:] for i in test]

y_test = np.zeros((len(test),10))
for i,y in enumerate(test):
    y_test[i][int(y[0])-1] = 1

N = len(train) # Number of examples in the training data
N_test = len(test)
x0 = 1 # Beta Term 
K = 10 # Number of letters 

# Helpful Functions
def Sig(x):
    return 1 / (1 + np.exp(-x))

def LINEAR_FORWARD(a,Alpha):
    a = a.astype(np.float64)
    Alpha = Alpha.astype(np.float64)
    b = Alpha @ a
    return b

def LINEAR_BACKWARD(a,Alpha,b,g_b,order):
    if order == 1:
        g_b = g_b.reshape((len(g_b),1)).astype(np.float64)
        a = a.reshape((1,len(a)))
        a = a.astype(np.float64)
        g_Alpha = g_b @ a
        g_a = Alpha[:,1:].T @ g_b
    else:
        g_b = g_b.reshape((len(g_b),1)).astype(np.float64)
        a = a.reshape((1,len(a)))
        a = a.astype(np.float64)
        g_Alpha = g_b @ a
        g_a = Alpha.T @ g_b
    return [g_Alpha,g_a]



def SIGMOID_FORWARD(b):
    z = Sig(b)
    return np.array([1,*z])

def SOFTMAX_FORWARD(c):
    return np.exp(c)/np.sum(np.exp(c))

def CROSS_ENTROPY_FORWARD(y,y_hat):
    global N
    y = np.array(y).astype(np.float64)
    y_hat = y_hat.astype(np.float64)
    J = - (1 / 1) * np.sum(y * np.log(y_hat))
    return J

def CROSS_ENTROPY_BACKWARDS(y,y_hat,J,g_J):
    global N
    y = np.array(y).astype(np.float64)
    y_hat = y_hat.astype(np.float64)
    g_y = (- (1/1) * np.sum(y*(1/y_hat)))*g_J
    return g_y

def SOFTMAX_BACKWARD(c,y_hat,g_y,y):
    return y_hat - y

def SIGMOID_BACKWARD(b,z,gz):
    z2 = gz*(Sig(b[:1])*(1-Sig(b[:1])))
    g_b = z2.reshape((len(z2),1))
    return g_b

# Putting it all together

# This class is to store the intermidiate values
class Object:
    def __init__(self,b,z,c,y_hat,J):
        self.b = b
        self.z = z
        self.c = c
        self.y_hat = y_hat
        self.J = J

def NN_FORWARD(x,y,Alpha,Beta):
    b = LINEAR_FORWARD(x,Alpha)
    z = SIGMOID_FORWARD(b)
    c = LINEAR_FORWARD(z,Beta)
    y_hat = SOFTMAX_FORWARD(c)
    J = CROSS_ENTROPY_FORWARD(y,y_hat)
    return Object(b,z,c,y_hat,J)

def NN_BACKWARD(x,y,Alpha,Beta,Obj):
    g_J = 1 # dJ/dJ = 1    
    g_y_hat = CROSS_ENTROPY_BACKWARDS(y,Obj.y_hat,Obj.J,g_J)
    g_c = SOFTMAX_BACKWARD(Obj.c,Obj.y_hat,g_y_hat,y)
    g_Beta, g_z = LINEAR_BACKWARD(Obj.z,Beta,Obj.c,g_c,1)
    g_b = SIGMOID_BACKWARD(Obj.b,Obj.z,g_z)
    g_Alpha, g_x = LINEAR_BACKWARD(x,Alpha,Obj.b,g_b,2)
    return [g_Alpha,g_Beta]

f = open(train_out,"w+")
g = open(test_out,"w+")
h = open(metric_out,"w+")

def SGD(x_train,y_train,h):
    # Initializing parameters
    if init_flag == 1:
        Beta = np.zeros(K)
        Beta = Beta.reshape(len(Beta),1)
        for i in range(hidden_units):
            if i == 0:
                Alpha = np.array([0,*np.random.uniform(-0.1,0.1,len(x_train[0]))])
                Betai = np.array([[i] for i in np.random.uniform(-0.1,0.1,K)])
                Beta = np.hstack((Beta,Betai))
            else:
                Alphai = np.array([0,*np.random.uniform(-0.1,0.1,len(x_train[0]))])
                Alpha = np.vstack((Alpha,Alphai))
                Betai = np.array([[i] for i in np.random.uniform(-0.1,0.1,K)])
                Beta = np.hstack((Beta,Betai))
            
                
    elif init_flag == 2: 
        Beta = np.zeros(K)
        Beta = Beta.reshape(len(Beta),1)
        for i in range(hidden_units):
            if i == 0:
                Alpha = np.array([0,*np.zeros(len(x_train[0]))])
                Betai = np.zeros(K)
                Betai = Betai.reshape((len(Betai)),1)
                Beta = np.hstack((Beta,Betai))
            else:
                Alphai = np.array([0,*np.zeros(len(x_train[0]))])
                Betai = np.zeros(K)
                Betai = Betai.reshape((len(Betai)),1)
                Alpha = np.vstack((Alpha,Alphai))
                Beta = np.hstack((Beta,Betai))
    
    
    for e in range(num_epoch):
        Cross_Entropy = 0
        Cross_Entropy_test = 0
        for i, xi in enumerate(x_train):
            x = np.array([x0,*xi])
            O = NN_FORWARD(x,y_train[i],Alpha,Beta)
            g_alpha, g_beta = NN_BACKWARD(x,y_train[i],Alpha,Beta,O)
            Alpha = Alpha - learning_rate*g_alpha
            Beta = Beta - learning_rate*g_beta
        
        for i, xi in enumerate(x_train):
            x = np.array([x0,*xi])
            O = NN_FORWARD(x,y_train[i],Alpha,Beta)
            Cross_Entropy += - O.J
        
        for i, xi in enumerate(x_test):
            x = np.array([x0,*xi])
            O = NN_FORWARD(x,y_test[i],Alpha,Beta)
            Cross_Entropy_test += - O.J
        
        CE = -(1/N)*Cross_Entropy
        CE_test = -(1/N_test)*Cross_Entropy_test
        h.write('epoch=' + str(e+1) + ' crossentropy(train): ' + str(CE) + '\n')
        h.write('epoch=' + str(e+1) + ' crossentropy(test): ' + str(CE_test) + '\n')
    return [Alpha,Beta]

# Making predictions
Alpha,Beta = SGD(x_train,y_train,h)

def Predict(x_test,y_test,Alpha,Beta,obj):
    counter = 0
    for i,xi in enumerate(x_test):
        x = np.array([x0,*xi])
        O = NN_FORWARD(x,y_test[i],Alpha,Beta)
        prediction = np.argmax(O.y_hat)
        if prediction != list(y_test[i]).index(1):
            counter += 1
            
        if prediction == 9:
            prediction = -1
        
        obj.write(str(prediction+1) + '\n')
    obj.close()
    return counter/len(x_test)

Error_test = Predict(x_test,y_test,Alpha,Beta,g)
Error_train = Predict(x_train,y_train,Alpha,Beta,f)

h.write('error(train): ' + str(Error_train) + '\n')
h.write('error(test): ' + str(Error_test))
h.close()