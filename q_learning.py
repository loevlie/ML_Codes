#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys 
from environment import MountainCar
import numpy as np 

mode = sys.argv[1]
weight_out = sys.argv[2]
returns_out = sys.argv[3]
episodes = int(sys.argv[4])
max_iterations = int(sys.argv[5])
epsilon = float(sys.argv[6])
gamma = float(sys.argv[7])
learning_rate = float(sys.argv[8])

# Useful Functions
def sparse_dot(X, Theta):
    product = 0.0
    for i, v in X.items():
        product += Theta[int(i)] * v
    return product 

def Q(s,w):
    global beta
    return sparse_dot(s, w) + beta

# Initializing vectors
Car = MountainCar(mode)
actions = np.array([0,1,2])
returns = np.zeros(episodes)
state_space = Car.state_space
weights = np.zeros((state_space,len(actions)))
beta = 0  
for i in range(episodes):
    s = Car.reset()
    reward = 0
    for j in range(max_iterations):
        q = Q(s,weights)
        if np.random.uniform(0,1) > 1 - epsilon:
            a = np.random.randint(len(actions))              
        else: 
            a = np.argmax(q)   
        s_prime, rewardi, done = Car.step(a)
        q_prime = Q(s_prime,weights)
        q_delta = np.zeros(weights.shape)
        if mode == 'raw':
            weights_a = np.array([s[i] for i in range(state_space)])
        else:
            weights_a = np.zeros(state_space)
            for k, l in s.items():
                weights_a[k] = l
        q_delta[:,a] = weights_a
        weights -= learning_rate * (q[a] - (rewardi + gamma * np.max(q_prime))) * q_delta
        beta -= learning_rate * (q[a] - (rewardi + gamma * np.max(q_prime)))
        reward += rewardi
        s = s_prime
        if done==True:
            break
    returns[i] = reward

f = open(weight_out,"w+")
g = open(returns_out,"w+")
weights = weights.reshape(3 * state_space)
f.write('{0}\n'.format(beta))
for w in weights:
    f.write('{0}\n'.format(w))
    
f.close()
for ret in returns:
    g.write('{0}\n'.format(ret))
g.close()