#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from io import StringIO   # StringIO behaves like a file object

test_input = sys.argv[1]
index_to_word = sys.argv[2]
index_to_tag = sys.argv[3]
hmmprior = sys.argv[4]
hmmemit = sys.argv[5]
hmmtrans = sys.argv[6]
predicted_file = sys.argv[7]
metric_file = sys.argv[8]


with open(test_input,'r') as f:
    test_in = f.read()

with open(index_to_word,'r') as g:
    index_to_word_data = g.read()
    
with open(index_to_tag,'r') as h:
    index_to_tag_data = h.read()
    
with open(hmmprior,'r') as i:
    hmm_prior = i.read()

with open(hmmemit,'r') as j:
    hmm_emit = j.read()
    
with open(hmmtrans,'r') as k:
    hmm_trans = k.read() 
    
testing = np.genfromtxt(StringIO(test_in),delimiter='\t',dtype='str',comments=None)
I2W_data= np.genfromtxt(StringIO(index_to_word_data),delimiter='\t',dtype='str',comments=None)
I2T_data = np.genfromtxt(StringIO(index_to_tag_data),delimiter='\t',dtype='str',comments=None)
prior_data = np.genfromtxt(StringIO(hmm_prior),delimiter=' ',dtype='str',comments=None)
emit_data= np.genfromtxt(StringIO(hmm_emit),delimiter=' ',dtype='str',comments=None)
trans_data = np.genfromtxt(StringIO(hmm_trans),delimiter=' ',dtype='str',comments=None)

test = [x.split() for x in testing]
sentences = []
labels = []
for i in test:
    words = [x.split('_')[0] for x in i]
    tag = [x.split('_')[1] for x in i]
    sentences.append(words)
    labels.append(tag)
    
I2W = {}
for i, w in enumerate(I2W_data):
    I2W.update({w:i})
I2T = {}
for i, t in enumerate(I2T_data):

    I2T.update({t:i})


## USEFUL FUNCTIONS 
P = np.asarray(prior_data,dtype='float64')
T = np.asarray(trans_data,dtype='float64')
E = np.asarray(emit_data,dtype='float64')

def FORWARD(x):
    alpha = np.zeros((len(x),len(I2T_data)))
    for t, w in enumerate(x):
        if t == 0:
            alpha1 = P * E[:,I2W[w]]
            alpha[t] = alpha1
        else:
            for i in range(len(I2T)):
                alpha[t,i] = E[i,I2W[w]] * (T[:,i] @ alpha[t-1,:])
    return alpha
    
def BACKWARD(x):
    beta = np.zeros((len(x),len(I2T_data)))
    for t, w in enumerate(x):
        if t == 0:
            beta[len(x)-t-1,:] = np.ones(len(I2T_data)) 
        else:
            for i in range(len(I2T)):
                beta[len(x)-t-1,i] = np.sum(E[:,I2W[x[len(x)-t]]]*beta[len(x)-t,:]*T[i,:])
    return beta

def FORWARD_BACKWARD(x):
    alpha = FORWARD(w)
    log_likelihood = np.log(np.sum(alpha[-1]))
    beta = BACKWARD(w)
    p = alpha*beta
    yt = [np.argmax(i) for i in p]
    return [log_likelihood, yt]

# IMPLEMENTING
f = open(predicted_file,"w+")
g = open(metric_file,"w+")
ll_total = 0
count = 0
total = sum([len(x) for x in labels])

for i, w in enumerate(sentences):
    log_likelihood, yt = FORWARD_BACKWARD(w)
    for k,word in enumerate(w):
        if k != len(w)-1:
            f.write(f'{word}_{I2T_data[yt[k]]} ')
        else:
            f.write(f'{word}_{I2T_data[yt[k]]}')
    ll_total += log_likelihood
    f.write('\n')
    counting = [1.0 if I2T_data[yt[j]] != labels[i][j] else 0.0 for j,label in enumerate(labels[i])]
    count += sum(counting)

accuracy = 1 - count/total
Av_log_likelihood = ll_total / len(sentences)
g.write(f'Average Log-Likelihood: {Av_log_likelihood}')
g.write('\n')
g.write(f'Accuracy: {accuracy}')
f.close()
g.close()
    
    
    

    


    