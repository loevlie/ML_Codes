#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from io import StringIO   # StringIO behaves like a file object

train_input = sys.argv[1]
index_to_word = sys.argv[2]
index_to_tag = sys.argv[3]
hmmprior = sys.argv[4]
hmmemit = sys.argv[5]
hmmtrans = sys.argv[6]

# Need a function that reads the train_words.txt and then estimates PI, A, and B using the maximum likelihood solutisons
# The outputs should be of the same form as hmmprior.txt, hmmtrans.txt, and hmmemit.txt

with open(train_input,'r') as f:
    train_in = f.read()

with open(index_to_word,'r') as g:
    index_to_word_data = g.read()
    
with open(index_to_tag,'r') as h:
    index_to_tag_data = h.read()
    
training = np.genfromtxt(StringIO(train_in),delimiter='\t',dtype='str',comments=None)
I2W_data= np.genfromtxt(StringIO(index_to_word_data),delimiter='\t',dtype='str',comments=None)
I2T_data = np.genfromtxt(StringIO(index_to_tag_data),delimiter='\t',dtype='str',comments=None)

    
train = [x.split() for x in training]
I2W = {}
for i, w in enumerate(I2W_data):
    I2W.update({w:i})
I2T = {}
for i, t in enumerate(I2T_data):
    I2T.update({t:i})

f = open(hmmprior,"w+")
g = open(hmmemit,"w+")
h = open(hmmtrans,"w+")

N_pi_j =np.zeros(len(I2T))

for sentence in train:
    check = sentence[0].split('_')
    t=I2T[check[1]]
    N_pi_j[t] += 1 

Pi_j = np.zeros(len(I2T))

for i,pi in enumerate(N_pi_j):
    Pi_j[i]=(pi+1)/(np.sum(N_pi_j + 1))

for i in Pi_j:
    f.write(f'{i:.18e}' + '\n')

f.close()
    
N_B_jk = np.zeros((len(I2T),len(I2W))) # j,k more dimensionality (number of times state sj is assosiated with teh word k in the training dataset)

for sentence in train:
    for i,word in enumerate(sentence):
        split = word.split('_')
        word_index = I2W[split[0]]
        tag_index = I2T[split[1]]
        N_B_jk[tag_index,word_index] += 1

    
beta_jk = np.zeros((len(I2T),len(I2W)))

for index1,Nb_jp in enumerate(N_B_jk):
    for index2, Nb_jk in enumerate(Nb_jp):
        beta_jk[index1,index2] = (Nb_jk + 1) / (np.sum(Nb_jp + 1))
        

for row in beta_jk:
    for column in row:
        g.write(f'{column:0.18e} ')
    g.write('\n')
g.close()

N_A_jk = np.zeros((len(I2T),len(I2T))) # j,k more dimensionality (number of times state sj is followed by state sk in the training dataset)

for sentence in train:
    for i,word in enumerate(sentence):
        if i+1 == len(sentence):
            continue
        tag1 = I2T[word.split('_')[1]]
        tag2 = I2T[sentence[i+1].split('_')[1]]
        N_A_jk[tag1,tag2] += 1

alpha_jk = np.zeros((len(I2T),len(I2T)))

for index1,NA_jp in enumerate(N_A_jk):
    for index2, NA_jk in enumerate(NA_jp):
        alpha_jk[index1,index2] = (NA_jk + 1) / (np.sum(NA_jp + 1))
    
for row in alpha_jk:
    for column in row:
        h.write(f'{column:0.18e} ')
    h.write('\n')
h.close()        
