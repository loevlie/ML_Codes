#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:39:42 2020

@author: Denny
"""
import sys
import numpy as np
from io import StringIO   # StringIO behaves like a file object

#import os 
#os.chdir('⁨C:/Users⁩/Denny⁩/Downloads⁩')
train_input = sys.argv[1]

#train_input = 'politicians_train.tsv'
test_input = sys.argv[2]
#test_input = 'politicians_test.tsv'
split_index = int(sys.argv[3])
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics_out = sys.argv[6]

with open(train_input,'r') as f:
    train_in = f.read()

with open(test_input,'r') as g:
    test_in = g.read()

training = np.genfromtxt(StringIO(train_in),delimiter='\t',dtype='str',skip_header=1)
testing = np.genfromtxt(StringIO(test_in),delimiter='\t',dtype='str',skip_header=1)

att_ans = np.unique(testing[:,split_index])
conclusion = np.unique(testing[:,-1])

#def main():
    #print('Number of arguments:', len(sys.argv), 'arguments.')
    #print('Argument List:', str(sys.argv))

    #infile = sys.argv[1]
    #outfile = sys.argv[2]
    #print("The input file is: %s" % (infile))
        
    #print("The output file is: %s" % (outfile))

#if __name__ == '__main__':
    #main()
count1 = np.array([0,0])
count2 = np.array([0,0])
# Trainging
for i in range(len(training[:,split_index])):
    
    if training[i,split_index]  == att_ans[0]:
        if training[i,-1] == conclusion[0]:
            count1[0] += 1
        elif training[i,-1] == conclusion[1]:
            count1[1] += 1
            
    elif training[i,split_index] == att_ans[1]:
        if training[i,-1] == conclusion[0]:
            count2[0] += 1
        elif training[i,-1] == conclusion[1]:
            count2[1] += 1

h = open(metrics_out,"w+")
trained_ans1 = conclusion[np.argmax(count1)]
trained_ans2 = conclusion[np.argmax(count2)]

# TESTING 
f = open(train_out,"w+")
g = open(test_out,"w+")
Positive = np.array([0,0])
negative = np.array([0,0])

for i in range(len(training[:,split_index])):
    
    if training[i,split_index]  == att_ans[0]:
        #output_train_data += trained_ans1
        f.write(trained_ans1 + '\n')
        if trained_ans1 == training[i,-1]:
            Positive[0] += 1
        else:
            negative[0] += 1 
            
    elif training[i,split_index] == att_ans[1]:
        #output_train_data += trained_ans2
        f.write(trained_ans2 + '\n')
        if trained_ans2 == training[i,-1]:
            Positive[0] += 1
        else:
            negative[0] += 1 
            
for i in range(len(testing[:,split_index])):
    
    if testing[i,split_index]  == att_ans[0]:
        #output_train_data += trained_ans1
        g.write(trained_ans1 + '\n')
        if trained_ans1 == testing[i,-1]:
            Positive[1] += 1
        else:
            negative[1] += 1 
            
    elif testing[i,split_index] == att_ans[1]:
        #output_train_data += trained_ans2
        g.write(trained_ans2 + '\n')
        if trained_ans2 == testing[i,-1]:
            Positive[1] += 1
        else:
            negative[1] += 1 

h.write(f'error(train): {negative[0]/np.sum([negative[0],Positive[0]])}' + '\n')
h.write(f'error(test): {negative[1]/np.sum([negative[1],Positive[1]])}')
    
f.close()
g.close()
h.close()
#print(f'{att_ans[0]} = {conclusion[np.argmax(count1)]}')
#print(f'{att_ans[1]} = {conclusion[np.argmax(count2)]}')
#print(count1,count2,count3)

