#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:44:36 2020

@author: Denny
"""
import sys
import numpy as np
from io import StringIO   # StringIO behaves like a file object

Train_input = sys.argv[1]
Valid_input = sys.argv[2]
test_input = sys.argv[3]
Dict_input = sys.argv[4]
Train_out = sys.argv[5]
Valid_out = sys.argv[6]
Test_out = sys.argv[7]
feature_flag = int(sys.argv[8])

with open(Dict_input,'r') as f:
    Dict_in = f.read()

with open(test_input,'r') as g:
    test_in = g.read()

with open(Train_input,'r') as h:
    train_in = h.read()
    
with open(Valid_input,'r') as m:
    valid_in = m.read()
    
Dict_ = np.genfromtxt(StringIO(Dict_in),delimiter='\t',dtype='str',comments=None)
testing = np.genfromtxt(StringIO(test_in),delimiter='\t',dtype='str',comments=None)
validation = np.genfromtxt(StringIO(valid_in),delimiter='\t',dtype='str',comments=None) 
training = np.genfromtxt(StringIO(train_in),delimiter='\t',dtype='str',comments=None) 

Dict = {}
for i in Dict_:
    Dict.update({i.split()[0]:int(i.split()[1])})
    
if feature_flag == 1:

    Dict_Test = {}
    Formated = []
    g = open(Test_out,"w+")
    for j, Value in enumerate(testing[:,0]):
        g.write(Value + '\t')
        for i, w in enumerate(testing[j,1].split()):
            if w in Dict and str(Dict[w]) + ':1' not in Formated:
                if i ==  len(testing[j,1].split()):
                    Formated.append(str(Dict[w]) + ':1')
                    g.write(str(Dict[w]) + ':1')
                else:
                    Formated.append(str(Dict[w]) + ':1')
                    g.write(str(Dict[w]) + ':1' + '\t')
        Formated = []
        g.write('\n')       
    
    g.close()
    
    Formated = []
    h = open(Train_out,"w+")
    for j, Value in enumerate(training[:,0]):
        h.write(Value + '\t')
        for i, w in enumerate(training[j,1].split()):
            if w in Dict and str(Dict[w]) + ':1' not in Formated:
                if i ==  len(training[j,1].split()):
                    Formated.append(str(Dict[w]) + ':1')
                    h.write(str(Dict[w]) + ':1')
                else:
                    Formated.append(str(Dict[w]) + ':1')
                    h.write(str(Dict[w]) + ':1' + '\t')
        Formated = []
        h.write('\n')       
    
    h.close()
    
    Formated = []
    m = open(Valid_out,"w+")
    for j, Value in enumerate(validation[:,0]):
        m.write(Value + '\t')
        for i, w in enumerate(validation[j,1].split()):
            if w in Dict and str(Dict[w]) + ':1' not in Formated:
                if i ==  len(validation[j,1].split()):
                    Formated.append(str(Dict[w]) + ':1')
                    m.write(str(Dict[w]) + ':1')
                else:
                    Formated.append(str(Dict[w]) + ':1')
                    m.write(str(Dict[w]) + ':1' + '\t')
        Formated = []
        m.write('\n')       
    
    m.close()

else:
    
    def trim(data, value):
        return [x for x in data if x != value]
    
    
    Formated = []
    g = open(Test_out,"w+")
    for j, Value in enumerate(testing[:,0]):
        g.write(Value + '\t')
        Unique,Count_ = np.unique(testing[j,1].split(),return_counts = True)
        for i, w in enumerate(testing[j,1].split()):  
            if Count_[list(Unique).index(w)] >= 4:
                continue
            if w in Dict and str(Dict[w]) + ':1' not in Formated:
                if i ==  len(testing[j,1].split()):
                    Formated.append(str(Dict[w]) + ':1')
                    g.write(str(Dict[w]) + ':1')
                else:
                    Formated.append(str(Dict[w]) + ':1')
                    g.write(str(Dict[w]) + ':1' + '\t')
        Formated = []
        g.write('\n')       
    
    g.close()
    
    Formated = []
    h = open(Train_out,"w+")
    for j, Value in enumerate(training[:,0]):
        h.write(Value + '\t')
        Unique,Count_ = np.unique(training[j,1].split(),return_counts = True)
        for i, w in enumerate(training[j,1].split()):
            if Count_[list(Unique).index(w)] >= 4:
                continue
            if w in Dict and str(Dict[w]) + ':1' not in Formated:
                if i ==  len(training[j,1].split()):
                    Formated.append(str(Dict[w]) + ':1')
                    h.write(str(Dict[w]) + ':1')
                else:
                    Formated.append(str(Dict[w]) + ':1')
                    h.write(str(Dict[w]) + ':1' + '\t')
        Formated = []
        h.write('\n')       
    
    h.close()
    
    Formated = []
    m = open(Valid_out,"w+")
    for j, Value in enumerate(validation[:,0]):
        m.write(Value + '\t')
        Unique,Count_ = np.unique(validation[j,1].split(),return_counts = True)
        for i, w in enumerate(validation[j,1].split()):
            if Count_[list(Unique).index(w)] >= 4:
                continue
            if w in Dict and str(Dict[w]) + ':1' not in Formated:
                if i ==  len(validation[j,1].split()):
                    Formated.append(str(Dict[w]) + ':1')
                    m.write(str(Dict[w]) + ':1')
                else:
                    Formated.append(str(Dict[w]) + ':1')
                    m.write(str(Dict[w]) + ':1' + '\t')  
        Formated = []
        m.write('\n')       
    
    m.close()

