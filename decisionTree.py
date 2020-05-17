#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:34:36 2020

@author: Denny
"""
import sys
import numpy as np
from io import StringIO   # StringIO behaves like a file object

train_input = sys.argv[1]
test_input = sys.argv[2]
max_depth = int(sys.argv[3])
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics_out = sys.argv[6]

with open(train_input,'r') as f:
    train_in = f.read()

with open(test_input,'r') as g:
    test_in = g.read()

training = np.genfromtxt(StringIO(train_in),delimiter='\t',dtype='str',skip_header=1)
testing = np.genfromtxt(StringIO(test_in),delimiter='\t',dtype='str',skip_header=1)
training_for_tree = np.genfromtxt(StringIO(train_in),delimiter='\t',dtype='str')
testing_for_tree = np.genfromtxt(StringIO(test_in),delimiter='\t',dtype='str')

Attributes = training_for_tree[0,:]
Attributes = list(Attributes)
Attributes.pop()
for i in range(len(Attributes)):
    Attributes[i] = str(Attributes[i]).strip()
Attributes = np.array(Attributes)

label, label_count = np.unique(training[:,-1],return_counts = True)
att_ans_full, att_count_full = np.unique(training[:,0],return_counts = True)

def error_count(data):
    label, count = np.unique(data[:,-1],return_counts = True)
    if len(label) == 1:
        ERROR = 0
    else:
        ERROR = np.min(count)/np.sum([count])
    return ERROR

def gini_gain(data,attribute_index):
    COUNT1 = np.array([0,0])
    COUNT2 = np.array([0,0])
    att_ans, att_count = np.unique(data[:,attribute_index],return_counts = True)
    
    if len(att_ans) == 2:
        for i in range(len(data[:,attribute_index])):
            if data[i,attribute_index] == att_ans[0]:
                if data[i,-1] == label[0]:
                    COUNT1[0] += 1 # Y --> Dem
                else:
                    COUNT1[1] += 1  # Y --> Rep
            else:
                if data[i,-1] == label[0]: 
                    COUNT2[0] += 1 # N --> Dem
                else:
                    COUNT2[1] += 1 # N --> Rep
        
        total_gini = (label_count[0]/np.sum(label_count))*(label_count[1]/np.sum(label_count)) + (label_count[1]/np.sum(label_count))*(label_count[0]/np.sum(label_count))
        att_gini_1 = (COUNT1[0]/np.sum(COUNT1)) * (COUNT1[1]/np.sum(COUNT1)) + (COUNT1[1]/np.sum(COUNT1)) * (COUNT1[0]/np.sum(COUNT1))
        att_gini_2 = (COUNT2[0]/np.sum(COUNT2)) * (COUNT2[1]/np.sum(COUNT2)) + (COUNT2[1]/np.sum(COUNT2)) * (COUNT2[0]/np.sum(COUNT2))
        att_gini_impurity = att_gini_1 * att_count[0]/np.sum(att_count) + att_gini_2 * att_count[1]/np.sum(att_count)
        Gini = total_gini - att_gini_impurity
    else:
        Gini = 0
        
    return Gini


# This class represents an individual node 
class Node:
    def __init__(self,key,data,node_att=None,left=None,right=None):
        self.left= left
        self.right = right
        self.val = key
        self.data = data
        self.node_att = node_att


class leaf:
    def __init__(self,answer,data,att):
        self.answer = answer
        self.data = data
        self.attribute = att
        data_check, data_count = np.unique(data[:,-1],return_counts = True)
        
        if len(data_check) == 2:  
            self.error = np.min(data_count)
        else:
            self.error = 0

NODE = {}
x = 0

def BUILD_TREE(node,depth,max_depth):
    global x
    depth += 1 
    GINI_TEST = np.zeros(len(node.data[0,:]) - 1)
    for i in range(len(node.data[0,:]) - 1):
        GINI_TEST[i] = gini_gain(node.data,i)
    node_att = Attributes[np.argmax(GINI_TEST)]
    att_index = np.argmax(GINI_TEST)
    list3 = [att_index]
    list4 = [str(node_att).strip()]
    
    for i in range(len(GINI_TEST)):
        if i == att_index:
            jjjj = 1
        elif np.sum(GINI_TEST) == 0:
            jjjj = 2
        else:
            if np.isclose(GINI_TEST[i],max(GINI_TEST)):
                list3.append([i])
                list4.append(str(Attributes[i]).strip())
        if len(list4) > 1:
            LIST5 = sorted(list4,reverse=1)
            CHECK = []
            for i in range(len(Attributes)):
                CHECK.append(str(Attributes[i]).strip())
            att_index = list(CHECK).index(LIST5[0])
            node_att = Attributes[att_index]
    
    List1 = []
    List2 = []
    
    label, label_count = np.unique(node.data[:,-1],return_counts = True)
    att_ans, att_count = np.unique(node.data[:,att_index],return_counts = True)
    
    if depth > max_depth:
        if len(label) ==  2:
            if label_count[0] == label_count[1]:
                label_2 = sorted(label,reverse=1)
                label_2 = label_2[0]
                xx,yy = np.unique(node.data[:,-1],return_counts = True)
                x += np.min(yy)
                return leaf(label_2,node.data,node_att)
            else:
                label_2 = label[np.argmax(label_count)]
                xx,yy = np.unique(node.data[:,-1],return_counts = True)
                x += np.min(yy)
                return leaf(label_2,node.data,node_att)
        else:
            label_2 = label[np.argmax(label_count)] 
            return leaf(label_2,node.data,node_att)
            
        
        return leaf(label_2,node.data,node_att)
    
    if len(att_ans)  == 2:
        
        for i in range(len(node.data[:,att_index])):
            if node.data[i,att_index] == att_ans_full[0]:
                List1.append(node.data[i,:])
            else:
                List2.append(node.data[i,:])
        
        left = BUILD_TREE(Node(depth,np.array(List1)), depth, max_depth)
        right= BUILD_TREE(Node(depth,np.array(List2)), depth, max_depth)
        
    else:
        return leaf(label[np.argmax(label_count)],node.data,node_att)
    
    
    return Node(depth,node.data,node_att,left,right)



GINI_TEST = np.zeros(len(training[0,:]) - 1)

for i in range(len(training[0,:]) - 1):
    GINI_TEST[i] = gini_gain(training,i)
node_att = Attributes[np.argmax(GINI_TEST)]

root = Node(1,training)
depth = 0

TREE  = BUILD_TREE(root,depth,max_depth)
j = 1
y = 0
def PRINTING(node, space = '|', j=2):
    
    global y
    
    if j == 1:   
        print('[' + str(label_count[0]) + ' ' + label[0] + ' /' + str(label_count[1]) + ' ' + label[1] + ']')
    else:
        j = 2
    j += 1 
    
    att_ans, att_count = np.unique(training[:,0],return_counts = True)

    if isinstance(node, leaf):
        xx,yy = np.unique(node.data[:,-1],return_counts = True)
        if len(yy) == 2:
            y += np.min(yy)
        else:
            y += 0
        return 
    
    label_left, label_left_count = np.unique(node.left.data[:,-1],return_counts = True)
    label_right, label_right_count = np.unique(node.right.data[:,-1],return_counts = True)  
    
    if len(label_right) == 2:
        print(space + node.node_att + '  = ' +  att_ans[1] + ': ' + '[' + str(label_right_count[0]) + ' ' + label_right[0] + ' /' + str(label_right_count[1]) + ' ' + label_right[1] + ']')
        PRINTING(node.right, space + " |")
    else:
        if label_right == label[0]:
            label_right = np.append(label_right,label[1]) 
            label_right_count = [label_right_count[0],0]
        else:
            label_right = np.append(label[0],label_right) 
            label_right_count = [0,label_right_count[0]]
            
        print(space + node.node_att + '  = ' +  att_ans[1] + ': ' + '[' + str(label_right_count[0]) + ' ' + label_right[0] + ' /' + str(label_right_count[1]) + ' ' + label_right[1] + ']')
        PRINTING(node.right, space + " |")
    
    if len(label_left) == 2:
        print(space + node.node_att + ' = ' +  att_ans[0] + ': ' + '[' + str(label_left_count[0]) + ' ' + label_left[0] + ' /' + str(label_left_count[1]) + ' ' + label_left[1] + ']')
        PRINTING(node.left, space + " |")
    else:
        if label_left == label[0]:
            label_left = np.append(label_left,label[1]) 
            label_left_count = [label_left_count[0],0]
        else:
            label_left = np.append(label[0],label_left)
            label_left_count = [0,label_left_count[0]]
            
        print(space + node.node_att + ' = ' +  att_ans[0] + ': ' + '[' + str(label_left_count[0]) + ' ' + label_left[0] + ' /' + str(label_left_count[1]) + ' ' + label_left[1] + ']')
        PRINTING(node.left, space + " |")
    return 
    
            


PRINTING(TREE,j=1)

f = open(train_out,"w+")
g = open(test_out,"w+")
h = open(metrics_out,"w+")
z =  0
def Classify(node,data,TEST):
    global z 
    
    if isinstance(node, leaf):
        if TEST == 1:
            g.write(node.answer + '\n')
            xx,yy = np.unique(node.data[:,-1],return_counts = True)
            
            if node.answer == data[-1]:
                z += 0
            else:
                z += 1        
        else:
            f.write(node.answer + '\n')
        return node.error
    
    Attribute = list(Attributes)
    att_index  = Attribute.index(node.node_att)
    if data[att_index] ==  att_ans_full[0]:
        Classify(node.left,data,TEST)
    else:
        Classify(node.right,data,TEST)
    return 


Error =  np.zeros(1000)
for i in range(len(training[:,0])):
    Error[i]  = Classify(TREE,training[i,:],2)

ERROR_TEST = np.zeros(1000)
for i in range(len(testing[:,0])):
    ERROR_TEST[i] = Classify(TREE,testing[i,:],1)
    
f.close()
g.close()
zzz = y/np.sum(label_count)

y = 0
label_TEST, label_count_TEST = np.unique(testing[:,-1],return_counts = True)
yyy = z/np.sum(label_count_TEST)
h.write(f'error(train): {zzz}' + '\n')
h.write(f'error(test): {yyy}')
h.close