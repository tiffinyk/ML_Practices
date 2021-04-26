# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 10:22:12 2021

@author: ASUS
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
#matplot inline
file_name = r"testSet.txt"
file = pd.read_table(file_name, header=None, names=["factor1","factor2","class"])
file.head()

positive = file[file["class"]==1]
negative = file[file["class"]==-1]
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(positive["factor1"], positive["factor2"], s=30, c="b", marker="o", label="class 1")
ax.scatter(negative["factor1"], negative["factor2"], s=30, c="r", marker="x", label="class -1")
ax.legend()
ax.set_xlabel("factor1")
ax.set_ylabel("factor2")

def load_data_set(file):
    orig_data = file.values
    cols = orig_data.shape[1] #取列数
    data_mat = orig_data[:,0:cols-1]
    label_mat = orig_data[:,cols-1:cols]
    #print(data_mat)
    #print(label_mat)
    #print(cols)
    return data_mat, label_mat

data_mat, label_mat = load_data_set(file) 
#输出一个与j不同的整数
def select_jrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

#将aj限制在L和H之间
def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smo_simple(data_mat, class_label, C, toler, max_iter):
    #循环外的初始化工作
    data_mat = np.mat(data_mat)
    label_mat = np.mat(class_label)
    b=0
    m, n = np.shape(data_mat)
    alphas = np.zeros((m,1))
    iter = 0
    while iter < max_iter:
        #内循环的初始化工作
        alpha_pairs_changed = 0
        for i in range(m):
            WT_i = np.dot(np.multiply(alphas, label_mat).T, data_mat)
            f_xi = float(np.dot(WT_i, data_mat[i,:].T)) + b
            Ei = f_xi - float(label_mat[i])
            if((label_mat[i]*Ei < -toler) and (alphas[i] < C)) or ((label_mat[i]*Ei > toler) and (alphas[i] > 0)):
                j = select_jrand(i, m)
                WT_j = np.dot(np.multiply(alphas, label_mat).T, data_mat)
                f_xj = float(np.dot(WT_j, data_mat[j,:].T)) + b
                Ej = f_xj - float(label_mat[j])
                alpha_iold = alphas[i].copy()
                alpha_jold = alphas[j].copy()
                
                if (label_mat[i] != label_mat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] -C)
                    H = min(C, alphas[j] + alphas[i])
                if H == L :
                    continue
                
                eta = 2.0 * data_mat[i,:]*data_mat[j,:].T - data_mat[i,:]*data_mat[i,:].T - \
                data_mat[j,:]*data_mat[j,:].T
                
                if eta >= 0:
                    continue
                alphas[j] = (alphas[j] - label_mat[j]*(Ei - Ej))/eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if (abs(alphas[j] - alpha_jold) < 0.00001):
                    continue
                
                alphas[i] = alphas[i] + label_mat[j]*label_mat[i]*(alpha_jold - alphas[j])
                
                
                b1 = b - Ei + label_mat[i]*(alpha_iold - alphas[i])*np.dot(data_mat[i,:], data_mat[i,:].T) + \
                label_mat[j]*(alpha_jold - alphas[j])*np.dot(data_mat[i,:], data_mat[j,:].T)
                b2 = b - Ej + label_mat[i]*(alpha_iold - alphas[i])*np.dot(data_mat[i,:], data_mat[j,:].T) + \
                label_mat[j]*(alpha_jold - alphas[j])*np.dot(data_mat[j,:], data_mat[j,:].T)
                
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2 )/2.0
                
                alpha_pairs_changed += 1
        
        if (alpha_pairs_changed == 0): iter += 1
        else: iter = 0
    return b, alphas


b, alphas = smo_simple(data_mat, label_mat, 0.6, 0.001, 10)
print(b, alphas[alphas>0])                
                
#画出分类后的图片
support_x = []
support_y = []
class1_x = []
class1_y = []
class2_x = []
class2_y = []
for i in range(100):
    if alphas[i] > 0.0:
        support_x.append(data_mat[i,0])
        support_y.append(data_mat[i,1])

for i in range(100):
    if label_mat[i] == 1:
        class1_x.append(data_mat[i,0])
        class1_y.append(data_mat[i,1])
    else:
        class2_x.append(data_mat[i,0])
        class2_y.append(data_mat[i,1])

w_best = np.dot(np.multiply(alphas, label_mat).T, data_mat)
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(support_x, support_y, s=100, c="y", marker="v", label="support_v")
ax.scatter(class1_x, class1_y, s=30, c="b", marker="o", label="class 1")
ax.scatter(class2_x, class2_y, s=30, c="r", marker="x", label="class -1")  
lin_x = np.linspace(3, 6)
lin_y = (-float(b) - w_best[0,0]*lin_x) / w_best[0,1]
plt.plot(lin_x, lin_y, color="black")
ax.legend()
ax.set_xlabel("factor1")
ax.set_ylabel("factor2")             
                
                
 
    