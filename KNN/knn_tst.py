# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 09:16:57 2021

@author: ASUS
"""

import random
import csv

#读取数据
with open('Prostate_Cancer.csv','r') as file:
    #以字典形式读取文件里的数据
    reader=csv.DictReader(file)
    #做一个推导，将每一行存到datas里
    datas=[row for row in reader]

#分组
random.shuffle(datas)
n=len(datas)//3

test_set=datas[0:n]
train_set=datas[n:]

#KNN
#计算距离（最好做一个归一化）
def distance(d1, d2):
    res=0
    
    for key in ("radius","texture","perimeter","area","smoothness","compactness","symmetry","fractal_dimension"):
        #这里的key是字符串列表，遍历列表里所有的项目
        res+=(float(d1[key])-float(d2[key]))**2
    
    return res**0.5

K=6
def knn(data):
    #1.距离
    res=[
        {"result": train['diagnosis_result'], "distance":distance(data, train)}
        for train in train_set
        ]
    #这里的res是一个字典性数据的列表，调用了distance函数
    
    #2.排序
    res=sorted(res, key=lambda item:item['distance'])
    #升序
    
    #3.取前K个
    res2=res[0:K]
    
    #4.总距离
    sum=0
    for r in res2:
        sum+=r['distance']
    
    #5.加权平均
    result={'B':0, 'M':0}  #字典初始化
           
    for r in res2:
        result[r['result']]+=1-r['distance']/sum #用1-的方式因而权重会大于1
        #print(result)
    
    if result['B']>result['M']:
        return 'B'
    else:
        return 'M'
    
#测试阶段
correct=0
for test in test_set:
    result=test['diagnosis_result']
    result2=knn(test)
    
    if result==result2:
        correct+=1
        
print("准确率：{:.2f}%".format(100*correct/len(test_set)))