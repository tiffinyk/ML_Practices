# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:38:29 2021

@author: ASUS
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time

def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split("\t")
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def randPickj(i,m):
    j=i
    while j==i:
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

def weight(data,label,alphas):
    dataMatrix=np.mat(data);labelMatrix=np.mat(label).transpose() #转置
    m,n=dataMatrix.shape
    w=np.mat(np.zeros((1,n)))
    
    for i in range(m):
        if alphas[i]>0:
            w+=labelMatrix[i]*alphas[i]*dataMatrix[i,:]
    return w.tolist() #转换为列表

def plotBestFit(weights,b,filename):
    dataMat,labelMat=loadDataSet(filename)
    dataArr=np.array(dataMat)
    n=dataArr.shape[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,0])
            ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0])
            ycord2.append(dataArr[i,1])
    
    fig=plt.figure(figsize=(12,8))
    plt.scatter(xcord1,ycord1,c="red",s=50,label="label=1")
    plt.scatter(xcord2,ycord2,c="blue",s=50,label="label=-1")
    
    #绘制决策边界
    x=np.arange(2.0,7.0,0.1)
    y=(-b-weights[0][0]*x)/weights[0][1]
    x.shape=(len(x),1);y.shape=(len(x),1)
    plt.plot(x,y,color="darkorange",linewidth=3.0,label="Boarder")
    
    plt.xlabel("X1", fontsize=16)
    plt.ylabel("X2", fontsize=16)
    plt.title("SMO BestFit",fontsize=20,fontweight="bold")
    plt.legend()
    plt.show()
    
    
class optStruct:
    def __init__(self,data,label,C,toler):
        self.X=data
        self.labelMatrix=label
        self.C=C
        self.toler=toler
        self.m=data.shape[0]
        
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.Es=np.mat(np.zeros((self.m,2)))
        self.b=0
        
def calcEk(oS,k):
    gxk=float(np.multiply(oS.alphas,oS.labelMatrix).transpose()*(oS.X*oS.X[k,:].transpose()))+oS.b
    Ek=gxk-float(oS.labelMatrix[k])
    return Ek

#选择相较ai具有最大步长的函数
def selectJ(oS,i,Ei):
    maxK=-1;maxDeltaE=0;Ej=0
    oS.Es[i]=[1,Ei]
    validEsList=np.nonzero(oS.Es[:,0].A)[0]
    
    if len(validEsList)>1:
        for k in validEsList:
            if k==i:
                continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if deltaE>maxDeltaE:
                maxDeltaE=deltaE;maxK=k;Ej=Ek
        return maxK,Ej
    else:
        j=randPickj(i,oS.m)
        Ej=calcEk(oS, j)
    return j,Ej

def updateEk(oS,k):
    Ek=calcEk(oS, k)
    oS.Es[k]=[1,Ek]
    
#内循环
def innerL(i,oS):
    Ei=calcEk(oS,i)
    
    if (oS.labelMatrix[i]*Ei<-oS.toler and oS.alphas[i]<oS.C) or \
    (oS.labelMatrix[i]*Ei>oS.toler and oS.alphas[i]>0):
        j,Ej=selectJ(oS,i,Ei)
        alphaIold=oS.alphas[i].copy();alphaJold=oS.alphas[j].copy()
        
        if oS.labelMatrix[i]!=oS.labelMatrix[j]:
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        
        #计算eta
        eta=oS.X[i,:]*oS.X[i,:].transpose()+oS.X[j,:]*oS.X[j,:].transpose()-2.0*oS.X[i,:]*oS.X[j,:].transpose()
        if eta==0:
            print("eta==0")
            return 0
        
        #根据学习方法中的结果公式得到alphaj的解析解，并更新Ej值
        oS.alphas[j]=oS.alphas[j]+oS.labelMatrix[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j], H, L)
        updateEk(oS,j)
        
        if abs(oS.alphas[j]-alphaJold)<0.00001:
            print("j not moving enough")
            return 0
        
        oS.alphas[i]=oS.alphas[i]+oS.labelMatrix[i]*oS.labelMatrix[j]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)

        b1=-Ei-oS.labelMatrix[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].transpose()\
        -oS.labelMatrix[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[i,:].transpose()+oS.b
        
        b2=-Ej-oS.labelMatrix[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].transpose()\
        -oS.labelMatrix[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].transpose()+oS.b
    
        if oS.alphas[i]>0 and oS.alphas[i]<oS.C:
            oS.b=b1
        elif oS.alphas[j]>0 and oS.alphas[j]<oS.C:
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    
    else:
        return 0
    
#外循环
def SMOpro(data,label,C,toler,maxIter,kTup=("lin",0)):
    oS=optStruct(np.mat(data), np.mat(label).transpose(), C, toler)
    iter=0;entireSet=True;alphaPairsChanged=0
    
    while (iter<maxIter) and (entireSet or alphaPairsChanged>0):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i, oS)
                print ("fullset, iter:%d i:%d, pairChanged: %d" %(iter,i,alphaPairsChanged))
            iter+=1
            
        else:
            boundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<oS.C))[0]
            for i in boundIs:
                alphaPairsChanged+=innerL(i, oS)
                print ("bound, iter:%d i:%d, pairsChanged: %d" %(iter,i,alphaPairsChanged))
            iter+=1
            
        if entireSet:
            entireSet=False
        elif alphaPairsChanged==0:
            entireSet=True
        print ("iteration number: %d" %iter)
    return oS.b,oS.alphas

data3,label3=loadDataSet("testSet.txt")
print(data3)
print(label3)
start=time.time()
b3,alphas3=SMOpro(data3,label3,0.6,0.001,60)
print ("\n","time used:.{0}s".format(time.time()-start))
w3=weight(data3,label3,alphas3)
plotBestFit(w3, b3, "testSet.txt")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    