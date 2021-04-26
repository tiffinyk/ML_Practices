# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:04:32 2019

@author: ASUS
"""
a1=list(map(int,input().split()))
t=[0,0];

for k in range(a1[0]-1):
    for j in range(a1[0]-1-k):
        if a1[2*(j+1)]>a1[2*(j+2)]:
           
            t[0]=a1[2*(j+1)-1];
            t[1]=a1[2*(j+1)];
            a1[2*(j+1)-1]=a1[2*(j+2)-1];
            a1[2*(j+1)]=a1[2*(j+2)];
            a1[2*(j+2)-1]=t[0];
            a1[2*(j+2)]=t[1];

    
for s in range(2*a1[0]):
    print(a1[s+1]);