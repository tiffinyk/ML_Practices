# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:17:56 2019

@author: ASUS
"""

class Node:
    def __init__(self,coef,exp):
        self.coef=coef
        self.exp=exp
        self.next=None
    def get_data(self):
        return [self.coef,self.exp]
class List:
    def __init__(self,head):
        self.head=head
 
    #添加节点
    def addNode(self,node):
        temp=self.head
        while temp.next is not None:
            temp=temp.next
        temp.next=node           

    #打印
    def printLink(self,head):
        res=[]
        while head is not None:
            res.append(head.get_data())
            head=head.next
        return res

def adds(l1,l2):#l1,l2为链表,且不为空
    p1=l1.head   
    p2=l2.head
    addRes=[]
    while (p1 is not None) and (p2 is not None) :
        tmp1_exp=p1.get_data()[1]
        tmp2_exp=p2.get_data()[1]
        #当指数相同时，系数相加
        if tmp1_exp == tmp2_exp:
            addRes.append([p1.get_data()[0]+p2.get_data()[0],p1.get_data()[1]])
            p1=p1.next
            p2=p2.next
        if tmp1_exp < tmp2_exp:
            addRes.append([p1.get_data()[0],p1.get_data()[1]])
            p1=p1.next
        if tmp1_exp > tmp2_exp:
            addRes.append([p2.get_data()[0],p2.get_data()[1]])
            p2=p2.next
    while p1 is not None:
        addRes.append([p1.get_data()[0],p1.get_data()[1]])
        p1=p1.next
    while p2 is not None:
        addRes.append([p2.get_data()[0],p2.get_data()[1]])
        p2=p2.next
        
    l3=[]
    for item in addRes:
        if item[0]!=0:
            l3.append(item[0])
            l3.append(item[1])
    if len(l3) == 0:
        return [0,0]
    return l3


def print_list(x):
    for i in x[:-1]:
        print(i,end=' ')
    print(x[-1],end='')
       
 #输入
print("请依次输入两个多项式的项数，系数以及指数：")
a1=list(map(int,input().split()))
a2=list(map(int,input().split()))

#变为链表
if a1[0]!=0:
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
    head1=Node(a1[1],a1[2])
    l1=List(head1)
    if a1[0]>1:
        for i in range(a1[0]-1):
            node=Node(a1[i*2+3],a1[i*2+4])
            l1.addNode(node)
    
if a2[0]!=0:
    t=[0,0];
    for k in range(a2[0]-1):
        for j in range(a2[0]-1-k):
            if a2[2*(j+1)]>a2[2*(j+2)]:
                t[0]=a2[2*(j+1)-1];
                t[1]=a2[2*(j+1)];
                a2[2*(j+1)-1]=a2[2*(j+2)-1];
                a2[2*(j+1)]=a2[2*(j+2)];
                a2[2*(j+2)-1]=t[0];
                a2[2*(j+2)]=t[1];
    head2=Node(a2[1],a2[2])
    l2=List(head2)
    if a2[0]>1:
        for i in range(a2[0]-1):
            node=Node(a2[i*2+3],a2[i*2+4])
            l2.addNode(node)
#考虑链表长度进行运算
if len(a1)==1 and len(a2)==1:        #都为0，则输出都为0
    print_list([0,0])
elif len(a1)==1 and len(a2)>1:    #一个为0，另一个为多项式
    print_list(a2[1:])
elif len(a2)==1 and len(a1)>1:
    print_list(a1[1:])
else:                                   #都为多项式
    print_list(adds(l1,l2))