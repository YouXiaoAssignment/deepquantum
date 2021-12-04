# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:30:23 2021

@author: shish
"""


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time


from deepquantum.gates.qmath import multi_kron, measure, IsUnitary, IsNormalized
import deepquantum.gates.qoperator as op
from deepquantum.gates.qcircuit import Circuit
from deepquantum.embeddings.qembedding import PauliEncoding
from deepquantum.layers.qlayers import YZYLayer, ZXLayer,ring_of_cnot, ring_of_cnot2, BasicEntangleLayer
from deepquantum.gates.qtensornetwork import StateVec2MPS,MPS2StateVec,MPS_expec
from deepquantum.embeddings import PauliEncoding


if 0:
    N = 4    #量子线路的qubit总数
    wires_lst = list(range(N))
    
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    MPS = StateVec2MPS(psi,N)
    for each in MPS:
        print(each.shape)
    
if 1:
    '''
    测试pauliencoding层加入TN功能后是否正确
    '''
    N = 5    #量子线路的qubit总数
    wires_lst = list(range(N))
    input_lst = [1,0.3,4.7]
    p1 = PauliEncoding(N,input_lst,wires_lst)
    
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    psif0 = (p1.U_expand() @ psi.view(-1,1)).view(1,-1)
    
    MPSf = p1.TN_operation( StateVec2MPS(psi,N) )
    psif1 = MPS2StateVec(MPSf)
    
    print(psif0)
    print(psif1)



#if __name__ == "__main__":
if 0:    
    N = 10    #量子线路的qubit总数
    wires_lst = list(range(N))
    weight = torch.rand(21*N) * 2 * torch.pi
    
    c1 = Circuit(N)
    
    c1.YZYLayer(wires_lst, weight[0:3*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[3*N:6*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[6*N:9*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[9*N:12*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[12*N:15*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[15*N:18*N])
    c1.ring_of_cnot(wires_lst)
    c1.YZYLayer(wires_lst, weight[18*N:21*N])
    
    c2 = Circuit(N)
    c2.BasicEntangleLayer(wires_lst, weight[0:18*N],repeat=6)
    c2.YZYLayer(wires_lst, weight[18*N:21*N])
    
    itern = 10
    T1 = 0.0;T2 = 0.0
    for ii in range(itern):
        
        #psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
        psi = torch.zeros(1,2**N)+0.0j
        psi[0,0] = 1.0+0j;#psi[0,-1] = 1.0+0j
        psi = nn.functional.normalize( psi,p=2,dim=1 )
        
        t1 = time.time()
        MPS0 = StateVec2MPS(psi, N)
        MPS_f = c1.TN_evolution(MPS0)
        psi_f0 = MPS2StateVec(MPS_f)
        t2 = time.time()
        
        
        
        T1 = T1 + (t2 - t1)
        
        t3 = time.time()
        U = c2.U()
        psi_f1 = (U @ psi.view(-1,1) ).view(1,-1) 
        t4 = time.time()
        T2 = T2 + (t4 - t3)
    
    print(psi_f0)
    print(psi_f1)
    print('矩阵相乘:',T2/itern,' 张量网络:',T1/itern)
    #13qubit,优化矩阵后313s对8.5s，引入SWAP后TN：0.21s
    #12qubit时，65s对3s,优化矩阵后40s对2.2s，引入SWAP后40ss对0.16s
    #11qubit时，引入SWAP后5.3s对0.11s
    #10qubit是，1.5s对0.22s,优化矩阵后0.8s对0.2s,引入SWAP后0.8s对0.1s
    #14qubit:0.3s，15qubit:0.45s，16qubit:0.91s，18qubit:4.7s
    
input('END')
































    


 














































































































        