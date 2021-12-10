# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:37:33 2021

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
from deepquantum.gates.qtensornetwork import StateVec2MPS,MPS2StateVec,MPS_expec,Rho2MPS,MPS2Rho
from deepquantum.embeddings import PauliEncoding


if 1:
    print('CDO线路：')
    N = 10
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    #psi1 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    # psi = torch.zeros(1,2**N)+0.0j
    # psi[0,0] = 1.0+0j;#psi[0,-1] = 1.0+0j
    # psi = nn.functional.normalize( psi,p=2,dim=1 )
    
    a1 = Circuit(N)
    #part1====================================================================
    a1.u3([torch.pi/2.0,0,0], 1)
    a1.ry(0.98, 2)
    a1.ry(1.45, 3)
    a1.ry(1.5, 4)
    a1.u1(0, 1)
    a1.u3([-0.125,0,0], 2)
    a1.u3([-0.191,0,0], 3)
    a1.u3([-0.116,0,0], 4)
    
    a1.cnot([1,0])
    a1.u3([0.791,0,0], 0)
    a1.cnot([1,0])
    a1.u3([torch.pi/2.0,0,0], 0)
    a1.cnot([1,0])
    a1.u1(0, 0)
    a1.cnot([1,0])
    a1.u1(0, 0)
    
    a1.cnot([0,2])
    a1.u3([0.125,0,0], 2)
    a1.cnot([0,2])
    a1.u3([-0.251,0,0], 2)
    a1.cnot([0,3])
    a1.cnot([1,2])
    a1.u3([0.191,0,0], 3)
    a1.u3([0.251,0,0], 2)
    
    a1.cnot([0,3])
    a1.cnot([1,2])
    a1.u3([-0.383,0,0], 3)
    a1.cnot([0,4])
    a1.cnot([1,3])
    a1.u3([0.116,0,0], 4)
    a1.u3([0.383,0,0], 3)
    a1.cnot([0,4])
    a1.cnot([1,3])
    a1.u3([-0.231,0,0], 4)
    a1.cnot([1,4])
    a1.u3([0.231,0,0], 4)
    a1.cnot([1,4])
    #part2====================================================================
    a1.toffoli([2,6,8])
    a1.cnot([2,6])
    a1.toffoli([2,8,7])
    a1.PauliX(6)
    a1.toffoli([2,6,8])
    a1.PauliX(6)
    a1.toffoli([3,6,8])
    a1.cnot([3,6])
    a1.toffoli([3,8,7])
    
    a1.PauliX(6)
    a1.toffoli([3,6,8])
    a1.PauliX(6)
    a1.toffoli([4,6,8])
    a1.cnot([4,6])
    a1.toffoli([4,8,7])
    a1.PauliX(6)
    a1.toffoli([4,6,8])
    a1.PauliX(6)
    #part3====================================================================
    a1.ry(3.0*torch.pi/8, 5)
    a1.cnot([6,9])
    a1.u3([torch.pi,0,torch.pi], 7)
    a1.u3([torch.pi,0,torch.pi], 8)
    a1.u3([torch.pi,0,torch.pi], 9)
    a1.toffoli([7,9,8])
    a1.u3([torch.pi,0,torch.pi], 7)
    a1.u3([torch.pi,0,torch.pi], 9)
    a1.cu3([-torch.pi/12.0,0,0], [8,5])
    a1.cnot([6,9])
    a1.cu3([torch.pi/24.0,0,0], [8,5])
    a1.toffoli([6,8,5])
    a1.cu3([-torch.pi/24.0,0,0], [8,5])
    a1.toffoli([6,8,5])
    
    a1.cu3([torch.pi/12.0,0,0], [8,5])
    a1.cnot([6,9])
    a1.toffoli([7,8,5])
    a1.u3([-torch.pi,-torch.pi,0], 9)
    a1.cu3([-torch.pi/12.0,0,0], [8,5])
    a1.toffoli([7,8,5])
    a1.u3([-torch.pi,-torch.pi,0], 7)
    a1.toffoli([7,9,8])
    a1.u3([-torch.pi,-torch.pi,0], 7)
    a1.u3([-torch.pi,-torch.pi,0], 8)
    a1.u3([-torch.pi,-torch.pi,0], 9)
    a1.cnot([6,9])
    #part4====================================================================
    a1.PauliX(6)
    a1.toffoli([4,6,8])
    a1.PauliX(6)
    a1.toffoli([4,8,7])
    a1.cnot([4,6])
    a1.toffoli([4,6,8])
    a1.PauliX(6)
    a1.toffoli([3,6,8])
    a1.PauliX(6)
    a1.toffoli([3,8,7])
    
    a1.cnot([3,6])
    a1.toffoli([3,6,8])
    a1.PauliX(6)
    a1.toffoli([2,6,8])
    a1.PauliX(6)
    a1.toffoli([2,8,7])
    a1.cnot([2,6])
    a1.toffoli([2,6,8])
    
    t1 = time.time()
    MPS = StateVec2MPS(psi,N)
    MPS = a1.TN_evolution(MPS)
    psif1 = MPS2StateVec(MPS).view(1,-1)
    t2 = time.time()
    
    print('耗时：',t2-t1)
    
    
input('END')
































    


 














































































































        