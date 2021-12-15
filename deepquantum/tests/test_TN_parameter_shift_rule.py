# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:39:37 2021

@author: shish
"""
import torch
import deepquantum as dq
from deepquantum.gates.qmath import multi_kron, measure, IsUnitary, IsNormalized
import deepquantum.gates.qoperator as op
from deepquantum.gates.qcircuit import Circuit,parameter_shift
from deepquantum.embeddings.qembedding import PauliEncoding
from deepquantum.layers.qlayers import YZYLayer, ZXLayer,ring_of_cnot, ring_of_cnot2, BasicEntangleLayer
from deepquantum.gates.qtensornetwork import StateVec2MPS,MPS2StateVec,MPS_expec,Rho2MPS,MPS2Rho
from deepquantum.embeddings import PauliEncoding

'''
N = 4
wires = list(range(N))
params_lst = torch.rand(6*N) * (2*torch.pi)

I = torch.eye(2) + 0j
lst = [I] * N
lst[0] = dq.PauliZ().matrix
M = multi_kron(lst)

c1 = Circuit(N)

c1.YZYLayer(wires, params_lst[0:3*N])
c1.ring_of_cnot(wires)
c1.YZYLayer(wires, params_lst[3*N:6*N])


psi = torch.zeros(1,2**N) + 0j
psi[0,0] = 1+0j
MPS_i = StateVec2MPS(psi, N)
MPS_f = c1.TN_evolution(MPS_i)
psi_f = MPS2StateVec(MPS_f)


rst = psi_f.view(1,-1).conj() @ M @ psi_f.view(-1,1)
rst = rst.squeeze()
'''

def forward(params_lst):
    N = 4
    wires = list(range(N))
    params_lst = params_lst + torch.tensor(0.0)

    I = torch.eye(2) + 0j
    lst = [I] * N
    lst[0] = dq.PauliZ().matrix
    M = multi_kron(lst)

    c1 = Circuit(N)

    c1.YZYLayer(wires, params_lst[0:3*N])
    c1.ring_of_cnot(wires)
    c1.YZYLayer(wires, params_lst[3*N:6*N])
    c1.ring_of_cnot(wires)
    c1.YZYLayer(wires, params_lst[6*N:9*N])


    psi = torch.zeros(1,2**N) + 0j
    psi[0,0] = 1+0j
    MPS_i = StateVec2MPS(psi, N)
    MPS_f = c1.TN_evolution(MPS_i)
    psi_f = MPS2StateVec(MPS_f)


    rst = psi_f.view(1,-1).conj() @ M @ psi_f.view(-1,1)
    rst = rst.squeeze().real
    return rst


def partial_M(params_lst):
    rst_lst = []
    for i,p in enumerate(params_lst):
        params_lst[i] = params_lst[i] + 0.5*torch.pi
        M1 = forward(params_lst)
        params_lst[i] = params_lst[i] - 1*torch.pi
        M2 = forward(params_lst)
        params_lst[i] = params_lst[i] + 0.5*torch.pi
        temp = 0.5*(M1-M2)
        #print(temp)
        rst_lst.append(temp)
    return torch.tensor(rst_lst)
'''
测试tensor network通过parameter shift rule计算梯度进而梯度下降
'''

N = 4
params_lst = torch.rand(9*N) * (2*torch.pi)
#print('p: ',params_lst[5])
lr = 0.1
for i in range(20):
    grad = partial_M(params_lst)
    # print('g: ',grad,'size: ',grad.shape)
    # print('p: ',params_lst[5])
    params_lst = params_lst - lr*grad
    #print('p: ',params_lst[5])
    print('i：',i,' 期望值：',forward(params_lst))

#==============================================================================

N = 4
wires = list(range(N))
params_lst = torch.rand(9*N) * (2*torch.pi)

I = torch.eye(2) + 0j
lstM = [I]*N
lstM[0] = dq.PauliZ().matrix
#lstM[1] = dq.PauliZ().matrix
M = multi_kron(lstM)

lr = 0.1
for i in range(50):
    c1 = Circuit(N)
    c1.YZYLayer(wires, params_lst[0:3*N])
    c1.ring_of_cnot(wires)
    c1.YZYLayer(wires, params_lst[3*N:6*N])
    c1.ring_of_cnot(wires)
    c1.YZYLayer(wires, params_lst[6*N:9*N])
    # c1.ring_of_cnot(wires)
    # c1.YZYLayer(wires, params_lst[9*N:12*N])
    # c1.ring_of_cnot(wires)
    # c1.YZYLayer(wires, params_lst[12*N:15*N])
    print(i,'  ',c1.cir_expectation(c1.state_init().view(1,-1), M))
    
    
    ps = parameter_shift(c1, c1.state_init().view(1,-1), M)
    grad = ps.cal_params_grad()
    params_lst = params_lst - lr*grad
    

    
input('END')
    


