import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from deepquantum.gates import Circuit as cir
from deepquantum.gates.qmath import multi_kron, dag
from deepquantum.gates.qoperator import rx, ry, rz, Operation

# ===============================encoding layer=================================

# def PauliEncoding(N, input_lst, pauli='X'):
#     if N < len(input_lst):
#         raise ValueError("number of inputs must be less than number of qubits")
#     num = len(input_lst)
#     if pauli == 'X':
#         E = cir.multi_kron([cir.rx(input_lst[i % num]) for i in range(N)])
#     elif pauli == 'Y':
#         E = cir.multi_kron([cir.ry(input_lst[i % num]) for i in range(N)])
#     elif pauli == 'Z':
#         E = cir.multi_kron([cir.rz(input_lst[i % num]) for i in range(N)])
#     else:
#         raise ValueError("pauli parameter must be one of X Y Z")
#     return E


# def AmplitudeEncoding(N, input_lst):
#     if 2 ** N < len(input_lst):
#         raise ValueError("number of inputs must be less than dimension 2^N")

#     num = len(input_lst)

#     norm = 0.0
#     for each in torch.abs(input_lst):
#         norm += each ** 2

#     input_lst = (1.0 / torch.sqrt(norm)) * input_lst
#     state = torch.zeros([2 ** N]) + 0j
#     for i in range(num):
#         state[i] = input_lst[i]
#     return state


class PauliEncoding(Operation):
    def __init__(self, N, input_lst, wires, pauli='X'):
        
        if N < len(input_lst):
            raise ValueError("number of inputs must be less than number of qubits")
        self.nqubits = N
        self.input_lst = input_lst
        self.pauli = pauli
        self.wires = wires
        
        self.state0 = torch.zeros(2**self.nqubits)
        self.state0[0] = 1
        self.state0 = self.state0 + 0j
        
        self.rho0 = torch.zeros( [ 2**self.nqubits, 2**self.nqubits ] )
        self.rho0[0][0] = 1
        self.rho0 = self.rho0 + 0j
        
    def U_expand(self):
        num = len(self.input_lst)
        
        lst1 = [torch.eye(2)]*self.nqubits
        
        if self.pauli == 'X':
            
            for i,qbit in enumerate(self.wires):
                lst1[qbit] = rx(self.input_lst[i % num]).matrix
            E = multi_kron(lst1)
        
        elif self.pauli == 'Y':
            
            for i,qbit in enumerate(self.wires):
                lst1[qbit] = ry(self.input_lst[i % num]).matrix
            E = multi_kron(lst1)
        
        elif self.pauli == 'Z':
            
            for i,qbit in enumerate(self.wires):
                lst1[qbit] = rz(self.input_lst[i % num]).matrix
            E = multi_kron(lst1)
        
        else:
            raise ValueError("pauli parameter must be one of X Y Z")
        
        return E
    


class AmplitudeEncoding(object):
    
    def __init__(self,N,input_lst):
        
        if 2 ** N < len(input_lst):
            raise ValueError("number of inputs must be less than dimension 2^N")
        self.nqubits = N
        self.input_lst = input_lst
        
        self.state0 = torch.zeros(2**self.nqubits)
        self.state0[0] = 1
        self.state0 = self.state0 + 0j
        
        self.rho0 = torch.zeros( [ 2**self.nqubits, 2**self.nqubits ] )
        self.rho0[0][0] = 1
        self.rho0 = self.rho0 + 0j
        
    def encoded_state(self):
        num = len(self.input_lst)

        norm = 0.0
        for each in torch.abs(self.input_lst):
            norm += each ** 2
    
        input_lst = (1.0 / torch.sqrt(norm)) * self.input_lst #输入数据归一化
        
        state = torch.zeros([ 2**self.nqubits ]) + 0j
        for i in range(num):
            state[i] = input_lst[i]
        
        return state + 0j #得到encoding之后的态矢
        
        
    pass




























