# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:16:17 2021

@author: shish
"""
import torch
from deepquantum.layers.qlayers import *
from deepquantum.gates.qoperator import *
from deepquantum.gates.qmath import multi_kron, measure, IsUnitary


class Circuit(object):
    def __init__(self, N):
        self.nqubits = N  # 总QuBit的个数
        self.gate = []  # 顺序添加各类门
        self._U = torch.eye(2**self.nqubits) + 0j     # 线路酉矩阵
        
        #线路的初始态，默认全为|0>态
        self.state_init = torch.zeros(2**self.nqubits)
        self.state_init[0] = 1
        self.state_init = self.state_init + 0j
        

    def add(self, gate):
        self.gate.append(gate)

    
    def U(self, left_to_right=True):
        U_overall = torch.eye(2 ** self.nqubits, 2 ** self.nqubits) + 0j
        
        for i,each_oper in enumerate( self.gate ):
            
            u_matrix = each_oper.U_expand()
            
            if left_to_right:
                U_overall = u_matrix @ U_overall
            else:
                U_overall = U_overall @ u_matrix
        
        self._U = U_overall
        
        return U_overall
    
    
    def clear(self):
        self.gate = []
        self._U = torch.eye(2**self.nqubits) + 0j
        
        
