# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:16:17 2021

@author: shish
"""
import torch
from deepquantum.layers.qlayers import *
from deepquantum.gates.qoperator import *


class Circuit(object):
    def __init__(self, N):
        self.nqubits = N  # 总QuBit的个数
        self.gate = []  # 顺序添加各类门
        self._U = torch.eye(2**self.nqubits) + 0j     # 线路酉矩阵

    def add(self, gate):
        self.gate.append(gate)

    def U(self, left_to_right=True):
        U_overall = torch.eye(2 ** self.nqubits, 2 ** self.nqubits) + 0j
        for U in self.gate:
            if left_to_right:
                U_overall = U.U_expand() @ U_overall
            else:
                U_overall = U_overall @ U.U_expand()
        self._U = U_overall
        return U_overall
    
    def clean(self):
        self.gate = []
        self._U = torch.eye(2**self.nqubits) + 0j
        
        
