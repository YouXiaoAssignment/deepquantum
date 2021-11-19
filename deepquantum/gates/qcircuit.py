# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:16:17 2021

@author: shish
"""
import torch
from collections.abc import Iterable
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

    
    def U(self, left_to_right=True,cuda=False):
        
        U_overall = torch.eye(2 ** self.nqubits, 2 ** self.nqubits) + 0j
        for i,each_oper in enumerate( self.gate ):
            u_matrix = each_oper.U_expand()        
            if left_to_right:
                U_overall = u_matrix @ U_overall
            else:
                U_overall = U_overall @ u_matrix
        self._U = U_overall
       
        return U_overall
    
    def draw(self):
        for each in self.gate:
            pass
        pass
    
    def clear(self):
        self.gate = []
        self._U = torch.eye(2**self.nqubits) + 0j
        
    def Hadamard(self, wires):
        if isinstance(wires, Iterable):
            self.add( HLayer(self.nqubits, wires) )
        else:
            self.add( Hadamard(self.nqubits, wires) )
    
    def PauliX(self, wires):
        self.add( PauliX(self.nqubits, wires) )
    
    def PauliY(self, wires):
        self.add( PauliY(self.nqubits, wires) )
    
    def PauliZ(self, wires):
        self.add( PauliZ(self.nqubits, wires) )
    
    def rx(self, theta, wires):
        self.add( rx(theta, self.nqubits, wires) )
    
    def ry(self, theta, wires):
        self.add( ry(theta, self.nqubits, wires) )
    
    def rz(self, theta, wires):
        self.add( rz(theta, self.nqubits, wires) )
    
    def rxx(self, theta, wires):
        self.add( rxx(theta, self.nqubits, wires) )
    
    def ryy(self, theta, wires):
        self.add( ryy(theta, self.nqubits, wires) )
    
    def rzz(self, theta, wires):
        self.add( rzz(theta, self.nqubits, wires) )
    
    def cnot(self, wires):
        self.add( cnot(self.nqubits, wires) )
    
    def cz(self, wires):
        self.add( cz(self.nqubits, wires) )
    
    def cphase(self, theta, wires):
        self.add( cphase(theta, self.nqubits, wires) )
    
    def SWAP(self, wires):
        self.add( SWAP(self.nqubits, wires) )
    
    def toffoli(self, wires):
        self.add( toffoli(self.nqubits, wires) )
    
    def multi_control_cnot(self, wires):
        self.add( multi_control_cnot(self.nqubits, wires) )
    #====================================================================    
    def XYZLayer(self, wires, params_lst):
        self.add( XYZLayer(self.nqubits, wires, params_lst) )
    
    def YZYLayer(self, wires, params_lst):
        self.add( YZYLayer(self.nqubits, wires, params_lst) )
    
    def XZXLayer(self, wires, params_lst):
        self.add( XZXLayer(self.nqubits, wires, params_lst) )
    
    def XZLayer(self, wires, params_lst):
        self.add( XZLayer(self.nqubits, wires, params_lst) )
    
    def ZXLayer(self, wires, params_lst):
        self.add( ZXLayer(self.nqubits, wires, params_lst) )
    
    def ring_of_cnot(self, wires):
        self.add( ring_of_cnot(self.nqubits, wires) )
    
    def ring_of_cnot2(self, wires):
        self.add( ring_of_cnot2(self.nqubits, wires) )
    
    def BasicEntangleLayer(self, wires, params_lst, repeat=1):
        self.add( BasicEntangleLayer(self.nqubits, wires, params_lst, repeat=1) )
    
    
if __name__ == '__main__':
    cir = Circuit(3)
    
    cir.cphase(1.2, [1,2])
    
    cir.Hadamard([0,1])
    
    cir.PauliX(0)
    
    cir.cnot([1,0])
    
    cir.rx(torch.pi/6.0, 1)
    
    cir.BasicEntangleLayer([0,1,2], torch.rand(9))
    
    cir.SWAP([0,2])
    
    for each in cir.gate:
        print(each.info()['label'])
    
    print('\n',cir.U())
    
    input('')
        
        
