# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:06:10 2021

@author: shish
"""
import torch
from deepquantum.gates import multi_kron
from deepquantum.gates.qoperator import Hadamard,rx,ry,rz,rxx,ryy,rzz,cnot,cz,Operation

class XYZLayer(Operation):
    label = "XYZLayer"
    
    def __init__(self,N,wires,params_lst):
        if 3*len(wires) != len(params_lst):
            raise ValueError("XYZLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XYZLayer: number of wires must less than N")
        self.nqubits = N
        self.wires = wires
        self.params = params_lst
        
    def U_expand(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            xm = rx(self.params[3*i+0]).matrix
            ym = ry(self.params[3*i+1]).matrix
            zm = rz(self.params[3*i+2]).matrix
            
            lst1[qbit] = zm @ ym @ xm
        
        return multi_kron(lst1) + 0j
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info
    







class YZYLayer(Operation):
    label = "YZYLayer"
    
    def __init__(self,N,wires,params_lst):
        if 3*len(wires) != len(params_lst):
            raise ValueError("YZYLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("YZYLayer: number of wires must less than N")
        self.nqubits = N
        self.wires = wires
        self.params = params_lst
        
    def U_expand(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            y1m = ry(self.params[3*i+0]).matrix
            zm = rz(self.params[3*i+1]).matrix
            y2m = ry(self.params[3*i+2]).matrix
            
            lst1[qbit] = y2m @ zm @ y1m
        
        return multi_kron(lst1) + 0j
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info










class XZXLayer(Operation):
    label = "XZXLayer"
    
    def __init__(self,N,wires,params_lst):
        if 3*len(wires) != len(params_lst):
            raise ValueError("XZXLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XZXLayer: number of wires must less than N")
        self.nqubits = N
        self.wires = wires
        self.params = params_lst
        
    def U_expand(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            x1m = rx(self.params[3*i+0]).matrix
            zm = rz(self.params[3*i+1]).matrix
            x2m = rx(self.params[3*i+2]).matrix
            
            lst1[qbit] = x2m @ zm @ x1m
        
        return multi_kron(lst1) + 0j
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info









class XZLayer(Operation):
    label = "XZLayer"
    
    def __init__(self,N,wires,params_lst):
        if 2*len(wires) != len(params_lst):
            raise ValueError("XZLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XZLayer: number of wires must less than N")
        self.nqubits = N
        self.wires = wires
        self.params = params_lst
        
    def U_expand(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            xm = rx(self.params[2*i+0]).matrix
            zm = rz(self.params[2*i+1]).matrix

            lst1[qbit] = zm @ xm
        
        return multi_kron(lst1) + 0j
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info












class ZXLayer(Operation):
    label = "ZXLayer"
    
    def __init__(self,N,wires,params_lst):
        if 2*len(wires) != len(params_lst):
            raise ValueError("ZXLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("ZXLayer: number of wires must less than N")
        self.nqubits = N
        self.wires = wires
        self.params = params_lst
        
    def U_expand(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):
            
            zm = rz(self.params[2*i+0]).matrix
            xm = rx(self.params[2*i+1]).matrix

            lst1[qbit] = xm @ zm
        
        return multi_kron(lst1) + 0j
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':self.params}
        return info






class HLayer(Operation):
    label = "HadamardLayer"
    
    def __init__(self,N,wires):
        if len(wires) > N:
            raise ValueError("HadamardLayer: number of wires must less than N")
        
        self.nqubits = N
        self.wires = wires
        
        
    def U_expand(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):

            lst1[qbit] = Hadamard.matrix
        
        return multi_kron(lst1) + 0j
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':None}
        return info





#==============================================================================




class ring_of_cnot(Operation):
    label = "ring_of_cnot_Layer"
    
    def __init__(self,N,wires):
        
        if len(wires) > N:
            raise ValueError("ring_of_cnotLayer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("ring_of_cnotLayer: number of wires must >= 2")
        self.nqubits = N
        self.wires = wires
        
        
    def U_expand(self):
        L = len(self.wires)
        if L == 2:
            return cnot( self.nqubits,[ self.wires[0],self.wires[1] ]).U_expand()
    
        I = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        for i,qbit in enumerate( self.wires ):
            
            rst = cnot(self.nqubits,[ self.wires[i],self.wires[(i+1)%L] ]).U_expand() @ I

        return rst
        
    def info(self):
        L = len(self.wires)
        target_lst = [self.wires[(i+1)%L] for i in range(L)]
        if L == 2:
            info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        else:
            info = {'label':self.label, 'contral_lst':self.wires, 'target_lst':target_lst,'params':None}
        return info








class ring_of_cnot2(Operation):
    label = "ring_of_cnot2_Layer"
    
    def __init__(self,N,wires):
        
        if len(wires) > N:
            raise ValueError("ring_of_cnotLayer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("ring_of_cnotLayer: number of wires must >= 2")
        self.nqubits = N
        self.wires = wires
        
        
    def U_expand(self):
        L = len(self.wires)
        if L == 2:
            return cnot( self.nqubits,[ self.wires[0],self.wires[1] ]).U_expand()
    
        I = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        for i,qbit in enumerate( self.wires ):
            
            rst = cnot(self.nqubits,[ self.wires[i],self.wires[(i+2)%L] ]).U_expand() @ I

        return rst
        
    def info(self):
        L = len(self.wires)
        target_lst = [self.wires[(i+2)%L] for i in range(L)]
        if L == 2:
            info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        else:
            info = {'label':self.label, 'contral_lst':self.wires, 'target_lst':target_lst,'params':None}
        return info







if __name__ == '__main__':
    print('start')
    N = 2
    p = torch.rand(3*N)
    a = ring_of_cnot(N,list(range(N)))
    print(a.label)
    print(a.U_expand())
    print(a.info())
    input('')