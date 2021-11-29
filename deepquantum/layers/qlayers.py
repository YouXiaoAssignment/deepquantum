# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:06:10 2021

@author: shish
"""
import torch
from deepquantum.gates import multi_kron
from deepquantum.gates.qoperator import Hadamard,rx,ry,rz,rxx,ryy,rzz,cnot,cz,Operation
import time

class XYZLayer(Operation):
    #label = "XYZLayer"
    
    def __init__(self,N,wires,params_lst):
        if 3*len(wires) != len(params_lst):
            raise ValueError("XYZLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XYZLayer: number of wires must less than N")
        self.label = "XYZLayer"
        
        self.nqubits = N
        self.wires = wires
        self.params = params_lst
        self.num_params = len(params_lst)
        
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
    
    def params_update(self,params_lst):
        pass
    







class YZYLayer(Operation):
    # label = "YZYLayer"
    
    def __init__(self,N,wires,params_lst):
        if 3*len(wires) != len(params_lst):
            raise ValueError("YZYLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("YZYLayer: number of wires must less than N")
        self.label = "YZYLayer"
        
        self.nqubits = N
        self.wires = wires
        self.params = params_lst
        self.num_params = len(params_lst)
        
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
    
    def params_update(self,params_lst):
        pass










class XZXLayer(Operation):
    # label = "XZXLayer"
    
    def __init__(self,N,wires,params_lst):
        if 3*len(wires) != len(params_lst):
            raise ValueError("XZXLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XZXLayer: number of wires must less than N")
        self.label = "XZXLayer"
        
        self.nqubits = N
        self.wires = wires
        self.params = params_lst
        self.num_params = len(params_lst)
        
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
    
    def params_update(self,params_lst):
        pass









class XZLayer(Operation):
    # label = "XZLayer"
    
    def __init__(self,N,wires,params_lst):
        if 2*len(wires) != len(params_lst):
            raise ValueError("XZLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("XZLayer: number of wires must less than N")
        self.label = "XZLayer"
        
        self.nqubits = N
        self.wires = wires
        self.params = params_lst
        self.num_params = len(params_lst)
        
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
    
    def params_update(self,params_lst):
        pass












class ZXLayer(Operation):
    # label = "ZXLayer"
    
    def __init__(self,N,wires,params_lst):
        if 2*len(wires) != len(params_lst):
            raise ValueError("ZXLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("ZXLayer: number of wires must less than N")
        self.label = "ZXLayer"
        
        self.nqubits = N
        self.wires = wires
        self.params = params_lst
        self.num_params = len(params_lst)
        
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
    
    def params_update(self,params_lst):
        pass






class HLayer(Operation):
    # label = "HadamardLayer"
    
    def __init__(self,N,wires):
        if len(wires) > N:
            raise ValueError("HadamardLayer: number of wires must less than N")
        
        self.label = "HadamardLayer"
        
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
        
        
    def U_expand(self):
        lst1 = [torch.eye(2,2)]*self.nqubits
        for i,qbit in enumerate( self.wires ):

            lst1[qbit] = Hadamard().matrix
        
        return multi_kron(lst1) + 0j
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires,'params':None}
        return info
    
    def params_update(self,params_lst):
        pass





#==============================================================================




class ring_of_cnot(Operation):
    # label = "ring_of_cnot_Layer"
    
    def __init__(self,N,wires):
        
        if len(wires) > N:
            raise ValueError("ring_of_cnotLayer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("ring_of_cnotLayer: number of wires must >= 2")
        self.label = "ring_of_cnot_Layer"
        
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
    
    def _gate_fusion_U_expand(self,N):
        if N < 3:
            raise ValueError('ring of cnot : gate_fusion error! N must be >= 3')
        I = torch.eye(2,2) + 0j
        rst = cnot( 2,[0,1] ).U_expand()
        for i in range( 1, N ):
            cur_M = cnot( min(2+i,N),[i, (i+1)%N] ).U_expand()
            if i == N-1:
                rst = cur_M @ rst
            else:
                rst = cur_M @ torch.kron(rst,I)
        return rst
        
    def U_expand(self):
        L = len(self.wires)
        if L == 2:
            return cnot( self.nqubits,[ self.wires[0],self.wires[1] ]).U_expand()
        
        if self.wires == list( range(self.nqubits) ):
            return self._gate_fusion_U_expand(self.nqubits)
        
        #I = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        rst = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        for i,qbit in enumerate( self.wires ):
            
            #rst = cnot(self.nqubits,[ self.wires[i],self.wires[(i+1)%L] ]).U_expand() @ I
            rst = cnot(self.nqubits,[ self.wires[i],self.wires[(i+1)%L] ]).U_expand() @ rst

        return rst
        
    def info(self):
        L = len(self.wires)
        target_lst = [self.wires[(i+1)%L] for i in range(L)]
        if L == 2:
            info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        else:
            info = {'label':self.label, 'contral_lst':self.wires, 'target_lst':target_lst,'params':None}
        return info
    
    def params_update(self,params_lst):
        pass








class ring_of_cnot2(Operation):
    # label = "ring_of_cnot2_Layer"
    
    def __init__(self,N,wires):
        
        if len(wires) > N:
            raise ValueError("ring_of_cnot2Layer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("ring_of_cnot2Layer: number of wires must >= 2")
        self.label = "ring_of_cnot2_Layer"
        
        self.nqubits = N
        self.wires = wires
        self.num_params = 0
    
    
    def _gate_fusion_U_expand(self,N):
        if N < 3:
            raise ValueError('ring of cnot : gate_fusion error! N must be >= 3')
        I = torch.eye(2,2) + 0j
        rst = cnot( 3,[0,2] ).U_expand()
        for i in range( 1, N ):
            cur_M = cnot( min(3+i,N),[i, (i+2)%N] ).U_expand()
            if i == N-2 or i == N-1:
                rst = cur_M @ rst
            else:
                rst = cur_M @ torch.kron(rst,I)
        return rst    

    
    def U_expand(self):
        L = len(self.wires)
        if L == 2:
            return cnot( self.nqubits,[ self.wires[0],self.wires[1] ]).U_expand()
        
        if self.wires == list( range(self.nqubits) ):
            return self._gate_fusion_U_expand(self.nqubits)
        
        #I = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        rst = torch.eye(2**self.nqubits,2**self.nqubits) + 0j
        for i,qbit in enumerate( self.wires ):
            
            rst = cnot(self.nqubits,[ self.wires[i],self.wires[(i+2)%L] ]).U_expand() @ rst

        return rst
        
    def info(self):
        L = len(self.wires)
        target_lst = [self.wires[(i+2)%L] for i in range(L)]
        if L == 2:
            info = {'label':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        else:
            info = {'label':self.label, 'contral_lst':self.wires, 'target_lst':target_lst,'params':None}
        return info
    
    def params_update(self,params_lst):
        pass







#=========================================================================================




class BasicEntangleLayer(Operation):
    # label = "BasicEntangleLayer"
    
    def __init__(self, N, wires, params_lst, repeat=1):
        
        if 3*len(wires)*repeat != len(params_lst):
            raise ValueError("BasicEntangleLayer: number of parameters not match")
        if len(wires) > N:
            raise ValueError("BasicEntangleLayer: number of wires must <= N")
        if len(wires) < 2:
            raise ValueError("BasicEntangleLayer: number of wires must >= 2")
        if repeat < 1:
            raise ValueError("BasicEntangleLayer: number of repeat must >= 1")
        self.label = "BasicEntangleLayer"
        
        self.nqubits = N
        self.wires = wires
        self.num_params = len(params_lst)
        self.params = params_lst
        self.repeat = repeat
        
        self.part1_lst, self.part2_lst = [], []
        for i in range(int(self.repeat)):
            self.part1_lst.append( YZYLayer(self.nqubits, self.wires, self.params[i*3*len(wires):(i+1)*3*len(wires)]) )
            self.part2_lst.append( ring_of_cnot(self.nqubits, self.wires) )
            
        
    def U_expand(self):
        rst = torch.eye(2**self.nqubits) + 0j
        cnot_ring = self.part2_lst[0].U_expand()
        for i in range(self.repeat):
            #rst = self.part2_lst[i].U_expand() @ self.part1_lst[i].U_expand() @ rst
            rst = cnot_ring @ self.part1_lst[i].U_expand() @ rst
        return rst
        
    def info(self):
        info = {'label':self.label, 'contral_lst':[], 'target_lst':self.wires, 'params':self.params}
        return info
    
    def params_update(self,params_lst):
        self.num_params = len(params_lst)
        self.params = params_lst
        self.part1_lst, self.part2_lst = [], []
        L = 3*len(self.wires)
        for i in range(self.repeat):
            self.part1_lst.append( YZYLayer(self.nqubits, self.wires, self.params[i*L:(i+1)*L]) )
            self.part2_lst.append( ring_of_cnot(self.nqubits, self.wires) )
        













if __name__ == '__main__':
    print('start')
    N = 10
    wires = list(range(N))
    roc = ring_of_cnot2(N,wires)
    
    t1 = time.time()
    for i in range(10):
        r1 = roc.U_expand()
    t2 = time.time()
    for i in range(10):
        r2 = roc._gate_fusion_U_expand(roc.nqubits)
    t3 = time.time()
    print('r1-r2:',r1-r2)
    #print('r2:',r2)
    print('old:',t2-t1)
    print('new:',t3-t2)
    print('耗时比：',(t3-t2)/(t2-t1))
    # N = 2
    # p = torch.rand(3*N)
    # a = ring_of_cnot(N,list(range(N)))
    # print(a.label)
    # print(a.U_expand())
    # print(a.info())
    input('')