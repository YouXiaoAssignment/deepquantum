# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 08:49:16 2021

@author: shish
"""
import torch
import torch.nn as nn
import random
import math
import time

from typing import List

def StateVec2MPS(psi:torch.Tensor, N:int, d:int=2)->List[torch.Tensor]:
    #t1 = time.time()
    #输入合法性检测：输入的态矢必须是1行2^N列的张量
    if len(psi.shape) != 2:
        raise ValueError('StateVec2MPS:input dimension error!')
    if psi.shape[0] != 1 or psi.shape[1] != 2**N:
        raise ValueError('StateVec2MPS:input shape must be 1 ROW 2^N COLUMN')
    
    c_tensor = psi + 0j
    rst_lst = []
    for i in range(N):
        #按照列(dim=1)把张量c_tensor对半分成2块(chunk=2)，再按照行叠加
        c_tensor_block = torch.chunk(c_tensor, chunks=2, dim=1)
        c_tensor = torch.cat((c_tensor_block[0], c_tensor_block[1]), dim=0)
        
        U,S,V = torch.svd( c_tensor )
        V_d = V.permute(1,0).conj()
        D = len( (S[torch.abs(S)>1e-6]).view(-1) ) #D：bond dimension
        #print(D)
        S = torch.diag(S) + 0j
        #根据bond dimension对张量进行缩减
        if D < S.shape[0]:
            U = torch.index_select(U, 1, torch.tensor(list(range(D))))
            S = torch.index_select(S, 0, torch.tensor(list(range(D))))
        
        rst_lst.append(U.view(2,-1,U.shape[1]))
        c_tensor = S @ V_d
    #t2 = time.time()
    #print('SV2MPS:',t2-t1)
    return rst_lst


# def TensorContraction(TensorA:torch.Tensor,TensorB:torch.Tensor,dimA:int,dimB:int):
#     '''
#     将TensorA的dimA维度与TensorB的dimB维度进行相乘收缩
#     '''
#     rankA = len(TensorA.shape)
#     rankB = len(TensorB.shape)
#     if dimA > rankA - 1 or dimB > rankB - 1:
#         raise ValueError('TensorContraction: dimA/dimB must less than rankA/rankB')
#     if TensorA.shape[dimA] != TensorB.shape[dimB]:
#         raise ValueError('TensorContraction: dimA&dimB not match')
    
#     permuteA = list(range(rankA)) 
#     permuteA.pop(dimA)
#     permuteA.append(dimA)
#     permuteA = tuple(permuteA)
    
#     permuteB = list(range(rankB))
#     permuteB.pop(dimB)
#     permuteB.append(dimB)
#     permuteB = tuple(permuteB)
    
#     TensorA = TensorA.permute(permuteA)
#     TensorB = TensorB.permute(permuteB)
#     pass

def MPS2StateVec(tensor_lst:List[torch.Tensor])->torch.Tensor:
    #t1 = time.time()
    N = len(tensor_lst)
    for i in range(N):
        temp = tensor_lst[i].unsqueeze(0)
        if i == 0:
            c_tensor = tensor_lst[i]
        else:
            c_tensor = c_tensor.unsqueeze(1) @ temp
            shape = c_tensor.shape
            c_tensor = c_tensor.view(shape[0]*shape[1],shape[2],shape[3])
    c_tensor = c_tensor.view(-1)
    #t2 = time.time()
    #print('MPS:',t2-t1)
    return c_tensor #返回1行2^N列的张量，表示态矢的系数


def TensorDecompAfterTwoQbitGate(tensor:torch.Tensor):
    #t1 = time.time()
    #tensor = tensor.reshape(tensor.shape[0]*tensor.shape[2],tensor.shape[1]*tensor.shape[3])
    block1 = torch.cat((tensor[0,0],tensor[0,1]),dim=1)
    block2 = torch.cat((tensor[1,0],tensor[1,1]),dim=1)
    tensor = torch.cat((block1,block2),dim=0)
    U,S,V = torch.svd( tensor )
    V_d = V.permute(1,0).conj()
    D = len( (S[torch.abs(S)>1e-6]).view(-1) ) #D：bond dimension
    S = torch.diag(S) + 0j
    #根据bond dimension对张量进行缩减
    if D < S.shape[0]:
        U = torch.index_select(U, 1, torch.tensor(list(range(D))))
        S = torch.index_select(S, 0, torch.tensor(list(range(D))))
    
    rst1 = (U.view(2,-1,U.shape[1]))
    
    rst2 = S @ V_d
    rst2_block = torch.chunk(rst2, chunks=2, dim=1)
    rst2 = torch.cat((rst2_block[0], rst2_block[1]), dim=0)
    rst2 = rst2.view(2,-1,rst2.shape[1])
    #print(rst1.shape,'  and  ',rst2.shape)
    #t2 = time.time()
    #print('DECOM:',t2-t1)
    return rst1,rst2


def MPS_inner_product(ketMPS,braMPS):
    #上限12个qubit
    N = len(ketMPS)
    lst = []
    for i in range(N):
        t1 = ketMPS[i]
        t2 = braMPS[i]
        t1 = t1.permute(1,2,0)
        t2 = t2.permute(1,2,0)
        
        for i in range(3):
            t1 = t1.unsqueeze(2*i+1)
            t2 = t2.unsqueeze(2*i)
        qbit_tensor = (t2.conj() @ t1).squeeze()
        #print(qbit_tensor.shape)
        lst.append(qbit_tensor)
    
    for i in range(len(lst)-1):
        if i == 0:
            t = lst[i]
            
        t_nxt = lst[i+1]
        if i != len(lst) - 2:
            t_nxt = t_nxt.permute(2,3,0,1)
        
        t = t.unsqueeze(-2).unsqueeze(-4) #1212
        t_nxt = t_nxt.unsqueeze(-1).unsqueeze(-3) #442121
        
        temp = t @ t_nxt #442211
        temp = temp.squeeze() #4422
        #print('temp:',temp.shape)
        if i != len(lst) - 2:
            trace_temp = torch.zeros(temp.shape[0],temp.shape[1]) + 0j
            for r,d1 in enumerate(temp):
                for c,d2 in enumerate(d1):
                    trace_temp[r,c] = torch.trace(d2)
            t = trace_temp
        else:
            t = torch.trace(temp)
        #print('new t:',t.shape)
    #return torch.abs(t)
    return t

def MPS_expec(MPS,wires:List[int],local_obserable:List[torch.Tensor]):
    '''
    local_obserable是一个包含局部力学量的list
    wires也是一个list，表明每个局部力学量作用在哪个qubit上
    '''
    SV0 = MPS2StateVec(MPS).view(1,-1)
    
    for i,qbit in enumerate(wires):
        temp = MPS[qbit]
        temp = temp.permute(1,2,0).unsqueeze(-1) #2421
        temp = torch.squeeze(local_obserable[i] @ temp, dim=3) #242 在指定维度squeeze
        MPS[qbit] = temp.permute(2,0,1)
    expec = SV0.conj() @ MPS2StateVec(MPS).view(-1,1)
    return expec.squeeze().real
    
        

if __name__ == "__main__":
    '''
    12qubit时，SV2MPS平均要5ms，MPS2SV平均要3ms
    '''
    N = 12 #19个就是上限了，20个比特我的电脑立刻死给你看
    psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    
    #psi = nn.functional.normalize( torch.ones(1,2**N)+torch.rand(1,2**N)*0.0j,p=2,dim=1 )
    
    # psi = torch.zeros(1,2**N)+0.0j
    # psi[0,0] = 1.0+0j;#psi[0,-1] = 1.0+0j
    # psi = nn.functional.normalize( psi,p=2,dim=1 )
    '''
    验证StateVec2MPS和MPS2StateVec的正确性
    '''
    psi0 = psi
    lst = StateVec2MPS(psi,N)
    psi1 = MPS2StateVec(lst)
    print('psi0:',psi0)
    print('psi1:',psi1)
    
    '''
    统计StateVec2MPS和MPS2StateVec的平均耗时
    '''
    T1 = 0.0;T2 = 0.0
    itern = 300
    for i in range(itern):
        psi = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
        
        # psi = torch.zeros(1,2**N)+0.0j
        # psi[0,0] = 1.0+0j;psi[0,-1] = 1.0+0j
        # psi = nn.functional.normalize( psi,p=2,dim=1 )
        
        t1 = time.time()
        lst = StateVec2MPS(psi,N)
        t2 = time.time()
        psi1 = MPS2StateVec(lst)
        t3 = time.time()
        T1 += t2 - t1
        T2 += t3 - t2
    print('SV2MPS:',T1/itern,'    MPS2SV:',T2/itern)
    
    # psi1 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    # psi2 = nn.functional.normalize( torch.rand(1,2**N)+torch.rand(1,2**N)*1j,p=2,dim=1 )
    # print(( psi2.conj() @ psi1.permute(1,0) ).squeeze())
    # lst1 = StateVec2MPS(psi1,N)
    # lst2 = StateVec2MPS(psi2,N)
    # rst = MPS_inner_product(lst1,lst2)
    
    input("")