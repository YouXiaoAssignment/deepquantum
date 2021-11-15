import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


def multi_kron(lst):
    rst = lst[0]
    for i in range( 1,len(lst) ):
        rst = torch.kron(rst, lst[i])
    return rst


def dag(x):
    """
    compute conjugate transpose of input matrix
    """
    x_conj = torch.conj(x)
    x_dag = x_conj.permute(1, 0)
    return x_dag


def IsUnitary(in_matrix):
    '''
    判断一个矩阵是否是酉矩阵
    '''
    if (in_matrix.shape)[0] != (in_matrix.shape)[1]:  # 验证是否为方阵
        raise ValueError("not square matrix!")
        return False

    n = in_matrix.shape[0]  # 行数

    for i in range(n):  # 每行是否归一
        summ = 0.0
        for j in range(n):
            summ += (torch.abs(in_matrix[i][j])) ** 2
        if torch.abs(summ - 1) > 1e-6:
            print("not unitary! not normalized")
            raise ValueError("not unitary matrix! not normalized")
            return False

    for i in range(n - 1):  # 行之间是否正交
        for k in range(i + 1, n):
            summ = 0.0 + 0.0 * 1j
            for j in range(n):
                summ += in_matrix[i][j] * (in_matrix[k][j]).conj()
            if torch.abs(summ) > 1e-6:
                print("not unitary! not orthogonal")
                raise ValueError("not unitary matrix! not orthogonal")
                return False

    return True



def IsNormalized(vector):
    '''
    判断一个矢量是否归一
    '''
    if len(vector.shape) != 1:  # 验证是否为方阵
        raise ValueError("not vector!")

    n = vector.shape[0]  # 向量元素数

    summ = 0.0
    for i in range(n):
        summ += (torch.abs(vector[i])) ** 2
    if torch.abs(summ - 1) > 1e-6:
        #print("vector is not normalized")
        return False
        #raise ValueError("vector is not normalized")
        
    return True



def IsHermitian(matrix):
    '''
    判断一个矩阵是否是厄米矩阵
    '''
    if (matrix.shape)[0] != (matrix.shape)[1]:  # 验证是否为方阵
        raise ValueError("not square matrix!")

    n = matrix.shape[0] #行数
    
    for i in range(n):
        for j in range(i,n,1):
            if torch.abs(matrix[i][j] - matrix[j][i].conj()) > 1e-6:
                return False
    
    return True
    



def ptrace(rhoAB, dimA, dimB):
    """
    rhoAB : density matrix
    dimA: n_qubits A keep
    dimB: n_qubits B trash
    """
    mat_dim_A = 2 ** dimA
    mat_dim_B = 2 ** dimB

    id1 = torch.eye(mat_dim_A, requires_grad=True) + 0.j
    id2 = torch.eye(mat_dim_B, requires_grad=True) + 0.j

    pout = 0
    for i in range(mat_dim_B):
        p = torch.kron(id1, id2[i]) @ rhoAB @ torch.kron(id1, id2[i].reshape(mat_dim_B, 1))
        pout += p
    return pout





def partial_trace(rho,N,trace_lst):
    '''
    trace_lst里面是想trace掉的qubit的索引号，须从小到大排列
    '''
    #输入合法性检测
    if abs(torch.trace(rho) - 1) > 1e-6:
        raise ValueError("trace of density matrix must be 1")
    if rho.shape[0] != 2**N:
        raise ValueError('rho dim error')
    
    trace_lst.sort()#必须从小到大排列
    rho = rho + 0j
    if len(trace_lst) == 0:
        return rho + 0j
    
    id1 = torch.eye(2**(trace_lst[0])) + 0j
    id2 = torch.eye(2**(N-1-trace_lst[0])) + 0j
    id3 = torch.eye(2) + 0j
    rho_nxt = torch.tensor(0)
    for i in range(2):
        A = torch.kron( torch.kron(id1,id3[i]), id2 ) + 0j
        rho_nxt = rho_nxt + A @ rho @ dag(A)
    
    new_lst = [ i-1 for i in trace_lst[1:] ] #trace掉一个qubit，他后面的qubit索引号要减1
    
    return partial_trace(rho_nxt,N-1,new_lst) + 0j







def measure(state,M,rho=False,physic=False):
    if not rho: #输入态为态矢，而非密度矩阵
        if len(state.shape) != 2: #state必须是二维张量，即便只有1个态矢也要view成(n,1)
            raise ValueError("state必须是二维张量,即便batch只有1个态矢也要view成(n,1)")
        else: #state为batch_size个态矢，即二维张量
            
            m1 = (dag(state) @ M @ state)
            
            rst = torch.diag(m1).view(-1,1) #取对角元变成1维张量，在被view成2维张量
            rst = rst.real
            return rst
                  
    else:#state是1个密度矩阵，此时不支持batch
        if torch.abs(torch.trace(state) - 1) > 1e-4:
            raise ValueError("trace of density matrix must be 1")
        return torch.trace(state @ M).real 


