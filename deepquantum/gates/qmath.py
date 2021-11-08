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


