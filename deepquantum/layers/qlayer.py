# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:00:55 2021

@author: shishunynag
"""
# import numpy as np
import torch
# import math
from qutorch.qgate import rx, ry, rz, multi_kron, cnot, cz, IsUnitary, Hadamard, rxx, ryy, rzz


def XYZLayer(N, parameter_lst):
    if N < 1:
        raise ValueError("number of qubits(N) must be >= 1")
    if len(parameter_lst) != 3 * N:
        raise ValueError("number of parameters must be equal to 3*N")

    single_gate_lst = []
    for i in range(N):
        theta1 = parameter_lst[3 * i + 0]
        theta2 = parameter_lst[3 * i + 1]
        theta3 = parameter_lst[3 * i + 2]
        single_gate = torch.matmul(rz(theta3), torch.matmul(ry(theta2), rx(theta1)))
        single_gate_lst.append(single_gate)

    return multi_kron(single_gate_lst)


def YZYLayer(N, parameter_lst):
    if N < 1:
        raise ValueError("number of qubits(N) must be >= 1")
    if len(parameter_lst) != 3 * N:
        raise ValueError("number of parameters must be equal to 3*N")

    single_gate_lst = []
    for i in range(N):
        theta1 = parameter_lst[3 * i + 0]
        theta2 = parameter_lst[3 * i + 1]
        theta3 = parameter_lst[3 * i + 2]
        single_gate = torch.matmul(ry(theta3), torch.matmul(rz(theta2), ry(theta1)))
        single_gate_lst.append(single_gate)

    return multi_kron(single_gate_lst)


def XZXLayer(N, parameter_lst):
    if N < 1:
        raise ValueError("number of qubits(N) must be >= 1")
    if len(parameter_lst) != 3 * N:
        raise ValueError("number of parameters must be equal to 3*N")

    single_gate_lst = []
    for i in range(N):
        theta1 = parameter_lst[3 * i + 0]
        theta2 = parameter_lst[3 * i + 1]
        theta3 = parameter_lst[3 * i + 2]
        single_gate = torch.matmul(rx(theta3), torch.matmul(rz(theta2), rx(theta1)))
        single_gate_lst.append(single_gate)

    return multi_kron(single_gate_lst)


def XZLayer(N, parameter_lst):
    if N < 1:
        raise ValueError("number of qubits(N) must be >= 1")
    if len(parameter_lst) != 2 * N:
        raise ValueError("number of parameters must be equal to 3*N")

    single_gate_lst = []
    for i in range(N):
        theta1 = parameter_lst[2 * i + 0]
        theta2 = parameter_lst[2 * i + 1]
        single_gate = torch.matmul(rz(theta2), rx(theta1))
        single_gate_lst.append(single_gate)

    return multi_kron(single_gate_lst)


def HLayer(N):
    if N < 1:
        raise ValueError("number of qubits(N) must be >= 1")
    single_gate_lst = []
    for i in range(N):
        single_gate_lst.append(Hadamard())
    return multi_kron(single_gate_lst)


# ===============================================================================
def ring_of_cnot(N, up2down=True):
    '''
    N表示qubit数目，up2down默认为True，表示以线路上方的qubit优先为控制比特
    '''
    if N < 2:
        raise ValueError("ring of cnot must be applied on at least 2 qubits")

    if N == 2:  # 当只有两个qubit时，ring of cnot就是1个cnot
        if up2down:
            return cnot(2, 0, 1)
        else:
            return cnot(2, 1, 0)

    rst = torch.eye(2 ** N, 2 ** N) + 0j
    if up2down:
        for i in range(N):
            rst = torch.matmul(cnot(N, i, (i + 1) % N), rst)
        return rst

    else:  # 当然，设置up2down为False就可优先以线路下方的qubit为控制比特
        for i in range(N - 1, -1, -1):
            rst = torch.matmul(cnot(N, i, (i - 1) % N), rst)
        return rst


def ring_of_cnot2(N, up2down=True):
    '''
    次近邻ring of cnot，即0控制2,1控制3,2控制4...N-2控制0，N-1控制1
    N表示qubit数目，up2down默认为True，表示以线路上方的qubit优先为控制比特
    '''
    if N < 2:
        raise ValueError("ring of cnot must be applied on at least 2 qubits")

    if N == 2:  # 当只有两个qubit时，ring of cnot就是1个cnot
        if up2down:
            return cnot(2, 0, 1)
        else:
            return cnot(2, 1, 0)

    rst = torch.eye(2 ** N, 2 ** N) + 0j
    if up2down:
        for i in range(N):
            rst = torch.matmul(cnot(N, i, (i + 2) % N), rst)
        return rst

    else:  # 当然，设置up2down为False就可优先以线路下方的qubit为控制比特
        for i in range(N - 1, -1, -1):
            rst = torch.matmul(cnot(N, i, (i - 2) % N), rst)
        return rst


def ring_of_cz(N, up2down=True):
    if N < 2:
        raise ValueError("ring of cz must be applied on at least 2 qubits")

    if N == 2:  # 当只有两个qubit时，ring of cz就是1个cz
        if up2down:
            return cz(2, 0, 1)
        else:
            return cz(2, 1, 0)

    rst = torch.eye(2 ** N, 2 ** N) + 0j
    if up2down:
        for i in range(N):
            rst = torch.matmul(cz(N, i, (i + 1) % N), rst)
        return rst

    else:
        for i in range(N - 1, -1, -1):
            rst = torch.matmul(cz(N, i, (i - 1) % N), rst)
        return rst


def nearest_neighbor_cnot(N, up2down=True):
    '''
    最近邻构型的CNOT模块
    '''
    if N < 2:
        raise ValueError("nearest_neighbor_cnot must be applied on at least 2 qubits")

    if N == 2:  # 当只有两个qubit时，nearest_neighbor_cnot模块就是1个cnot
        if up2down:
            return cnot(2, 0, 1)
        else:
            return cnot(2, 1, 0)

    rst = torch.eye(2 ** N, 2 ** N) + 0j
    if up2down:
        for i in range(0, N - 1, 2):
            rst = torch.matmul(cnot(N, i, i + 1), rst)
        for i in range(1, N - 1, 2):
            rst = torch.matmul(cnot(N, i, i + 1), rst)
        return rst

    else:  # 当然，设置up2down为False就可优先以线路下方的qubit为控制比特
        for i in range(N - 1, 0, -2):
            rst = torch.matmul(cnot(N, i, i - 1), rst)
        for i in range(N - 2, 0, -2):
            rst = torch.matmul(cnot(N, i, i - 1), rst)
        return rst


def nearest_neighbor_cz(N, up2down=True):
    '''
    最近邻构型的CZ模块
    '''
    if N < 2:
        raise ValueError("nearest_neighbor_cz must be applied on at least 2 qubits")

    if N == 2:  # 当只有两个qubit时，nearest_neighbor_cz模块就是1个cz
        if up2down:
            return cz(2, 0, 1)
        else:
            return cz(2, 1, 0)

    rst = torch.eye(2 ** N, 2 ** N) + 0j
    if up2down:
        for i in range(0, N - 1, 2):
            rst = torch.matmul(cz(N, i, i + 1), rst)
        for i in range(1, N - 1, 2):
            rst = torch.matmul(cz(N, i, i + 1), rst)
        return rst

    else:  # 当然，设置up2down为False就可优先以线路下方的qubit为控制比特
        for i in range(N - 1, 0, -2):
            rst = torch.matmul(cz(N, i, i - 1), rst)
        for i in range(N - 2, 0, -2):
            rst = torch.matmul(cz(N, i, i - 1), rst)
        return rst


def all2all_cnot(N, up2down=True):
    if N < 2:
        raise ValueError("all to all cnot must be applied on at least 2 qubits")

    if N == 2:  # 当只有两个qubit时，nearest_neighbor_cz模块就是1个cz
        if up2down:
            return cnot(2, 0, 1)
        else:
            return cnot(2, 1, 0)

    rst = torch.eye(2 ** N, 2 ** N) + 0j
    if up2down:
        for i in range(N):
            for j in range(N):
                if j != i:
                    rst = torch.matmul(cnot(N, i, j), rst)
        return rst

    else:  # 当然，设置up2down为False就可优先以线路下方的qubit为控制比特
        for i in range(N - 1, -1, -1):
            for j in range(N - 1, -1, -1):
                if j != i:
                    rst = torch.matmul(cnot(N, i, j), rst)
        return rst


# ==========================组合而成的一些更复杂layer============================

def basic_entangle_layer(N, param_lst, single_gate='YZY'):
    if len(single_gate) == 3 and len(param_lst) != 3 * N:
        raise ValueError("num of parameters not match")
    if len(single_gate) == 2 and len(param_lst) != 2 * N:
        raise ValueError("num of parameters not match")

    if single_gate == 'YZY':
        S1 = YZYLayer(N, param_lst[0:3 * N])

    elif single_gate == 'XZX':
        S1 = XZXLayer(N, param_lst[0:3 * N])

    elif single_gate == 'XYZ':
        S1 = XYZLayer(N, param_lst[0:3 * N])

    elif single_gate == 'XZ':
        S1 = XZLayer(N, param_lst[0:2 * N])

    else:
        raise ValueError("single_gate='YZY' or 'XZX' or 'XYZ'.")

    C1 = ring_of_cnot(N)
    return C1 @ S1


def basic_entangle_layer2(N, param_lst, single_gate='YZY'):
    if len(single_gate) == 3 and len(param_lst) != 6 * N:
        raise ValueError("num of parameters not match")
    if len(single_gate) == 2 and len(param_lst) != 4 * N:
        raise ValueError("num of parameters not match")

    nosg = len(single_gate)
    L1 = basic_entangle_layer(N, param_lst[:nosg * N], single_gate)
    L2 = basic_entangle_layer(N, param_lst[nosg * N:], single_gate)
    return L2 @ L1


def strong_entangle_layer(N, param_lst, single_gate='YZY'):
    if len(single_gate) == 3 and len(param_lst) != 6 * N:
        raise ValueError("num of parameters not match")
    if len(single_gate) == 2 and len(param_lst) != 4 * N:
        raise ValueError("num of parameters not match")

    if single_gate == 'YZY':
        S1 = YZYLayer(N, param_lst[0:3 * N])
        S2 = YZYLayer(N, param_lst[3 * N:6 * N])
    elif single_gate == 'XZX':
        S1 = XZXLayer(N, param_lst[0:3 * N])
        S2 = XZXLayer(N, param_lst[3 * N:6 * N])
    elif single_gate == 'XYZ':
        S1 = XYZLayer(N, param_lst[0:3 * N])
        S2 = XYZLayer(N, param_lst[3 * N:6 * N])
    elif single_gate == 'XZ':
        S1 = XZLayer(N, param_lst[0:2 * N])
        S2 = XZLayer(N, param_lst[2 * N:4 * N])
    else:
        raise ValueError("single_gate='YZY' or 'XZX' or 'XYZ'.")

    C1 = ring_of_cnot(N)
    C2 = ring_of_cnot2(N)
    return C2 @ S2 @ C1 @ S1


def RxxLayer(N, param_lst):
    if N - 1 != len(param_lst):
        raise ValueError("number of parameters not match")
    if N < 2:
        raise ValueError("number of qubits must >= 2")
    if N == 2:
        return rxx(param_lst[0])

    if N % 2 == 0:  # 偶数个qubit
        U1 = multi_kron([rxx(p) for p in param_lst[0:int(N / 2)]])
        U2 = multi_kron([torch.eye(2, 2)] + [rxx(p) for p in param_lst[int(N / 2):N - 1]] + [torch.eye(2, 2)])
        rst = U2 @ U1

    elif N % 2 == 1:  # 奇数个qubit
        U1 = multi_kron([rxx(p) for p in param_lst[0:int((N - 1) / 2)]] + [torch.eye(2, 2)])
        U2 = multi_kron([torch.eye(2, 2)] + [rxx(p) for p in param_lst[int((N - 1) / 2):N - 1]])
        rst = U2 @ U1
    return rst


def RyyLayer(N, param_lst):
    if N - 1 != len(param_lst):
        raise ValueError("number of parameters not match")
    if N < 2:
        raise ValueError("number of qubits must >= 2")
    if N == 2:
        return ryy(param_lst[0])

    if N % 2 == 0:  # 偶数个qubit
        U1 = multi_kron([ryy(p) for p in param_lst[0:int(N / 2)]])
        U2 = multi_kron([torch.eye(2, 2)] + [ryy(p) for p in param_lst[int(N / 2):N - 1]] + [torch.eye(2, 2)])
        rst = U2 @ U1

    elif N % 2 == 1:  # 奇数个qubit
        U1 = multi_kron([ryy(p) for p in param_lst[0:int((N - 1) / 2)]] + [torch.eye(2, 2)])
        U2 = multi_kron([torch.eye(2, 2)] + [ryy(p) for p in param_lst[int((N - 1) / 2):N - 1]])
        rst = U2 @ U1
    return rst


def RzzLayer(N, param_lst):
    if N - 1 != len(param_lst):
        raise ValueError("number of parameters not match")
    if N < 2:
        raise ValueError("number of qubits must >= 2")
    if N == 2:
        return rzz(param_lst[0])

    if N % 2 == 0:  # 偶数个qubit
        U1 = multi_kron([rzz(p) for p in param_lst[0:int(N / 2)]])
        U2 = multi_kron([torch.eye(2, 2)] + [rzz(p) for p in param_lst[int(N / 2):N - 1]] + [torch.eye(2, 2)])
        rst = U2 @ U1

    elif N % 2 == 1:  # 奇数个qubit
        U1 = multi_kron([rzz(p) for p in param_lst[0:int((N - 1) / 2)]] + [torch.eye(2, 2)])
        U2 = multi_kron([torch.eye(2, 2)] + [rzz(p) for p in param_lst[int((N - 1) / 2):N - 1]])
        rst = U2 @ U1
    return rst




# def simplified2design(N, param_lst):
#     C0 = torch.eye( 2**N, 2**N ) + 0j
#     s1_lst = []
#     s2_lst = []
#     for i in range(0,N-1,2):
#         C1 = cz( N, i, i+1 ) @ C0

#     for i in range(1,N-1,2):
#         C2 = cz( N, i, i+1 ) @ C0
#     return C2 @ C1


if __name__ == "__main__":
    b = all2all_cnot(3)
    print(b)
    print(IsUnitary(b))
    a = XYZLayer(3, [1, 1, 1, 2, 2, 2, 3, 3, 3])
    print(a)
    print(IsUnitary(a))
