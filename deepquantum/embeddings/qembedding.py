import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


# ===============================encoding layer=================================

def PauliEncoding(N, input_lst, pauli='X'):
    if N < len(input_lst):
        raise ValueError("number of inputs must be less than number of qubits")
    num = len(input_lst)
    if pauli == 'X':
        E = multi_kron([rx(input_lst[i % num]) for i in range(N)])
    elif pauli == 'Y':
        E = multi_kron([ry(input_lst[i % num]) for i in range(N)])
    elif pauli == 'Z':
        E = multi_kron([rz(input_lst[i % num]) for i in range(N)])
    else:
        raise ValueError("pauli parameter must be one of X Y Z")
    return E


def AmplitudeEncoding(N, input_lst):
    if 2 ** N < len(input_lst):
        raise ValueError("number of inputs must be less than dimension 2^N")

    num = len(input_lst)

    norm = 0.0
    for each in torch.abs(input_lst):
        norm += each ** 2

    input_lst = (1.0 / torch.sqrt(norm)) * input_lst
    state = torch.zeros([2 ** N]) + 0j
    for i in range(num):
        state[i] = input_lst[i]
    return state
