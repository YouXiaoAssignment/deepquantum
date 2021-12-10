import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from deepquantum.gates import Circuit as cir
from deepquantum.gates.qmath import dag

def PrepareState(state):
    return torch.tensor([i+0j for i in state])


def PrepareRho(state):
    basic_rho = PrepareState(state).unsqueeze(0)
    return basic_rho.T @ basic_rho  # out-produt




if __name__ == '__main__':


    print(PrepareState([0,1]))
    print(PrepareRho([0,1]))

