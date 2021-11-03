import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


def dag(x):
    """
    compute conjugate transpose of input matrix
    """
    x_conj = torch.conj(x)
    x_dag = x_conj.permute(1, 0)
    return x_dag