import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from deepquantum.gates.qmath import dag
from deepquantum.gates import Circuit



class QuanConv2D(nn.Module):

    def __init__(self, n_qubits, stride, kernel_size, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super(QuanConv2D, self).__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(n_qubits*3), a=0.0, b=2 * np.pi) * init_std)

        if n_qubits != kernel_size**2:
            raise ValueError("number of qubits must == kernel_size")
        self.n_qubits = n_qubits
        self.stride = stride
        self.kernel_size = kernel_size

    def _input(self, data):
        cir1 = Circuit(self.n_qubits)
        # print("data,", data)
        for which_q in range(0, self.n_qubits, 1):
            cir1.ry(target_qubit=which_q, phi=np.pi * data[which_q])
        out = cir1.U()
        return out

    def _qconv(self, rho):
        cir2 = Circuit(self.n_qubits)
        w = self.weight * self.w_mul

        for which_q in range(0, self.n_qubits, 1):
            cir2.rx(target_qubit=which_q, phi=w[3*which_q+0])
            cir2.rz(target_qubit=which_q, phi=w[3*which_q+1])
            cir2.rx(target_qubit=which_q, phi=w[3*which_q+2])

        for which_q in range(0, self.n_qubits, 1):
            cir2.cnot(which_q, (which_q+1) % self.n_qubits)
        U = cir2.U()
        qconv_out = U @ rho @ dag(U)

        return qconv_out

    def forward(self, input):
        """Convolves the input image with many applications of the same quantum circuit."""
        """kernel_size数为量子比特数"""
        cir3 = Circuit(self.n_qubits)
        # print("input::", input.shape)
        batch, len_x_in, len_y_in, _ = input.shape
        len_x_out = (len_x_in - self.kernel_size) // self.stride + 1
        len_y_out = (len_y_in - self.kernel_size) // self.stride + 1
        out_channel = self.kernel_size**2

        out = torch.zeros((batch, len_x_out, len_y_out, out_channel))
        for batch_i in range(batch):
            for j in range(0, len_x_in, self.stride):
                for k in range(0, len_y_in, self.stride):

                    x = input[batch_i, j:j+self.kernel_size, k:k+self.kernel_size, 0].reshape(-1)
                    # print("x::", x, j, j+self.kernel_size, k, k+self.kernel_size)
                    init_matrix = torch.zeros(2**self.n_qubits, 2**self.n_qubits) + 0j
                    init_matrix[0, 0] = 1 + 0j
                    U1 = self._input(x)
                    rho_out1 = U1 @ init_matrix @ dag(U1)
                    rho_out2 = self._qconv(rho_out1)
                    classical_value = cir3.measure(rho_out2, self.n_qubits)

                    for c in range(out_channel):
                        out[batch_i, j // self.stride, k // self.stride, c] = classical_value[c]
        return out



#######################################################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.quanv = QuanConv2D(n_qubits=4, stride=2, kernel_size=2)
        self.fc1 = nn.Linear(14*14*4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.quanv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



