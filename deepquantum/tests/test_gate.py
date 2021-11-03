import torch
from deepquantum import qgate, qmath
from deepquantum.gates import Circuit as cir

a = torch.tensor(3.1416)
b = 0
# print(rz(a),'\n',rz(b))
# print(multi_control_cnot(3,[0,1],2))

N = 5
rho = torch.rand(2 ** N, 2 ** N)
tra = torch.trace(rho)
rho1 = rho * (1 / tra)
print("rho1", rho1)

p_rho = qmath.ptrace(rho1, N, [0, 4, 2])
print(p_rho)