from deepquantum import qlayer, qgate, qmath
from deepquantum.gates import Circuit as cir

b = qlayer.all2all_cnot(3)
print(b)
print(qmath.IsUnitary(b))
a = qlayer.XYZLayer(3, [1, 1, 1, 2, 2, 2, 3, 3, 3])
print(a)
print(qmath.IsUnitary(a))