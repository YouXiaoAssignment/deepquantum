from deepquantum import qlayer, qgate

b = qlayer.all2all_cnot(3)
print(b)
print(qgate.IsUnitary(b))
a = qlayer.XYZLayer(3, [1, 1, 1, 2, 2, 2, 3, 3, 3])
print(a)
print(qgate.IsUnitary(a))