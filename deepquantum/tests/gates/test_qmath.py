import torch
import numpy as np
from deepquantum.gates.qmath import multi_kron, dag, IsUnitary, IsNormalized,IsHermitian,ptrace,partial_trace,measure

# 测试文件以test_开头（以_test结尾也可以）
# 测试类以Test开头，并且不能带有 init 方法
# 测试函数以test_开头
# 断言使用基本的assert即可

# content of test_sample.py
# def func(x):
#     return x + 1
#
# def test_answer():
#
#     assert func(3) == 5


class TestCorrectness:
    def test_multi_kron(self):
        a = torch.randn(2, 2) + 0j
        b = torch.randn(2, 2) + 0j
        c = torch.randn(2, 2) + 0j
        d = multi_kron([a, b, c])
        print(d.shape)
        # assert d.type == torch.tensor
        assert list(d.shape) == [2**3, 2**3]
        pass

    def test_dag(self):
        a = torch.randn(2, 2) + 0j
        b = torch.randn(2, 2) + 0j
        print(dag(a))
        print(dag(b))
        pass

    def test_IsUnitary(self):
        a = torch.randn(2, 2) + 0j
        b = torch.randn(2, 2) + 0j
        # print(IsUnitary(a))
        # print(IsUnitary(b))
        pass

    def test_IsNormalized(self):
        a = torch.normal(mean=0.1, std=0, size=(1, 10))
        # print(IsNormalized(a))
        pass
    def test_IsHermitian(self):
        pass
    def test_ptrace(self):
        pass
    def test_partial_trace(self):
        pass
    def test_measure(self):
        rho = torch.tensor([[0.5, 0],
                            [0, 0.5]])+0j

        M= torch.tensor([[1, 0],
                            [0, -1]])+0j
        print(measure(state=rho, M=M, rho=False))

        pass

if __name__ == '__main__':
    a = torch.randn(4, 4) + 0j
    print(a)