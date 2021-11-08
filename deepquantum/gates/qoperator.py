# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:43:13 2021

@author: shish
"""
import torch
from deepquantum.gates import multi_kron



class Operator(object):
    
    @staticmethod #为了让该函数既可以实例化调用，也可以不实例化直接Operator.gate_expand_1toN()调用
    def gate_expand_1toN(gate,N,index):
        '''
        不要直接用这个函数
        '''
        if N < 1:
            raise ValueError("number of qubits N must be >= 1")
        if index < 0 or index > N - 1:
            raise ValueError("index must between 0~N-1")
        lst1 = [torch.eye(2,2)]*N
        lst1[index] = gate
        return multi_kron(lst1) + 0j
    
    pass

class Operation(Operator):
    
    @staticmethod
    def two_qubit_control_gate(U,N,control,target):
        '''
        不建议直接使用该函数
        two_qubit_control_gate该函数可实现任意两比特受控门
        代码照抄田泽卉的，但建议用我这个函数名，注意这里的U是controlled-U里的U，而非controlled-U整体
        比如想实现cnot门，cnot表示controlled-not gate，那么U就是not门，即sigma_x(paulix)
        比如想实现cz门，cnot表示controlled-z gate，那么U就是z门，即sigma_z(pauliz)
        '''
        if N < 1:
            raise ValueError("number of qubits(interger N) must be >= 1")
        if max(control,target) > N-1:
            raise ValueError("control&target must <= number of qubits - 1")
        if min(control,target) < 0:
            raise ValueError("control&target must >= 0")
        if control == target:
            raise ValueError("control cannot be equal to target")
        
        zero_zero = torch.tensor( [[1,0],[0,0]] ) + 0j
        one_one = torch.tensor( [[0,0],[0,1]] ) + 0j
        
        lst1 = [torch.eye(2,2)] * N
        lst1[control] = zero_zero
        
        lst2 = [torch.eye(2,2)] * N
        lst2[control] = one_one
        lst2[target] = U
        return multi_kron(lst1) + multi_kron(lst2) + 0j
    
    
    
    @staticmethod
    def multi_control_gate(U,N,control_lst,target):
        '''
        多控制比特受控门，比如典型的toffoli gate就是2个控制1个受控
        control_lst:一个列表，内部是控制比特的索引号
        '''
        if N < 1:
            raise ValueError("number of qubits(interger N) must be >= 1")
            
        if max(max(control_lst),target) > N-1:
            raise ValueError("control&target must <= number of qubits - 1")
            
        if min(min(control_lst),target) < 0:
            raise ValueError("control&target must >= 0")
            
        for each in control_lst:
            if each == target:
                raise ValueError("control cannot be equal to target")
        
        U = U + 0j
        one_one = torch.tensor( [[0,0],[0,1]] ) + 0j
        
        lst1 = [torch.eye(2,2)] * N
        for each in control_lst:
            lst1[each] = one_one
        lst1[target] = U
        
        lst2 = [torch.eye(2,2)] * N
        
        lst3 = [torch.eye(2,2)] * N
        for each in control_lst:
            lst3[each] = one_one
        #multi_kron(lst2) - multi_kron(lst3)对应不做操作的哪些情况
        return multi_kron(lst2) - multi_kron(lst3) + multi_kron(lst1) + 0j
    
    
    
    
    
    def IsUnitary(matrix):
        #判断一个矩阵是否幺正
        pass
    
    pass

class Observable(Operator):
    
    def IsHermitian(matrix):
        #判断一个矩阵是否厄米
        pass
    
    pass

class DiagonalOperation(Operation):
    pass


#==============================无参数单比特门==================================

class Hadamard(Observable, Operation):
    #没有可调参数
    #只作用在1个qubit上
    #自己是自己的逆操作
    #以下属于类的属性，而非实例的属性
    label = "Hadamard"
    num_params = 0
    num_wires = 1               
    self_inverse = True
    matrix = torch.sqrt( torch.tensor(0.5) ) * torch.tensor([[1,1],[1,-1]]) + 0j
    
    def __init__(self,N=None,wires=None):
        self.nqubits = N
        self.wires = wires
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            return self.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("Hadamard gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':None}
        return info
    
    def params_update(self,params_lst):
        pass
        
    

class PauliX(Observable, Operation):
    label = "PauliX"
    num_params = 0
    num_wires = 1               
    self_inverse = True
    matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,N=None,wires=None):
        self.nqubits = N
        self.wires = wires
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            return self.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("PauliX gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':None}
        return info
    
    def params_update(self,params_lst):
        pass
    

class PauliY(Observable, Operation):
    label = "PauliY"
    num_params = 0
    num_wires = 1               
    self_inverse = True
    matrix = torch.tensor([[0,-1j],[1j,0]]) + 0j
    
    def __init__(self,N=None,wires=None):
        self.nqubits = N
        self.wires = wires
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            return self.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("PauliY gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':None}
        return info
    
    def params_update(self,params_lst):
        pass


class PauliZ(Observable, DiagonalOperation):
    label = "PauliZ"
    num_params = 0
    num_wires = 1               
    self_inverse = True
    matrix = torch.tensor([[1,0],[0,-1]]) + 0j
    
    def __init__(self,N=None,wires=None):
        self.nqubits = N
        self.wires = wires
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            return self.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("PauliZ gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':None}
        return info
    
    def params_update(self,params_lst):
        pass

#==============================带参数单比特门==================================

class rx(Operation):
    label = "Rx"
    num_params = 1
    num_wires = 1            
    self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=None,wires=None):
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(theta/2.0)*PauliX.matrix + 0j
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            return self.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("Rx gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(self.params/2.0)*PauliX.matrix + 0j







class ry(Operation):
    label = "Ry"
    num_params = 1
    num_wires = 1            
    self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=None,wires=None):
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(theta/2.0)*PauliY.matrix + 0j
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            return self.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("Ry gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(self.params/2.0)*PauliY.matrix + 0j







class rz(Operation):
    label = "Rz"
    num_params = 1
    num_wires = 1
    self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=None,wires=None):
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(theta/2.0)*PauliZ.matrix + 0j
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            return self.gate_expand_1toN(self.matrix, self.nqubits, self.wires)
        else:
            raise ValueError("Rz gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[], 'target_lst':[self.wires],'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(2,2) \
            - 1j*torch.sin(self.params/2.0)*PauliZ.matrix + 0j


#==============================带参数两比特门==================================




class rxx(Operation):
    label = "Rxx"
    num_params = 1
    num_wires = 2           
    self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=None,wires=None):#wires以list形式输入
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(theta/2.0)*torch.kron(PauliX.matrix,PauliX.matrix) + 0j
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            if self.nqubits < 1:
                raise ValueError("number of qubits N must be >= 1")
            if self.wires[0] < 0 or self.wires[0] > self.nqubits - 1 or self.wires[1] < 0  or self.wires[0] > self.nqubits - 1:
                raise ValueError("qbit index must between 0~N-1")
            if self.wires[0] == self.wires[1]:
                raise ValueError("qbit1 cannot be equal to qbit2")
            lst1 = [torch.eye(2,2)]*self.nqubits
            lst2 = [torch.eye(2,2)]*self.nqubits
            lst2[self.wires[0]] =  PauliX.matrix
            lst2[self.wires[1]] =  PauliX.matrix
            rst = torch.cos(self.params/2.0)*multi_kron(lst1) - 1j*torch.sin(self.params/2.0)*multi_kron(lst2)
            return rst + 0j
        else:
            raise ValueError("Rxx gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[], 'target_lst':list(self.wires),'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(self.params/2.0)*torch.kron(PauliX.matrix,PauliX.matrix) + 0j










class ryy(Operation):
    label = "Ryy"
    num_params = 1
    num_wires = 2           
    self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=None,wires=None):#wires以list形式输入
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(theta/2.0)*torch.kron(PauliY.matrix,PauliY.matrix) + 0j
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            if self.nqubits < 1:
                raise ValueError("number of qubits N must be >= 1")
            if self.wires[0] < 0 or self.wires[0] > self.nqubits - 1 or self.wires[1] < 0  or self.wires[0] > self.nqubits - 1:
                raise ValueError("qbit index must between 0~N-1")
            if self.wires[0] == self.wires[1]:
                raise ValueError("qbit1 cannot be equal to qbit2")
            lst1 = [torch.eye(2,2)]*self.nqubits
            lst2 = [torch.eye(2,2)]*self.nqubits
            lst2[self.wires[0]] =  PauliY.matrix
            lst2[self.wires[1]] =  PauliY.matrix
            rst = torch.cos(self.params/2.0)*multi_kron(lst1) - 1j*torch.sin(self.params/2.0)*multi_kron(lst2)
            return rst + 0j
        else:
            raise ValueError("Ryy gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[], 'target_lst':list(self.wires),'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(self.params/2.0)*torch.kron(PauliY.matrix,PauliY.matrix) + 0j










class rzz(Operation):
    label = "Rzz"
    num_params = 1
    num_wires = 2           
    self_inverse = False
    #matrix = torch.tensor([[0,1],[1,0]]) + 0j
    
    def __init__(self,theta,N=None,wires=None):#wires以list形式输入
        self.nqubits = N
        self.wires = wires
        self.params = theta
        self.matrix = torch.cos(theta/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(theta/2.0)*torch.kron(PauliZ.matrix,PauliZ.matrix) + 0j
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            if self.nqubits < 1:
                raise ValueError("number of qubits N must be >= 1")
            if self.wires[0] < 0 or self.wires[0] > self.nqubits - 1 or self.wires[1] < 0  or self.wires[0] > self.nqubits - 1:
                raise ValueError("qbit index must between 0~N-1")
            if self.wires[0] == self.wires[1]:
                raise ValueError("qbit1 cannot be equal to qbit2")
            lst1 = [torch.eye(2,2)]*self.nqubits
            lst2 = [torch.eye(2,2)]*self.nqubits
            lst2[self.wires[0]] =  PauliZ.matrix
            lst2[self.wires[1]] =  PauliZ.matrix
            rst = torch.cos(self.params/2.0)*multi_kron(lst1) - 1j*torch.sin(self.params/2.0)*multi_kron(lst2)
            return rst + 0j
        else:
            raise ValueError("Rzz gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[], 'target_lst':list(self.wires),'params':self.params}
        return info
    
    def params_update(self,params):
        self.params = params
        self.matrix = torch.cos(self.params/2.0)*torch.eye(4,4) \
            - 1j*torch.sin(self.params/2.0)*torch.kron(PauliZ.matrix,PauliZ.matrix) + 0j



#==============================无参数两比特门==================================



class cnot(Operation):
    label = "cnot"
    num_params = 0
    num_wires = 2          
    self_inverse = True
    matrix = torch.tensor([[1,0,0,0],\
                           [0,1,0,0],\
                           [0,0,0,1],\
                           [0,0,1,0]]) + 0j
    
    def __init__(self,N=None,wires=None):#wires以list形式输入
        self.nqubits = N
        self.wires = wires
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            sigma_x = torch.tensor( [[0,1],[1,0]] ) + 0j
            control = self.wires[0]
            target = self.wires[1]
            return self.two_qubit_control_gate( sigma_x, self.nqubits, control, target )
        else:
            raise ValueError("cnot gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        return info
    
    def params_update(self,params):
        pass
    





class cz(Operation):
    label = "cz"
    num_params = 0
    num_wires = 2          
    self_inverse = True
    matrix = torch.tensor([[1,0,0,0],\
                           [0,1,0,0],\
                           [0,0,1,0],\
                           [0,0,0,-1]]) + 0j
    
    def __init__(self,N=None,wires=None):#wires以list形式输入
        self.nqubits = N
        self.wires = wires
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            sigma_z = torch.tensor( [[1,0],[0,-1]] ) + 0j
            control = self.wires[0]
            target = self.wires[1]
            return self.two_qubit_control_gate( sigma_z, self.nqubits, control, target )
        else:
            raise ValueError("cz gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':[self.wires[0]], 'target_lst':[self.wires[1]],'params':None}
        return info
    
    def params_update(self,params):
        pass
    



#==============================无参数多比特门==================================

class toffoli(Operation):
    label = "toffoli"
    num_params = 0
    num_wires = 3       
    self_inverse = True
    matrix = torch.tensor([[1,0,0,0,0,0,0,0],\
                           [0,1,0,0,0,0,0,0],\
                           [0,0,1,0,0,0,0,0],\
                           [0,0,0,1,0,0,0,0],\
                           [0,0,0,0,1,0,0,0],\
                           [0,0,0,0,0,1,0,0],\
                           [0,0,0,1,0,0,0,1],\
                           [0,0,0,1,0,0,1,0]]) + 0j
    
    def __init__(self,N=None,wires=None):#wires以list形式输入
        self.nqubits = N
        if len(wires) != 3:
            raise ValueError("toffoli gate must be applied on 3 qubits")
        self.wires = wires
        self.control_lst = [ wires[0], wires[1] ]
        self.target_lst = [ wires[2] ]
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            sigma_x = torch.tensor( [[0,1],[1,0]] ) + 0j
            
            return self.multi_control_gate( sigma_x, self.nqubits, self.control_lst, self.target_lst[0] )
        else:
            raise ValueError("toffoli gate input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':self.control_lst, 'target_lst':self.target_lst,'params':None}
        return info
    
    def params_update(self,params):
        pass







class multi_control_cnot(Operation):
    #label = "multi_control_cnot"
    num_params = 0
    #num_wires = 3       
    self_inverse = True
    #matrix = None
    
    def __init__(self,N=None,wires=None):#wires以list形式输入
        
        self.label = str(len(wires)-1)+"_control_cnot"
        self.num_wires = len(wires)
        
        self.nqubits = N
        self.wires = wires
        self.control_lst =  list(wires[0:len(wires)-1]) 
        self.target_lst = [ wires[-1] ]
        #self.U = self.U_expand()
    
    def U_expand(self):
        if self.nqubits != None and self.wires != None:
            sigma_x = torch.tensor( [[0,1],[1,0]] ) + 0j
            return self.multi_control_gate( sigma_x, self.nqubits, self.control_lst, self.target_lst[0] )
        else:
            raise ValueError(self.label+" input error!")
    
    def info(self):
        #将门的信息整合后return，用来添加到circuit的gate_lst中
        info = {'gate':self.label, 'contral_lst':self.control_lst, 'target_lst':self.target_lst,'params':None}
        return info
    
    def params_update(self,params):
        pass










if __name__ == "__main__":
    # print('start')
    # h0 = rx(torch.tensor(0.5),3,1)
    # print(h0.matrix)
    # #h = Hadamard(1,0)
    # print(h0.U_expand())
    # #print(h.label,' ',h.self_inverse)
    #c1 = cnot(5,[3,0])
    c1 = multi_control_cnot(6,[0,1,2,3])
    print(c1.matrix)
    print(c1.U_expand())
    print(c1.info())
    input("")
    