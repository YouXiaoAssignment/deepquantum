# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:53:04 2021

@author: shish
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time


from deepquantum.gates.qmath import multi_kron, measure, IsUnitary, IsNormalized
import deepquantum.gates.qoperator as op
from deepquantum.gates.qcircuit import Circuit
from deepquantum.embeddings.qembedding import PauliEncoding
from deepquantum.layers.qlayers import YZYLayer, ZXLayer,ring_of_cnot, ring_of_cnot2, BasicEntangleLayer






#==============================================================================
class qcir(nn.Module):
    def __init__(self, nqubits):
        super().__init__()
        #属性：量子线路qubit数目，随机初始化的线路参数，测量力学量列表
        self.nqubits = nqubits
        self.weight = \
            nn.Parameter( nn.init.uniform_(torch.empty(21*self.nqubits), a=0.0, b=2*torch.pi) )
        
        self.M_lst = self.Zmeasure()

        
    def Zmeasure(self):
        #生成测量力学量的列表
        M_lst = []
        for i in range(self.nqubits):
            Mi = op.PauliZ(self.nqubits, i).U_expand()
            M_lst.append( Mi )
        
        return M_lst
            
    
    def build_circuit(self,input_lst_batch):
        '''
        构建变分量子线路的演化矩阵
        '''
        wires_lst = [i for i in range(self.nqubits)]
        #创建线路
        c1 = Circuit(self.nqubits) 
        
        #encoding编码部分
        batch_size = len(input_lst_batch)
        phi_encoded_batch = torch.zeros( batch_size, 2**self.nqubits) + 0j
        for i, inputs in enumerate(input_lst_batch):
            e1 = PauliEncoding(self.nqubits, inputs, wires_lst,pauli='Y')
            e2 = PauliEncoding(self.nqubits, inputs, wires_lst,pauli='Z')
            E1 = e1.U_expand() #编码矩阵
            E2 = e2.U_expand()
            phi_encoded_batch[i] = E2 @ E1 @ c1.state_init() #矩阵与列向量相乘
        
        #variation变分部分
        repeat = 6
        c1.add( BasicEntangleLayer(self.nqubits, 
                                   wires_lst, 
                                   self.weight[0:3*repeat*self.nqubits], 
                                   repeat=repeat) )
        
        c1.add( YZYLayer(self.nqubits, 
                         wires_lst, 
                         self.weight[3*repeat*self.nqubits:3*(repeat+1)*self.nqubits]) )
        
        U = c1.U()

        #最终返回线路变分部分的 演化酉矩阵 和 编码后的态矢
        return U + 0j, phi_encoded_batch.permute(1,0) 
    
    
    def forward(self,input_lst_batch):
        #计算编码后的态和变分线路的演化矩阵
        U, phi_encoded_batch = self.build_circuit(input_lst_batch)
        #计算线路演化终态
        phi_out = U @ phi_encoded_batch # 8 X batch_size
        phi_out = phi_out.permute(1,0)
        #模拟测量得到各个测量力学量的期望值
        # measure_rst = []
        # for Mi in self.M_lst: 
        #     measure_rst.append( measure(phi_out, Mi) )
        
        # #以3个qubit的线路为例，把3个[batch,1]的矩阵拼接为[batch,3]
        # rst = torch.cat( tuple(measure_rst),dim=1 ) 
        
        # #把值域做拉伸，从[-1,1]到[-10,10]
        # rst = ( rst + 0 ) * 10
        # return rst
        measure_rst = []
        for Mi in self.M_lst:
            measure_rst.append((phi_out.conj().unsqueeze(-2) @ Mi @ phi_out.unsqueeze(-1) ).squeeze(dim=2) )
        #measure_rst =  measure(self.nqubits, phi_out) 
        
        #以3个qubit的线路为例，把3个[batch,1]的矩阵拼接为[batch,3]
        rst = torch.cat( tuple(measure_rst),dim=1 ) 
        
        #把值域做拉伸，从[-1,1]变为[-4,4]
        rst = ( rst + 0 ) * 4
        return rst.real
        
        
        # rst_average = rst[:,0]
        # for i in range(1,rst.shape[1]):
        #     rst_average += rst[:,i]
        
        # rst_average = rst_average * (1.0/rst.shape[1])
            
        # return rst_average.view(-1,1)
        
        
        

class qnet(nn.Module):
    
    def __init__(self,nqubits):
        super().__init__()
        
        self.nqubits = nqubits
        self.circuit = qcir(self.nqubits)
        self.FC1 = nn.Linear(len(self.circuit.M_lst),8)
        self.FC2 = nn.Linear(8,8)
        self.FC3 = nn.Linear(8,1)
      
   
    
    def forward(self,x_batch):
        
        #输入数据的非线性预处理
        #pre_batch = torch.sqrt( 0.5*(1 + torch.sigmoid(x_batch)) )
        #pre_batch = torch.arctan( x_batch )
        pre_batch = torch.arcsin( x_batch )
        #pre_batch = torch.arcsin( x_batch*x_batch )
        #pre_batch = x_batch
        
        cir_out = self.circuit ( pre_batch )
        
        return torch.sigmoid( cir_out[:,0] )
        # out = nn.functional.leaky_relu(self.FC1(cir_out))
        # out = nn.functional.leaky_relu(self.FC2(out))
        # out = nn.functional.leaky_relu(self.FC3(out))
        # return out





def foo(x1,x2):
    if x1**2 + x2**2 > 0.5:
        y = 1
    else:
        y = 0
    return y




if __name__ == "__main__":
    
    N = 4    #量子线路的qubit总数
    num_examples = 2048
    num_inputs = 2
    num_outputs = 1
    
    features = torch.empty( num_examples,num_inputs )
    labels = torch.empty( num_examples,num_outputs )
    for i in range(num_examples):
        features[i] = torch.rand(num_inputs)*2 - 1 #输入值范围是[-1,1]

    for i in range(num_examples):
        labels[i] = foo( features[i][0], features[i][1] )
    
    def data_iter(batch_size, features, labels):
        #输入batch_size，输入训练集地数据features+标签labels
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices) #把indices列表顺序随机打乱
        for i in range(0,num_examples,batch_size):
            #每次取batch_size个训练样本,j代表索引
            j = torch.LongTensor( indices[i:min(i+batch_size,num_examples)] ) 
            yield features.index_select(0,j), labels.index_select(0,j)
            #把张量沿着0维，只保留取出索引号对应的元素
    
#=============================================================================
    
    
    net1 = qnet(N)      #构建训练模型
    loss = nn.MSELoss() #平方损失函数
    
    #定义优化器，也就是选择优化器，选择Adam梯度下降，还是随机梯度下降，或者其他什么
    #optimizer = optim.SGD(net1.parameters(), lr=0.1) #lr为学习率
    optimizer = optim.Adam(net1.parameters(), lr=0.01) #lr为学习率
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,10,15], gamma=0.8)
    
    
    
    
    #torch.cuda.set_device(0)
    num_epochs = 10;
    batch_size = 1024;
    
    #记录loss随着epoch的变化，用于后续绘图
    epoch_lst = [i+1 for i in range(num_epochs)]
    loss_lst = []
    
    for epoch in range(1,num_epochs+1):
        t1 = time.time()
        for x,y in data_iter(batch_size,features,labels):
            
            x.requires_grad_(True)
            output = net1(x);
            #squeeze是为了把y维度从1x3变成3
            l = loss(output.squeeze(),y.squeeze());
            #梯度清0
            optimizer.zero_grad()
            l.backward()
            #print("weights_grad2:",net1.circuit.weight.grad,'  weight is leaf?:',net1.circuit.weight.is_leaf)
            optimizer.step()
            
        #lr_scheduler.step()
        t2 = time.time()
        loss_lst.append(l.item())
        print("epoch:%d, loss:%f" % (epoch,l.item()),\
              ';current lr:', optimizer.state_dict()["param_groups"][0]["lr"],'  耗时：',t2-t1)
    
    
    
    plt.cla()
    
    x1_lst = list(features[:num_examples,0])
    x2_lst = list(features[:num_examples,1])
    
    plt.subplot(131)
    y_t = [foo(x1_lst[i],x2_lst[i]) for i in range(len(x1_lst))]
    color_lst = ['b' if label==1 else 'r' for label in y_t]
    plt.scatter(x1_lst, x2_lst, s=1, c=color_lst)
    
    plt.subplot(132)
    y_prediction = [float(each) for each in net1( features[:num_examples,:] ).squeeze() ]
    color_pred_lst = ['m' if label>0.5 else 'black' for label in y_prediction]
    plt.scatter(x1_lst, x2_lst, s=1, c=color_pred_lst)
    
    plt.subplot(133)
    plt.plot(epoch_lst,loss_lst,'r^--',linewidth=1, markersize=1.5)
    plt.show()
    
    input("")
    
































    


 














































































































        