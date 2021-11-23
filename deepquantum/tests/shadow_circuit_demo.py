# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:02:05 2021

@author: shish
"""
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

import random
import time

from deepquantum import Circuit
import deepquantum as dp

class qcircuit(nn.Module):
    
    def __init__(self, N, n_sqc):
        super().__init__()
        self.nqubits = N
        self.n_sqc = n_sqc
        self.weight = \
            nn.Parameter( nn.init.uniform_(torch.empty(9*self.n_sqc), a=0.0, b=2*torch.pi) )
            
        self.M = dp.multi_kron([dp.PauliX.matrix]*self.n_sqc)
        pass
    
    def normlization(self,vector_batch):
        num_row = vector_batch.shape[0]
        num_col = vector_batch.shape[1]
        rst_vector_batch = torch.empty(num_row, num_col)
        for i in range(num_row):
            summ = 0.0
            for j in range(num_col):
                summ += torch.abs( vector_batch[i,j] )**2
            for j in range(num_col):
                rst_vector_batch[i,j] = ( 1.0/torch.sqrt(summ) ) * vector_batch[i,j]
        
        return rst_vector_batch + 0j
    
    def amplitude_encoding(self, norm_vector_batch):
        '''
        N表示circuit的qubit总数，norm_vector_batch这是已经归一化的输入矢量
        '''
        N = self.nqubits
        num_row = norm_vector_batch.shape[0]
        num_col = norm_vector_batch.shape[1]
        #print(num_col.dtype)
        if num_col > 2**N:
            raise ValueError('amplitude_encoding error: vector dimension too large! qubits not enough!')
        
        zero_m = torch.zeros(num_row, 2**N-num_col)
        rst = torch.cat( (norm_vector_batch, zero_m), 1 )
        return rst + 0j
    
    def build_circuit(self):
        '''
        n_sqc表示shadow circuit的qubit数目，必须满足n_sqc < N
        n_sqc一般不超过4
        '''
        n_sqc = self.n_sqc
        
        cir = Circuit(n_sqc)
        
        #wires = list( range(start_qbit, start_qbit+n_sqc, 1) )
        wires = list( range(n_sqc) )
        
        repeat = 2
        cir.BasicEntangleLayer(wires, self.weight[0:3*n_sqc*repeat], repeat=repeat)
        
        cir.YZYLayer(wires, self.weight[3*n_sqc*repeat:3*n_sqc*(repeat+1)])
        
        U = cir.U()
        
        return U
    
    def cal_o_lst(self,state):
        rho =  state.view(-1,1) @ dp.dag(state.view(-1,1))
        o_lst = torch.zeros(self.nqubits - self.n_sqc + 1)
        for start_qbit in range(0,self.nqubits - self.n_sqc + 1):
                
            U = self.build_circuit()
            temp = dp.dag(U) @ self.M @ U
            temp_lst = [torch.eye(2)+0j]*start_qbit \
                + [temp] \
                + [torch.eye(2)+0j]*( self.nqubits - self.n_sqc - start_qbit )
            
            o = torch.trace( rho @ dp.multi_kron(temp_lst) ).real
            o_lst[start_qbit] = o
        return o_lst
        
    
    def forward(self,vector_batch):
        batch_size = vector_batch.shape[0]
        state_batch = self.amplitude_encoding( self.normlization(vector_batch) )
        
        o_batch = torch.empty( batch_size, self.nqubits - self.n_sqc + 1 )
        for idx,state in enumerate( state_batch ):
            o_lst = self.cal_o_lst(state)
        o_batch[idx] = o_lst
        
        return o_batch
        

     
                
        
        
#============================================================


class qnet(nn.Module):
    
    def __init__(self,nqubits,n_sqc):
        super().__init__()
        
        self.nqubits = nqubits
        self.n_sqc = n_sqc
        self.circuit = qcircuit( self.nqubits, self.n_sqc )
        
        num_qout = self.circuit.nqubits - self.circuit.n_sqc + 1
        self.FC1 = nn.Linear(num_qout,10)
      
   
    
    def forward(self,x_batch):
        
        #输入数据的非线性预处理
        #pre_batch = torch.sqrt( 0.5*(1 + torch.sigmoid(x_batch)) )
        #pre_batch = torch.arctan( x_batch )
        #pre_batch = torch.arcsin( x_batch )
        #pre_batch = torch.arcsin( x_batch*x_batch )
        pre_batch = x_batch
        
        cir_out = self.circuit ( pre_batch )
        
        #return torch.sigmoid( cir_out[:,0] )
        out = nn.functional.softmax(self.FC1(cir_out),dim=1) #dim=1同一行的元素做softmax
        #out = self.FC1(cir_out)  #无需在此处做softmax，交叉熵loss计算中会自动帮你计算
        #print(out)
        # out = nn.functional.leaky_relu(self.FC2(out))
        # out = nn.functional.leaky_relu(self.FC3(out))
        #mmax,idx = torch.max( out, dim=1 )
        #idx = idx.view(-1,1) + torch.tensor(0.0)
        
        return out




if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_examples = 16
    num_inputs = 784
    num_outputs = 10
    
    num_epochs = 10;
    batch_size = 16;
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307),(0.3081))])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                               train=True, 
                                               transform=transforms.ToTensor(),  
                                               download=True)
    n_train = num_examples
    train_dataset.data = train_dataset.data[:n_train]
    train_dataset.targets = train_dataset.targets[:n_train]
    # test_dataset = torchvision.datasets.MNIST(root='./data', 
    #                                           train=False, 
    #                                           transform=transforms.ToTensor())
     
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)
     
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
    #                                           batch_size=batch_size, 
    #                                           shuffle=False)
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
 
    
    net1 = qnet(10,2)      #构建训练模型
    #loss = nn.MSELoss() #平方损失函数
    loss = nn.CrossEntropyLoss() #交叉熵损失函数
    optimizer = optim.Adam(net1.parameters(), lr=0.01) #lr为学习率   
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,10,15], gamma=0.8)
    

    
    #记录loss随着epoch的变化，用于后续绘图
    epoch_lst = [i+1 for i in range(num_epochs)]
    loss_lst = []
    
    print('start')
    for epoch in range(1,num_epochs+1):
        t1 = time.time()
        #for x,y in data_iter(batch_size,features,labels):
        for i,data in enumerate(train_loader):
            #print(i)
            x,y = data
            # x = x.to(device)
            # y = y.to(device)
            '''
            y_new = torch.zeros((y.shape[0],10))
            for i,label in enumerate(y):
                y_new[i][int(label)] = 1 
            '''
            #print('y :',y)
            x = x.squeeze().view(-1,784)
            #print(x)
            # print(x.shape)
            # print(y.shape)
            x.requires_grad_(True)
            output = net1(x);
            #print('yp:',output)
            #squeeze是为了把y维度从1x3变成3
            '''
            参考：
            Pytorch中的CrossEntropyLoss()函数案例解读和结合one-hot编码计算Loss
            '''
            l = loss(output, y);
            #梯度清0
            optimizer.zero_grad()
            l.requires_grad_(True)
            l.backward()
            #print("weights_grad2:",net1.circuit.weight.grad,'  weight is leaf?:',net1.circuit.weight.is_leaf)
            optimizer.step()
            
        #lr_scheduler.step()
        
        loss_lst.append(l.item())
        t2 = time.time()
        print("epoch:%d, loss:%f" % (epoch,l.item()),\
              ';current lr:', optimizer.state_dict()["param_groups"][0]["lr"],'time:',t2-t1)
    
    input('')