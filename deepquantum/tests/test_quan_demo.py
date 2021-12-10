"""
    更新门后版本
"""

#导入库文件
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from door_lxy import Circuit
# from calculate import dag, measure, IsUnitary
from deepquantum import qgate, qmath
from deepquantum.gates.qmath import dag
from deepquantum.gates import Circuit
from deepquantum.nn.modules.quanv import QuanConv2D

BATCH_SIZE = 4
EPOCHS = 1     # Number of optimization epochs
n_train = 5    # Size of the train dataset
n_test = 1   # Size of the test dataset

SAVE_PATH = "./"  # Data saving folder
PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
seed = 42
np.random.seed(seed)        # Seed for NumPy random number generator
torch.manual_seed(seed)     # Seed for TensorFlow random number generator


DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root="./data",
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

train_dataset.data = train_dataset.data[:n_train]
train_dataset.targets = train_dataset.targets[:n_train]

test_dataset = datasets.MNIST(root="./data",
                              train=False,
                              transform=transforms.ToTensor())

test_dataset.data = test_dataset.data[:n_test]
test_dataset.targets = test_dataset.targets[:n_test]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.quan1 = QuanConv2D(n_qubits=4, stride=2, kernel_size=2)
        self.fc1 = nn.Linear(14*14*4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.quan1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

loss_list = []

model.train().to(DEVICE)
for epoch in range(EPOCHS):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.to(DEVICE)
        optimizer.zero_grad()
        data = data.to(torch.float32).to(DEVICE)

        # Forward pass
        output = model(data).to(DEVICE)

        # Calculating loss
        loss = loss_func(output, target).to(DEVICE)

        # Backward pass
        loss.backward()

        # Optimize the weights
        optimizer.step()

        total_loss.append(loss.item())
    loss_list.append(sum(total_loss) / len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / EPOCHS, loss_list[-1]))

model.eval()
with torch.no_grad():
    correct = 0
    total_loss = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(torch.float32).to(DEVICE)
        target = target.to(DEVICE)
        output = model(data).to(DEVICE)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_func(output, target)
        total_loss.append(loss.item())
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100 / BATCH_SIZE)
        )