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


BATCH_SIZE = 4
EPOCHS = 10     # Number of optimization epochs
n_layers = 1    # Number of  layers
n_train = 5    # Size of the train dataset
n_test = 1   # Size of the test dataset

SAVE_PATH = "./"  # Data saving folder
PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
seed = 42
np.random.seed(seed)        # Seed for NumPy random number generator
torch.manual_seed(seed)     # Seed for TensorFlow random number generator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

train_images = torch.unsqueeze(train_dataset.data, -1)
test_images = torch.unsqueeze(test_dataset.data, -1)


class QuanConv2D(nn.Module):

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(12), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits


    def input(self, data):
        cir1 = Circuit(self.n_qubits)

        for which_q in range(0, self.n_qubits, 1):
            cir1.ry(target_qubit=which_q, phi=np.pi * data[which_q])
        out = cir1.U()
        return out

    def qconv(self):
        cir2 = Circuit(self.n_qubits)
        w = self.weight * self.w_mul

        for which_q in range(0, self.n_qubits, 1):
            cir2.rx(target_qubit=which_q, phi=w[3*which_q+0])
            cir2.rz(target_qubit=which_q, phi=w[3*which_q+1])
            cir2.rx(target_qubit=which_q, phi=w[3*which_q+2])

        for which_q in range(0, self.n_qubits, 1):
            cir2.cnot(which_q, (which_q+1) % self.n_qubits)
        U = cir2.U()

        return U

    def forward(self, x):
        cir3 = Circuit(self.n_qubits)
        E_qconv = self.qconv()
        qconv_out = dag(E_qconv) @ x @ E_qconv
        classical_value = cir3.measure(qconv_out, self.n_qubits)
        return classical_value

circuit = QuanConv2D(4)

def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((14, 14, 4))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            # Process a squared 2x2 region of the image with a quantum circuit

            x = torch.FloatTensor(([image[j, k, 0],
                                    image[j, k + 1, 0],
                                    image[j + 1, k, 0],
                                    image[j + 1, k + 1, 0]]))

            init_matrix = torch.zeros(2**4, 2**4) + 0j
            init_matrix[0, 0] = 1 + 0j
            q_input = circuit.input(x)
            q_out = q_input @ init_matrix @ dag(q_input)
            q_results = circuit.forward(q_out)

            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out

if PREPROCESS == True:
    q_train_images = []
    print("Quantum pre-processing of train images:")
    for idx, img in enumerate(train_images):
        print("{}/{}        ".format(idx + 1, n_train), end="\r")
        q_train_images.append(quanv(img))
    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    print("\nQuantum pre-processing of test images:")
    for idx, img in enumerate(test_images):
        print("{}/{}        ".format(idx + 1, n_test), end="\r")
        q_test_images.append(quanv(img))
    q_test_images = np.asarray(q_test_images)

    # Save pre-processed images
    np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
    np.save(SAVE_PATH + "q_test_images.npy", q_test_images)

# Load pre-processed images
q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
q_test_images = np.load(SAVE_PATH + "q_test_images.npy")


n_samples = 4
n_channels = 4
fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
for k in range(n_samples):
    axes[0, 0].set_ylabel("Input")
    if k != 0:
        axes[0, k].yaxis.set_visible(False)
    axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")

    # Plot all output channels
    for c in range(n_channels):
        axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
        if k != 0:
            axes[c, k].yaxis.set_visible(False)
        axes[c + 1, k].imshow(q_train_images[k, :, :, c], cmap="gray")

plt.tight_layout()
plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(14*14*4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()


train_data = []
train_target = []
for i in range(len(q_train_images)):
    train_data.append(q_train_images[i])
    train_target.append(train_dataset.targets[i])

test_data = []
test_target = []
for i in range(len(q_test_images)):
    test_data.append(q_test_images[i])
    test_target.append(test_dataset.targets[i])


class Train_dataset(Dataset):
    def __init__(self):
        self.src = train_data
        self.trg = train_target

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.trg[index]


class Test_dataset(Dataset):
    def __init__(self):
        self.src = test_data
        self.trg = test_target

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.trg[index]

train_dataset = Train_dataset()
test_dataset = Test_dataset()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

model = Net().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

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
