import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as dset
from hw2_ResNet import ResNet
import struct
import os
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)


root = './hw2_data'
if not os.path.exists(root):
    os.mkdir(root)

normalization = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=normalization, download=True)
test_set = dset.MNIST(root=root, train=False, transform=normalization, download=True)
trainLoader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
testLoader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

net = ResNet(64)

numparams = 0
for f in net.parameters():
    print(f.size())
    numparams += f.numel()

optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=0)
optimizer.zero_grad()

criterion = nn.CrossEntropyLoss()

train_losses = []
test_losses = []
test_accs = []

def test(net, testLoader):
    net.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for (data,target) in testLoader:
            output = net(data)
            loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        test_losses.append(loss / len(testLoader))
        test_accs.append(100.*correct/len(testLoader.dataset))
        print("Test Accuracy: %f" % (100.*correct/len(testLoader.dataset)))

test(net, testLoader)

for epoch in range(400):
    net.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(trainLoader):
        pred = net(data)
        loss = criterion(pred, target)
        loss.backward()
        gn = 0
        for f in net.parameters():
            gn = gn + torch.norm(f.grad)
        #print("E: %d; B: %d; Loss: %f; ||g||: %f" % (epoch, batch_idx, loss, gn))
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(trainLoader))
    test(net, testLoader)

plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.legend()
plt.show()
