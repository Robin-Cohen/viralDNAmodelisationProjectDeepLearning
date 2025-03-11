import torch
import torchvision
import torchvision.transforms as transforms
from extractParameterFromName import getMrcFeatures
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import mrcfile
import numpy as np
import matplotlib.pyplot as plt

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class MrcDataset(Dataset):
    def __init__(self, mrcDic, transform=None): 
        self.MrcDic = mrcDic
        self.transform = transform

    def __len__(self):
        return len(self.MrcDic)
    
    def __getitem__(self, idx):
        mrcfileName = self.MrcDic[idx]["filename"]
        label = self.MrcDic[idx]["radius"]
        # Loading data
        try:
            with mrcfile.open(mrcfileName, mode='r+') as mrc:
                mrcData = mrc.data
        except:
            print(f"Error loading file {mrcfileName}")
            return None

        # Normalisation --> TO DO chech how to do it with transform
        mrcData = (mrcData - np.min(mrcData)) / (np.max(mrcData) - np.min(mrcData))
        #tensor conversion
        mrcData = torch.from_numpy(mrcData).float()
        label = torch.tensor(label, dtype=torch.float32)
        # Add channel dimension
        mrcData = mrcData.unsqueeze(0)
        if self.transform:
            mrcData = self.transform(mrcData)
        return mrcData, label


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #padding= kernel_size-1/2
        self.conv1 = torch.nn.Conv3d(1,16, kernel_size=(5,5,5), stride=2, padding=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=1)

        self.conv2 = torch.nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=1)

        self.fc1 = torch.nn.LazyLinear(1)
        

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))   
        x = self.pool2(self.relu2(self.conv2(x))) 
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x.squeeze(-1)
    





dataset = MrcDataset(getMrcFeatures())

# print("------------------------------------------------")
# print(dataset.shape()))
# print("------------------------------------------------")
# print(dataset[0][0].size())
# devTestLoader = DataLoader(dataset, batch_size=1, shuffle=True)


trainDataset, testDataset = random_split(dataset, [0.8, 0.2])

trainDataloader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testDataloader = DataLoader(testDataset, batch_size=64, shuffle=False)

# print(trainDataloader)
print("------------------------------------------------")
print(len(trainDataloader))
print("------------------------------------------------")

model= Model()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.
last_loss = 0.

for inputs, labels in trainDataloader:
    print(f"Input shape: {inputs.shape}")
    print(f"Label shape: {labels.shape}")
    break
print("------------------------------------------------")
print("training started")
print(len(trainDataloader))

running_losses = []
losses = []
for epoch in range(10):
    print(f"Epoch {epoch}")
    for i, data in enumerate(trainDataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_losses.append(running_loss)
        losses.append(loss.item()/len(trainDataloader))
        print(f"Batch: Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")
        if i == len(trainDataloader) - 1:
            print("------------------------------------------------")
            print(f"Epoch {epoch+1}, loss: {running_loss/len(trainDataloader):.4f}")
            losses.append(running_loss/len(trainDataloader))
            running_loss = 0.0
            running_losses.append(running_loss)
print("Finished Training")
plt.plot(losses)
plt.savefig("loss.png")
plt.plot(running_losses)
plt.savefig("running_loss.png")
# Save the model
torch.save(model.state_dict(), "./model.pth")