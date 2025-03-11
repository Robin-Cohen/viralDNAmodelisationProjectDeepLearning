import torch
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
        self.drop1=nn.Dropout3d(0.2)

        self.conv2 = torch.nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.drop2=nn.Dropout3d(0.2)

        self.conv3 = torch.nn.Conv3d(32, 64, kernel_size=(2,2,2), stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=1)


        self.fc1 = torch.nn.LazyLinear(1)
    def forward(self, x):
        x = self.drop1(self.relu1(self.conv1(x)))
        x = self.drop2(self.relu2(self.conv2(x)))
        x = self.pool1(self.relu3(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x.squeeze(-1)