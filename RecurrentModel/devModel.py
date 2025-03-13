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
        mindata=np.min(mrcData)
        for i in range(mrcData.shape[0]):
            mrcData[i][0][0]=0.0
        fileStd=np.std(mrcData)
        fileMean=np.mean(mrcData)
        mrcData = (mrcData - fileMean) / fileStd
        #old way: mrcData = (mrcData - np.min(mrcData)) / (np.max(mrcData) - np.min(mrcData))
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
        self.conv1 = nn.Conv3d(1, 16, kernel_size=15, stride=2, padding=7)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(16, 32, kernel_size=10, stride=1, padding=4)
        self.relu2 = nn.ReLU()
        # self.drop2 = nn.Dropout3d(0.2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=5, stride=2, padding=2)
        self.relu3 = nn.ReLU()
        # self.drop1 = nn.Dropout3d(0.2)

        self.conv4 = nn.Conv3d(64, 128, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=2)
        self.relu5 = nn.ReLU()
        
        self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))  

        self.fc1 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = (self.relu2(self.conv2(x)))
        x = (self.relu3(self.conv3(x)))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.pool1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x.squeeze(-1)
#######################################################""
class FullyConnectedRegressor(nn.Module):
    def __init__(self):
        super(FullyConnectedRegressor, self).__init__()
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(35*35*35, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.flatten(x)
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        x = self.tanh(self.fc6(x))
        x = self.fc7(x)
        return x.squeeze(-1)

