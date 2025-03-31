import torch
from extractParameterFromName import *
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

class MrcDataset2v(Dataset):
    def __init__(self, mrcDic, transform=None): 
        self.MrcDic = mrcDic
        self.transform = transform

    def __len__(self):
        return len(self.MrcDic)
    
    def __getitem__(self, idx):
        
        mrcfileName = self.MrcDic[idx]["filename"]
        
        # label = [self.MrcDic[idx]["radius"], self.MrcDic[idx]["pitch"]]
        label = [self.MrcDic[idx]["radius"], self.MrcDic[idx]["pitch"]]

        # Loading data
        with mrcfile.open(mrcfileName, mode='r+') as mrc:
            mrcData = mrc.data

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
        
class ModelMultiReg(nn.Module):
    def __init__(self, input_shape=(1, 35, 35, 35)):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 7, stride=2, padding=3),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(3, stride=2),
            
            nn.Conv3d(16, 32, 5, padding=2),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Dropout3d(0.3),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Dropout3d(0.3),

            nn.Conv3d(64,2,1, stride=1,padding=1),
            nn.LeakyReLU(),

            nn.Conv3d(2, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Dropout3d(0.2),

            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.Dropout3d(0.2),

            nn.AdaptiveAvgPool3d((4, 4, 4))
        )

class MrcDataset3v(Dataset):
    def __init__(self, mrcDic, transform=None): 
        self.MrcDic = mrcDic
        self.transform = transform

    def __len__(self):
        return len(self.MrcDic)
    
    def __getitem__(self, idx):
        
        mrcfileName = self.MrcDic[idx]["filename"]
        
        # label = [self.MrcDic[idx]["radius"], self.MrcDic[idx]["pitch"]]
        label = [self.MrcDic[idx]["radius"], self.MrcDic[idx]["pitch"],self.MrcDic[idx]["phi"]]

        # Loading data
        with mrcfile.open(mrcfileName, mode='r+') as mrc:
            mrcData = mrc.data

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
        
class ModelMultiReg(nn.Module):
    def __init__(self, input_shape=(1, 35, 35, 35)):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 7, stride=2, padding=3),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(3, stride=2),
            
            nn.Conv3d(16, 32, 5, padding=2),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Dropout3d(0.3),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Dropout3d(0.3),

            nn.Conv3d(64,2,1, stride=1,padding=1),
            nn.LeakyReLU(),

            nn.Conv3d(2, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Dropout3d(0.2),

            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.Dropout3d(0.2),

            nn.AdaptiveAvgPool3d((4, 4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*4*4*4, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SkipBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels*2, 3, padding=1)
        self.conv2 = nn.Conv3d(in_channels*2, in_channels, 1)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class ModelWithSkipConn(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 7, 2, 3),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(3, 2),
            
            SkipBlock(16),
            nn.Conv3d(16, 32, 5, padding=2),
            nn.MaxPool3d(2, 2),
            nn.Dropout3d(0.2),

            SkipBlock(32),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.Dropout3d(0.2),

            nn.Conv3d(64, 128, 1),
            nn.LeakyReLU(),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4*4, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
