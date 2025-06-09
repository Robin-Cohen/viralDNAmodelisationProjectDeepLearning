import torch
from extractParameterFromName import *
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split , ConcatDataset
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import csv
leakySlope=0.2

class SkippConnnection(nn.Module):
    def __init__(self, in_channels, expansion=4):  
        super().__init__()
        hidden_dim = in_channels * expansion
        
        self.block = nn.Sequential(

            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(leakySlope),
            nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
            
            nn.BatchNorm3d(hidden_dim),
            nn.LeakyReLU(leakySlope),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            
            nn.BatchNorm3d(hidden_dim),
            nn.LeakyReLU(leakySlope),
            nn.Conv3d(hidden_dim, in_channels, 1, bias=False),
        )
        
        self.shortcut = nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
    

class Model4(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(

            nn.Conv3d(in_channels, 32, 3, padding=1, bias=False),
            nn.Dropout3d(0.2),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(leakySlope),
            nn.MaxPool3d(2, 2),
            

            SkippConnnection(32),
            SkippConnnection(32),

            nn.Conv3d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(leakySlope),
            # CBAM(64),  
            nn.Dropout3d(0.3),
            
            SkippConnnection(64),
            SkippConnnection(64),

            nn.Conv3d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.Dropout3d(0.3),
            nn.LeakyReLU(leakySlope),

            nn.Conv3d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.Dropout3d(0.2),
            nn.LeakyReLU(leakySlope),

            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*4*4*4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(leakySlope),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))
