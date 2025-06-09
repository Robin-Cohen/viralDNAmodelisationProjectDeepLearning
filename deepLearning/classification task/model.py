import torch
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split , ConcatDataset
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import csv
import random

leakySlope=0.15

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class MrcDataset1vMetaDataWithNoiseFile(Dataset):
    def __init__(self, metaFile, noiseDirectory,noNoiseDirectory,onlyNoise,training= False, transform=None): 
        self.transform = transform
        self.augmenation_prob = 0.5
        self.MrcFiles =[]
        self.filterMetaData(metaFile,noiseDirectory,noNoiseDirectory)
        self.denoise_prob = 0.05
        self.training = training
        self.loadOnlyNoisyFile(onlyNoise)
        

    def setTraining(self,training:bool)->None: #set so test dataset will not be denoised
        self.training = training
    
    def filterMetaData(self,metaFile,NoisyDirectory,noNoiseDirectory):
        mrcDic={
            "filename": None,
            "radius": None,
            "pitch": None,
            "dataPoints": None,
            "zEnd": None,
            "denoiseFileName": None,
            "onlyNoise" : False,
        }
        with open(metaFile, mode="r+") as metaData:
            files= [os.path.basename(file) for file in os.listdir(NoisyDirectory)]
            csv_reader = csv.DictReader(metaData)
            totaleFind=0
            
            for row in csv_reader:
                # print(row["Box_file_name_With_Noise"]) if row["Box_file_name_With_Noise"] is not None else None
                for file in files:
                    if row["Box_file_name_With_Noise"] in files:
                        totaleFind+=1
                        mrcDic["filename"]=(os.path.join(NoisyDirectory,row["Box_file_name_With_Noise"]))
                        mrcDic["radius"]=float(row["radius"])
                        mrcDic["pitch"]=float(row["pitch"])
                        mrcDic["dataPoints"]=int(row["numberOfPointBox"])
                        if mrcDic["dataPoints"]<500:
                            mrcDic["onlyNoise"] = True
                        mrcDic["zEnd"]=row["zEndpoint"] # will often be None
                        if mrcDic["zEnd"]=="None":
                            mrcDic["zEnd"] = 0
                        else:
                            mrcDic["zEnd"] = 1
                        mrcDic["denoiseFileName"]=os.path.join(noNoiseDirectory,row["Box_file_name_No_Noise"])
                        files.remove(file)
                        self.MrcFiles.append(mrcDic.copy())
    def loadOnlyNoisyFile(self,onlyNoiseDirectory):
        mrcDic={
            "filename": None,
            "radius": None,
            "pitch": None,
            "dataPoints": None,
            "zEnd": None,
            "denoiseFileName": None,
            "onlyNoise" : True,
        }
        files=os.listdir(onlyNoiseDirectory)
        random.shuffle(files)
        for filename in files[0:1000]:
            if filename.endswith(".mrc"):
                mrcDic["filename"]=os.path.join(onlyNoiseDirectory,filename)
                self.MrcFiles.append(mrcDic.copy())
            else:
                continue
        return
    def __len__(self):
        return len(self.MrcFiles)

    def filp(self,mrcDataTorch:torch,flip_prob:float)->torch:
        if np.random.rand() < flip_prob:
            axes = [0, 2]#z axis
            
            mrcDataTorch = torch.flip(mrcDataTorch, dims=[axes[np.random.randint(len(axes))]])
        return mrcDataTorch

    def rot(self,mrcDataTorch:torch)->torch:
        turnNumber=np.random.randint(4)# 25% chance of rotation (also 25% that nothing happen--> case 0 )
        axe= [0,1]
        torch.rot90(mrcDataTorch, k=turnNumber, dims=axe) 
        return mrcDataTorch
    
            #mrcDataTorch = torch.nn.functional.dropout(mrcDataTorch, p=0.5, training=True)
    def augmentation(self, mrcDataTorchFormated:torch)->torch:
        
        mrcDataTorchFormated=self.filp(mrcDataTorchFormated,0.5)
        mrcDataTorchFormated = self.rot(mrcDataTorchFormated)
        return mrcDataTorchFormated
    
    def doesNeedDenoiseAugmentation(self, fileName:str,denoiseFileName:str)->str: #randomly replace noisy file with equivalent denoised File
        if  np.random.rand() <= self.denoise_prob and not self.training:
            isExist = os.path.exists(denoiseFileName)
            if isExist:
                fileName = denoiseFileName
        
        return fileName
    
    def __getitem__(self, idx):
        
        mrcDic = self.MrcFiles[idx]
        mrcfileName = mrcDic["filename"]
        

        if mrcDic["onlyNoise"]:
            label = [2] #score when no data present
        else:
            mrcfileName = self.doesNeedDenoiseAugmentation(mrcfileName,mrcDic["denoiseFileName"])
            label = [mrcDic["zEnd"]]

        # Loading data
        try:
            with mrcfile.open(mrcfileName, mode='r+') as mrc:
                mrcData = mrc.data #extraction of the data
        except (ValueError, OSError) as e:
            print(f"file {mrcfileName} is corrupted. Error : {str(e)}")
            return None
        mindata=np.min(mrcData)
        for i in range(mrcData.shape[0]):
            mrcData[i][0][0]=0.0
        fileStd=np.std(mrcData)
        fileMean=np.mean(mrcData)
        mrcData = (mrcData - fileMean) / fileStd
        mrcData = torch.from_numpy(mrcData).float()
        label = torch.tensor(label, dtype=torch.long)
        #data augmentation
        
        mrcData = self.augmentation(mrcData)
        
        # Add channel dimension
        mrcData = mrcData.unsqueeze(0)
        if self.transform:
            mrcData = self.transform(mrcData)
        return mrcData, label


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



class ModelClassificationMultival(nn.Module):
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
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 3)
        )
    def forward(self, x):
        return self.classifier(self.features(x))