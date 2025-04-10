# This script takes a mrc file and cuts it into smaller boxes
#Redo this script to use proper function of mrcfile library-->update_header_from_data()

import mrcfile
import sys
import os
import numpy as np

DISCARD_RATE = 0.05 # 1% of the data points in the box
def putHeader(header, box):
    header.mx = box.shape[2]
    header.my = box.shape[1]
    header.mz = box.shape[0]
    header.cella.x = box.shape[2]
    header.cella.y = box.shape[1]
    header.cella.z = box.shape[0]
    header.map = b'MAP '
    header.machst = 0x44440000
    header.dmin = box.min()
    header.dmax = box.max()
    header.dmean = box.mean()
    header.ispg = 0
    header.nsymbt = 0
    header.origin.x = 0
    header.origin.y = 0
    header.origin.z = 0
    header.rms = box.std()
    header.nlabl = 0
    header.label = b''
    return header

def saveWithHeader(inputMrcFile, dataNewMrcFile, i, inputFileName, output_dir):
    # newHeader=inputMrcFile.header.copy()
    newVsize = inputMrcFile.voxel_size.copy()
                    
                    
    input_file_name = os.path.basename(inputFileName).split(".mrc")[0] # take  the basneme of the file and remove the .mrc extension
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path =  f"{output_dir}/{input_file_name}_box_{i}.mrc"

    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(dataNewMrcFile)
        # newHeader = putHeader(newHeader, dataNewMrcFile)
        # mrc.header.set = newHeader
        mrc.voxel_size = newVsize
        mrc.update_header_from_data()
        
    
def getNumDataPoint(data:np.array)->int:
    numDataPoint=0
    for z in range(data.shape[0]):
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):
                if data[z][y][x]!=0:
                    numDataPoint+=1
    return numDataPoint


def cut_mrc_file_to_boxes(file_path,  output_dir, inputFileName, box_size=35,stride=35):
    with mrcfile.open(file_path, permissive=True) as mrc:
        data = mrc.data
        counterBox=0
        print(f"Data shape: {data.shape}")
        fullDataPoint=getNumDataPoint(data)
        for z in range(0, data.shape[0], stride):
            for y in range(0, data.shape[1], stride):
                for x in range(0, data.shape[2] , stride):
                    box = data[z:z+box_size, y:y+box_size, x:x+box_size]
                    if box.shape != (box_size, box_size, box_size):
                        pad_width = [(0, max(0, box_size - s)) for s in box.shape]
                        box = np.pad(box, pad_width, mode='constant', constant_values=0)
                    if not box.any():
                        continue
                    boxDataPoint=getNumDataPoint(box)
                    print(f"Percent box data point: {(boxDataPoint/fullDataPoint)*100}")
                    if (boxDataPoint/fullDataPoint)<DISCARD_RATE:
                        print(f"Box has less than {DISCARD_RATE*100}%\ of the data points, skipping")
                        continue
                    counterBox+=1
                    saveWithHeader(mrc, box, counterBox, inputFileName, output_dir)
                    
        return counterBox

def save_boxes_as_mrc_files(boxes, output_dir, inputFileName):
    input_file_name = os.path.basename(inputFileName).split(".mrc")[0] # take  the basneme of the file and remove the .mrc extension
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, box in enumerate(boxes):
        output_path =  f"{output_dir}/{input_file_name}_box_{i}.mrc"
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(box)

if __name__ == "__main__":
    InputMrcFiles=[]
    isAInputDir = False
    output_dir = "boxesOutput" #default output directory
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-output_dir":
            output_dir = sys.argv[i+1]
            isAInputDir = True
        elif isAInputDir:
            isAInputDir = False
            continue
        else:
            InputMrcFiles.append(sys.argv[i])
    for mrcFilePath in InputMrcFiles:
        nbox=cut_mrc_file_to_boxes(mrcFilePath, output_dir, mrcFilePath, box_size=35)
        # save_boxes_as_mrc_files(boxes, output_dir, mrcFilePath)
        print(f" {nbox} boxes created from {mrcFilePath} to {output_dir}")
        