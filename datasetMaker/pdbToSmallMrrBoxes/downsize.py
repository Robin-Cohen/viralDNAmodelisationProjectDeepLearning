# This script takes a mrc file and cuts it into smaller boxes
#Redo this script to use proper function of mrcfile library-->update_header_from_data()

import mrcfile
import sys
import os
import numpy as np
from Bio.PDB import PDBParser
import re

DISCARD_RATE = 0

def moveBox(box, boxCoordinate, orginalData, variableRange=3):
    box_size = 35
    variation= np.random.randint(-variableRange, variableRange+1)
    boxCoordinate[0] += variation
    boxCoordinate[1] += variation
    boxCoordinate[2] += variation
    z, y, x = boxCoordinate

    box = orginalData[z:z+box_size, y:y+box_size, x:x+box_size]
    if box.shape != (box_size, box_size, box_size):
        pad_width = [(0, max(0, box_size - s)) for s in box.shape]
        box = np.pad(box, pad_width, mode='constant', constant_values=0)

    if not box.any():
        return None
    boxDataPoint=getNumDataPoint(box)
    fullDataPoint=getNumDataPoint(orginalData)
    if (boxDataPoint/fullDataPoint)<DISCARD_RATE:
        # print(f"Box has less than {DISCARD_RATE*100}%\ of the data points, skipping")
        return None
    return [box, boxCoordinate]

def get_radius_value(filename):
    match = re.search(r'radius(\d+.\d+)', filename)
    if match:
        return float(match.group(1))
    return None
def get_pitch_value(filename):
    match = re.search(r'pitch(\d)', filename)
    match_decimal = re.search(r'pitch(\d+\.\d+)', filename)
    if match_decimal:
        return float(match_decimal.group(1))

def putInfoInCsv(inputFileName, dataPointBox, dataPointOriginal,relative_coordinates, endPoint):
    csv_file_path = os.path.join("../..", "box_info.csv")
    print(f"csv file path: {csv_file_path}")
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w") as csv_file:
            csv_file.write("Box_file_name_No_Noise,radius,pitch,xEndpoint,yEndpoint,zEndpoint,xOriginBox,yOriginBox,zOriginBox,numberOfPointBox,numberOfPointOrigin\n")
    with open(csv_file_path, "a") as csv_file:
        radius = get_radius_value(inputFileName)
        pitch = get_pitch_value(inputFileName)
        for data in endPoint:
            if data is None:
                data = "NaN"
        csv_file.write(f"{inputFileName},{radius},{pitch},{endPoint[0]},{endPoint[1]},{endPoint[2]},{relative_coordinates[0]},{relative_coordinates[1]},{relative_coordinates[2]},{dataPointBox},{dataPointOriginal}\n")
        
        
def parse_pdbForFirstAndLastAtom(pdb_path):
    iter=0
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb_path)
    atom_coords = []
    last_atom = None
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if iter == 0:
                        iter+=1
                        atom_coords.append(atom.coord)
                    else:
                        last_atom = atom.coord
                    
    atom_coords.append(last_atom)
    
    return np.array(atom_coords)

def convertToMrcCoordinates(atom_coords, origin, voxel_size,mapc, mapr, maps,mrc_path, mx, my, mz):
    x, y, z = atom_coords
    i = (x - origin.x)/voxel_size[0]
    j = (y - origin.y)/voxel_size[1]
    k = (z - origin.z)/voxel_size[2]
    
    # orientation gestion --> put on good axis
    axis_order = {int(mapc):0, int(mapr):1, int(maps):2}
    sorted_axes = [axis_order[key] for key in sorted(axis_order)]
    final_indices = np.round([i, j, k])[sorted_axes].astype(int)
    
    with mrcfile.open(mrc_path) as mrc:
        if np.any(final_indices < 0) or np.any(final_indices >= [mx, my, mz]):
            raise ValueError("Coordonnées hors de la grille MRC")
        
    return final_indices

def convert_first_last_coordinate_to_mrc_coordinates(pdb_path, mrc_path):
    atom_coords = parse_pdbForFirstAndLastAtom(pdb_path)
    
    #mrc reading
    with mrcfile.open(mrc_path,"r+") as mrc:
        origin = mrc.header.origin  # x₀,y₀,z₀
        cell = mrc.header.cella     # dimensions en Å
        mx, my, mz = mrc.header.nx, mrc.header.ny, mrc.header.nz
        mapc, mapr, maps = mrc.header.mapc, mrc.header.mapr, mrc.header.maps
        
    # voxel size
    voxel_size = (cell.x/mx, cell.y/my, cell.z/mz)
    
    # conversion
    final_coordinates = []
    for elem in atom_coords:
        final_coordinates.append(convertToMrcCoordinates(elem, origin, voxel_size, mapc, mapr, maps,mrc_path, mx, my, mz))
    
    return final_coordinates

 # Check if one of the two 3D coordinates in firstLastCoordinate is within the box
def is_coordinate_within_box(firstLastCoordinate, relative_coordinates, maxRelative_coordinates):
    for index, coord in enumerate(firstLastCoordinate):
        if all(relative_coordinates[i] <= coord[i] < maxRelative_coordinates[i] for i in range(3)):
            return True, index
    return False, None

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
                    
                    
    input_file_name = os.path.basename(inputFileName) # take  the basneme of the file and remove the .mrc extension
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path =  f"{output_dir}/{input_file_name}"

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


def cut_mrc_file_to_boxes(inputFileName,  output_dir, pdb_path, box_size=35,stride=35):
    # pdb_path = "/home/robin/viralDNAmodelisationProjectDeepLearning/datasetMaker/dataPointMaker/data/spiralDNAcenter0.00_0.00_0.00-radius15-pitch.5.pdb"
    
    firstLastCoordinate =convert_first_last_coordinate_to_mrc_coordinates(pdb_path,inputFileName)
    with mrcfile.open(inputFileName, permissive=True) as mrc:
        data = mrc.data
        counterBox=0
        # print(f"Data shape: {data.shape}")
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
                    # print(f"Percent box data point: {(boxDataPoint/fullDataPoint)*100}")
                    if (boxDataPoint/fullDataPoint)<DISCARD_RATE:
                        # print(f"Box has less than {DISCARD_RATE*100}%\ of the data points, skipping")
                        continue
                    counterBox+=1

                    baseOutputFileName = os.path.basename(inputFileName).split(".mrc")[0] # take  the basneme of the file and remove the .mrc extension
                    relative_coordinates = [x, y, z]
                    maxRelative_coordinates = [x+box_size, y+box_size, z+box_size]
                    strCoordinate = map(str, relative_coordinates)
                    baseOutputFileName+="box"+ "_".join(strCoordinate)
                    endCoordinate=[None, None, None]
                    orginalName= baseOutputFileName
                    is_within, which_coordinate = is_coordinate_within_box(firstLastCoordinate, relative_coordinates, maxRelative_coordinates)
                    if is_within:
                        endCoordinate= firstLastCoordinate[which_coordinate]
                        endCoordinateStr= map(str, endCoordinate)
                        baseOutputFileName +="endpoint"+"_".join(endCoordinateStr)
                        for i in range(3):
                            boxes = moveBox(box, relative_coordinates, data)
                            if boxes is not None:
                                counterBox+=1
                                strCoordinate = map(str, boxes[1])
                                orginalName+="box"+ "_".join(strCoordinate)
                                orginalName +="endpoint"+"_".join(endCoordinateStr) + ".mrc"
                                saveWithHeader(mrc, boxes[0], counterBox, orginalName, output_dir)
                                putInfoInCsv(orginalName,boxDataPoint,fullDataPoint,relative_coordinates, endCoordinate)
                            else:
                                print(baseOutputFileName)
                                continue
                    baseOutputFileName=baseOutputFileName+".mrc"
                    # print(f"creating box {baseOutputFileName}")
                    # print(f"Box {counterBox} relative coordinates: {relative_coordinates} regulate coordinates: {relative_coordinates}")
                    # print(f"first-last coordinate :{firstLastCoordinate}")
                    saveWithHeader(mrc, box, counterBox, baseOutputFileName, output_dir)
                    putInfoInCsv(baseOutputFileName,boxDataPoint,fullDataPoint,relative_coordinates, endCoordinate)
                    
        return counterBox

if __name__ == "__main__":
    InputMrcFiles=[]
    isAInputDir = False
    output_dir = "boxesOutput" #default output directory
    mrc_path = None
    pdb_path = None
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-output_dir":
            output_dir = sys.argv[i+1]
            isAInputDir = True
        elif isAInputDir:
            isAInputDir = False
        elif sys.argv[i] == "-input_pdb":
            pdb_path = sys.argv[i+1]
            isAInputDir = True
        elif sys.argv[i] == "-input_mrc":
            mrc_path = sys.argv[i+1]
            isAInputDir = True
        # print(sys.argv)
    cut_mrc_file_to_boxes(mrc_path, output_dir, pdb_path, box_size=35)