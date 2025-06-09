#/home/robin/opt/miniconda3/envs/ReccurentModel/bin/python
import numpy as np
import mrcfile
import sys
import os
import csv
from filelock import FileLock

class XPLOR_Density_Map:
    def __init__(self, path):
        self.path = path
        self._read_header()

    def _read_header(self):
        with open(self.path, 'rb') as f:
            f.readline()  # first empty line
            ntitle = int(f.readline().split()[0])  # Number of coment lines
            
            for _ in range(ntitle):
                f.readline()  # Ignore comments

            extent_line = f.readline().split()
            self.na, self.amin, self.amax = map(int, extent_line[:3])
            self.nb, self.bmin, self.bmax = map(int, extent_line[3:6])
            self.nc, self.cmin, self.cmax = map(int, extent_line[6:9])
            self.grid_size = (self.amax - self.amin + 1, 
                              self.bmax - self.bmin + 1, 
                              self.cmax - self.cmin + 1)

            cell_line = f.readline().split()
            self.cell_size = list(map(float, cell_line[:3]))

            f.readline()  

            self.data_offset = f.tell()

    def read_data(self):

        data = np.zeros(self.grid_size, dtype=np.float32)
        with open(self.path, 'r') as f:
            f.seek(self.data_offset)
            for c in range(self.grid_size[2]):
                f.readline()  # section number c
                values = []
                while len(values) < (self.grid_size[0] * self.grid_size[1]):
                    line = f.readline().split()
                    values.extend(map(float, line))  # Ajouter les valeurs en tenant compte des lignes multiples
                data[c, :, :] = np.array(values, dtype=np.float32).reshape(self.grid_size[1], self.grid_size[0])
                
            f.readline()  # 
            f.readline()  #
        
        return data

    def to_mrc(self, output_path):
        data = self.read_data()
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = self.cell_size  #put into voxel
            mrc.update_header_from_data()

def putInCsv(mrcNoisyFileName, mrcClean, csvFile, noiseFactor, tempFactor):
    updated_rows = []
    found = False
    print("putInCsv")
    mrcClean = mrcClean.split("/")[-1]
    print(f"mrc clean: {mrcClean} mrc noisy: {mrcNoisyFileName}")
    #temp modification
    csvFile = os.path.dirname(csvFile)
    csvFile = os.path.join(csvFile, "boxInfoNoise.csv")
    if not os.path.exists(csvFile):
        print(f"CSV file {csvFile} does not exist. Creating a new one.")
        with open(csvFile, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["Box_file_name_No_Noise", "mrcNoisyFileName", "noiseFactor", "tempFactor"])  # Write header
    with open(csvFile, 'a+') as file:
        # write the new row
        writer = csv.writer(file)
        writer.writerow([mrcClean, mrcNoisyFileName, noiseFactor, tempFactor])  # Write the new row


# usage--> to do replace with choice of user
# xplor_file = "/home/robin/viralDNAmodelisationProjectDeepLearning/datasetMaker/makeNoise/correctedXplor/noisyMap.xplor"
mrc_output = os.environ["noiseDirPath"]
os.mkdir(mrc_output) if not os.path.exists(mrc_output) else None
# xplor_map = XPLOR_Density_Map(xplor_file)
# xplor_map.to_mrc(mrc_output)

def process_file(xplor_file,mrcClean, noiseFactor, tempFactor):

    csvfile = os.environ["boxinfo"]
    xplor_map = XPLOR_Density_Map(xplor_file)
    print(f"Processing {xplor_file}")
    mrcName= os.path.basename(xplor_file).split(".xplor")[0] + ".mrc"
    mrcFullOutputpath = os.path.join(mrc_output, mrcName) 
    xplor_map.to_mrc(mrcFullOutputpath)
    print(f"{xplor_file} converted to mrc")
    putInCsv(mrcName,mrcClean, csvfile, noiseFactor, tempFactor)

# repetory = sys.argv[1]
# xplor_file = [os.path.join(repetory, file) for file in os.listdir(repetory) if file.endswith('.xplor')]
# print(len(xplor_file))
xplor_file = sys.argv[1]
mrcClean = sys.argv[2]
noiseFactor = sys.argv[3]
tempFactor = sys.argv[4]
process_file(xplor_file,mrcClean, noiseFactor, tempFactor)