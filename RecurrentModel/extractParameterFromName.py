import os
import re
import mrcfile

def get_radius_value(filename):
    match = re.search(r'radius(\d+)', filename)
    if match:
        return int(match.group(1))
    return None
def getMrcFeatures(directory="/home/robin/viralDNAmodelisationProjectDeepLearning/dataset/")->dict:
    fileFeatures = {}
    for num, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".mrc"):
            fileFeatures[num]={}
            fileFeatures[num]["filename"]=os.path.join(directory, filename)
            

            #get radius
            radius = get_radius_value(filename)
            fileFeatures[num]["radius"]=radius

    #if continue with filename use this solution, otherwise would be good to build a csv file with the features at the file generation
    return fileFeatures

def get_corruptedFile(mrcDir):
    corruptedFile=[]
    for filename in os.listdir(mrcDir):
        if filename.endswith(".mrc"):
            try:
                with mrcfile.open(os.path.join(mrcDir, filename), mode='r+') as mrc:
                    mrcData = mrc.data
            except:
                print(f"Error loading file {filename}")
                corruptedFile.append(filename)
    return corruptedFile

if __name__ == "__main__":
    print(get_corruptedFile("/home/robin/viralDNAmodelisationProjectDeepLearning/dataset/"))