#!/bin/bash

makeUniqueDirectory() {
    # Create a new directory with a unique name
    target_name=$1
    if [ -d "$target_name" ]; then
        i=1
        while [ -d "${target_name}_$i" ]; do
            i=$((i + 1))
        done
        target_name="${target_name}_$i"
    fi

    echo $target_name
}
dataDir=$(makeUniqueDirectory "datasetMrcBox")
mkdir -p $dataDir
dataDir=$(readlink -f $dataDir)
if [ -d "$dataDir" ]; then
    echo "Data will be stored  in $dataDir"
    
else
    echo "Couln't create directory $dataDir"
    exit
fi


echo "Creating pdb dataset..."
cd dataPointMaker/
bash multi_param.sh
echo "done"
echo "converting to mrc boxes..."
cd ../pdbToSmallMrrBoxes/
bash pdbToMrc.sh
wait
echo "done"
cd ../

mv box_info.csv $dataDir
cat $dataDir/box_info.csv | wc -l
export boxinfo=$(readlink -f "$dataDir/box_info.csv")

# Create a new directory with a unique name
noNoiseDir=$(makeUniqueDirectory "noNoise")
export noNoiseDirPath="$dataDir/$noNoiseDir"

echo $dataDir
echo $noNoiseDirPath
# Move the contents of the directory to the new name
echo "Moving boxes files to $noNoiseDirPath ..."
mv pdbToSmallMrrBoxes/boxData/ $noNoiseDirPath



rm dataPointMaker/data/*
mv dataPointMaker/labels.csv $dataDir
rm dataPointMaker/out.log
rm pdbToSmallMrrBoxes/fullMrcData/*.mrc
cd makeNoise/
#take a lot of time
# # Generate noise
echo "Generating noise on data in $noNoiseDirPath"


noiseDir=$(makeUniqueDirectory "noisy")
export noiseDirPath="$dataDir/$noiseDir"

bash noiseMaker.sh $noNoiseDirPath 
echo "done making Noise"
cd ../
echo "cleaning "




