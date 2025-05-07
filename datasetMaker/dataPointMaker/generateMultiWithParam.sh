#\bin\bash
# This script generates multiple spiral DNA structures with random parameters and saves them in a specified directory.
# It also creates a CSV file containing the parameters used for each generated structure.
# Check if the script is being run with a parameter
# If no parameter is provided, source the default parameter file
# If a parameter is provided, use the provided values
if [ "$1" == "" ]; then
        source ../parameter
else
        fileByProcess=10
        centerMin=20
        centerMax=20
        radiusMin=1
        radiusMax=50
        pitchMin=1
        pitchMax=30
        # rightHanded=1
fi

generate_random_number() {
    #redo after discover of shuff function--> will be replace
    local min=$1
    local max=$2
    # echo $((RANDOM % (max - min + 1) + min))
    shuf -i $min-$max -n 1
}

#try to create a directory, if it fail, try again
generate_a_work_dir() {
    local dirRandom="tmp$RANDOM"
    if mkdir "$dirRandom/"; then
        echo "$dirRandom"
    else
        generate_a_work_dir
    fi
    
}

dirRandom=$(generate_a_work_dir)
cd $dirRandom
# Write labels to a CSV file
    csv_file="../labels.csv"
#create a data directory if it doesn't exist
mkdir -p ../data
#one call wil generate n(define in $fileByProcess) random spiralDNA
for ((i=0; i<=$fileByProcess; i++))
do
    echo "-----------------------------------"
    echo "i: $i"
    #generation of center and radius(only thing that change in this stepp)
    centerx="$(seq -f'%.2f' -50 0.1 50 | shuf -n1)"
    centery="$(seq -f'%.2f' -50 0.1 50 | shuf -n1)"
    centerz="$(seq -f'%.2f' -50 0.1 50 | shuf -n1)"
    radius="$(seq -f'%.2f' $radiusMin 0.1 $radiusMax | shuf -n1)"
    # radius=$(echo "$radius * 0.1" | bc)
    # radius=10
    pitch="$(seq -f'%.2f' $pitchMin 0.1 $pitchMax | shuf -n1)"
    # pitch=$(echo "$pitch * 0.1" | bc)
    # pitch=3
    phi="$(seq -f'%.2f' 0 1 360 | shuf -n1)"
    # phi=319
    rightHanded="$(seq -f '%.0f' 0 1|shuf -n1)"
    # theta1or=31
    #teta0 and teta1 are between 0 and 25
    # theta0="$(seq -f'%.3f' 0 0.01 3.14 | shuf -n1)"
    # theta1="$(seq -f'%.3f' 0 0.01 3.14 | shuf -n1)"
    theta0=1.0
    theta1=2.2
    # echo "center: $center"
    echo "radius: $radius"
    echo "pitch: $pitch"
    echo "phi: $phi"
    echo "teta0: $theta0"
    echo "teta1: $theta1"
    echo "rightHanded: $rightHanded"
    cp ../workParameter/* .
    sed -i "s/spiralDNA center spiralCenterValues/spiralDNA center $centerx $centery $centerz/" "command.generateSpiral.dat"
    sed -i "s/spiralDNA radius spiralRadiusValue/spiralDNA radius $radius/" "command.generateSpiral.dat"
    sed -i "s/spiralDNA pitch pitchValue/spiralDNA pitch $pitch/" "command.generateSpiral.dat"
    sed -i "s/spiralDNA phivar/spiralDNA phiOffset  $phi*@pi\/180/" "command.generateSpiral.dat"
    sed -i "s/spiralDNA startTheta/spiralDNA startTheta $theta0/" "command.generateSpiral.dat"
    sed -i "s/spiralDNA endTheta/spiralDNA endTheta $theta1/" "command.generateSpiral.dat"
    sed -i "s/spiralDNA spiralIsRightHanded/spiralDNA spiralIsRightHanded $rightHanded/" "command.generateSpiral.dat"
    cat command.generateSpiral.dat

 
    MMB -c command.generateSpiral.dat > tmp.log
    # center=$(echo "$center" | sed 's/ /_/g')
    fileName="spiralDNAcenter$centerx-$centery-$centerz-rightHanded$rightHanded-radius$radius-pitch$pitch-phi$phi-teta0-$theta0-teta1-$theta1.pdb"
    gemmi convert last.2.cif ../data/$fileName
    

    echo "$fileName,$centerx,$centery,$centerz,$rightHanded,$radius,$pitch,$phi,$theta0,$theta1" >> "$csv_file"
done
cd ..
rm -rf "$dirRandom"