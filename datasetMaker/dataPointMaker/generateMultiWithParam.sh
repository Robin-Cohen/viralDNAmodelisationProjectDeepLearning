#\bin\bash
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

fi

generate_random_number() {
    local min=$1
    local max=$2
    echo $((RANDOM % (max - min + 1) + min))
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
#create a data directory if it doesn't exist
mkdir -p ../data
#one call wil generate n(define in $fileByProcess) random spiralDNA
for ((i=0; i<=$fileByProcess; i++))
do
    echo "-----------------------------------"
    echo "i: $i"
    #generation of center and radius(only thing that change in this stepp)
    center="$(generate_random_number $centerMin $centerMax).00 $(generate_random_number $centerMin $centerMax).00 $(generate_random_number $centerMin $centerMax).00"
    radius=$(generate_random_number $radiusMin $radiusMax)
    pitch=$(generate_random_number $pitchMin $pitchMax)
    pitch=$(echo "$pitch * 0.1" | bc)

    echo "center: $center"
    echo "radius: $radius"
    echo "pitch: $pitch"
    cp ../workParameter/* .
    sed -i "s/spiralDNA center spiralCenterValues/spiralDNA center $center/" "command.generateSpiral.dat"
    # sed -i "s/spiralDNA radius spiralRadiusValue/spiralDNA radius 0.5/" "command.generateSpiral.dat"
    sed -i "s/spiralDNA radius spiralRadiusValue/spiralDNA radius $radius\.00/" "command.generateSpiral.dat"
    sed -i "s/spiralDNA pitch pitchValue/spiralDNA pitch $pitch/" "command.generateSpiral.dat"
    MMB -c command.generateSpiral.dat 
    center=$(echo "$center" | sed 's/ /_/g')
    gemmi convert last.2.cif ../data/"spiralDNAcenter$center-radius$radius-pitch$pitch.pdb"
done
cd ..
rm -rf "$dirRandom"