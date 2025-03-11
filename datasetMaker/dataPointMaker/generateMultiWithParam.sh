#\bin\bash
if [ "$1" == "" ]; then
        source ../parameter.sh
else
        fileByProcess=$1
        centerMin=20
        centerMax=20
        radiusMin=1
        radiusMax=50

fi

generate_random_number() {
    local min=$1
    local max=$2
    echo $((RANDOM % (max - min + 1) + min))
}
dirRandom="tmp$RANDOM"
mkdir "$dirRandom/"
cd "$dirRandom/"
mkdir -p ../data
#one call wil generate n(define in $fileByProcess) random spiralDNA
for i in {1..$fileByProcess}
do
    #generation of center and radius(only thing that change in this stepp)
    center="$(generate_random_number $centerMin $centerMax).00 $(generate_random_number $centerMin $centerMax).00 $(generate_random_number $centerMin $centerMax).00"
    radius=$(generate_random_number $radiusMin $radiusMax)
    cp ../workParameter/* .
    sed -i "s/spiralDNA center i/spiralDNA center $center/" "command.generateSpiral.dat"
    sed -i "s/spiralDNA radius j/spiralDNA radius 0.5/" "command.generateSpiral.dat"
    sed -i "s/spiralDNA radius j/spiralDNA radius $radius\.00/" "command.generateSpiral.dat"
    MMB -c command.generateSpiral.dat 
    center=$(echo "$center" | sed 's/ /_/g')
    mv last.2.pdb ../data/"spiralDNAcenter$center-radius$radius.pdb"
    
done
cd ..
rm -rf "$dirRandom"