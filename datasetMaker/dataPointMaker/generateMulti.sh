#\bin\bash
generate_random_number() {
    local min=$1
    local max=$2
    echo $((RANDOM % (max - min + 1) + min))
}
dirRandom="tmp$RANDOM"
mkdir "$dirRandom/"
cd "$dirRandom/"
mkdir -p ../data
#one call wil generate 10 random spiralDNA
for i in {1..10}
do
    center="$(generate_random_number -20 20).00 $(generate_random_number -20 20).00 $(generate_random_number -20 20).00"
    radius=$(generate_random_number 1 50)
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