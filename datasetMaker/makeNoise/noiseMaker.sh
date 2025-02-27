generate_random_number() {
    local min=$1
    local max=$2
    echo $((RANDOM % (max - min + 1) + min))
}

mkdir -p ./dataXplor

#one call wil generate 10 random spiralDNA
# numberofMrcFile=ls ../pdbToSmallMrrBoxes/boxesOutput/*.mrc | wc -l
generate_noise() {
    mrcFile=$(readlink -f $1)
    dirRandom="tmp$RANDOM"
    mkdir -p "$dirRandom/"
    cp "$mrcFile" "$dirRandom/"
    cd "$dirRandom/"
    for i in {1..2}
    do
        
        cp ../workParameter/* .
        number=$(generate_random_number 0 30)
        temp=$(generate_random_number 50 300)
        noise=$(echo "scale=2;$number*0.01+0.18"|bc)
        sed -i "s/densityNoiseScale x/densityNoiseScale $noise/" "noise.dat"
        sed -i "s/densityNoiseTemperature y/densityNoiseTemperature $temp/" "noise.dat"
        sed -i "s|density densityFileName z|density densityFileName $mrcFile|" "noise.dat"
        # echo $mrcFile
        MMB -c "noise.dat" >> mmb.log
        cat "===================================================" >> ../mmb.log
        cat mmb.log >> ../mmb.log
        cp noisyMap.xplor "../dataXplor/noisyMap$(basename $mrcFile)0$noise-$temp.xplor"
        
        
    done
    cd ..
    rm -rf "$dirRandom"
    
}
maxNumberofProcess=64

for file in ../pdbToSmallMrrBoxes/boxesOutput/*.mrc 
do
    # echo $file
    generate_noise $file &
    if [ $(jobs | wc -l) -ge $maxNumberofProcess ]; then
        wait
    fi
done
wait