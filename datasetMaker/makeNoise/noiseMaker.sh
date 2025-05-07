generate_random_number() {
    #redo after discover of shuff function--> will be replace
    local min=$1
    local max=$2
    # echo $((RANDOM % (max - min + 1) + min))
    shuf -i $min-$max -n 1
}
maxNumberofProcess=20
source ../parameter

mkdir -p ./dataXplor

#one call wil generate 10 random spiralDNA
# numberofMrcFile=ls ../pdbToSmallMrrBoxes/boxesOutput/*.mrc | wc -l
generate_noise() {
    lineNumber=$3
    relPathFile="$1/$2"
    if [ ! -f "$relPathFile" ]; then
    echo "File $relPathFile not found!"
        return
    fi
    mrcFile=$(readlink -f $relPathFile)
    dirRandom="tmp$RANDOM"
    mkdir -p "$dirRandom/"
    echo "cp $mrcFile $dirRandom/"
    cp "$mrcFile" "$dirRandom/"
    echo "done"
    cd "$dirRandom/"
        
    cp ../workParameter/* .
    number=$(generate_random_number 0 70)
    temp=$(generate_random_number 50 500)
    noise=$(echo "scale=2;$number*0.01+0.18"|bc)
    sed -i "s/densityNoiseScale x/densityNoiseScale $noise/" "noise.dat"
    sed -i "s/densityNoiseTemperature y/densityNoiseTemperature $temp/" "noise.dat"
    sed -i "s|density densityFileName z|density densityFileName $mrcFile|" "noise.dat"
    echo "makeing noise for $mrcFile with noise $noise and temperature $temp" >> mmb.log
    echo "MMB"
    xplorName=(noisyMap$(basename $mrcFile)0$noise-$temp.xplor)
    echo "$xplorName"
    MMB -c "noise.dat" >> mmb.log
    echo "MMB done"
    echo "===================================================" >> ../mmb.log
    cat mmb.log >> ../mmb.log
    
    cp noisyMap.xplor "$xplorName"
    cp noise.xplor "../noise/noise$(basename $mrcFile)0$noise-$temp.xplor"
    echo "putting the noise in the box info file"
    echo "python ../xplorToMrc.py $xplorName $mrcFile $noise $temp"
    python ../xplorToMrc.py $xplorName $mrcFile $noise $temp
    # awk -F "," -v line="$number" -v value="noisyMap$(basename $mrcFile)0$noise-$temp.xplor" 'BEGIN {OFS = FS} NR == line {$12 = value} 1' ../../box_info.csv > ../../box_info.csv.tmp && mv ../../box_info.csv.tmp ../../box_info.csv
    cd ..
    rm -rf "$dirRandom"
    
}
if [ -z "${maxNumberofProcess}" ]; # verify that a max number is set so thread don't go crazy
then
    exit
fi

boxpath=$noNoiseDirPath
line=0
echo "Generating noise on data $boxpath"

files=$(cat $boxinfo| awk -F "," '{print $1}')
for file in $files 
do
    source ../parameter #see if update in number of process
    echo "Processing $maxNumberofProcess files in paralel"
    # echo $file
    line=$((line+1))
    generate_noise $boxpath $file $line &
    if [ $(jobs | wc -l) -ge $maxNumberofProcess ]; then
        wait
    fi
done
wait
    