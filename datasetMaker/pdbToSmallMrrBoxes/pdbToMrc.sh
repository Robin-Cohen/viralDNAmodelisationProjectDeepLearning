mkdir -p fullMrcData
cd fullMrcData
for file in /home/robin/viralDNAmodelisationProjectDeepLearning/datasetMaker/dataPointMaker/data/*
do
    base=$(basename "$file" .pdb)
    output_file="${base}.mrc"
    cat ../pdb2volParameter| pdb2vol $file $output_file &
done
wait
python ../downsize.py -output_dir ../boxData *.mrc
#To do : parallelise the process

# dirRandom="tmp$RANDOM"
# mkdir "$dirRandom/"
# cd "$dirRandom/"
# mkdir -p ../data
# #one call wil generate 10 random spiralDNA
# for i in {1..10}
# do
#     center="$(generate_random_number -20 20).00 $(generate_random_number -20 20).00 $(generate_random_number -20 20).00"
#     radius=$(generate_random_number 1 50)
#     cp ../workParameter/* .
#     sed -i "s/spiralDNA center i/spiralDNA center $center/" "command.generateSpiral.dat"
#     sed -i "s/spiralDNA radius j/spiralDNA radius 0.5/" "command.generateSpiral.dat"
#     sed -i "s/spiralDNA radius j/spiralDNA radius $radius\.00/" "command.generateSpiral.dat"
#     MMB -c command.generateSpiral.dat 
#     # cat command.generateSpiral.dat
#     center=$(echo "$center" | sed 's/ /_/g')
#     mv last.2.pdb ../data/"spiralDNAcenter$center-radius$radius.pdb"
#     # mv last.2.pdb ../data/"spiralDNAcenter$center-radius0.5.pdb"
    
# done
# cd ..
# rm -rf "$dirRandom"