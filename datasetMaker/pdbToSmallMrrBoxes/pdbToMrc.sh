mkdir -p fullMrcData
mkdir -p boxData
cd fullMrcData
echo "spliting into mrc box"
for file in ../../dataPointMaker/data/*
do
    base=$(basename "$file" .pdb)
    output_file="${base}.mrc"
    cat ../pdb2volParameter| pdb2vol $file $output_file > ../log.txt &
    # echo "pdb2vol $file $output_file"

    python ../downsize.py -output_dir ../boxData/ -input_pdb $file -input_mrc $output_file
    echo "python ../downsize.py -output_dir ../boxData/ -input_pdb $file -input_mrc $output_file"
done
wait
echo "done spliting"
rm -rf fullMrcData

cat ../pdb2volParameter| pdb2vol /home/robin/viralDNAmodelisationProjectDeepLearning/datasetMaker/dataPointMaker/dataCylinder/spiralDNAcenter0.00_0.00_0.00-radius5-pitch1.0.pdb -input_mrc /spiralDNAcenter0.00_0.00_0.00-radius5-pitch1.0.mrc test.mrc
python ../downsize.py -output_dir ../boxData/ -input_pdb /home/robin/viralDNAmodelisationProjectDeepLearning/datasetMaker/dataPointMaker/dataCylinder/spiralDNAcenter0.00_0.00_0.00-radius5-pitch1.0.pdb -input_mrc test.mrc
