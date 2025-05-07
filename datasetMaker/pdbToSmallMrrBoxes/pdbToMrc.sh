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