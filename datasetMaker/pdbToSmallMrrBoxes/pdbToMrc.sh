mkdir -p fullMrcData
mkdir -p boxData
cd fullMrcData
for file in ../../dataPointMaker/data/*
do
    base=$(basename "$file" .pdb)
    output_file="${base}.mrc"
    cat ../pdb2volParameter| pdb2vol $file $output_file &
    python ../downsize.py -output_dir ../boxData/ -input_pdb $file -input_mrc $output_file 
done
wait
# python ../downsize.py -output_dir ../boxData *.mrc
rm -rf fullMrcData
