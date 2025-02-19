for ((i=0; i<=$1; i++))
do
        bash generateMulti.sh > out_$i.log &
done
wait
#compile all log file into one
for ((i=0; i<=$1; i++))
echo "----------------------------------------------------------------------"new generation"----------------------------------------------------------------------" >> out.log
do
        echo "---------------------------------------------------------------------->> out.log
        cat out_$i.log >> out.log
done
rm out_*.log