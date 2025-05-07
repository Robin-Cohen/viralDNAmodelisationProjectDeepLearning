# Write labels to a CSV file
csv_file="labels.csv"
echo "fileName,centerx,centery,centerz,rightHanded,radius,pitch,phi,theta0,theta1" > "$csv_file"
    
if [ "$1" == "" ]; then
        source ../parameter
else
        numProcess=$1
fi

for ((i=0; i<=numProcess; i++))
do
        bash generateMultiWithParam.sh >> out.log &
done
wait
# # #compile all log file into one
# echo "----------------------------------------------------------------------new generation----------------------------------------------------------------------" >> out.log
# for ((i=0; i<=$1; i++))
# do
#         echo "----------------------------------------------------------------------" >> out.log
#         cat out_$i.log >> out.log
# done
# rm out_*.b