for i in 10, 50, 100, 150, 200 do
    #echo 'Super AlexNet with mid_units ${i}'
    python test.py $i > super_alexnet_$i.txt
done