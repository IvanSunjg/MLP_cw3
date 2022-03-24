@echo off

: base model
: echo "Original AlexNet"
: python original_alexnet.py > alexnet.txt
: echo "Original EfficientNet"
: python original_efficientnet.py > efficientnet.txt

: super model
: echo "Super AlexNet with mid_units 10"
: python super_alexnet.py 10 > super_alexnet_10.txt
echo "Super AlexNet with mid_units 50"
python super_alexnet.py 50 300 > super_alexnet_50.txt
: echo "Super AlexNet with mid_units 100"
: python super_alexnet.py 100 > super_alexnet_100.txt
: echo "Super AlexNet with mid_units 150"
: python super_alexnet.py 150 > super_alexnet_150.txt
: echo "Super AlexNet with mid_units 200"
: python super_alexnet.py 200 > super_alexnet_200.txt

echo "Super EfficientNet"
python super_efficientnet.py 300 > super_efficientnet.txt