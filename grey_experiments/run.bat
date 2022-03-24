@echo off

echo "original alexnet"
python -c "import model_train; model_train.original_alexnet_main()" > original_alexnet.txt
echo "pretrained alexnet"
python -c "import model_train; model_train.pretrained_alexnet_main()" > pretrained_alexnet.txt
echo "pretrained sub alexnet"
python -c "import model_train; model_train.pretrained_sub_alexnet_main()" > pretrained_sub_alexnet.txt
echo "sub alexnet"
python -c "import model_train; model_train.sub_alexnet_main()" > sub_alexnet.txt