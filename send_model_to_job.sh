#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:2
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

# base model
echo "Original AlexNet"
python original_alexnet.py > alexnet.txt
echo "Original EfficientNet"
pthon original_efficientnet.py > efficientnet.txt

# super model
echo "Super AlexNet with mid_units 10"
python super_alexnet.py 10 > super_alexnet_10.txt
echo "Super AlexNet with mid_units 50"
python super_alexnet.py 50 > super_alexnet_50.txt
echo "Super AlexNet with mid_units 100"
python super_alexnet.py 100 > super_alexnet_100.txt
echo "Super AlexNet with mid_units 150"
python super_alexnet.py 150 > super_alexnet_150.txt
echo "Super AlexNet with mid_units 200"
python super_alexnet.py 200 > super_alexnet_200.txt

echo "Super EfficientNet"
python super_efficientnet.py > super_efficientnet.txt