#spile_data.py

from cgi import test
import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


file = 'archive'

with open(file + '/' +'name of the animals.txt') as f:
    lines = f.readlines()

cd_class = [l[:-1] for l in lines]

mkfile('data/train')
for cla in cd_class:
    mkfile('data/train/'+cla)

mkfile('data/val')
for cla in cd_class:
    mkfile('data/val/'+cla)

mkfile('data/test')
for cla in cd_class:
    mkfile('data/test/'+cla)

split_rate_val = 0.3
split_rate_test = 0.1

for cla in cd_class:
    cla_path = file + '/animals/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    non_train_index = random.sample(images, k=int(num*(split_rate_test+split_rate_val)))
    val_index = random.sample(non_train_index, k=int(len(non_train_index)*(split_rate_val/(split_rate_val+split_rate_test))))
    test_index = list(set(non_train_index) - set(val_index))


    for index, image in enumerate(images):
        if image in val_index:
            image_path = cla_path + image
            new_path = 'data/val/' + cla
            copy(image_path, new_path)
        elif image in test_index:
            image_path = cla_path + image
            new_path = 'data/test/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)

print("processing done!")