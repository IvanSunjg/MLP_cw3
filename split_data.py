#spile_data.py

import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


file = 'data'
cd_class = ['CARS','CATS','DOGS','FACES','daisy','dandelion','roses','sunflowers','tulips']
#cd_class = ['daisy','dandelion','roses','sunflowers','tulips']
mkfile('data/train')
for cla in cd_class:
    mkfile('data/train/'+cla)

mkfile('data/val')
for cla in cd_class:
    mkfile('data/val/'+cla)

split_rate = 0.2
for cla in cd_class:
    cla_path = file + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'data/val/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)

print("processing done!")