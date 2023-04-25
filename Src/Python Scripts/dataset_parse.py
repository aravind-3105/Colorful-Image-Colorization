# Script to move images from subfolders to a single folder

import os
import shutil
import tqdm

source = '../../Datasets/mini_imagenet/imagenet-mini/val'
destination = '../../Datasets/mini_imagenet/imagenet-mini/test_images'
if not os.path.exists(destination):
    os.makedirs(destination)

for folder in tqdm.tqdm(os.listdir(source)):
    for file in os.listdir(source + '/' + folder):
        shutil.copy(source + '/' + folder + '/' + file, destination)
