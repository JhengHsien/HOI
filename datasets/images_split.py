import os
import shutil
import subprocess
import json

with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/55_90_10/train_uv.json', "r") as f:
    train = json.load(f)
with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/55_90_10/test_uv.json', "r") as f:
    test = json.load(f)

for item in train:
    img_path = item["file_name"]
    if (img_path[5:9] == "test"):
        
        original_prefix = "/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/images/test2015/"
        prefix = "/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/new_split/train/"
        shutil.copyfile(original_prefix + img_path, prefix + img_path)
    if (img_path[5:10] == "train"):
        
        original_prefix = "/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/images/train2015/"
        prefix = "/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/new_split/train/"
        shutil.copyfile(original_prefix + img_path, prefix + img_path)

for item in test:
    img_path = item["file_name"]
    if (img_path[5:9] == "test"):
        
        original_prefix = "/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/images/test2015/"
        prefix = "/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/new_split/test/"
        shutil.copyfile(original_prefix + img_path, prefix + img_path)
    if (img_path[5:10] == "train"):
        
        original_prefix = "/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/images/train2015/"
        prefix = "/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/new_split/test/"
        shutil.copyfile(original_prefix + img_path, prefix + img_path)
    
