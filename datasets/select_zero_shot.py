import os
import sys
import random
sys.path.append(os.getcwd())

from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
import json

# original traininig datas: 38118
# original test datas: 9658


train_datas = {}
test_datas = {}

random.seed(3)

valid_obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                82, 84, 85, 86, 87, 88, 89, 90]

hico_text_label_list = [key for key in hico_text_label.keys()]

####  By Samples  ####
with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/trainval_hico.json', "r") as f:
    train = json.load(f)
    for item in train:
        for anno in item["hoi_annotation"]:
            if(anno["hoi_category_id"] in train_datas.keys()):
                train_datas[anno["hoi_category_id"]]+=1
            else:
                train_datas[anno["hoi_category_id"]] = 1

with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/test_hico.json', "r") as f:
    test = json.load(f)
    for item in test:

        obj_info = [i["category_id"] for i in item["annotations"]]
        for anno in item["hoi_annotation"]:
            obj_id = anno["object_id"]
            verb_id = anno["category_id"]
            pair_id = (verb_id -1, valid_obj_ids.index(obj_info[obj_id]))
            hoi_id = hico_text_label_list.index(pair_id)+1

            if(hoi_id in test_datas.keys()):
                test_datas[hoi_id]+=1
            else:
                test_datas[hoi_id] = 1

###  By Image  ###
train_images = []
with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/trainval_hico.json', "r") as f:
    train = json.load(f)
    train_images = train
test_images = []
with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/test_hico.json', "r") as f:
    test = json.load(f)
    test_images = test

obj_idx = [i for i in valid_obj_ids]
verb_idx = [i for i in range(1, 118)]

for random_times in range(1000):

    test_object_choice = random.sample(obj_idx, 20)
    test_verb_choice = random.sample(verb_idx, 29)
    train_object_choice = [i for i in valid_obj_ids if i not in test_object_choice]
    train_verb_choice = [i for i in range(1, 118) if i not in test_verb_choice]


    test_set_labels = []
    for obj in test_object_choice:
        for verb in test_verb_choice:
            if (verb-1, valid_obj_ids.index(obj)) in hico_text_label_list:
                test_set_labels.append(hico_text_label_list.index((verb-1, valid_obj_ids.index(obj))) +1)
    # print(test_set_labels)

    train_set_labels = []
    for obj in train_object_choice:
        for verb in train_verb_choice:
            if (verb-1, valid_obj_ids.index(obj)) in hico_text_label_list:
                train_set_labels.append(hico_text_label_list.index((verb-1, valid_obj_ids.index(obj))) +1)

    total_test_number = 0
    for each_id in test_set_labels:
        total_test_number += test_datas[each_id]

    total_train_number = 0
    for each_id in train_set_labels:
        total_train_number += train_datas[each_id]

    # if (total_test_number/total_train_number > 0.1):
        # print("Test Set: ", f'Selected samples: {total_test_number}   ', f'Total Samples: {sum(test_datas.values())}   ', f'Picked rate: {total_test_number/sum(test_datas.values()):.2f}')
        # print("Train Set: ", f'Selected samples: {total_train_number}   ', f'Total Samples: {sum(train_datas.values())}   ', f'Picked rate: {total_train_number/sum(train_datas.values()):.2f}')
        # print("Propotion of split: ", f'{total_test_number/total_train_number:.2f}')
        # print("The Samples we throw out: ", sum(test_datas.values()) + sum(train_datas.values()) - total_test_number - total_train_number)

    
    ###  By images  ###
    train_by_image = []
    for item in train_images:
        valid = True
        for hoi_item in item["hoi_annotation"]:
            if (hoi_item["hoi_category_id"] in test_set_labels):
                valid = False
                break
        if (valid): train_by_image.append(item)
        Valid = True

    test_by_image = []
    for item in test_images:
        valid = True
        obj_info = [i["category_id"] for i in item["annotations"]]
        for anno in item["hoi_annotation"]:
            obj_id = anno["object_id"]
            verb_id = anno["category_id"]
            pair_id = (verb_id -1, valid_obj_ids.index(obj_info[obj_id]))
            hoi_id = hico_text_label_list.index(pair_id)+1

            if (hoi_id in train_set_labels):
                valid = False
                break
        if (valid): test_by_image.append(item)    
        valid = True

    print("Test Set: ", f'Selected images: {len(test_by_image)}   ', f'Total images: {len(test_images)}   ', f'Picked rate: {len(test_by_image)/len(test_images):.2f}')
    print("Train Set: ", f'Selected images: {len(train_by_image)}   ', f'Total Samples: {len(train_images)}   ', f'Picked rate: {len(train_by_image)/len(train_images):.2f}')
    print("Propotion of split: ", f'{len(test_by_image)/len(train_by_image):.2f}')
    print("The Samples we throw out: ", len(test_images) + len(train_images) - len(test_by_image) - len(train_by_image))
    


