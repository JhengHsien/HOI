import os
import sys
import random
sys.path.append(os.getcwd())

from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
import json
import matplotlib.pylab as plt
import csv
import seaborn as sns
import numpy as np
import copy

valid_obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                82, 84, 85, 86, 87, 88, 89, 90]

hico_text_label_list = [key for key in hico_text_label.keys()]

with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/trainval_hico.json', "r") as f:
    copy_train = json.load(f)
    train = copy.deepcopy(copy_train)

with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/test_hico.json', "r") as f:
    copy_test = json.load(f)
    test = copy.deepcopy(copy_test)

# Case 1
############################################
top_group = [
    '36', '76', '87'
] # 4

rare_group = [
    '109', '111', '99', '9', '15', 
    '79', '104', '67', '16', '32', 
    '75', '52', '47', '3', '44', 
    '108', '20', '54', '6', '35', 
    '101', '14', '85', '17', '2', 
    '107', '45', '103', '62', '71', 
    '48', '18', '25', '19', '80', 
    '102', '50', '83', '38', '11', 
    '10', '106', '59', '7', '105', 
    '0', '27', '89', '13', '64', 
    '95', '81', '40', '66', '68', 
    '37', '12', '96', '46', '1', 
    '5', '69', '33', '82', '55', 
    '60', '51', '97', '90', '63', 
    '31', '74', '42', '61', '53', 
    '88', '22', '84', '91', '56', 
    '92', '116', '100', '29', '113'
] # 85

general_group = [
    '8', '98', '93', '114', '86', 
    '24', '43', '112', '41', '73', 
    '21', '94', '115', '30', '110', 
    '58', '23', '77', '34', '49', 
    '72', '39', '4', '26', '78', 
    '65', '70', '28'
] # 28


###########################################

# Case 2
# 55 - 95 - 5
# top_group = [
#     '36', '76', '87'
# ] # 4

# rare_group = [
#     '108', '20', '54', '6', '35', 
#     '101', '14', '85', '17', '2', 
#     '107', '45', '103', '62', '71', 
#     '48', '18', '25', '19', '80', 
#     '102', '50', '83', '38', '11', 
#     '10', '106', '59', '7', '105', 
#     '0', '27', '89', '13', '64', 
#     '95', '81', '40', '66', '68', 
#     '37', '12', '96', '46', '1', 
#     '5', '69', '33', '82', '55', 
#     '60', '51', '97', '90', '63', 
#     '31', '74', '42', '61', '53', 
#     '88', '22', '84', '91', '56', 
#     '92', '116', '100', '29', '113'
# ] # 70

# general_group = [
#     '8', '98', '93', '114', '86', 
#     '24', '43', '112', '41', '73', 
#     '21', '94', '115', '30', '110', 
#     '58', '23', '77', '34', '49', 
#     '72', '39', '4', '26', '78', 
#     '65', '70', '28', '109', '111', 
#     '99', '9', '15', '79', '104', 
#     '67', '16', '32', '75', '52', 
#     '47', '3', '44'
# ] # 43

# # Case 3: 
# "General" means: almost class should be there
# right? 

# top_group = [
#     '36', '76', '87'
# ] # 4

# rare_group = [
#     '13', '64', '95', '81', '40', 
#     '66', '68', '37', '12', '96', 
#     '46', '1', '5', '69', '33', 
#     '82', '55', '60', '51', '97', 
#     '90', '63', '31', '74', '42', 
#     '61', '53', '88', '22', '84', 
#     '91', '56', '92', '116', '100', 
#     '29', '113'
# ] # 37

# general_group = [
#     '8', '98', '93', '114', '86', 
#     '24', '43', '112', '41', '73', 
#     '21', '94', '115', '30', '110', 
#     '58', '23', '77', '34', '49', 
#     '72', '39', '4', '26', '78', 
#     '65', '70', '28', '109', '111', 
#     '99', '9', '15', '79', '104', 
#     '67', '16', '32', '75', '52', 
#     '47', '3', '44', '108', '20', 
#     '54', '6', '35', '101', '14', 
#     '85', '17', '2', '107', '45', 
#     '103', '62', '71', '48', '18', 
#     '25', '19', '80', '102', '50', 
#     '83', '38', '11', '10', '106', 
#     '59', '7', '105', '0', '27', 
#     '89'
# ] # 76


# Case 4:
# Split by every 5%

# top_55 = [
#     '36', '76', '87'
# ] # 4

# general_55_75 = [
#     '8', '98', '93', '114', '86', 
#     '24', '43'
# ] # 7

# general_75_85 = [
#     '112', '41', '73', '21', '94', 
#     '115', '30', '110', '58', '23', 
#     '77'
# ] # 11

# general_85_95 = [
#     '34', '49', '72', '39', '4', 
#     '26', '78', '65', '70', '28', 
#     '109', '111', '99', '9', '15', 
#     '79', '104', '67', '16', '32', 
#     '75', '52', '47', '3', '44'
# ] # 25

# rare_group = [
#     '13', '64', '95', '81', '40', 
#     '66', '68', '37', '12', '96', 
#     '46', '1', '5', '69', '33', 
#     '82', '55', '60', '51', '97', 
#     '90', '63', '31', '74', '42', 
#     '61', '53', '88', '22', '84', 
#     '91', '56', '92', '116', '100', 
#     '29', '113'
# ] # 37

## 75:25 labels ##
# train: 88 test: 29 #

random.seed(3)

# case 1:
test_verbs = []
train_verbs = [str(i) for i in range(117)]

top = random.sample(top_group, 1)
general = random.sample(general_group, 7)
rare = random.sample(rare_group, 21)

test_verbs = top + general + rare
train_verbs = [i for i in train_verbs if i not in test_verbs and i != str(57)]

hoi_label_idx = []
for i in train_verbs:
    for j in hico_text_label_list:
        if i == str(j[0]):
            hoi_label_idx.append(hico_text_label_list.index(j))
for j in hico_text_label_list:
        if str(j[0]) == "57":
            hoi_label_idx.append(hico_text_label_list.index(j))
print(hoi_label_idx)
print(len(hoi_label_idx))
exit()

train_annotations = []
test_annotations = []

test_images_by_test = random.sample(test, 2386) # 0.75
train_images_by_test = [item for item in test if item not in test_images_by_test]


test_images_by_train = random.sample(train, 9408) # 0.25
train_images_by_train = [item for item in train if item not in test_images_by_train]

test_recycle = []
train_recycle = []

# test annotations
all_annotations = 0
remove_annotations = 0
test_images_total = 0

for item_idx, item in enumerate(test_images_by_test):
    recycle_bin = []
    test_images_total +=1
    all_annotations += len(item["hoi_annotation"])
    for anno in item["hoi_annotation"]:
        obj_id = anno["object_id"]
        verb_id = anno["category_id"]-1

        if (str(verb_id) in train_verbs):
            recycle_bin.append(anno)
            remove_annotations +=1
    for anno in recycle_bin:
        test_images_by_test[item_idx]["hoi_annotation"].remove(anno)
    test_annotations.append(test_images_by_test[item_idx])

for item_idx, item in enumerate(test_images_by_train):
    test_images_total +=1
    recycle_bin = []
    all_annotations += len(item["hoi_annotation"])
    for anno in item["hoi_annotation"]:
        hoi_label = anno["hoi_category_id"]
        verb_id = hico_text_label_list[hoi_label-1][0]
        if (str(verb_id) in train_verbs):
            recycle_bin.append(anno)
            remove_annotations +=1
    # for anno in recycle_bin:
    #     test_images_by_train[item_idx]["hoi_annotation"].remove(anno)
    test_annotations.append(test_images_by_train[item_idx])

after_remove_annotations = 0
empty = 0
remove_list = []
for item in test_annotations:
    after_remove_annotations += len(item["hoi_annotation"])
    if (len(item["hoi_annotation"]) == 0): 
        empty +=1
#         test_recycle.append(item["file_name"])
#         remove_list.append(item)
# for item in remove_list:
#     test_annotations.remove(item)

print("------- test set analysis------- ")
print("total images: ", len(test_images_by_test) + len(test_images_by_train))
print("All annotations: ", all_annotations)
# print("Remove annotations: ", remove_annotations)
# print("After removing annotations ", after_remove_annotations)
print("Empty annotation: ", empty)

# train annotations
all_annotations = 0
remove_annotations = 0
train_images_total = 0

for item_idx, item in enumerate(train_images_by_test):
    train_images_total +=1
    recycle_bin = []
    all_annotations += len(item["hoi_annotation"])
    for anno in item["hoi_annotation"]:
        obj_id = valid_obj_ids.index(item["annotations"][anno["object_id"]]["category_id"])
        verb_id = anno["category_id"]-1
        index = hico_text_label_list.index((verb_id, obj_id)) +1
        anno["hoi_category_id"] = index
        if (str(verb_id) in test_verbs):
            recycle_bin.append(anno)
            remove_annotations +=1
    for anno in recycle_bin:
        train_images_by_test[item_idx]["hoi_annotation"].remove(anno)
    train_annotations.append(train_images_by_test[item_idx])

for item_idx, item in enumerate(train_images_by_train):
    train_images_total +=1
    recycle_bin = []
    all_annotations += len(item["hoi_annotation"])
    for anno in item["hoi_annotation"]:
        hoi_label = anno["hoi_category_id"]
        verb_id = hico_text_label_list[hoi_label-1][0]
        if (str(verb_id) in test_verbs):
            recycle_bin.append(anno)
            remove_annotations +=1
    for anno in recycle_bin:
        train_images_by_train[item_idx]["hoi_annotation"].remove(anno)
    train_annotations.append(train_images_by_train[item_idx])

after_remove_annotations = 0
empty = 0
remove_list = []
for item in train_annotations:
    after_remove_annotations += len(item["hoi_annotation"])
    if (len(item["hoi_annotation"]) == 0):
        empty +=1
        train_recycle.append(item["file_name"])
        remove_list.append(item)
for item in remove_list:
    train_annotations.remove(item)

print("------- train set analysis------- ")
print("total images: ", len(train_images_by_test) + len(train_images_by_train))
print("All annotations: ", all_annotations)
print("Remove annotations: ", remove_annotations)
print("After remove ", after_remove_annotations)
print("Empty annotation: ", empty)

# recycle
print("------ Exchange empty images: ----------")
print('test set before exchanging: ', len(test_annotations))
print('train set before exchanging: ', len(train_annotations))
for item in copy_test:
    if item["file_name"] in train_recycle:
        test_annotations.append(item)
    # if item["file_name"] in test_recycle:
    #     for anno in item["hoi_annotation"]:
    #         obj_id = valid_obj_ids.index(item["annotations"][anno["object_id"]]["category_id"]) # WTF is that annotations = = 
    #         verb_id = anno["category_id"]-1
    #         index = hico_text_label_list.index((verb_id, obj_id)) + 1
    #         anno["hoi_category_id"] = index
    #     train_annotations.append(item)

for item in copy_train:
    if item["file_name"] in train_recycle:
        test_annotations.append(item)
    # if item["file_name"] in test_recycle:
    #     train_annotations.append(item)
       
all_annotations = 0
empty = 0
for item in test_annotations:
    all_annotations += len(item["hoi_annotation"])
    if (len(item["hoi_annotation"]) == 0): 
        empty +=1

print('test set after exchanging: ', len(test_annotations))
print('train set after exchanging: ', len(train_annotations))
# print(all_annotations, empty)
# all_annotations = 0
# empty = 0
# for item in train_annotations:
#     all_annotations += len(item["hoi_annotation"])
#     if (len(item["hoi_annotation"]) == 0):
#         empty +=1





# with open("data/hico_20160224_det/annotations/55_95_5/test_uv.json", "w") as outfile: 
#     json.dump(test_annotations, outfile)

# with open("data/hico_20160224_det/annotations/55_95_5/train_uv.json", "w") as outfile: 
#     json.dump(train_annotations, outfile)

# visulization
import pandas as pd
def distr(images_set, mode=None, HOI_dist = None, verb_dist = None):
    distribution = range(1,601)
    
    for i in distribution:
        if (str(i) not in HOI_dist.keys()):
            HOI_dist[str(i)] = 0
    distribution = range(0,117)
    for i in [i for i in range(0, 117)]:
        if (str(i) not in verb_dist.keys()):
            verb_dist[str(i)] = 0
    
    

    if (mode == "train"):
        for item in images_set:

            for hoi_item in item["hoi_annotation"]:
                hoi_label = hoi_item["hoi_category_id"]
                verb_id = hico_text_label_list[hoi_label-1][0]
                # if (verb_id == 58):continue
                verb_dist[str(verb_id)] +=1
            
                hoi_idx = hoi_label
                HOI_dist[str(hoi_idx)] +=1

    else:
        for item in images_set:
            obj_info = [i["category_id"] for i in item["annotations"]]
            for anno in item["hoi_annotation"]:
                obj_id = anno["object_id"]
                verb_id = anno["category_id"]-1
                # if (verb_id == 58): continue
                pair_id = (verb_id, valid_obj_ids.index(obj_info[obj_id]))
                hoi_id = hico_text_label_list.index(pair_id)+1

                HOI_dist[str(hoi_id)] +=1
                verb_dist[str(verb_id)] +=1


    hoi_result_lists = sorted(HOI_dist.items(), key=lambda x:x[1]) # sorted by key, return a list of tuples
    hoi_verb = []
    for i in hoi_result_lists:
        action_id = hico_text_label_list[int(i[0])-1]
        hoi_verb.append(action_id[0])

    verb_list = sorted(verb_dist.items(), key=lambda x:x[1])
    verb_dist = dict(verb_list)
    keys = list(verb_dist.keys())
    vals = [verb_dist[k] for k in keys]
    hoi_dict = dict(hoi_result_lists)
    verb_dist['group'] = []

    for item in verb_list:
        if (item[0] in top_group): verb_dist['group'].append('top')
        elif (item[0] in rare_group): verb_dist['group'].append('rare')
        else: verb_dist['group'].append('general')

    # verb_dist = pd.DataFrame.from_dict(verb_dist)

    
    data = {'verb id': keys,
        'samples': vals,
        'colors': verb_dist['group']}
    plt.gcf().set_size_inches(40,10)
    sns.barplot(data, x='verb id', y='samples', hue='colors').set_title('Set verb distribution of 75_25 labels split')
    if (mode == "test"):
        plt.savefig(f'data/hico_20160224_det/annotations/55_90_10/test_verb_distribution.png')
    else:
        plt.savefig(f'data/hico_20160224_det/annotations/55_90_10/train_verb_distribution.png')
    plt.close()

with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/55_90_10/train_uv.json', "r") as f:
    train = json.load(f)

with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/55_90_10/test_uv.json', "r") as f:
    test = json.load(f)

HOI_dist = {}
verb_dist = {}
distr(test, mode="test", HOI_dist=HOI_dist, verb_dist=verb_dist)
distr(train, mode="train", HOI_dist=HOI_dist, verb_dist=verb_dist)

exit()