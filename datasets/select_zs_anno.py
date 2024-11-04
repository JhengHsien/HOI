import os
import sys
import random
sys.path.append(os.getcwd())

from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
import json
import matplotlib.pylab as plt
import csv
import seaborn as sns


def distr(images_set, mode=None, HOI_dist = None, verb_dist = None):
    distribution = range(1,601)
    
    for i in distribution:
        if (str(i) not in HOI_dist.keys()):
            HOI_dist[str(i)] = 0
    distribution = range(0,117)
    for i in [i for i in range(1, 118)]:
        if (str(i) not in verb_dist.keys()):
            verb_dist[str(i)] = 0

    if (mode == "train"):
        for item in images_set:

            for hoi_item in item["hoi_annotation"]:
                hoi_label = hoi_item["hoi_category_id"]
                verb_id = hico_text_label_list[hoi_label-1][0]+1
                if (verb_id == 58):continue
                verb_dist[str(verb_id)] +=1
            
                hoi_idx = hoi_label
                HOI_dist[str(hoi_idx)] +=1

    else:
        for item in images_set:
            obj_info = [i["category_id"] for i in item["annotations"]]
            for anno in item["hoi_annotation"]:
                obj_id = anno["object_id"]
                verb_id = anno["category_id"]
                if (verb_id == 58):continue
                pair_id = (verb_id -1, valid_obj_ids.index(obj_info[obj_id]))
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
    plt.gcf().set_size_inches(70,10)
    sns.barplot(x=keys, y=vals)

    plt.savefig(f'data_split/unseen_obj/train_verb_distribution.png')
    plt.close()


    # print(sorted(verb_dist.items(), key=lambda x:x[1]))
    # exit()
    
    top_verb_result_lists = sorted(verb_dist.items(), key=lambda x:x[1])[-3:]
    medium_verb_result_lists = sorted(verb_dist.items(), key=lambda x:x[1])[54:-3]
    bottom_verb_result_lists = sorted(verb_dist.items(), key=lambda x:x[1])[20:54]
    rare_verb_result_lists = sorted(verb_dist.items(), key=lambda x:x[1])[:20]
    # print(verb_result_lists[:20], verb_result_lists[-20:-1]) # most popular verbs/
    # verb_result_lists = dict(verb_result_lists)
    return top_verb_result_lists, medium_verb_result_lists, bottom_verb_result_lists, rare_verb_result_lists

obj_dict = {}
hico_text_label_list = [key for key in hico_text_label.keys()]
for each_pairs in hico_text_label_list:
    if (each_pairs[0])  not in obj_dict.keys():
        obj_dict[each_pairs[0]] = {}
        obj_dict[each_pairs[0]][each_pairs[1]] =1
    else:
        if (each_pairs[1]) not in obj_dict[each_pairs[0]].keys():
            obj_dict[each_pairs[0]][each_pairs[1]] =1
        else:
            obj_dict[each_pairs[0]][each_pairs[1]] +=1

# for each_obj in obj_dict:
#     keys = list(obj_dict[each_obj].keys())
#     vals = [obj_dict[each_obj][k] for k in keys]
#     sns.barplot(x=keys, y=vals)
#     plt.savefig(f'data_split/unseen_obj/{each_obj}.png')
#     plt.close()

random.seed(3)

valid_obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                82, 84, 85, 86, 87, 88, 89, 90]
verb_id = [i for i in range(1,118)]

with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/trainval_hico.json', "r") as f:
    train = json.load(f)
    train_images = train
test_images = []
with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/test_hico.json', "r") as f:
    test = json.load(f)
    test_images = test

HOI_dist = {}
verb_dist = {}
top_verb, medium_verb, bottom_verb, rare_verb = distr(test, mode="test", HOI_dist=HOI_dist, verb_dist=verb_dist)
# distr(test, mode="test", HOI_dist=HOI_dist, verb_dist=verb_dist)
top_obj_cand = []
medium_obj_cand = []
botom_obj_cand = []
rare_obj_cand = []

top_verb_cand = dict(random.sample(top_verb, 1))
medium_verb_cand = dict(random.sample(medium_verb, 11))
botom_verb_cand = dict(random.sample(bottom_verb, 13))
rare_verb_cand = dict(random.sample(rare_verb, 4))

all_test_verb = set(list(top_verb_cand)) | set(list(medium_verb_cand)) | set(list(botom_verb_cand)) | set(list(rare_verb_cand))
print(len(valid_obj_ids))
print(len(all_test_verb))
print(all_test_verb)
# Case 1: consider unseen objects & unseen verbs
for each_triplet in hico_text_label_list:
    if (str(each_triplet[0]+1) in top_verb_cand.keys() and str(each_triplet[0]+1) != '58'):
        if (each_triplet[1] not in top_obj_cand):
            top_obj_cand.append(each_triplet[1])
    if (str(each_triplet[0]+1) in medium_verb_cand.keys() and str(each_triplet[0]+1) != '58'):
        if (each_triplet[1] not in medium_obj_cand):
            medium_obj_cand.append(each_triplet[1])
    if (str(each_triplet[0]+1) in botom_verb_cand.keys() and str(each_triplet[0]+1) != '58'):
        if (each_triplet[1] not in botom_obj_cand):
            botom_obj_cand.append(each_triplet[1])
    if (str(each_triplet[0]+1) in rare_verb_cand.keys() and str(each_triplet[0]+1) != '58'):
        if (each_triplet[1] not in rare_obj_cand):
            rare_obj_cand.append(each_triplet[1])

common_obj = set(top_obj_cand) & set(medium_obj_cand) & set(botom_obj_cand) & set(rare_obj_cand)



# # remains_obj = 

print(len(common_obj))

others_verb = [i for i in verb_id if str(i) not in all_test_verb]
# # print(len(others_verb))
# train_obj_cand = []

# # # Case 1: consider unseen objects & unseen verbs
# for each_triplet in hico_text_label_list:
#     if (str(each_triplet[0]+1) in others_verb and str(each_triplet[0]+1) != '58'):
#         if (each_triplet[1] not in top_obj_cand):
#             train_obj_cand.append(each_triplet[1])
    
# test_annotation = []
# delete_count = 0
# all_anno = 0
# for item in test:
#     obj_info = [i["category_id"] for i in item["annotations"]]
#     for anno in item["hoi_annotation"]:
#         all_anno +=1
#         obj_id = anno["object_id"]
#         verb_id = anno["category_id"]
#         pair_id = (verb_id -1, valid_obj_ids.index(obj_info[obj_id]))
#         # hoi_id = hico_text_label_list.index(pair_id)+1

#         if (obj_info[obj_id] in train_obj_cand or verb_id in others_verb):
#             del anno
#             delete_count +=1
# print(all_anno)
# print(delete_count)
# all_anno = 0
# delete_count = 0
# for item in train_images:
#     for hoi_item in item["hoi_annotation"]:
#         all_anno +=1
#         hoi_label = hoi_item["hoi_category_id"]
#         if (valid_obj_ids[hico_text_label_list[hoi_label-1][1]] in common_obj or hico_text_label_list[hoi_label-1][0]+1 in all_test_verb):
#             del hoi_item
#             delete_count +=1
# print(all_anno)
# print(delete_count)
# # print(top_obj_cand)
# # remains_obj = 