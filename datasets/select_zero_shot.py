import os
import sys
import random
sys.path.append(os.getcwd())

from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
import json
import matplotlib.pylab as plt
import csv

# original traininig datas: 38118
# original test datas: 9658


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
                verb_dist[str(verb_id)] +=1
            
                hoi_idx = hoi_label
                HOI_dist[str(hoi_idx)] +=1

    else:
        for item in images_set:
            obj_info = [i["category_id"] for i in item["annotations"]]
            for anno in item["hoi_annotation"]:
                obj_id = anno["object_id"]
                verb_id = anno["category_id"]
                pair_id = (verb_id -1, valid_obj_ids.index(obj_info[obj_id]))
                hoi_id = hico_text_label_list.index(pair_id)+1

                HOI_dist[str(hoi_id)] +=1
                verb_dist[str(verb_id)] +=1

    hoi_result_lists = sorted(HOI_dist.items(), key=lambda x:x[1]) # sorted by key, return a list of tuples
    hoi_result_lists = dict(hoi_result_lists)

    verb_result_lists = sorted(verb_dist.items(), key=lambda x:x[1])
    verb_result_lists = dict(verb_result_lists)



    # fig, ax = plt.subplots(2)
    # ax[0].bar(list(hoi_result_lists.keys()), list(hoi_result_lists.values()))
    # ax[1].bar(list(verb_result_lists.keys()), list(verb_result_lists.values()))
    # plt.savefig(f'data_split/{mode}_distribution.png')
    # plt.close()

    # with open(f'{mode}_dist.csv', 'w') as csv_file:
    #     all_values = sum(hoi_result_lists.values())
    #     for key, value in hoi_result_lists.items():
    #         csv_file.write('{0},{1}\n'.format(key, value/all_values))

    



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
            # 1~600
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
verb_idx = [i for i in range(1, 118) if i != 58]

for random_times in range(1000):

    # calculate the distribution
    HOI_dist = {}
    verb_dist = {}
    distr(train, mode="train", HOI_dist=HOI_dist, verb_dist=verb_dist)
    distr(test, mode="test", HOI_dist=HOI_dist, verb_dist=verb_dist)
    verb_result_lists = sorted(verb_dist.items(), key=lambda x:x[1])

    # test_verb_choice = []
    # for step in range(0,len(verb_result_lists),4):
    #     if (int(verb_result_lists[step][0]) != 58):
    #         test_verb_choice.append(int(verb_result_lists[step][0]))
    # test_verb_choice.append(58)
    test_object_choice = random.sample(obj_idx, 32)
    test_verb_choice = random.sample(verb_idx, 46)
    train_object_choice = [i for i in valid_obj_ids if i not in test_object_choice]
    train_verb_choice = [i for i in range(1, 118) if i not in test_verb_choice and i != 58]
    # train_verb_choice.append(58)
    # print(train_verb_choice)
    # exit()


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
    #     print("Test Set: ", f'Selected samples: {total_test_number}   ', f'Total Samples: {sum(test_datas.values())}   ', f'Picked rate: {total_test_number/sum(test_datas.values()):.2f}')
    #     print("Train Set: ", f'Selected samples: {total_train_number}   ', f'Total Samples: {sum(train_datas.values())}   ', f'Picked rate: {total_train_number/sum(train_datas.values()):.2f}')
    #     print("Propotion of split: ", f'{total_test_number/total_train_number:.2f}')
    #     print("The Samples we throw out: ", sum(test_datas.values()) + sum(train_datas.values()) - total_test_number - total_train_number)

    
    ###  By images  ###
    train_by_image = []
    test_by_image = []
    # analysis each object and verb numbers
    train_obj_idx = {}
    test_obj_idx = {}
    train_verb_idx = {}
    test_verb_idx = {}
    train_hoi_idx = {}
    test_hoi_idx = {}

    # calculate the distribution
    HOI_dist = {}
    verb_dist = {}


    for item in train_images:
        valid = True
        for hoi_item in item["hoi_annotation"]:
            hoi_label = hoi_item["hoi_category_id"]
            if (valid_obj_ids[hico_text_label_list[hoi_label-1][1]] in test_object_choice or hico_text_label_list[hoi_label-1][0]+1 in test_verb_choice):
                valid = False
                break
        if (valid): 
            train_by_image.append(item)
        valid = True

        for hoi_item in item["hoi_annotation"]:
            hoi_label = hoi_item["hoi_category_id"]
            if (valid_obj_ids[hico_text_label_list[hoi_label-1][1]] in train_object_choice or hico_text_label_list[hoi_label-1][0]+1 in train_verb_choice):
                valid = False
                break
        if (valid): 
            test_by_image.append(item)
        valid = True
    
    for item in test_images:
        valid = True
        obj_info = [i["category_id"] for i in item["annotations"]]
        for anno in item["hoi_annotation"]:
            obj_id = anno["object_id"]
            verb_id = anno["category_id"]
            pair_id = (verb_id -1, valid_obj_ids.index(obj_info[obj_id]))
            # hoi_id = hico_text_label_list.index(pair_id)+1

            if (obj_info[obj_id] in train_object_choice or verb_id in train_verb_choice):
                valid = False
                break
        if (valid): test_by_image.append(item)    
        valid = True

        for anno in item["hoi_annotation"]:
            obj_id = anno["object_id"]
            verb_id = anno["category_id"]
            pair_id = (verb_id -1, valid_obj_ids.index(obj_info[obj_id]))
            # hoi_id = hico_text_label_list.index(pair_id)+1

            if (obj_info[obj_id] in test_object_choice or verb_id in test_verb_choice):
                valid = False
                break
        if (valid): train_by_image.append(item)    
        valid = True

    for item in train_by_image:
        for hoi_item in item["hoi_annotation"]:
            # from training set
            if ("hoi_category_id" in hoi_item.keys()):
                # verb
                if str(hico_text_label_list[hoi_item["hoi_category_id"]-1][0]) not in train_verb_idx.keys():
                    train_verb_idx[str(hico_text_label_list[hoi_item["hoi_category_id"]-1][0])] = 1
                else:
                    train_verb_idx[str(hico_text_label_list[hoi_item["hoi_category_id"]-1][0])] += 1

                # object
                if hico_text_label_list[hoi_item["hoi_category_id"]-1][1] not in train_obj_idx.keys():
                    train_obj_idx[hico_text_label_list[hoi_item["hoi_category_id"]-1][1]] = 1
                else:
                    train_obj_idx[hico_text_label_list[hoi_item["hoi_category_id"]-1][1]] += 1

                # HOI 
                if (str(hoi_item["hoi_category_id"]-1) not in train_hoi_idx.keys()):
                    train_hoi_idx[str(hoi_item["hoi_category_id"]-1)] = 1
                else:
                    train_hoi_idx[str(hoi_item["hoi_category_id"]-1)] += 1
            # from testing set
            else:
                obj_info = [i["category_id"] for i in item["annotations"]]
                for anno in item["hoi_annotation"]:
                    obj_id = anno["object_id"]
                    verb_id = anno["category_id"]
                    pair_id = (verb_id -1, valid_obj_ids.index(obj_info[obj_id]))
                    hoi_id = hico_text_label_list.index(pair_id)

                    # verb
                    if (str(verb_id+1) not in train_verb_idx.keys()):
                        train_verb_idx[str(verb_id+1)] = 1
                    else:
                        train_verb_idx[str(verb_id+1)] += 1


                    # object
                    if (valid_obj_ids.index(obj_info[obj_id]) not in train_obj_idx.keys()):
                        train_obj_idx[valid_obj_ids.index(obj_info[obj_id])] = 1
                    else:
                        train_obj_idx[valid_obj_ids.index(obj_info[obj_id])] += 1

                    # HOI
                    if (str(hoi_id) not in train_hoi_idx.keys()):
                        train_hoi_idx[str(hoi_id)] = 1
                    else:
                        train_hoi_idx[str(hoi_id)] += 1


    for item in test_by_image:
        if ("hoi_category_id" in hoi_item.keys()):
            # verb
            if str(hico_text_label_list[hoi_item["hoi_category_id"]-1][0]) not in test_verb_idx.keys():
                test_verb_idx[str(hico_text_label_list[hoi_item["hoi_category_id"]-1][0])] = 1
            else:
                test_verb_idx[str(hico_text_label_list[hoi_item["hoi_category_id"]-1][0])] += 1

            # object
            if hico_text_label_list[hoi_item["hoi_category_id"]-1][1] not in test_obj_idx.keys():
                test_obj_idx[hico_text_label_list[hoi_item["hoi_category_id"]-1][1]] = 1
            else:
                test_obj_idx[hico_text_label_list[hoi_item["hoi_category_id"]-1][1]] += 1

            # HOI 
            if (str(hoi_item["hoi_category_id"]-1) not in test_hoi_idx.keys()):
                test_hoi_idx[str(hoi_item["hoi_category_id"]-1)] = 1
            else:
                test_hoi_idx[str(hoi_item["hoi_category_id"]-1)] += 1
        else:
            obj_info = [i["category_id"] for i in item["annotations"]]
            for anno in item["hoi_annotation"]:
                obj_id = anno["object_id"]
                verb_id = anno["category_id"]
                pair_id = (verb_id -1, valid_obj_ids.index(obj_info[obj_id]))
                hoi_id = hico_text_label_list.index(pair_id)

                # verb
                if (str(verb_id+1) not in test_verb_idx.keys()):
                    test_verb_idx[str(verb_id+1)] = 1
                else:
                    test_verb_idx[str(verb_id+1)] += 1


                # object
                if (valid_obj_ids.index(obj_info[obj_id]) not in test_obj_idx.keys()):
                    test_obj_idx[valid_obj_ids.index(obj_info[obj_id])] = 1
                else:
                    test_obj_idx[valid_obj_ids.index(obj_info[obj_id])] += 1

                # HOI
                if (str(hoi_id) not in test_hoi_idx.keys()):
                    test_hoi_idx[str(hoi_id)] = 1
                else:
                    test_hoi_idx[str(hoi_id)] += 1


    print(f'Test set Selected images: {len(test_by_image)}') # Total images: {len(test_images)}  Picked rate: {len(test_by_image)/len(test_images):.2f}')
    print(f'Train set  Selected images: {len(train_by_image)}') # Total Samples: {len(train_images)}  Picked rate: {len(train_by_image)/len(train_images):.2f}')
    print(f'Propotion of split: {len(test_by_image)/len(train_by_image):.2f}')
    print(f'The images we throw out: {len(test_images) + len(train_images) - len(test_by_image) - len(train_by_image)}')
    
    # print(train_object_choice)
    # print(test_object_choice)
    # print(train_obj_idx.keys())
    # print([valid_obj_ids[key] for key in test_obj_idx.keys()])
    # plot results
    test_hoi_result_lists = sorted(test_hoi_idx.items(), key=lambda x:x[1]) # sorted by key, return a list of tuples
    # print(test_hoi_result_lists[-1])
    test_hoi_result_lists = dict(test_hoi_result_lists)
    train_hoi_result_lists = sorted(train_hoi_idx.items(), key=lambda x:x[1]) # sorted by key, return a list of tuples
    # print(train_hoi_result_lists[-1])
    train_hoi_result_lists = dict(train_hoi_result_lists)
    # print(hico_text_label_list[75])
    # exit()

    fig, ax = plt.subplots(2,2)
    ax[0][0].bar(list(test_hoi_result_lists.keys()), list(test_hoi_result_lists.values()))
    ax[0][1].bar(list(train_hoi_result_lists.keys()), list(train_hoi_result_lists.values()))
    
    ax[1][0].axis([0, 10, 0, 10])
    ax[1][1].axis([0, 10, 0, 10])
    ax[1][1].text(-15, 8, f'Train set Selected images: {len(train_by_image)}  Total Samples: {len(train_images)}  Picked rate: {len(train_by_image)/len(train_images):.2f}', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax[1][1].text(-15, 5, f'Test set Selected images: {len(test_by_image)}  Total images: {len(test_images)}  Picked rate: {len(test_by_image)/len(test_images):.2f}', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax[1][1].text(-15, 2, f'Propotion of split: {len(test_by_image)/len(train_by_image):.2f}', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax[1][1].text(-15, -2, f'The images we throw out: {len(test_images) + len(train_images) - len(test_by_image) - len(train_by_image)}', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    # plt.plot(list(train_obj_result_lists.keys()), list(train_obj_result_lists.values()))
    plt.savefig(f'data_split/randomly/full/60_40/60_40Labels_{random_times}.png')
    plt.close()


