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
    train = json.load(f)
    train_images = train
test_images = []
with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/test_hico.json', "r") as f:
    test = json.load(f)
    test_images = test

# make a form of 117 X 117
rows,cols = (117,117)
arr = [[0 for i in range(cols)] for j in range(rows)]

# verb static
verb_static = {}

####  By Samples  ####
with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/trainval_hico.json', "r") as f:
    train = json.load(f)
    for item in train:
        verb_queue = []
        for anno in item["hoi_annotation"]:
            hoi_label = anno["hoi_category_id"]
            verb_queue.append(hico_text_label_list[hoi_label-1][0])

            if (str(hico_text_label_list[hoi_label-1][0]) not in verb_static.keys()):
                verb_static[str(hico_text_label_list[hoi_label-1][0])] = 1
            else:
                verb_static[str(hico_text_label_list[hoi_label-1][0])] +=1

        front = 0
        back = len(verb_queue)-1
        for i in range(len(verb_queue)):
            for j in range(len(verb_queue)):
                arr[verb_queue[front+i]][verb_queue[back-j]] += 1
        


with open('/home/nii/Desktop/sin/HOICLIP/data/hico_20160224_det/annotations/test_hico.json', "r") as f:
    test = json.load(f)
    for item in test:

        obj_info = [i["category_id"] for i in item["annotations"]]
        verb_queue = []
        for anno in item["hoi_annotation"]:
            obj_id = anno["object_id"]
            verb_id = anno["category_id"]-1
            verb_queue.append(verb_id)

            if (str(verb_id) not in verb_static.keys()):
                verb_static[str(verb_id)] = 1
            else:
                verb_static[str(verb_id)] += 1

        front = 0
        back = len(verb_queue)-1
        for i in range(len(verb_queue)):
            for j in range(len(verb_queue)):
                arr[verb_queue[front+i]][verb_queue[back-j]] += 1

verb_non_zero = []
for i in range(117):
    arr[i][i] = 0
    for j in range(117):
        if (i+j >116): break
        arr[i][i+j] - arr[i][i+j] + arr[i+j][i]
        arr[i+j][i] = arr[i][i+j]

        if (arr[i][i+j] !=0):
            verb_non_zero.append((f'({i},{i+j})', arr[i][i+j]))

verb_static = dict(reversed(sorted(verb_static.items(), key=lambda x:x[1])))
verb_proportion = {}
num_verb = sum(verb_static.values())
num_20 = 0.5 * num_verb

keys = [str(key) for key in verb_static.keys()]
accumulate = 0
for key in keys:
    values = verb_static[key]
    accumulate += values
    verb_proportion[key] = accumulate / num_verb

# vals = [verb_proportion[k] for k in keys]
vals = [verb_static[k] for k in keys]
data = {'verb id': keys,
    'samples': vals,
}
# print(verb_proportion)
# plt.gcf().set_size_inches(70,10)
# plt.plot(keys, vals, marker='.', linestyle='none', color='tab:blue')
# plt.savefig(f'verb_accu.png')
# plt.close()
# show_verb = [i[0] for i in verb_proportion.items() if i[1] < 0.95 and i[1] >= 0.85]
# print(show_verb)
# exit()
sns.violinplot(data, cut=0)
plt.savefig(f'55-90verb_qua.png')
plt.close()

# arr = np.array(arr)
# sns.heatmap(arr)
# plt.savefig(f'verb_heat_map.png')
# plt.close()

verb_dict = {}
for item in verb_non_zero:
    verb_dict[item[0]] = item[1]

# Case 1
############################################
# top_group = [
#     '36', '76', '87'
# ] # 4

# rare_group = [
#     '109', '111', '99', '9', '15', 
#     '79', '104', '67', '16', '32', 
#     '75', '52', '47', '3', '44', 
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
# ] # 85

# general_group = [
#     '8', '98', '93', '114', '86', 
#     '24', '43', '112', '41', '73', 
#     '21', '94', '115', '30', '110', 
#     '58', '23', '77', '34', '49', 
#     '72', '39', '4', '26', '78', 
#     '65', '70', '28']

# verb_list = sorted(verb_static.items(), key=lambda x:x[1])
# verb_dict = dict(verb_list)
# keys = list(verb_dict.keys())
# vals = [verb_dict[k] for k in keys]
# verb_dict['group'] = []

# for item in verb_list:
#     if (str(item[0]) in top_group): verb_dict['group'].append('top')
#     elif (str(item[0]) in rare_group): verb_dict['group'].append('rare')
#     else: verb_dict['group'].append('general')

#     # verb_dist = pd.DataFrame.from_dict(verb_dist)

    
# data = {'verb id': keys,
#     'samples': vals,
#     'colors': verb_dict['group']}

# plt.gcf().set_size_inches(50,10)
# sns.barplot(data, x='verb id', y='samples', hue='colors').set_title('HOI-DET verb distribution')

# plt.savefig(f'verb_bar.png')
# plt.close()

# print(arr)
