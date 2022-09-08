import numpy as np
from tqdm import tqdm
class_samples = {}
# num_per_class = np.zeros((261255,))
for i in range(261255):
    class_samples[i] = []
with open('../split_path/train_split.txt', 'r') as t_f:
    for data in t_f.readlines():
        im_dir, cont_id = data.strip().split("---")
        class_samples[int(cont_id)].append(im_dir)
t_f.close()  
delete_class = []
remain_class = []
for i in sorted(list(class_samples.copy().keys())):
    if len(class_samples[i]) <= 5 :
        del class_samples[i]
        delete_class.append(i)

print(len(class_samples.keys()))
for k in class_samples.copy().keys():
    remain_class.append(int(k))
remain_class = sorted(list(set(remain_class)))
remain_class_dict = {}
for i, cls in enumerate(remain_class):
    remain_class_dict[cls] = i
with open('../split_path/train_split_reduce.txt', 'w+') as f_train:
    for k in class_samples.copy().keys():
        for v in list(class_samples[k]):
            f_train.write(v + '---' + str(remain_class_dict[k]) + '\n')

keep_test = []
with open('../split_path/test_split.txt', 'r') as t_f:
    for data in tqdm(t_f.readlines()):
        im_dir, cont_id = data.strip().split("---")
        if int(cont_id) in delete_class:
            continue
        else:
            keep_test.append([im_dir, cont_id])
with open('../split_path/test_split_reduce.txt', 'w+') as f_test:
    for d in keep_test:
        if int(d[1]) in remain_class_dict:
            f_test.write(d[0] + '---' + str(remain_class_dict[int(d[1])]) + '\n')
