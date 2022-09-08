import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from shutil import copyfile
import json
import csv
import pandas as pd
import scipy.io as sio
import cv2
import random

def load_json_to_dict(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data

def get_args_parser():
    parser = argparse.ArgumentParser('Create split', add_help=False)
    parser.add_argument('--data_path', default='all', type=str)
    parser.add_argument('--split_ratio', default=0.04, type=float)
    
    return parser
def processing_gglandmarkv2(args):
    args.data_path =  '../gglandmark-v2-clean'
    df = pd.read_csv(os.path.join(args.data_path,'train.csv'))
    train_list = []
    val_list = []
    current_label = -1
    current_label_idx = -1
    labels = df['landmark_id'].values
    im_files = df['id'].values
    for idx in tqdm(np.argsort(labels)):
        imfile, label = im_files[idx], labels[idx]
        imfile = 'train/' + imfile[0] + '/' + imfile[1] + '/' + imfile[2] + '/' + imfile +'.jpg'
        if label > current_label :
            current_label_idx += 1
            current_label = label
            train_list.append([imfile, current_label_idx])
        else:
            if np.random.uniform() >= args.split_ratio:
                train_list.append([imfile, current_label_idx])
            else:
                val_list.append([imfile, current_label_idx])
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for train_data in tqdm(train_list):
        file, label = train_data
        f_train.write(args.data_path + '/' + file + '---' + str(label) + '\n')
    for test_data in tqdm(val_list):
        file, label = test_data
        f_test.write(args.data_path + '/' + file + '---' + str(label) + '\n')
    return 81313
def processing_food_recog_2022(args, start = 0):
    args.data_path =  '../food-recog-2022'
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for spilt in ['public_training_set_release_2.0', 'public_validation_set_2.0']:
        food_rec_meta = load_json_to_dict(args.data_path + "/raw_data/" + spilt + "/annotations.json")
        category_label_map = {x["id"]: ids + start for ids,x in enumerate(food_rec_meta["categories"])}
        imageid_category_map = {x["image_id"]: x["category_id"] for x in food_rec_meta["annotations"]}
        imageid_imagepath_map = {x["id"]:x["file_name"] for x in food_rec_meta['images']}
        for image_id in imageid_category_map.keys():
            file, label = imageid_imagepath_map[image_id], category_label_map[imageid_category_map[image_id]]
            file = "raw_data/" + spilt + '/images/' + file
            if np.random.uniform() >= args.split_ratio:
                f_train.write(args.data_path + '/' + file + '---' + str(label) + '\n')
            else:
                f_test.write(args.data_path + '/' + file + '---' + str(label) + '\n')
    return len(category_label_map.keys()) + start 

def processing_HMFashion(args,start = 0):
    args.data_path =  '../HMFashion'
    df = pd.read_csv(args.data_path +'/articles.csv')
    num_label = 0
    all_possible_images = df['article_id'].to_numpy()
    all_possible_labels = df['prod_name'].to_numpy()
    cur_label = start - 1
    cur_prod_name = ''
    sort_label_idx = np.argsort(all_possible_labels)
    all_possible_labels = all_possible_labels[sort_label_idx]
    all_possible_images = all_possible_images[sort_label_idx]
    images, labels = [], []
    for image_id, product_name in tqdm(zip(all_possible_images,all_possible_labels)):
        image_id = '0' + str(image_id)
        if product_name != cur_prod_name:
            cur_label += 1 
            cur_prod_name = product_name
        if os.path.exists(args.data_path + '/images/' + image_id[:3] + '/' + image_id + '.jpg') :
            images.append(args.data_path + '/images/' + image_id[:3] + '/' + image_id + '.jpg')
            labels.append(cur_label)
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for file, label in zip(images, labels):
        if np.random.uniform() >= args.split_ratio:
            f_train.write(file + '---' + str(label) + '\n')
        else:
            f_test.write(file + '---' + str(label) + '\n')

    return labels[-1] + 1
    
def processing_sketch(args,start = 0):
    args.data_path =  '../imagenet_sketch'
    images, labels = [], []
    for label_idx, folder in enumerate(os.listdir(args.data_path + '/sketch')):
        label_idx += start
        for img_path in os.listdir(args.data_path + '/sketch/'+ folder):
            images.append(args.data_path + '/sketch/'+ folder + '/' + img_path)
            labels.append(label_idx)
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for file, label in zip(images, labels):
        if np.random.uniform() >= args.split_ratio:
            f_train.write(file + '---' + str(label) + '\n')
        else:
            f_test.write(file + '---' + str(label) + '\n')

    return labels[-1] + 1

def processing_product_10k(args, start = 0):
    args.data_path =  '../product_10k'
    images, labels = [], []
    for csv_file, image_folder in zip(['test_kaggletest.csv','train.csv'], ['test','train']):
        df = pd.read_csv(args.data_path +'/' + csv_file)
        all_possible_images = df['name'].to_numpy()
        all_possible_labels = df['class'].to_numpy()
        for image_id, class_id in tqdm(zip(all_possible_images,all_possible_labels)):
            class_id += start
            images.append(args.data_path + '/' + image_folder + '/' + image_id)
            labels.append(class_id)
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for file, label in zip(images, labels):
        if np.random.uniform() >= args.split_ratio:
            f_train.write(file + '---' + str(label) + '\n')
        else:
            f_test.write(file + '---' + str(label) + '\n')

    return labels[-1] + 1

def processing_met_dataset(args, start = 0) :
    args.data_path = '../met_artwork'
    images, labels, nums = [], [], []
    label_idx = start
    for _, folder in tqdm(enumerate(os.listdir(args.data_path + '/MET/MET'))):
        process = False
        n_samp = len(os.listdir(args.data_path + '/MET/MET/'+ folder))        
        if n_samp >= 2:
            process = True
        elif np.random.uniform() < 0.2 :
            process = True
        if process :
            for img_path in os.listdir(args.data_path + '/MET/MET/'+ folder):
                images.append(args.data_path + '/MET/MET/'+ folder + '/' + img_path)
                labels.append(label_idx)
                nums.append(n_samp)
            label_idx += 1
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for file, label, num in zip(images, labels, nums):
        if np.random.uniform() >= args.split_ratio or num <=2 :
            f_train.write(file + '---' + str(label) + '\n')
        else:
            f_test.write(file + '---' + str(label) + '\n')

    return labels[-1] + 1
    
def processing_stanfordcar(args, start = 0) :
    args.data_path = '../StanfordCars'
    images, labels = [], []
    for mat_file, image_folder in zip(['cars_train_annos.mat','cars_test_annos.mat'], ['cars_train/cars_train','cars_test/cars_test']):
        cars_annos = sio.loadmat(args.data_path + '/' + mat_file)
        annotations = cars_annos['annotations']
        annotations = np.transpose(annotations)
        for annotation in annotations:
            fname = annotation[0][5][0]
            car_class = annotation[0][4][0][0]
            images.append(args.data_path + '/' + image_folder + '/' +fname)
            labels.append(car_class + start)
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for file, label in zip(images, labels):
        if np.random.uniform() >= args.split_ratio:
            f_train.write(file + '---' + str(label) + '\n')
        else:
            f_test.write(file + '---' + str(label) + '\n')
    
    return np.max(labels) + 1

def processing_small_shapenet(args, start = 0):
    args.data_path = '../shapenet2dverysmall'
    ind = 0
    images, labels = [], []
    for _ , img_path in tqdm(enumerate(os.listdir(args.data_path + '/Input2'))):
        try:
            img = cv2.imread(args.data_path + '/Input2/' + img_path)
            imh, imw, _ = img.shape
        except:
            continue
        im1 = img[0:imh//2,0:imw//2,:]
        im2 = img[imh//2:imh,0:imw//2,:]
        im3 = img[0:imh//2,imw//2:imw,:]
        im4 = img[imh//2:imh,imw//2:imw,:]
        ims = [im1,im2,im3,im4]
        for i in range(4):
            cv2.imwrite(args.data_path + '/' + 'images/' + str(ind+i) + '.jpg',ims[i])
            images.append('images/' + str(ind+i) + '.jpg')
            labels.append(ind//4 + start)
        ind += 4
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for file, label in zip(images, labels):
        if np.random.uniform() >= args.split_ratio:
            f_train.write(args.data_path + '/' + file + '---' + str(label) + '\n')
        else:
            f_test.write(args.data_path + '/' + file + '---' + str(label) + '\n')
    return labels[-1] + 1

def processing_deepfhasionv2(args, start = 0):
    args.data_path = '../DeepFashinv2'
    modes = ['train','validation']
    labels = []
    images = []
    for mode in modes:
        img_folder = args.data_path + '/' + mode +'/image'
        anno_folder = args.data_path + '/' + mode +'/annos'
        for img_path in tqdm(os.listdir(img_folder)):
            anno_path = anno_folder + '/' + img_path.split('.')[0] +'.json'
            with open(anno_path) as json_file:
                data = json.load(json_file)
                labels.append(data['pair_id'] - 1 + start)
                images.append(img_folder + '/' + img_path )
                
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for file, label in zip(images, labels):
        if np.random.uniform() >= args.split_ratio:
            f_train.write(file + '---' + str(label) + '\n')
        else:
            f_test.write(file + '---' + str(label) + '\n')
    return np.max(np.array(labels)) + 1
def processing_auto_cars(args, start = 0):
    args.data_path = '../myautoge_cars'
    labels = []
    images = []
    all_folders = os.listdir(args.data_path + '/images/images')
    random.shuffle(all_folders)
    for label_idx,img_folder in tqdm(enumerate(all_folders[:2*len(all_folders)//5])):
        cur_im_paths = []
        for img_path in os.listdir(args.data_path + '/images/images/' + img_folder):
            cur_im_paths.append(int(img_path.split('.')[0]))
        cur_im_paths = sorted(cur_im_paths)
        cur_im_paths = [args.data_path + '/images/images/' + img_folder + '/' + str(im_path) + '.jpg'
                        for im_path in cur_im_paths]
        images.extend(cur_im_paths[:len(cur_im_paths)//2])
        labels.extend([label_idx + start] * (len(cur_im_paths)//2))
    
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for file, label in zip(images, labels):
        if np.random.uniform() >= args.split_ratio:
            f_train.write(file + '---' + str(label) + '\n')
        else:
            f_test.write(file + '---' + str(label) + '\n')
    return labels[-1] + 1
        
def processing_food_fgc6(args, start = 0):
    args.data_path = '../food-fgvc6'
    modes = ['train_set', 'val_set']
    images = []
    labels = []
    for mode in modes :
        df = pd.read_csv(os.path.join(args.data_path, mode + '.csv'))
        images.extend([args.data_path + '/'+ mode + '/' + im_file for im_file in df['img_name'].to_numpy()])
        label = df['label'].to_numpy()
        labels.extend([start + l for l in label])
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for file, label in zip(images, labels):
        if np.random.uniform() >= args.split_ratio:
            f_train.write(file + '---' + str(label) + '\n')
        else:
            f_test.write(file + '---' + str(label) + '\n')
    return np.max(np.array(labels)) + 1
        
def processing_alibaba_product(args, start = 0):
#     for i in os.listdir('../alibaba_product/goods_categories'):
#         os.rename('../alibaba_product/goods_categories/' + i, '../alibaba_product/goods_categories/' + 
#                   i.replace("ChildrenтАЩs_Shoes", 'ChildrenShoes'))

    args.data_path = '../alibaba_product'
    images = []
    labels = []
    df = pd.read_csv(os.path.join(args.data_path,'meta.csv'))
    for cate, im, lab in zip(df['category'],df['img_name'],df['label_group']):
        cate = cate.replace('Children’s Shoes', 'ChildrenShoes')
        cate = cate.replace('Men’s Shoes', 'MenShoes')
        cate = cate.replace('Men’s Shoes', 'MenShoes')
        cate = cate.replace('Women’s Shoes', 'WomenShoes')
        cate = cate.replace('Men’s Clothing', 'MenClothing')
        cate = cate.replace('Girls’ Clothing', 'GirlClothing')
        cate = cate.replace('Boy’s Clothing', 'BoyClothing')    
        
        cate = cate.replace(' ', '_')
        images.append(args.data_path + '/goods_categories/' + cate + '/' + im )
        labels.append(lab + start)
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for file, label in zip(images, labels):
        if np.random.uniform() >= args.split_ratio:
            f_train.write(file + '---' + str(label) + '\n')
        else:
            f_test.write(file + '---' + str(label) + '\n')
    return np.max(np.array(labels)) + 1
    
###DATASETV1####      
# def process_all_data(args):
#     ncl = processing_gglandmarkv2(args)
#     print('ncl',  ncl)
#     ncl = processing_food_recog_2022(args,ncl)
#     print('ncl',  ncl)
#     ncl = processing_HMFashion(args,ncl)
#     print('ncl',  ncl)
#     ncl = processing_sketch(args,ncl)
#     print('ncl',  ncl)
#     ncl = processing_product_10k(args,ncl)
#     print('ncl',  ncl)
#     ncl = processing_met_dataset(args,ncl)
#     print('ncl',  ncl)
#     ncl = processing_stanfordcar(args,ncl)
#     print('ncl',  ncl)
#     ncl = processing_small_shapenet(args,ncl)
#     print('ncl',  ncl)    

###DATASETV3####      

def process_all_data(args):
    ncl = processing_gglandmarkv2(args)
    print('ncl',  ncl)
    ncl = processing_food_recog_2022(args,ncl)
    print('ncl',  ncl)
    ncl = processing_deepfhasionv2(args,ncl)
    print('ncl',  ncl)
    ncl = processing_sketch(args,ncl)
    print('ncl',  ncl)
    ncl = processing_product_10k(args,ncl)
    print('ncl',  ncl)
    ncl = processing_met_dataset(args,ncl)
    print('ncl',  ncl)
    ncl = processing_stanfordcar(args,ncl)
    print('ncl',  ncl)
    ncl = processing_alibaba_product(args,ncl)
    print('ncl', ncl)
#     ncl = processing_small_shapenet(args,ncl)
#     print('ncl',  ncl)  
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('XCiT Retrieval training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    list_data_paths = ['../gglandmark-v2-clean','../food-recog-2022','../DeepFashinv2','../imagenet_sketch',
                       '../product_10k','../met_artwork','../StanfordCars', 
                       '../alibaba_product']
    process_all_data(args)
    args.data_path = '../split_path'
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for data in list_data_paths:
        with open(data + '/train_split.txt', 'r') as train_f:
            for d in tqdm(train_f.readlines()):
                f_train.write(d)
        with open(data + '/test_split.txt', 'r') as test_f:
            for d in test_f.readlines():
                f_test.write(d)
    f_train.close()
    f_test.close()
    
    with open('../split_path/train_split.txt', 'r') as t_f:
        a = t_f.readlines()
    print(a[-1])
    
    
    
        
        

                
                
            
        
        
    
    