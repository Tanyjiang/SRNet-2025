import cv2
from torch.utils import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from transforms_MAN import Transforms
import glob
from torchvision.transforms import functional
import pandas as pd
import json


class Dataset1(data.Dataset):
    def __init__(self, dataset, increntmal_phase, exemplar_set, exemplar_set_gt, dataset_dir):
        self.dataset = dataset
        self.label_list = []
        self.image_list = []
        self.target = []
        train_stage = ['category1', 'category2', '...', 'categoryn']
        test_stage = {
            0: ['category1'],
            1: ['category2', 'category1'],
            2: ['category3', 'category1', 'category2'],
            ...: ...,
            'n': ['categoryn', 'category1', 'category2', '...', 'categoryn-1']
        }
        self.class_name = ['others', 'your_class_name_1', 'your_class_name_2', '...', 'your_class_name_n']

        dataset = dataset + '_data'
        if increntmal_phase == 0:
            dataset_path = os.path.join(dataset_dir, train_stage[increntmal_phase], dataset, 'images')
            self.image_list = glob.glob(os.path.join(dataset_path, '*.jpg'))
            for index in range(len(self.image_list)):
                image = self.image_list[index]
                image_name = image.split('images' + os.sep)[1]
                if image_name.startswith('class1'):
                    self.label_list.append(
                        self.image_list[index].replace('.jpg', '.npy').replace('images', 'ground_truth'))
                    if self.dataset == 'train':
                        self.target.append(1)
                else:
                    img = cv2.imread(image)
                    height = img.shape[0]
                    width = img.shape[1]
                    label = np.zeros((height, width))
                    self.label_list.append(label)
                    self.target.append(0)

        elif increntmal_phase >= 1:
            if self.dataset == 'test' or self.dataset == 'val':
                for phase in range(len(test_stage[increntmal_phase])):
                    dataset_path = os.path.join(dataset_dir, test_stage[increntmal_phase][phase], dataset, 'images')
                    img_list_buff = glob.glob(os.path.join(dataset_path, '*.jpg'))
                    for index in range(len(img_list_buff)):
                        self.image_list.append(img_list_buff[index])
                        self.label_list.append(
                            img_list_buff[index].replace('.jpg', '.npy').replace('images', 'ground_truth'))


            elif self.dataset == 'train':
                dataset_path = os.path.join(dataset_dir, train_stage[increntmal_phase], dataset, 'images')
                img_list_buff = glob.glob(os.path.join(dataset_path, '*.jpg'))
                for index in range(len(img_list_buff)):
                    self.image_list.append(img_list_buff[index])
                    self.label_list.append(
                        img_list_buff[index].replace('.jpg', '.npy').replace('images', 'ground_truth'))
                    image_name = img_list_buff[index].split('images' + os.sep)[1]
                    image_name = image_name.split('_')[0]
                    if image_name in self.class_name:
                        self.target.append(self.class_name.index(image_name))
                    else:
                        print('error!')

                if exemplar_set != None:
                    for index in range(len(exemplar_set)):
                        for num in range(len(exemplar_set[index])):
                            self.image_list.append(exemplar_set[index][num])
                            self.label_list.append(exemplar_set_gt[index][num])
                            image_name = exemplar_set[index][num].split('images' + os.sep)[1]
                            image_name = image_name.split('_')[0]
                            if image_name in self.class_name:
                                self.target.append(self.class_name.index(image_name))
                            else:
                                self.target.append(0)


    def __getitem__(self, index):

        image = Image.open(self.image_list[index]).convert('RGB')

        if self.dataset == 'train':
            target = self.target[index]
            img = self.image_list[index].split('images' + os.sep)[1]
            img = img.split('_')[0]

            if img in self.class_name:

                keypoints = np.load(self.label_list[index])
                gt = len(keypoints)
            else:

                keypoints = np.array([])
                gt = 0
        else:

            keypoints = np.load(self.label_list[index])
            gt = len(keypoints)



        trans = Transforms((0.8, 1.2), (512, 512), 1, (0.5, 1.5), self.dataset)
        if self.dataset == 'train':
            image, points, targets, st_sizes = trans(image, keypoints, None)
            return image, points, targets, st_sizes, target

        else:
            height, width = image.size[1], image.size[0]
            height = round(height / 16) * 16
            width = round(width / 16) * 16
            image = image.resize((width, height), Image.BILINEAR)

            image = functional.to_tensor(image)
            image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return image, gt

    def __len__(self):
        return len(self.image_list)

