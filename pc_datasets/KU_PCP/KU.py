"""
Created on Tue April 19 2022
@author: Wang Zhicheng
"""

import cv2
import numpy as np
from PIL import Image
import os
import torch

from torch.utils.data import Dataset


def read_image(x):
    # the order of the image dimension: (height, width, channel)
    img_arr = np.array(Image.open(x))
    if len(img_arr.shape) == 2:
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr


# def read_data_list_test(data_list,data_dir):
#     data_ground_list = [list(map(int,name.split(' '))) for name in open(data_list,'r').read().splitlines()]
#     data_list = os.listdir(data_dir)
#     img_map = dict(zip(data_list, data_ground_list))
#     img = read_image(os.path.join(data_dir,data_list[0]))[:,:,0:3]
#     img = cv2.resize(img,(200,128),interpolation=cv2.INTER_CUBIC)
#     print(1)

class KU_PCPDataset(Dataset):
    def __init__(self, data_dir, data_list, input_size, preload=True, transform=None):
        self.data_dir = data_dir
        self.data_list = os.listdir(data_dir)
        self.data_list.sort()
        self.data_ground_list = [list(map(int, name.split(' '))) for name in open(data_list, 'r').read().splitlines()]
        self.img_map = dict(zip(self.data_list, self.data_ground_list))
        self.preload = preload
        self.transform = transform

        # input size
        self.input_size = input_size  # input_size = [width, height]

        # store images and generate ground truths
        self.image_list = []
        self.images = {}
        self.labels = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        self.image_list.append(file_name)

        if file_name in self.images:
            image = self.images[file_name]
            label = self.labels[file_name]

        else:
            image_path = os.path.join(self.data_dir, file_name)

            image = read_image(image_path)[:, :, 0:3]
            label = self.img_map[file_name]

            image = cv2.resize(image, (self.input_size[0], self.input_size[1]), interpolation=cv2.INTER_CUBIC)
            if self.preload:
                self.images.update({file_name: image})
                self.labels.update({file_name: label})
        label = torch.tensor(label, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        sample = {
            'image': image,
            'label': label
        }

        return sample