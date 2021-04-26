'''
data generator for the global local net

v0: for resnet34 only
v1: for global local with only local path, prepare the data for the input['local']
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import pandas as pd
import random
import os
import math
# from skimage import io, transform
import numpy as np
import cv2
from time import time
from PIL import Image
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
plt.ion()



class dataconfig(object):
    def __init__(self, dataset = 'defaut',subset = '0', **kwargs):
        self.dataset = dataset
        # self.dir = r'E:\Xing\mass0508\Crops_M'   % original
        # self.csv = 'Gl_summery_csv_valid_dn.csv' % original
        self.dir = r'E:\Xing\Covid_19_xray\code'
        self.csv = 'df_main.csv'
        self.subset = subset
        self.csv_file = os.path.join(self.dir,self.csv)

class batch_sampler():
    def __init__(self, batch_size, class_list):
        self.batch_size = batch_size
        self.class_list = class_list
        self.unique_value = np.unique(class_list)
        self.iter_list = []
        self.len_list = []
        for v in self.unique_value:
            indexes = np.where(self.class_list == v)[0]
            self.iter_list.append(self.shuffle_iterator(indexes))
            self.len_list.append(len(indexes))
        self.len = len(class_list) // batch_size
        # print('self.len: ', self.len)

    def __iter__(self):
        index_list = []
        for _ in range(self.len):
            for index in range(self.batch_size):
                index_list.append(next(self.iter_list[index % len(self.unique_value)]))
            np.random.shuffle(index_list)
            yield index_list
            index_list = []

    def __len__(self):
        return self.len

    @staticmethod
    def shuffle_iterator(iterator):
        # iterator should have limited size
        index = list(iterator)
        total_size = len(index)
        i = 0
        random.shuffle(index)
        while True:
            yield index[i]
            i += 1
            if i >= total_size:
                i = 0
                random.shuffle(index)


class DataGenerator(Dataset):
    def __init__(self, config=None,transform = None):
        self.config = config
        self.debug = False
        self.df = self.parse_csv(self.config.csv_file, self.config.subset)
        self.df.reset_index(drop=True, inplace=True)
        self.transform = transform
        if self.config.subset == '0':
            self.data_folder = r'E:\Xing\Covid_19_xray\data\data\train'
        else:
            self.data_folder = r'E:\Xing\Covid_19_xray\data\data\test'

    def __len__(self):
        print('len = {}'.format(len(self.df)))
        return len(self.df)

    def img_augmentation(self, img, seq_det):

        img = img.transpose(2, 0, 1)

        for i in range(len(img)):
            img[i, :, :] = seq_det.augment_image(img[i, :, :])

        img = img.transpose(1, 2, 0)

        # img = seq_det.augment_images(img)

        return img

    def __getitem__(self, index):

        # print(index)

        img_name = self.df.loc[index, 'filename']
        img_path = os.path.join(self.data_folder,img_name)
        # print(img_path)
        # image = cv2.imread(img_path)
        image = Image.open(img_path).convert('RGB')

        image = image.resize((224,224))

        # label = self.df.loc[index,'Shape']
        label = self.df.loc[index, 'label_num']
        # label = label.reshape(-1,1)
        # landmarks = landmarks.reshape(-1, 2)
        # sample = {'image': image, 'label': label}

        if self.transform:


            #
            # if len(image.shape) == 2:
            #     image = np.transpose(np.array([image]*3),(1,2,0))

            # dec = random.choice(range(2))
            # if dec == 1 and self.df.loc[index, 'valid'] == 0:
            if self.df.loc[index, 'valid'] == 0:
                # print('{} is img_auged'.format(index))


                seq = iaa.SomeOf((3, 6), [
                    iaa.Fliplr(0.8),
                    iaa.Flipud(0.8),
                    iaa.Multiply((0.8, 1.2)),
                    iaa.GaussianBlur(sigma=(0.0, 0.2)),
                    iaa.PiecewiseAffine((0.02, 0.06)),
                    iaa.Affine(
                        rotate=(-5, 5),
                        shear=(-5, 5),
                        scale=({'x': (0.8, 1.1), 'y': (0.8, 1.1)})  # to strentch the image along x,y axis
                    )
                ])

                seq_det = seq.to_deterministic()

                # image = seq_det.augment_image(image)
                image = np.array(image)
                image = self.img_augmentation(image, seq_det=seq_det)
                # plt.imshow(image),plt.show()
                image = Image.fromarray(image)

            image = self.transform(image)

        if self.debug:
            pass

        return image,label

    @staticmethod
    def parse_csv(csv_file, subset):
        data_frame = pd.read_csv(csv_file)
        data_frame = data_frame[data_frame['valid'] == int(subset)]
        return data_frame


def show_landmarks(image, landmarks):
    """SHow image with landmarks"""
    plt.imshow(image)
    # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker=".", c="r")


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.456], [0.224])
    ])

    valconfig = {"dataset": "mura","subset": '1'}
    val_config = dataconfig(**valconfig)
    validation_data = DataGenerator(val_config,transform= transform)
    val_loader = DataLoader(validation_data, num_workers=1)

    for i, (images, labels) in enumerate(val_loader):
        print(images.shape)

