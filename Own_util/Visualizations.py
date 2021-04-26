# -*- encoding: utf-8 -*-
'''
@File    :   Visualizations.py    
@Contact :   lvxingvir@gmail.com
@License :   (C)Copyright Xing Lu

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
04/14/2021 13:52  Xing        1.0         inital for the cam visulazations
'''

import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import cv2
import matplotlib.pyplot as plt


def parse_arg():
    parser = argparse.ArgumentParser(description='Visualizations')
    parser.add_argument('--img_num', help='img_num to show', default=6, type=int)
    parser.add_argument('--cls_name', help='number of classification', default={}, type=dict)
    args = parser.parse_args()
    return args


class Visualizations(object):
    def __init__(self, img_num = 6, class_names = 1):
        self.image_num = img_num
        self.class_names = class_names


    @staticmethod
    def generate_grad_cam(net, ori_image, Trans):
        """
        :param net: deep learning network(ResNet DataParallel object)
        :param ori_image: the original image
        :return: gradient class activation map
        """
        input_image = Trans(ori_image)

        feature = None
        gradient = None

        def func_f(module, input, output):
            nonlocal feature
            feature = output.data.cpu().numpy()

        def func_b(module, grad_in, grad_out):
            nonlocal gradient
            gradient = grad_out[0].data.cpu().numpy()

        # net.module.layer4.register_forward_hook(func_f)
        # net.module.layer4.register_backward_hook(func_b)

        net.layer4.register_forward_hook(func_f)
        net.layer4.register_backward_hook(func_b)

        out = net(input_image.unsqueeze(0))

        pred = (out.data > 0.5)

        net.zero_grad()

        loss = F.binary_cross_entropy(out, pred.float())
        loss.backward()

        feature = np.squeeze(feature, axis=0)
        gradient = np.squeeze(gradient, axis=0)

        weights = np.mean(gradient, axis=(1, 2), keepdims=True)

        cam = np.sum(weights * feature, axis=0)

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = 1.0 - cam
        cam = np.uint8(cam * 255)

        return cam

    @staticmethod
    def generate_grad_cam_inter(net, ori_image, label, criterion, num_classes=1):
        """
        :param net: deep learning network(ResNet DataParallel object)
        :param ori_image: the original image
        :return: gradient class activation map
        """
        input_image = ori_image

        feature = None
        gradient = None

        def func_f(module, input, output):
            nonlocal feature
            feature = output.data.cpu().numpy()

        def func_b(module, grad_in, grad_out):
            nonlocal gradient
            gradient = grad_out[0].data.cpu().numpy()

        # net.module.layer4.register_forward_hook(func_f)
        # net.module.layer4.register_backward_hook(func_b)

        net.layer4.register_forward_hook(func_f)
        net.layer4.register_full_backward_hook(func_b)

        #     pred = net(input_image.unsqueeze(0))
        #     print(pred)
        out = net(input_image.unsqueeze(0))
        label = label.unsqueeze(0)

        if num_classes == 1:
            pred = out.data > 0.5
        else:
            _, pred = torch.max(out.data, 1)

        net.zero_grad()

        loss = criterion(out, label)
        #     loss = criterion(out,pred) # by the original, but maybe wrong.

        loss.backward()

        feature = np.squeeze(feature, axis=0)
        gradient = np.squeeze(gradient, axis=0)

        weights = np.mean(gradient, axis=(1, 2), keepdims=True)

        cam = np.sum(weights * feature, axis=0)

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = 1.0 - cam
        cam = np.uint8(cam * 255)

        return cam

    @staticmethod
    def localize(cam_feature, ori_image, frac=0.6):
        """
        localize the abnormality region using grad_cam feature
        :param cam_feature: cam_feature by generate_grad_cam
        :param ori_image: the original image, with range of 255
        :return: img with heatmap, the abnormality region is highlighted
        """
        ori_image = np.array(ori_image)
        activation_heatmap = cv2.applyColorMap(cam_feature, cv2.COLORMAP_JET)
        activation_heatmap = cv2.resize(activation_heatmap, (ori_image.shape[1], ori_image.shape[0]))
        img_with_heatmap = frac * np.float32(activation_heatmap) + (1 - frac) * np.float32(ori_image)
        img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
        return img_with_heatmap

    @staticmethod
    def localize2(cam_feature, ori_image):
        """
        localize the abnormality region using grad_cam feature
        :param cam_feature: cam_feature by generate_grad_cam
        :param ori_image: input of the network
        :return: img with heatmap, the abnormality region is in a red window
        """
        ori_image = np.array(ori_image)
        cam_feature = cv2.resize(cam_feature, (ori_image.shape[1], ori_image.shape[0]))
        crop = np.uint8(cam_feature > 0.7 * 255)
        h = ori_image.shape[0]
        w = ori_image.shape[1]
        ret, markers = cv2.connectedComponents(crop)
        branch_size = np.zeros(ret)
        for i in range(h):
            for j in range(w):
                t = int(markers[i][j])
                branch_size[t] += 1
        branch_size[0] = 0
        max_branch = np.argmax(branch_size)
        mini = h
        minj = w
        maxi = -1
        maxj = -1
        for i in range(h):
            for j in range(w):
                if markers[i][j] == max_branch:
                    if i < mini:
                        mini = i
                    if i > maxi:
                        maxi = i
                    if j < minj:
                        minj = j
                    if j > maxj:
                        maxj = j
        img_with_window = np.uint8(ori_image)
        img_with_window[mini:mini + 2, minj:maxj, 0:1] = 255
        img_with_window[mini:mini + 2, minj:maxj, 1:3] = 0
        img_with_window[maxi - 2:maxi, minj:maxj, 0:1] = 255
        img_with_window[maxi - 2:maxi, minj:maxj, 1:3] = 0
        img_with_window[mini:maxi, minj:minj + 2, 0:1] = 255
        img_with_window[mini:maxi, minj:minj + 2, 1:3] = 0
        img_with_window[mini:maxi, maxj - 2:maxj, 0:1] = 255
        img_with_window[mini:maxi, maxj - 2:maxj, 1:3] = 0

        return img_with_window

    @staticmethod
    def generate_local(cam_features, inputs, invTrans, Normal):
        """
        :param cam_features: numpy array of shape = (B, 224, 224), pixel value range [0, 255]
        :param inputs: tensor of size = (B, 3, 224, 224), with mean and std as Imagenet
        :return: local image
        """
        b = cam_features.shape[0]
        local_out = []
        for k in range(b):
            ori_img = invTrans(inputs[k]).cpu().numpy()
            ori_img = np.transpose(ori_img, (1, 2, 0))
            ori_img = np.uint8(ori_img * 255)

            crop = np.uint8(cam_features[k] > 0.7)
            ret, markers = cv2.connectedComponents(crop)
            branch_size = np.zeros(ret)
            h = 224
            w = 224
            for i in range(h):
                for j in range(w):
                    t = int(markers[i][j])
                    branch_size[t] += 1
            branch_size[0] = 0
            max_branch = np.argmax(branch_size)
            mini = h
            minj = w
            maxi = -1
            maxj = -1
            for i in range(h):
                for j in range(w):
                    if markers[i][j] == max_branch:
                        if i < mini:
                            mini = i
                        if i > maxi:
                            maxi = i
                        if j < minj:
                            minj = j
                        if j > maxj:
                            maxj = j
            local_img = ori_img[mini: maxi + 1, minj: maxj + 1, :]
            local_img = cv2.resize(local_img, (224, 224))
            local_img = Image.fromarray(local_img)
            local_img = Normal(local_img)
            local_out += [local_img]
        local_out = torch.stack(local_out)
        return local_out

    # @staticmethod
    def show_images(self, images, labels, preds, save_path = None, title = ''):
        plt.figure(figsize=(12, 10))

        colum = np.sqrt(self.image_num)

        for i, image in enumerate(images):
            if i > self.image_num-1:
                break
            plt.subplot(round(colum), round(self.image_num/colum), i + 1, xticks=[], yticks=[])  # x & y ticks are set to blank
            image = image.numpy().transpose((1, 2, 0))  # Channel first then height and width
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0., 1.)
            plt.imshow(image)

            col = 'green' if preds[i] == labels[i] else 'red'

            plt.xlabel(f'{self.class_names[int(labels[i].cpu().numpy())]}')
            plt.ylabel(f'{self.class_names[int(preds[i].cpu().numpy())]}', color=col)
        plt.suptitle(title)
        plt.tight_layout()
        if save_path:
            fig_name = os.path.join(save_path,title+'.png')
            plt.savefig(fig_name,dpi=300)
        plt.show()

    # @staticmethod
    def show_cam_images(self,model_ckpt, images, labels, preds, criteration,frac = 0.6):
        '''

        :param model_ckpt: input model
        :param images: input an iter images
        :param labels:
        :param preds:
        :param criteration: loss_fn
        :param class_names: diction for class names
        :return: no retrun but can save images
        '''

        num_cls = len(self.class_names)
        plt.figure(figsize=(12, 10))
        for i, (ori_image, label) in enumerate(zip(images, labels)):
            #         print(i,ori_image.shape,label)

            if i > self.image_num-1:
                break
            plt.subplot(2, round(self.image_num/2), i + 1, xticks=[], yticks=[])  # x & y ticks are set to blank

            #         ori_image = Image.open(args.img_path).convert('RGB')
            cam_feature = Visualizations.generate_grad_cam_inter(model_ckpt, ori_image, label, criteration, num_classes=num_cls)
            #         print(cam_feature.shape)

            image = ori_image.numpy().transpose((1, 2, 0))  # Channel first then height and width

            image = Visualizations.localize(cam_feature, image,frac=frac)

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0., 1.)
            plt.imshow(image)

            col = 'green' if preds[i] == labels[i] else 'red'

            plt.xlabel(f'{self.class_names[int(labels[i].numpy())]}')
            plt.ylabel(f'{self.class_names[int(preds[i].numpy())]}', color=col)

        plt.tight_layout()
        plt.show()


    def __call__(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    args = parse_arg()

    Visualizations(args)()