# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 18:01:24 2020

@author: crystal
"""
import os
import sys
import random
import math
import subprocess
from glob import glob
from collections import OrderedDict
import numpy as np
import cv2
from skimage import measure
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.optimizer import Optimizer, required
import pretrainedmodels
sys.path.append("../input/resnext50")
sys.path.append("../input/resnext101")
sys.path.append("../input/pretrain-models")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform = None, img_size = 288, save_img = True):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        self.save_img = save_img

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]
        
        if 'train' in img_path:
            img = cv2.imread('../input/aptos2019-dataset/images_288_scaled/images_288_scaled/%s' %os.path.basename(img_path))
        
        else:
            if os.path.exists('processed/%s' %os.path.basename(img_path)):
                img = cv2.imread('processed/%s' %os.path.basename(img_path))

            else:
                img = cv2.imread(img_path)
                try:
                    img = scale_radius(img, img_size=self.img_size, padding = False)
                except Exception as e:
                    img = img
                if self.save_img:
                    cv2.imwrite('processed/%s' %os.path.basename(img_path), img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.img_paths)
    
    
def get_model(model_name = 'resnet18', num_outputs = None, pretrained = True,
              freeze_bn = False, dropout_p = 0, **kwargs):

    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__[model_name](num_classes = 1000,
                                                  pretrained = pretrained)

    if 'dpn' in model_name:
        in_channels = model.last_linear.in_channels
        model.last_linear = nn.Conv2d(in_channels, num_outputs,
                                      kernel_size = 1, bias = True)
    else:
        if 'resnet' in model_name:
            model.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = model.last_linear.in_features
        
        if dropout_p == 0:
            model.last_linear = nn.Linear(in_features, num_outputs)
        else:
            model.last_linear = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_outputs))

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    return model


l1_probs = {}

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
dir_name = '../input/aptos2019-blindness-detection/test_images'
test_img_paths = dir_name + '/' + test_df['id_code'].values + '.png'
test_labels = np.zeros(len(test_img_paths))

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_set = Dataset(
    test_img_paths,
    test_labels,
    transform = test_transform)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size = 8,
    shuffle = False,
    num_workers = 2)

# inference se_resnext50_32x4d
model = get_model(model_name = 'se_resnext50_32x4d',
                  num_outputs = 1,
                  pretrained = False,
                  freeze_bn = True,
                  dropout_p = 0)
model = model.cuda()
model.eval()

probs = []
for fold in range(5):
    print('Fold [%d/%d]' %(fold+1, 5))

    model.load_state_dict(torch.load('../input/resnext50/se_resnext50_32x4d/model_%d.pth' % (fold+1)))

    probs_fold = []
    with torch.no_grad():
        for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            input = input.cuda()
            output = model(input)
            probs_fold.extend(output.data.cpu().numpy()[:, 0])
    probs_fold = np.array(probs_fold)
    probs.append(probs_fold)

probs = np.mean(probs, axis=0)
l1_probs['se_resnext50_32x4d'] = probs

# inference se_resnext101_32x4d
model = get_model(model_name='se_resnext101_32x4d',
                  num_outputs=1,
                  pretrained=False,
                  freeze_bn=True,
                  dropout_p=0)
model = model.cuda()
model.eval()

probs = []
for fold in range(5):
    print('Fold [%d/%d]' %(fold+1, 5))

    model.load_state_dict(torch.load('../input/resnext101/se_resnext101_32x4d/model_%d.pth' % (fold+1)))

    probs_fold = []
    with torch.no_grad():
        for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            input = input.cuda()
            output = model(input)
            probs_fold.extend(output.data.cpu().numpy()[:, 0])   
    probs_fold = np.array(probs_fold)
    probs.append(probs_fold)

probs = np.mean(probs, axis=0)
l1_probs['se_resnext101_32x4d'] = probs

preds = 0.5 * l1_probs['se_resnext50_32x4d'] + 0.5 * l1_probs['se_resnext101_32x4d']
thrs = [0.5, 1.5, 2.5, 3.5]
preds[preds < thrs[0]] = 0
preds[(preds >= thrs[0]) & (preds < thrs[1])] = 1
preds[(preds >= thrs[1]) & (preds < thrs[2])] = 2
preds[(preds >= thrs[2]) & (preds < thrs[3])] = 3
preds[preds >= thrs[3]] = 4
preds = preds.astype('int')

test_df['diagnosis'] = preds
test_df.to_csv('submission.csv', index = False)