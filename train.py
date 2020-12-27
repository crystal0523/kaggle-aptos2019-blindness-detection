import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from skimage.io import imread

import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from lib.dataset import RetinopathyLoader
from lib.models.model_factory import get_model
from lib.utils import *
from lib.metrics import *
from lib.preprocess import preprocess


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default = None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar = 'ARCH', default='resnet34',
                        help='model architecture: ' +
                        ' (default: resnet34)')
    parser.add_argument('--freeze_bn', default = True, type = str2bool)
    
    parser.add_argument('--reg_coef', default = 1.0, type = float)
    parser.add_argument('--cls_coef', default = 0.1, type = float)
    parser.add_argument('--epochs', default = 30)             
    parser.add_argument('-b', '--batch_size', default = 4)
    parser.add_argument('--lr', '--learning-rate', default = 1e-3, type = float, metavar = 'LR')
    parser.add_argument('--min_lr', default = 1e-5, type = float, help = 'minimum learning rate')
    parser.add_argument('--factor', default = 0.5, type = float)
    parser.add_argument('--patience', default = 5, type = int)
    
    # dataset
    parser.add_argument('--train_dataset', default = 'aptos2015 + aptos2019')
    parser.add_argument('--cv', default = True, type = str2bool)
    parser.add_argument('--n_splits', default = 5, type = int)

    parser.add_argument('--pretrained_model')
    parser.add_argument('--pseudo_labels')
    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    for i, (input, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        label = label.cuda()
        output = model(input)
        
        loss = criterion(output.view(-1), label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        thrs = [0.5, 1.5, 2.5, 3.5]
        output[output < thrs[0]] = 0
        output[(output >= thrs[0]) & (output < thrs[1])] = 1
        output[(output >= thrs[1]) & (output < thrs[2])] = 2
        output[(output >= thrs[2]) & (output < thrs[3])] = 3
        output[output >= thrs[3]] = 4
        
      
        label[label < thrs[0]] = 0
        label[(label >= thrs[0]) & (label < thrs[1])] = 1
        label[(label >= thrs[1]) & (label < thrs[2])] = 2
        label[(label >= thrs[2]) & (label < thrs[3])] = 3
        label[label >= thrs[3]] = 4
        score = quadratic_weighted_kappa(output, label)

        losses.update(loss.item(), input.size(0))
        scores.update(score, input.size(0))

    return losses.avg, scores.avg


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

  
    model.eval()

    with torch.no_grad():
        for i, (input, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            label = label.cuda()
            output = model(input)     
            loss = criterion(output.view(-1), label.float())    
            thrs = [0.5, 1.5, 2.5, 3.5]
            output[output < thrs[0]] = 0
            output[(output >= thrs[0]) & (output < thrs[1])] = 1
            output[(output >= thrs[1]) & (output < thrs[2])] = 2
            output[(output >= thrs[2]) & (output < thrs[3])] = 3
            output[output >= thrs[3]] = 4
            score = quadratic_weighted_kappa(output, label)

            losses.update(loss.item(), input.size(0))
            scores.update(score, input.size(0))

    return losses.avg, scores.avg


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s_%s' % (args.arch, datetime.now().strftime('%m%d%H'))

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('- %s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('- %s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)
    criterion = nn.MSELoss().cuda()
    cudnn.benchmark = True

    model = get_model(model_name=args.arch,
                      num_outputs = 1,
                      freeze_bn=args.freeze_bn,)

    train_transform = []
    train_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomAffine(degrees = (180, 180),
            translate = (0, 0),
            scale = (0.8889, 1),
            shear = (-36, 36)),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # data loading code
    if 'aptos2015' in args.train_dataset:
        aptos2015_dir = preprocess('aptos2015', 288, scale = True, norm = True, pad = True, remove = False)
        aptos2015_df = pd.read_csv('C:/Users/USER/Desktop/crystal/cv/final/trainLabels2.csv')
        aptos2015_img_paths = aptos2015_dir+'/' + aptos2015_df['image'].values + '.jpeg'
        aptos2015_labels = aptos2015_df['level'].values

    if 'aptos2019' in args.train_dataset:
        aptos2019_dir = preprocess(
            'aptos2019',
            288,
            scale = True,
            norm = True,
            pad = True,
            remove = True)
        aptos2019_df = pd.read_csv('C:/Users/USER/Desktop/crystal/cv/final/train.csv')
        aptos2019_img_paths = aptos2019_dir + '/' + aptos2019_df['id_code'].values + '.png'
        aptos2019_labels = aptos2019_df['diagnosis'].values

    if args.train_dataset == 'aptos2019':
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=41)
        img_paths = []
        labels = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(aptos2019_img_paths, aptos2019_labels)):
            img_paths.append((aptos2019_img_paths[train_idx], aptos2019_img_paths[val_idx]))
            labels.append((aptos2019_labels[train_idx], aptos2019_labels[val_idx]))
    elif args.train_dataset == 'aptos2015':
        img_paths = [(diabetic_retinopathy_img_paths, aptos2019_img_paths)]
        labels = [(diabetic_retinopathy_labels, aptos2019_labels)]
    elif 'aptos2015' in args.train_dataset and 'aptos2019' in args.train_dataset:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=41)
        img_paths = []
        labels = []
        print(img_paths)
        for fold, (train_idx, val_idx) in enumerate(skf.split(aptos2019_img_paths, aptos2019_labels)):
            img_paths.append((np.hstack((aptos2019_img_paths[train_idx], aptos2015_img_paths)),aptos2019_img_paths[val_idx]))
            labels.append((np.hstack((aptos2019_labels[train_idx], aptos2015_labels)), aptos2019_labels[val_idx]))
    # else:
    #     raise NotImplementedError

    if args.pseudo_labels:
        test_df = pd.read_csv('probs/%s.csv' % args.pseudo_labels)
        test_dir = preprocess(
            'test',
            288,
            scale = True,
            norm = True,
            pad = True,
            remove = True)
        test_img_paths = test_dir + '/' + test_df['id_code'].values + '.png'
        test_labels = test_df['diagnosis'].values
        for fold in range(len(img_paths)):
            img_paths[fold] = (np.hstack((img_paths[fold][0], test_img_paths)), img_paths[fold][1])
            labels[fold] = (np.hstack((labels[fold][0], test_labels)), labels[fold][1])

    folds = []
    best_losses = []
    best_scores = []

    for fold, ((train_img_paths, val_img_paths), (train_labels, val_labels)) in enumerate(zip(img_paths, labels)):
        print('Fold [%d/%d]' %(fold+1, len(img_paths)))

        if os.path.exists('models/%s/model_%d.pth' % (args.name, fold+1)):
            log = pd.read_csv('models/%s/log_%d.csv' %(args.name, fold+1))
            best_loss, best_score = log.loc[log['val_loss'].values.argmin(), ['val_loss', 'val_score']].values
            folds.append(str(fold + 1))
            best_losses.append(best_loss)
            best_scores.append(best_score)
            continue

        # train
        train_set = RetinopathyLoader(
            train_img_paths,
            train_labels,
            transform=train_transform)

        _, class_sample_counts = np.unique(train_labels, return_counts=True)
      
        train_loader = DataLoader(train_set, batch_size = args.batch_size,
            shuffle = False, num_workers = 4)

        val_set = RetinopathyLoader(val_img_paths, val_labels, transform = val_transform)
        val_loader = DataLoader(val_set, batch_size = args.batch_size, 
                                shuffle = False, num_workers = 4)

        # create model
        model = get_model(model_name = args.arch,num_outputs = 1, 
                          freeze_bn = args.freeze_bn)
        model = model.cuda()
        if args.pretrained_model is not None:
            model.load_state_dict(torch.load('models/%s/model_%d.pth' % (args.pretrained_model, fold+1)))


        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr,
                                  momentum = 0.9, weight_decay = 1e-4)

        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        best_loss = float('inf')
        best_score = 0
        for epoch in range(args.epochs):
            print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

           
            train_loss, train_score = train(args, train_loader, model, criterion, optimizer, epoch)
            val_loss, val_score = validate(args, val_loader, model, criterion)            
            scheduler.step()
            
            print('loss %.4f - score %.4f - val_loss %.4f - val_score %.4f'
                  % (train_loss, train_score, val_loss, val_score))

            if val_loss < best_loss:
                torch.save(model.state_dict(), 'models/%s/model_%d.pth' % (args.name, fold+1))
                best_loss = val_loss
                best_score = val_score
                print("saved best model")

        print('val_loss:  %f' % best_loss)
        print('val_score: %f' % best_score)

        folds.append(str(fold + 1))
        best_losses.append(best_loss)
        best_scores.append(best_score)

        results = pd.DataFrame({
            'fold': folds + ['mean'],
            'best_loss': best_losses + [np.mean(best_losses)],
            'best_score': best_scores + [np.mean(best_scores)],
        })

        print(results)
        results.to_csv('models/%s/results.csv' % args.name, index=False)
        torch.cuda.empty_cache()

        if not args.cv:
            break


if __name__ == '__main__':
    main()
