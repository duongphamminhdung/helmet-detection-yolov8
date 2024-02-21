from __future__ import division

import numpy as np #for set the seed...
import os
import random
import argparse
import time
import imgaug

import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from data.transform import Augmentation, BaseTransform

from models.build import build_yolov2

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv2 Detection')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--eval_epoch', type=int,
                            default=5, help='interval between evaluations')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Save folder')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')

    parser.add_argument('-bs', '--batch_size', default=8, type=int, 
                        help='Batch size for training')
    parser.add_argument('-accu', '--accumulate', default=8, type=int, 
                        help='gradient accumulate.')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--max_epoch', type=int, default=200,
                        help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[100, 150], type=int,
                        help='lr epoch to decay')

    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--seed', default=2506, type=int, 
                        help='seed for iaa random')

    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('--root', default='/root/PyTorch_YOLOv2/setup',
                        help='data root')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

def build_dataloader(args, dataset):
    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )
    
    return dataloader


if __name__ == '__main__':
    train()