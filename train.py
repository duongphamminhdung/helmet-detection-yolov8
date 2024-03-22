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

from data.transform import Augmentation, BaseTransform

from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Helmet Detection')
    parser.add_argument('-p', '--path', default=None, type=str, 
                        help='path to model')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    parser.add_argument('--save_period', type=int,
                            default=10, help='How long does the program save the model')
    parser.add_argument('--epochs', type=int,
                            default=100, help='Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.')
    parser.add_argument('--save_folder', default="", type=str, 
                        help='Save folder')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('-bs', '--batch_size', default=8, type=int, 
                        help='Batch size for training')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')

    parser.add_argument('--lr0', default=1e-3, type=float, 
                        help='Initial learning rate (i.e. SGD=1E-2, Adam=1E-3) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.')
    parser.add_argument('--lrf', default=1e-2, type=float, 
                        help='Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--seed', default=2506, type=int)
    parser.add_argument('--name', default=None, type=str)
    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    
    if args.cuda:
        print("Using CUDA")
        device = 0
    model_name = os.path.split(args.path)[-1]
    if model_name.endswith('.pt'):
        model_name = model_name[:-3]
    if args.seed is not None:
        imgaug.random.RNG(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    else:
        imgaug.random.RNG(0)
        random.seed(0)
        np.random.seed(0)
        

    model = YOLO(
        model=args.path,
        # save_period=10,
        # project=args.save_folder+model_name,
        # name=args.name,
        # exist_ok=True,
        # device = 0,
        # workers=args.num_workers,
        # seed=args.seed,
        # lr0=args.lr0,
        # lrf=args.lrf,
        # momentum=args.momentum,
        # weight_decay=args.weight_decay,
        # warmup_epochs=args.warmup_epochs,
    )
    results = model.train(data='/root/helmet-detection-yolov8/data/data.yaml', 
                            epochs=args.epochs, 
                            imgsz=640, 
                            save=True,
                            save_period=args.save_period,
                            project=args.name,
                            name=args.save_folder+model_name,
                            exist_ok=True,
                            device = 0,
                            workers=args.num_workers,
                            seed=args.seed,
                            lr0=args.lr0,
                            lrf=args.lrf,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            warmup_epochs=args.warmup_epochs,)



if __name__ == '__main__':
    train()
