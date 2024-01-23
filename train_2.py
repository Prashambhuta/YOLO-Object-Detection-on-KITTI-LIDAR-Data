import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import logging
# from logger import Logger
import datetime
import os
import argparse

# from model import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss
from ultralytics import YOLO

num_classes = 8
num_anchors = 5
batch_size = 12

train_data_path = os.path.join("../data")
dataset = KittiDataset(root=train_data_path, set='train')
data_loader = data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=False)

model = YOLO()
model.train(data=data_loader)