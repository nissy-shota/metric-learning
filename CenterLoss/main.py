import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from sklearn.metrics import classification_report
from torch.autograd.function import Function
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import trange



def main():

    arg_parser = argparse.ArgumentParser(description="parser for focus one")
    arg_parser.add_argument("--dataset_dir", type=str, default='../inputs/')
    arg_parser.add_argument("--model_path_temp", type=str, default='../outputs/models/checkpoints/mnist_original_softmax_center_epoch_{}.pth')
    arg_parser.add_argument("--vis_img_path_temp", type=str, default='../outputs/visual/epoch_{}.png')
    args = arg_parser.parse_args()



if __name__ == '__main__':
    main()