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
from torchinfo import summary
from sklearn.metrics import classification_report
from torch.autograd.function import Function
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets, transforms
from tqdm import trange

import MNIST_utils
import models
import loss_original

DEVEICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(device, train_loader, model, nllloss, loss_weight, centerloss, dnn_optimizer, center_optimizer):
    running_loss = 0.0
    pred_list = []
    label_list = []
    ip1_loader = []
    idx_loader = []

    model.train()
    for i,(imgs, labels) in enumerate(train_loader):
        # Set batch data.
        imgs, labels = imgs.to(device), labels.to(device)
        # Predict labels.
        ip1, pred = model(imgs)
        # Calculate loss.
        loss = nllloss(pred, labels) + loss_weight * centerloss(labels, ip1)
        # Initilize gradient.
        dnn_optimizer.zero_grad()
        center_optimizer.zero_grad()
        # Calculate gradient.
        loss.backward()
        # Update parameters.
        dnn_optimizer.step()
        center_optimizer.step()
        # For calculation.
        running_loss += loss.item()
        pred_list += [int(p.argmax()) for p in pred]
        label_list += [int(l) for l in labels]
        # For visualization.
        ip1_loader.append(ip1)
        idx_loader.append((labels))

    # Calculate training accurary and loss.
    result = classification_report(pred_list, label_list, output_dict=True)
    train_acc = round(result['weighted avg']['f1-score'], 6)
    train_loss = round(running_loss / len(train_loader.dataset), 6)

    # Concatinate features and labels.
    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)

    return train_acc, train_loss, feat, labels

def main():

    # arg_parser = argparse.ArgumentParser(description="parser for focus one")
    # arg_parser.add_argument("--dataset_dir", type=str, default='../inputs/')
    # arg_parser.add_argument("--model_path_temp", type=str, default='../outputs/models/checkpoints/mnist_original_softmax_center_epoch_{}.pth')
    # arg_parser.add_argument("--vis_img_path_temp", type=str, default='../outputs/visual/epoch_{}.png')
    # args = arg_parser.parse_args()

    # load data
    dataset_dir = '/home/shota/Research/metric-learning/data'
    train_loader, test_loader, classes = MNIST_utils.load_dataset(dataset_dir)
    # MNIST_utils.show_data(train_loader)
    # model
    model = models.ResNet18()
    model.to(DEVEICE)
    # print(model)

#     loss
    nllloss = nn.NLLLoss()
    loss_weight = 1
    centerloss = loss_original.CenterLoss(10, 2)

    # Optimizer
    dnn_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    sheduler = lr_scheduler.StepLR(dnn_optimizer, 20, gamma=0.8)
    center_optimizer = optim.SGD(centerloss.parameters(), lr =0.5)

    print('Start training...')

    for epoch in range(100):
        # Update parameters.
        epoch += 1
        # sheduler.step()

        # Train and test a model.
        train_acc, train_loss, feat, labels = train(DEVEICE, train_loader, model, nllloss, loss_weight, centerloss, dnn_optimizer, center_optimizer)
        # test_acc, test_loss = test(device, test_loader, model, nllloss, loss_weight, centerloss)
        stdout_temp = 'Epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
        # print(stdout_temp.format(epoch, train_acc, train_loss, test_acc, test_loss))



if __name__ == '__main__':
    main()
