import argparse
import os
import yaml

from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim

import models
import dataset

def calc_score(output_list, target_list, running_loss, data_loader):
    # Calculate accuracy.
    result = classification_report(output_list, target_list, output_dict=True)
    acc = round(result['weighted avg']['f1-score'], 6)
    loss = round(running_loss / len(data_loader.dataset), 6)

    return acc, loss

def train(model, device, train_loader, criterion, optimizer):
    model.train()

    output_list = []
    target_list = []
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward processing.
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward processing.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Set data to calculate score.
        output_list += [int(o.argmax()) for o in outputs]
        target_list += [int(t) for t in targets]
        running_loss += loss.item()

        # Calculate score at present.
        train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)
        if batch_idx % 10 == 0 and batch_idx != 0:
            stdout_temp = 'batch: {:>3}/{:<3}, train acc: {:<8}, train loss: {:<8}'
            print(stdout_temp.format(batch_idx, len(train_loader), train_acc, train_loss))

    # Calculate score.
    train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)

    return train_acc, train_loss

def test(model, device, test_loader, criterion):
    model.eval()

    output_list = []
    target_list = []
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # Forward processing.
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Set data to calculate score.
        output_list += [int(o.argmax()) for o in outputs]
        target_list += [int(t) for t in targets]
        running_loss += loss.item()

    test_acc, test_loss = calc_score(output_list, target_list, running_loss, test_loader)

    return test_acc, test_loss

def main():

    parser = argparse.ArgumentParser(description='cifar10 model by ResNet50')
    parser.add_argument("--yaml_file", type=str, default='./config.yaml')
    args = parser.parse_args()

    yaml_file = args.yaml_file
    with open(yaml_file) as stream:
        config = yaml.safe_load(stream)

    data_dir = config['data_directory']
    dataset_name = config['dataset']['data_name']
    model_name = config['model']['model_name']
    model_ckpt_dir = config['model_directory']

    os.makedirs(model_ckpt_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset.
    train_loader, test_loader, class_names = dataset.load_data(data_dir)

    model = models.resnet50()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay'])

    # Train and test.
    epochs = config['training']['epochs']
    for epoch in range(epochs+1):
        # Train and test a model.
        train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)
        test_acc, test_loss = test(model, device, test_loader, criterion)

        # Output score.
        stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
        print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))

        # Save a model checkpoint.
        model_ckpt_path_temp = './experiments/models/checkpoints/{}_{}_epoch={}.pth'
        model_ckpt_path = model_ckpt_path_temp.format(dataset_name, model_name, epoch+1)
        torch.save(model.state_dict(), model_ckpt_path)
        print('Saved a model checkpoint at {}'.format(model_ckpt_path))
        print('')

if __name__ == '__main__':
    main()

