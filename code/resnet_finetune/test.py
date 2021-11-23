import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# import cv2
from torchvision import datasets, models, transforms

from fer import FER2013

resnet_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]),
}

# test_path = './faces_299'
batch_size = 20
model_path = './output/best.pkl'
torch.backends.cudnn.benchmark = True

test_dataset = FER2013(split='PrivateTest',
                       transform=resnet_data_transforms['val'])
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)


# test_dataset = torchvision.datasets.ImageFolder(test_path, transform=resnet_data_transforms['val'])
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)

test_dataset_size = len(test_dataset)
corrects = 0
acc = 0

model = models.resnet18()
num_fc_ftr = model.fc.in_features
model.fc = nn.Linear(num_fc_ftr, 7)

model.load_state_dict(torch.load(model_path))
# if isinstance(model, torch.nn.DataParallel):
# 	model = model.module

model = model.cuda()
model.eval()
print('start')
with torch.no_grad():
    for (image, labels) in test_loader:
        # print('into loop')
        image = image.cuda()
        labels = labels.cuda()
        outputs = model(image)
        _, preds = torch.max(outputs.data, 1)
        corrects += torch.sum(preds == labels.data).to(torch.float32)
        print('Iteration Acc {:.4f}'.format(
            torch.sum(preds == labels.data).to(torch.float32)/batch_size))
    acc = corrects / test_dataset_size
    print('Test Acc: {:.4f}'.format(acc))
