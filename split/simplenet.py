#!/usr/bin/env python
# coding=utf-8
import numpy as np
from PIL import Image
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import sys
sys.path.append('..')
from exp_dataset import ExpDataset
from .util_ext import SplitDP

class GlobalMaxPooling(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()
    
    def forward(self, inputs):
        out = F.max_pool2d(inputs, kernel_size=inputs.size()[2:]) 
        return out

def get_flatten_size(model, input_shape):
    x = torch.zeros(10, *input_shape)
    x = model(x)
    shape = x.shape
    return np.prod(shape[1:])

class SimpleNet(nn.Module):
    def _get_flatten_size(self):
        x = torch.zeros(10, *self.input_shape)
        for layer in self.features:
            x = layer(x)
        shape = x.shape
        return np.prod(shape[1:])
        
    def __init__(self, input_shape, n_class, drop_p=0.1):
        # input_shape: [channel, width, height]
        super(SimpleNet, self).__init__()
        
        # self.features = nn.Sequential()
        self.features = []
        # self.classifier = nn.Sequential()
        self.classifier = []
        self.input_shape = input_shape
        in_channel = self.input_shape[0]
        self.n_class = n_class
        
        self.layer1 = nn.Conv2d(in_channel, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        # self.features.add_module('layer1', self.layer1)
        self.features.append(self.layer1)
        self.layer2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer2', self.layer2)
        self.features.append(self.layer2)
        self.layer3 = nn.ReLU(inplace=True)
        # self.features.add_module('layer3', self.layer3)
        self.features.append(self.layer3)

        self.layer4 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        # self.features.add_module('layer4', self.layer4)
        self.features.append(self.layer4)
        self.layer5 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer5', self.layer5)
        self.features.append(self.layer5)
        self.layer6 = nn.ReLU(inplace=True)
        # self.features.add_module('layer6', self.layer6)
        self.features.append(self.layer6)

        self.layer7 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        # self.features.add_module('layer7', self.layer7)
        self.features.append(self.layer7)
        self.layer8 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer8', self.layer8)
        self.features.append(self.layer8)
        self.layer9 = nn.ReLU(inplace=True)
        # self.features.add_module('layer9', self.layer9)
        self.features.append(self.layer9)

        self.layer10 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        # self.features.add_module('layer10', self.layer10)
        self.features.append(self.layer10)
        self.layer11 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer11', self.layer11)
        self.features.append(self.layer11)
        self.layer12 = nn.ReLU(inplace=True)
        # self.features.add_module('layer12', self.layer12)
        self.features.append(self.layer12)

        self.layer13 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        # self.features.add_module('layer13', self.layer13)
        self.features.append(self.layer13)
        self.layer14 = nn.Dropout2d(p=drop_p)
        # self.features.add_module('layer14', self.layer14)
        self.features.append(self.layer14)

        self.layer15 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        # self.features.add_module('layer15', self.layer15)
        self.features.append(self.layer15)
        self.layer16 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer16', self.layer16)
        self.features.append(self.layer16)
        self.layer17 = nn.ReLU(inplace=True)
        # self.features.add_module('layer17', self.layer17)
        self.features.append(self.layer17)

        self.layer18 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        # self.features.add_module('layer18', self.layer18)
        self.features.append(self.layer18)
        self.layer19 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer19', self.layer19)
        self.features.append(self.layer19)
        self.layer20 = nn.ReLU(inplace=True)
        # self.features.add_module('layer20', self.layer20)
        self.features.append(self.layer20)

        self.layer21 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        # self.features.add_module('layer21', self.layer21)
        self.features.append(self.layer21)
        self.layer22 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer22', self.layer22)
        self.features.append(self.layer22)
        self.layer23 = nn.ReLU(inplace=True)
        # self.features.add_module('layer23', self.layer23)
        self.features.append(self.layer23)

        self.layer24 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        # self.features.add_module('layer24', self.layer24)
        self.features.append(self.layer24)
        self.layer25 = nn.Dropout2d(p=drop_p)
        # self.features.add_module('layer25', self.layer25)
        self.features.append(self.layer25)

        self.layer26 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        # self.features.add_module('layer26', self.layer26)
        self.features.append(self.layer26)
        self.layer27 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer27', self.layer27)
        self.features.append(self.layer27)
        self.layer28 = nn.ReLU(inplace=True)
        # self.features.add_module('layer28', self.layer28)
        self.features.append(self.layer28)

        self.layer29 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        # self.features.add_module('layer29', self.layer29)
        self.features.append(self.layer29)
        self.layer30 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer30', self.layer30)
        self.features.append(self.layer30)
        self.layer31 = nn.ReLU(inplace=True)
        # self.features.add_module('layer31', self.layer31)
        self.features.append(self.layer31)

        self.layer32 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        # self.features.add_module('layer32', self.layer32)
        self.features.append(self.layer32)
        self.layer33 = nn.Dropout2d(p=drop_p)
        # self.features.add_module('layer33', self.layer33)
        self.features.append(self.layer33)

        self.layer34 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        # self.features.add_module('layer34', self.layer34)
        self.features.append(self.layer34)
        self.layer35 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer35', self.layer35)
        self.features.append(self.layer35)
        self.layer36 = nn.ReLU(inplace=True)
        # self.features.add_module('layer36', self.layer36)
        self.features.append(self.layer36)

        self.layer37 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        # self.features.add_module('layer37', self.layer37)
        self.features.append(self.layer37)
        self.layer38 = nn.Dropout2d(p=drop_p)
        # self.features.add_module('layer38', self.layer38)
        self.features.append(self.layer38)

        self.layer39 = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0))
        # self.features.add_module('layer39', self.layer39)
        self.features.append(self.layer39)
        self.layer40 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True)
        # self.features.add_module('layer40', self.layer40)
        self.features.append(self.layer40)
        self.layer41 = nn.ReLU(inplace=True)
        # self.features.add_module('layer41', self.layer41)
        self.features.append(self.layer41)
        
        if self.n_class <= 128:
            self.layer42 = nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0))
            # self.features.add_module('layer42', self.layer42)
            self.features.append(self.layer42)
            self.layer43 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
            # self.features.add_module('layer43', self.layer43)
            self.features.append(self.layer43)
        elif self.n_class <= 300:
            self.layer42 = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0))
            # self.features.add_module('layer42', self.layer42)
            self.features.append(self.layer42)
            self.layer43 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True)
            # self.features.add_module('layer43', self.layer43)
            self.features.append(self.layer43)
        else:
            self.layer42 = nn.Conv2d(2048, 1024, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0))
            # self.features.add_module('layer42', self.layer42)
            self.features.append(self.layer42)
            self.layer43 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.05, affine=True)
            # self.features.add_module('layer43', self.layer43)
            self.features.append(self.layer43)
            
        self.layer44 = nn.ReLU(inplace=True)
        # self.features.add_module('layer44', self.layer44)
        self.features.append(self.layer44)

        self.layer45 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        # self.features.add_module('layer45', self.layer45)
        self.features.append(self.layer45)
        self.layer46 = nn.Dropout2d(p=drop_p)
        # self.features.add_module('layer46', self.layer46)
        self.features.append(self.layer46)

        if self.n_class <= 128:
            self.layer47 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
            # self.features.add_module('layer47', self.layer47)
            self.features.append(self.layer47)
            self.layer48 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
            # self.features.add_module('layer48', self.layer48)
            self.features.append(self.layer48)
        elif self.n_class <= 300:
            self.layer47 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
            # self.features.add_module('layer47', self.layer47)
            self.features.append(self.layer47)
            self.layer48 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True)
            # self.features.add_module('layer48', self.layer48)
            self.features.append(self.layer48)
        else:
            self.layer47 = nn.Conv2d(1024, 1024, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
            # self.features.add_module('layer47', self.layer47)
            self.features.append(self.layer47)
            self.layer48 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.05, affine=True)
            # self.features.add_module('layer48', self.layer48)
            self.features.append(self.layer48)
            
        self.layer49 = nn.ReLU(inplace=True)
        # self.features.add_module('layer49', self.layer49)
        self.features.append(self.layer49)

        
        self.layer50 = GlobalMaxPooling()
        # self.features.add_module('layer50', self.layer50)
        self.features.append(self.layer50)
        self.layer51 = nn.Dropout(p=drop_p)
        # self.features.add_module('layer51', self.layer51)
        self.features.append(self.layer51)

        self.layer52 = nn.Flatten()
        # self.features.add_module('layer51', self.layer51)
        self.features.append(self.layer52)
        
        
        flatten_size = self._get_flatten_size()
        
        self.layer53 = nn.Linear(flatten_size, flatten_size)
        self.classifier.append(self.layer53)
        self.layer54 = nn.Linear(flatten_size, flatten_size)
        self.classifier.append(self.layer54)
        self.layer55 = nn.Linear(flatten_size, self.n_class)
        self.classifier.append(self.layer55)
    
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    
    def _features(self, x):
        out = x
        for layer in self.features:
            out = layer(out)
        return out
    
    def _classifier(self, x):
        out = x
        for layer in self.classifier:
            out = layer(out)
        return out
    
    def forward(self, x):
        out = self._features(x)
        out = self._classifier(out)
        return out

def surrogate_top(prev_model, input_shape, n_class, drop_p=0.1):
    sub_model = nn.Sequential()
    if len(input_shape) == 1:
        sub_model.add_module('fc1', 
                             nn.Linear(input_shape[0], input_shape[0]))
        sub_model.add_module('fc2', 
                             nn.Linear(input_shape[0], input_shape[0]))
        sub_model.add_module('fc3', 
                             nn.Linear(input_shape[0], n_class))
    else:
        if input_shape[1:].numel() > 1:
            sub_model.add_module('gmp', GlobalMaxPooling())
        # sub_model.add_module('dp1', nn.Dropout(p=drop_p))
        sub_model.add_module('flatten', nn.Flatten())
        flatten_size = get_flatten_size(sub_model, input_shape)
        sub_model.add_module('fc1',
                             nn.Linear(flatten_size, flatten_size))
        sub_model.add_module('fc2',
                             nn.Linear(flatten_size, flatten_size))
        sub_model.add_module('fc3',
                             nn.Linear(flatten_size, n_class))
    return sub_model

def benign_surrogate_top(prev_model, input_shape, n_class, drop_p=0.1):
    sub_model = nn.Sequential()
    if len(input_shape) == 1:
        sub_model.add_module('fc1', 
                             nn.Linear(input_shape[0], n_class))
    else:
        if input_shape[0] > 1:
            sub_model.add_module('gmp', GlobalMaxPooling())
            sub_model.add_module('dp1', nn.Dropout(p=drop_p))
        sub_model.add_module('flatten', nn.Flatten())
        flatten_size = get_flatten_size(sub_model, input_shape)
        sub_model.add_module('fc1',
                             nn.Linear(flatten_size, n_class))
    return sub_model

def calc_layer_num(module):
    local_children = []
    queue = list(module.children())
    while len(queue) > 0:
        child = queue.pop(0)
        child_children = list(child.children())
        if len(child_children) > 0:
            queue.extend(child_children)
        else:
            local_children.append(child)

    if len(local_children) == 0:
        local_children.append(module)
    return len(local_children)

def get_model_children(model):
    children = []
    modules = model.children()
    for module in modules:
        if isinstance(module, torch.nn.Sequential):
            children.extend(list(module.children()))
        else:
            children.append(module)
            if isinstance(module, torch.nn.AdaptiveAvgPool2d):
                children.append(nn.Flatten())

    return children

def get_layer_modules(step=4):
    model = SimpleNet((1, 32, 32), 10)
    children = get_model_children(model)
    
    indexes = []
    index = 0
    for child in children:
        child_length = calc_layer_num(child)
        index += child_length
        indexes.append(index)

    pick_indexes, pick_position = [], []
    prev_index = 0
    position = 0
    delta = 0
    for index, child in zip(indexes, children):
        delta += index - prev_index
        if delta >= step and (not isinstance(child, nn.ReLU)) and \
                (not isinstance(child, nn.BatchNorm2d)):
            # AdaptiveAvgPool2d is meanless
            pick_indexes.append(index)
            pick_position.append(position)
            delta = 0

        prev_index = index
        position += 1

    if pick_indexes[-1] > indexes[-2]:
        # have chosen the last layer
        pick_indexes[-1] = indexes[-2]
        pick_position[-1] = len(indexes) - 2
    elif pick_indexes[-1] < indexes[-2]:
        # miss the penultimate layer
        pick_indexes.append(indexes[-2])
        pick_position.append(len(indexes) - 2)
    return pick_indexes, pick_position, children
