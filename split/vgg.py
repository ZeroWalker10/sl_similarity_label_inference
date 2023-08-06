#!/usr/bin/env python
# coding=utf-8
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import alexnet, resnet18, inception_v3, vgg16, densenet121, squeezenet1_1
from torchvision.models import vgg19, resnet50, resnet101, densenet121, resnet34
import torchvision

import numpy as np
import sys, copy, pdb
sys.path.append('..')
from exp_dataset import ExpDataset
from .util_ext import SplitDP 

def get_flatten_size(model, input_shape):
    x = torch.zeros(10, *input_shape)
    x = model(x)
    shape = x.shape
    return np.prod(shape[1:])

def specific_vgg16_three_channels(n_class, pretrained=True):
    model = vgg16(pretrained=pretrained)
    return model

def specific_vgg16_single_channel(n_class, pretrained=True):
    model = vgg16(pretrained=pretrained)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    return model

def surrogate_top(prev_model, input_shape, n_class):
    sub_model = nn.Sequential()
    if len(input_shape) == 1:
        sub_model.add_module('fc1', 
                             nn.Linear(input_shape[0], 4096))
        sub_model.add_module('fc2', 
                             nn.Linear(4096, n_class))
    else:
        if input_shape[0] > 1 and (input_shape[1] > 1 or input_shape[2] > 1):
            sub_model.add_module('avg', nn.AdaptiveAvgPool2d(output_size=(7, 7)))
        sub_model.add_module('flatten', nn.Flatten())
        flatten_size = get_flatten_size(sub_model, input_shape)
        sub_model.add_module('fc1',
                             nn.Linear(flatten_size, 4096))
        sub_model.add_module('fc2',
                             nn.Linear(4096, n_class))
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

def get_layer_modules(label_num, step=4):
    model = specific_vgg16_three_channels(label_num)
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
        if delta >= step:
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
