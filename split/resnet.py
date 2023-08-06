#!/usr/bin/env python
# coding=utf-8
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import alexnet, resnet18, inception_v3, vgg16, densenet121, squeezenet1_1
from torchvision.models import vgg19, resnet50, resnet101, densenet121, resnet34, wide_resnet50_2
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

def specific_resnet101_three_channels(num_classes, pretrained=True):
    model = resnet101(pretrained=pretrained)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, num_classes)
    )
    return model

def specific_resnet50_three_channels(num_classes, pretrained=True):
    model = resnet50(pretrained=pretrained)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, num_classes)
    )
    return model

def specific_wresnet50_three_channels(num_classes, pretrained=True):
    model = wide_resnet50_2(pretrained=pretrained)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model

def specific_resnet50_single_channel(n_class, pretrained=True):
    model = resnet50(pretrained=pretrained)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(2048, n_class)
    return model

def specific_resnet18_three_channels(n_class, pretrain=True):
    model = resnet18(pretrained=pretrain)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
    model.fc = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.Linear(2048, 2048),
        nn.Linear(2048, n_class)
    )
    '''
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.Linear(512, 512),
        nn.Linear(512, n_class)
    )
    '''
    return model

def specific_resnet18_single_channel(n_class, pretrain=True):
    model = resnet18(pretrained=pretrain)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
    model.fc = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.Linear(2048, 2048),
        nn.Linear(2048, n_class)
    )
    '''
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.Linear(512, 512),
        nn.Linear(512, n_class)
    )
    '''
    return model

def surrogate_top(prev_model, input_shape, n_class):
    sub_model = nn.Sequential()
    if len(input_shape) == 1:
        sub_model.add_module('fc1', 
                             nn.Linear(input_shape[0], input_shape[0]))
        sub_model.add_module('act1', nn.ReLU())
        sub_model.add_module('fc2', 
                             nn.Linear(input_shape[0], input_shape[0]))
        sub_model.add_module('act2', nn.ReLU())
        sub_model.add_module('fc3', 
                             nn.Linear(input_shape[0], n_class))
    else:
        if input_shape[0] > 1 and (input_shape[1] > 1 or input_shape[2] > 1):
            sub_model.add_module('avg', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        sub_model.add_module('flatten', nn.Flatten())
        flatten_size = get_flatten_size(sub_model, input_shape)
        sub_model.add_module('fc1',
                             nn.Linear(flatten_size, flatten_size))
        sub_model.add_module('act1', nn.ReLU())
        sub_model.add_module('fc2',
                             nn.Linear(flatten_size, flatten_size))
        sub_model.add_module('act2', nn.ReLU())
        sub_model.add_module('fc3',
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

def get_layer_modules(label_num, resnet=18, step=4):
    if resnet == 50:
        model = specific_resnet50_three_channels(label_num)
    elif resnet == 101:
        model = specific_resnet101_three_channels(label_num)
    elif resnet == 18:
        model = specific_resnet18_three_channels(label_num)
    else:
        raise 'Invalid resnet choice!!!'


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

def get_layers(resnet=50):
    if resnet == 50:
        # dummy label number, just to generate model
        model = specific_resnet50_three_channels(10, step=16)
    elif resnet == 101:
        # dummy label number, just to generate model
        model = specific_resnet101_three_channels(10, step=8)
    elif resnet == 18:
        # dummy label number, just to generate model
        model = specific_resnet18_three_channels(10, step=4)
    else:
        raise 'Invalid resnet choice!!!'

    layer_candidates = []
    module_candidates = []
    for layer_name, module in list(model.named_modules())[1:]:
        if isinstance(module, nn.Sequential) or isinstance(module, torchvision.models.resnet.Bottleneck) \
                    or isinstance(module, torchvision.models.resnet.BasicBlock) \
                or isinstance(module, nn.AdaptiveAvgPool2d):
            continue
        layer_candidates.append(layer_name)
        module_candidates.append(module)

    layers = []
    prev_major, prev_minor = None, None
    for i, (layer_name, module) in enumerate(zip(layer_candidates, module_candidates)):
        if i == 0:
            continue

        if not layer_name.startswith('layer'):
            if not isinstance(module, nn.ReLU) and not isinstance(module, nn.BatchNorm2d):
                layers.append((i + 1, layer_name))
            continue
    
        tokens = layer_name[len('layer'):].split('.')
        if len(tokens) == 3:
            major, minor, spec_layer = tokens
        elif len(tokens) == 4:
            major, minor, spec_layer, spec_index = tokens
    
        if major != prev_major:
            if not isinstance(module, nn.ReLU) and not isinstance(module, nn.BatchNorm2d):
                layers.append((i + 1, layer_name))
        elif minor != prev_minor:
            if not isinstance(module, nn.ReLU) and not isinstance(module, nn.BatchNorm2d):
                layers.append((i + 1, layer_name))
        prev_major, prev_minor = major, minor
    return layers
