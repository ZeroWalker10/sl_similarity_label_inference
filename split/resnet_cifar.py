#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class GlobalMaxPooling(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()

    def forward(self, inputs):
        out = F.max_pool2d(inputs, kernel_size=inputs.size()[2:])
        return out

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out

class ResNetCIFAR10(nn.Module):
    """
    A Residual network.
    """
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            
            ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        )

        self.classifier = nn.Sequential(
            GlobalMaxPooling(),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x

def get_model_children(model):
    children = []
    modules = model.children()
    for module in modules:
        if isinstance(module, torch.nn.Sequential):
            children.extend(list(module.children()))
        else:
            children.append(module)
    return children

def get_flatten_size(model, input_shape):
    x = torch.zeros(10, *input_shape)
    x = model(x)
    shape = x.shape
    return np.prod(shape[1:])

def surrogate_top(prev_model, input_shape, n_class):
    if len(input_shape) == 1:
        sub_model = nn.Sequential(
            nn.Linear(input_shape[0], input_shape[0]),
            nn.ReLU(),
            nn.Linear(input_shape[0], input_shape[0]),
            nn.ReLU(),
            nn.Linear(input_shape[0], n_class),
        )
    else:
        sub_model = nn.Sequential()
        if input_shape[0] > 1 and (input_shape[1] > 1 or input_shape[2] > 1):
            sub_model.add_module('max', GlobalMaxPooling())
        sub_model.add_module('flatten', nn.Flatten())
        flatten_size = get_flatten_size(sub_model, input_shape)
        sub_model.add_module('fc1',
                             nn.Linear(flatten_size, 256))
        sub_model.add_module('act1', nn.ReLU())
        sub_model.add_module('fc2',
                             nn.Linear(256, 128))
        sub_model.add_module('act2', nn.ReLU())
        sub_model.add_module('fc3',
                             nn.Linear(128, n_class))
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

def get_layer_modules(step=4):
    model = ResNetCIFAR10(10)
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
            if isinstance(children[position], nn.Linear):
                if position + 1 < len(children) and isinstance(children[position+1], nn.ReLU):
                    index = index + 1
                    position = position + 1
            elif position + 1 < len(children) and isinstance(children[position+1], GlobalMaxPooling):
                index = index + 1
                position = position + 1

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
