#!/usr/bin/env python
# coding=utf-8
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from importlib import reload
import exp_dataset
import split.util_ext
reload(exp_dataset)
reload(split.util_ext)

from exp_dataset import ExpDataset
from sklearn.decomposition import IncrementalPCA

from split.util_mixup import training_loop, validate
from split.util_ext import SplitDP, cluster_accuracy_based_on_known, ResultManagement
from split.util_ext import smash_cluster
from split.simplenet import get_layer_modules, get_model_children, SimpleNet
from split.split_simplenet import SplitSimplenet
import pdb
import os
from exp_constant import *
from split.util_mixmatch import training_loop as training_mixmatch
from split.attack.labelleakage.clusterattack_stages import collect_training
from sklearn.cluster import KMeans

skip_datasets = ['image-net']
save_pathes = ['output/fashion-mnist-base-simplenet.pickle',
               'output/cifar10-base-simplenet.pickle',
               'output/dogs-vs-cats-base-simplenet.pickle',
               'output/hist-base-simplenet.pickle',
               'output/intel-base-simplenet.pickle',
               'output/flowers-base-simplenet.pickle',
               'output/imagenet-base-simplenet.pickle',]

oriention = 'logger'
pick_indexes, pick_position, _ = get_layer_modules()
print_log(['test layer', pick_indexes], oriention=oriention)
split_index = pick_indexes[-2]
split_pos = pick_position[-2]

for (n_epoch, lr, batch_size, data_path, dataset, composer, save_path, metric) in \
        zip(params['simplenet']['epochs'], 
            params['simplenet']['lrs'], 
            params['simplenet']['batches'],
            test_pathes, test_datasets, 
            test_composers, save_pathes,
            params['resnet18']['metric']):
    if dataset in skip_datasets:
        continue

    print_log(['test dataset:', dataset], oriention=oriention)
    layer_attacks = {}
    torch.cuda.empty_cache()

    layer_name = 'layer' + str(split_index)
    if os.path.exists(save_path):
        rm = ResultManagement(save_path)
        layer_attacks = rm.load()

    if layer_name in layer_attacks:
        continue

    # load databset
    SplitDP.static_set_seed()
    data = ExpDataset(data_path, dataset, composer)
    train_loader, val_loader = data.load_dataset(batch_size)
    labels_set = data.load_labels_set()

    imgs, _ = next(iter(train_loader))
    input_shape = np.array(imgs.shape[1:])

    # load model
    SplitDP.static_set_seed()
    model = SimpleNet(input_shape, len(labels_set))
    children = get_model_children(model)

    # prepare sub models
    prev_model = nn.Sequential(*children[:split_pos+1])
    back_model = nn.Sequential(*children[split_pos+1:])
    pdb.set_trace()

    # split simplenet
    split_model = SplitSimplenet([prev_model, back_model],
                                     len(labels_set),
                                     lr=lr)

    client_pipeline, server_pipeline, total_pipeline = split_model.split(device)

    training_loop(n_epoch, total_pipeline,
                      train_loader, device)

    prev_model_name = prev_model_name_fmt.format(dataset, layer_name)
    torch.save({'model': prev_model.state_dict()}, prev_model_name)
    back_model_name = back_model_name_fmt.format(dataset, layer_name)
    torch.save({'model': back_model.state_dict()}, back_model_name)

    orig_acc = validate(total_pipeline, val_loader, device)
    layer_attacks[layer_name] = [orig_acc]
    print_log([layer_name, 'orig accuracy:', orig_acc], oriention=oriention)

    rm = ResultManagement(save_path)
    rm.save(layer_attacks)
    data = rm.load()
    print_log(['current performance:', data], oriention=oriention)
