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

from split.util import training_loop, validate, grads_training, training_loop_pick_enjoy
from split.util_ext import SplitDP, cluster_accuracy, ResultManagement
from split.util_ext import CollecterExt
from split.resnet import specific_resnet_tree_channels, specific_resnet_single_channel
import split.simplenet as simplenet
from split.simplenet import SimpleNet, sweep_model, surrogate_top, sweep_front_model
from split.split_simplenet import SplitSimplenet
import pdb
import os
from exp_constant import *
from split.util_mixmatch import training_loop as training_mixmatch
from split.util import validate

# skip_datasets = ['fashion-mnist', 'cifar10']
skip_datasets = []
save_pathes = ['output/fashion-mnist-defense-simplenet.pickle',
               'output/cifar10-defense-simplenet.pickle',
               'output/dogs-vs-cats-defense-simplenet.pickle',
               'output/hist-defense-simplenet.pickle',
               'output/intel-defense-simplenet.pickle',
               'output/flowers-defense-simplenet.pickle',]

epochs = [30, 30, 30, 30, 30, 30]
milestones = [[10, 15], [10, 15], [10, 15], [10, 15], [10, 15], [10, 15]]
# define test layers
layers = simplenet.get_layers()
split_layer = layers[-3]
print('split layer:', split_layer)

for n_epoch, lr, batch_size, data_path, dataset, composer, save_path, ms in \
        zip(epochs, params['simplenet']['lrs'], 
            params['simplenet']['batches'],
            test_pathes, test_datasets, 
            test_composers, save_pathes, milestones):
    if dataset in skip_datasets:
        continue

    layer_attacks = {}
    if os.path.exists(save_path):
        rm = ResultManagement(save_path)
        layer_attacks = rm.load()

    print('test dataset:', dataset)
    for defense, defense_arg in zip(defenses, defense_args):
        torch.cuda.empty_cache()

        if defense in layer_attacks:
            continue

        # load databset
        SplitDP.static_set_seed()
        data = ExpDataset(data_path, dataset, composer)
        train_loader, val_loader = data.load_dataset(batch_size, drop_last=True)
        labels_set = data.load_labels_set()
        imgs, _ = next(iter(train_loader))
        input_shape = np.array(imgs.shape[1:])
        
        # load model
        SplitDP.static_set_seed()
        model = SimpleNet(input_shape, len(labels_set))

        # prepare sub models
        prev_model = sweep_model(model, split_layer, freeze=False)
        back_model = sweep_front_model(model, split_layer, freeze=False)
        pdb.set_trace()

        # split simplenet
        split_model = SplitSimplenet([prev_model, fake_top],
                                     len(labels_set),
                                     lr=lr,
                                     defense=defense,
                                     defense_args={'ms': {'milestones': ms}})
        client_pipeline, server_pipeline, total_pipeline = split_model.split(device)
   
        rm = ResultManagement(save_path)
        rm.save(layer_attacks)
        data = rm.load()
        print('current performance:', data)
