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
from split.resnet import specific_resnet_tree_channels, specific_resnet_single_channel
import split.resnet as resnet50
from split.resnet import sweep_model, surrogate_top, sweep_front_model
from split.split_resnet import SplitResnet
import pdb
import os
from exp_constant import *
from split.util_mixmatch import training_loop as training_mixmatch
from split.util import validate
from split.attack.labelleakage.clusterattack_stages import collect_training
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans


# skip_datasets = ['fashion-mnist', 'cifar10']
skip_datasets = []
save_pathes = ['output/fashion-mnist-pruning-defense-resnet50-smash.pickle',
               'output/cifar10-pruning-defense-resnet50-smash.pickle',
               'output/dogs-vs-cats-pruning-defense-resnet50-smash.pickle',
               'output/hist-pruning-defense-resnet50-smash.pickle',
               'output/intel-pruning-defense-resnet50-smash.pickle',
               'output/flowers-pruning-defense-resnet50-smash.pickle',]

epochs = [10, 30, 10, 10, 10, 10]
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
# define test layers
layers = resnet50.get_layers()
layer_index, split_layer = layers[-3]
print('split layer:', split_layer)
defense = 'grad_pruning'

for n_epoch, lr, batch_size, data_path, dataset, composer, save_path, gen_model in \
        zip(epochs, params['resnet50']['lrs'], 
            params['resnet50']['batches'],
            test_pathes, test_datasets, 
            test_composers, save_pathes,
            params['resnet50']['models']):
    if dataset in skip_datasets:
        continue

    print('test dataset:', dataset)
    attacks = {}
    for threshold in thresholds:
        torch.cuda.empty_cache()

        if os.path.exists(save_path):
            rm = ResultManagement(save_path)
            attacks = rm.load()

        if threshold in attacks:
            continue

        defense_args = {
            'ratio': threshold,
        }

        # load databset
        SplitDP.static_set_seed()
        data = ExpDataset(data_path, dataset, composer)
        train_loader, val_loader = data.load_dataset(batch_size)
        labels_set = data.load_labels_set()
        imgs, _ = next(iter(train_loader))
        input_shape = np.array(imgs.shape[1:])

        # load model
        SplitDP.static_set_seed()
        model = gen_model(len(labels_set), True)

        # prepare sub models
        prev_model = sweep_model(model, split_layer, freeze=False)
        back_model = sweep_front_model(model, split_layer, freeze=False)

        # split simplenet
        split_model = SplitResnet([prev_model, back_model],
                                     len(labels_set),
                                     lr=lr,
                                     defense=defense,
                                     defense_args=defense_args)

        # client_pipeline, server_pipeline, total_pipeline, train_loader = split_model.split(device)
        client_pipeline, server_pipeline, total_pipeline = split_model.split(device)

        # begin to train
        training_loop(n_epoch, total_pipeline,
                      train_loader, device)

        orig_acc = validate(total_pipeline, val_loader, device)
        attacks[threshold] = [orig_acc]
        print(threshold, 'orig accuracy:', orig_acc)

        # collect smash
        cache_size = 512
        upper_dims = 256
        dims = max(upper_dims, batch_size)

        collects = []
        for loader in [train_loader, val_loader]:
            pca = IncrementalPCA(n_components=dims, batch_size=batch_size)
            for i in range(2):
                if i == 0:
                    train_pca = True
                else:
                    train_pca = False

                collect_data, labels = collect_training(total_pipeline,
                             loader,
                             device,
                             cache_size,
                             pca,
                             train_pca,
                             victim='smash')

                if collect_data is not None:
                    break

            collects.append((collect_data, labels))

        for collect_data, labels in collects:
            km = KMeans(n_clusters=len(labels_set))
            km.fit(collect_data)
            attack_acc = cluster_accuracy(labels, km.labels_)
            attacks[threshold].append(attack_acc)
            print(threshold, 'attack accuracy:', attack_acc)

        rm = ResultManagement(save_path)
        rm.save(attacks)
        data = rm.load()
        print('current performance:', data)

