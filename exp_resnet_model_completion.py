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
from split.resnet import get_layers, get_layer_modules, get_model_children
from split.resnet import surrogate_top
from split.split_resnet import SplitResnet
import pdb, copy
import os
from exp_constant import *
from split.util_mixmatch import training_loop as training_mixmatch
from split.attack.labelleakage.clusterattack_stages import collect_training
from sklearn.cluster import KMeans

skip_datasets = ['fruits', 'fashion-mnist', 'cifar10', 'dogs_vs_cats',
                 'hist_cancer_detection', 'intel_classification',]
save_pathes = ['output/fashion-mnist-mc-attack-resnet18.pickle',
               'output/cifar10-mc-attack-resnet18.pickle',
               'output/dogs-vs-cats-mc-attack-resnet18.pickle',
               'output/hist-mc-attack-resnet18.pickle',
               'output/intel-mc-attack-resnet18.pickle',
               'output/flowers-mc-attack-resnet18.pickle',
               'output/caltech256-mc-attack-resnet18.pickle',
               'output/imagenet-mc-attack-resnet18.pickle',]

oriention = 'logger'
pick_indexes, pick_position, _ = get_layer_modules(10, resnet=18)
print_log(['test layer', pick_indexes], oriention=oriention)
known = 1

split_layer = 'layer' + str(pick_indexes[-2])
split_pos = pick_position[-2]

for (n_epoch, lr, batch_size, data_path, dataset, composer, save_path, gen_model,
    pretrain, metric) in \
        zip(params['resnet18']['mc_epochs'], 
            params['resnet18']['lrs'], 
            params['resnet18']['batches'],
            test_pathes, test_datasets, 
            test_composers, save_pathes,
            params['resnet18']['models'],
            params['resnet18']['pretrain'],
            params['resnet18']['metric']):
    if dataset in skip_datasets:
        continue

    # load databset
    SplitDP.static_set_seed()
    data = ExpDataset(data_path, dataset, composer)
    train_loader, val_loader = data.load_dataset(batch_size)
    labels_set = data.load_labels_set()

    this_labeled_batch_size = min(labeled_batch_size, len(labels_set))
    this_unlabeled_batch_size = batch_size - this_labeled_batch_size

    imgs, _ = next(iter(train_loader))

    # load model
    SplitDP.static_set_seed()
    model = gen_model(len(labels_set), pretrain)
    children = get_model_children(model)

    # prepare sub models
    prev_model = nn.Sequential(*children[:split_pos+1])
    back_model = nn.Sequential(*children[split_pos+1:])

    # load save weights
    prev_model_name = prev_model_name_fmt.format(dataset, 'layer50')
    prev_state_dict = torch.load(prev_model_name)
    prev_model.load_state_dict(prev_state_dict['model'])
    back_model_name = back_model_name_fmt.format(dataset, 'layer50')
    back_state_dict = torch.load(back_model_name)
    back_model.load_state_dict(back_state_dict['model'])

    print_log(['test dataset:', dataset], oriention=oriention)
    layer_attacks = {}
    for index, pos in zip(pick_indexes[::-1], pick_position[::-1]):
        torch.cuda.empty_cache()

        layer_name = 'layer' + str(index)
        if os.path.exists(save_path):
            rm = ResultManagement(save_path)
            layer_attacks = rm.load()

        if layer_name in layer_attacks:
            continue

        train_scores, test_scores = [], []
        for i in range(max_tries):
            # prepare sub models
            prev_model = nn.Sequential(*copy.deepcopy(children[:pos+1]))
            inter_outputs = prev_model(imgs)
            inter_shape = inter_outputs.shape[1:]
            fake_top = surrogate_top(prev_model, inter_shape, len(labels_set))
            split_model = SplitResnet([prev_model, fake_top],
                                 len(labels_set),
                                 lr=lr,
                                 defense='mixmatch',)

            client_pipeline, server_pipeline, total_pipeline = split_model.split(device)

            labeled_loader, unlabeled_loader, val_loader = \
            data.load_dataset_partial_labeled(this_labeled_batch_size,
                                              this_unlabeled_batch_size,
                                              known_labels_per_class,
                                              drop_last=True)

            training_mixmatch(n_epoch, total_pipeline,
                      labeled_loader, unlabeled_loader, labels_set,
                      device, mixup_alpha, lamb_u,
                      temp)

            train_acc = validate(total_pipeline, train_loader, device)
            train_scores.append(train_acc)
            print_log([layer_name, 'train attack accuracy:', train_acc], oriention=oriention)
            val_acc = validate(total_pipeline, val_loader, device)
            test_scores.append(val_acc)
            print_log([layer_name, 'test attack accuracy:', val_acc], oriention=oriention)

        layer_attacks[layer_name] = [train_scores, test_scores]
        rm = ResultManagement(save_path)
        rm.save(layer_attacks)
        cur_data = rm.load()
        print_log(['current performance:', cur_data], oriention=oriention)
