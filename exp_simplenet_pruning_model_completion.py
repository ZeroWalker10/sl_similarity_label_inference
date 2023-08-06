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
from split.util_ext import smash_cluster, bad_knn
from split.simplenet import get_layer_modules, get_model_children, SimpleNet
from split.simplenet import surrogate_top
from split.split_simplenet import SplitSimplenet
import pdb, copy
import os
from exp_constant import *
from split.util_mixmatch import training_loop as training_mixmatch
from split.attack.labelleakage.clusterattack_stages import collect_training
from sklearn.cluster import KMeans

skip_datasets = ['fashion-mnist', 'intel_classification', 'image-net']
save_pathes = ['output/fashion-mnist-mc-pruning-defense-simplenet.pickle',
               'output/cifar10-mc-pruning-defense-simplenet.pickle',
               'output/dogs-vs-cats-mc-pruning-defense-simplenet.pickle',
               'output/hist-mc-pruning-defense-simplenet.pickle',
               'output/intel-mc-pruning-defense-simplenet.pickle',
               'output/fruits-mc-pruning-defense-simplenet.pickle',
               'output/imagenet-mc-pruning-defense-simplenet.pickle',]

epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]
oriention = 'logger'
pick_indexes, pick_position, _ = get_layer_modules()
print_log(['test layer', pick_indexes], oriention=oriention)
split_index = pick_indexes[-2]
split_pos = pick_position[-2]

split_layer = 'layer' + str(split_index)
print_log(['split layer:', split_layer], oriention=oriention)
known = 1
defense = 'grad_pruning'

for (n_epoch, mc_epoch, lr, batch_size, data_path, dataset, composer, save_path, metric) in \
        zip(params['simplenet']['epochs'], 
            params['simplenet']['mc_epochs'], 
            params['simplenet']['lrs'], 
            params['simplenet']['batches'],
            test_pathes, test_datasets, 
            test_composers, save_pathes,
            params['simplenet']['metric']):
    if dataset in skip_datasets:
        continue

    print_log(['test dataset:', dataset], oriention=oriention)
    layer_attacks = {}
    for epsilon in epsilons:
        torch.cuda.empty_cache()

        if os.path.exists(save_path):
            rm = ResultManagement(save_path)
            layer_attacks = rm.load()

        if epsilon in layer_attacks:
            continue

        defense_args = {
            'ratio': epsilon,
        }

        # load databset
        SplitDP.static_set_seed()
        data = ExpDataset(data_path, dataset, composer)
        train_loader, val_loader = data.load_dataset(batch_size)
        labels_set = data.load_labels_set()

        this_labeled_batch_size = min(labeled_batch_size, len(labels_set))
        this_unlabeled_batch_size = batch_size - this_labeled_batch_size
        
        imgs, _ = next(iter(train_loader))
        input_shape = np.array(imgs.shape[1:])

        # load model
        SplitDP.static_set_seed()
        model = SimpleNet(input_shape, len(labels_set))
        children = get_model_children(model)

        # prepare sub models
        prev_model = nn.Sequential(*children[:split_pos+1])
        back_model = nn.Sequential(*children[split_pos+1:])

        # split resnet
        split_model = SplitSimplenet([prev_model, back_model],
                                     len(labels_set),
                                     lr=lr,
                                     defense=defense,
                                     defense_args=defense_args)

        client_pipeline, server_pipeline, total_pipeline = split_model.split(device)
        
        training_loop(n_epoch, total_pipeline,
                      train_loader, device)

        orig_acc = validate(total_pipeline, val_loader, device)
        layer_attacks[epsilon] = [orig_acc]
        print_log([epsilon, 'orig accuracy:', orig_acc], oriention=oriention)

        train_scores, test_scores = [], []
        for i in range(max_tries):
            # prepare sub models
            nprev_model = nn.Sequential(*copy.deepcopy(list(prev_model.children()))).cpu()
            inter_outputs = nprev_model(imgs)
            inter_shape = inter_outputs.shape[1:]
            fake_top = surrogate_top(nprev_model, inter_shape, len(labels_set))
            split_model = SplitSimplenet([nprev_model, fake_top],
                                 len(labels_set),
                                 lr=lr,
                                 defense='mixmatch',)
            client_pipeline, server_pipeline, total_pipeline = split_model.split(device)

            labeled_loader, unlabeled_loader, val_loader = \
            data.load_dataset_partial_labeled(this_labeled_batch_size,
                                              this_unlabeled_batch_size,
                                              known_labels_per_class,
                                              drop_last=True)

            training_mixmatch(mc_epoch, total_pipeline,
                      labeled_loader, unlabeled_loader, labels_set,
                      device, mixup_alpha, lamb_u,
                      temp)

            train_acc = validate(total_pipeline, train_loader, device)
            train_scores.append(train_acc)
            print_log([epsilon, 'train attack accuracy:', train_acc], oriention=oriention)
            val_acc = validate(total_pipeline, val_loader, device)
            test_scores.append(val_acc)
            print_log([epsilon, 'test attack accuracy:', val_acc], oriention=oriention)

        layer_attacks[epsilon] = [train_scores, test_scores]
        rm = ResultManagement(save_path)
        rm.save(layer_attacks)
        result = rm.load()
        print_log(['current performance:', result], oriention=oriention)
