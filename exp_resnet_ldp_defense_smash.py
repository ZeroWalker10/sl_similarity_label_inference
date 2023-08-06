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

from split.util_mixup import training_loop, validate
from split.util_ext import SplitDP, cluster_accuracy_based_on_known, ResultManagement
from split.util_ext import smash_cluster
from split.resnet import get_layers
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


skip_datasets = ['fashion-mnist', 'hist_cancer_detection',
                 'intel_classification', 'image-net']
save_pathes = ['output/fashion-mnist-ldp-defense-resnet18.pickle',
               'output/cifar10-ldp-defense-resnet18.pickle',
               'output/dogs-vs-cats-ldp-defense-resnet18.pickle',
               'output/hist-ldp-defense-resnet18.pickle',
               'output/intel-ldp-defense-resnet18.pickle',
               'output/flowers-ldp-defense-resnet18.pickle',
               'output/caltech256-ldp-defense-resnet18.pickle',
               'output/imagenet-ldp-defense-resnet18.pickle',]

epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
# define test layers
layers = get_layers(resnet=18)
layer_index, split_layer = layers[-3]
oriention = 'logger'
print_log(['split layer:', split_layer], oriention=oriention)
defense = 'rrlabel'
known = 1

for (n_epoch, lr, batch_size, data_path, dataset, composer, save_path, gen_model,
     pretrain, metric) in \
        zip(params['resnet18']['epochs'], 
            params['resnet18']['lrs'], 
            params['resnet18']['batches'],
            test_pathes, test_datasets, 
            test_composers, save_pathes,
            params['resnet18']['models'],
            params['resnet18']['pretrain'],
            params['resnet18']['metric']):
    if dataset in skip_datasets:
        continue

    print_log(['test dataset:', dataset], oriention=oriention)
    layer_attacks = {}
    for epsilon in epsilons[::-1]:
        torch.cuda.empty_cache()

        if os.path.exists(save_path):
            rm = ResultManagement(save_path)
            layer_attacks = rm.load()

        if epsilon in layer_attacks:
            continue

        defense_args = {
            'epsilon': epsilon,
            'prior': None
        }

        # load databset
        SplitDP.static_set_seed()
        data = ExpDataset(data_path, dataset, composer)
        train_loader, val_loader = data.load_dataset(batch_size)
        labels_set = data.load_labels_set()
        imgs, _ = next(iter(train_loader))
        input_shape = np.array(imgs.shape[1:])

        defense_args['prior'] = data.load_distribution()

        # load model
        SplitDP.static_set_seed()
        model = gen_model(len(labels_set), pretrain)

        # prepare sub models
        prev_model = sweep_model(model, split_layer, freeze=False)
        back_model = sweep_front_model(model, split_layer, freeze=False)

        # split simplenet
        split_model = SplitResnet([prev_model, back_model],
                                     len(labels_set),
                                     lr=lr,
                                     defense=defense,
                                     defense_args=defense_args)

        client_pipeline, server_pipeline, total_pipeline = split_model.split(device)

        training_loop(n_epoch, total_pipeline,
                      train_loader, device,
                      mixup_alpha=mixup_alpha)

        orig_acc = validate(total_pipeline, val_loader, device)
        layer_attacks[epsilon] = [orig_acc]
        print_log([epsilon, 'orig accuracy:', orig_acc], oriention=oriention)

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

        known_labels, known_raw = None, None
        for collect_data, labels in collects:
            km = KMeans(n_clusters=len(labels_set))
            km.fit(collect_data)
            known_labels, known_raw, attack_score = smash_cluster(collect_data, 
                                                                  labels,
                                                           labels_set, metric, known,
                                                           known_labels,
                                                           known_raw)

            layer_attacks[epsilon].append(attack_score)
            print_log([epsilon, 'attack accuracy:', attack_score], oriention=oriention)

        rm = ResultManagement(save_path)
        rm.save(layer_attacks)
        data = rm.load()
        print_log(['current performance:', data], oriention=oriention)
