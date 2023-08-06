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
from split.split_simplenet import SplitSimplenet
import pdb
import os
from exp_constant import *
from split.util_mixmatch import training_loop as training_mixmatch
from split.attack.labelleakage.clusterattack_stages import collect_training
from sklearn.cluster import KMeans

skip_datasets = ['image-net']
save_pathes = ['output/fashion-mnist-smashed-knn-attack-simplenet.pickle',
               'output/cifar10-smashed-knn-attack-simplenet.pickle',
               'output/dogs-vs-cats-smashed-knn-attack-simplenet.pickle',
               'output/hist-smashed-knn-attack-simplenet.pickle',
               'output/intel-smashed-knn-attack-simplenet.pickle',
               'output/fruits-smashed-knn-attack-simplenet.pickle',
               'output/imagenet-smashed-knn-attack-simplenet.pickle',]

oriention = 'logger'
pick_indexes, pick_position, _ = get_layer_modules()
print_log(['test layer', pick_indexes], oriention=oriention)
known = 1

split_layer = 'layer' + str(pick_indexes[-2])
split_pos = pick_position[-2]

for (n_epoch, lr, batch_size, data_path, dataset, composer, save_path, metric) in \
        zip(params['simplenet']['epochs'], 
            params['simplenet']['lrs'], 
            params['simplenet']['batches'],
            test_pathes, test_datasets, 
            test_composers, save_pathes,
            params['simplenet']['metric']):
    if dataset in skip_datasets:
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

        # prepare sub models
        prev_model = nn.Sequential(*children[:pos+1])
        back_model = nn.Sequential(*children[pos+1:])

        # split resnet
        split_model = SplitSimplenet([prev_model, back_model],
                                     len(labels_set),
                                     lr=lr)

        client_pipeline, server_pipeline, total_pipeline = split_model.split(device)

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


        train_scores, test_scores = [], []
        for i in range(max_tries):
            known_labels, known_raw = None, None
            for j, (collect_data, labels) in enumerate(collects):
                indexes = np.arange(len(labels))
                np.random.shuffle(indexes)
                collect_data = collect_data[indexes]
                labels = labels[indexes]

                known_labels, known_raw, attack_score = bad_knn(collect_data, 
                                                                  labels,
                                                           metric, known,
                                                           known_labels,
                                                           known_raw)

                if j == 0:
                    # train loader
                    train_scores.append(attack_score)
                else:
                    test_scores.append(attack_score)
                print_log([layer_name, 'attack accuracy:', train_scores, test_scores], 
                          oriention=oriention)

        layer_attacks[layer_name] = [train_scores, test_scores]
        rm = ResultManagement(save_path)
        rm.save(layer_attacks)
        result = rm.load()
        print_log(['current performance:', result], oriention=oriention)
