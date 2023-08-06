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

skip_datasets = ['fashion-mnist', 'intel_classification', 'image-net',
                 'cifar10', 'fruits']
save_pathes = ['output/fashion-mnist-smash-kl-defense-simplenet.pickle',
               'output/cifar10-smash-kl-defense-simplenet.pickle',
               'output/dogs-vs-cats-smash-kl-defense-simplenet.pickle',
               'output/hist-smash-kl-defense-simplenet.pickle',
               'output/intel-smash-kl-defense-simplenet.pickle',
               'output/fruits-smash-kl-defense-simplenet.pickle',
               'output/imagenet-smash-kl-defense-simplenet.pickle',]

epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
oriention = 'logger'
pick_indexes, pick_position, _ = get_layer_modules()
print_log(['test layer', pick_indexes], oriention=oriention)
split_index = pick_indexes[-2]
split_pos = pick_position[-2]

split_layer = 'layer' + str(split_index)
print_log(['split layer:', split_layer], oriention=oriention)
known = 1
defense = 'klperturb'

for (n_epoch, lr, batch_size, data_path, dataset, composer, save_path, metric) in \
        zip(params['simplenet']['epochs'], 
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
            'threshold': epsilon,
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

        cluster_train_scores, cluster_test_scores = [], []
        knn_train_scores, knn_test_scores = [], []
        for i in range(max_tries):
            known_labels, known_raw = None, None
            knn_known_labels, knn_known_raw = None, None
            for j, (collect_data, labels) in enumerate(collects):
                indexes = np.arange(len(labels))
                np.random.shuffle(indexes)
                collect_data = collect_data[indexes]
                labels = labels[indexes]

                km = KMeans(n_clusters=len(labels_set))
                km.fit(collect_data)
                known_labels, known_raw, cluster_attack_score = smash_cluster(collect_data, 
                                                                  labels,
                                                           labels_set, metric, known,
                                                           known_labels,
                                                           known_raw)

                if knn_known_labels is None:
                    knn_known_labels, knn_known_raw = [], []
                    for label, raw in known_raw.items():
                        knn_known_raw.extend(raw)
                        knn_known_labels.extend([label] * len(raw))

                knn_known_labels, knn_known_raw, knn_attack_score = bad_knn(collect_data, 
                                                                  labels,
                                                           metric, known,
                                                           knn_known_labels,
                                                           knn_known_raw)
                if j == 0:
                    # train loader
                    cluster_train_scores.append(cluster_attack_score)
                    knn_train_scores.append(knn_attack_score)
                else:
                    cluster_test_scores.append(cluster_attack_score)
                    knn_test_scores.append(knn_attack_score)
                print_log([epsilon, 'attack accuracy:', 
                           cluster_train_scores, cluster_test_scores,
                           knn_train_scores, knn_test_scores], 
                          oriention=oriention)

        layer_attacks[epsilon].append([cluster_train_scores, cluster_test_scores,
                                       knn_train_scores, knn_test_scores])
        rm = ResultManagement(save_path)
        rm.save(layer_attacks)
        result = rm.load()
        print_log(['current performance:', result], oriention=oriention)
