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
from split.attack.labelleakage.random_guess import random_guess_attack
import pdb
import os
from exp_constant import *
from split.util_mixmatch import training_loop as training_mixmatch
from split.attack.labelleakage.clusterattack_stages import collect_training
from sklearn.cluster import KMeans

skip_datasets = []
save_pathes = ['output/fashion-mnist-guess-attack-simplenet.pickle',
               'output/cifar10-guess-attack-simplenet.pickle',
               'output/dogs-vs-cats-guess-attack-simplenet.pickle',
               'output/hist-guess-attack-simplenet.pickle',
               'output/intel-guess-attack-simplenet.pickle',
               'output/fruits-guess-attack-simplenet.pickle',
               'output/imagenet-guess-attack-simplenet.pickle',]

oriention = 'logger'
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
    if os.path.exists(save_path):
        rm = ResultManagement(save_path)
        layer_attacks = rm.load()

    if dataset in layer_attacks:
        continue

    # load databset
    SplitDP.static_set_seed()
    data = ExpDataset(data_path, dataset, composer)
    train_loader, val_loader = data.load_dataset(batch_size)
    labels_set = data.load_labels_set()
    
    train_scores, test_scores = [], []
    for i in range(max_tries):
        pred_labels, true_labels = random_guess_attack(train_loader, labels_set, device)
        attack_score = metric(true_labels, pred_labels)
        train_scores.append(attack_score)

        pred_labels, true_labels = random_guess_attack(val_loader, labels_set, device)
        attack_score = metric(true_labels, pred_labels)
        test_scores.append(attack_score)
        print_log([dataset, 'attack accuracy:', train_scores, test_scores], 
                   oriention=oriention)

    layer_attacks[dataset] = [train_scores, test_scores]
    rm = ResultManagement(save_path)
    rm.save(layer_attacks)
    result = rm.load()
    print_log(['current performance:', result], oriention=oriention)
