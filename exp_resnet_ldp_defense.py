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
from split.resnet import get_layer_modules, get_model_children
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
                 'intel_classification', 'image-net', 'caltech256']
save_pathes = ['output/fashion-mnist-ldp-defense-resnet18.pickle',
               'output/cifar10-ldp-defense-resnet18.pickle',
               'output/dogs-vs-cats-ldp-defense-resnet18.pickle',
               'output/hist-ldp-defense-resnet18.pickle',
               'output/intel-ldp-defense-resnet18.pickle',
               'output/flowers-ldp-defense-resnet18.pickle',
               'output/caltech256-ldp-defense-resnet18.pickle',
               'output/imagenet-ldp-defense-resnet18.pickle',]

epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
oriention = 'logger'

# define test layers
pick_indexes, pick_position, _ = get_layer_modules(10, resnet=18)
print_log(['test layer', pick_indexes], oriention=oriention)
split_index = pick_indexes[-2]
split_pos = pick_position[-2]
split_layer = 'layer' + str(split_index)

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

        dims = max(upper_dims, batch_size)
        pca = IncrementalPCA(n_components=dims, batch_size=batch_size)
        for i in range(2):
            # load model
            SplitDP.static_set_seed()
            model = gen_model(len(labels_set), pretrain)
            children = get_model_children(model)

            # prepare sub models
            prev_model = nn.Sequential(*children[:split_pos+1])
            back_model = nn.Sequential(*children[split_pos+1:])

            # split simplenet
            split_model = SplitResnet([prev_model, back_model],
                                     len(labels_set),
                                     lr=lr,
                                     defense=defense,
                                     defense_args=defense_args)
            client_pipeline, server_pipeline, total_pipeline = split_model.split(device)

            if i == 0:
                train_pca = True
            else:
                train_pca = False

            collect_data, labels = collect_training(total_pipeline,
                             train_loader,
                             device,
                             cache_size,
                             pca,
                             train_pca,
                             victim='gradient',
                             mixup_alpha=mixup_alpha)

            if collect_data is not None:
                break

        scores = []
        for i in range(max_tries):
            rindexes = np.arange(len(labels))
            np.random.shuffle(rindexes)
            labels = labels[rindexes]
            collect_data = collect_data[rindexes]

            km = KMeans(n_clusters=len(labels_set))
            km.fit(collect_data)
            known_labels, pred_labels = cluster_accuracy_based_on_known(labels, 
                                                                        km.labels_,
                                                                    known=known)
            attack_score = metric(labels, pred_labels)
            print_log([epsilon, 'attack accuracy:', attack_score], oriention=oriention)
            scores.append(attack_score)

        layer_attacks[epsilon] = scores
        rm = ResultManagement(save_path)
        rm.save(layer_attacks)
        data = rm.load()
        print_log(['current performance:', data], oriention=oriention)
