#!/usr/bin/env python
# coding=utf-8
from split.split_mnist import SplitMnist
# from split.split_cifar import SplitCifar
from split.split_cifar_simplenet import SplitCifar
from split.split_dogs2cats import SplitDC
from split.split_hist_cancer import SplitHist
from split.split_imagenet import SplitImagenet
import torch

class ExpModel:
    def __init__(self, name, defense=None, defense_args=None):
        self.name = name
        self.model = None
        self.defense = defense
        self.defense_args = defense_args

        if self.name == 'mnist':
            self.model = SplitMnist(1, 576, 10, self.defense, self.defense_args)
        elif self.name == 'cifar10':
            # self.model = SplitCifar(3, 4096, 10, self.defense, self.defense_args)
            self.model = SplitCifar(3, 256, 10, self.defense, self.defense_args)
        elif self.name == 'cifar100':
            # self.model = SplitCifar(3, 4096, 100, self.defense, self.defense_args)
            self.model = SplitCifar(3, 256, 100, self.defense, self.defense_args)
        elif self.name == 'dogs_vs_cats':
            self.model = SplitDC(3, 8192, 2, self.defense, self.defense_args)
        elif self.name == 'hist_cancer_detection':
            self.model = SplitHist(3, 4096, 2, self.defense, self.defense_args)
        elif self.name == 'image-net':
            self.model = SplitImagenet(3, 4096, 1000, self.defense, self.defense_args)
        else:
            raise 'invalid model name!!!'

    def split_model(self, device, top_layers=1):
        if top_layers >= self.model.layer_depth():
            raise 'the number of layers of top model should less than {}'.format(self.model.layer_depth())

        cut_layers = [
            (self.model.layer_depth() - top_layers, 'client', device),
            (self.model.layer_depth(), 'server', device),
        ]

        return self.model.split(cut_layers)

    def total_layers(self):
        return self.model.layer_depth()


if __name__ == '__main__':
    names = ['mnist', 'cifar10', 'cifar100', 'dogs_vs_cats', 'hist_cancer_detection']
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    for name in names:
        m = ExpModel(name)
        client_pipeline, server_pipeline, total_pipeline = m.split_model(device)
        for entry in total_pipeline.entries:
            print(entry.role.sub_model)
