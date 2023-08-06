#!/usr/bin/env python
# coding=utf-8
from dataset.custom_dataset import CustomDataset
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from split.defense.labelleakage.antipodes.alibi import Ohm, RandomizedLabelPrivacy, NoisedCIFAR
import pdb

class ExpDatasetLDP(CustomDataset):
    def __init__(self, datapath, name, ldp_args=None):
        super(ExpDatasetLDP, self).__init__(datapath)
        self.name = name

        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.labels_set = None
        self.ldp_args = ldp_args
        if self.name == 'mnist':
            self.load_mnist()
        elif self.name == 'fashion-mnist':
            self.load_fashion_mnist()
        elif self.name == 'dogs_vs_cats':
            self.load_dogs_vs_cats()
        elif self.name == 'cifar10':
            self.load_cifar10()
        elif self.name == 'cifar100':
            self.load_cifar100()
        elif self.name == 'hist_cancer_detection':
            self.load_hist_cancer_detection()
        else:
            raise 'invalid dataset name!!!'

        if self.ldp_args['name'] == 'alibi':
            self._alibi_ldp()

    def _alibi_ldp(self):
        self.randomized_label_privacy = RandomizedLabelPrivacy(
            sigma=self.ldp_args['sigma'],
            delta=self.ldp_args['delta'],
            mechanism=self.ldp_args['mechanism'],
            device=self.ldp_args['device']
        )

        self.load_labels_set()
        self.train_dataset = NoisedCIFAR(self.train_dataset,
                                         len(self.labels_set),
                                         self.randomized_label_privacy,
                                         )


    def load_mnist(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_dataset = datasets.MNIST(self.datapath,
                                           train=True,
                                           download=False,
                                           transform=transform)
        self.valid_dataset = datasets.MNIST(self.datapath,
                                           train=False,
                                           download=False,
                                           transform=transform)


    def load_fashion_mnist(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_dataset = datasets.FashionMNIST(self.datapath,
                                           train=True,
                                           download=False,
                                           transform=transform)
        self.valid_dataset = datasets.FashionMNIST(self.datapath,
                                           train=False,
                                           download=False,
                                           transform=transform)

    def load_cifar10(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_dataset = datasets.CIFAR10(self.datapath,
                                           train=True,
                                           download=False,
                                           transform=transform)
        self.valid_dataset = datasets.CIFAR10(self.datapath,
                                           train=False,
                                           download=False,
                                           transform=transform)

    def load_cifar100(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_dataset = datasets.CIFAR100(self.datapath,
                                           train=True,
                                           download=False,
                                           transform=transform)
        self.valid_dataset = datasets.CIFAR100(self.datapath,
                                           train=False,
                                           download=False,
                                           transform=transform)

    def load_dogs_vs_cats(self):
        size = 256
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor()
        ])

        ds = datasets.ImageFolder(self.datapath,
                                  transform)
        self.train_dataset, self.valid_dataset = self._split_dataset(ds)

    def load_hist_cancer_detection(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        ds = datasets.ImageFolder(self.datapath,
                                  transform)
        self.train_dataset, self.valid_dataset = self._split_dataset(ds)


    def _split_dataset(self, ds):
        train_size = int(0.8 * len(ds))
        valid_size = len(ds) - train_size
        train_dataset, valid_dataset = random_split(ds,
                                                    [train_size, valid_size],
                                                    generator=torch.Generator().manual_seed(0))
        return train_dataset, valid_dataset

    def load_labels_set(self):
        if self.labels_set is None:
            if isinstance(self.train_dataset, torch.utils.data.dataset.Subset):
                labels_dict = self.train_dataset.dataset.class_to_idx
                self.train_dataset.targets = self.train_dataset.dataset.targets
            else:
                labels_dict = self.train_dataset.class_to_idx
            labels_set = [label for name, label in labels_dict.items()]
            self.labels_set = torch.LongTensor(labels_set).reshape(-1, 1)
        return self.labels_set

    def load_dataset(self, batch_size, shuffle=True):
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle, drop_last=True)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       drop_last=True)
        return self.train_loader, self.valid_loader



if __name__ == '__main__':
    batch_size = 128
    names = ['mnist', 'fashion-mnist', 'dogs_vs_cats',
             'cifar10', 'cifar100', 'hist_cancer_detection']
    datapathes = ['./dataset', './dataset', '~/DATA/dataset/competitions/dogs-vs-cats/train',
                  './dataset', './dataset', 
                  '~/DATA/dataset/competitions/histopathologic-cancer-detection/small_train']
    for name, datapath in zip(names, datapathes):
        ds = ExpDataset(datapath, name)
        train_loader, valid_loader = ds.load_dataset(batch_size)
        print(name)
        print(len(train_loader.dataset))
        print(len(valid_loader.dataset))
        print(len(ds.load_labels_set()))

