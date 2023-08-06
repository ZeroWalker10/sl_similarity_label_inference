#!/usr/bin/env python
# coding=utf-8
from custom_dataset import CustomDataset
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
import collections
import pdb

class ExpDataset(CustomDataset):
    def __init__(self, datapath, name, composers=None, num_workers=8):
        # composers: [0] train composer, [1] val composer
        super(ExpDataset, self).__init__(datapath)
        self.name = name
        self.num_workers = num_workers

        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.labels_set = None
        self.distribution = None
        if self.name == 'mnist':
            self.load_mnist(composers)
        elif self.name == 'emnist-letters':
            self.load_emnist('letters', composers)
        elif self.name == 'omniglot':
            self.load_omniglot(composers)
        elif self.name == 'fashion-mnist':
            self.load_fashion_mnist(composers)
        elif self.name == 'dogs_vs_cats':
            self.load_dogs_vs_cats(composers)
        elif self.name == 'clothing':
            self.load_clothing(composers)
        elif self.name == 'cifar10':
            self.load_cifar10(composers)
        elif self.name == 'cinic10':
            self.load_cinic10(composers)
        elif self.name == 'cifar100':
            self.load_cifar100(composers)
        elif self.name == 'hist_cancer_detection':
            self.load_hist_cancer_detection(composers)
        elif self.name == 'intel_classification':
            self.load_intel_classification(composers)
        elif self.name == '102flowers':
            self.load_102flowers(composers)
        elif self.name == '101foods':
            self.load_101foods(composers)
        elif self.name == 'caltech256':
            self.load_caltech256(composers)
        elif self.name == 'caltech101':
            self.load_caltech101(composers)
        elif self.name == 'cars':
            self.load_cars(composers)
        elif self.name == 'fruits':
            self.load_fruits(composers)
        elif self.name == 'image-net':
            self.load_image_net(composers)
        elif self.name == 'tiny-image-net':
            self.load_tiny_image_net(composers)
        elif self.name == 'stanford-dogs':
            self.load_stanford_dogs(composers)
        elif self.name == 'butterfly-moths':
            self.load_butterfly_moths(composers)
        elif self.name == '100sports':
            self.load_100sports(composers)
        else:
            raise 'invalid dataset name!!!'

    def load_mnist(self, composers=None):
        if composers is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        self.train_dataset = datasets.MNIST(self.datapath,
                                           train=True,
                                           download=False,
                                           transform=train_transform)
        self.valid_dataset = datasets.MNIST(self.datapath,
                                           train=False,
                                           download=False,
                                           transform=val_transform)

    def load_emnist(self, split, composers):
        if composers is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        self.train_dataset = datasets.EMNIST(self.datapath,
                                           train=True,
                                           download=False,
                                           split=split,
                                           transform=train_transform)
        self.valid_dataset = datasets.EMNIST(self.datapath,
                                           train=False,
                                           split=split,
                                           download=False,
                                           transform=val_transform)

    def load_omniglot(self, composers=None):
        if composers is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        self.train_dataset = datasets.Omniglot(self.datapath,
                                           background=True,
                                           download=False,
                                           transform=train_transform)
        self.valid_dataset = datasets.Omniglot(self.datapath,
                                           download=False,
                                           background=False,
                                           transform=val_transform)
    def load_fashion_mnist(self, composers):
        if composers is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        self.train_dataset = datasets.FashionMNIST(self.datapath,
                                           train=True,
                                           download=False,
                                           transform=train_transform)
        self.valid_dataset = datasets.FashionMNIST(self.datapath,
                                           train=False,
                                           download=False,
                                           transform=val_transform)

    def load_cifar10(self, composers=None):
        if composers is None:
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            val_transform = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform, val_transform = composers

        self.train_dataset = datasets.CIFAR10(self.datapath,
                                           train=True,
                                           download=False,
                                           transform=train_transform)
        self.valid_dataset = datasets.CIFAR10(self.datapath,
                                           train=False,
                                           download=False,
                                           transform=val_transform)

    def load_cifar100(self, composers=None):
        if composers is None:
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            val_transform = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform, val_transform = composers

        self.train_dataset = datasets.CIFAR100(self.datapath,
                                           train=True,
                                           download=False,
                                           transform=train_transform)
        self.valid_dataset = datasets.CIFAR100(self.datapath,
                                           train=False,
                                           download=False,
                                           transform=val_transform)

    def load_102flowers(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        ds = datasets.ImageFolder(self.datapath,
                                  train_transform)
        self.train_dataset, self.valid_dataset = self._split_dataset(ds)
        
    def load_caltech256(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        ds = datasets.ImageFolder(self.datapath,
                                  train_transform)
        self.train_dataset, self.valid_dataset = self._split_dataset(ds)

    def load_cars(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        train_path = os.path.join(self.datapath, 'cars_train')
        self.train_dataset  = datasets.ImageFolder(train_path, train_transform)
        val_path = os.path.join(self.datapath, 'cars_test')
        self.valid_dataset = datasets.ImageFolder(val_path, val_transform)

    def load_butterfly_moths(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        train_path = os.path.join(self.datapath, 'train')
        self.train_dataset  = datasets.ImageFolder(train_path, train_transform)
        val_path = os.path.join(self.datapath, 'VALID')
        self.valid_dataset = datasets.ImageFolder(val_path, val_transform)

    def load_100sports(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        train_path = os.path.join(self.datapath, 'train')
        self.train_dataset  = datasets.ImageFolder(train_path, train_transform)
        val_path = os.path.join(self.datapath, 'VALID')
        self.valid_dataset = datasets.ImageFolder(val_path, val_transform)

    def load_cinic10(self, composers=None):
        if composers is None:
            train_transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            test_transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            train_transform, val_transform = composers

        train_path = os.path.join(self.datapath, 'train')
        self.train_dataset  = datasets.ImageFolder(train_path, train_transform)
        val_path = os.path.join(self.datapath, 'valid')
        self.valid_dataset = datasets.ImageFolder(val_path, val_transform)

    def load_fruits(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        train_path = os.path.join(self.datapath, 'Training')
        self.train_dataset  = datasets.ImageFolder(train_path, train_transform)
        val_path = os.path.join(self.datapath, 'Test')
        self.valid_dataset = datasets.ImageFolder(val_path, val_transform)

    def load_caltech101(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        train_path = os.path.join(self.datapath, 'train')
        self.train_dataset  = datasets.ImageFolder(train_path, train_transform)
        val_path = os.path.join(self.datapath, 'test')
        self.valid_dataset = datasets.ImageFolder(val_path, val_transform)

    def load_101foods(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        ds = datasets.ImageFolder(self.datapath,
                                  train_transform)
        self.train_dataset, self.valid_dataset = self._split_dataset(ds)

    def load_dogs_vs_cats(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        ds = datasets.ImageFolder(self.datapath,
                                  train_transform)
        self.train_dataset, self.valid_dataset = self._split_dataset(ds)

    def load_stanford_dogs(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        ds = datasets.ImageFolder(self.datapath,
                                  train_transform)
        self.train_dataset, self.valid_dataset = self._split_dataset(ds)

    def load_clothing(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        ds = datasets.ImageFolder(self.datapath,
                                  train_transform)
        self.train_dataset, self.valid_dataset = self._split_dataset(ds)

    def load_hist_cancer_detection(self, composers=None):
        if composers is None:
            size = 224
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        ds = datasets.ImageFolder(self.datapath,
                                  train_transform)
        self.train_dataset, self.valid_dataset = self._split_dataset(ds)

    def load_intel_classification(self, composers=None):
        if composers is None:
            size = 256
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        train_path = os.path.join(self.datapath, 'seg_train/seg_train')
        self.train_dataset  = datasets.ImageFolder(train_path, train_transform)
        val_path = os.path.join(self.datapath, 'seg_test/seg_test')
        self.valid_dataset = datasets.ImageFolder(val_path, val_transform)

    def load_tiny_image_net(self, composers=None):
        if composers is None:
            size = 256
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers

        train_path = os.path.join(self.datapath, 'train')
        self.train_dataset  = datasets.ImageFolder(train_path, train_transform)
        val_path = os.path.join(self.datapath, 'val/images')
        self.valid_dataset = datasets.ImageFolder(val_path, val_transform)

    def load_image_net(self, composers=None):
        if composers is None:
            size = 256
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor()
            ])
            train_transform = transform
            val_transform = transform
        else:
            train_transform, val_transform = composers


        ds = datasets.ImageFolder(self.datapath,
                                  train_transform)
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
            else:
                labels_dict = self.train_dataset.class_to_idx
            labels_set = [label for name, label in labels_dict.items()]
            self.labels_set = torch.LongTensor(labels_set).reshape(-1, 1)
        return self.labels_set

    def load_distribution(self):
        if self.distribution is None:
            if isinstance(self.train_dataset, torch.utils.data.dataset.Subset):
                cnt = collections.Counter(self.train_dataset.dataset.targets)
            else:
                if not isinstance(self.train_dataset.targets, list):
                    cnt = collections.Counter(self.train_dataset.targets.tolist())
                else:
                    cnt = collections.Counter(self.train_dataset.targets)

            dist = [0] * len(cnt)
            for label, num in cnt.items():
                dist[label] = num
            self.distribution = np.array(dist) / np.sum(dist)
        return self.distribution

    def load_dataset_indexed(self, batch_size, shuffle=True): 
        self.indexed_features = []
        for i, (feature, label) in enumerate(self.train_dataset):
            self.indexed_features.append((feature, [label, i]))

        self.indexed_train_loader = DataLoader(self.indexed_features,
                                       batch_size=batch_size,
                                       shuffle=shuffle)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle)
        return self.indexed_train_loader, self.valid_loader

    def load_dataset_partial_labeled(self, labeled_batch_size, 
                                     unlabeled_batch_size, known_per_class, shuffle=True,
                                     drop_last=False):
        labeled_features, unlabeled_features = [], []
        labeled_count = {}
        for feature, label in self.train_dataset:
            if label not in labeled_count:
                labeled_count[label] = 1
                labeled_features.append((feature, label))
            elif labeled_count[label] < known_per_class:
                labeled_count[label] += 1
                labeled_features.append((feature, label))
            else:
                unlabeled_features.append((feature, label))

        self.labeled_train_dataset = labeled_features
        self.unlabeled_train_dataset = unlabeled_features
        self.labeled_train_loader = DataLoader(self.labeled_train_dataset,
                                       batch_size=labeled_batch_size,
                                       shuffle=shuffle,
                                               drop_last=drop_last)
        self.unlabeled_train_loader = DataLoader(self.unlabeled_train_dataset,
                                       batch_size=unlabeled_batch_size,
                                       shuffle=shuffle, drop_last=drop_last)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=unlabeled_batch_size + labeled_batch_size,
                                       shuffle=shuffle,
                                       drop_last=drop_last)
        return self.labeled_train_loader, self.unlabeled_train_loader, self.valid_loader

    def load_dataset(self, batch_size, shuffle=True, drop_last=False):
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       drop_last=drop_last,
                                       num_workers=self.num_workers)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=batch_size,
                                       drop_last=drop_last,
                                       num_workers=self.num_workers)
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
        print(ds.load_distribution())

