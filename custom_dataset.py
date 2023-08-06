#!/usr/bin/env python
# coding=utf-8
class CustomDataset:
    def __init__(self, datapath):
        self.datapath = datapath

    def load_dataset(self, batch_size, shuffle=True):
        raise 'Derived class must implement this method!!!'
