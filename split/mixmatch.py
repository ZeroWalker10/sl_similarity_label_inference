#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn.functional as F
import numpy as np
import pdb

def sharpen(p, temp):
    sharpened_p = p ** (1.0 / temp)
    return sharpened_p / torch.sum(sharpened_p, dim=1).reshape(-1, 1)

class MixMatch:
	def __init__(self, unlabeled_dataloader, labeled_dataloader, labels_set,
	      model, mixup_alpha, lamb_u=50, temp=0.8):
		self.unlabeled_dataloader = unlabeled_dataloader
		self.labeled_dataloader = labeled_dataloader
		self.labels_set = labels_set
		self.model = model
		self.mixup_alpha = mixup_alpha
		self.lamb_u = lamb_u
		self.temp = temp
		self.critical_point = None
	
	def __iter__(self):
		return self

	def __next__(self):
		for unlabeled_features, dummy_labels in self.unlabeled_dataloader:
			for labeled_features, labeled_labels in self.labeled_dataloader:
				break
			true_labels = F.one_hot(labeled_labels, num_classes=len(labels_set))
			self.critical_point = len(true_labels)

			pred_labels = self.model(unlabeled_features)
			pred_labels = F.softmax(pred_labels, dim=1)
			pred_labels = sharpen(pred_labels, self.temp)

			features = torch.cat([labeled_features, unlabeled_features], dim=0)
			labels = torch.cat([true_labels, pred_labels], dim=0)

			if self.mixup_alpha is not None:
				lamb = np.random.beta(self.mixup_alpha, self.mixup_alpha)
				lamb = np.maximum(lamb, 1 - lamb)

				indexes = torch.randperm(len(features))
				features = lamb * features + (1 - lamb) * feagtures[indexes, :]
				labels = lamb * labels + (1 - lamb) * labels[indexes, :]

			yield features, labels
		raise StopIteration

	def loss_fn(self, pred_outputs, true_outputs):
		pred_softmax = F.softmax(pred_outputs, dim=1)
		labeled = true_outputs[:self.critical_point]
		unlabeled = true_output[self.critical_point:]
		labeled_softmax = pred_softmax[:self.critical_point]
		unlabeled_softmax = pred_softmax[self.critical_point:]

		labeled_log = torch.log(labeled_softmax + 1e-8)
		labeled_loss = -torch.sum(labeled_log * labeled) / len(labeled)
		unlabeled_loss = torch.sum((unlabeled - unlabeled_softmax) ** 2) / \
			(len(unlabeled) + len(self.labels_set))

		loss = labeled_loss + unlabeled_loss
		return loss
