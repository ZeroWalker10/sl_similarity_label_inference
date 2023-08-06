#!/usr/bin/env python
# coding=utf-8
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class CoAE(nn.Module):
    def __init__(self, n_class, times=6):
        super().__init__()
        
        self.n_class = n_class
        
        n_neuron = times * self.n_class + 2
        self.encoder = nn.Sequential(
            nn.Linear(self.n_class, n_neuron * n_neuron),
            nn.ReLU(),
            nn.Linear(n_neuron * n_neuron, self.n_class),
            nn.Softmax(dim=1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.n_class, n_neuron * n_neuron),
            nn.ReLU(),
            nn.Linear(n_neuron * n_neuron, self.n_class),
            nn.Softmax(dim=1)
        )
    
    def forward(self, X):
        X_tile = self.encoder(X)
        X_hat = self.decoder(X_tile)
        
        return X_tile, X_hat

def cross_entropy(pred, target):
    return torch.mean(torch.sum(-target * torch.log(pred + 1e-30), 1))

def entropy(pred):
    return torch.mean(torch.sum(-pred * torch.log2(pred + 1e-30), 1))

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=1), 1))

def train_coae(coae, n_class, lambda_2=0.5):
    batch_size = 256
    epochs = 30
    batch_num = 500
    lr = 1e-4
    
    optimizer = torch.optim.Adam(coae.parameters(), lr=lr)
    lambda_1 = 0.1
    
    coae.train()
    for e in range(epochs):
        total_loss = 0.0
        total_contra_loss = 0.0
        total_entropy_loss = 0.0
        
        for _ in range(batch_num):
            optimizer.zero_grad()
            
            y = F.one_hot(torch.randint(0, n_class, (batch_size,)), n_class).float()
            y_tile, y_hat = coae(y)
            contra_loss = cross_entropy(y_hat, y) - lambda_1 * cross_entropy(y_tile, y)
            entropy_loss = entropy(y_tile)
            loss = contra_loss - lambda_2 * entropy_loss
            loss.backward()
            
            optimizer.step()
        
            total_loss = total_loss + loss.detach().cpu().numpy()
            total_contra_loss = total_contra_loss + contra_loss.detach().cpu().numpy()
            total_entropy_loss = total_entropy_loss + entropy_loss.detach().cpu().numpy()
        
        if (e + 1) % 10 == 0:
            print('epoch {}'.format(e + 1), 'loss:', total_loss / batch_num, 'contrast loss:', total_contra_loss / batch_num,
             'entropy loss:', total_entropy_loss / batch_num)

def sae_softlabel(y_onehot, n, steps=9):
    n_class = y_onehot.shape[1]
    candidate_steps = np.arange(0.0, 1.0, 0.1)[:steps]
    lambs = np.random.choice(candidate_steps, size=(y_onehot.shape[0],), replace=True)
    y_soft = np.copy(y_onehot)
    for i in range(y_onehot.shape[0]):
        candidates = list(range(n_class))
        del candidates[np.argmax(y_soft[i])]
        choices = np.random.choice(candidates, size=(n,), replace=False)
        y_soft[i, choices] = lambs[i]
        y_soft[i] = y_soft[i] / np.sum(y_soft[i])
    return y_soft

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x + 1e-30
        return x / torch.sum(x, dim=1, keepdims=True)
    
class SAE(nn.Module):
    def __init__(self, n_class, times):
        super().__init__()
        
        self.n_class = n_class
        
        if self.n_class <= 10:
            n_neuron = times * self.n_class + 2
        else:
            n_neuron = self.n_class + 2
        self.encoder = nn.Sequential(
            nn.Linear(self.n_class, n_neuron * n_neuron),
            nn.ReLU(),
            nn.Linear(n_neuron * n_neuron, self.n_class),
            nn.Sigmoid(),
            Normalize(),
            # nn.Softmax(dim=1),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.n_class, n_neuron * n_neuron),
            nn.ReLU(),
            nn.Linear(n_neuron * n_neuron, self.n_class),
            nn.Sigmoid(),
            Normalize(),
            # nn.Softmax(dim=1),
        )
    
    def forward(self, X):
        X_tile = self.encoder(X)
        X_hat = self.decoder(X_tile)
        
        return X_tile, X_hat

def train_sae(sae, n_class, pad_n, steps, device):
    batch_size = 256
    epochs = 100
    batch_num = 100
    lr = 5e-4
    
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    lambda_1, lambda_2 = 0.05, 0.05
    
    sae.train()
    for e in range(epochs):
        total_loss = 0.0
        total_contra_loss = 0.0
    
        for _ in range(batch_num):
            optimizer.zero_grad()
    
            y = F.one_hot(torch.randint(0, n_class, (batch_size,)), n_class).float().numpy()
            y_soft = torch.from_numpy(sae_softlabel(y, pad_n, steps=steps)).float().to(device)
            y_tile, y_hat = sae(y_soft)
            contra_loss = cross_entropy(y_hat, y_soft) - lambda_1 * cross_entropy(y_tile, y_soft)
            # entropy_loss = entropy(y_tile)
            # loss = contra_loss + lambda_2 * entropy_loss
            loss = contra_loss
            loss.backward()
    
            optimizer.step()
    
            total_loss = total_loss + loss.detach().cpu().numpy()
        if (e + 1) % 20 == 0:
            print('epoch {}'.format(e + 1), 'loss:', total_loss / batch_num)
