import torch
import numpy as np
import pdb

def random_guess_attack(train_loader, labels_set, device):
    guess_labels = []
    all_labels = []
    for imgs, labels in train_loader:
        labels = labels.to(device)
        guess_labels.append(np.random.choice(labels_set.cpu().numpy().flatten(), len(labels)))
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(guess_labels), np.concatenate(all_labels)
