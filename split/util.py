import torch
from torch import nn
import pdb
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csr_matrix
import datetime

def training_loop(n_epochs, pipeline, train_loader, device):
    # training mode
    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.train()

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            pipeline.reset()
            inputs = imgs
            # forward
            while not pipeline.is_end():
                entry = pipeline.next()
                outputs = entry.role.forward(inputs)
                inputs = outputs

            predictions = outputs.to(device)

            # backward
            pipeline.r_reset()
            inputs = labels
            while not pipeline.r_is_end():
                entry = pipeline.r_next()
                outputs = entry.role.backward(inputs)
                inputs = outputs
                
            if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
                with torch.no_grad():
                    _, predictions = torch.max(predictions, dim=1)
                    total += labels.shape[0]
                    correct += int((predictions == labels).sum())

            # release GPU memory
            del imgs, labels, predictions
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print('{} Epoch {}, Accuracy {:.2f}'.format(
                datetime.datetime.now(), epoch, correct / total
            ))

        # learning rate schedule
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.scheduler_step()

def training_loop_pick_enjoy(n_epochs, pipeline, train_loader_indexed, 
                             pick_epochs, device, pick_choice='grad', normalize=True):
    # pick choice: grad, smashed
    # training mode
    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.train()

    n = len(train_loader_indexed.dataset)
    pick_enjoys = [[] for _ in range(n)]
    pick_labels = [-1 for _ in range(n)]
    channel_enjoys = None
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        correct = 0
        total = 0
        for imgs, [labels, indexes] in train_loader_indexed:
            imgs = imgs.to(device)
            labels = labels.to(device)

            pipeline.reset()
            inputs = imgs
            # forward
            while not pipeline.is_end():
                entry = pipeline.next()
                outputs = entry.role.forward(inputs)
                inputs = outputs

                if pick_choice == 'smashed' and entry.role.step != -1 and epoch in pick_epochs:
                    smashed = outputs.view(len(outputs), -1).cpu().detach().numpy().tolist()
                    if normalize:
                        smashed = preprocessing.normalize(smashed)
                    for i, index in enumerate(indexes):
                        pick_enjoys[index].append(smashed[i])
                        if pick_labels[index] == -1:
                            pick_labels[index] = labels[i].cpu().detach().item()

            predictions = outputs.to(device)

            # backward
            pipeline.r_reset()
            inputs = labels
            while not pipeline.r_is_end():
                entry = pipeline.r_next()
                outputs = entry.role.backward(inputs)
                inputs = outputs

                if pick_choice == 'grad' and entry.role.step == -1 and epoch in pick_epochs:
                    # last step
                    # grads = outputs.view(len(outputs), -1).cpu().detach().numpy().tolist()
                    grads = outputs.cpu().detach().numpy()
                    shape = grads.shape
                    if len(shape[1:]) > 1:
                        # sum along the channel
                        # grads = np.sum(grads, axis=1) # not a good idea
                        grads = grads[:, 0, :, :]
                    grads = grads.reshape(len(outputs), -1).tolist()
                    if normalize:
                        grads = preprocessing.normalize(grads)
                    for i, index in enumerate(indexes):
                        pick_enjoys[index].append(grads[i])
                        if pick_labels[index] == -1:
                            pick_labels[index] = labels[i].cpu().detach().item()

            if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
                with torch.no_grad():
                    _, predictions = torch.max(predictions, dim=1)
                    total += labels.shape[0]
                    correct += int((predictions == labels).sum())
            
            # release GPU memory
            del imgs, labels, predictions
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print('{} Epoch {}, Accuracy {:.2f}'.format(
                datetime.datetime.now(), epoch, correct / total
            ))

        # learning rate schedule
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.scheduler_step()
    return pick_enjoys, pick_labels

def validate(pipeline, val_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        # eval mode
        pipeline.reset()
        while not pipeline.is_end():
            entry = pipeline.next()
            entry.role.eval()
            
        for imgs, labels in val_loader:
            pipeline.reset()
            inputs = imgs.to(device)
            labels = labels.to(device)
            while not pipeline.is_end():
                entry = pipeline.next()
                outputs = entry.role.forward(inputs)
                inputs = outputs
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
    return correct / total

def grads_training(pipeline, train_loader, labels_set, device):
    all_grads = []
    all_labels = []
    known_labels = {}
    offset = 0

    # training mode
    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.train()

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        pipeline.reset()
        inputs = imgs
        # forward
        while not pipeline.is_end():
            entry = pipeline.next()
            outputs = entry.role.forward(inputs)
            inputs = outputs

        predictions = outputs.to(device)

        # backward
        pipeline.r_reset()
        inputs = labels
        while not pipeline.r_is_end():
            entry = pipeline.r_next()
            outputs = entry.role.backward(inputs)
            if entry.role.step == -1:
                # last step
                for label in labels_set:
                  if label.item() in known_labels:
                    continue
                  label_indexes = torch.nonzero(labels == label,as_tuple=True)[0]
                  if len(label_indexes) == 0:
                      continue
                  # choose one grad randomly
                  index = label_indexes[torch.randperm(len(label_indexes))][0]
                  known_labels[label.item()] = offset + index.item()
                grads = outputs.reshape(len(outputs), -1)
                all_grads.append(grads)
                all_labels.append(labels)
            inputs = outputs
        offset += len(labels)
    return torch.cat(all_grads), torch.cat(all_labels), all_grads, known_labels

def smashed_training(pipeline, train_loader, labels_set, device):
    all_smashed = []
    all_labels = []
    known_labels = {}
    offset = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        pipeline.reset()
        inputs = imgs
        # forward
        while not pipeline.is_end():
            entry = pipeline.next()
            outputs = entry.role.forward(inputs)
            inputs = outputs
            if entry.role.step != -1:
                # not last step
                for label in labels_set:
                  if label.item() in known_labels:
                    continue
                  label_indexes = torch.nonzero(labels == label,as_tuple=True)[0]
                  if len(label_indexes) == 0:
                      continue
                  # choose one grad randomly
                  index = label_indexes[torch.randperm(len(label_indexes))][0]
                  known_labels[label.item()] = offset + index.item()
                smashed = outputs.detach().cpu().reshape(len(outputs), -1)
                all_smashed.append(smashed)
                all_labels.append(labels)

        predictions = outputs.to(device)

        # backward
        pipeline.r_reset()
        inputs = labels
        while not pipeline.r_is_end():
            entry = pipeline.r_next()
            outputs = entry.role.backward(inputs)
            inputs = outputs
        offset += len(labels)
    return torch.cat(all_smashed), torch.cat(all_labels), all_smashed, known_labels

def smashed_collecting(pipeline, train_loader, labels_set, device):
    all_smashed = []
    all_labels = []

    pipeline.reset()
    while not pipeline.is_end():
        entry = pipeline.next()
        entry.role.eval()

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        pipeline.reset()
        inputs = imgs
        # forward
        while not pipeline.is_end():
            entry = pipeline.next()
            outputs = entry.role.forward(inputs)
            inputs = outputs
            if entry.role.step != -1:
                smashed = outputs.detach().cpu().reshape(len(outputs), -1)
                all_smashed.append(smashed)
                all_labels.append(labels.cpu())

        predictions = outputs.to(device)

    return torch.cat(all_smashed), torch.cat(all_labels)

def l1_regularization(model):
    reg_loss = 0.0
    for param in model.parameters():
        if not isinstance(param, nn.BatchNorm2d):
            reg_loss = torch.sum(torch.abs(param)) + reg_loss

    return reg_loss

def l2_regularization(model):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss = torch.sum(param ** 2) / 2.0 + reg_loss

    return reg_loss

