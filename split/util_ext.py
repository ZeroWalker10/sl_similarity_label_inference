#!/usr/bin/env python
# coding=utf-8
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize
from torch.optim.lr_scheduler import LambdaLR
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
import copy
import pickle
import pdb

from scipy.optimize import linear_sum_assignment as linear_assignment

def cluster_match(true_labels, pred_labels):
    dims = max(np.max(true_labels), np.max(pred_labels)) + 1
    w = np.zeros((dims, dims), dtype=np.int64)
    for i in range(len(true_labels)):
        w[true_labels[i], pred_labels[i]] += 1
    indexes = linear_assignment(w.max() - w)
    return w, indexes

def cluster_accuracy(true_labels, pred_labels):
    w, indexes = cluster_match(true_labels, pred_labels)
    return sum([w[i, j] for i, j in zip(indexes[0], indexes[1])]) * 1.0 / len(true_labels)

def generate_known_dict(true_labels, known=1):
    known_label_dict = {}
    for index, label in enumerate(true_labels):
        if isinstance(label, torch.Tensor):
            label = label.item()
        elif isinstance(label, np.ndarray):
            label = int(label)
        elif (not isinstance(label, int)) and (not isinstance(label, np.int64)) \
            and (not isinstance(label, np.int32)):
            raise Exception('Invalid label type')

        if label not in known_label_dict:
            known_label_dict[label] = [index]
        elif len(known_label_dict[label]) < known:
            known_label_dict[label].append(index)

    return known_label_dict

def cluster_accuracy_based_on_known(true_labels, pred_labels, known=1,
                                    known_label_dict=None):
    # pick known label dict
    true_candidates, pred_candidates = [], []
    if known_label_dict is None:
        known_label_dict = {}
        for index, label in enumerate(true_labels):
            if isinstance(label, torch.Tensor):
                label = label.item()
            elif isinstance(label, np.ndarray):
                label = int(label)
            elif (not isinstance(label, int)) and (not isinstance(label, np.int64)) \
                and (not isinstance(label, np.int32)):
                raise Exception('Invalid label type')

            if label not in known_label_dict:
                known_label_dict[label] = [index]
                true_candidates.append(label)
                pred_candidates.append(pred_labels[index])
            elif len(known_label_dict[label]) < known:
                known_label_dict[label].append(index)
                true_candidates.append(label)
                pred_candidates.append(pred_labels[index])
    else:
        for label, indexes in known_label_dict.items():
            for index in indexes:
                true_candidates.append(label)
                pred_candidates.append(pred_labels[index])

    return_labels = np.random.choice(list(known_label_dict.keys()), pred_labels.shape)
    if len(np.unique(pred_candidates)) > 1:
        w, indexes = cluster_match(true_candidates, pred_candidates)
        for tlabel, plabel in zip(indexes[0], indexes[1]):
            return_labels[pred_labels==plabel] = tlabel
    '''
    return_labels = np.random.choice(list(known_label_dict.keys()), 
                                     size=len(pred_labels))
    if known == 1:
        for tlabel, plabel in zip(true_candidates, pred_candidates):
            return_labels[pred_labels==plabel] = tlabel
    elif known > 1:
        raise 'unimplement the multiple knonw'
    else:
        raise 'invalid known argument'
    '''

    return known_label_dict, return_labels

def split_based_on_known(datum, labels, known_labels_per_class):
    labeled_data, unlabeled_data= [], []
    labeled_labels, unlabeled_labels = [], []
    labeled_count = {}
    for data, label in zip(datum, labels):
        if label not in labeled_count:
            labeled_count[label] = 1
            labeled_data.append(data)
            labeled_labels.append(label)
        elif labeled_count[label] < known_labels_per_class:
            labeled_count[label] += 1
            labeled_data.append(data)
            labeled_labels.append(label)
        else:
            unlabeled_data.append(data)
            unlabeled_labels.append(label)
    return labeled_data, labeled_labels, unlabeled_data, unlabeled_labels

class SplitDP(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0
        
        self.train_step = self._make_train_step()
        self.val_step  = self._make_val_step()

        self.visualization = {}
        self.activation_grads = {}
        self.forward_handles = {}
        self.backward_handles = {}
        self.train_labels = np.array([])
        self.val_labels = np.array([])
        self.collect_labels = None

        self._gradients = {}
        self._parameters = {}

        self.scheduler = None
        self.is_batch_lr_scheduler = False

        self.learning_rates = []
        self.clipping = None
    
    def to(self, device):
        self.device = device
        self.model.to(device)
    
    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def set_tensorboard(self, name, folder='runs'):
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter('{}/{}_{}'.format(
            folder, name, suffix
        ))
    
    @staticmethod
    def static_set_seed(seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass
    
    def train(self, n_epochs, seed=10, verbose=True):
        self.set_seed(seed)
        for epoch in range(1, n_epochs + 1):
            self.total_epochs += 1
            
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)
            
            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            
            self._epoch_schedulers(val_loss)

            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                self.writer.add_scalars(main_tag='loss',
                                       tag_scalar_dict=scalars,
                                       global_step=epoch)
            
            if (epoch == 1 or epoch % 10 == 0 or epoch == n_epochs) and verbose:
                if self.train_loader:
                    train_acc = SplitDP.loader_apply(self.train_loader, self.correct).sum(dim=0)
                    train_acc = train_acc[0] / train_acc[1]
                else:
                    train_acc = 0.0
                
                if self.val_loader:
                    val_acc = SplitDP.loader_apply(self.val_loader, self.correct).sum(dim=0)
                    val_acc = val_acc[0] / val_acc[1]
                else:
                    val_acc = 0.0
                    
                print('{} Epoch {}, Training loss {:.2f}, Acc {:.2f}; Validation loss {:.2f}, Acc {:.2f}'.format(
                    datetime.datetime.now(), epoch,
                    loss, train_acc, val_loss, val_acc
                ))
                
        if self.writer:
            self.writer.flush()
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses,
            'val_loss': self.val_losses
        }
        
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        
        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        
        self.model.train()
    
    def predict(self, x):
        self.model.eval()
        
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        
        self.model.train()
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        if self.val_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def lr_range_test(self, data_loader, end_lr, num_iter=100,
                      step_mode='exp', alpha=0.05, ax=None):
        previous_states = {
            'model': deepcopy(self.model.state_dict()),
            'optimizer': deepcopy(self.optimizer.state_dict())
        }

        start_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        lr_fn = SplitDP.make_lr_fn(start_lr, end_lr, num_iter)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)

        tracking = {'loss': [], 'lr': []}
        iteration = 0
        while iteration < num_iter:
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                yhat = self.model(x_batch)
                loss = self.loss_fn(yhat, y_batch)
                loss.backward()

                tracking['lr'].append(scheduler.get_last_lr()[0])
                if iteration == 0:
                    tracking['loss'].append(loss.item())
                else:
                    prev_loss = tracking['loss'][-1]
                    smoothed_loss = (alpha * loss.item() + (1 - alpha) * prev_loss)
                    tracking['loss'].append(smoothed_loss)

                iteration += 1
                if iteration == num_iter:
                    break

                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

        self.optimizer.load_state_dict(previous_states['optimizer'])
        self.model.load_state_dict(previous_states['model'])

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.get_figure()

        ax.plot(tracking['lr'], tracking['loss'])
        if step_mode == 'exp':
            ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        return tracking, fig

    def add_graph(self):
        if self.train_loader and self.writer:
            x_dummy, y_dummy = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_dummy.to(self.device))

    def count_parameters(self):
        return sum(p.numel()
                   for p in self.model.parameters()
                   if p.requires_grad)
    
    def visualize_filters(self, layer_name, **kwargs):
        try:
            layer = self.model
            for name in layer_name.split('.'):
                layer =getattr(layer, name)

            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.data.cpu().numpy()
                n_filters, n_channels, _, _ = weights.shape
                size = (2 * n_channels + 2, 2 * n_filters)
                fig, axes = plt.subplots(n_filters, n_channels,
                                     figsize=size)
                axes = np.atleast_2d(axes)
                axes = axes.reshape(n_filters, n_channels)
                for i in range(n_filters):
                    SplitDP._visualize_tensors(
                        axes[i, :],
                        weights[i],
                        layer_name=f'Filter #{i}',
                        title='Channel'
                    )

                for ax in axes.flat:
                    ax.label_outer()

                fig.tight_layout()
                return fig
        except AttributeError: 
            pass

    def attach_forward_hooks(self, layers_to_hook, hook_fn=None):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]
            
        self.visualization = {}
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]}

        if hook_fn is None:
            def hook_fn(layer, inputs, outputs):
                name = layer_names[layer]
                values = inputs[0].detach().cpu().numpy().astype('float32')

                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([self.visualization[name], values])
        
        for name, layer in modules:
            if name in layers_to_hook:
                self.visualization[name] = None
                self.forward_handles[name] = layer.register_forward_hook(hook_fn)
    
    def attach_backward_hooks(self, layers_to_hook, hook_fn=None):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]
            
        self.activation_grads = {}
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]}

        if hook_fn is None:
            def hook_fn(layer, inputs, outputs):
                name = layer_names[layer]
                values = inputs[0].detach().cpu().numpy().astype('float32')
                # values = inputs[0].detach().cpu().numpy().astype('float32')

                if self.activation_grads[name] is None:
                    self.activation_grads[name] = values
                else:
                    self.activation_grads[name] = np.concatenate([self.activation_grads[name], values])
        
        for name, layer in modules:
            if name in layers_to_hook:
                self.activation_grads[name] = None
                self.backward_handles[name] = layer.register_full_backward_hook(hook_fn)

    def remove_hooks(self):
        for handle in self.forward_handles.values():
            handle.remove()
        self.forward_handles = {}
        
        for handle in self.backward_handles.values():
            handle.remove()
        self.backward_handles = {}

    def visualize_outputs(self, layers, n_images=10, y=None, yhat=None):
        layers = filter(lambda l: l in self.visualization.keys(), layers)
        layers = list(layers)
        shapes = [self.visualization[layer].shape for layer in layers]
        n_rows = [shape[1] if len(shape) == 4 else 1
                  for shape in shapes]
        total_rows = np.sum(n_rows)

        fig, axes = plt.subplots(total_rows, n_images,
                                 figsize=(1.5*n_images, 1.5*total_rows))
        axes = np.atleast_2d(axes).reshape(total_rows, n_images)

        row = 0
        for i, layer in enumerate(layers):
            start_row = row
            output = self.visualization[layer]

            is_vector = len(output.shape) == 2
            for j in range(n_rows[i]):
                SplitDP._visualize_tensors(
                    axes[row, :],
                    output if is_vector else output[:, j].squeeze(),
                    y,
                    yhat,
                    layer_name=layers[i] if is_vector else f'{layer[i]}\nfil#{row-start_row}',
                    title='Image' if row == 0 else None
                )
                row += 1

        for ax in axes.flat:
            ax.label_outer()

        plt.tight_layout()
        return fig

    def collect_labels_fun(self, x, y):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        ys = y.detach().cpu().numpy()
        if self.collect_labels is None:
            self.collect_labels = ys
        else:
            self.collect_labels = np.concatenate([self.collect_labels, ys])
        self.model.train()

    def correct(self, x, y, threshold=0.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()

        n_samples, n_dims = yhat.shape
        if n_dims > 1:
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            if isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Sigmoid):
                predicted = (yhat > threshold).long()
            else:
                predicted = (torch.sigmoid(yhat) > threshold).long()

        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result)

    def capture_parameters(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layer_to_hook = [layers_to_hook]

        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules}

        self._parameters = {}
        for name, layer in modules:
            if name in layers_to_hook:
                self._parameters.update({name: {}})
                for param_id, p in layer.named_parameters():
                    self._parameters[name].update({param_id: []})

        def fw_hook_fn(layer, inputs, outputs):
            name = layer_names[layer]
            for param_id, parameter in layer.named_parameters():
                self._parameters[name][param_id].append(
                    parameter.tolist()
                )

        self.attach_hooks(layers_to_hook, fw_hook_fn)

    def capture_gradients(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]

        modules = list(self.model.named_modules())
        self._gradients = {}

        def make_log_fn(name, param_id):
            def log_fn(grad):
                self._gradients[name][param_id].append(grad.tolist())
                return None
            return log_fn
        
        for name, layer in self.model.named_modules():
            if name in layers_to_hook:
                self._gradients.update({name: {}})
                for param_id, p in layer.named_parameters():
                    if p.requires_grad:
                        self._gradients[name].update({param_id: []})
                        log_fn = make_log_fn(name, param_id)
                        self.handles[f'{name}.{param_id}.grad'] = p.register_hook(log_fn)

    def set_lr_scheduler(self, scheduler):
        if scheduler.optimizer == self.optimizer:
            self.scheduler = scheduler
            if (isinstance(scheduler, optim.lr_scheduler.CyclicLR) or
                isinstance(scheduler, optim.lr_scheduler.OneCycleLR) or
                isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts)):
                self.is_batch_lr_scheduler = True
            else:
                self.is_batch_lr_scheduler = False

    def set_clip_grad_value(self, clip_value):
        self.clipping = lambda: nn.utils.clip_grad_value_(
            self.model.parameters(), clip_value=clip_value
        )

    def set_clip_grad_norm(self, max_norm, norm_type=2):
        self.clipping = lambda: nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm, norm_type
        )

    def remove_clip(self):
        self.clipping = None

    def set_clip_backprop(self, clip_value):
        if self.clipping is None:
            self.clipping = []

        for p in self.model.parameters():
            if p.requires_grad:
                func = lambda grad: torch.clamp(grad,
                                                -clip_value,
                                                clip_value)
                handle = p.register_hook(func)
                self.clipping.append(handle)

    def remove_clip(self):
        if isinstance(self.clipping, list):
            for handle in self.clipping:
                handle.remove()
        self.clipping = None

    def _make_train_step(self):
        def perform_train_step(x, y):

            self.model.train()
            
            self.optimizer.zero_grad()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()

            if callable(self.clipping):
                self.clipping()

            self.optimizer.step()
            
            return loss.item()
        return perform_train_step
    
    def _make_val_step(self):
        def perform_val_step(x, y):
            self.model.eval()
            
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            
            return loss.item()
        return perform_val_step

    def _mini_batch_schedulers(self, frac_epoch):
        if self.scheduler:
            if self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.scheduler.step(self.total_epochs + frac_epoch)
                else:
                    self.scheduler.step()

            current_lr = list(
                map(lambda d: d['lr'],
                    self.scheduler.optimizer.state_dict()['param_groups'])
            )
            self.learning_rates.append(current_lr)

    def _epoch_schedulers(self, val_loss):
        if self.scheduler:
            if not self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = list(
                    map(lambda d: d['lr'],
                        self.scheduler.optimizer.state_dict()['param_groups'])
                )

                self.learning_rates.append(current_lr)
    
    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step = self.val_step
            self.val_labels = None
        else:
            data_loader = self.train_loader
            step = self.train_step
            self.train_labels = None
        
        if data_loader is None:
            return None
        
        n_batches = len(data_loader)
        mini_batch_losses = []
        for i, (x_batch, y_batch) in enumerate(data_loader):
            batch_labels = y_batch.detach().cpu().numpy()
            if not validation:
                if self.train_labels is None:
                    self.train_labels = batch_labels
                else:
                     self.train_labels = np.concatenate([self.train_labels, batch_labels])
            else:
                if self.val_labels is None:
                    self.val_labels = batch_labels
                else:
                     self.val_labels = np.concatenate([self.val_labels, batch_labels])
            
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            mini_batch_loss = step(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

            if not validation:
                self._mini_batch_schedulers(i / n_batches)
            
        loss = np.mean(mini_batch_losses)
        return loss

    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None,
                           layer_name='', title=None):
        n_images = len(axs)
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]

            if title is not None:
                ax.set_title(f'{title} #{j}', fontsize=12)
            shp = np.atleast_2d(image).shape
            ax.set_ylabel(
                f'{layer_name}\n{shp[0]}x{shp[1]}',
                rotation=0, labelpad=40
            )
            xlabel1 = '' if y is None else f'\nLabel: {y[j]}'
            xlabel2 = '' if yhat is None else f'\nPredicted: {yhat[j]}'
            xlabel = f'{xlabel1}{xlabel2}'
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='gray',
                vmin=minv,
                vmax=maxv
            )

    @staticmethod
    def naiv_loader_apply(loader, func, reduce='sum'):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        return results

    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)

        return results

    @staticmethod
    def statistics_per_channel(images, labels):
        n_samples, n_channels, n_height, n_weight = images.size()
        flatten_per_channel = images.reshape(n_samples, n_channels, -1)

        means = flatten_per_channel.mean(axis=2)
        stds = flatten_per_channel.std(axis=2)

        sum_means = means.sum(axis=0)
        sum_stds = stds.sum(axis=0)

        n_samples = torch.tensor([n_samples] * n_channels).float()

        return torch.stack([n_samples, sum_means, sum_stds], axis=0)

    @staticmethod
    def make_normalizer(loader):
        total_samples, total_means, total_stds = SplitDP.loader_apply(
            loader, SplitDP.statistics_per_channel
        )
        norm_mean = total_means / total_samples
        norm_std = total_stds / total_samples

        return Normalize(mean=norm_mean, std=norm_std)

    @staticmethod
    def make_lr_fn(start_lr, end_lr, num_iter, step_mode='exp'):
        if step_mode == 'linear':
            factor = (end_lr / start_lr - 1) / num_iter
            def lr_fn(iteration):
                return 1 + iteration * factor
        else:
            factor = (np.log(end_lr) - np.log(start_lr)) / num_iter
            def lr_fn(iteration):
                return np.exp(factor) ** iteration
        return lr_fn

class Collecter:
    def __init__(self, max_dims=2048*4):
        self.collect_grads = None
        self.collect_activations = None
        self.max_dims = max_dims
        self.pca = IncrementalPCA(n_components=self.max_dims)
        self.batch_samples = None
        
    def get_grad_hook(self, normalize=True):
        def hook_fn(layer, inputs, outputs):
            # grads = outputs[0].cpu().detach().numpy()
            grads = inputs[0].cpu().detach().numpy().astype('float32')
            # pdb.set_trace()
            shape = grads.shape
            if np.prod(shape[1:]) <= max_dims:
                grads = grads.reshape(len(grads), -1)
            else:
                dims_per_channel = int(max_dims / shape[1])
                dims = min(dims_per_channel, len(grads))
                final_batch_grads = []
                for channel in range(shape[1]):
                    grads_per_channel = grads[:, channel, :, :].reshape(len(grads), -1)
                    pca = PCA(n_components=self.max_dims)
                    reduced_grads = pca.fit_transform(grads_per_channel)
                    final_batch_grads.append(reduced_grads)
                grads = np.hstack(final_batch_grads)
            if normalize:
                grads = preprocessing.normalize(grads).astype('float32')
            if self.collect_grads is None:
                self.collect_grads = grads
            else:
                self.collect_grads = np.concatenate([self.collect_grads, grads])
        return hook_fn

class CollecterExt:
    def __init__(self, batch_size, max_dims=2048*3, train_pca=False, normalize=True, cache_size=1024, upper_dims=None):
        self.collect_grads = None
        self.collect_smashed = None
        self.collect_labels = None
        self.max_dims = max_dims
        self.batch_size = batch_size
        self.upper_dims = upper_dims
        self.cache_size = cache_size
        self.cache = None
        
        self.dims = min(self.batch_size, self.max_dims)
        if self.upper_dims is None:
            self.dims = max(self.dims, self.cache_size)
        else:
            self.dims = max(self.dims, self.upper_dims)
        self.pca = IncrementalPCA(n_components=self.dims, batch_size=self.batch_size)
        self.batch_samples = None
        self.train_pca = train_pca
        self.normalize = normalize

    def get_grad_hook(self):
        def hook_fn(layer, inputs, outputs):
            # grads = outputs[0].cpu().detach().numpy()
            grads = inputs[0].cpu().detach().numpy().astype('float32')
            # pdb.set_trace()
            shape = grads.shape
            grads = grads.reshape(len(grads), -1)
            if np.prod(shape[1:]) > self.max_dims:
                if self.train_pca:
                    if self.cache is None:
                        self.cache = grads
                    else:
                        self.cache = np.concatenate([self.cache, grads])
                    
                    if len(self.cache) >= self.dims:
                        self.pca.partial_fit(self.cache)
                        self.cache = None
                else:
                    grads = self.pca.transform(grads).astype('float32')
            else:
                self.train_pca = False
            
            if not self.train_pca:
                if self.normalize:
                    grads = preprocessing.normalize(grads).astype('float32')

                if self.collect_grads is None:
                    self.collect_grads = grads
                else:
                    self.collect_grads = np.concatenate([self.collect_grads, grads])
        return hook_fn

    def get_smashed_hook(self):
        def hook_fn(layer, inputs, outputs):
            smashed = inputs[0].cpu().detach().numpy().astype('float32')
            shape = smashed.shape
            smashed = smashed.reshape(len(smashed), -1)
            if np.prod(shape[1:]) > self.max_dims:
                if self.train_pca:
                    if self.cache is None:
                        self.cache = smashed
                    else:
                        self.cache = np.concatenate([self.cache, smashed])
                    
                    if len(self.cache) >= self.dims:
                        self.pca.partial_fit(self.cache)
                        self.cache = None
                else:
                    smashed = self.pca.transform(smashed).astype('float32')
            else:
                self.train_pca = False
            
            if not self.train_pca:
                if self.normalize:
                    smashed = preprocessing.normalize(smashed).astype('float32')

                if self.collect_smashed is None:
                    self.collect_smashed = smashed
                else:
                    self.collect_smashed = np.concatenate([self.collect_smashed, smashed])
        return hook_fn

class ResultManagement:
    def __init__(self, path):
        self.path = path
        
    def save(self, data):
        with open(self.path, 'wb') as fp:
            pickle.dump(data, fp)
    
    def load(self):
        with open(self.path, 'rb') as fp:
            data = pickle.load(fp)
        return data

def apply_loader(layer, data_path, dataset, composer, batch_size,
                 weight_path, gen_model, pretrain, train=True):
    collecter = CollecterExt(batch_size=batch_size, train_pca=True, 
                             cache_size=cache_size, upper_dims=upper_dims)
    for i in range(0, 2):
        # clear gpu
        torch.cuda.empty_cache()

        # load dataset
        SplitDP.static_set_seed()
        data = ExpDataset(data_path, dataset, composer)
        train_loader, val_loader = data.load_dataset(batch_size, drop_last=True)
        labels_set = data.load_labels_set()
        
        # load model
        SplitDP.static_set_seed()
        model = gen_model(len(labels_set), pretrain) 
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999)) 

        # extract grads
        splitdp = SplitDP(model, loss_fn, optimizer)
        splitdp.load_checkpoint(weight_path)
        if i == 0:
            collecter.train_pca = True
        else:
            collecter.train_pca = False
        smashed_hook = collecter.get_smashed_hook()
        splitdp.attach_forward_hooks(layer, smashed_hook)

        splitdp.collect_labels = None
        if train:
            # extract training activations
            rs = splitdp.naiv_loader_apply(train_loader, splitdp.collect_labels_fun)
        else:
            # extract test activations
            rs = splitdp.naiv_loader_apply(val_loader, splitdp.collect_labels_fun)

        if collecter.collect_smashed is not None:
            # no pca reduction, break
            break
    return collecter.collect_smashed, splitdp.collect_labels, labels_set

def smash_cluster(smashed, labels, labels_set, 
                 metric, 
                 known, 
                 known_labels=None, known_raw=None):
    if known_labels is None:
        # km = KMeans(n_clusters=len(labels_set))
        # km.fit(smashed)
        # known_labels, pred_labels = cluster_accuracy_based_on_known(labels, 
        #                                                             km.labels_, known,
        #                                                             known_labels)
        # attack_score = metric(labels, pred_labels)
        known_labels = generate_known_dict(labels, known)
        metric_index = smashed.shape[0]
        init_smashes = []
        known_raw = {}
        for label, indexes in known_labels.items():
            known_raw[label] = []
            for index in indexes:
                known_raw[label].append(smashed[index].reshape(1, -1))
            known_raw[label] = np.concatenate(known_raw[label])

            init_smashes.append(known_raw[label][np.random.choice(len(known_raw[label]))])
    else:
        # mix the known data into the unknown
        init_smashes = []
        metric_index = smashed.shape[0]
        for label, raw in known_raw.items():
            base_index = smashed.shape[0]
            smashed = np.concatenate([smashed, raw])
            labels = np.concatenate([labels,
                                     np.array([label] * len(raw))])

            init_smashes.append(raw[np.random.choice(len(raw))])
            # update known indexes
            for i, index in enumerate(known_labels[label]):
                known_labels[label][i] = base_index + i

    km = KMeans(n_clusters=len(labels_set), init=np.vstack(init_smashes), n_init=1)
    km.fit(smashed)
    known_labels, pred_labels = cluster_accuracy_based_on_known(labels, 
                                                 km.labels_, known, known_labels)
    attack_score = metric(labels[:metric_index], 
                                  pred_labels[:metric_index])

    return known_labels, known_raw, attack_score

def bad_knn(raw_data, labels,
              metric, 
              known, 
              known_labels=None, known_raw=None, neighbors=1):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    if known_labels is None:
        known_raw, known_labels, unknown_raw, unknown_labels = \
                   split_based_on_known(raw_data, labels, known)
    else:
        unknown_raw = raw_data
        unknown_labels = labels

    knn.fit(known_raw, known_labels)
    pred_labels = knn.predict(unknown_raw)
    attack_score = metric(unknown_labels, pred_labels)
    return known_labels, known_raw, attack_score

def calc_layer_num(module):
    local_children = []
    queue = list(module.children())
    while len(queue) > 0:
        child = queue.pop(0)
        child_children = list(child.children())
        if len(child_children) > 0:
            queue.extend(child_children)
        else:
            local_children.append(child)

    if len(local_children) == 0:
        local_children.append(module)
    return len(local_children)

def extract_feature(model, loader, device):
    model = model.to(device)
    features = None
    labels = None
    model.eval()
    for i, (x, y) in enumerate(loader):
        output = model(x.to(device))
        if i == 0:
            features = output.detach().cpu()
            labels = y.cpu()
        else:
            features = torch.cat([features, output.detach().cpu()])
            labels = torch.cat([labels, y.cpu()])
    return features, labels

def get_model_children(model):
    children = []
    modules = model.children()
    for module in modules:
        if isinstance(module, torch.nn.Sequential):
            children.extend(list(module.children()))
        else:
            children.append(module)
            if isinstance(module, torch.nn.AdaptiveAvgPool2d):
                children.append(nn.Flatten())

    return children
