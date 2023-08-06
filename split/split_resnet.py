#!/usr/bin/env python
# coding=utf-8
from .pipeline import Pipeline
from .role import Role, MaxNormRole, OpacusRole, GradPruningRole
from .role import RRLabelRole, MixMatchRole, KLPerturbRole
from .entry import Entry
from .defense.labelleakage.coae import cross_entropy, cross_entropy_for_onehot
from torch import optim, nn
from opacus.validators import ModuleValidator
import torch.optim.lr_scheduler as lr_scheduler
import pdb
import numpy as np

class SplitResnet:
    def __init__(self, sub_models, n_class, lr, defense=None, defense_args=None):
        self.defense = defense
        self.defense_args = defense_args

        self.n_class = n_class
        self.sub_models = sub_models
        self.lr = lr

    def split(self, device, optimizer_args=None, scheduler_args=None):
        client_pipeline, server_pipeline, total_pipeline = Pipeline([]), Pipeline([]), Pipeline([])
        for i, sub_model in enumerate(self.sub_models):
            if self.defense == 'opacus' and i == len(self.sub_models) - 1:
              sub_model = ModuleValidator.fix(sub_model)

            loss_fn = nn.CrossEntropyLoss()
            if i != len(self.sub_models) - 1:
                role_type = 'client'
                step = i
                last_step = False
            else:
                role_type = 'server'
                step = -1
                last_step = True

            lr = self.lr

            if self.defense == 'AutoEncoderRole' and last_step:
              ae = self.defense_args['autoencoder']

            sub_model = sub_model.to(device)
            if optimizer_args is None:
              optimizer = optim.SGD(sub_model.parameters(), lr=lr)
            else:
              opt_method = optimizer_args.get('opt_method', 'sgd')
              if opt_method == 'sgd':
                momentum = optimizer_args.get('momentum', 0.9)
                weight_decay = optimizer_args.get('weight_decay', 5e-4)
                if self.defense == 'AutoEncoderRole' and last_step:
                  optimizer = optim.SGD(list(sub_model.parameters()) + list(ae.parameters()), 
                                        lr=lr, momentum=momentum, weight_decay=weight_decay)
                else:
                  optimizer = optim.SGD(sub_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
              elif opt_method == 'adam':
                betas = optimizer_args.get('betas', (0.9, 0.999))
                weight_decay = optimizer_args.get('weight_decay', 0.001)
                eps = optimizer_args.get('eps', 0.001)
                if self.defense == 'AutoEncoderRole' and last_step:
                  optimizer = optim.Adam(list(sub_model.parameters()) + list(ae.parameters()), lr=lr, eps=eps, weight_decay=weight_decay)
                else:
                  optimizer = optim.Adam(sub_model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
              elif opt_method == 'rmsprop':
                if self.defense == 'AutoEncoderRole' and last_step:
                  optimizer = optim.RMSprop(list(sub_model.parameters()) + list(ae.parameters()), lr=lr)
                else:
                  optimizer = optim.RMSprop(sub_model.parameters(), lr=lr)

            if scheduler_args is not None:
              sch_method = scheduler_args.get('sch_method', None)
              if sch_method == 'steplr':
                step_size = scheduler_args['step_size']
                gamma = scheduler_args.get('gamma', 0.1)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
              elif sch_method == 'multi_steplr':
                milestones = scheduler_args['milestones']
                gamma = scheduler_args.get('gamma', 0.1)
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
              elif sch_method == 'one_cyclelr':
                epochs = scheduler_args['epochs']
                steps_per_epoch = scheduler_args['step_size']
                scheduler = lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs,
                                                    steps_per_epoch=steps_per_epoch)
              else:
                scheduler = None
            else:
              scheduler = None


            if self.defense == 'max_norm' and last_step:
              role = MaxNormRole(role_type, step, sub_model, optimizer, loss_fn)
            elif self.defense == 'opacus' and last_step:
              role = OpacusRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'grad_pruning' and last_step:
              role = GradPruningRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'opacus_pruning' and last_step:
              role = OpacusPruningRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'maxnorm_pruning' and last_step:
              role = MaxNormPruningRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'softlabel' and last_step:
              role = SoftLabelRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'labelsmoothing' and last_step:
              role = LabelSmoothingRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'noiselabel' and last_step:
              role = NoiseLabelRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'rrlabel' and last_step:
              role = RRLabelRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'klperturb' and i == last_step:
              role = KLPerturbRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'mixup' and last_step:
              role = MixupRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'mixmatch' and last_step:
              role = MixMatchRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'multihead' and last_step:
              role = MultiHeadRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'CoAE' and last_step:
              role = CoAERole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'SAE' and last_step:
              loss_fn = cross_entropy
              role = SAERole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'PELoss' and last_step:
              role = PELossRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'DiffusionLoss' and last_step:
              role = DiffusionLossRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'FakeDiffusionLoss' and last_step:
              role = DiffusionLossWithFakeLabelRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'DistCorrelationLoss' and last_step:
              role = DistCorrelationLossRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'NaiveDiffusionLoss' and last_step:
              role = NaiveDiffusionLossRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'NaiveDiffusionExtLoss' and last_step:
              role = NaiveDiffusionLossExtRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'NaiveDiffusionExtWithFakeLabelLoss' and last_step:
              # loss_fn = cross_entropy
              role = NaiveDiffusionLossExtWithFakeLabelRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'NaiveMultiLoss' and last_step:
              role = NaiveMultiLossRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'AutoEncoderRole' and last_step:
              role = AutoEncoderRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'CrossEntropyLoss' and last_step:
              loss_fn = cross_entropy_for_onehot
              role = Role(role_type, step, sub_model, optimizer, loss_fn)
            else:
              role = Role(role_type, step, sub_model, optimizer, loss_fn)

            if scheduler is not None:
              role.set_scheduler(scheduler)

            if role_type == 'client':
              client_pipeline.push(Entry('local', role))
              server_pipeline.push(Entry('remote', None))
            elif role_type == 'server':
              client_pipeline.push(Entry('remote', None))
              server_pipeline.push(Entry('local', role))

            total_pipeline.push(Entry('local', role))

        return client_pipeline, server_pipeline, total_pipeline
