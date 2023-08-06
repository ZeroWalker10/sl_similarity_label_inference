#!/usr/bin/env python
# coding=utf-8
from .pipeline import Pipeline
from .role import Role, MaxNormRole, OpacusRole, GradPruningRole 
from .role import RRLabelRole, MixMatchRole, KLPerturbRole
from .entry import Entry
from torch import optim, nn
from .defense.labelleakage.coae import cross_entropy, cross_entropy_for_onehot
import torch
from opacus.validators import ModuleValidator
import numpy as np

class SplitSimplenet:
    def __init__(self, sub_models, n_class, lr, defense=None, defense_args=None):
        self.defense = defense
        self.defense_args = defense_args

        self.n_class = n_class
        self.sub_models = sub_models
        self.lr = lr

    def split(self, device):
        client_pipeline, server_pipeline, total_pipeline = Pipeline([]), Pipeline([]), Pipeline([])
        dataloader = None
        for i, sub_model in enumerate(self.sub_models):
            if self.defense == 'opacus' and i == len(self.sub_models) - 1:
                sub_model = ModuleValidator.fix(sub_model)

            sub_model = sub_model.to(device)
            # optimizer = optim.Adadelta(sub_model.parameters(), lr=self.lr, rho=0.9, eps=1e-3, weight_decay=0.001)
            optimizer = optim.SGD(sub_model.parameters(), lr=self.lr)
            loss_fn = nn.CrossEntropyLoss()
            if i != len(self.sub_models) - 1:
                role_type = 'client'
                step = i
                last_step = False
            else:
                role_type = 'server'
                step = -1
                last_step = True

            if self.defense == 'max_norm' and last_step:
              role = MaxNormRole(role_type, step, sub_model, optimizer, loss_fn)
            elif self.defense == 'opacus' and last_step:
              role = OpacusRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
              # dataloader = role.get_dataloader()
            elif self.defense == 'grad_pruning' and last_step:
              role = GradPruningRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'opacus_pruning' and last_step:
              role = OpacusPruningRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
              # dataloader = role.get_dataloader()
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
            elif self.defense == 'NaiveMultiLoss' and last_step:
              role = NaiveMultiLossRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'CrossEntropyLoss' and last_step:
              loss_fn = cross_entropy_for_onehot
              role = Role(role_type, step, sub_model, optimizer, loss_fn)
            else:
              role = Role(role_type, step, sub_model, optimizer, loss_fn)

            if self.defense_args is not None and 'ms' in self.defense_args:
              role.set_scheduler(self.defense_args['ms'])

            if role_type == 'client':
              client_pipeline.push(Entry('local', role))
              server_pipeline.push(Entry('remote', None))
            elif role_type == 'server':
              client_pipeline.push(Entry('remote', None))
              server_pipeline.push(Entry('local', role))

            total_pipeline.push(Entry('local', role))
        '''
        if self.defense == 'opacus':
          return client_pipeline, server_pipeline, total_pipeline, dataloader
        else:
          return client_pipeline, server_pipeline, total_pipeline
        '''
        return client_pipeline, server_pipeline, total_pipeline
