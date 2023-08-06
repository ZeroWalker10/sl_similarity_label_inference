import torch
from torch.autograd import Variable
from torch import nn
from .defense.labelleakage.max_norm import max_norm 
from .defense.labelleakage.gradient_pruning import grad_pruning
from .defense.labelleakage.gradient_KL_perturb import gradient_KL_perturb
from .defense.labelleakage.ldp.ldp import rrwith_prior
from .defense.labelleakage.max_norm import max_norm 
from .defense.labelleakage.coae import cross_entropy, cross_entropy_for_onehot
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import copy, pdb, math
from opacus import PrivacyEngine 
from opacus.validators import ModuleValidator
from .util import l1_regularization, l2_regularization
import numpy as np

class Role:
  def __init__(self, role_type, step, sub_model, optimizer, loss_fn=None):
    '''
    role_type: 'client' or 'server'
    step: which step this role locates on for the whole model
          0: first step, -1: last step
    sub_model: sub model that the role will execute
    optimizer: optimizer to update arguments
    loss_fn: loss function
    '''
    self.role_type = role_type
    self.step = step
    self.sub_model = sub_model
    self.intermediate = None
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.predictions = None
    self.scheduler = None # learning rate scheduler
    self.is_freeze = False
    self.is_batch_lr_scheduler = False

  def on_epoch_start(self, args=None):
    pass

  def on_epoch_end(self, args=None):
    pass

  def on_batch_start(self, args=None):
    pass

  def on_batch_end(self, args=None):
    pass

  def train(self):
    self.sub_model.train()

  def eval(self):
    self.sub_model.eval()

  def unfreeze(self):
    for parameter in self.sub_model.parameters():
      parameter.requires_grad = True
    self.is_freeze = False

  def freeze(self):
    for parameter in self.sub_model.parameters():
      parameter.requires_grad = False
    self.is_freeze = True

  def save(self, filename):
    state = {
      'model_state_dict': self.sub_model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict()
    }
    torch.save(state, filename)

  def restore(self, filename):
    state = torch.load(filename)
    self.sub_model.load_state_dict(state['model_state_dict'])
    self.optimizer.load_state_dict(state['optimizer_state_dict'])

  def set_scheduler(self, scheduler=None):
    self.scheduler = scheduler
    if (isinstance(scheduler, lr_scheduler.CyclicLR) or
        isinstance(scheduler, lr_scheduler.OneCycleLR) or
        isinstance(scheduler, lr_scheduler.CosineAnnealingWarmRestarts)):
      self.is_batch_lr_scheduler = True
    else:
      self.is_batch_lr_scheduler = False

  def scheduler_step(self, is_batch):
    if self.scheduler is not None:
      if (is_batch and self.is_batch_lr_scheduler) or (not is_batch and not self.is_batch_lr_scheduler):
        self.scheduler.step()
  
  def clone(self):
    role_type = self.role_type
    step = self.step
    sub_model = copy.deepcopy(self.sub_model)
    optimizer = copy.deepcopy(self.optimizer)
    loss_fn = copy.deepcopy(self.loss_fn)

    # reset model parameters
    def reset_parameters(m):
      if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)

    sub_model.apply(reset_parameters)

    # replace optimizer with new parameters
    optimizer.param_groups[0]['params'] = list(sub_model.parameters())
    return Role(role_type, step, sub_model, optimizer, loss_fn)

  def _forward(self, inputs):
    return self.sub_model(inputs)
  
  def forward(self, inputs):
    if self.step != -1:
      self.intermediate = self.sub_model(inputs)
      # outputs = self.intermediate.detach().requires_grad_(True)
      outputs = Variable(self.intermediate.data, requires_grad=True)
      if self.intermediate.requires_grad:
        self.intermediate.retain_grad()
    else:
      # last step
      self.intermediate = inputs
      outputs = self.sub_model(self.intermediate)
      self.predictions = outputs
    return outputs 
  
  def backward(self, inputs):
    if not self.is_freeze:
      self.optimizer.zero_grad()
    if self.step != -1 and not self.is_freeze:
      # inputs are gradients
      self.intermediate.backward(inputs)
    elif self.step == -1:
      # inputs are labels
      loss = self.loss_fn(self.predictions, inputs)
      loss.backward()

    if not self.is_freeze:
      self.optimizer.step()
    return self.intermediate.grad

class MaxNormRole(Role):
  def __init__(self, role_type, step, sub_model, optimizer, loss_fn=None):
    super(MaxNormRole, self).__init__(role_type, step, sub_model, optimizer, loss_fn)

  def backward(self, inputs):
    if not self.is_freeze:
      self.optimizer.zero_grad()
    if self.step != -1 and not self.is_freeze:
      # inputs are gradients
      self.intermediate.backward(inputs)
    elif self.step == -1:
      # inputs are labels
      loss = self.loss_fn(self.predictions, inputs)
      loss.backward()

    if not self.is_freeze:
      self.optimizer.step()
    
    # defense
    if self.step != -1:
      grad = self.intermediate.grad
    else:
      grad = self.intermediate.grad.clone()
      grad = max_norm(grad)
    return grad 

class RRLabelRole(Role):
  def __init__(self, role_type, step, sub_model, optimizer, loss_fn=None, rrlabel_args=None):
    super(RRLabelRole, self).__init__(role_type, step, sub_model, optimizer, loss_fn)
    self.rrlabel_args = rrlabel_args

  def backward(self, inputs, indexes=None, lamb=None):
    if not self.is_freeze:
      self.optimizer.zero_grad()
    if self.step != -1 and not self.is_freeze:
      # inputs are gradients
      self.intermediate.backward(inputs)
    elif self.step == -1:
      # inputs are labels
      nlabels = []
      for label in inputs:
        nlabel = rrwith_prior(label.cpu().item(), self.rrlabel_args['prior'],
                     self.rrlabel_args['epsilon'])
        nlabels.append(nlabel)

      nlabels = torch.from_numpy(np.array(nlabels)).to(inputs.device)
      nlabels_alt = nlabels[indexes]
      loss = lamb * self.loss_fn(self.predictions, nlabels) + \
          (1 - lamb) * self.loss_fn(self.predictions, nlabels_alt)
      loss.backward()
    if not self.is_freeze:
      self.optimizer.step()
    return self.intermediate.grad

class MixMatchRole(Role):
  def __init__(self, role_type, step, sub_model, optimizer, loss_fn=None, mixup_args=None):
    super(MixMatchRole, self).__init__(role_type, step, sub_model, optimizer, loss_fn)
    self.mixup_args = mixup_args


  def backward(self, inputs, lamb, labels_set, critical_point):
    # critical point is the boundary between the labeled and unlabeled samples
    if not self.is_freeze:
      self.optimizer.zero_grad()
    if self.step != -1 and not self.is_freeze:
      # inputs are gradients
      self.intermediate.backward(inputs)
    elif self.step == -1:
      # inputs are labels over distribution
      pred_softmax = F.softmax(self.predictions, dim=1)

      labeled = inputs[:critical_point]
      unlabeled = inputs[critical_point:]
      labeled_softmax = pred_softmax[:critical_point]
      unlabeled_softmax = pred_softmax[critical_point:]

      labeled_log = torch.log(labeled_softmax + 1.0e-8)
      labeled_loss = -torch.sum(labeled_log * labeled) / len(labeled)

      unlabeled_loss = torch.sum((unlabeled - unlabeled_softmax) ** 2) /  \
                          (len(unlabeled) + len(labels_set))

      loss = labeled_loss + lamb * unlabeled_loss
      loss.backward()
    if not self.is_freeze:
      self.optimizer.step()
    return self.intermediate.grad

class GradPruningRole(Role):
  def __init__(self, role_type, step, sub_model, optimizer, loss_fn=None, pruning_args=None):
    super(GradPruningRole, self).__init__(role_type, step, sub_model, optimizer, loss_fn)
    self.pruning_args = pruning_args
    # self.residuals = None

  def backward(self, inputs):
    if not self.is_freeze:
      self.optimizer.zero_grad()
    if self.step != -1 and not self.is_freeze:
      # inputs are gradients
      self.intermediate.backward(inputs)
    elif self.step == -1:
      # inputs are labels
      loss = self.loss_fn(self.predictions, inputs)
      loss.backward()

    if not self.is_freeze:
      self.optimizer.step()
    
    # defense
    if self.step != -1:
      grad = self.intermediate.grad
    else:
      raw_grad = self.intermediate.grad.clone()
      grad = grad_pruning(raw_grad, self.pruning_args['ratio'])
    return grad 

class KLPerturbRole(Role):
  def __init__(self, role_type, step, sub_model, optimizer, loss_fn=None, kl_args=None):
    super(KLPerturbRole, self).__init__(role_type, step, sub_model, optimizer, loss_fn)
    self.kl_args = kl_args
    # self.residuals = None

  def backward(self, inputs):
    if not self.is_freeze:
      self.optimizer.zero_grad()
    if self.step != -1 and not self.is_freeze:
      # inputs are gradients
      self.intermediate.backward(inputs)
    elif self.step == -1:
      # inputs are labels
      loss = self.loss_fn(self.predictions, inputs)
      loss.backward()

    if not self.is_freeze:
      self.optimizer.step()
    
    # defense
    if self.step != -1:
      grad = self.intermediate.grad
    else:
      raw_grad = self.intermediate.grad.clone()
      grad = gradient_KL_perturb(raw_grad, inputs, self.kl_args['threshold'])
    return grad 

class OpacusRole(Role):
  def __init__(self, role_type, step, sub_model, optimizer, loss_fn=None, opacus_args=None):
    super(OpacusRole, self).__init__(role_type, step, sub_model, optimizer, loss_fn)

    # self.sub_model = ModuleValidator.fix(self.sub_model)
    # ModuleValidator changes the model parameters, need to update optimizer
    # self.optimizer.param_groups[0]['params'] = list(self.sub_model.parameters())

    # opacus 0.15.0
    '''
    self.privacy_engine = PrivacyEngine(
      self.sub_model,
      #epochs=opacus_args['epochs'],
      batch_size=opacus_args['batch_size'],
      sample_size=opacus_args['sample_size'],
      target_epsilon=opacus_args['epsilon'],
      target_delta=opacus_args['delta'],
      max_grad_norm=opacus_args['max_grad_norm'],
      secure_rng=True,
    )
    self.privacy_engine.attach(self.optimizer)
    '''
    # opacus 1.1.3
    self.privacy_engine = PrivacyEngine(secure_mode=False)
    self.sub_model, self.optimizer, self.data_loader = self.privacy_engine.make_private_with_epsilon(
      module=self.sub_model,
      optimizer=self.optimizer,
      data_loader=opacus_args['data_loader'],
      epochs=opacus_args['epochs'],
      target_epsilon=opacus_args['epsilon'],
      target_delta=opacus_args['delta'],
      max_grad_norm=opacus_args['max_grad_norm'],
    )
    '''
    self.sub_model, self.optimizer, self.data_loader = self.privacy_engine.make_private(
      module=self.sub_model,
      optimizer=self.optimizer,
      data_loader=opacus_args['data_loader'],
      noise_multiplier=opacus_args['noise_multiplier'],
      # epochs=opacus_args['epochs'],
      # target_epsilon=opacus_args['epsilon'],
      # target_delta=opacus_args['delta'],
      max_grad_norm=opacus_args['max_grad_norm'],
    )
    '''

    # self.sub_model = self.sub_model.to(opacus_args['device'])
  def get_dataloader(self):
    return self.data_loader
