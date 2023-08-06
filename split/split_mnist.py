from .split_model import SplitModel
from .pipeline import Pipeline
from .role import Role, MaxNormRole, OpacusRole, GradPruningRole, OpacusPruningRole, MaxNormPruningRole, SoftLabelRole, LabelSmoothingRole, NoiseLabelRole
from .role import RRLabelRole, MixupRole, MixMatchRole
from .entry import Entry
from torch import optim, nn
import numpy as np

class SplitMnist(SplitModel):
    def __init__(self, in_channels, inter_view, n_class, defense=None, defense_args=None):
        super(SplitMnist, self).__init__()

        self.defense = defense
        # defense arguments dictionary
        self.defense_args = defense_args
        
        # layer 1
        self.layers.append([nn.Conv2d(in_channels, 6, kernel_size=3), nn.ReLU()])
        
        # layer 2
        self.layers.append([nn.MaxPool2d(kernel_size=2, stride=2)])
        
        # layer 3
        self.layers.append([nn.Conv2d(6, 16, kernel_size=2), nn.ReLU()])
        
        # layer 4
        self.layers.append([nn.MaxPool2d(kernel_size=2, stride=2)])
        
        # layer 5
        self.layers.append([nn.Flatten(), nn.Linear(inter_view, 120), nn.ReLU()])
        
        # layer 6
        self.layers.append([nn.Linear(120, 84), nn.ReLU()])
        
        # layer 7
        self.layers.append([nn.Linear(84, n_class)])
        self.n_class = n_class

    def surrogate_top(self, input_shape, device):
        sub_model = nn.Sequential()
        if len(input_shape) == 1:
            # dense layer
            sub_model.add_module('classifier', nn.Linear(input_shape[0],
                                                         self.n_class))
        else:
            sub_model.add_module('flatten', nn.Flatten())
            sub_model.add_module('dense1', nn.Linear(np.prod(input_shape),
                                                     120))
            sub_model.add_module('relu1', nn.ReLU())
            sub_model.add_module('dense2', nn.Linear(120, 84))
            sub_model.add_module('relu2', nn.ReLU())
            sub_model.add_module('dense3', nn.Linear(84, self.n_class))

        optimizer = optim.SGD(sub_model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        role_type = 'server'
        step = -1
        role = MixMatchRole(role_type, step, sub_model.to(device), 
                            optimizer, loss_fn, self.defense_args)
        return role
    
    def split(self, cut_layers):
        client_pipeline, server_pipeline, total_pipeline = Pipeline([]), Pipeline([]), Pipeline([])
        
        start_layer = 0
        last_step = len(cut_layers) - 1
        for i, (end_layer, role_type, device) in enumerate(cut_layers):
            # construct sub model
            sub_model = nn.Sequential()
            for layer_index, layer in enumerate(self.layers[start_layer:end_layer]):
                for module_index, module in enumerate(layer):
                    module_name = ''.join(['module', str(i), '-', str(layer_index), '-', str(module_index)])
                    sub_model.add_module(module_name, module)
            start_layer = end_layer
            
            sub_model = sub_model.to(device)
            if i != last_step:
                optimizer = optim.SGD(sub_model.parameters(), lr=1e-2)
                loss_fn = None
                step = i
            else:
                # last step
                optimizer = optim.SGD(sub_model.parameters(), lr=1e-2)
                loss_fn = nn.CrossEntropyLoss()
                step = -1
            
            if self.defense == 'max_norm' and i == last_step:
              role = MaxNormRole(role_type, step, sub_model, optimizer, loss_fn)
            elif self.defense == 'opacus' and i == last_step:
              role = OpacusRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'grad_pruning' and i == last_step:
              role = GradPruningRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'opacus_pruning' and i == last_step:
              role = OpacusPruningRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'maxnorm_pruning' and i == last_step:
              role = MaxNormPruningRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'softlabel' and i == last_step:
              role = SoftLabelRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'labelsmoothing' and i == last_step:
              role = LabelSmoothingRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'noiselabel' and i == last_step:
              role = NoiseLabelRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'rrlabel' and i == last_step:
              role = RRLabelRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'mixup' and i == last_step:
              role = MixupRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            elif self.defense == 'mixmatch' and i == last_step:
              role = MixMatchRole(role_type, step, sub_model, optimizer, loss_fn, self.defense_args)
            else:
              role = Role(role_type, step, sub_model, optimizer, loss_fn)

            if role_type == 'client':
                client_pipeline.push(Entry('local', role))
                server_pipeline.push(Entry('remote', None))
            elif role_type == 'server':
                client_pipeline.push(Entry('remote', None))
                server_pipeline.push(Entry('local', role))
                
            total_pipeline.push(Entry('local', role))
        return client_pipeline, server_pipeline, total_pipeline
