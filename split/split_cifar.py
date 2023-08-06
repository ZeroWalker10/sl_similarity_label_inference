from .split_model import SplitModel
from .pipeline import Pipeline
from .role import Role, MaxNormRole, OpacusRole, GradPruningRole, OpacusPruningRole, MaxNormPruningRole, SoftLabelRole, LabelSmoothingRole, NoiseLabelRole
from .role import RRLabelRole
from .entry import Entry
from torch import optim, nn
from .defense.labelleakage.antipodes.alibi import Ohm

class SplitCifar(SplitModel):
    def __init__(self, in_channels, inter_view, n_class, defense=None, defense_args=None):
        super(SplitCifar, self).__init__()

        self.defense = defense
        self.defense_args = defense_args

        # layer 1
        self.layers.append([nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)])

        # layer 2
        self.layers.append([nn.Conv2d(64, 64, kernel_size=3, padding=1)])

        # layer 3
        self.layers.append([nn.MaxPool2d(kernel_size=2, stride=2)])

        # layer 4
        self.layers.append([nn.BatchNorm2d(64), nn.ReLU()])

        # layer 5
        self.layers.append([nn.Conv2d(64, 128, kernel_size=3, padding=1)])

        # layer 6
        self.layers.append([nn.Conv2d(128, 128, kernel_size=3, padding=1)])

        # layer 7
        self.layers.append([nn.MaxPool2d(kernel_size=2, stride=2)])

        # layer 8
        self.layers.append([nn.BatchNorm2d(128), nn.ReLU()])

        # layer 9
        self.layers.append([nn.Conv2d(128, 256, kernel_size=3, padding=1)])

        # layer 10
        self.layers.append([nn.Conv2d(256, 256, kernel_size=3, padding=1),])

        # layer 11
        self.layers.append([nn.MaxPool2d(kernel_size=2, stride=2)])

        # layer 12
        self.layers.append([nn.BatchNorm2d(256), nn.ReLU()])

        # layer 13
        self.layers.append([nn.Flatten(), nn.Linear(inter_view, 512)])

        # layer 14
        self.layers.append([nn.Linear(512, 512)])

        # layer 15
        self.layers.append([nn.Linear(512, n_class)])

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
            optimizer = optim.SGD(sub_model.parameters(), lr=1e-2)
            if i != last_step:
                loss_fn = None
                step = i
            else:
                # last step
                if self.defense != 'noiselabel':
                    loss_fn = nn.CrossEntropyLoss()
                else:
                    loss_fn = Ohm(
                        privacy_engine=self.defense_args['randomized_label_privacy'],
                        post_process=self.defense_args['post_process'])
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
