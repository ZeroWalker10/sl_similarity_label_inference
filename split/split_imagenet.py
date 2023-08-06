from .split_model import SplitModel
from .pipeline import Pipeline
from .role import Role, MaxNormRole, OpacusRole, GradPruningRole, OpacusPruningRole, MaxNormPruningRole, SoftLabelRole, LabelSmoothingRole, NoiseLabelRole
from .entry import Entry
from torch import optim, nn
import torch.nn.functional as F
from .defense.labelleakage.antipodes.alibi import Ohm

class GlobalMaxPooling(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()

    def forward(self, inputs):
        out = F.max_pool2d(inputs, kernel_size=inputs.size()[2:])
        return out

class SplitImagenet(SplitModel):
    def __init__(self, in_channels, inter_view, n_class, defense=None, defense_args=None):
        super(SplitCifar, self).__init__()

        self.defense = defense
        self.defense_args = defense_args

        # layer 1
        self.layers.append([
            nn.Conv2d(in_channels, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        ])

        # layer 2
        self.layers.append([
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 3
        self.layers.append([
            nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        ])

        # layer 4
        self.layers.append([
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])


        # layer 5
        self.layers.append([
            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        ])

        # layer 6
        self.layers.append([
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 7
        self.layers.append([
            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        ])

        # layer 8
        self.layers.append([
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 9
        self.layers.append([
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            #nn.Dropout2d(p=0.1),
        ])

        # layer 10
        self.layers.append([
            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        ])

        # layer 11
        self.layers.append([
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 12
        self.layers.append([
            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        ])

        # layer 13
        self.layers.append([
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 14
        self.layers.append([
            nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        ])

        # layer 15
        self.layers.append([
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 16
        self.layers.append([
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            #nn.Dropout2d(p=0.1),
        ])

        # layer 17
        self.layers.append([
            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        ])

        # layer 18
        self.layers.append([
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 19
        self.layers.append([
            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        ])

        # layer 20
        self.layers.append([
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 21
        self.layers.append([
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            #nn.Dropout2d(p=0.1),
        ])

        # layer 22
        self.layers.append([
            nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        ])

        # layer 23
        self.layers.append([
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 24
        self.layers.append([
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            #nn.Dropout2d(p=0.1),
        ])

        # layer 25
        self.layers.append([
            nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
        ])

        # layer 26
        self.layers.append([
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 27
        self.layers.append([
            nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
        ])

        # layer 28
        self.layers.append([
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 29
        self.layers.append([
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            #nn.Dropout2d(p=0.1),
        ])

        # layer 30
        self.layers.append([
            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        ])

        # layer 31
        self.layers.append([
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
        ])

        # layer 32
        self.layers.append([
            # GlobalMaxPooling(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            #nn.Dropout(p=0.1),
        ])

        # layer 33
        self.layers.append([
            nn.Flatten(),
            nn.Linear(inter_view, n_class)
        ])

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

            # initilize weights
            for m in sub_model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

            start_layer = end_layer

            sub_model = sub_model.to(device)
            optimizer = optim.Adadelta(sub_model.parameters(), lr=0.1, rho=0.9, eps=1e-3, weight_decay=0.001)
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
