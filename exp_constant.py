#!/usr/bin/env python
# coding=utf-8
# configure transform
import torch, logging
from torch import nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage, CenterCrop, RandomResizedCrop
from torchvision.transforms import Grayscale
from split.resnet import specific_resnet50_three_channels, specific_resnet50_single_channel
from split.resnet import specific_resnet18_three_channels, specific_resnet18_single_channel
from split.vgg import specific_vgg16_single_channel, specific_vgg16_three_channels
from sklearn.metrics import f1_score, accuracy_score
from torchvision.models import alexnet, vgg16, inception_v3, resnet50, resnet18

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
composer = Compose([Resize(224), CenterCrop(224),
                   ToTensor(), normalizer])
composers2 = [composer, composer]

composer = Compose([Grayscale(num_output_channels=3),
                     Resize(224), CenterCrop(224), ToTensor(), normalizer])
composers4 = [composer, composer]

composer = Compose([Resize([64, 64]),
                  ToTensor(), normalizer])
composers3 = [composer, composer]

# composer = Compose([Resize(32), CenterCrop(32), ToTensor()])
composer = Compose([Resize([32, 32]), ToTensor()])
composers1 = [composer, composer]

# define dataset path
data_path = '~/Data/split_dp/dataset'
# data_path = './dataset/dataset'
dogs_vs_cats_path = '~/Data/split_dp/dataset/dogs-vs-cats/train'
# dogs_vs_cats_path = './dataset/competitions/dogs-vs-cats/train'
hist_path = '~/Data/split_dp/dataset/histopathologic-cancer-detection/small_train'
intel_path = '~/Data/split_dp/dataset/intel-image-classification'
# flowers_path = './dataset/102flowers/jpg'
flowers_path = '~/Data/split_dp/dataset/102flowers/jpg'
foods_path = '~/Data/split_dp/dataset/food-101/images'
caltech256_path = '~/Data/split_dp/dataset/caltech256/256_ObjectCategories'
caltech101_path = '~/Data/split_dp/dataset/caltech101/train_test'
imagenet_path = '~/Data/split_dp/dataset/imagenet/train'
cars_path = '~/Data/split_dp/dataset/stanfordcars'
fruits_path = '~/Data/split_dp/dataset/fruits-360_dataset/fruits-360'

test_pathes = [data_path, data_path, dogs_vs_cats_path, hist_path, intel_path,
               fruits_path, imagenet_path, caltech256_path]
test_composers = [composers1, None, composers3, composers3,
                  composers3, composers3, composers2, composers3]
transfer_composers = [composers4, composers2, composers2, composers2,
                  composers2, composers2, composers2, composers2]
test_datasets = ['fashion-mnist', 'cifar10', 'dogs_vs_cats',
                 'hist_cancer_detection', 'intel_classification',
                 'fruits', 'image-net', 'caltech256']
model_zoo = ['alexnet', 'vgg16', 'resnet18', 'resnet50']

# dataset-layername
prev_model_name_fmt = 'models/{}-{}-prev.pt'
back_model_name_fmt = 'models/{}-{}-back.pt'

cache_size = 512
upper_dims = 512

known_labels_per_class = 1
labeled_batch_size = 8

mixup_alpha = 0.2
lamb_u = 50
temp = 0.8

max_tries = 5

params = {
    'simplenet': {
	'lrs':     [0.005, 0.005, 0.002, 0.002, 0.003, 0.005, 0.005],
        'batches': [128,   128,   64,    64,    128,   128,   64],
        'epochs':  [50,    150,   150,   140,   150,   15,    300],
        'mc_epochs': [20, 20, 20, 20, 20, 20, 20],
	# 'lrs': [0.005, 0.005, 0.005, 0.005, 0.005, 0.001,  0.001],
        # 'batches': [128, 128, 64, 64, 64, 32, 32],
        # 'epochs': [50, 1, 60, 45, 80, 70, 30],
        # 'mc_epochs': [20, 20, 20, 20, 20, 20, 20],
        # 'mc_epochs': [20, 20, 20, 20, 20, 20, 20],
        'transfer-suffix': ['multi', 'multi', 'multi', 'multi', 'multi', 'multi',
                            'multi'],
        'transfer_batches': [64, 64, 64, 64, 64, 64, 64],
        'metric': [accuracy_score, accuracy_score, f1_score, f1_score,
                   accuracy_score, accuracy_score, accuracy_score],
    },
    'resnet50': {
	'lrs': [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5],
        'models': [specific_resnet50_single_channel,
                   specific_resnet50_three_channels,
                   specific_resnet50_three_channels,
                   specific_resnet50_three_channels,
                   specific_resnet50_three_channels,
                   specific_resnet50_three_channels,
                   specific_resnet50_three_channels,
                   specific_resnet50_three_channels],
        'batches': [64, 64, 32, 32, 32, 32, 32, 32],
        'opts': ['Adam', 'Adam', 'Adam', 'Adam', 'Adam', 'Adam', 'Adam', 'Adam'],
        'weight_pathes': ['fmnist-resnet.pt', 'cifar10-resnet.pt',
                 'dogs_vs_cats-resnet.pt', 'hist-resnet.pt',
                 'intel-resnet.pt', 'flowers-resnet.pt',
                 'caltech-resnet.pt', 'imagenet-resnet.pt']
    },
    'resnet18': {
	'lrs': [0.001, 0.01, 0.0005, 0.0005, 0.002, 0.01, 0.0005, 0.0005],
        'models': [specific_resnet18_single_channel,
                   specific_resnet18_three_channels,
                   specific_resnet18_three_channels,
                   specific_resnet18_three_channels,
                   specific_resnet18_three_channels,
                   specific_resnet18_three_channels,
                   specific_resnet18_three_channels,
                   specific_resnet50_three_channels],
        'batches': [128, 64, 64, 64, 32, 32, 32, 32],
        'mc_epochs': [10, 20, 20, 20, 5, 20, 1, 1],
        'pretrain': [False, False, False, False, False, False, True, True],
        'epochs': [30, 50, 60, 50, 15, 70, 100, 50],
        'metric': [accuracy_score, accuracy_score, f1_score, f1_score,
                   accuracy_score, accuracy_score, accuracy_score, accuracy_score, accuracy_score],
        'weight_pathes': ['fmnist-resnet18-raw.pt', 'cifar10-resnet18-raw.pt',
                 'dogs_vs_cats-resnet18-raw.pt', 'hist-resnet18-raw.pt',
                 'intel-resnet18-raw.pt', 'flowers-resnet18-raw.pt',
                 'caltech-resnet18-raw.pt', 'imagenet-resnet18-raw.pt']
    },
    'vgg16': {
	'lrs': [1e-6, 1e-3, 0.01, 0.01, 0.001, 1e-5, 1e-5, 1e-5],
        'models': [specific_vgg16_single_channel,
                   specific_vgg16_three_channels,
                   specific_vgg16_three_channels,
                   specific_vgg16_three_channels,
                   specific_vgg16_three_channels,
                   specific_vgg16_three_channels,
                   specific_vgg16_three_channels,
                   specific_vgg16_three_channels],
        'batches': [128, 64, 32, 32, 32, 32, 32, 32],
        'pretrain': [False, False, False, False, False, True, True, True],
        'epochs': [40, 50, 30, 20, 20, 10, 10, 25],
        'opts': ['Adam', 'Adam', 'Adam', 'Adam', 'Adam', 'Adam', 'Adam', 'Adam'],
        'metric': [accuracy_score, accuracy_score, f1_score, f1_score,
                   accuracy_score, accuracy_score, accuracy_score, accuracy_score],
        'weight_pathes': ['fmnist-vgg16-raw.pt', 'cifar10-vgg16-raw.pt',
                 'dogs_vs_cats-vgg16-raw.pt', 'hist-vgg16-raw.pt',
                 'intel-vgg16-raw.pt', 'flowers-vgg16-raw.pt',
                 'caltech-vgg16-raw.pt', 'imagenet-vgg16-raw.pt']
    }
}


def config_logger(mode='train', fname='train.log'):
    logger = logging.getLogger(mode)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    fh = logging.FileHandler(fname, 'w', encoding='utf-8')

    formatter = logging.Formatter('%(asctime)s-%(filename)s-%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

config_logger(mode='train', fname='./logs/split.log')
logger = logging.getLogger('train')

def print_log(msg_list, oriention='print'):
    msg = ''
    for m in msg_list:
        if msg != '':
            msg = msg + ' ' + str(m)
        else:
            msg = str(m)

    if oriention == 'print':
        print(msg)
    elif oriention == 'logger':
        logger.info(msg)

def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

def get_transfer_model(model_name):
    if 'alexnet' in model_name:
        return alexnet_transfer(model_name)
    elif 'vgg16' in model_name:
        return vgg16_transfer(model_name)
    elif 'inception_v3' in model_name:
        return inception_transfer(model_name)
    elif 'resnet18' in model_name:
        return resnet_transfer(18, model_name)
    elif 'resnet50' in model_name:
        return resnet_transfer(50, model_name)
    else:
        return 'invalid model name'

def alexnet_transfer(model_name):
    model = alexnet(pretrained=True)
    if 'single1' in model_name:
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if 'single3' in model_name:
        model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.classifier[6] = nn.Identity()
    freeze_model(model)
    model.eval()
    return model

def vgg16_transfer(model_name):
    model = vgg16(pretrained=True)
    if 'single1' in model_name:
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if 'single3' in model_name:
        model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.classifier[6] = nn.Identity()
    freeze_model(model)
    model.eval()
    return model

def inception_transfer(model_name):
    model = inception_v3(pretrained=True)
    if 'single1' in model_name:
        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if 'single3' in model_name:
        model.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.fc = nn.Identity()
    freeze_model(model)
    model.eval()
    return model

def resnet_transfer(scale, model_name):
    if scale == 50:
        model = resnet50(pretrained=True)
    elif scale == 18:
        model = resnet18(pretrained=True)
    elif scale == 34:
        model = resnet34(pretrained=True)
    if 'single1' in model_name:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if 'single3' in model_name:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.fc = nn.Identity()
    freeze_model(model)
    model.eval()
    return model

def transfer_surrogate_top(input_shape, n_class):
    sub_model = nn.Sequential()
    sub_model.add_module('final-fc',
                         nn.Linear(input_shape[0], n_class))
    return sub_model


