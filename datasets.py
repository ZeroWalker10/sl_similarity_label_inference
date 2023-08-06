from torchvision import datasets
import pdb

data_path = './dataset'

'''
# MNIST
print('Downloading MNIST dataset...')
mnist = datasets.MNIST(data_path, train=True, download=True)
mnist_val = datasets.MNIST(data_path, train=False, download=True)
print('Finish downloading MNIST!!!')

# Fashion-MNIST
print('Downloading Fashion-MNIST dataset...')
fashion_mnist = datasets.FashionMNIST(data_path, train=True, download=True)
fashion_mnist_val = datasets.FashionMNIST(data_path, train=False, download=True)
print('Finish downloading Fashion-MNIST!!!')

# CIFAR-10
print('Downloading CIFAR10 dataset...')
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)
print('Finish downloading CIFAR10!!!')
'''
# CIFAR-100
# print('Downloading CIFAR100 dataset...')
# cifar10 = datasets.CIFAR100(data_path, train=True, download=True)
# cifar10_val = datasets.CIFAR100(data_path, train=False, download=True)
# print('Finish downloading CIFAR100!!!')

# print('Downloading Places365 dataset...')
# sun397 = datasets.places365.Places365(data_path, download=True)
#sun397_val = datasets.places365.Places365(data_path, train=False, download=True)
# print('Finish downloading SUN365!!!')

#print('Downloading caltech dataset...')
#caltech = datasets.caltech.Caltech265(data_path, download=True)
#caltech_val = datasets.caltech.Caltech265(data_path, train=False, download=True)
#print('Finish downloading caltech!!!')

# datasets.Food101(data_path, split='train', download=True)
# datasets.Food101(data_path, split='test', download=True)

datasets.Caltech101(data_path, target_type='category', download=True)
# datasets.Caltech101(data_path, target_type='category', split='test', download=True)

print('Finish downloading all datasets!!!')
