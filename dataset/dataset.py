import numpy as np


from .mnist import MNIST
from .cifar10 import Cifar10
from .imagenet import ImageNet



dataset_dict = {
    'mnist' : MNIST,
    'cifar10' : Cifar10,
    'imagenet' : ImageNet
}



def get_dataset(name, config):
    if name in dataset_dict:
        return dataset_dict[name](config)
    else:
        raise Exception('None dataset named ' + name)




