import numpy as np


from .mnist import MNIST


dataset_dict = {
    'mnist' : MNIST,
}





def get_dataset(name, config):
    if name in dataset_dict:
        return dataset_dict[name](config)
    else:
        raise Exception('None dataset named ' + name)




