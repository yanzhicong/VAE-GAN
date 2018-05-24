
import os
import sys
sys.path.append('../')
# from network.weightsinit import get_weightsinit

from network.vgg import VGG16


discriminator_dict = {
    "DiscriminatorVGG16" : VGG16
}

discriminator_params_dict = {
    "DiscriminatorVGG16" : {
        'name':'DiscriminatorVGG16',
    }
}


def get_discriminator(name, config, model_config):
    if name in discriminator_dict:
        return discriminator_dict[name](config, model_config, **discriminator_params_dict[name])
    else : 
        raise Exception("None discriminator named " + name)



