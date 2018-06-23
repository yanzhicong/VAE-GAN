
import os
import sys
sys.path.append('../')
# from network.weightsinit import get_weightsinit

# from network.vgg import VGG16


# discriminator_dict = {
#     "DiscriminatorVGG16" : VGG16
# }

# discriminator_params_dict = {
#     "DiscriminatorSimple" : {
#         'name':'DiscriminatorVGG16',
#     }
# }


def get_discriminator(name, config, is_training):
    if name == 'DiscriminatorSimple':
        return discriminator_dict[name](config, is_training)
    else : 
        raise Exception("None discriminator named " + name)



