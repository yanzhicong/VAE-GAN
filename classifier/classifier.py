

import os
import sys

sys.path.append('../')


from network.vgg import VGG16

from classifier_pixel import ClassifierPixel



classifier_dict = {
    # 'googlenet' : None,
    # 'resnet-101' : None,
    'classifier_pixel' : ClassifierPixel,
    'vgg' : VGG16,
}



classifier_params_dict = {
    'vgg' : {
        'name' : 'ClassifierVGG16'
    },
}



def get_classifier(name, config, model_config):

    if name in classifier_dict:
        return classifier_dict[name](config, model_config, **classifier_params_dict[name])
    else:
        raise Exception("No classifier named " + name)

    # pass




