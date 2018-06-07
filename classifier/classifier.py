

import os
import sys

sys.path.append('../')


from network.vgg import VGG16
from network.inception_v3 import InceptionV3

from .classifier_pixel import ClassifierPixel
# from .classifier_simple import ClassifierSimple


classifier_dict = {
    # 'googlenet' : None,
    # 'resnet-101' : None,
    'GoogleNet' : InceptionV3,
    'ClassifierPixel' : ClassifierPixel,
    # 'ClassifierSimple' : ClassifierSimple
    'VGG' : VGG16,
}



def get_classifier(name, config, model_config, is_training, **kwargs):
    if name in classifier_dict:
        return classifier_dict[name](config, model_config, is_training, **kwargs)
    else:
        raise Exception("No classifier named " + name)


