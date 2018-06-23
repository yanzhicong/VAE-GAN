

import os
import sys

sys.path.append('../')


# from network.vgg import VGG16
# from network.inception_v3 import InceptionV3

# from .classifier_pixel import ClassifierPixel
# # from .classifier_simple import ClassifierSimple


# classifier_dict = {
#     # 'googlenet' : None,
#     # 'resnet-101' : None,
#     'GoogleNet' : InceptionV3,
#     'ClassifierPixel' : ClassifierPixel,
#     # 'ClassifierSimple' : ClassifierSimple
#     'VGG' : VGG16,
# }



def get_classifier(name, config, is_training):
    if name == 'ClassifierSimple':
        from .classifier_simple import ClassifierSimple
        return ClassifierSimple(config, is_training)
    else:
        raise Exception("No classifier named " + name)


