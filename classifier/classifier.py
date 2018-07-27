
import os
import sys

sys.path.append('../')


def get_classifier(name, config, is_training):
    if name == 'classifier' or name == 'simple classifier' or name == 'ClassifierSimple':
        from .classifier_simple import ClassifierSimple
        return ClassifierSimple(config, is_training)
    elif name == 'classifier_unet' or name == 'unet classifier':
   		from .classifier_unet import ClassifierUNet
   		return ClassifierUNet(config, is_training)
    else:
        raise Exception("No classifier named " + name)

