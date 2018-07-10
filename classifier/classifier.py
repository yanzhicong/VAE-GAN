
import os
import sys

sys.path.append('../')


def get_classifier(name, config, is_training):
    if name == 'classifier':
        from .classifier_simple import ClassifierSimple
        return ClassifierSimple(config, is_training)
    else:
        raise Exception("No classifier named " + name)

