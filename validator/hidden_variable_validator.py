import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm







class HiddenVariableValidator(object):
    def __init__(self, config):

        self.z_dim = config['z_dim']

        self.generate_method = config.get('generate_method', 'normppf')
        self.generate_method_params = config.get('generate_method_params', '')

        self.nb_samples = config.get('num_samples', 1000)


    def validate(self, model, dataset, step):

        

        pass


    




