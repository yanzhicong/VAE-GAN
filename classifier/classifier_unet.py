# SOFTWARE.
# ==============================================================================

import os
import sys


import tensorflow as tf
import tensorflow.contrib.layers as tcl


sys.path.append('../')


from utils.weightsinit import get_weightsinit
from utils.activation import get_activation
from utils.normalization import get_normalization

from network.basenetwork import BaseNetwork
from network.unet import UNet


class ClassifierUNet(BaseNetwork):
	def __init__(self, config, is_training):
		BaseNetwork.__init__(self, config, is_training)
		self.network = UNet(config, is_training)
		
	def __call__(self, i):
		x, end_points = self.network(i)
		return x

	def features(self, i):
		x, end_points = self.network(i)
		return x, end_points

