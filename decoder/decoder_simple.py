import os
import sys

import tensorflow as tf
import tensorflow.contrib.layers as tcl

sys.path.append('../')


from utils.weightsinit import get_weightsinit
from utils.activation import get_activation
from utils.normalization import get_normalization

from network.devgg import DEVGG
from network.base_network import BaseNetwork


class DecoderSimple(BaseNetwork):

	def __init__(self, config, is_training):
		super(DecoderSimple, self).__init__(config, is_training)
		self.name = config.get('name', 'DecoderSimple')
		self.config = config
		network_config = config.copy()
		self.network = DEVGG(network_config, is_training)

	def __call__(self, x, condition=None):
		if condition is not None:
			x = tf.concatenate([x, condition], axis=-1)
		x, end_points = self.network(x)
		
		return x

