
import os
import sys


import tensorflow as tf
import tensorflow.contrib.layers as tcl


sys.path.append('../')



from network.weightsinit import get_weightsinit
from network.activation import get_activation
from network.normalization import get_normalization


class DecoderSimple(object):

	def __init__(self, config, model_config, is_training, name="DecoderSimple"):

		self.name = name
		self.training = is_training
		self.normalizer_params = {
			'decay' : 0.999,
			'center' : True,
			'scale' : False,
			'is_training' : self.training
		}

		self.config = config
		self.model_config = model_config

	def __call__(self, x, reuse=False):
		act_fn = get_activation(
					self.config.get('activation', 'relu'),
					self.config.get('activation_params', {}))

		norm_fn, norm_params = get_normalization(
					self.config.get('batch_norm', 'batch_norm'),
					self.config.get('batch_norm_params', self.normalizer_params))

		winit_fn = get_weightsinit(
					self.config.get('weightsinit', 'normal'),
					self.config.get('weightsinit_params', '0.00 0.02'))

		if 'nb_nodes' in self.config: 
			nb_nodes = self.config['nb_nodes']
		else:
			nb_nodes = [256,]

		output_act_fn = get_activation(
					self.config.get('output_activation', 'sigmoid'),
					self.config.get('output_activation_params', ''))

		output_dim = self.config['output_dims']


		with tf.variable_scope(self.name):
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False

			for ind, nodes in enumerate(nb_nodes):
				x = tcl.fully_connected(x, nodes, activation_fn=act_fn, 
							weights_initializer=winit_fn, scope='dfc%d'%(ind+1))

			x = tcl.fully_connected(x, output_dim, 
							activation_fn=output_act_fn, scope='dfc_x')

			return x

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


