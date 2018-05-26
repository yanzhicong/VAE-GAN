
import os
import sys


import tensorflow as tf
import tensorflow.contrib.layers as tcl


sys.path.append('../')



from network.weightsinit import get_weightsinit
from network.activation import get_activation
from network.normalization import get_normalization



class DecoderSimple(object):

	def __init__(self, config, model_config, name="DecoderSimple"):

		self.name = name
		self.training = model_config["is_training"]
		self.normalizer_params = {
			'decay' : 0.999,
			'center' : True,
			'scale' : False,
			'is_training' : self.training
		}

		self.config = config
		self.model_config = model_config


	def __call__(self, i, reuse=False):

		if 'activation' in self.config:
			act_fn = get_activation(self.config['activation'], self.config['activation_params'])
		elif 'activation' in self.model_config:
			act_fn = get_activation(self.model_config['activation'], self.model_config['activation_params'])
		else:
			act_fn = get_activation('relu')


		if 'weightsinit' in self.config:
			winit_fn = get_weightsinit(self.config['weightsinit'], self.config['weightsinit_params'])
		elif 'weightsinit' in self.model_config:
			winit_fn = get_weightsinit(self.model_config['weightsinit'], self.config['weightsinit_params'])
		else:
			winit_fn = tf.random_normal_initializer(0, 0.02)

		if 'nb_nodes' in self.config: 
			nb_nodes = self.config['nb_nodes']
		else:
			nb_nodes = [64, 256]

		if 'out_activation' in self.config:
			out_act_fn = get_activation(self.config['out_activation'], self.config['out_activation_params'])
		else:
			out_act_fn = get_activation('sigmoid', None)

		output_dim = self.config['output_dim']

		with tf.variable_scope(self.name):
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False


			x = i
            for ind, nodes in enumerate(nb_nodes):
                x = tcl.fully_connected(x, nodes, activation_fn=act_fn, 
                            weighits_initializer=winit_fn, scope='dfc%d'(ind+1))

            x = tcl.fully_connected(x, output_dims, 
							activation_fn=act_fn, scope='dfc_x')

			return x

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


