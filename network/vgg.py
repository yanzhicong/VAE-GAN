import tensorflow as tf
import tensorflow.contrib.layers as tcl


from .weightsinit import get_weightsinit
from .activation import get_activation
from .normalization import get_normalization


class VGG16(object):

	def __init__(self, config, model_config, is_training, name="VGG16"):

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

		# convolution structure parameters
		nb_blocks = int(self.config.get('nb_conv_blocks', 5))
		nb_filters = self.config.get('nb_filters', [64, 128, 256, 512, 512])
		nb_layers = self.config.get('nb_layers', [2, 2, 3, 3, 3])
		ksize = self.config.get('ksize', [3, 3, 3, 3, 3])

		no_maxpooling = self.config.get('no_maxpooling', False)

		# fully connected parameters
		including_top = self.config.get('including_top', True)
		nb_fc_nodes = self.config.get('nb_fc_nodes', [1024, 1024])

		# output stage parameters
		output_dims = self.config.get('output_dims', 0)  # zero for no output layer
		output_act_fn = get_activation(
					self.config.get('output_activation', 'none'),
					self.config.get('output_activation_params', ''))


		with tf.variable_scope(self.name):
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False

			endpoints = {}

			# construct convolution layers
			for block_ind in range(nb_blocks):
				for layer_ind in range(nb_layers[block_ind]):

					if layer_ind == nb_layers[block_ind]-1:
						if no_maxpooling:
							x = tcl.conv2d(x, nb_filters[block_ind], 3,
									stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
									padding='SAME', weights_initializer=winit_fn, scope='conv%d_%d'%(block_ind+1, layer_ind))
						else:
							x = tcl.conv2d(x, nb_filters[block_ind], 3,
									stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
									padding='SAME', weights_initializer=winit_fn, scope='conv%d_%d'%(block_ind+1, layer_ind))
							x = tcl.max_pool2d(x, 2, stride=2, padding='SAME')							
					else:
						x = tcl.conv2d(x, nb_filters[block_ind], 3,
								stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
								padding='SAME', weights_initializer=winit_fn, scope='conv%d_%d'%(block_ind+1, layer_ind))
					endpoints['conv%d_%d'%(block_ind+1, layer_ind)] = x

			# construct top fully connected layer
			if including_top: 
				x = tcl.flatten(x)
				for ind, nb_nodes in enumerate(nb_fc_nodes):
					x = tcl.fully_connected(x, nb_nodes, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							weights_initializer=winit_fn, scope='fc%d'%ind)
					endpoints['fc%d'%ind] = x

				if output_dims != 0:
					x = tcl.fully_connected(x, output_dims, activation_fn=output_act_fn, weights_initialzr=winit_fn, scope='fc_out')
					endpoints['fc_out'] = x

			# else construct a convolution layer for output
			elif output_dims != 0:
				x = tcl.conv2d(x, output_dims, 1, 
							stride=1, activation_fn=output_act_fn, padding='SAME', weights_initializer=winit_fn, scope='conv_out')
				endpoints['conv_out'] = x

			return x, endpoints

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)








