import tensorflow as tf
import tensorflow.contrib.layers as tcl




from weightsinit import get_weightsinit
from activation import get_activation
from normalization import get_normalization


class VGG16(object):

	def __init__(self, config, model_config, name="VGG16"):

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
			act_fn = get_activation(self.config['activation'])
		elif 'activation' in self.model_config:
			act_fn = get_activation(self.model_config['activation'])
		else:
			act_fn = get_activation('relu')

		if 'batch_norm' in self.config:
			norm_fn, norm_params = get_normalization(self.config['batch_norm'])
		elif 'batch_norm' in self.model_config:
			norm_fn, norm_params = get_normalization(self.model_config['batch_norm'])
		else:
			norm_fn = tcl.batch_norm

		if 'weightsinit' in self.config:
			winit_fn = get_weightsinit(self.config['weightsinit'])
		elif 'weightsinit' in self.model_config:
			winit_fn = get_weightsinit(self.model_config['weightsinit'])
		else:
			winit_fn = tf.random_normal_initializer(0, 0.02)


		if 'nb_filters' in self.config: 
			filters = int(self.config['nb_filters'])
		else:
			filters = 64


		if 'no_maxpooling' in self.config:
			no_maxpooling = self.config['no_maxpooling']
		else:
			no_maxpooling = False


		if 'including_top' in self.config:
			including_top = self.config['including_top']
			including_top_params = self.config['including_top_params']
		else:
			including_top = True
			including_top_params = [1024, 1024]


		if 'out_activation' in self.config:
			out_act_fn = get_activation(self.config['out_activation'])
		else:
			out_act_fn = None

		output_classes = self.config['output_classes']


		with tf.variable_scope(self.name):
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False



			x = tcl.conv2d(i, filters, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv1_0')
			x = tcl.conv2d(x, filters, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv1_1')

			if no_maxpooling:
				x = tcl.conv2d(x, filters*2, 3,
								stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
								padding='SAME', weights_initializer=winit_fn, scope='conv2_0')               
			else:
				x = tf.nn.maxpooling(x, ksize=2, strides=2, padding='SAME')
				x = tcl.conv2d(x, filters*2, 3,
								stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
								padding='SAME', weights_initializer=winit_fn, scope='conv2_0')
			x = tcl.conv2d(x, filters*2, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv2_1')


			if no_maxpooling:
				x = tcl.conv2d(x, filters*4, 3,
								stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
								padding='SAME', weights_initializer=winit_fn, scope='conv3_0')               
			else:
				x = tf.nn.maxpooling(x, ksize=2, strides=2, padding='SAME')
				x = tcl.conv2d(x, filters*4, 3,
								stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
								padding='SAME', weights_initializer=winit_fn, scope='conv3_0')
			x = tcl.conv2d(x, filters*4, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv3_1')

			x = tcl.conv2d(x, filters*4, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv3_2')


			if no_maxpooling:
				x = tcl.conv2d(x, filters*8, 3,
								stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
								padding='SAME', weights_initializer=winit_fn, scope='conv4_0')               
			else:
				x = tf.nn.maxpooling(x, ksize=2, strides=2, padding='SAME')
				x = tcl.conv2d(x, filters*8, 3,
								stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
								padding='SAME', weights_initializer=winit_fn, scope='conv4_0')
			x = tcl.conv2d(x, filters*8, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv4_1')

			x = tcl.conv2d(x, filters*8, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv4_2')


			if including_top: 
				x = tf.flatten(x)
				
				for ind, nb_nodes in enumerate(including_top_params):
					x = tcl.fully_connected(x, nb_nodes, avtivation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							weights_initializer=winit_fn, scope='fc%d'%ind)

				x = tcl.fully_connected(x, output_classes, activation_fn=out_act_fn, weights_initialzr=winit_fn, scope='fc_out')

			else : 
				
				x = tcl.conv2d(x, output_classes, 1, 
							stride=1, activation_fn=out_act_fn, padding='SAME', weights_initializer=winit_fn, scope='conv_out')

			return x

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)








