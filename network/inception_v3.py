

import tensorflow as tf
import tensorflow.contrib.layers as tcl

from weightsinit import get_weightsinit
from activation import get_activation
from normalization import get_normalization



from inception_block import inception_v3_figure4
from inception_block import inception_v3_figure5
from inception_block import inception_v3_figure6
from inception_block import inception_v3_figure7


class InceptionV3(object):
	def __init__(self, config, model_config, name="InceptionV3"):

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
			norm_params = self.normalizer_params

		if 'weightsinit' in self.config:
			winit_fn = get_weightsinit(self.config['weightsinit'])
		elif 'weightsinit' in self.model_config:
			winit_fn = get_weightsinit(self.model_config['weightsinit'])
		else:
			winit_fn = tf.random_normal_initializer(0, 0.02)


		if 'nb_filters' in self.config: 
			filters = int(self.config['nb_filters'])
		else:
			filters = 32


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

			end_points = {}

			# x : 299 * 299 * 3
			x = tcl.conv2d(i, filters, 3,
							stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='VALID', weights_initializer=winit_fn, scope='conv1')
			end_points['conv1'] = x
			
			# x : 149 * 149 * 32 
			x = tcl.conv2d(x, filters, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='VALID', weights_initializer=winit_fn, scope='conv2')
			end_points['conv2'] = x

			# x : 147 * 147 * 32
			x = tcl.conv2d(x, 2*filters, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv3')
			end_points['conv3'] = x


			# x : 147 * 147 * 64
			x = tcl.max_pool2d(x, 3, stride=2, 
							padding='VALID', scope='pool1')
			end_points['pool1'] = x

			# x : 73 * 73 * 64
			x = tcl.conv2d(x, 80, 3, 
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='VALID', weights_initializer=winit_fn, scope='conv4')
			end_points['conv4'] = x


			# x : 71 * 71 * 80
			x = tcl.conv2d(x, 192, 3, 
							stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='VALID', weights_initializer=winit_fn, scope='conv5')
			end_points['conv5'] = x


			# x : 35 * 35 * 192
			x = tcl.conv2d(x, 288, 3, 
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv6')
			end_points['conv6'] = x

			# x : 35 * 35 * 288
			x, end_points = inception_v3_figure5('inception1a', x, end_points, 
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure5('inception1b', x, end_points, 
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure5('inception1c', x, end_points, 
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn, downsample=True)


			# x : 17 * 17 * 768
			x, end_points = inception_v3_figure6('inception2a', x, end_points, n=7,
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure6('inception2b', x, end_points, n=7,
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure6('inception2c', x, end_points, n=7,
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure6('inception2d', x, end_points, n=7,
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure6('inception2e', x, end_points, n=7,
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn, downsample=True)
  

			# x : 8 * 8 * 1280
			x, end_points = inception_v3_figure7('inception3a', x, end_points, 
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure7('inception3b', x, end_points, 
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			# x, end_points = inception_v3_figure7('inception3c', x, end_points, 
			# 			act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn, downsample=True)
  
			return x, end_points

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)




if __name__ == '__main__':
	config = {
		'output_classes' : 10
	}
	model_config = {
		'is_training' : True
	}

	inception_model = InceptionV3(config, model_config)


	x = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input')

	y, end_points = inception_model(x)


	for name, value in end_points.items():
		print(name, '  --> ', value.get_shape())


