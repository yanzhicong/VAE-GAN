# -*- coding: utf-8 -*-
# MIT License
# 
# Copyright (c) 2018 ZhicongYan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import os
import sys


import tensorflow as tf
import tensorflow.contrib.layers as tcl


sys.path.append('../')


import numpy as np
from utils.weightsinit import get_weightsinit
from utils.activation import get_activation
from utils.normalization import get_normalization


from network.vgg import VGG
from network.basenetwork import BaseNetwork




# _params = {}
# _param_aliases = {}
# def param(name, *args, **kwargs):
#     """
#     A wrapper for `tf.Variable` which enables parameter sharing in models.
	
#     Creates and returns theano shared variables similarly to `tf.Variable`, 
#     except if you try to create a param with the same name as a 
#     previously-created one, `param(...)` will just return the old one instead of 
#     making a new one.

#     This constructor also adds a `param` attribute to the shared variables it 
#     creates, so that you can easily search a graph for all params.
#     """

#     if name not in _params:
#         kwargs['name'] = name
#         param = tf.Variable(*args, **kwargs)
#         param.param = True
#         _params[name] = param
#     result = _params[name]
#     i = 0
#     while result in _param_aliases:
#         # print 'following alias {}: {} to {}'.format(i, result, _param_aliases[result])
#         i += 1
#         result = _param_aliases[result]
#     return result


def Param(name, value, trainable=True):
	return tf.get_variable(name, shape=value.shape, trainable=trainable, initializer=tf.constant_initializer(value))



def Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1, weightnorm=None, biases=True, gain=1.):
	"""
	inputs: tensor of shape (batch size, num channels, height, width)
	mask_type: one of None, 'a', 'b'

	returns: tensor of shape (batch size, num channels, height, width)
	"""
	with tf.variable_scope(name) as scope:

		if mask_type is not None:
			mask_type, mask_n_channels = mask_type

			mask = np.ones(
				(filter_size, filter_size, input_dim, output_dim), 
				dtype='float32'
			)
			center = filter_size // 2

			# Mask out future locations
			# filter shape is (height, width, input channels, output channels)
			mask[center+1:, :, :, :] = 0.
			mask[center, center+1:, :, :] = 0.

			# Mask out future channels
			for i in xrange(mask_n_channels):
				for j in xrange(mask_n_channels):
					if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
						mask[
							center,
							center,
							i::mask_n_channels,
							j::mask_n_channels
						] = 0.


		def uniform(stdev, size):
			return np.random.uniform(
				low=-stdev * np.sqrt(3),
				high=stdev * np.sqrt(3),
				size=size
			).astype('float32')

		fan_in = input_dim * filter_size**2
		fan_out = output_dim * filter_size**2 / (stride**2)

		if mask_type is not None: # only approximately correct
			fan_in /= 2.
			fan_out /= 2.

		if he_init:
			filters_stdev = np.sqrt(4./(fan_in+fan_out))
		else: # Normalized init (Glorot & Bengio)
			filters_stdev = np.sqrt(2./(fan_in+fan_out))

		# if _weights_stdev is not None:
		#     filter_values = uniform(
		#         _weights_stdev,
		#         (filter_size, filter_size, input_dim, output_dim)
		#     )
		# else:
		filter_values = uniform(
			filters_stdev,
			(filter_size, filter_size, input_dim, output_dim)
		)

		# print "WARNING IGNORING GAIN"
		filter_values *= gain

		# scale = tf.get_variable('weights', shape=filter_values.shape, trainable=trainable, initializer=tf.constant_initializer(filter_values))

		filters = Param('weights', filter_values, trainable=True)

		# if weightnorm==None:
		# #     weightnorm = _default_weightnorm
		# if weightnorm:
		#     norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2)))

		#     target_norms = lib.param(
		#         name + '.g',
		#         norm_values
		#     )
		#     with tf.variable_scope('weightnorm') as scope:
		#         norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,2]))
		#         filters = filters * (target_norms / norms)

		if mask_type is not None:
			with tf.variable_scope('filter_mask'):
				filters = filters * mask

		result = tf.nn.conv2d(
			input=inputs, 
			filter=filters, 
			strides=[1, stride, stride, 1],
			padding='SAME',
			data_format='NHWC'
		)

		if biases:
			# _biases = lib.param(
			#     name+'.Biases',
			#     np.zeros(output_dim, dtype='float32')
			# )

			_biases = Param('biases', np.zeros(output_dim, dtype='float32'))

			result = tf.nn.bias_add(result, _biases, data_format='NHWC')
		return result




def Linear(
		name, 
		input_dim, 
		output_dim, 
		inputs,
		biases=True,
		initialization=None,
		weightnorm=None,
		gain=1.
		):
	"""
	initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
	"""
	with tf.variable_scope(name) as scope:

		def uniform(stdev, size):
			# if _weights_stdev is not None:
				# stdev = _weights_stdev
			return np.random.uniform(
				low=-stdev * np.sqrt(3),
				high=stdev * np.sqrt(3),
				size=size
			).astype('float32')

		if initialization == 'lecun':# and input_dim != output_dim):
			# disabling orth. init for now because it's too slow
			weight_values = uniform(
				np.sqrt(1./input_dim),
				(input_dim, output_dim)
			)

		elif initialization == 'glorot' or (initialization == None):

			weight_values = uniform(
				np.sqrt(2./(input_dim+output_dim)),
				(input_dim, output_dim)
			)

		elif initialization == 'he':

			weight_values = uniform(
				np.sqrt(2./input_dim),
				(input_dim, output_dim)
			)

		elif initialization == 'glorot_he':

			weight_values = uniform(
				np.sqrt(4./(input_dim+output_dim)),
				(input_dim, output_dim)
			)

		elif initialization == 'orthogonal' or \
			(initialization == None and input_dim == output_dim):
			
			# From lasagne
			def sample(shape):
				if len(shape) < 2:
					raise RuntimeError("Only shapes of length 2 or more are "
									   "supported.")
				flat_shape = (shape[0], np.prod(shape[1:]))
				 # TODO: why normal and not uniform?
				a = np.random.normal(0.0, 1.0, flat_shape)
				u, _, v = np.linalg.svd(a, full_matrices=False)
				# pick the one with the correct shape
				q = u if u.shape == flat_shape else v
				q = q.reshape(shape)
				return q.astype('float32')
			weight_values = sample((input_dim, output_dim))
		
		elif initialization[0] == 'uniform':
		
			weight_values = np.random.uniform(
				low=-initialization[1],
				high=initialization[1],
				size=(input_dim, output_dim)
			).astype('float32')

		else:
			raise Exception('Invalid initialization!')

		weight_values *= gain

		weight = Param('weights', weight_values)

		# weight = lib.param(
		#     name + '.W',
		#     weight_values
		# )

		# if weightnorm==None:
		#     weightnorm = _default_weightnorm
		# if weightnorm:
		#     norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
		#     # norm_values = np.linalg.norm(weight_values, axis=0)

		#     target_norms = lib.param(
		#         name + '.g',
		#         norm_values
		#     )

		#     with tf.variable_scope('weightnorm') as scope:
		#         norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
		#         weight = weight * (target_norms / norms)

		# if 'Discriminator' in name:
		#     print "WARNING weight constraint on {}".format(name)
		#     weight = tf.nn.softsign(10.*weight)*.1
		if inputs.get_shape().ndims == 2:
			result = tf.matmul(inputs, weight)
		else:
			reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
			result = tf.matmul(reshaped_inputs, weight)
			result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

		if biases:

			_biases = Param('biases', np.zeros((output_dim, ), dtype='float32'))

			result = tf.nn.bias_add(result, _biases)

		return result




def LeakyReLU(x, alpha=0.2):
	return tf.maximum(alpha*x, x)



class DiscriminatorCifar10(BaseNetwork):

	def __init__(self, config, is_training):

		BaseNetwork.__init__(self, config, is_training)
		# self.name = config.get('name', 'DiscriminatorCifar10')
		# self.config = config
		# self.reuse = False
		# self.is_training = is_training
		
		# self.normalizer_params = {
		# 	'decay' : 0.999,
		# 	'center' : True,
		# 	'scale' : False,
		# 	'is_training' : self.is_training
		# }


	def __call__(self, x):
	
		# act_fn = get_activation('lrelu 0.2')
		norm_fn = get_normalization('batch_norm')
		# winit_fn = get_weightsinit('normal 0.00 0.02')
		# binit_fn = get_weightsinit('zeros')


		# output_dims = self.config.get('output_dims', 100)

		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

				DIM=128
				# output = tf.reshape(inputs, [-1, 32, 32, 3])


				print('x', x.get_shape())

				output = Conv2D('conv1', 3, DIM, 5, x, stride=2)

				print('output', output.get_shape())

				output = LeakyReLU(output)

				output = Conv2D('conv2', DIM, 2*DIM, 5, output, stride=2)


				print('output', output.get_shape())
				output = LeakyReLU(output)

				output = Conv2D('conv3', 2*DIM, 4*DIM, 5, output, stride=2)
				# if MODE != 'wgan-gp':
				# output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
				output = LeakyReLU(output)

				output = tf.reshape(output, [-1, 4*4*4*DIM])
				output = Linear('fc1', 4*4*4*DIM, 1, output)

				# return tf.reshape(output, [-1, 1])

				return output

		return x

	def features(self, i, condition=None):
		return NotImplementedError


	# @property
	# def vars(self):
	# 	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


