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
sys.path.append('../')

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl

from utils.weightsinit import get_weightsinit
from utils.activation import get_activation
from utils.normalization import get_normalization


# from network.vgg import VGG

from network.base_network import BaseNetwork
from network.devgg import DEVGG





def Param(name, value, trainable=True, collections=None):
	if collections is not None:


		if isinstance(collections, list):
			collections.append(tf.GraphKeys.GLOBAL_VARIABLES)
		else:
			collections = [collections, tf.GraphKeys.GLOBAL_VARIABLES]
		return tf.get_variable(name, shape=value.shape, trainable=trainable, initializer=tf.constant_initializer(value), collections=collections)
	else:
		return tf.get_variable(name, shape=value.shape, trainable=trainable, initializer=tf.constant_initializer(value))





def Batchnorm(name, axes, inputs, is_training=None, stats_iter=100, update_moving_stats=True, fused=True):
	with tf.variable_scope(name) as scope:

		if (axes == [0,1,2]) and fused==True:
			# if axes==[0,2]:
			#     inputs = tf.expand_dims(inputs, 3)
			# Old (working but pretty slow) implementation:
			##########

			# inputs = tf.transpose(inputs, [0,2,3,1])

			# mean, var = tf.nn.moments(inputs, [0,1,2], keep_dims=False)
			# offset = lib.param(name+'.offset', np.zeros(mean.get_shape()[-1], dtype='float32'))
			# scale = lib.param(name+'.scale', np.ones(var.get_shape()[-1], dtype='float32'))
			# result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-4)

			# return tf.transpose(result, [0,3,1,2])

			# New (super fast but untested) implementation:
			offset = Param('offset', np.zeros(inputs.get_shape()[-1], dtype='float32'))
			scale = Param('scale', np.ones(inputs.get_shape()[-1], dtype='float32'))

			moving_mean = Param('moving_mean', np.zeros(inputs.get_shape()[-1], dtype='float32'), trainable=False, collections=['BATCH_NORM_MOVING_VARS'])
			moving_variance = Param('moving_variance', np.ones(inputs.get_shape()[-1], dtype='float32'), trainable=False, collections=['BATCH_NORM_MOVING_VARS'])

			def _fused_batch_norm_training():
				return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-5, data_format='NHWC')
			def _fused_batch_norm_inference():
				# Version which blends in the current item's statistics
				batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
				mean, var = tf.nn.moments(inputs, [0,1,2])
				mean = ((1./batch_size)*mean) + (((batch_size-1.)/batch_size)*moving_mean)
				var = ((1./batch_size)*var) + (((batch_size-1.)/batch_size)*moving_variance)


				print('inputs : ', inputs.get_shape())
				print('mean : ', mean.get_shape())
				print('var : ', var.get_shape())
				return tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5), mean, var

				# Standard version
				# return tf.nn.fused_batch_norm(
				#     inputs,
				#     scale,
				#     offset,
				#     epsilon=1e-2, 
				#     mean=moving_mean,
				#     variance=moving_variance,
				#     is_training=False,
				#     data_format='NCHW'
				# )

			if is_training is None:
				outputs, batch_mean, batch_var = _fused_batch_norm_training()
			else:
				outputs, batch_mean, batch_var = tf.cond(is_training,
														   _fused_batch_norm_training,
														   _fused_batch_norm_inference)
				if update_moving_stats:
					def _no_updates(): 
						return outputs


					def _force_updates():
						"""Internal function forces updates moving_vars if is_training."""
						float_stats_iter = tf.cast(stats_iter, tf.float32)

						update_moving_mean = tf.assign(moving_mean, ((float_stats_iter/(float_stats_iter+1))*moving_mean) + ((1/(float_stats_iter+1))*batch_mean))
						update_moving_variance = tf.assign(moving_variance, ((float_stats_iter/(float_stats_iter+1))*moving_variance) + ((1/(float_stats_iter+1))*batch_var))

						with tf.control_dependencies([update_moving_mean, update_moving_variance]):
							return tf.identity(outputs)


					print(is_training)
					outputs = tf.cond(pred=is_training, true_fn=_force_updates, false_fn=_no_updates)

			# if axes == [0,2]:
				# return outputs[:,:,:,0] # collapse last dim
			# else:
			return outputs
		else:
			# raise Exception('old BN')
			#     # TODO we can probably use nn.fused_batch_norm here too for speedup
			print(inputs.get_shape())
			mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
			shape = mean.get_shape().as_list()
			if 0 not in axes:
				print("WARNING ({}): didn't find 0 in axes, but not using separate BN params for each item in batch".format(name))
				shape[0] = 1
			offset = Param('offset', np.zeros(shape, dtype='float32'))
			scale = Param('scale', np.ones(shape, dtype='float32'))
			result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

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

			result = tf.nn.bias_add(
				result, _biases, data_format='NHWC'
				# lib.param(
				#     name + '.b',
				#     np.zeros((output_dim,), dtype='float32')
				# )
			)

		return result




def Deconv2D(
	name, 
	input_dim, 
	output_dim, 
	filter_size, 
	inputs, 
	he_init=True,
	weightnorm=None,
	biases=True,
	gain=1.,
	mask_type=None,
	):
	"""
	inputs: tensor of shape (batch size, height, width, input_dim)
	returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
	"""
	with tf.variable_scope(name) as scope:

		if mask_type != None:
			raise Exception('Unsupported configuration')

		def uniform(stdev, size):
			return np.random.uniform(
				low=-stdev * np.sqrt(3),
				high=stdev * np.sqrt(3),
				size=size
			).astype('float32')

		stride = 2
		fan_in = input_dim * filter_size**2 / (stride**2)
		fan_out = output_dim * filter_size**2

		if he_init:
			filters_stdev = np.sqrt(4./(fan_in+fan_out))
		else: # Normalized init (Glorot & Bengio)
			filters_stdev = np.sqrt(2./(fan_in+fan_out))


		# if _weights_stdev is not None:
		# 	filter_values = uniform(
		# 		_weights_stdev,
		# 		(filter_size, filter_size, output_dim, input_dim)
		# 	)
		# else:
		filter_values = uniform(
			filters_stdev,
			(filter_size, filter_size, output_dim, input_dim)
		)

		filter_values *= gain

		filters = Param(
			'weights',
			filter_values
		)

		# if weightnorm==None:
		#     weightnorm = _default_weightnorm
		# if weightnorm:
		#     norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,3)))
		#     target_norms = lib.param(
		#         name + '.g',
		#         norm_values
		#     )
		#     with tf.variable_scope('weightnorm') as scope:
		#         norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,3]))
		#         filters = filters * tf.expand_dims(target_norms / norms, 1)

		# inputs = tf.transpose(inputs, [0,2,3,1], name='NCHW_to_NHWC')

		# input_shape = tf.shape(inputs)
		input_shape = inputs.get_shape()
		# try: # tf pre-1.0 (top) vs 1.0 (bottom)
			# output_shape = tf.pack([input_shape[0], 2*input_shape[1], 2*input_shape[2], output_dim])
		# except Exception as e:
			# output_shape = tf.stack([input_shape[0], 2*input_shape[1], 2*input_shape[2], output_dim])

		# output_shape = [2*int(input_shape[1]), 2*int(input_shape[2]), output_dim]

		# print('conv2d transpose', output_shape)
		result = tf.nn.conv2d_transpose(
			value=inputs, 
			filter=filters,
			# output_shape=output_shape, 
			strides=[1, 2, 2, 1],
			padding='SAME'
		)

		if biases:
			_biases = Param(
				'biases',
				np.zeros(output_dim, dtype='float32')
			)
			result = tf.nn.bias_add(result, _biases, data_format='NHWC')

		# result = tf.transpose(result, [0,3,1,2], name='NHWC_to_NCHW')


		return result


from network.base_network import BaseNetwork


class GeneratorCifar10(BaseNetwork):

	def __init__(self, config, is_training):

		BaseNetwork.__init__(self, config, is_training)

		self.name = config.get('name', 'GeneratorCifar10')
		self.config = config

		self.reuse=False
		self.is_training = is_training
		
		# self.normalizer_params = {
		# 	'decay' : 0.999,
		# 	'center' : True,
		# 	'scale' : False,
		# 	'is_training' : self.is_training
		# }


	def __call__(self, x):


		# act_fn = get_activation('relu')
		# norm_fn, norm_params = get_normalization('batch_norm', self.normalizer_params)
		# winit_fn = get_weightsinit('normal 0.00 0.02')
		# binit_fn = get_weightsinit('zeros')
		# output_act_fn = get_activation('sigmoid')

		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True


			DIM=128
			output = Linear('fc1', int(x.get_shape()[-1]), 4*4*4*DIM, x)
			print(output.get_shape())

			output = Batchnorm('bn1', [0], output, is_training=self.is_training)
			output = tf.nn.relu(output)
			output = tf.reshape(output, [-1, 4, 4, 4*DIM])
			print(output.get_shape())
			output = Deconv2D('deconv1', 4*DIM, 2*DIM, 5, output)
			print(output.get_shape())
			output = Batchnorm('bn2', [0,1,2], output, is_training=self.is_training)
			output = tf.nn.relu(output)
			print(output.get_shape())

			output = Deconv2D('deconv2', 2*DIM, DIM, 5, output)
			output = Batchnorm('bn3', [0,1,2], output, is_training=self.is_training)
			output = tf.nn.relu(output)
			print(output.get_shape())

			output = Deconv2D('deconv3', DIM, 3, 5, output)
			output = tf.tanh(output)

			# return tf.reshape(output, [-1, 32, 32, 3])
			return output


			# x = tcl.fully_connected(x, 512, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
			# 				weights_initializer=winit_fn, scope='fc0')


			# x = tcl.fully_connected(x, 4 * 4 * 512, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
			# 				weights_initializer=winit_fn, scope='fc1')

			# x = tf.reshape(x, [-1, 4, 4, 512])

			# x = tcl.conv2d_transpose(x, 256, 3,
			# 						stride=2, 
			# 						activation_fn=act_fn, 
			# 						normalizer_fn=norm_fn, 
			# 						normalizer_params=norm_params,
			# 						padding='SAME', weights_initializer=winit_fn, scope='deconv0')

			# # x = tcl.conv2d(x, 256, 3,
			# # 						stride=1, 
			# # 						activation_fn=act_fn, 
			# # 						normalizer_fn=norm_fn, 
			# # 						normalizer_params=norm_params,
			# # 						padding='SAME', weights_initializer=winit_fn, scope='conv0')


			# x = tcl.conv2d_transpose(x, 128, 3,
			# 						stride=2, 
			# 						activation_fn=act_fn, 
			# 						normalizer_fn=norm_fn, 
			# 						normalizer_params=norm_params,
			# 						padding='SAME', weights_initializer=winit_fn, scope='deconv1')

			# x = tcl.conv2d(x, 128, 3,
			# 						stride=1, 
			# 						activation_fn=act_fn, 
			# 						normalizer_fn=norm_fn, 
			# 						normalizer_params=norm_params,
			# 						padding='SAME', weights_initializer=winit_fn, scope='conv1')


			# x = tcl.conv2d_transpose(x, 64, 3,
			# 						stride=2, 
			# 						activation_fn=act_fn, 
			# 						normalizer_fn=norm_fn, 
			# 						normalizer_params=norm_params,
			# 						padding='SAME', weights_initializer=winit_fn, scope='deconv2')


			# x = tcl.conv2d(x, 64, 3,
			# 						stride=1, 
			# 						activation_fn=act_fn, 
			# 						normalizer_fn=norm_fn, 
			# 						normalizer_params=norm_params,
			# 						padding='SAME', weights_initializer=winit_fn, scope='conv2')
												
			# x = tcl.conv2d_transpose(x, 3, 1,
			# 						stride=1, 
			# 						activation_fn=output_act_fn, 
			# 						normalizer_fn=None, 
			# 						normalizer_params=None,
			# 						padding='SAME', weights_initializer=winit_fn, scope='deconv3')

		return x


	# @property
	# def vars(self):
	# 	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


