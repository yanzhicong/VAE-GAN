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




import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.python.training import moving_averages


def batch_norm(inputs,
					decay=0.99,
					center=True,
					scale=False,
					epsilon=0.001,
					is_training=True,
					moving_vars_collection=None,
					trainable=True,
					zero_debias_moving_mean=False):

	if not isinstance(moving_vars_collection, list):
		moving_vars_collection = list([moving_vars_collection])
	moving_vars_collection.append(tf.GraphKeys.GLOBAL_VARIABLES)

	with tf.variable_scope('BatchNorm') as sc:
		inputs = tf.convert_to_tensor(inputs)
		original_shape = inputs.get_shape()
		original_inputs = inputs
		original_rank = original_shape.ndims
		if original_rank is None:
			raise ValueError('Inputs %s has undefined rank' % inputs.name)
		elif original_rank not in [2, 4]:
			raise ValueError('Inputs %s has unsupported rank.'
											 ' Expected 2 or 4 but got %d' % (inputs.name,
																							original_rank))
		if original_rank == 2:
			channels = inputs.get_shape()[-1].value
			if channels is None:
				raise ValueError('`C` dimension must be known but is None')
			new_shape = [-1, 1, 1, channels]
			inputs = tf.reshape(inputs, new_shape)
		inputs_shape = inputs.get_shape()
		params_shape = inputs_shape[-1:]
		if not params_shape.is_fully_defined():
			raise ValueError('Inputs %s has undefined `C` dimension %s.' %
											 (inputs.name, params_shape))

		variable_dtype = tf.float32

		if center:
			beta = tf.get_variable(
					'beta',
					shape=params_shape,
					dtype=variable_dtype,
					initializer=tf.zeros_initializer(),
					trainable=trainable)
		else:
			beta = tf.constant(0.0, dtype=variable_dtype, shape=params_shape)

		if scale:
			gamma = tf.get_variable(
					'gamma',
					shape=params_shape,
					dtype=variable_dtype,
					initializer=tf.ones_initializer(),
					trainable=trainable)
		else:
			gamma = tf.constant(1.0, dtype=variable_dtype, shape=params_shape)

		moving_mean = tf.get_variable(
				'moving_mean',
				shape=params_shape,
				dtype=variable_dtype,
				collections=moving_vars_collection,
				initializer=tf.zeros_initializer(),
				trainable=False)

		moving_variance = tf.get_variable(
				'moving_variance',
				shape=params_shape,
				dtype=variable_dtype,
				collections=moving_vars_collection,
				initializer=tf.ones_initializer(),
				trainable=False,)

		def batch_norm_training():
			mean, variance = tf.nn.moments(inputs, [0,1,2])
			update_moving_mean = moving_averages.assign_moving_average(
					moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
			update_moving_variance = moving_averages.assign_moving_average(
					moving_variance, variance, decay, zero_debias=zero_debias_moving_mean)
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				outputs = tf.nn.batch_normalization(inputs, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=epsilon)
				return outputs

		def batch_norm_inference():
			outputs = tf.nn.batch_normalization(inputs, mean=moving_mean, 
										variance=moving_variance, offset=beta, scale=gamma, variance_epsilon=epsilon)
			return outputs

		outputs = tf.cond(
				pred=is_training, true_fn=batch_norm_training, false_fn=batch_norm_inference)

		outputs.set_shape(inputs_shape)
		if original_shape.ndims == 2:
			outputs = tf.reshape(outputs, tf.shape(original_inputs))
		return outputs




def fused_batch_norm(inputs, 
					decay=0.99,
					epsilon=1e-5,
					is_training=True,
					moving_vars_collection=None,
					trainable=True):
	'''
		fused batch normalization
	'''
	if not isinstance(moving_vars_collection, list):
		moving_vars_collection = list([moving_vars_collection])
	moving_vars_collection.append(tf.GraphKeys.GLOBAL_VARIABLES)

	inputs_shape = inputs.get_shape()
	inputs_rank = inputs_shape.ndims

	if inputs_rank == 4:
		params_shape = inputs_shape[-1:]

		with tf.variable_scope('BatchNorm') as sc:
			offset = tf.get_variable('offset', shape=params_shape, trainable=trainable, initializer=tf.zeros_initializer())
			scale = tf.get_variable('scale', shape=params_shape, trainable=trainable, initializer=tf.ones_initializer())
			moving_mean = tf.get_variable('moving_mean', shape=params_shape, trainable=False, initializer=tf.zeros_initializer(),
												collections=moving_vars_collection)
			moving_variance = tf.get_variable('moving_variance', shape=params_shape, trainable=False, initializer=tf.ones_initializer(),
												collections=moving_vars_collection)

		def fused_batch_norm_training():
			outputs, batch_mean, batch_variance = tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=epsilon, data_format='NHWC')
			update_moving_mean = tf.assign_add(moving_mean, (1.0-decay)*(batch_mean-moving_mean))
			update_moving_variance = tf.assign_add(moving_variance, (1.0-decay)*(batch_variance-moving_variance))
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				return tf.identity(outputs)
			return outputs

		def fused_batch_norm_inference():
			outputs = tf.nn.batch_normalization(inputs, moving_mean, moving_variance, offset, scale, epsilon)
			return outputs
		outputs = tf.cond(pred=is_training,
								true_fn=fused_batch_norm_training,
								false_fn=fused_batch_norm_inference)
		return outputs

	elif inputs_rank == 2:
		with tf.variable_scope('BatchNorm') as sc:
			mean, variance = tf.nn.moments(inputs, [0], keep_dims=True)
			shape = mean.get_shape().as_list()
			offset = tf.get_variable('offset', shape=shape, trainable=trainable, initializer=tf.zeros_initializer()) 
			scale = tf.get_variable('scale', shape=shape, trainable=trainable, initializer=tf.ones_initializer())
			result = tf.nn.batch_normalization(inputs, mean, variance, offset, scale, 1e-5)
			return result




def fused_batch_norm2(inputs, 
					decay=0.99,
					epsilon=1e-5,
					is_training=True,
					moving_vars_collection=None,
					trainable=True):

	if not isinstance(moving_vars_collection, list):
		moving_vars_collection = list([moving_vars_collection])
	moving_vars_collection.append(tf.GraphKeys.GLOBAL_VARIABLES)

	inputs_shape = inputs.get_shape()
	inputs_rank = inputs_shape.ndims

	if inputs_rank == 4:
		params_shape = inputs_shape[-1:]

		with tf.variable_scope('BatchNorm') as sc:
			offset = tf.get_variable('offset', shape=params_shape, trainable=trainable, initializer=tf.zeros_initializer())
			scale = tf.get_variable('scale', shape=params_shape, trainable=trainable, initializer=tf.ones_initializer())
			moving_mean = tf.get_variable('moving_mean', shape=params_shape, trainable=False, initializer=tf.zeros_initializer(),
												collections=moving_vars_collection)
			moving_variance = tf.get_variable('moving_variance', shape=params_shape, trainable=False, initializer=tf.ones_initializer(),
												collections=moving_vars_collection)

		def fused_batch_norm_training():
			return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=epsilon, data_format='NHWC')

		def fused_batch_norm_inference():
			batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
			mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])
			mean = ( (1.0/batch_size) * moving_mean + ((batch_size-1.0)/batch_size) * mean )
			variance = ( (1.0/batch_size) * moving_variance + ((batch_size-1.0)/batch_size) * variance )
			return tf.nn.batch_normalization(inputs, mean, variance, offset, scale, epsilon), mean, variance

		outputs, batch_mean, batch_var = tf.cond(pred=is_training,
												true_fn=fused_batch_norm_training,
												false_fn=fused_batch_norm_inference)

		def no_updates():
			return outputs
		def force_updates():
			update_moving_mean = tf.assign_add(moving_mean, (1.0-decay)*(batch_mean-moving_mean))
			update_moving_variance = tf.assign_add(moving_variance, (1.0-decay)*(batch_var-moving_variance))
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				return tf.identity(outputs)
		outputs = tf.cond(pred=is_training, true_fn=force_updates, false_fn=no_updates)

		return outputs


	elif inputs_rank == 2:
		with tf.variable_scope('BatchNorm') as sc:
			mean, variance = tf.nn.moments(inputs, [0], keep_dims=True)
			shape = mean.get_shape().as_list()
			offset = tf.get_variable('offset', shape=shape, trainable=trainable, initializer=tf.zeros_initializer()) 
			scale = tf.get_variable('scale', shape=shape, trainable=trainable, initializer=tf.ones_initializer())
			result = tf.nn.batch_normalization(inputs, mean, variance, offset, scale, 1e-5)
			return result



def fused_batch_norm3(inputs, 
					decay=0.99,
					epsilon=1e-5,
					is_training=True,
					moving_vars_collection=None,
					trainable=True):

	if not isinstance(moving_vars_collection, list):
		moving_vars_collection = list([moving_vars_collection])
	moving_vars_collection.append(tf.GraphKeys.GLOBAL_VARIABLES)

	inputs_shape = inputs.get_shape()
	inputs_rank = inputs_shape.ndims

	if inputs_rank == 4:
		params_shape = inputs_shape[-1:]

		with tf.variable_scope('BatchNorm') as sc:
			offset = tf.get_variable('offset', shape=params_shape, trainable=trainable, initializer=tf.zeros_initializer())
			scale = tf.get_variable('scale', shape=params_shape, trainable=trainable, initializer=tf.ones_initializer())
			moving_mean = tf.get_variable('moving_mean', shape=params_shape, trainable=False, initializer=tf.zeros_initializer(),
												collections=moving_vars_collection)
			moving_variance = tf.get_variable('moving_variance', shape=params_shape, trainable=False, initializer=tf.ones_initializer(),
												collections=moving_vars_collection)

		def fused_batch_norm_training():
			return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=epsilon, data_format='NHWC')

		def fused_batch_norm_inference():
			return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=epsilon, data_format='NHWC')

		outputs, batch_mean, batch_var = tf.cond(pred=is_training,
												true_fn=fused_batch_norm_training,
												false_fn=fused_batch_norm_inference)

		def no_updates():
			return outputs
		def force_updates():
			update_moving_mean = tf.assign_add(moving_mean, (1.0-decay)*(batch_mean-moving_mean))
			update_moving_variance = tf.assign_add(moving_variance, (1.0-decay)*(batch_var-moving_variance))
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				return tf.identity(outputs)
		outputs = tf.cond(pred=is_training, true_fn=force_updates, false_fn=no_updates)
		return outputs

	elif inputs_rank == 2:
		with tf.variable_scope('BatchNorm') as sc:
			mean, variance = tf.nn.moments(inputs, [0], keep_dims=True)
			shape = mean.get_shape().as_list()
			offset = tf.get_variable('offset', shape=shape, trainable=trainable, initializer=tf.zeros_initializer()) 
			scale = tf.get_variable('scale', shape=shape, trainable=trainable, initializer=tf.ones_initializer())
			result = tf.nn.batch_normalization(inputs, mean, variance, offset, scale, 1e-5)
			return result



def get_normalization(name):
	if name == 'batch_norm':
		return batch_norm
	elif name == 'fused_batch_norm':
		return fused_batch_norm
	elif name == 'fused_batch_norm2':
		return fused_batch_norm2
	elif name == 'fused_batch_norm3':
		return fused_batch_norm3
	elif name == 'none':
		return None
	else:
		raise Exception("None normalization named " + name)


