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

sys.path.append('./')
sys.path.append('../')

import tensorflow as tf
from tensorflow import layers as tl
import tensorflow.contrib.layers as tcl
import numpy as np

from utils.weightsinit import get_weightsinit
from utils.activation import get_activation
from utils.normalization import get_normalization

class BaseNetwork(object):
	def __init__(self, config, is_training):
		assert('name' in config)
		
		self.name = config['name']
		self.is_training = is_training
		self.moving_variables_collection = 'BATCH_NORM_MOVING_VARS'
		self.using_tcl_library = config.get('use tcl library', False)

		self.norm_params = {
			'is_training' : self.is_training,
			'moving_vars_collection' : self.moving_variables_collection
		}
		self.config = config

		# when first applying the network to input tensor, the reuse is false
		self.reuse = False
		self.end_points = {}

		act_fn = self.config.get('activation', 'relu')
		norm_fn = self.config.get('normalization', 'batch_norm')
		has_bias = self.config.get('has bias', True)
		conv_has_bias = self.config.get('conv has bias', has_bias)
		fc_has_bias = self.config.get('fc has bias', has_bias)
		out_has_bias = self.config.get('out has bias', has_bias)
		norm_params = self.norm_params.copy()
		norm_params.update(self.config.get('normalization params', {}))
		winit_fn = self.config.get('weightsinit', 'xavier')
		binit_fn = self.config.get('biasesinit', 'zeros')
		padding = self.config.get('padding', 'SAME')
		output_act_fn = self.config.get('output_activation', 'none')

		self.conv_args = {
			'norm_fn':norm_fn,
			'norm_params':norm_params,
			'act_fn':act_fn,
			'winit_fn':winit_fn,
			'binit_fn':binit_fn,
			'padding':padding,
			'has_bias':conv_has_bias,
		}

		self.fc_args = {
			'norm_fn':norm_fn,
			'norm_params':norm_params,
			'act_fn':act_fn,
			'winit_fn':winit_fn,
			'binit_fn':binit_fn,
			'has_bias':fc_has_bias,
		}

		self.deconv_args = {
			'norm_fn':norm_fn,
			'norm_params':norm_params,
			'act_fn':act_fn,
			'winit_fn':winit_fn,
			'binit_fn':binit_fn,
			'padding':padding,
			'has_bias' : conv_has_bias,
		}

		self.out_conv_args = {
			'act_fn':output_act_fn,
			'winit_fn':winit_fn,
			'binit_fn':binit_fn,
			'padding':padding,
			'has_bias' : out_has_bias,
		}

		self.out_fc_args = {
			'act_fn':output_act_fn,
			'winit_fn':winit_fn,
			'binit_fn':binit_fn,
			'has_bias': out_has_bias,
		}

	def uniform_initializer(self, stdev):
		return tf.random_uniform_initializer(-stdev*np.sqrt(3), stdev*np.sqrt(3))

	def conv2d(self, name, x, nb_filters, ksize, stride=1, *,
				norm_fn='none', norm_params=None, act_fn='none', winit_fn='xavier', binit_fn='zeros', padding='SAME', has_bias=True,
				disp=True, collect_end_points=True):

		if callable(act_fn):
			_act_fn = 'func'
			l_act_fn = act_fn
		else:
			_act_fn = self.config.get(name + ' activation', act_fn)
			l_act_fn = get_activation(_act_fn)
		
		if callable(norm_fn):
			_norm_fn = 'func'
			l_norm_fn = norm_fn
		else:
			_norm_fn = self.config.get(name + ' normalization', norm_fn)
			l_norm_fn = get_normalization(_norm_fn)

		_winit_fn = self.config.get(name + ' weightsinit', winit_fn)
		if 'special' in _winit_fn:
			split = _winit_fn.split()
			winit_name = split[0]
			if winit_name == 'he_uniform':
				input_nb_filters = int(x.get_shape()[-1])
				fan_in = input_nb_filters * (ksize**2)
				fan_out = nb_filters * (ksize**2) / (stride**2)
				filters_stdev = np.sqrt(4.0/(fan_in + fan_out))
				l_winit_fn = self.uniform_initializer(filters_stdev)
			else:
				raise Exception('Error weights initializer function name : ' + _winit_fn)
		else:
			l_winit_fn = get_weightsinit(_winit_fn)

		_binit_fn = self.config.get(name + ' biasesinit', binit_fn)
		l_binit_fn = get_weightsinit(_binit_fn)
		_padding = self.config.get(name + ' padding', padding)


		if self.using_tcl_library:
			x = tcl.conv2d(x, nb_filters, ksize, stride=stride,  
												activation_fn=l_act_fn,
												normalizer_fn=l_norm_fn,
												normalizer_params=norm_params,
												weights_initializer=l_winit_fn,
												padding=_padding,
												scope=name)
		else:
			x = tl.conv2d(x, nb_filters, ksize, strides=stride, 
						padding=_padding, 
						use_bias=has_bias,
						kernel_initializer=l_winit_fn,
						bias_initializer=l_binit_fn,
						trainable=True,	
						name=name)

			with tf.variable_scope(name):
				if l_norm_fn is not None:
					norm_params = norm_params or {}
					x = l_norm_fn(x, **norm_params)
				if l_act_fn is not None:
					x = l_act_fn(x)

		if disp:
			print('\t\tConv2D(' + str(name) + ') --> ', x.get_shape(), '  ', (_act_fn, _norm_fn, _winit_fn, _padding))
		if collect_end_points:
			self.end_points[name] = x
		return x

	def deconv2d(self, name, x, nb_filters, ksize, stride, *,
				norm_fn='none', norm_params=None, act_fn='relu', winit_fn='xavier', binit_fn='zeros', padding='SAME', has_bias=True,
				disp=True, collect_end_points=True):

		if callable(act_fn):
			_act_fn = 'func'
			l_act_fn = act_fn
		else:
			_act_fn = self.config.get(name + ' activation', act_fn)
			l_act_fn = get_activation(_act_fn)
		
		if callable(norm_fn):
			_norm_fn = 'func'
			l_norm_fn = norm_fn
		else:
			_norm_fn = self.config.get(name + ' normalization', norm_fn)
			l_norm_fn = get_normalization(_norm_fn)

		_winit_fn = self.config.get(name + ' weightsinit', winit_fn)
		if 'special' in _winit_fn:
			split = _winit_fn.split()
			winit_name = split[0]
			if winit_name == 'he_uniform':
				input_nb_filters = int(x.get_shape()[-1])
				fan_in = input_nb_filters * (ksize**2) / (stride**2)
				fan_out = nb_filters * (ksize**2)
				filters_stdev = np.sqrt(4.0/(fan_in + fan_out))
				l_winit_fn = self.uniform_initializer(filters_stdev)
			else:
				raise Exception('Error weights initializer function name : ' + _winit_fn)
		else:
			l_winit_fn = get_weightsinit(_winit_fn)
		_binit_fn = self.config.get(name + ' biasesinit', binit_fn)
		l_binit_fn = get_weightsinit(_binit_fn)
		_padding = self.config.get(name + ' padding', padding)

		if self.using_tcl_library:
			x = tcl.conv2d_transpose(x, nb_filters, ksize, stride=stride,  
												use_bias=True,
												activation_fn=l_act_fn,
												normalizer_fn=l_norm_fn,
												normalizer_params=norm_params,
												weights_initializer=l_winit_fn,
												padding=_padding,
												scope=name)
		else:
			x = tl.conv2d_transpose(x, nb_filters, ksize, strides=stride, 
							padding=_padding, 
							use_bias=has_bias, 
							kernel_initializer=l_winit_fn, 
							bias_initializer=l_binit_fn,
							trainable=True, name=name)
			with tf.variable_scope(name):
				if l_norm_fn is not None:
					norm_params = norm_params or {}
					x = l_norm_fn(x, **norm_params)
				if l_act_fn is not None:
					x = l_act_fn(x)

		if disp:
			print('\t\tDeonv2D(' + str(name) + ') --> ', x.get_shape(), '  ', (_act_fn, _norm_fn, _winit_fn, _padding))
		if collect_end_points:
			self.end_points[name] = x
		return x


	def fc(self, name, x, nb_nodes, *,
				norm_fn='none', norm_params=None, act_fn='none', winit_fn='xavier', binit_fn='zeros', has_bias=True,
				disp=True, collect_end_points=True):
		
		if callable(act_fn):
			_act_fn = 'func'
			l_act_fn = act_fn
		else:
			_act_fn = self.config.get(name + ' activation', act_fn)
			l_act_fn = get_activation(_act_fn)
		
		if callable(norm_fn):
			_norm_fn = 'func'
			l_norm_fn = norm_fn
		else:
			_norm_fn = self.config.get(name + ' normalization', norm_fn)
			l_norm_fn = get_normalization(_norm_fn)


		_winit_fn = self.config.get(name + ' weightsinit', winit_fn)
		if 'special' in _winit_fn:
			split = _winit_fn.split()
			winit_name = split[0]
			if winit_name == 'glorot_uniform':
				input_nb_nodes = int(x.get_shape()[-1])
				filters_stdev = np.sqrt(2.0/(input_nb_nodes + nb_nodes))
				l_winit_fn = self.uniform_initializer(filters_stdev)
			else:
				raise Exception('Error weights initializer function name : ' + _winit_fn)
		else:
			l_winit_fn = get_weightsinit(_winit_fn)

		_binit_fn = self.config.get(name + ' biasesinit', binit_fn)
		l_binit_fn = get_weightsinit(_binit_fn)

		if self.using_tcl_library:
			x = tcl.fully_connected(x, nb_nodes, 
					activation_fn=l_act_fn, normalizer_fn=l_norm_fn, normalizer_params=norm_params,
					weights_initializer=l_winit_fn, scope=name)
		else:
			x = tl.dense(x, nb_nodes, use_bias=has_bias, kernel_initializer=l_winit_fn,
													bias_initializer=l_binit_fn,
						trainable=True, name=name)

			with tf.variable_scope(name):
				if l_norm_fn is not None:
					norm_params = norm_params or {}
					x = l_norm_fn(x, **norm_params)
				if l_act_fn is not None:
					x = l_act_fn(x)


		if disp:
			print('\t\tFC(' + str(name) + ') --> ', x.get_shape(), '  ', (_act_fn, _norm_fn, _winit_fn))
		if collect_end_points:
			self.end_points[name] = x
		return x


	def concat(self, name, x_list, disp=True, collect_end_points=True):
		x = tf.concat(x_list, axis=3)
		if disp:
			print('\t\tConcat(' + str(name) + ') --> ', x.get_shape())
		if collect_end_points:
			self.end_points[name] = x
		return x

	def maxpool2d(self, name, x, size, stride, padding='SAME', disp=True, collect_end_points=True):
		_padding = self.config.get(name + ' padding', padding)
		x = tcl.max_pool2d(x, size, stride=stride, padding=_padding, scope=name)
		if disp:
			print('\t\tMaxPool(' + str(name) + ') --> ', x.get_shape())
		if collect_end_points:
			self.end_points[name] = x
		return x

	def upsample2d(self, name, x, size):
		# tf.resize_images
		pass

	def activation(self, x, act_fn='relu'):
		if not callable(act_fn):
			act_fn = get_activation(act_fn)
		return act_fn(x)

	def zero_padding2d(self, x, padding):
		if isinstance(padding, int):
			padding = ((padding, padding), (padding, padding))
		elif isinstance(padding, list) or isinstance(padding, tuple) and isinstance(padding[0], int) and isinstance(padding[1], int):
			padding = ((padding[0], padding[0]), (padding[1], padding[1]))
		
		else:
			raise ValueError('BaseNetwork : padding error')
		return tf.spatial_2d_padding(x, padding=padding, data_format='channels_last')

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

	@property
	def trainable_vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

	@property
	def store_vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name) + tf.get_collection(self.moving_variables_collection, scope=self.name)

	@property
	def moving_vars(self):
		return tf.get_collection(self.moving_variables_collection, scope=self.name)

	@property
	def all_vars(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

	@property
	def histogram_summary_list(self):
		return [tf.summary.histogram(var.name, var) for var in self.store_vars]