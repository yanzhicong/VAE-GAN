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


from utils.weightsinit import get_weightsinit
from utils.activation import get_activation
from utils.normalization import get_normalization


class BaseNetwork(object):
	def __init__(self, config, is_training):
		self.name = config['name']
		self.is_training = is_training
		self.moving_variables_collection = 'BATCH_NORM_MOVING_VARS'

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
		}

		self.fc_args = {
			'norm_fn':norm_fn,
			'norm_params':norm_params,
			'act_fn':act_fn,
			'winit_fn':winit_fn,
			'binit_fn':binit_fn,
		}

		self.deconv_args = {
			'norm_fn':norm_fn,
			'norm_params':norm_params,
			'act_fn':act_fn,
			'winit_fn':winit_fn,
			'binit_fn':binit_fn,
			'padding':padding,
		}

		self.out_conv_args = {
			'act_fn':output_act_fn,
			'winit_fn':winit_fn,
			'binit_fn':binit_fn,
			'padding':padding,
		}


		self.out_fc_args = {
			'act_fn':output_act_fn,
			'winit_fn':winit_fn,
			'binit_fn':binit_fn,
		}


	def conv2d(self, name, x, nb_filters, ksize, stride, 
				norm_fn='none', norm_params=None, act_fn='none', winit_fn='xavier', binit_fn='zeros', padding='SAME', disp=True, collect_end_points=True):

		_act_fn = self.config.get(name + ' activation', act_fn)
		l_act_fn = get_activation(_act_fn)

		_norm_fn = self.config.get(name + ' normalization', norm_fn)
		l_norm_fn = get_normalization(_norm_fn)

		_winit_fn = self.config.get(name + ' weightsinit', winit_fn)
		l_winit_fn = get_weightsinit(_winit_fn)

		_binit_fn = self.config.get(name + ' biasesinit', binit_fn)
		l_binit_fn = get_weightsinit(_binit_fn)

		_padding = self.config.get(name + ' padding', padding)

		# x = tcl.conv2d(x, nb_filters, ksize, stride=stride,  
		# 									activation_fn=l_act_fn,
		# 									normalizer_fn=l_norm_fn,
		# 									normalizer_params=norm_params,
		# 									weights_initializer=l_winit_fn,
		# 									padding=_padding,
		# 									scope=name)

		x = tl.conv2d(x, nb_filters, ksize, strides=stride, 
					padding=_padding, 
					use_bias=True,
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

	def deconv2d(self, name, x, nb_filters, ksize, stride, 
				norm_fn='none', norm_params=None, act_fn='relu', winit_fn='xavier', binit_fn='zeros', padding='SAME', disp=True, collect_end_points=True):

		_act_fn = self.config.get(name + ' activation', act_fn)
		l_act_fn = get_activation(_act_fn)

		_norm_fn = self.config.get(name + ' normalization', norm_fn)
		l_norm_fn = get_normalization(_norm_fn)

		_winit_fn = self.config.get(name + ' weightsinit', winit_fn)
		l_winit_fn = get_weightsinit(_winit_fn)

		_binit_fn = self.config.get(name + ' biasesinit', binit_fn)
		l_binit_fn = get_weightsinit(_binit_fn)

		_padding = self.config.get(name + ' padding', padding)

		# x = tcl.conv2d_transpose(x, nb_filters, ksize, stride=stride,  
		# 									use_bias=True,
		# 									activation_fn=l_act_fn,
		# 									normalizer_fn=l_norm_fn,
		# 									normalizer_params=norm_params,
		# 									weights_initializer=l_winit_fn,
		# 									padding=_padding,
		# 									scope=name)

		x = tl.conv2d_transpose(x, nb_filters, ksize, strides=stride, 
						padding=_padding, use_bias=True, kernel_initializer=l_winit_fn, bias_initializer=l_binit_fn,
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


	def fc(self, name, x, nb_nodes, norm_fn='none', norm_params=None, act_fn='none', winit_fn='xavier', binit_fn='zeros', disp=True, collect_end_points=True):
		_act_fn = self.config.get(name + ' activation', act_fn)
		l_act_fn = get_activation(_act_fn)

		_norm_fn = self.config.get(name + ' normalization', norm_fn)
		l_norm_fn = get_normalization(_norm_fn)

		_winit_fn = self.config.get(name + ' weightsinit', winit_fn)
		l_winit_fn = get_weightsinit(_winit_fn)

		_binit_fn = self.config.get(name + ' biasesinit', binit_fn)
		l_binit_fn = get_weightsinit(_binit_fn)

		# x = tcl.fully_connected(x, nb_nodes, 
		# 		activation_fn=l_act_fn, normalizer_fn=l_norm_fn, normalizer_params=norm_params,
		# 		weights_initializer=l_winit_fn, scope=name)

		x = tl.dense(x, nb_nodes, use_bias=True, kernel_initializer=l_winit_fn,
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

	# def upsample2d(self, name, x, stride,  disp=True, collect_end_points=True):
	# 	return x

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


