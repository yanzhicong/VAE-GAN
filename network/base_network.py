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

from cfgs.networkconfig import get_config

import tensorflow as tf
from tensorflow import layers as tl
import tensorflow.contrib.layers as tcl
import numpy as np
from tensorflow.python import pywrap_tensorflow


from netutils.weightsinit import get_weightsinit
from netutils.activation import get_activation
from netutils.normalization import get_normalization

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
		output_act_fn = self.config.get('output_activation', 'none')

		has_bias = self.config.get('has bias', True)
		conv_has_bias = self.config.get('conv has bias', has_bias)
		fc_has_bias = self.config.get('fc has bias', has_bias)
		out_has_bias = self.config.get('out has bias', has_bias)

		norm_fn = self.config.get('normalization', 'batch_norm')
		norm_params = self.norm_params.copy()
		norm_params.update(self.config.get('normalization params', {}))

		winit_fn = self.config.get('weightsinit', 'xavier')
		binit_fn = self.config.get('biasesinit', 'zeros')

		padding = self.config.get('padding', 'SAME')

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
			act_fn_str = 'func'
			act_fn = act_fn
		else:
			act_fn_str = self.config.get(name + ' activation', act_fn)
			act_fn = get_activation(act_fn_str)
		
		if callable(norm_fn):
			norm_fn_str = 'func'
			norm_fn = norm_fn
		else:
			norm_fn_str = self.config.get(name + ' normalization', norm_fn)
			norm_fn = get_normalization(norm_fn_str)

		winit_fn_str = self.config.get(name + ' weightsinit', winit_fn)
		if 'special' in winit_fn_str:
			split = winit_fn_str.split()
			winit_name = split[0]
			if winit_name == 'he_uniform':
				input_nb_filters = int(x.get_shape()[-1])
				fan_in = input_nb_filters * (ksize**2)
				fan_out = nb_filters * (ksize**2) / (stride**2)
				filters_stdev = np.sqrt(4.0/(fan_in + fan_out))
				winit_fn = self.uniform_initializer(filters_stdev)
			else:
				raise Exception('Error weights initializer function name : ' + winit_fn_str)
		else:
			winit_fn = get_weightsinit(winit_fn_str)

		binit_fn_str = self.config.get(name + ' biasesinit', binit_fn)
		binit_fn = get_weightsinit(binit_fn_str)
		_padding = self.config.get(name + ' padding', padding)


		if self.using_tcl_library:
			x = tcl.conv2d(x, nb_filters, ksize, stride=stride,  
												activation_fn=act_fn,
												normalizer_fn=norm_fn,
												normalizer_params=norm_params,
												weights_initializer=winit_fn,
												padding=_padding,
												scope=name)
		else:
			x = tl.conv2d(x, nb_filters, ksize, strides=stride, 
						padding=_padding, 
						use_bias=has_bias,
						kernel_initializer=winit_fn,
						bias_initializer=binit_fn,
						trainable=True,	
						name=name)

			with tf.variable_scope(name):
				if norm_fn is not None:
					norm_params = norm_params or {}
					x = norm_fn(x, **norm_params)
				if act_fn is not None:
					x = act_fn(x)

		if disp:
			print('\t\tConv2D(' + str(name) + ') --> ', x.get_shape(), '  ', (act_fn_str, norm_fn_str, winit_fn_str, _padding))
		if collect_end_points:
			self.end_points[name] = x
		return x

	def deconv2d(self, name, x, nb_filters, ksize, stride, *,
				norm_fn='none', norm_params=None, act_fn='relu', winit_fn='xavier', binit_fn='zeros', padding='SAME', has_bias=True,
				disp=True, collect_end_points=True):

		if callable(act_fn):
			act_fn_str = 'func'
			act_fn = act_fn
		else:
			act_fn_str = self.config.get(name + ' activation', act_fn)
			act_fn = get_activation(act_fn_str)
		
		if callable(norm_fn):
			norm_fn_str = 'func'
			norm_fn = norm_fn
		else:
			norm_fn_str = self.config.get(name + ' normalization', norm_fn)
			norm_fn = get_normalization(norm_fn_str)

		winit_fn_str = self.config.get(name + ' weightsinit', winit_fn)
		if 'special' in winit_fn_str:
			split = winit_fn_str.split()
			winit_name = split[0]
			if winit_name == 'he_uniform':
				input_nb_filters = int(x.get_shape()[-1])
				fan_in = input_nb_filters * (ksize**2) / (stride**2)
				fan_out = nb_filters * (ksize**2)
				filters_stdev = np.sqrt(4.0/(fan_in + fan_out))
				winit_fn = self.uniform_initializer(filters_stdev)
			else:
				raise Exception('Error weights initializer function name : ' + winit_fn_str)
		else:
			winit_fn = get_weightsinit(winit_fn_str)
		binit_fn_str = self.config.get(name + ' biasesinit', binit_fn)
		binit_fn = get_weightsinit(binit_fn_str)
		_padding = self.config.get(name + ' padding', padding)

		if self.using_tcl_library:
			x = tcl.conv2d_transpose(x, nb_filters, ksize, stride=stride,  
												use_bias=True,
												activation_fn=act_fn,
												normalizer_fn=norm_fn,
												normalizer_params=norm_params,
												weights_initializer=winit_fn,
												padding=_padding,
												scope=name)
		else:
			x = tl.conv2d_transpose(x, nb_filters, ksize, strides=stride, 
							padding=_padding, 
							use_bias=has_bias, 
							kernel_initializer=winit_fn, 
							bias_initializer=binit_fn,
							trainable=True, name=name)
			with tf.variable_scope(name):
				if norm_fn is not None:
					norm_params = norm_params or {}
					x = norm_fn(x, **norm_params)
				if act_fn is not None:
					x = act_fn(x)

		if disp:
			print('\t\tDeonv2D(' + str(name) + ') --> ', x.get_shape(), '  ', (act_fn_str, norm_fn_str, winit_fn_str, _padding))
		if collect_end_points:
			self.end_points[name] = x
		return x


	def fc(self, name, x, nb_nodes, *,
				norm_fn='none', norm_params=None, act_fn='none', winit_fn='xavier', binit_fn='zeros', has_bias=True,
				disp=True, collect_end_points=True):
		
		if callable(act_fn):
			act_fn_str = 'func'
			act_fn = act_fn
		else:
			act_fn_str = self.config.get(name + ' activation', act_fn)
			act_fn = get_activation(act_fn_str)
		
		if callable(norm_fn):
			norm_fn_str = 'func'
			norm_fn = norm_fn
		else:
			norm_fn_str = self.config.get(name + ' normalization', norm_fn)
			norm_fn = get_normalization(norm_fn_str)


		winit_fn_str = self.config.get(name + ' weightsinit', winit_fn)
		if 'special' in winit_fn_str:
			split = winit_fn_str.split()
			winit_name = split[0]
			if winit_name == 'glorot_uniform':
				input_nb_nodes = int(x.get_shape()[-1])
				filters_stdev = np.sqrt(2.0/(input_nb_nodes + nb_nodes))
				winit_fn = self.uniform_initializer(filters_stdev)
			else:
				raise Exception('Error weights initializer function name : ' + winit_fn_str)
		else:
			winit_fn = get_weightsinit(winit_fn_str)

		binit_fn_str = self.config.get(name + ' biasesinit', binit_fn)
		binit_fn = get_weightsinit(binit_fn_str)

		if self.using_tcl_library:
			x = tcl.fully_connected(x, nb_nodes, 
					activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
					weights_initializer=winit_fn, scope=name)
		else:
			x = tl.dense(x, nb_nodes, use_bias=has_bias, kernel_initializer=winit_fn,
													bias_initializer=binit_fn,
						trainable=True, name=name)

			with tf.variable_scope(name):
				if norm_fn is not None:
					norm_params = norm_params or {}
					x = norm_fn(x, **norm_params)
				if act_fn is not None:
					x = act_fn(x)


		if disp:
			print('\t\tFC(' + str(name) + ') --> ', x.get_shape(), '  ', (act_fn_str, norm_fn_str, winit_fn_str))
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
	def conv_vars(self):
		return [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name) if self.name+'/conv' in var.name]

	@property
	def top_vars(self):
		return [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name) if self.name+'/fc' in var.name]

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


	def find_pretrained_weights_path(self, weights_filename, throw_not_found_error=False):
		model_path = os.path.join('C:\\Models', weights_filename)
		if not os.path.exists(model_path):
			model_path = os.path.join('E:\\Models', weights_filename)
		if not os.path.exists(model_path):
			model_path = os.path.join('F:\\Models', weights_filename)
		if not os.path.exists(model_path):
			model_path = os.path.join('/mnt/data01/models/', weights_filename)
		if not os.path.exists(model_path):
			model_path = os.path.join('/mnt/data02/models/', weights_filename)
		if not os.path.exists(model_path):
			model_path = os.path.join('/mnt/data03/models/', weights_filename)
		if not os.path.exists(model_path):
			model_path = os.path.join('/mnt/data04/models/', weights_filename)
		if not os.path.exists(model_path):
			if throw_not_found_error:
				raise ValueError('Base Network : the pretrained weights file ' + weights_filename + ' is not found')
			else:
				model_path = None
		return model_path


	def load_pretrained_weights(self, sess):

		print('base network load pretrained weights')
		return False


	def load_pretrained_model_weights(self, sess, cfg, network_name, only_bottom=True):
		config_file = get_config(cfg)
		asset_filepath = config_file['assets dir']
		ckpt_path = os.path.join(asset_filepath, config_file["trainer params"].get("checkpoint dir", "checkpoint"))
		ckpt_name = ''
		with open(os.path.join(ckpt_path, 'checkpoint'), 'r') as infile:
			for line in infile:
				if line.startswith('model_checkpoint_path'):
					ckpt_name = line[len("model_checkpoint_path: \""):-2]
		checkpoint_path = os.path.join(ckpt_path, ckpt_name)

		print("Load checkpoint : ", checkpoint_path)
		reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
		var_to_shape_map = reader.get_variable_to_shape_map()

		assign_list = []
		var_list = self.all_vars
		var_dict = {var.name.split(':')[0] : var for var in var_list}

		for key in var_to_shape_map:
			if key.startswith(network_name):
				if only_bottom and 'fc' in key:
					continue
				var_name = self.name + '/' + key[len(network_name)+1:]
				assign_list.append(tf.assign(var_dict[var_name], reader.get_tensor(key)))

		assign_op = tf.group(assign_list)
		sess.run(assign_op)
		return True