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
from time import *

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl


from .base_network import BaseNetwork

class VGG(BaseNetwork):

	'''
		a flexible image feature encoding network class, which can be customized easily.
		
		this class implements vgg network structure, which composed by several convolution layers 
		and fc layers behind.

		the convolution layers are divied into blocks, in the end of each blocks there is a max pooling or avg pooling 
		behind(except the last block). the max pooling params is always ksize=2 and strides=2, and padding of convolution 
		layers is always 'SAME'.
		the convolution layers params(in @params:config):
			"conv nb blocks":	5
			"conv nb filters":	[64, 128, 256, 512, 512]
			"conv nb layers":	[2, 2, 3, 3, 3]
			"conv ksize":	[3, 3, 3, 3, 3]
			"no maxpooling":	True or False

		the fc layers are appended behind the convolution layers. if "including top" is false, then there is 
		no fc layers.
		the fc layers params(in @params:config):
			"including top":	
			"fc nb nodes":

		the output nodes is defined by "output dims", if there is fc layers, the output tensor shape is [batchsize, output_dims]
		else if there is no fc layers, the output tensor shape is [batchsize, h, w, output_dims], and if "output dims" == 0, then 
		there is no extra layers append to the network
		output params:
			"output dims":
			'output_activation':
			'output_activation_params'
	'''

	def __init__(self, config, is_training):
		BaseNetwork.__init__(self, config, is_training)
		self.config = config
		self.reuse = False


	def __call__(self, x):

		# convolution structure parameters
		including_conv = self.config.get("including conv", True)
		conv_nb_blocks = int(self.config.get("conv nb blocks", 5))
		conv_nb_filters = self.config.get("conv nb filters", [64, 128, 256, 512, 512])
		conv_nb_layers = self.config.get("conv nb layers", [2, 2, 3, 3, 3])
		conv_ksize = self.config.get("conv ksize", [3 for i in range(conv_nb_blocks)])
		no_maxpooling = self.config.get("no maxpooling", False)

		# fully connected parameters
		including_top = self.config.get("including top", True)
		fc_nb_nodes = self.config.get("fc nb nodes", [1024, 1024])

		# output stage parameters
		output_dims = self.config.get("output dims", 0)  # zero for no output layer

		debug = self.config.get('debug', False)

		self.end_points = {}
		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

			if debug:
				print('VGG : (' + self.name + ')')
				print('\tincluding_conv :           ', self.config.get("including conv", ''))
				print('\tconv network :')

			if including_conv:
				for block_ind in range(conv_nb_blocks):
					for layer_ind in range(conv_nb_layers[block_ind]):

						conv_layer_name = 'conv%d_%d'%(block_ind+1, layer_ind+1)
						maxpool_layer_name = 'maxpool%d'%(block_ind+1)

						# construct a downsample layer in the end of each blocks except the last block
						if layer_ind == conv_nb_layers[block_ind]-1 and block_ind != conv_nb_blocks-1:
							if no_maxpooling:
								x = self.conv2d(conv_layer_name, x, conv_nb_filters[block_ind], conv_ksize[block_ind], 
																stride=2, **self.conv_args, disp=debug)
							else:
								x = self.conv2d(conv_layer_name, x, conv_nb_filters[block_ind], conv_ksize[block_ind], 
																stride=1, **self.conv_args, disp=debug)
								x = self.maxpool2d(maxpool_layer_name, x, 2, stride=2, padding='SAME')
						else:
							x = self.conv2d(conv_layer_name, x, conv_nb_filters[block_ind], conv_ksize[block_ind], 
															stride=1, **self.conv_args, disp=debug)

			if debug:
				print('\tincluding_top :            ', self.config.get("including top", ''))
				print('\tfc network :')

			# construct top fully connected layer
			if including_top: 
				x = tcl.flatten(x)
				for ind, nb_nodes in enumerate(fc_nb_nodes):
					x = self.fc('fc%d'%(ind+1), x, nb_nodes, **self.fc_args, disp=debug)

			if debug:
				print('\toutput_dims :              ', self.config.get("output dims", ''))
				print('\toutput network : ')

			if output_dims != 0:
				# construct a fc layer for output
				if including_top: 
					x = self.fc('fc_out', x, output_dims, **self.out_fc_args, disp=debug)

				# else construct a convolution layer for output
				else:
					x = self.conv2d('conv_out', x, output_dims, 1, stride=1, **self.out_conv_args, disp=debug)

			if debug:
				print('')
			return x, self.end_points


	def __load_pretrained_h5py_weights(self, sess, weights_h5_fp):
		var_list = self.all_vars
		var_dict = {var.name.split(':')[0] : var for var in var_list}
		import h5py
		f = h5py.File(weights_h5_fp, mode='r')
		if 'layer_names' not in f.attrs and 'model_weights' in f:
			f = f['model_weights']
		assign_list = []

		for key, value in f.items():
			if 'block' in key and 'conv' in key:
				block_ind = 0
				conv_ind = 0
				for split in key.split('_'):
					if split.startswith('block'):
						block_ind = int(split[len('block'):])
					elif split.startswith('conv'):
						conv_ind = int(split[len('conv'):])
				for key2, value2 in value.items():
					if '_W_' in key2:
						var_name = self.name + '/conv%d_%d/kernel'%(block_ind, conv_ind)
						assign_value = np.array(value2)
						assign_shape = assign_value.shape
						var_shape = var_dict[var_name].get_shape()
						assert(int(var_shape[2]) <= assign_shape[2])
						assert(int(var_shape[3]) <= assign_shape[3])
						if int(var_shape[2]) < assign_shape[2]:
							assign_value = assign_value[:, :, 0:int(var_shape[2]), :]
						if int(var_shape[3]) < assign_shape[3]:
							assign_value = assign_value[:, :, :, 0:int(var_shape[3])]
					elif '_b_' in key2:
						var_name = self.name + '/conv%d_%d/bias'%(block_ind, conv_ind)
						assign_value = np.array(value2)
						assign_shape = assign_value.shape
						var_shape = var_dict[var_name].get_shape()
						assert(int(var_shape[0]) <= assign_shape[0])
						if int(var_shape[0]) < assign_shape[0]:
							assign_value = assign_value[0:int(var_shape[0])]
					else:
						continue

					assign_list.append(tf.assign(var_dict[var_name], assign_value))

		assign_op = tf.group(assign_list)
		sess.run(assign_op)

	def load_pretrained_weights(self, sess):
		pretrained_weights = self.config.get('load pretrained weights', '')

		w_name = pretrained_weights.split(' ')[0]
		print(w_name)

		if w_name == 'vgg16':
			weights_h5_fp = self.find_pretrained_weights_path('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', throw_not_found_error=True)
			if weights_h5_fp is not None:
				self.__load_pretrained_h5py_weights(sess, weights_h5_fp)
				return True
			else:
				return False
		elif w_name == 'vgg19':
			weights_h5_fp = self.find_pretrained_weights_path('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', throw_not_found_error=True)
			if weights_h5_fp is not None:
				self.__load_pretrained_h5py_weights(sess, weights_h5_fp)
				return True
			else:
				return False

		elif w_name == 'config':
			config_path = pretrained_weights.split(' ')[1]
			config_network_name = pretrained_weights.split(' ')[2]
			return self.load_pretrained_model_weights(sess, config_path, config_network_name)
		else:
			return False
