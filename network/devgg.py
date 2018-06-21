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

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl

from netutils.weightsinit import get_weightsinit
from netutils.activation import get_activation
from netutils.normalization import get_normalization

from .base_network import BaseNetwork

arg_scope = tf.contrib.framework.arg_scope

class DEVGG(BaseNetwork):
	'''
		a flexible generative model network class, can be customized easily.
		from z(one-dimensional or three-dimensional) to x(three-dimensional)

	
		at first some fc layers are constructed. if 'including_bottom' is false, then there is 
		no fc layers.
		the fc layers params(in @params:config):
			'including_bottom':
			"fc nb nodes":

		the deconvolution layers are divied into blocks, in the end of each blocks the deconv layer stride is 2 and other deconv layer stride is 1.
		the convolution layers params(in @params:config):

			'including_deconv':
			'deconv nb blocks':
			'deconv nb filters':
			'deconv nb layers':
			'deconv_ksize':


		the interval layers params(both convolution layer and fc layer):
			'activation':
			'batch_norm':
			'batch_norm_params':
			'weightsinit':

		the output nodes is defined by "output dims", if there is fc layers, the output tensor shape is [batchsize, output_dims]
		else if there is no fc layers, the output tensor shape is [batchsize, h, w, output_dims], and if "output dims" == 0, then 
		there is no extra layers append to the network
		output params:
			"output dims":
			'output_activation':
			'output_stride':
			'output_ksize':
			'output_shape':
	'''

	def __init__(self, config, is_training):
		BaseNetwork.__init__(self, config, is_training)
		# when first applying the network to input tensor, the reuse is false
		self.reuse = False


	def __call__(self, x):
		# fully connected parameters
		including_bottom = self.config.get('including_bottom', True)
		fc_nb_nodes = self.config.get("fc nb nodes", [1024, 1024])
		fc_output_reshape = self.config.get('fc_output_reshape', None)

		# convolution structure parameters
		including_deconv = self.config.get('including_deconv', False)
		deconv_nb_blocks = int(self.config.get('deconv nb blocks', 5))
		deconv_nb_filters = self.config.get('deconv nb filters', [512, 512, 256, 128, 64])
		deconv_nb_layers = self.config.get('deconv nb layers', [2, 2, 3, 3, 3])
		deconv_ksize = self.config.get('deconv_ksize', [3, 3, 3, 3, 3])

		# output stage parameters
		output_dims = self.config.get("output dims", 0)  # zero for no output layer
		output_shape = self.config.get('output_shape', None)
		output_stride = self.config.get('output_stride', 1)
		output_ksize = self.config.get('output_ksize', 1)
		debug = self.config.get('debug', False)

		self.end_points={}

		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

			self.end_points = {}

			if debug:
				print('DEVGG : (' + self.name + ')')
				print('\tincluding_bottom :          ', self.config.get('including_bottom', ''))
				print('\tfc_output_reshape :         ', self.config.get('fc_output_reshape', ''))
				print('\tfc network :')

			# construct bottom fully connected layer
			if including_bottom: 
				x = tcl.flatten(x)
				for ind, nb_nodes in enumerate(fc_nb_nodes):
					x = self.fc('fc%d'%ind, x, nb_nodes, **self.fc_args, disp=debug)

				if fc_output_reshape is not None:
					x = self.fc('fc%d'%(len(fc_nb_nodes)+1), x, int(np.product(list(fc_output_reshape))), **self.fc_args, disp=debug)
					x = tf.reshape(x, [-1, ] + list(fc_output_reshape))

			if debug:
					print('\tincluding_deconv :          ', self.config.get('including_deconv', ''))
					print('\tdeconv network : ')

			# construct deconvolution layers
			if including_deconv:
				for block_ind in range(deconv_nb_blocks):
					for layer_ind in range(deconv_nb_layers[block_ind]):
						deconv_layer_name = 'deconv%d_%d'%(block_ind+1, layer_ind)
						if layer_ind == deconv_nb_layers[block_ind]-1 and block_ind != deconv_nb_blocks-1:
							x = self.deconv2d(deconv_layer_name, x, deconv_nb_filters[block_ind], deconv_ksize[block_ind], stride=2, **self.deconv_args, disp=debug)
						else:
							x = self.deconv2d(deconv_layer_name, x, deconv_nb_filters[block_ind], deconv_ksize[block_ind], stride=1, **self.deconv_args, disp=debug)

			# construct output layers
			if debug:
					print('\toutput network : ')
			if output_dims != 0:
				if including_deconv:
					x = self.deconv2d('deconv_out', x, output_dims, output_ksize, stride=output_stride, **self.out_conv_args, disp=debug)
				else:
					x = self.fc('fc_out', x, output_dims, **self.out_fc_args, disp=debug)
					if output_shape is not None:
						x = tf.reshape(x, [-1,] + list(output_shape))
			if debug:
				print('')

			return x, self.end_points

