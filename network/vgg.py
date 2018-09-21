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

						conv_layer_name = 'conv%d_%d'%(block_ind+1, layer_ind)
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


	def load_pretrained_weights(self):
		print("load pretrained weights")

		pretrained_weights = self.config.get('load pretrained weights', 'vgg16')

		if pretrained_weights == 'vgg16':
			weights_h5_fp = self.find_pretrained_weights_path('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

			print(weights_h5_fp)


