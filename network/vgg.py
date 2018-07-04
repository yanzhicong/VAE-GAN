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


from utils.weightsinit import get_weightsinit
from utils.activation import get_activation
from utils.normalization import get_normalization



class VGG(object):

	'''
		a flexible image feature encoding network class, which can be customized easily.
		
		the structure imitates the vgg network structure, which consists several convolution layers 
		and fc layer behind.

		the convolution layers are divied into blocks, in the end of each blocks there is a max pooling behind.
		the max pooling params is always ksize=2 and strides=2, and padding of convolution layers is always 'SAME'
		the convolution layers params(in @params:config):
			'nb_conv_blocks':	5
			'nb_conv_filters':	[64, 128, 256, 512, 512]
			'nb_conv_layers':	[2, 2, 3, 3, 3]
			'nb_conv_ksize':	[3, 3, 3, 3, 3]
			'no_maxpooling':	True or False

		the fc layers are appended behind the convolution layers. if 'including_top' is false, then there is 
		no fc layers.
		the fc layers params(in @params:config):
			'including_top':	
			'nb_fc_nodes':

		the interval layers params(both convolution layer and fc layer):
			'activation':
			'activation params':
			'batch_norm':
			'batch_norm_params':
			'weightsinit':
			'weightsinit_params':

		the output nodes is defined by 'output_dims', if there is fc layers, the output tensor shape is [batchsize, output_dims]
		else if there is no fc layers, the output tensor shape is [batchsize, h, w, output_dims], and if 'output_dims' == 0, then 
		there is no extra layers append to the network
		output params:
			'output_dims':
			'output_activation':
			'output_activation_params'
	'''


	def __init__(self, config, is_training):
		self.name = config.get('name', 'VGG')
		self.is_training = is_training
		self.normalizer_params = {
			'decay' : 0.999,
			'center' : True,
			'scale' : False,
			'is_training' : self.is_training
		}
		self.config = config

		# when first applying the network to input tensor, the reuse is false
		self.reuse = False


	def __call__(self, x):
		act_fn = get_activation(
					self.config.get('activation', 'relu'),
					self.config.get('activation params', ''))

		norm_fn, norm_params = get_normalization(
					self.config.get('batch_norm', 'batch_norm'),
					self.config.get('batch_norm params', None))

		winit_fn = get_weightsinit(
					self.config.get('weightsinit', 'normal'),
					self.config.get('weightsinit_params', '0.00 0.02'))

		binit_fn = get_weightsinit(
					self.config.get('biasesinit', 'normal'),
					self.config.get('biasesinit_params', '0.00 0.02'))


		# convolution structure parameters
		nb_conv_blocks = int(self.config.get('nb_conv_blocks', 5))
		nb_conv_filters = self.config.get('nb_conv_filters', [64, 128, 256, 512, 512])
		nb_conv_layers = self.config.get('nb_conv_layers', [2, 2, 3, 3, 3])
		nb_conv_ksize = self.config.get('nb_conv_ksize', [3, 3, 3, 3, 3])
		no_maxpooling = self.config.get('no_maxpooling', False)

		# fully connected parameters
		including_top = self.config.get('including_top', True)
		nb_fc_nodes = self.config.get('nb_fc_nodes', [1024, 1024])

		# output stage parameters
		output_dims = self.config.get('output_dims', 0)  # zero for no output layer
		output_act_fn = get_activation(
					self.config.get('output_activation', 'none'),
					self.config.get('output_activation_params', ''))


		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

			end_points = {}

			# construct convolution layers
			for block_ind in range(nb_conv_blocks):
				for layer_ind in range(nb_conv_layers[block_ind]):
					if layer_ind == nb_conv_layers[block_ind]-1:
					 # and block_ind != nb_conv_blocks-1:
						if no_maxpooling:
							x = tcl.conv2d(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], 
									stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
									padding='SAME', weights_initializer=winit_fn, biases_initializer=binit_fn, scope='conv%d_%d'%(block_ind+1, layer_ind))
						else:
							x = tcl.conv2d(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], 
									stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
									padding='SAME', weights_initializer=winit_fn, biases_initializer=binit_fn, scope='conv%d_%d'%(block_ind+1, layer_ind))
							x = tcl.max_pool2d(x, 2, stride=2, padding='SAME')							
					else:
						x = tcl.conv2d(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], 
								stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
								padding='SAME', weights_initializer=winit_fn, biases_initializer=binit_fn, scope='conv%d_%d'%(block_ind+1, layer_ind))
					end_points['conv%d_%d'%(block_ind+1, layer_ind)] = x
 
			# construct top fully connected layer
			if including_top: 
				x = tcl.flatten(x)
				for ind, nb_nodes in enumerate(nb_fc_nodes):
					x = tcl.fully_connected(x, nb_nodes, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							weights_initializer=winit_fn, biases_initializer=binit_fn, scope='fc%d'%ind)
					end_points['fc%d'%ind] = x

				if output_dims != 0:
					x = tcl.fully_connected(x, output_dims, activation_fn=output_act_fn, weights_initializer=winit_fn, biases_initializer=binit_fn, scope='fc_out')
					end_points['fc_out'] = x

			# else construct a convolution layer for output
			elif output_dims != 0:
				x = tcl.conv2d(x, output_dims, 1, 
							stride=1, activation_fn=output_act_fn, padding='SAME', weights_initializer=winit_fn, scope='conv_out')
				end_points['conv_out'] = x


			if self.config.get('debug', False):
				print('VGG : (' + self.name + ')')
				print('\tactivation :               ', self.config.get('activation', ''))
				print('\tactivation_params :        ', self.config.get('activation_params', ''))
				print('\tbatch_norm :               ', self.config.get('batch_norm', ''))
				print('\tbatch_norm_params :        ', self.config.get('batch_norm_params', ''))
				print('\tweightsinit :              ', self.config.get('weightsinit', ''))
				print('\tweightsinit_params :       ', self.config.get('weightsinit_params', ''))
				print('\tnb_conv_blocks :           ', self.config.get('nb_conv_blocks', ''))
				print('\tnb_conv_filters :          ', self.config.get('nb_conv_filters', ''))
				print('\tnb_conv_layers :           ', self.config.get('nb_conv_layers', ''))
				print('\tnb_conv_ksize :            ', self.config.get('nb_conv_ksize', ''))
				print('\tno_maxpooling :            ', self.config.get('no_maxpooling', ''))
				print('\tincluding_top :            ', self.config.get('including_top', ''))
				print('\tnb_fc_nodes :              ', self.config.get('nb_fc_nodes', ''))
				print('\toutput_dims :              ', self.config.get('output_dims', ''))
				print('\toutput_activation :        ', self.config.get('output_activation', ''))
				print('\toutput_activation_params : ', self.config.get('output_activation_params', ''))
				for var_name, var in end_points.items():
					print('\t\t' + var_name, ' --> ', var.get_shape())
				print('')

			return x, end_points

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

