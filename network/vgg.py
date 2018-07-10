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


from .basenetwork import BaseNetwork


arg_scope = tf.contrib.framework.arg_scope

class VGG(BaseNetwork):

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
		BaseNetwork.__init__(self, config, is_training)
		self.config = config
		self.reuse = False


	def __call__(self, x):
		act_fn = get_activation(self.config.get('activation', 'relu'))

		norm_fn = get_normalization(self.config.get('normalization', 'batch_norm'))
		norm_params = self.norm_params.copy()
		norm_params.update(self.config.get('normalization params', {}))

		winit_fn = get_weightsinit(self.config.get('weightsinit', 'normal 0.00 0.02'))
		padding = self.config.get('padding', 'SAME')

		# convolution structure parameters
		including_conv = self.config.get('including_conv', True)
		nb_conv_blocks = int(self.config.get('nb_conv_blocks', 5))
		nb_conv_filters = self.config.get('nb_conv_filters', [64, 128, 256, 512, 512])
		nb_conv_layers = self.config.get('nb_conv_layers', [2, 2, 3, 3, 3])
		nb_conv_ksize = self.config.get('nb_conv_ksize', [3 for i in range(nb_conv_blocks)])
		no_maxpooling = self.config.get('no_maxpooling', False)

		# fully connected parameters
		including_top = self.config.get('including_top', True)
		nb_fc_nodes = self.config.get('nb_fc_nodes', [1024, 1024])

		# output stage parameters
		output_dims = self.config.get('output_dims', 0)  # zero for no output layer
		output_act_fn = get_activation(self.config.get('output_activation', 'none'))


		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

			end_points = {}

			if self.config.get('debug', False):
				print('VGG : (' + self.name + ')')
				print('\tactivation :               ', self.config.get('activation', ''))
				print('\tnormalization :             ', self.config.get('normalization', ''))
				print('\tweightsinit :              ', self.config.get('weightsinit', ''))
				print('')
				print('\tincluding_conv :           ', self.config.get('including_conv', ''))
				print('\tnb_conv_blocks :           ', self.config.get('nb_conv_blocks', ''))
				print('\tnb_conv_filters :          ', self.config.get('nb_conv_filters', ''))
				print('\tnb_conv_layers :           ', self.config.get('nb_conv_layers', ''))
				print('\tnb_conv_ksize :            ', self.config.get('nb_conv_ksize', ''))
				print('\tno_maxpooling :            ', self.config.get('no_maxpooling', ''))
				print('\tconv network :')

			if including_conv:
				for block_ind in range(nb_conv_blocks):
					for layer_ind in range(nb_conv_layers[block_ind]):

						_conv_layer_name = 'conv%d_%d'%(block_ind+1, layer_ind)
						_maxpool_layer_name = 'maxpool%d'%(block_ind+1)

						_act_fn = self.config.get(_conv_layer_name + ' activation', None)
						l_act_fn = get_activation(_act_fn) if _act_fn else act_fn

						_norm_fn = self.config.get(_conv_layer_name + ' normalization', None)
						l_norm_fn = get_normalization(_norm_fn) if _norm_fn else norm_fn

						_winit_fn = self.config.get(_conv_layer_name + ' weightsinit', None)
						l_winit_fn = get_weightsinit(_winit_fn) if _winit_fn else winit_fn

						_padding = self.config.get(_conv_layer_name + ' padding', None)
						l_padding = _padding if _padding else padding

						_mp_padding = self.config.get(_conv_layer_name + ' maxpool padding', None)
						l_padding = _mp_padding if _mp_padding else padding

						def print_conv(x):
							if self.config.get('debug', False):
								print('\t\t', _conv_layer_name, ' --> ', x.get_shape(), '  ', (_act_fn, _norm_fn, _winit_fn, _padding))
						def print_maxpool(x):
							if self.config.get('debug', False):
								print('\t\t', _maxpool_layer_name, ' --> ', x.get_shape())

						with arg_scope([tcl.conv2d], 
									padding='SAME',
									activation_fn=l_act_fn,
									normalizer_fn=l_norm_fn,
									normalizer_params=norm_params,
									weights_initializer=l_winit_fn):
							if layer_ind == nb_conv_layers[block_ind]-1 and block_ind != nb_conv_blocks-1:
								if no_maxpooling:
									x = tcl.conv2d(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], stride=2, 
											scope=_conv_layer_name)
									print_conv(x)
									end_points[_conv_layer_name] = x
								else:
									x = tcl.conv2d(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], stride=1, 
											scope=_conv_layer_name)
									print_conv(x)
									end_points[_conv_layer_name] = x

									x = tcl.max_pool2d(x, 2, stride=2, padding='SAME')
									print_maxpool(x)
									end_points[_maxpool_layer_name] = x

							else:
								x = tcl.conv2d(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], stride=1, 
											scope=_conv_layer_name)
								print_conv(x)
								end_points[_conv_layer_name] = x


			if self.config.get('debug', False):
				print('\tincluding_top :            ', self.config.get('including_top', ''))
				print('\tnb_fc_nodes :              ', self.config.get('nb_fc_nodes', ''))
				print('\tfc network :')

			# construct top fully connected layer
			if including_top: 
				x = tcl.flatten(x)
				for ind, nb_nodes in enumerate(nb_fc_nodes):

					_fc_layer_name = 'fc%d'%ind

					_act_fn = self.config.get(_fc_layer_name + ' activation', None)
					l_act_fn = get_activation(_act_fn) if _act_fn else act_fn

					_norm_fn = self.config.get(_fc_layer_name + ' normalization', None)
					l_norm_fn = get_normalization(_norm_fn) if _norm_fn else norm_fn

					_winit_fn = self.config.get(_fc_layer_name + ' weightsinit', None)
					l_winit_fn = get_weightsinit(_winit_fn) if _winit_fn else winit_fn

					def print_fc(x):
						if self.config.get('debug', False):
							print('\t\t', _fc_layer_name, ' --> ', x.get_shape(), '  ', (_act_fn, _norm_fn, _winit_fn))	

					x = tcl.fully_connected(x, nb_nodes, 
							activation_fn=l_act_fn, normalizer_fn=l_norm_fn, normalizer_params=norm_params,
							weights_initializer=l_winit_fn, scope=_fc_layer_name)
					print_fc(x)
					end_points[_fc_layer_name] = x

			if self.config.get('debug', False):
				print('\toutput_dims :              ', self.config.get('output_dims', ''))
				print('\toutput_activation :        ', self.config.get('output_activation', ''))
				print('\toutput network : ')

			if output_dims != 0:
				if including_top: 
					x = tcl.fully_connected(x, output_dims, 
							activation_fn=output_act_fn, 
							weights_initializer=winit_fn, scope='fc_out')
					if self.config.get('debug', False):
						print('\t\tfc_out --> ', x.get_shape())
					end_points['fc_out'] = x

				# else construct a convolution layer for output
				else:
					x = tcl.conv2d(x, output_dims, 1, stride=1, padding='SAME',
						activation_fn=output_act_fn,  
						weights_initializer=winit_fn, scope='conv_out')
					if self.config.get('debug', False):
						print('\t\tconv_out --> ', x.get_shape())
					end_points['conv_out'] = x

			return x, end_points


