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




class UNet(BaseNetwork):

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

		nb_conv_blocks = self.config.get('nb_conv_blocks', 4)
		nb_conv_layers = self.config.get('nb_conv_layers', [2, 2, 2, 2, 2])
		nb_conv_filters = self.config.get('nb_conv_filters', [64, 128, 256, 512, 1024])
		nb_conv_ksize = self.config.get('nb_conv_ksize', [3 for i in range(nb_conv_blocks+1)])
		no_maxpooling = self.config.get('no_maxpooling', False)
		no_upsampling = self.config.get('no_upsampling', False)

		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True


			block_end = {}
			end_points = {}

			with arg_scope([tcl.conv2d, tcl.conv2d_transpose], 
						padding='SAME',
						activation_fn=act_fn,
						normalizer_fn=norm_fn,
						normalizer_params=norm_params,
						weights_initializer=winit_fn):

				for block_ind in range(nb_conv_blocks):
					for layer_ind in range(nb_conv_layers[block_ind]):

						_conv_layer_name = 'conv_ds%d_%d'%(block_ind+1, layer_ind)
						_maxpool_layer_name = 'maxpool_ds%d'%(block_ind+1)

						if layer_ind == nb_conv_layers[block_ind]-1:
							if no_maxpooling:
								block_end[block_ind] = x
								x = tcl.conv2d(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], stride=2, 
										scope=_conv_layer_name)
								end_points[_conv_layer_name] = x
							else:
								x = tcl.conv2d(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], stride=1, 
										scope=_conv_layer_name)
								block_end[block_ind] = x
								end_points[_conv_layer_name] = x
								x = tcl.max_pool2d(x, 2, stride=2, padding='SAME')
								end_points[_maxpool_layer_name] = x
						else:
							x = tcl.conv2d(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], stride=1, 
									scope=_conv_layer_name)
							end_points[_conv_layer_name] = x

				for layer_ind in range(nb_conv_layers[nb_conv_blocks]):
					_conv_layer_name = 'conv_bn%d_%d'%(nb_conv_blocks, layer_ind)
					x = tcl.conv2d(x, nb_conv_filters[nb_conv_blocks], nb_conv_ksize[nb_conv_blocks], stride=1, 
							scope=_conv_layer_name)
					end_points[_conv_layer_name] = x

				for block_ind in range(nb_conv_blocks)[::-1]:
					for layer_ind in range(nb_conv_layers[block_ind]):
						_conv_layer_name = 'conv_us%d_%d'%(block_ind+1, layer_ind)
						if layer_ind == 0:
							x = tcl.conv2d_transpose(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], stride=2, 
										scope=_conv_layer_name)
							end_points[_conv_layer_name] = x
							x = tf.concat([x, block_end[block_ind]], axis=3)
							end_points[_conv_layer_name + '_concat'] = x

						else:
							x = tcl.conv2d(x, nb_conv_filters[block_ind], nb_conv_ksize[block_ind], stride=1, 
									scope=_conv_layer_name)
							end_points[_conv_layer_name] = x





			return x, end_points
