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

from .base_network import BaseNetwork

arg_scope = tf.contrib.framework.arg_scope


class UNet(BaseNetwork):

	def __init__(self, config, is_training):
		BaseNetwork.__init__(self, config, is_training)
		self.config = config
		self.reuse = False

	def __call__(self, x):

		conv_nb_blocks = self.config.get('conv_nb_blocks', 4)
		conv_nb_layers = self.config.get('conv_nb_layers', [2, 2, 2, 2, 2])
		conv_nb_filters = self.config.get('conv_nb_filters', [64, 128, 256, 512, 1024])
		conv_ksize = self.config.get('conv_ksize', [3 for i in range(conv_nb_blocks+1)])
		no_maxpooling = self.config.get('no_maxpooling', False)
		no_upsampling = self.config.get('no_upsampling', False)


		output_dims = self.config.get('output_dims', 0)  # zero for no output layer
		output_act_fn = get_activation(self.config.get('output_activation', 'none'))

		debug = self.config.get('debug', False)

		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

			block_end = {}
			self.end_points = {}

			if debug:
				print('UNet : (' + str(self.name) + ')')
				print('downsapmle network : ')

			for block_ind in range(conv_nb_blocks):
				for layer_ind in range(conv_nb_layers[block_ind]):

					conv_name = 'conv_ds%d_%d'%(block_ind+1, layer_ind)
					maxpool_name = 'maxpool_ds%d'%(block_ind+1)

					if layer_ind == conv_nb_layers[block_ind]-1:
						if no_maxpooling:
							block_end[block_ind] = x
							x = self.conv2d(conv_name, x, conv_nb_filters[block_ind], conv_ksize[block_ind], stride=2, **self.conv_args, disp=debug)
						else:
							x = self.conv2d(conv_name, x, conv_nb_filters[block_ind], conv_ksize[block_ind], stride=1, **self.conv_args, disp=debug)
							block_end[block_ind] = x
							x = self.maxpool2d(maxpool_name, x, 2, stride=2, padding='SAME')
					else:
						x = self.conv2d(conv_name, x, conv_nb_filters[block_ind], conv_ksize[block_ind], stride=1, **self.conv_args, disp=debug)

			if debug:
				print('bottleneck network : ')
			for layer_ind in range(conv_nb_layers[conv_nb_blocks]):
				conv_name = 'conv_bn%d'%(layer_ind)
				x = self.conv2d(conv_name, x, conv_nb_filters[conv_nb_blocks], conv_ksize[conv_nb_blocks], stride=1, **self.conv_args, disp=debug)

			if debug:
				print('upsample network : ')

			for block_ind in range(conv_nb_blocks)[::-1]:
				for layer_ind in range(conv_nb_layers[block_ind]):
					deconv_name = 'deconv_us%d'%block_ind
					conv_name = 'conv_us%d_%d'%(block_ind+1, layer_ind)
					if layer_ind == 0:
						x = self.deconv2d(deconv_name, x, conv_nb_filters[block_ind], conv_ksize[block_ind], stride=2, **self.conv_args, disp=debug)
						x = self.concat(deconv_name + '_concat', [x, block_end[block_ind]])
						x = self.conv2d(conv_name, x, conv_nb_filters[block_ind], conv_ksize[block_ind], stride=1, **self.conv_args, disp=debug)
					else:
						x = self.conv2d(conv_name, x, conv_nb_filters[block_ind], conv_ksize[block_ind], stride=1, **self.conv_args, disp=debug)

			if debug:
				print('output network : ')
			if output_dims != 0:
				x = self.conv2d('conv_out', x, output_dims, 1, stride=1, **self.out_conv_args, disp=debug)

			if debug:
				print('')

			return x, self.end_points

