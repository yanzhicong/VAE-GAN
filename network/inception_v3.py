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

from __future__ import absolute_import

import os
import sys

sys.path.append('.')
sys.path.append('../')



import tensorflow as tf
import tensorflow.contrib.layers as tcl

from .weightsinit import get_weightsinit
from .activation import get_activation
from .normalization import get_normalization

from .inception_block import inception_v3_figure4
from .inception_block import inception_v3_figure5
from .inception_block import inception_v3_figure5_downsample
from .inception_block import inception_v3_figure6
from .inception_block import inception_v3_figure6_downsample
from .inception_block import inception_v3_figure7



class InceptionV3(object):
	"""Inception model from http://arxiv.org/abs/1512.00567.

	"Rethinking the Inception Architecture for Computer Vision"
	Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
	Zbigniew Wojna.
	"""

	def __init__(self, config, model_config, is_training, name="InceptionV3"):
		self.name = name
		self.training = is_training
		self.normalizer_params = {
			'decay' : 0.999,
			'center' : True,
			'scale' : False,
			'is_training' : self.training
		}

		self.config = config
		self.model_config = model_config


	def __call__(self, i, reuse=False):

		act_fn = get_activation(
					self.config.get('activation', 'relu'),
					self.config.get('activation_params', {}))

		norm_fn, norm_params = get_normalization(
					self.config.get('batch_norm', 'batch_norm'),
					self.config.get('batch_norm_params', self.normalizer_params))

		winit_fn = get_weightsinit(
					self.config.get('weightsinit', 'normal'),
					self.config.get('weightsinit_params', '0.00 0.02'))

		filters = self.config.get('nb_filters', 32)

		# fully connected parameters
		including_top = self.config.get('including_top', True)
		nb_fc_nodes = self.config.get('nb_fc_nodes', [1024, 1024])

		# output stage parameters
		output_dims = self.config.get('output_dims', 0)  # zero for no output layer
		output_act_fn = get_activation(
					self.config.get('output_activation', 'none'),
					self.config.get('output_activation_params', ''))

		with tf.variable_scope(self.name):
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False

			end_points = {}

			# x : 299 * 299 * 3
			x = tcl.conv2d(i, filters, 3,
							stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='VALID', weights_initializer=winit_fn, scope='conv1')
			end_points['conv1'] = x
			
			# x : 149 * 149 * 32 
			x = tcl.conv2d(x, filters, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='VALID', weights_initializer=winit_fn, scope='conv2')
			end_points['conv2'] = x

			# x : 147 * 147 * 32
			x = tcl.conv2d(x, 2*filters, 3,
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv3')
			end_points['conv3'] = x


			# x : 147 * 147 * 64
			x = tcl.max_pool2d(x, 3, stride=2, 
							padding='VALID', scope='pool1')
			end_points['pool1'] = x

			# x : 73 * 73 * 64
			x = tcl.conv2d(x, 80, 3, 
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='VALID', weights_initializer=winit_fn, scope='conv4')
			end_points['conv4'] = x


			# x : 71 * 71 * 80
			x = tcl.conv2d(x, 192, 3, 
							stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='VALID', weights_initializer=winit_fn, scope='conv5')
			end_points['conv5'] = x


			# x : 35 * 35 * 192
			x = tcl.conv2d(x, 288, 3, 
							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
							padding='SAME', weights_initializer=winit_fn, scope='conv6')
			end_points['conv6'] = x

			# x : 35 * 35 * 288
			x, end_points = inception_v3_figure5('inception1a', x, end_points, 
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure5('inception1b', x, end_points, 
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure5_downsample('inception1c', x, end_points, 
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)


			# x : 17 * 17 * 768
			x, end_points = inception_v3_figure6('inception2a', x, end_points, n=7,
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure6('inception2b', x, end_points, n=7,
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure6('inception2c', x, end_points, n=7,
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure6('inception2d', x, end_points, n=7,
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure6_downsample('inception2e', x, end_points, n=7,
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
  

			# x : 8 * 8 * 1280
			x, end_points = inception_v3_figure7('inception3a', x, end_points, 
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)
			x, end_points = inception_v3_figure7('inception3b', x, end_points, 
						act_fn=act_fn, norm_fn=norm_fn, norm_params=norm_params, winit_fn=winit_fn)

			# construct top fully connected layer
			if including_top: 
				with tf.variable_scope("logits"):  
					x = tcl.avg_pool2d(x,kernel_size=[8,8],padding="VALID",  
										  scope="avgpool_1a_8x8")  
					x = tcl.dropout(x,keep_prob=0.5,scope="dropout_1b")  
					end_points["global_avg_pooling"] = x  

					x = tcl.flatten(x)


					if output_dims != 0:
						x = tcl.fully_connected(x, output_dims, activation_fn=output_act_fn, normalizer_fn=None,
								weights_initializer=winit_fn, scope='fc_out')
						end_points['fc_out'] = x
	
			return x, end_points

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)



