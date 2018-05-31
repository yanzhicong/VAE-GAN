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


import tensorflow as tf
import tensorflow.contrib.layers as tcl


sys.path.append('../')



from network.weightsinit import get_weightsinit
from network.activation import get_activation
from network.normalization import get_normalization


from network.vgg import VGG16


class EncoderSimple(object):

	def __init__(self, config, model_config, is_training, name="EncoderSimple"):
		self.name = name
		self.config = config
		self.model_config = model_config


		network_config = config.copy()
		network_config['output_dims'] = 0
		self.network = VGG16(network_config, model_config, is_training, name=self.name)

		self.output_distribution = self.config.get('output_distribution', 'gaussian')
		

	def __call__(self, i, reuse=False):

		output_act_fn = get_activation(
						self.config.get('output_activation', 'none'),
						self.config.get('output_activation_params', ''))

		winit_fn = get_weightsinit(
						self.config.get('weightsinit', 'normal'), 
						self.config.get('weightsinit_params', '0.00 0.02'))

		output_dims = self.config.get('output_dims', 3)

	
		x, end_points = self.network(i)

		with tf.variable_scope(self.name):
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False


			if self.output_distribution == 'gaussian':
				z_mean = tcl.fully_connected(x, output_dims, activation_fn=output_act_fn,
							weights_initializer=winit_fn, scope='efc_mean')
				z_log_var = tcl.fully_connected(x, output_dims, activation_fn=output_act_fn,
							weights_initializer=winit_fn, scope='efc_log_var')
				return z_mean, z_log_var
			elif self.output_distribution == 'mean':
				z_mean = tcl.fully_connected(x, output_dims, activation_fn=output_act_fn,
							weights_initializer=winit_fn, scope='efc_mean')
				return z_mean
			else:
				raise Exception("None output distribution named " + self.output_distribution)




	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


