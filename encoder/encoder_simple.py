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


from utils.weightsinit import get_weightsinit
from utils.activation import get_activation
from utils.normalization import get_normalization


from network.vgg import VGG
from network.base_network import BaseNetwork

class EncoderSimple(BaseNetwork):

	def __init__(self, config, is_training):

		super(EncoderSimple, self).__init__(config, is_training)
		
		self.name = config.get('name', 'EncoderSimple')
		self.config = config

		network_config = config.copy()
		network_config['name'] = self.name
		network_config['output_dims'] = 0

		self.network = VGG(network_config, is_training)
		self.output_distribution = self.config.get('output_distribution', 'gaussian')
		self.reuse=False
		
	def __call__(self, i, condition=None):

		output_dims = self.config.get('output_dims', 3)
		output_act_fn = get_activation(self.config.get('output_activation', 'none'))


		x, end_points = self.network(i)

		x = tcl.flatten(x)
		if condition is not None:
			x = tf.concatenate([x, condition], axis=-1)

		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse=True

			if self.output_distribution == 'gaussian':
				mean = self.fc('fc_out_mean', x, output_dims, **self.out_fc_args)
				log_var = self.fc('fc_out_log_var', x, output_dims, **self.out_fc_args)
				return mean, log_var

			elif self.output_distribution == 'mean':
				mean = self.fc('fc_out_mean', x, output_dims, **self.out_fc_args)
				return mean

			elif self.output_distribution == 'none':
				out = self.fc('fc_out_mean', x, output_dims, **self.out_fc_args)
				return out
			else:
				raise Exception("None output distribution named " + self.output_distribution)


