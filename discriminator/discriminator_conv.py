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


from network.base_network import BaseNetwork



def lrelu(x, leak=0.1, name="lrelu"):
	# r = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
	# x >= 0 : r = x
	# x < 0  : r = leak * x

	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)



class D_conv(BaseNetwork):

	def __init__(self, config, is_training):
		BaseNetwork.__init__(self, config, is_training)
		self.name = config.get('name', 'D_conv')
		self.config = config
		self.reuse = False
		
	def __call__(self, x):
		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

			end_points = {}
			
			size = 64
			shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
						weights_initializer=tf.random_normal_initializer(0, 0.02), 
						stride=2, activation_fn=lrelu)
			end_points['conv1'] = shared
			shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
						weights_initializer=tf.random_normal_initializer(0, 0.02), 
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			end_points['conv2'] = shared
			shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
						weights_initializer=tf.random_normal_initializer(0, 0.02), 
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			end_points['conv3'] = shared
			shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=4, # 4x4x512
						weights_initializer=tf.random_normal_initializer(0, 0.02), 
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			end_points['conv4'] = shared
			shared = tcl.flatten(shared)
			d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
			return d

	def features(self, i, condition=None):
		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

			end_points = {}
			
			size = 64
			shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
						weights_initializer=tf.random_normal_initializer(0, 0.02), 
						stride=2, activation_fn=lrelu)
			end_points['conv1'] = shared
			
			shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
						weights_initializer=tf.random_normal_initializer(0, 0.02), 
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			end_points['conv2'] = shared
			shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
						weights_initializer=tf.random_normal_initializer(0, 0.02), 
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			end_points['conv3'] = shared
			shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=4, # 4x4x512
						weights_initializer=tf.random_normal_initializer(0, 0.02), 
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			end_points['conv4'] = shared
			shared = tcl.flatten(shared)
			d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
			return d, end_points


