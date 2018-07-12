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
sys.path.append('../')

import tensorflow as tf
import tensorflow.contrib.layers as tcl

from utils.weightsinit import get_weightsinit
from utils.activation import get_activation
from utils.normalization import get_normalization


# from network.vgg import VGG
from network.devgg import DEVGG
from network.basenetwork import BaseNetwork

class G_conv(BaseNetwork):

	def __init__(self, config, is_training):
		BaseNetwork.__init__(self, config, is_training)

		self.name = config.get('name', 'GeneratorSimple')
		self.config = config

		network_config = config.copy()
		network_config['name'] = self.name
		self.reuse = False
		self.size = 64//16
		self.channel = 1
		
	def __call__(self, i):
		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True
			g = tcl.fully_connected(i, self.size * self.size * 1024, activation_fn=tf.nn.relu, 
									normalizer_fn=tcl.batch_norm)
			g = tf.reshape(g, (-1, self.size, self.size, 1024))  # size
			g = tcl.conv2d_transpose(g, 512, 3, stride=2, # size*2
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.conv2d_transpose(g, 256, 3, stride=2, # size*4
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.conv2d_transpose(g, 128, 3, stride=2, # size*8
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			
			g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16
										activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			return g

		return x


