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


class DiscriminatorCifar10(object):

	def __init__(self, config, is_training):
		self.name = config.get('name', 'DiscriminatorCifar10')
		self.config = config
		self.reuse = False
		self.is_training = is_training
		
		self.normalizer_params = {
			'decay' : 0.999,
			'center' : True,
			'scale' : False,
			'is_training' : self.is_training
		}


	def __call__(self, x):
	
		act_fn = get_activation('lrelu 0.2')
		norm_fn, norm_params = get_normalization('batch_norm', self.normalizer_params)
		winit_fn = get_weightsinit('normal 0.00 0.02')
		binit_fn = get_weightsinit('zeros')


		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True


			x = tcl.conv2d(x, 128, 5, stride=2, 
										activation_fn=act_fn, 
										normalizer_fn=norm_fn, 
										normalizer_params=norm_params,
										padding='SAME', 
										weights_initializer=winit_fn, 
										biases_initializer=binit_fn, scope='conv0')
			
			x = tcl.conv2d(x, 256, 5, stride=2, 
										activation_fn=act_fn, 
										normalizer_fn=norm_fn, 
										normalizer_params=norm_params,
										padding='SAME', 
										weights_initializer=winit_fn, 
										biases_initializer=binit_fn, scope='conv1')

			x = tcl.conv2d(x, 512, 5, stride=2, 
										activation_fn=act_fn, 
										normalizer_fn=norm_fn, 
										normalizer_params=norm_params,
										padding='SAME', 
										weights_initializer=winit_fn, 
										biases_initializer=binit_fn, scope='conv2')

			x = tcl.flatten(x)

			x = tcl.fully_connected(x, 512, 
										activation_fn=act_fn, 
										normalizer_fn=norm_fn, 
										normalizer_params=norm_params,
										weights_initializer=winit_fn, 
										biases_initializer=binit_fn, scope='fc0')

			x = tcl.fully_connected(x, 1, 
										activation_fn=None, 
										normalizer_fn=None, 
										normalizer_params=None,
										weights_initializer=winit_fn, 
										biases_initializer=binit_fn, scope='fc1')

		return x

	def features(self, i, condition=None):
		return NotImplementedError


	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


