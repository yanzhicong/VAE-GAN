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



class BaseNetwork(object):

	def __init__(self, config, is_training):
		self.name = config['name']
		self.is_training = is_training
		self.batch_norm_params_collection = 'BATCH_NORM_MOVING_VARS'

		self.norm_params = {
			'is_training' : self.is_training,
			'moving_vars_collection' : self.batch_norm_params_collection
		}
		self.config = config

		# when first applying the network to input tensor, the reuse is false
		self.reuse = False


	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

	@property
	def vars_to_save_and_restore(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name) + tf.get_collection(self.batch_norm_params_collection, scope=self.name)

	@property
	def moving_vars(self):
		return tf.get_collection(self.batch_norm_params_collection, scope=self.name)


	@property
	def all_vars(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

