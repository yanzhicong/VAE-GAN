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

from netutils.weightsinit import get_weightsinit
from netutils.activation import get_activation
from netutils.normalization import get_normalization


# from network.vgg import VGG
from network.devgg import DEVGG
from network.base_network import BaseNetwork

class GeneratorSimple(BaseNetwork):

	def __init__(self, config, is_training):
		BaseNetwork.__init__(self, config, is_training)

		self.name = config.get('name', 'GeneratorSimple')
		self.config = config

		network_config = config.copy()
		network_config['name'] = self.name
		self.network = DEVGG(network_config, is_training)

		self.reuse=False
		
	def __call__(self, i):
		x, end_points = self.network(i)
		return x

	# @property
	# def vars(self):
	# 	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


