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

from netutils.weightsinit import get_weightsinit
from netutils.activation import get_activation
from netutils.normalization import get_normalization

from network.base_network import BaseNetwork

class ClassifierSimple(BaseNetwork):
	def __init__(self, config, is_training):
		BaseNetwork.__init__(self, config, is_training)

		self.base_network = config.get('base network', 'vgg')
		if self.base_network == 'vgg':
			from network.vgg import VGG
			self.network = VGG(config, is_training)
		elif self.base_network == 'resnet':
			from network.resnet import Resnet
			self.network = Resnet(config, is_training)
		else:
			raise ValueError("no base network named " + self.base_network )
		
	def __call__(self, i):
		x, end_points = self.network(i)
		return x

	def features(self, i):
		x, end_points = self.network(i)
		return x, end_points

	def load_pretrained_weights(self, sess):
		return self.network.load_pretrained_weights(sess)
	