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


import tensorflow as tf
import tensorflow.contrib.layers as tcl


def get_weightsinit(name_config):

	split = name_config.split()
	name = split[0]
	if len(split) > 1:
		params = split[1]

	if name == 'normal': 
		if len(split) == 3:
			init_mean = float(split[1])
			init_var = float(split[2])
		else:
			init_mean = 0.0
			init_var = 0.02
		return tf.random_normal_initializer(init_mean, init_var)

	elif name == 'xvarial':
		return None

	elif name == 'zeros':
		return tf.zeros_initializer()
		
	elif name == 'ones':
		return tf.ones_initializer()

	else :
		raise Exception("None weights initializer named " + name)

