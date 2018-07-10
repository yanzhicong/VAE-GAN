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

def get_learning_rate(name, initial_learning_rate, global_step, config):

	initial_learning_rate = float(initial_learning_rate)

	if name == 'constant':
		return tf.constant(initial_learning_rate)

	elif name == 'exponential':
		return tf.train.exponential_decay(initial_learning_rate, global_step, 
					decay_steps=config['decay_steps'],
					decay_rate=config['decay_rate'],
					staircase=config.get('staircase', True))

	elif name == 'piecewise':
		'''
			config parameters:
			e.g.	
				boundaries: [10000, 30000]
				values : [1.0, 0.5, 0.1]
		'''
		return tf.train.piecewise_constant(global_step, 
				boundaries=config['boundaries'],
				values=[value * initial_learning_rate for value in config['values']]
			)
	else:
		raise Exception('None learning rate scheme named ' + name)


def get_global_step(name='global_step'):
	global_step = tf.Variable(0, trainable=False, name=name)
	global_step_update = tf.assign(global_step, global_step+1)
	return global_step, global_step_update

