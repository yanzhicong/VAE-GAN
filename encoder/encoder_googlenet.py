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


# class EncoderGoogleNet(object):

# 	def __init__(self, config, model_config, name="EncoderGoogleNet"):
# 		self.name = name
# 		self.training = model_config["is_training"]
# 		self.normalizer_params = {
# 			'decay' : 0.999,
# 			'center' : True,
# 			'scale' : False,
# 			'is_training' : self.training
# 		}
# 		self.config = config
# 		self.model_config = model_config


#     def inception_layer(inputs,
#                     conv_11_size,
#                     conv_33_reduce_size, conv_33_size,
#                     conv_55_reduce_size, conv_55_size,
#                     pool_size,
#                     data_dict={},
#                     trainable=False,
#                     name='inception'):

#         # arg_scope = tf.contrib.framework.arg_scope
#         # with arg_scope([conv], nl=tf.nn.relu, trainable=trainable,
#         #             data_dict=data_dict):
#             conv_11 = tcl.conv2d(inputs, 1, conv_11_size, '{}_1x1'.format(name))

#             conv_33_reduce = conv(inputs, 1, conv_33_reduce_size,
#                                 '{}_3x3_reduce'.format(name))
#             conv_33 = conv(conv_33_reduce, 3, conv_33_size, '{}_3x3'.format(name))

#             conv_55_reduce = conv(inputs, 1, conv_55_reduce_size,
#                                 '{}_5x5_reduce'.format(name))
#             conv_55 = conv(conv_55_reduce, 5, conv_55_size, '{}_5x5'.format(name))

#             pool = max_pool(inputs, '{}_pool'.format(name), stride=1,
#                             padding='SAME', filter_size=3)
#             convpool = conv(pool, 1, pool_size, '{}_pool_proj'.format(name))

#         return tf.concat([conv_11, conv_33, conv_55, convpool],
#                         3, name='{}_concat'.format(name))

# 	def __call__(self, x, c, reuse=False):

# 		if 'activation' in self.config:
# 			act_fn = get_activation(self.config['activation'], self.config['activation_params'])
# 		elif 'activation' in self.model_config:
# 			act_fn = get_activation(self.model_config['activation'], self.model_config['activation_params'])
# 		else:
# 			act_fn = get_activation('lrelu', '0.2')

# 		if 'batch_norm' in self.config:
# 			norm_fn, norm_params = get_normalization(self.config['batch_norm'])
# 		elif 'batch_norm' in self.model_config:
# 			norm_fn, norm_params = get_normalization(self.model_config['batch_norm'])
# 		else:
# 			norm_fn = tcl.batch_norm

# 		if 'weightsinit' in self.config:
# 			winit_fn = get_weightsinit(self.config['weightsinit'], self.config['weightsinit_params'])
# 		elif 'weightsinit' in self.model_config:
# 			winit_fn = get_weightsinit(self.model_config['weightsinit'], self.config['weightsinit_params'])
# 		else:
# 			winit_fn = tf.random_normal_initializer(0, 0.02)

# 		if 'nb_filters' in self.config: 
# 			filters = int(self.config['nb_filters'])
# 		else:
# 			filters = 64

# 		if 'out_activation' in self.config:
# 			out_act_fn = get_activation(self.config['out_activation'], self.config['out_activation_params'])
# 		else:
# 			out_act_fn = None

# 		output_classes = self.config['output_classes']

# 		with tf.variable_scope(self.name):
# 			if reuse:
# 				tf.get_variable_scope().reuse_variables()
# 			else:
# 				assert tf.get_variable_scope().reuse is False

# 			x = tcl.conv2d(x, filters, 3,
# 							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
# 							padding='SAME', weights_initializer=winit_fn, scope='conv1_0')
# 			x = tcl.conv2d(x, filters, 3,
# 							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
# 							padding='SAME', weights_initializer=winit_fn, scope='conv1_1')

# 			x = tcl.conv2d(x, filters*2, 3,
# 							stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
# 							padding='SAME', weights_initializer=winit_fn, scope='conv2_0')               
# 			x = tcl.conv2d(x, filters*2, 3,
# 							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
# 							padding='SAME', weights_initializer=winit_fn, scope='conv2_1')

# 			x = tcl.conv2d(x, filters*4, 3,
# 							stride=2, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
# 							padding='SAME', weights_initializer=winit_fn, scope='conv3_0')               
# 			x = tcl.conv2d(x, filters*4, 3,
# 							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
# 							padding='SAME', weights_initializer=winit_fn, scope='conv3_1')
# 			x = tcl.conv2d(x, filters*4, 3,
# 							stride=1, activation_fn=act_fn, normalizer_fn=norm_fn, normalizer_params=norm_params,
# 							padding='SAME', weights_initializer=winit_fn, scope='conv3_2')
# 			x = tcl.conv2d(x, filters*4, 3,
# 							stride=1, activation_fn=act_fn,  normalizer_fn=norm_fn, normalizer_params=norm_params,
# 							padding='SAME', weights_initializer=winit_fn, scope='conv3_3')
# 			x = tcl.conv2d(x, output_classes, 1,
# 							stride=1, activation_fn=out_act_fn,
# 							padding='SAME', weights_initializer=winit_fn, scope='conv_out')
# 			return x

# 	@property
# 	def vars(self):
# 		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


