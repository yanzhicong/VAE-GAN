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
from .base_network import BaseNetwork

class Resnet(BaseNetwork):

	def __init__(self, config, is_training):
		BaseNetwork.__init__(self, config, is_training)
		self.config = config
		self.reuse = False
		self.architecture = self.config.get('architecture', 'resnet50')
		self.debug = self.config.get('debug', False)

		assert self.architecture in ['resnet50', 'resnet101']


	def identity_block(self, input_tensor, kernel_size, filters, stage, block):
		"""The identity_block is the block that has no conv layer at shortcut
		# Arguments
			input_tensor: input tensor
			kernel_size: default 3, the kernel size of middle conv layer at main path
			filters: list of integers, the nb_filters of 3 conv layer at main path
			stage: integer, current stage label, used for generating layer names
			block: 'a','b'..., current block label, used for generating layer names
		"""
		nb_filter1, nb_filter2, nb_filter3 = filters
		conv_name_base = 'res' + str(stage) + block + '_branch'

		x = self.conv2d(conv_name_base+'2a', input_tensor, 
						nb_filter1, 1, stride=1, 
						norm_fn='fused_batch_norm', norm_params=self.norm_params,
						act_fn='relu', disp=self.debug)

		x = self.conv2d(conv_name_base+'2b', x, 
						nb_filter2, kernel_size, stride=1, 
						norm_fn='fused_batch_norm', norm_params=self.norm_params,
						act_fn='relu', disp=self.debug)
		
		x = self.conv2d(conv_name_base+'2c', x, 
						nb_filter3, kernel_size, stride=1, 
						norm_fn='fused_batch_norm', norm_params=self.norm_params,
						act_fn='none', disp=self.debug)
				
		x = tf.add(x, input_tensor)
		x = self.activation(x, 'relu')
		return x

	def conv_block(self, input_tensor, kernel_size, filters, stage, block,
				stride=(2, 2), 
				# use_bias=True, train_bn=True
				):
		"""conv_block is the block that has a conv layer at shortcut
		# Arguments
			input_tensor: input tensor
			kernel_size: default 3, the kernel size of middle conv layer at main path
			filters: list of integers, the nb_filters of 3 conv layer at main path
			stage: integer, current stage label, used for generating layer names
			block: 'a','b'..., current block label, used for generating layer names
		Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
		And the shortcut should have subsample=(2,2) as well
		"""
		nb_filter1, nb_filter2, nb_filter3 = filters
		conv_name_base = 'res' + str(stage) + block + '_branch'

		x = self.conv2d(conv_name_base+'2a', input_tensor, 
						nb_filter1, 1, stride=stride, 
						norm_fn='fused_batch_norm', norm_params=self.norm_params,
						act_fn='relu', disp=self.debug)

		x = self.conv2d(conv_name_base+'2b', x, 
						nb_filter2, kernel_size, stride=1, 
						norm_fn='fused_batch_norm', norm_params=self.norm_params,
						act_fn='relu', disp=self.debug)
		
		x = self.conv2d(conv_name_base+'2c', x, 
						nb_filter3, kernel_size, stride=1, 
						norm_fn='fused_batch_norm', norm_params=self.norm_params,
						act_fn='none', disp=self.debug)

		shortcut = self.conv2d(conv_name_base+'1', input_tensor, 
						nb_filter3, 1, stride=stride, 
						norm_fn='fused_batch_norm', norm_params=self.norm_params,
						act_fn='none', disp=self.debug)
				
		x = tf.add(x, shortcut)
		x = self.activation(x, 'relu')
		return x


	def __call__(self, x):

		# fully connected parameters
		including_top = self.config.get("including top", True)
		fc_nb_nodes = self.config.get("fc nb nodes", [1024, 1024])

		# output stage parameters
		output_dims = self.config.get("output dims", 0)  # zero for no output layer
		output_act_fn = self.config.get('output_activation', 'none')

		self.end_points = {}
		with tf.variable_scope(self.name):
			if self.reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse is False
				self.reuse = True

			if self.debug:
				print('Resnet : (' + self.name + ')')

			# Stage 1
			x = self.conv2d('conv1', x, 64, 7, stride=2, 
						norm_fn='fused_batch_norm', norm_params=self.norm_params,
						act_fn='relu', disp=self.debug)
			x = self.maxpool2d('pool1', x, 3, stride=2, disp=self.debug)

			self.end_points['stage1'] = x

			# Stage 2
			x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', stride=(1, 1))
			x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
			x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

			self.end_points['stage2'] = x
			# Stage 3
			x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
			x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
			x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
			x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

			self.end_points['stage3'] = x
			# Stage 4
			x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
			block_count = {"resnet50": 5, "resnet101": 22}[self.architecture]
			for i in range(block_count):
				x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
			self.end_points['stage4'] = x

			x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
			x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
			x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

			self.end_points['stage5'] = x


			if self.debug:
				print('\tincluding_top :            ', self.config.get("including top", ''))
				print('\tfc network :')

			# construct top fully connected layer
			if including_top: 
				x = tcl.flatten(x)
				for ind, nb_nodes in enumerate(fc_nb_nodes):
					x = self.fc('fc%d'%(ind+1), x, nb_nodes, **self.fc_args, disp=self.debug)

			if self.debug:
				print('\toutput_dims :              ', self.config.get("output dims", ''))
				print('\toutput network : ')

			if output_dims != 0:
				# construct a fc layer for output
				if including_top: 
					x = self.fc('fc_out', x, output_dims, **self.out_fc_args, disp=self.debug)

				# else construct a convolution layer for output
				else:
					x = self.conv2d('conv_out', x, output_dims, 1, stride=1, **self.out_fc_args, disp=self.debug)

			return x, self.end_points
