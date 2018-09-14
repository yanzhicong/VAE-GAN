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
sys.path.append('.')
sys.path.append("../")

import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np

from math import sin, cos, sqrt

from utils.sample import get_sampler
from utils.learning_rate import get_global_step
from utils.loss import get_loss

from .base_model import BaseModel


class AAE(BaseModel):
	""" Implementation of "Adversarial Autoencoders"
		Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey

		@article{DBLP:journals/corr/MakhzaniSJG15,
				author    = {Alireza Makhzani and
										Jonathon Shlens and
										Navdeep Jaitly and
										Ian J. Goodfellow},
				title     = {Adversarial Autoencoders},
				journal   = {CoRR},
				volume    = {abs/1511.05644},
				year      = {2015},
				url       = {http://arxiv.org/abs/1511.05644},
				archivePrefix = {arXiv},
				eprint    = {1511.05644},
				timestamp = {Wed, 07 Jun 2017 14:42:14 +0200},
				biburl    = {https://dblp.org/rec/bib/journals/corr/MakhzaniSJG15},
				bibsource = {dblp computer science bibliography, https://dblp.org}
		}
	"""

	def __init__(self, config,
				 **kwargs
				 ):

		super(AAE, self).__init__(config, **kwargs)
		self.config = config

		self.input_shape = self.config['input shape']
		self.z_dim = self.config['z_dim']
		self.nb_classes = self.config['nb classes']
		self.has_label = self.config.get('has label', True)

		assert('encoder' in self.config)
		assert('decoder' in self.config)
		assert('discriminator' in self.config)

		self.prior_distribution = self.config.get('prior distribution', 'mixGaussian')
		assert(self.prior_distribution in ['mixGaussian', 'swiss_roll', 'normal'])

		# self.z_sampler = get_sampler()
		if self.prior_distribution == 'mixGaussian':
			self.z_sampler = get_sampler(self.prior_distribution, z_dim=self.z_dim, nb_classes=self.nb_classes)
		elif self.prior_distribution == 'normal':
			self.z_sampler = get_sampler(self.prior_distribution, z_dim=self.z_dim)

		self.build_model()
		self.build_summary()


	def build_model(self):
		# network config
		self.discriminator = self.build_discriminator('discriminator', params={
			'name':'Discriminator',
			'output_dims':1,
			'output_activation':'none'})
		self.encoder = self.build_encoder('encoder', params={
			'name':'Encoder',
			'output_dims':self.z_dim})
		self.decoder = self.build_decoder('decoder', params={'name':'Decoder'})

		# build model
		self.img = tf.placeholder(tf.float32, shape=[None, ] + list(self.input_shape), name='img')
		if self.has_label:
			self.label = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='label')
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')
		if self.has_label:
			self.z_label = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='z_label')

		self.z_sample = self.encoder(self.img)

		self.img_recon = self.decoder(self.z_sample)

		if self.has_label:
			self.dis_real = self.discriminator(tf.concat([self.z, self.z_label], axis=1))
			self.dis_fake = self.discriminator(tf.concat([self.z_sample, self.label], axis=1))
		else:
			self.dis_real = self.discriminator(self.z)
			self.dis_fake = self.discriminator(self.z_sample)

		# generate image from z:
		self.img_generate = self.decoder(self.z)

		# loss config
		self.loss_adv_down = get_loss('adversarial down', 'cross entropy', {'dis_real': self.dis_real, 'dis_fake': self.dis_fake})
		self.loss_adv_up = get_loss('adversarial up', 'cross entropy', {'dis_fake': self.dis_fake})
		self.loss_recon = get_loss('reconstruction', 'l2', {'x': self.img, 'y': self.img_recon})

		# optimizer config
		self.global_step, self.global_step_update = self.build_step_var('global_step')

		self.train_auto_encoder, _ = self.build_train_function('auto-encoder', 
												self.loss_recon, self.encoder.vars + self.decoder.vars, 
												step=self.global_step, build_summary=self.has_summary)
		 
		self.train_discriminator, _ = self.build_train_function('discriminator', 
												self.loss_adv_down, self.discriminator.vars,
												step=self.global_step, build_summary=self.has_summary)

		self.train_encoder, _ = self.build_train_function('encoder', 
												self.loss_adv_up, self.encoder.vars,
												step=self.global_step, build_summary=self.has_summary)

		# model saver
		self.saver = tf.train.Saver(self.discriminator.store_vars
									+ self.encoder.store_vars
									+ self.decoder.store_vars
									+ [self.global_step])

	def build_summary(self):
		if self.has_summary:
			sum_list = []
			sum_list += [tf.summary.histogram('discriminator/'+var.name, var) for var in self.discriminator.vars]
			sum_list += [tf.summary.histogram('encoder/'+var.name, var) for var in self.encoder.vars]
			sum_list += [tf.summary.histogram('decoder/'+var.name, var) for var in self.decoder.vars]
			self.sum_hist = tf.summary.merge(sum_list)
		else:
			self.sum_hist = None

	#
	#	train operations
	#
	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		assert self.has_label == True

		z_batch, z_label_batch = self.z_sampler(x_batch.shape[0])
		summary_list = []

		feed_dict = {
			self.img: x_batch,
			self.is_training: True,
		}
		step_ae, lr_ae, loss_ae, summary_ae = self.train_auto_encoder(sess, feed_dict)
		summary_list.append((step_ae, summary_ae))

		feed_dict = {
			self.img: x_batch,
			self.label: y_batch,
			self.is_training: True,
			self.z: z_batch,
			self.z_label: z_label_batch
		}
		step_d, lr_d, loss_d, summary_d = self.train_discriminator(sess, feed_dict)
		summary_list.append((step_d, summary_d))

		feed_dict = {
			self.img: x_batch,
			self.label: y_batch,
			self.is_training: True,
		}
		step_e, lr_e, loss_e, summary_e = self.train_encoder(sess, feed_dict)
		summary_list.append((step_e, summary_e))

		step, _ = sess.run([self.global_step, self.global_step_update])
		return step, "[ae:%0.6f, d:%0.6f, e:%0.6f]"%(lr_ae, lr_d, lr_e), "[ae:%0.4f, d:%0.4f, e:%0.4f]"%(loss_ae, loss_d, loss_e), summary_list
		

	def train_on_batch_unsupervised(self, sess, x_batch):
		assert self.has_label == False

		z_batch = self.z_sampler(x_batch.shape[0])
		summary_list = []

		feed_dict = {
			self.img: x_batch,
			self.is_training: True,
		}
		step_ae, lr_ae, loss_ae, summary_ae = self.train_auto_encoder(sess, feed_dict)
		summary_list.append((step_ae, summary_ae))

		feed_dict = {
			self.img: x_batch,
			self.z: z_batch,
			self.is_training: True,
		}
		step_d, lr_d, loss_d, summary_d = self.train_discriminator(sess, feed_dict)
		summary_list.append((step_d, summary_d))

		feed_dict = {
			self.img: x_batch,
			self.is_training: True,
		}
		step_e, lr_e, loss_e, summary_e = self.train_encoder(sess, feed_dict)
		summary_list.append((step_e, summary_e))

		step, _ = sess.run([self.global_step, self.global_step_update])

		return step, "[ae:%0.6f, d:%0.6f, e:%0.6f]"%(lr_ae, lr_d, lr_e), "[ae:%0.4f, d:%0.4f, e:%0.4f]"%(loss_ae, loss_d, loss_e), summary_list

	#
	#	test operations
	#
	def generate(self, sess, z_batch):
		feed_dict = {
			self.z: z_batch,
			self.is_training: False
		}
		x_batch = sess.run([self.img_generate], feed_dict=feed_dict)[0]
		return x_batch

	def hidden_variable(self, sess, x_batch):
		feed_dict = {
			self.img: x_batch,
			self.is_training: False
		}
		sample_z = sess.run([self.z_sample], feed_dict=feed_dict)[0]
		return sample_z

	#
	#	summary operations
	#
	def summary(self, sess):
		if self.has_summary:
			summ = sess.run(self.sum_hist)
			return summ
		else:
			return None
