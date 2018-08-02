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

from utils.learning_rate import get_learning_rate
from utils.learning_rate import get_global_step
from utils.optimizer import get_optimizer
from utils.optimizer import get_optimizer_by_config
from utils.loss import get_loss


from math import sin, cos, sqrt

from .basemodel import BaseModel


class AAE(BaseModel):

	"""
		Implementation of "Adversarial Autoencoders"
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

		self.input_shape = config['input shape']
		self.z_dim = config['z_dim']
		self.nb_classes = config['nb classes']
		self.config = config

		assert('encoder' in self.config)
		assert('decoder' in self.config)
		assert('discriminator' in self.config)

		self.prior_distribution = self.config.get('prior distribution', 'mixGaussian')
		assert(self.prior_distribution in ['mixGaussian', 'swiss_roll', 'normal'])

		self.build_model()
		self.build_summary()


	def sample_prior(self, batch_size):
		assert(self.z_dim == 2)
		if self.prior_distribution == 'mixGaussian':		
			def sample(x, y, label, n_labels):
				shift = 3
				r = 2.0 * np.pi / n_labels * label
				new_x = x * np.cos(r) - y * np.sin(r)
				new_y = x * np.sin(r) + y * np.cos(r)
				new_x += shift * np.cos(r)
				new_y += shift * np.sin(r)
				return new_x, new_y
			x_var = 0.5
			y_var = 0.1
			x = np.random.normal(0, x_var, [batch_size, 1])
			y = np.random.normal(0, y_var, [batch_size, 1])
			label = np.random.randint(0, self.nb_classes, size=[batch_size, 1]).astype(np.float32)
			label_onehot = np.zeros(shape=(batch_size, self.nb_classes)).astype(np.float32)
			for i in range(batch_size):
				label_onehot[i, int(label[i,0])] = 1

			x, y = sample(x, y, label, self.nb_classes)
			return np.concatenate([x, y], axis=1).astype(np.float32), label_onehot
			
	def discriminator_prior(self, z_batch):
		assert(self.z_dim == 2)
		if self.prior_distribution == 'mixGaussian':
			pass

	def build_model(self):
		# network config
		self.config['discriminator params']['name'] = 'Discriminator'
		self.config['encoder params']['name'] = 'Encoder'
		self.config['decoder params']['name'] = 'Decoder'
		self.discriminator = self.build_discriminator('discriminator')
		self.encoder = self.build_encoder('encoder')
		self.decoder = self.build_decoder('decoder')

		# build model
		self.img = tf.placeholder(tf.float32, shape=[None, ] + list(self.input_shape), name='img')
		self.label = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='label')
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')
		self.z_label = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='z_label')

		self.z_sample = self.encoder(self.img)
		
		self.img_recon = self.decoder(self.z_sample)

		self.dis_real = self.discriminator(tf.concat([self.z, self.z_label], axis=1))
		self.dis_fake = self.discriminator(tf.concat([self.z_sample, self.label], axis=1))

		# generate image from z:
		self.img_generate = self.decoder(self.z)

		# loss config
		self.loss_adv_down = get_loss('adversarial down', 'cross entropy', {'dis_real' : self.dis_real, 'dis_fake' : self.dis_fake})
		self.loss_adv_up = get_loss('adversarial up', 'cross entropy', {'dis_fake' : self.dis_fake})
		self.loss_recon = get_loss('reconstruction', 'l2', {'x' : self.img, 'y' : self.img_recon})

		self.ae_loss = self.loss_recon
		self.d_loss = self.loss_adv_down
		self.e_loss = self.loss_adv_up


		# optimizer config
		self.global_step, self.global_step_update = get_global_step()

		# optimizer of discriminator configured without global step update
		# so we can keep the learning rate of discriminator the same as generator
		(self.ae_train_op, 
			self.ae_learning_rate, 
				self.ae_global_step) = get_optimizer_by_config(self.config['auto-encoder optimizer'],
																self.config['auto-encoder optimizer params'],
																self.loss_recon, self.encoder.vars + self.decoder.vars,
																self.global_step)
		(self.d_train_op, 
			self.d_learning_rate, 
				self.d_global_step) = get_optimizer_by_config(self.config['discriminator optimizer'],
																self.config['discriminator optimizer params'],
																self.loss_adv_down, self.discriminator.vars,
																self.global_step)

		(self.e_train_op, 
					self.e_learning_rate, 
						self.e_global_step) = get_optimizer_by_config(self.config['encoder optimizer'],
																		self.config['encoder optimizer params'],
																		self.loss_adv_up, self.encoder.vars,
																		self.global_step)
														
		# model saver
		self.saver = tf.train.Saver(self.discriminator.store_vars 
									+ self.encoder.store_vars
									+ self.decoder.store_vars
									+ [self.global_step])


	def build_summary(self):
		if self.is_summary:
			# summary scalars are logged per step
			sum_list = []
			sum_list.append(tf.summary.scalar('auto-encoder/loss', self.ae_loss))
			sum_list.append(tf.summary.scalar('auto-encoder/lr', self.ae_learning_rate))
			self.ae_sum_scalar = tf.summary.merge(sum_list)

			sum_list = []
			sum_list.append(tf.summary.scalar('discrimintor/loss', self.d_loss))
			sum_list.append(tf.summary.scalar('discrimintor/lr', self.d_learning_rate))
			self.d_sum_scalar = tf.summary.merge(sum_list)

			sum_list = []
			sum_list.append(tf.summary.scalar('encoder/loss', self.e_loss))
			sum_list.append(tf.summary.scalar('encoder/lr', self.e_learning_rate))
			self.e_sum_scalar = tf.summary.merge(sum_list)


			# summary hists are logged by calling self.summary()
			sum_list = []
			sum_list += [tf.summary.histogram('discriminator/'+var.name, var) for var in self.discriminator.vars]
			sum_list += [tf.summary.histogram('encoder/'+var.name, var) for var in self.encoder.vars]
			sum_list += [tf.summary.histogram('decoder/'+var.name, var) for var in self.decoder.vars]
			self.sum_hist = tf.summary.merge(sum_list)
		else:
			self.d_sum_scalar = None
			self.g_sum_scalar = None
			self.sum_hist = None

	
	'''
		train operations
	'''
	def train_on_batch_supervised(self, sess, x_batch, y_batch):
		
		z_batch, z_label_batch = self.sample_prior(x_batch.shape[0])
		summary_list = []


		feed_dict = {
			self.img : x_batch,
			self.is_training : True,
		}
		step_ae, lr_ae, loss_ae, summary_ae = self.train(sess, feed_dict, update_op=self.ae_train_op,
															step=self.ae_global_step,
															learning_rate=self.ae_learning_rate,
															loss=self.ae_loss,
															summary=self.ae_sum_scalar)
		
		summary_list.append((step_ae, summary_ae))

		feed_dict = {
			self.img : x_batch,
			self.label : y_batch,
			self.is_training : True,
			self.z : z_batch, 
			self.z_label : z_label_batch
		}
		step_d, lr_d, loss_d, summary_d = self.train(sess, feed_dict, update_op=self.d_train_op,
															step=self.d_global_step,
															learning_rate=self.d_learning_rate,
															loss=self.d_loss,
															summary=self.d_sum_scalar)
		summary_list.append((step_d, summary_d))
		
		feed_dict = {
			self.img : x_batch,
			self.label : y_batch,
			self.is_training : True,
		}
		step_e, lr_e, loss_e, summary_e = self.train(sess, feed_dict, update_op=self.e_train_op,
															step=self.e_global_step,
															learning_rate=self.e_learning_rate,
															loss=self.e_loss,
															summary=self.e_sum_scalar)

		summary_list.append((step_e, summary_e))


		step, _ = sess.run([self.global_step,self.global_step_update])

		return step, {'ae':lr_ae, 'd':lr_d, 'e':lr_e}, {'ae':loss_ae, 'd':loss_d,'e':loss_e}, summary_list, 



	def train_on_batch_unsupervised(self, sess, x_batch):
		raise NotImplementedError
		

	'''
		test operation
	'''
	def generate(self, sess, z_batch):
		feed_dict = {
			self.z : z_batch,
			self.is_training : False
		}
		x_batch = sess.run([self.img_generate], feed_dict = feed_dict)[0]
		return x_batch

	def hidden_variable_distribution(self, sess, x_batch):
		feed_dict = {
			self.img : x_batch,
			self.is_training : False
		}
		sample_z = sess.run([self.z_sample], feed_dict=feed_dict)[0]
		return sample_z


	'''
		summary operation
	'''
	def summary(self, sess):
		if self.is_summary:
			summ = sess.run(self.sum_hist)
			return summ
		else:
			return None
