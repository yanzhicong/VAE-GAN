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
sys.path.append("../")

import tensorflow as tf
import tensorflow.contrib.layers as tcl

# from encoder.encoder import get_encoder
# from decoder.decoder import get_decoder
# from classifier.classifier import get_classifier
# from discriminator.discriminator import get_discriminator

# from utils.sample import get_sample
# from utils.loss import get_loss

from .base_model import BaseModel


class StarGAN(BaseModel):
	def __init__(self, config):
		
		super(StarGAN, self).__init__(config)

		self.config = config
		self.input_shape = self.config['input shape']

		self.nb_classes = self.config['nb_classes']
		self.adv_type = self.config.get('adv_type', 'wgan')

		self.build_model()
		self.build_summary()


	def build_model(self):
		raise NotImplementedError
		# self.real_img = tf.placeholder(tf.float32, shape=[None, ] + self.input_shape, name='real_img')
		# self.real_img_attr = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='real_img_attr')

		# self.fake_img_attr = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='fake_img_attr')
		# self.epsilon = tf.placeholder(tf.float32, [None, 1, 1, 1], name = 'gp_random_num')
	
		# self.generator = get_encoder(self.config['generator'], self.config['generator params'], self.is_trainings)
		# self.discriminator = get_decoder(self.config['discriminator'], self.config['discriminator params'], self.is_training)


		# real_img_with_fake_attr = tf.concat(
		# 	self.real_img,
		# 	tf.tile(
		# 		tf.reshape(self.fake_img_attr, [-1, 1, 1, self.nb_classes]),
		# 		[1, self.input_shape[0], self.input_shape[1], 1]
		# 	), axis=3
		# )

		# self.fake_img = self.generator(real_img_with_fake_attr)
		
		# fake_img_with_real_attr = tf.concat(
		# 	self.fake_img,
		# 	tf.tile(
		# 		tf.reshape(self.real_img_attr, [-1, 1, 1, self.nb_classes]),
		# 		[1, self.input_shape[0], self.input_shape[1], 1]
		# 	), axis=3			
		# )

		# self.recon_img = self.generator(fake_img_with_real_attr)

		# self.dis_real_img, self.cls_real_img = self.discriminator(self.real_img)
		# self.dis_fake_img, self.cls_fake_img = self.discriminator(self.fake_img)

		# # discriminator loss
		# if self.adv_type == 'wgan':
		# 	self.d_gp_loss = (get_loss('gradient penalty', self.config.get('gradient penalty', 'l2'),
		# 				{}) 
		# 				* self.config.get('gradient penalty loss weight', 1.0))

		# 	self.d_adv_loss = - tf.reduce_mean(self.dis_real_img) + tf.reduce_mean(self.dis_fake_img)
		# 	self.d_adv_loss += self.d_gp_loss

		# elif self.adv_type == 'gan':
		# 	self.d_adv_loss = get_loss('classification', self.config.get('gan loss', 'cross entropy'),{
		# 		'preds' : self.dis_real_img, 'labels': tf.ones_like(self.dis_real_img)
		# 	})
		# 	self.d_adv_loss += get_loss('classification', self.config.get('gan loss', 'cross entropy'), {
		# 		'preds' : self.dis_fake_img, 'labels': tf.zeros_like(self.dis_fake_img)
		# 	})
		# else:
		# 	raise Exception('None adverserial type of ' + self.adv_type)

		
		# self.d_cls_loss = (get_loss('classification', self.config.get('classification loss', 'cross entropy'),
		# 						{'preds' : self.dis_fake_img, 'labels': tf.ones_like(self.dis_fake_img) })
		# 						* self.config.get('classification loss weight', 1.0))

		# self.d_loss = self.d_adv_loss + self.d_cls_loss



		# if self.adv_type == 'wgan':
		# 	self.g_adv_loss = -tf.reduce_mean(self.src_fake_img)
		# elif self.adv_type == 'gan':
		# 	self.g_adv_loss = get_loss('classification', self.config.get('gan loss', 'cross entropy'), 
		# 						{'preds' : self.dis_fake_img, 'labels' : self.ones_like(self.dis_fake_img)})

		# self.g_recon_loss = (get_loss('reconstruction', self.config.get('reconstruction loss', 'mse'), 
		# 						{'x' : self.real_img, 'y' : self.recon_img})
		# 					* self.config.get('reconstruction loss weight', 1.0))

		# self.g_cls_loss = (get_loss('classification', self.config.get('classification loss', 'cross entropy'), 
		# 						{'preds' : self.cls_fake_img, 'labels' : self.fake_img_attr})
		# 					* self.config.get('classification loss weight', 1.0))

		# self.g_loss = self.g_adv_loss + g_recon_loss + g_cls_loss


		# ###########################################################################
		# # optimizer configure
		# self.global_step, self.global_step_update = get_global_step()

		# if 'lr' in self.config:
		# 	self.learning_rate = get_learning_rate(self.config['lr_scheme'], float(self.config['lr']), self.global_step, self.config['lr_params'])
		# 	optimizer_params = {'learning_rate' : self.learning_rate}
		# else:
		# 	optimizer_params = {}

		# self.discriminator_optimizer = get_optimizer(self.config['optimizer'], optimizer_params, self.d_loss, 
		# 				self.discrimintor_vars)
		# self.generator_optimizer = get_optimizer(self.config['optimizer'], optimizer_params, self.g_loss, 
		# 				self.generator_vars)

		# self.discriminator_train_op = tf.group([self.discriminator_optimizer, self.global_step_update])
		# self.generator_train_op = tf.group([self.generator_optimizer, self.global_step_update])


		###########################################################################
		# model saver
	# 	self.saver = tf.train.Saver(self.vars + [self.global_step,])


	# @property
	# def discrimintor_vars(self):
	# 	return self.discriminator.vars

	# @property
	# def generator_vars(self):
	# 	return self.generator.vars

	# @property
	# def vars(self):
	# 	return self.discriminator.vars + self.generator_vars

	def build_summary(self):

		# if self.has_summary:
		# 	sum_list = []
		# 	sum_list.append(tf.summary.scalar('discriminator/adv_loss', self.d_adv_loss))
		# 	sum_list.append(tf.summary.scalar('discriminator/cls_loss', self.d_cls_loss))
		# 	if self.adv_type == 'wgan':
		# 		sum_list.append(tf.summary.scalar('discriminator/gp_loss', self.d_gp_loss))
		# 	sum_list.append(tf.summary.scalar('discriminator/loss', self.d_loss))

		# 	self.discriminator_summary = tf.summary.merge(sum_list)


		# 	sum_list = []
		# 	sum_list.append(tf.summary.scalar('generator/adv_loss', self.g_adv_loss))
		# 	sum_list.append(tf.summary.scalar('generator/cls_loss', self.g_cls_loss))
		# 	sum_list.append(tf.summary.scalar('generator/reconstruction_loss', self.g_recon_loss))
		# 	sum_list.append(tf.summary.scalar('generator/loss', self.g_loss))
		# 	self.generator_summary = tf.summary.merge(sum_list)


		# 	sum_list = [tf.summary.histogram(var.name, var) for var in self.vars]
		# 	self.histogram_summary = tf.summary.merge(sum_list)


		# else:
		# 	self.discriminator_summary = None
		# 	self.generator_summary = None
		pass

	def train_on_batch_supervised(self, x_batch, y_batch):
		raise NotImplementedError


	def train_on_batch_unsupervised(self, x_batch):
		raise NotImplementedError   


	def predict(self, z_sample):
		raise NotImplementedError


	#
	#	summary operations
	#
	def summary(self, sess):
		# if self.has_summary:
		# 	sum = sess.run(self.histogram_summary)
		# 	return sum
		# else:
		return None

