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

from encoder.encoder import get_encoder
from decoder.decoder import get_decoder
from classifier.classifier import get_classifier
from discriminator.discriminator import get_discriminator

from utils.sample import get_sample
from utils.loss import get_loss

from .basemodel import BaseModel


class StarGAN(BaseModel):


	def __init__(self, config,
		**kwargs
	):
		
		super(StarGAN, self).__init__(input_shape=config['input_shape'], **kwargs)

		self.config = config
		self.input_shape = config['input_shape']
		# self.z_dim = config['z_dim']
		# self.config = config

		self.nb_classes = config['nb_classes']
		self.adv_type = config.get('adv_type', 'wgan')

		self.build_model()

		if self.is_summary:
			self.get_summary()


	def build_model(self):
		self.real_img = tf.placeholder(tf.float32, shape=[None, ] + self.input_shape, name='real_img')
		self.real_img_attr = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='real_img_attr')

		self.fake_img_attr = tf.placeholder(tf.float32, shape=[None, self.nb_classes], name='fake_img_attr')
		self.epsilon = tf.placeholder(tf.float32, [None, 1, 1, 1], name = 'gp_random_num')
	
		self.generator = get_encoder(self.config['generator'], self.config['generator params'], self.config, self.is_trainings)
		self.discriminator = get_decoder(self.config['discriminator'], self.config['discriminator params'], self.config, self.is_training)


		real_img_with_fake_attr = tf.concat(
			self.real_img,
			tf.tile(
				tf.reshape(self.fake_img_attr, [-1, 1, 1, self.nb_classes]),
				[1, self.input_shape[0], self.input_shape[1], 1]
			), axis=3
		)

		self.fake_img = self.generator(real_img_with_fake_attr, reuse=False)
		
		fake_img_with_real_attr = tf.concat(
			self.fake_img,
			tf.tile(
				tf.reshape(self.real_img_attr, [-1, 1, 1, self.nb_classes]),
				[1, self.input_shape[0], self.input_shape[1], 1]
			), axis=3			
		)

		self.recon_img = self.generator(fake_img_with_real_attr, reuse=True)

		self.dis_real_img, self.cls_real_img = self.discriminator(self.real_img, reuse=False)
		self.dis_fake_img, self.cls_fake_img = self.discriminator(self.fake_img, reuse=True)


		# discriminator loss
		if self.adv_type == 'wgan':
			self.d_gp_loss = get_loss('gradient penalty', self.config.get('gradient penalty', 'l2'),
						{})
			self.d_gp_loss = self.d_gp_loss * self.config.get('gradient penalty prod', 1.0)

			self.d_adv_loss = - tf.reduce_mean(self.dis_real_img)
			self.d_adv_loss += tf.reduce_mean(self.dis_fake_img)
			self.d_adv_loss += self.d_gp_loss

		elif self.adv_type == 'gan':
			self.d_adv_loss = get_loss('classification', self.config.get('gan loss', 'cross entropy'),{
				'preds' : self.dis_real_img, 'labels': tf.ones_like(self.dis_real_img)
			})
			self.d_adv_loss += get_loss('classification', self.config.get('gan loss', 'cross entropy'), {
				'preds' : self.dis_fake_img, 'labels': tf.ones_like(self.dis_fake_img)
			})
		else:
			raise Exception('None adverserial type of ' + self.adv_type)

		
		self.d_cls_loss = get_loss('classification', self.config.get('cls loss', 'binary entropy'),
			{'preds' : self.dis_fake_img, 'labels': tf.ones_like(self.dis_fake_img) }
		)
		self.d_cls_loss *= self.config.get('cls loss prod', 1.0)
		self.d_loss = self.d_adv_loss + self.d_cls_loss


		self.x_fake = self.decoder(tf.concat([z, self.label_real], axis=1))

		z_possible = tf.placeholder(tf.float32, shape=(None, self.z_dim))
		c_possible = tf.placeholder(tf.float32, shape=(None, self.nb_classes))

		x_possible = self.decoder(tf.concat([z_possible, c_possible],axis=1), reuse=True)

		d_real, feature_disc_real = self.discriminator(self.x_real)
		d_fake, feature_disc_fake = self.discriminator(self.x_fake, reuse=True)
		d_possible, feature_disc_possible = self.discriminator(x_possible, reuse=True)

		c_real, feature_clas_real = self.classifier(self.x_real)
		c_fake, feature_clas_fake = self.classifier(self.x_fake)
		c_possible, feature_clas_possible = self.classifier(self.x_possible)


	# def get_kl_loss(self, )

	def train_on_batch_supervised(self, x_batch, y_batch):
		raise NotImplementedError


	def train_on_batch_unsupervised(self, x_batch):
		raise NotImplementedError   


	def predict(self, z_sample):
		raise NotImplementedError


	def get_summary(self):
		pass


