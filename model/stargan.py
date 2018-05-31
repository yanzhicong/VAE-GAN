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

from .basemodel import BaseModel


class StarGAN(BaseModel):


    def __init__(self, config,
        **kwargs
    ):
        
        super(StarGAN, self).__init__(input_shape=config['input_shape'], **kwargs)

        self.input_shape = config['input_shape']
        self.nb_classes = config['nb_classes']
        self.z_dim = config['z_dim']
        self.config = config
        self.build_model()

    def build_model(self):
        self.encoder = get_encoder(self.config['encoder'], self.config['encoder params'], self.config)
        self.decoder = get_decoder(self.config['decoder'], self.config['decoder params'], self.config)
        self.classifier = get_classifier(self.config['classifier'], self.config['classifier params'], self.config)
        self.discriminator = get_discriminator(self.config['discriminator'], self.config['discriminator params'], self.config)
        
        self.x_real = tf.placeholder(tf.float32, shape=[None, ] + self.input_shape, name='xinput')
        self.label_real = tf.placeholder(tf.float32, shape=[None, self.num_classes,], name='cls')


        z_params = self.encoder([self.x_real, self.label_real])


        z_avg = z_params[:, :self.z_dim]
        z_log_var = z_params[:, self.z_dim:]

        z = get_sample(self.config['sample_func'], (z_avg, z_log_var))


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



        # self.encoder_loss = 


        pass


    # def get_kl_loss(self, )

    def train_on_batch_supervised(self, x_batch, y_batch):
        raise NotImplementedError


    def train_on_batch_unsupervised(self, x_batch):
        raise NotImplementedError   


    def predict(self, z_sample):
        raise NotImplementedError

