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

from .base_model import BaseModel


class CVAEGAN(BaseModel):

    def __init__(self, config):
        
        super(CVAEGAN, self).__init__(config)

        raise NotImplementedError

        self.input_shape = config['input shape']
        self.nb_classes = config['nb classes']
        self.z_dims = config['z dims']
        self.config = config
        self.build_model()

    def build_model(self):

        self.encoder = self.build_encoder('encoder', params={
            'name' : 'Encoder'
        })

        self.decoder = self.build_decoder('decoder', params={
            'name' : 'Decoder'
        })

        self.classifier = self.build_classifier('classifier', params={
            'name' : 'Classifier'
        })

        self.discriminator = self.build_discriminator('discriminator', params={
            'name' : 'Discriminator'
        })
        
        self.x_real = tf.placeholder(tf.float32, shape=[None, ] + self.input_shape, name='xinput')
        self.y_real = tf.placeholder(tf.float32, shape=[None, self.num_classes,], name='cls')


        z_mean, z_log_var = self.encoder(self.x_real, condition=self.y_real)
        z_sample = self.draw_sample(z_mean, z_log_var)
        x_fake = self.decoder(z_sample, condition=self.y_real)


        self.z_possible = tf.placeholder(tf.float32, shape=(None, self.z_dim))
        x_possible = self.decoder(self.z_possible, condition=self.y_real)

        dis_real, dis_real_feature = self.discriminator.features(self.x_real)
        dis_fake, dis_fake_feature = self.discriminator.features(x_fake)
        dis_possible, dis_possible_feature = self.discriminator.features(x_possible)

        cls_real, cls_real_feature = self.classifier.features(self.x_real)
        cls_fake, dis_fake_feature = self.classifier.features(x_fake)
        cls_possible, cls_possible_feature = self.classifier.features(x_possible)

        # encoder loss

        self.d_loss_adv = get_loss('discriminator adversarial', 'wassterstein', { 'dis_real' : dis_real, 'dis_fake' : dis_fake })

        self.d_loss_fm = get_loss('feature matching', 'l2', {'f1' : dis_fake_feature, 'f2' : dis_possible_feature})


        self.g_loss_kl = get_loss('kl', 'gaussian', {'mean' : z_mean, 'log_var' : z_log_var})

        self.g_loss_adv = get_loss('generator adversarial', 'wassterstein', {'dis_fake' : dis_fake })

        self.g_loss


        # self.encoder_loss_generator = get_loss('')


        # self.x_real = tf.placeholder(tf.float32, shape=[None, ] + self.input_shape, name='xinput')


        # z_avg = z_params[:, :self.z_dim]
        # z_log_var = z_params[:, self.z_dim:]



        # x_fake = self.decoder(tf.concat([z, self.label_real], axis=1))

        # c_possible = tf.placeholder(tf.float32, shape=(None, self.nb_classes))

        # x_possible = self.decoder(tf.concat([z_possible, c_possible],axis=1), reuse=True)


        # d_real, feature_disc_real = self.discriminator(self.x_real)
        # d_fake, feature_disc_fake = self.discriminator(x_fake, reuse=True)
        # d_possible, feature_disc_possible = self.discriminator(x_possible, reuse=True)


        # c_real, feature_clas_real = self.classifier(self.x_real)
        # c_fake, feature_clas_fake = self.classifier(x_fake)
        # c_possible, feature_clas_possible = self.classifier(self.x_possible)

    # def get_kl_loss(self, )

    def train_on_batch_supervised(self, x_batch, y_batch):
        raise NotImplementedError


    def train_on_batch_unsupervised(self, x_batch):
        raise NotImplementedError   


    def predict(self, z_sample):
        raise NotImplementedError

