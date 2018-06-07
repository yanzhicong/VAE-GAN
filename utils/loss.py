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



def kl_loss(mean, log_var):
    return -0.5 * tf.reduce_mean(1.0 + log_var - tf.exp(log_var) - tf.square(mean), axis=-1)

def kl_bernoulli_loss(logits=None, probs=None):
    if logits is not None:
        probs = tf.nn.softmax(logits)
    logprobs = tf.log(probs)
    return tf.reduce_sum(probs * logprobs)


def l2_loss(x, y):
    return tf.reduce_mean(tf.square(x - y), axis=-1)


def binary_cls_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))

loss_dict = {
    'kl' : {
        'gaussian' : kl_loss,
        'bernoulli' : kl_bernoulli_loss
    },
    'reconstruction' : {
        'mse' : l2_loss,
        'l2' : l2_loss
    },
    'classification' : {
        'cross entropy' : binary_cls_loss
    }
}


def get_loss(loss_name, loss_type, loss_params):
    if loss_name in loss_dict:
        if loss_type in loss_dict[loss_name]:
            return loss_dict[loss_name][loss_type](**loss_params)
    raise Exception("None loss named " + loss_name + " " + loss_type)


