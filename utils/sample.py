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


from functools import partial
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as tcl






def sample_mix_gaussian(batch_size, nb_classes, z_dim, radis=2.0, x_var=0.5, y_var=0.1):
    assert z_dim == 2
    def sample(x, y, label, n_labels):
        shift = 3
        r = radis * np.pi / n_labels * label
        new_x = x * np.cos(r) - y * np.sin(r)
        new_y = x * np.sin(r) + y * np.cos(r)
        new_x += shift * np.cos(r)
        new_y += shift * np.sin(r)
        return new_x, new_y
    x = np.random.normal(0, x_var, [batch_size, 1])
    y = np.random.normal(0, y_var, [batch_size, 1])
    label = np.random.randint(0, nb_classes, size=[batch_size, 1]).astype(np.float32)
    label_onehot = np.zeros(shape=(batch_size, nb_classes)).astype(np.float32)
    for i in range(batch_size):
        label_onehot[i, int(label[i, 0])] = 1

    x, y = sample(x, y, label, nb_classes)
    return np.concatenate([x, y], axis=1).astype(np.float32), label_onehot


def sample_normal(batch_size, z_dim, var=1.0):
    if isinstance(z_dim, list):
        shape = [batch_size] + z_dim
    elif isinstance(z_dim, int):
        shape = [batch_size, z_dim]
    return np.random.normal(0, var, shape).astype(np.float32)


def sample_categorical(batch_size, nb_classes):
    def to_categorical(y, nb_classes):
        input_shape = y.shape
        y = y.ravel().astype(np.int32)
        n = y.shape[0]
        ret = np.zeros((n, nb_classes), dtype=np.float32)
        indices = np.where(y >= 0)[0]
        ret[np.arange(n)[indices], y[indices]] = 1.0
        ret = ret.reshape(list(input_shape) + [nb_classes, ])
        return ret

    label = np.random.randint(0, nb_classes, size=[
				batch_size]).astype(np.float32)
    label_onehot = to_categorical(label, nb_classes)
    return label_onehot


sample_dict = {
    'mixGaussian' : sample_mix_gaussian,
    'normal' : sample_normal,
    'categorical' : sample_categorical,
    'mix gaussian' : sample_mix_gaussian,
    'gaussian' : sample_normal
}


def get_sampler(name, **kwargs):
    """ get sample function by name
    """
    if name in sample_dict:
        return partial(sample_dict[name], **kwargs)
    else:
        raise ValueError('No sampler named ' + name)


def get_sample(name, args):
    if name == 'normal' : 
        z_avg, z_log_var = args
        eps = tf.random_normal(shape=tf.shape(z_avg), mean=0.0, stddev=1.0)
        return z_avg + tf.exp(z_log_var / 2.0) * eps
    else:
        raise Exception("None sample function named " + name)


