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
import tensorflow.contrib.metrics as tcm


def _assign_moving_average(variable, value, decay):
    with tf.name_scope(None, 'AssignMovingAvg', [variable, value, decay]) as scope:
        decay = tf.convert_to_tensor(decay, name='decay')
        update_delta = (variable - value) * decay
        return tf.assign_sub(variable, update_delta, name=scope)


def accuracy_top_1(labels, logits=None, probs=None, decay=0.01):
    if probs is not None:
        acc = tcm.accuracy(predictions=tf.argmax(probs, axis=-1), labels=tf.argmax(labels, axis=-1))
    elif logits is not None:
        acc = tcm.accuracy(predictions=tf.argmax(logits, axis=-1), labels=tf.argmax(labels, axis=-1)) 
    else:
        raise Exception('in metric accuracy, the probability vector cannot be None')

    if decay == 1.0:
        return acc
    else:
        var = tf.Variable(0.0, name='acc_top_1')
        return _assign_moving_average(var, acc, decay)


# def accuracy_top_5(labels,logits=None, probs=None, decay=0.01):
#     if probs is not None:
#         acc = tcm.accuracy(predictions=tf.argmax(probs, axis=-1), labels=tf.argmax(labels, axis=-1))
#     elif logits is not None:
#         acc = tcm.accuracy(predictions=tf.argmax(logits, axis=-1), labels=tf.argmax(labels, axis=-1)) 
#     else:
#         raise Exception('in metric accuracy, the probability vector cannot be None')

#     if decay == 1.0:
#         return acc
#     else:
#         var = tf.Variable(0.0, name='acc_top_1')
#         return _assign_moving_average(var, acc, decay)

metric_dict = {
    'accuracy' :  {
        'top1' : accuracy_top_1
    }
}


def get_metric(metric_name, metric_type, metric_params):
    if metric_name in metric_dict:
        if metric_type in metric_dict[metric_name]:
            return metric_dict[metric_name][metric_type](**metric_params)
    raise Exception("None metric named " + metric_name + ' of type ' + metric_type)


