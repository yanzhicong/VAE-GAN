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

def accuracy_top_1(logits, labels, moveing_weight=0.9):
    acc = tcm.accuracy(predictions=tf.argmax(logits, axis=-1), labels=tf.argmax(labels, axis=-1))
    # tf.Variable(0.0, )
    return acc


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


