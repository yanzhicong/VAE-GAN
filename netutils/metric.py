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
        decay = tf.convert_to_tensor(decay, name='decay', dtype=tf.float32)
        update_delta = (variable - value) * decay
        return tf.assign_sub(variable, update_delta, name=scope)

def accuracy_top_1(labels, logits=None, probs=None, decay=0.01):
    """ calculate moving accuracy for classification
    Arguments:
        labels : [batch_size, nb_classes],   must be one-hot
        logits or probs : [batch_size, nb_classes]
        decay : float in range [0, 1]
    """
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


def accuracy_multi_class_acc(labels, probs, threshold=0.5, decay=0.01):
    """
    """
    preds = tf.cast(probs > threshold, tf.int32)
    labels = tf.cast(labels > threshold, tf.int32)
    acc = tf.cast(tf.reduce_sum(tf.cast(tf.equal(labels, preds), tf.int32)), tf.float32) / tf.cast(tf.reduce_sum(tf.ones_like(labels)), tf.float32)

    if decay == 1.0:
        return acc
    else:
        var = tf.Variable(0.0, name='acc_top_1')
        return _assign_moving_average(var, acc, decay)


def accuracy_multi_class_acc2(labels, probs, threshold=0.5, decay=0.01):
    """
    """

    # print("accuracy_multi_class_acc2")

    # print("input : ")
    # print(labels.get_shape())
    # print(probs.get_shape())
    # preds = tf.cast(probs > threshold, tf.int32)
    # labels = tf.cast(labels > threshold, tf.int32)

    probs = tf.reshape(probs, [-1, tf.shape(probs)[-1]])
    labels = tf.reshape(labels, [-1, tf.shape(labels)[-1]])

    normal_class1 = tf.ones([tf.shape(probs)[0], 1]) * threshold
    normal_class2 = tf.ones([tf.shape(labels)[0], 1]) * threshold


    probs = tf.concat([normal_class1, probs], axis=-1)
    labels = tf.concat([normal_class2, labels], axis=-1)


    # print(labels.get_shape())
    # print(probs.get_shape())

    pred_class = tf.argmax(probs, axis=-1)
    label_class = tf.argmax(labels, axis=-1)

    # print("class : ")
    # print(pred_class.get_shape())
    # print(label_class.get_shape())


    acc = tf.cast(tf.reduce_sum(tf.cast(tf.equal(pred_class, label_class), tf.int32)), tf.float32) / tf.cast(tf.reduce_sum(tf.ones_like(label_class)), tf.float32)

    if decay == 1.0:
        return acc
    else:
        var = tf.Variable(0.0, name='acc_top_1')
        return _assign_moving_average(var, acc, decay)


def segmentation_miou(mask, nb_classes, logits=None, probs=None):
    if probs is not None:
        pred = tf.argmax(probs, axis=-1)
    elif logits is not None:
        pred = tf.argmax(logits, axis=-1)
    else:
        raise Exception('logits and probs are none')

    mask_ori = tf.argmax(mask, axis=-1)
    weights = tf.logical_and(tf.greater_equal(mask, -0.5), tf.less_equal(mask, nb_classes))
    weights = tf.cast(weights, tf.int32)

    miou, miou_update_op = tcm.streaming_mean_iou(pred, mask_ori, num_classes=nb_classes, weights=weights,
                            # metrics_collections=[tf.GraphKeys.GLOBAL_VARIABLES]
                            )

    with tf.control_dependencies([miou_update_op]):
        miou = tf.identity(miou)
    return miou


metric_dict = {
    'accuracy' :  {
        'top1' : accuracy_top_1,
        'multi-class acc' : accuracy_multi_class_acc,
        'multi-class acc2' : accuracy_multi_class_acc2,
    },
    'moving accuracy' : {
        'top1' : accuracy_top_1,
        'multi-class acc' : accuracy_multi_class_acc,
        'multi-class acc2' : accuracy_multi_class_acc2,
    },
    'segmentation' : {
        'miou' : segmentation_miou
    }
}

def get_metric(metric_name, metric_type, metric_params):
    if metric_name in metric_dict:
        if metric_type in metric_dict[metric_name]:
            return metric_dict[metric_name][metric_type](**metric_params)
    raise Exception("None metric named " + metric_name + ' of type ' + metric_type)

