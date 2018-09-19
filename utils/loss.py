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

#
#	kl divergence loss
#
def kl_gaussian_loss(mean, log_var, instance_weight=None):
	if instance_weight is None:
		return -0.5 * tf.reduce_mean(1.0 + log_var - tf.exp(log_var) - tf.square(mean))
	else:
		return tf.reduce_mean(-0.5 * tf.reduce_mean(1.0 + log_var - tf.exp(log_var) - tf.square(mean), axis=-1) * instance_weight)


def kl_bernoulli_loss(logits=None, probs=None, instance_weight=None, lossen=0.02):
	if logits is not None:
		probs = tf.nn.softmax(logits)
	if probs is None:
		raise Exception("Probs can not be none")
	if instance_weight is None:
		return -tf.reduce_mean(tf.log((probs + lossen) / (1 + lossen)) * probs)
	else:
		return -tf.reduce_mean(tf.reduce_mean(tf.log((probs + lossen) / (1 + lossen)) * probs, axis=-1) * instance_weight) 


def kl_categorical_loss(logits=None, probs=None, instance_weight=None, lossen=0.02):
	"""

	"""
	if logits is not None:
		probs = tf.nn.softmax(logits)
	if probs is None:
		raise Exception("Probs can not be none")
	if instance_weight is None:
		return -tf.reduce_mean(tf.log((probs + lossen) / (1 + lossen)) * probs)
	else:
		return -tf.reduce_mean(tf.reduce_mean(tf.log((probs + lossen) / (1 + lossen)) * probs, axis=-1) * instance_weight) 


def l2_loss(x, y, instance_weight=None):
	x = tcl.flatten(x)
	y = tcl.flatten(y)
	if instance_weight is None:
		return tf.reduce_mean(tf.square(x - y))
	else:
		return tf.reduce_mean(tf.reduce_mean(tf.square(x-y), axis=-1) * instance_weight)


def l1_loss(x, y, instance_weight=None):
	x = tcl.flatten(x)
	y = tcl.flatten(y)
	if instance_weight is None:
		return tf.reduce_mean(tf.abs(x - y))
	else:
		return tf.reduce_mean(tf.reduce_mean(tf.abs(x-y), axis=-1) * instance_weight)


#
#	Cross entropy loss
#
def classify_cross_entropy_loss(logits, labels, instance_weight=None):
	""" classification cross entropy loss
	the last channel of logits and labels is nb_class, each sample's label must be one hot
	"""
	if instance_weight is None:
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
	else:
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits) * instance_weight)


def classify_binary_entropy_loss(logits, labels, instance_weight=None, prob_clamp=1e-5):
	""" classification cross entropy loss
	the last channel of logits and labels is nb_class and each class is indiviual, 
	which means each sample's label need not to be one-hot
	"""

	nb_classes = tf.cast(tf.shape(logits)[-1], tf.float32)
	logits = tf.reshape(logits, [tf.shape(logits)[0], -1])
	labels = tf.reshape(labels, [tf.shape(labels)[0], -1])
	probs = tf.sigmoid(logits)

	probs = tf.maximum(probs, prob_clamp)
	probs = tf.minimum(probs, 1.0-prob_clamp)
	if instance_weight is None:
		return - tf.reduce_mean(labels * tf.log(probs) + (1.0 - labels) / nb_classes * tf.log(1.0 - probs))
	else:
		return - tf.reduce_mean(tf.reduce_mean(labels * tf.log(probs) + (1.0 - labels) / nb_classes * tf.log(1.0 - probs), axis=-1) * instance_weight)


def segmentation_cross_entropy_loss(logits, mask, instance_weight=None):
	if instance_weight is None:
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=mask, logits=logits))
	else:
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=mask, logits=logits) * instance_weight)


#
#	regulation loss
# 
def regularization_l1_loss(var_list):
	loss = 0
	for var in var_list:
		loss += tf.reduce_mean(tf.square(var))
	return loss

def regularization_l2_loss(var_list):
	loss = 0
	for var in var_list:
		loss += tf.reduce_mean(tf.abs(var))
	return loss

def feature_matching_l2_loss(fx, fy, fnames):
	assert(isinstance(fnames, list))
	loss = tf.constant(0.0, dtype=tf.float32)
	for feature_name in fnames:
		loss += tf.reduce_mean(tf.square(fx[feature_name] - fy[feature_name]))
	return loss


#
#	adversarial down loss:
#		for training discriminator in gan networks
#
def adv_down_wassterstein_loss(dis_real, dis_fake):
	"""the adversarial loss of discriminator,
	which has only 1 output of real/fake prob
	"""
	assert(dis_real.get_shape()[1:] == dis_fake.get_shape()[1:])
	return - tf.reduce_mean(dis_real) + tf.reduce_mean(dis_fake)

def adv_down_cross_entropy_loss(dis_real, dis_fake):
	"""the adversarial loss of discriminator,
	which has only 1 output of real/fake prob
	"""
	assert(dis_real.get_shape()[1:] == dis_fake.get_shape()[1:])
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.zeros_like(dis_fake)))
	loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real, labels=tf.ones_like(dis_fake)))
	loss /= 2.0
	return loss

def adv_down_supervised_cross_entropy_loss(dis_real, dis_fake, label, label_smooth=1.0):
	""" the supervised adversarial loss of discriminator,
	which has nb_class+1 outputs, the first nb_class outputs is the probs of each classes,
	and the last output is the prob of fake images
	"""
	assert(dis_real.get_shape()[1:] == dis_fake.get_shape()[1:])
	real_label = tf.concat([label, tf.zeros(shape=[tf.shape(label)[0], 1])], axis=1)
	fake_label = tf.concat([tf.zeros_like(label), tf.ones(shape=[tf.shape(label)[0], 1])], axis=1)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real, labels=real_label*label_smooth))
	loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=fake_label*label_smooth))
	# loss /= 2.0
	return loss

def adv_down_unsupervised_cross_entropy_loss(dis_real, dis_fake):
	""" the unsupervised adversarial loss of discriminator,
	which has nb_class+1 outputs, the first nb_class outputs is the probs of each classes,
	and the last output is the prob of fake images
	"""
	assert(dis_real.get_shape()[1:] == dis_fake.get_shape()[1:])
	assert(dis_real.get_shape().ndims == 2)

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.zeros_like(dis_fake))[:, -1])
	loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real, labels=tf.ones_like(dis_fake))[:, -1])
	loss /= 2.0
	return loss

#
#	adversarial up loss:
#		for training generator in gan networks
#
def adv_up_wassterstein_loss(dis_fake):
	return - tf.reduce_mean(dis_fake)

def adv_up_cross_entropy_loss(dis_fake):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.ones_like(dis_fake)))

def adv_up_supervised_cross_entropy_loss(dis_fake, label, label_smooth=0.9):
	return tf.constant(0.0, dtype=tf.float32)

def adv_up_unsupervised_cross_entropy_loss(dis_fake):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake[:,-1], labels=tf.ones_like(dis_fake[:,-1])))


def gradient_penalty_l2_loss(x, y):
	gradients = tf.gradients(y, xs=[x])[0]
	g_rank = len(gradients.get_shape())
	slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[i for i in range(1, g_rank)]))
	gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
	return gradient_penalty


loss_dict = {
	'kl' : {
		'gaussian' : kl_gaussian_loss,
		'bernoulli' : kl_bernoulli_loss,
		'categorical' : kl_categorical_loss,
	},
	'reconstruction' : {
		'mse' : l2_loss,
		'l2' : l2_loss,
		'l1' : l1_loss
	},
	'classification' : {
		'cross entropy' : classify_cross_entropy_loss,
		'binary entropy' : classify_binary_entropy_loss
	},
	'segmentation' : {
		'cross entropy' : segmentation_cross_entropy_loss,
	},
	'regularization' : {
		'l2' : regularization_l2_loss,
		'l1' : regularization_l1_loss,
	},
	'feature matching' : {
		'l2' : feature_matching_l2_loss
	},
	'adversarial down' : {
		'wassterstein' : adv_down_wassterstein_loss,
		'cross entropy' : adv_down_cross_entropy_loss,
		'supervised cross entropy' : adv_down_supervised_cross_entropy_loss,
		'unsupervised cross entropy' : adv_down_unsupervised_cross_entropy_loss,
	},
	'adversarial up' : {
		'wassterstein' : adv_up_wassterstein_loss,
		'cross entropy' : adv_up_cross_entropy_loss,
		'supervised cross entropy' : adv_up_supervised_cross_entropy_loss,
		'unsupervised cross entropy' : adv_up_unsupervised_cross_entropy_loss,
	},
	'gradient penalty' : {
		'l2' : gradient_penalty_l2_loss
	}
}


def get_loss(loss_name, loss_type, loss_params):
	if loss_name in loss_dict:
		if loss_type in loss_dict[loss_name]:
			return loss_dict[loss_name][loss_type](**loss_params)
	raise Exception("None loss named " + loss_name + " " + loss_type)

