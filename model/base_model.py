import os
import sys
import time
import numpy as np
import tensorflow as tf
# from abc import ABCMeta, abstractmethod



from encoder.encoder import get_encoder
from decoder.decoder import get_decoder
from classifier.classifier import get_classifier
from generator.generator import get_generator
from discriminator.discriminator import get_discriminator


from utils.optimizer import get_optimizer_by_config


class BaseModel(object):


	def __init__(self, config, **kwargs):
		"""Initialization
		"""
		self.name = config['name']
		self.has_summary = config.get('summary', False)
		self.is_training = tf.placeholder(tf.bool, name='is_training')
		self.config = config


	def checkpoint_load(self, sess, log_dir):
		print(" [*] Reading checkpoint... : " + log_dir)
		ckpt = tf.train.get_checkpoint_state(log_dir)      
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(sess, os.path.join(log_dir, ckpt_name))
			return True
		else:
			return False
		
	def checkpoint_save(self, sess, log_dir, step):
		self.saver.save(sess,
						os.path.join(log_dir, self.name),
						global_step=step)




	#
	#	train methods (Please override those method in the derived model!)
	#
	def train_on_batch_supervised(self, x_batch, y_batch):
		""" Given a batch of input data and desired output data, training the model by a step
		
		Return: step, learning_rate, loss, summary
		1. if there are multiple learning rates and losses, the returned learning_rate and loss can be a string.
		2. the summary is recorded per step, and given the input data, which is used differently from summary method.
		if there is no summary needed, please return None
		"""
		raise NotImplementedError

	def train_on_batch_unsupervised(self, x_batch):
		""" Given a batch of input data, training the model by a step
		
		Return: step, learning_rate, loss, summary
		1. if there are multiple learning rates and losses, the returned learning_rate and loss can be a string.
		2. the summary is recorded per step, and given the input data, which is used differently from summary method.
		if there is no summary needed, please return None
		"""
		raise NotImplementedError


	#
	#	test methods (Please override those method in the derived model!)
	#
	def predict(self, sess, x_batch):
		""" Given a batch of input data, return the predict class probs
		"""
		raise NotImplementedError

	def generate(self, sess, z_batch):
		""" Given a batch of hidden variable, return the generated data
		"""
		raise NotImplementedError

	def hidden_distribution(self, sess, x_batch):
		""" Given a batch of input data, return the corresponding hidden variable
		"""
		raise NotImplementedError


	#
	# summary methods (Please override those method in the derived model!)
	#
	def summary(self, sess):
		"""Run the model summary variable and return the summary result, this function is without
		any data input, it just record the state and the weight variables of the model. 
		
		If there is	no summary needed or you want to skip logging summary for speed, please return 
		None in the derived	function.
		"""
		return None


	#
	#	utils functions for building model
	#
	def build_encoder(self, name, params=None):
		net_config = self.config[name + ' params'].copy()
		if params is not None:
			net_config.update(params)
		return get_encoder(self.config[name], net_config, self.is_training)

	def build_decoder(self, name, params=None):
		net_config = self.config[name + ' params'].copy()
		if params is not None:
			net_config.update(params)
		return get_decoder(self.config[name], net_config, self.is_training)

	def build_classifier(self, name, params=None):
		net_config = self.config[name + ' params'].copy()
		if params is not None:
			net_config.update(params)
		return get_classifier(self.config[name], net_config, self.is_training)

	def build_generator(self, name, params=None):
		net_config = self.config[name + ' params'].copy()
		if params is not None:
			net_config.update(params)
		return get_generator(self.config[name], net_config, self.is_training)

	def build_discriminator(self, name, params=None):
		net_config = self.config[name + ' params'].copy()
		if params is not None:
			net_config.update(params)
		return get_discriminator(self.config[name], net_config, self.is_training)


	def build_loss(self, type, name, args):
		pass
	
	def build_optimizer(self, name, loss, vars, step=None, step_update=None):
		if step == None and hasattr(self, 'global_step'):
			step = self.global_step

		if name == '' or name == 'optimizer':
			(train_op,
				learning_rate_var,
					step_var) = get_optimizer_by_config(self.config['optimizer'],
															self.config['optimizer params'],
															loss, vars,
															step, step_update)
		else:
			(train_op,
				learning_rate_var,
					step_var) = get_optimizer_by_config(self.config[name + ' optimizer'],
															self.config[name + ' optimizer params'],
															loss, vars,
															step, step_update)

		return train_op, learning_rate_var, step_var
	
	def draw_sample( self, mean, log_var):
		epsilon = tf.random_normal( ( tf.shape( mean ) ), 0, 1 )
		sample = mean + tf.exp( 0.5 * log_var ) * epsilon
		return sample

	#
	#
	#
	def train(self, sess, feed_dict,
				update_op=None, 
				step=None,
				learning_rate=None,
				loss=None,
				summary=None,
				):
		if step is None and hasattr(self, 'global_step'):
			step = self.global_step
		if learning_rate is None:
			learning_rate = self.learning_rate
		if loss is None:
			loss = self.loss

		if update_op is None:
			s, lr = sess.run([step, learning_rate])
			return s, lr, 0, None
		elif self.has_summary and summary is not None:
			_, s, lr, l, s_sum = sess.run([update_op, step, learning_rate, loss, summary],	
						feed_dict = feed_dict)
			return s, lr, l, s_sum
		else:
			_, s, lr, l = sess.run([update_op, step, learning_rate, loss],	feed_dict = feed_dict)
			return s, lr, l, None

