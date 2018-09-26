import os
import sys
import time
import numpy as np
import tensorflow as tf
from functools import partial
# from abc import ABCMeta, abstractmethod



from encoder.encoder import get_encoder
from decoder.decoder import get_decoder
from classifier.classifier import get_classifier
from generator.generator import get_generator
from discriminator.discriminator import get_discriminator


from utils.optimizer import get_optimizer_by_config
from utils.learning_rate import get_global_step

class BaseModel(object):


	def __init__(self, config):
		"""Initialization
		"""
		self.name = config['name']
		self.has_summary = config.get('summary', False)
		self.is_training = tf.placeholder(tf.bool, name='is_training')
		self.config = config

		self.global_step = None
		self.learning_rate = None
		self.sum_hist = None
		self.saver = None



	#
	#	model initialize methods
	#
	def load_checkpoint(self, sess, log_dir):
		print(" [*] Reading checkpoint... : " + log_dir)
		ckpt = tf.train.get_checkpoint_state(log_dir)      
		if self.saver is not None and ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(sess, os.path.join(log_dir, ckpt_name))
			return True
		else:
			return False
		
	def save_checkpoint(self, sess, log_dir, step):
		if self.saver is not None:
			self.saver.save(sess,
						os.path.join(log_dir, self.name),
						global_step=step)


	def load_pretrained_weights(self, sess):
		pass


	#
	#	train methods (Please override those method in the derived model!)
	#
	def train_on_batch_supervised(self, x_batch, y_batch):
		""" Given a batch of input data and desired output data, training the model by a step
		
		Return: 
			step, learning_rate, loss, summary
		1. if there are multiple learning rates and losses, the returned learning_rate and loss can be a string.
		2. the summary is recorded per step by giving the input data and is for recording the model output and performance, 
			it is different from summary method, which just logs the model varable destribution.
			if there is no summary needed, please return None
		"""
		raise NotImplementedError

	def train_on_batch_unsupervised(self, x_batch):
		""" Given a batch of input data, training the model by a step
		
		Return: 
			step, learning_rate, loss, summary
		1. if there are multiple learning rates and losses, the returned learning_rate and loss can be a string.
		2. the summary is recorded per step by giving the input data and is for recording the model output and performance, 
			it is different from summary method, which just logs the model varable destribution.
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

	def hidden_variable(self, sess, x_batch):
		""" Given a batch of input data, return the corresponding hidden variable
		"""
		raise NotImplementedError

	def attention(self, sess, x_batch):
		""" Given a batch of input data, return the corresponding attention weight
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
		if self.sum_hist is not None:
			summ = sess.run(self.sum_hist)
			return summ
		else:
			return None



	#
	#	utils functions for building model
	#
	def _build_encoder(self, name, params=None):
		net_config = self.config[name + ' params'].copy()
		if params is not None:
			net_config.update(params)
		return get_encoder(self.config[name], net_config, self.is_training)

	def _build_decoder(self, name, params=None):
		net_config = self.config[name + ' params'].copy()
		if params is not None:
			net_config.update(params)
		return get_decoder(self.config[name], net_config, self.is_training)

	def _build_classifier(self, name, params=None):
		net_config = self.config[name + ' params'].copy()
		if params is not None:
			net_config.update(params)
		return get_classifier(self.config[name], net_config, self.is_training)

	def _build_generator(self, name, params=None):
		net_config = self.config[name + ' params'].copy()
		if params is not None:
			net_config.update(params)
		return get_generator(self.config[name], net_config, self.is_training)

	def _build_discriminator(self, name, params=None):
		net_config = self.config[name + ' params'].copy()
		if params is not None:
			net_config.update(params)
		return get_discriminator(self.config[name], net_config, self.is_training)

	def _build_step_var(self, name):
		step, step_update = get_global_step(name)
		return step, step_update

	def _build_loss(self, type, name, args):
		raise NotImplementedError
	
	def _build_optimizer(self, name, loss, vars, step=None, step_update=None):
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
	
	def _build_train_function(self, name, loss, vars, *,
			step=None, 
			step_update=None, 
			summary=None, 
			build_summary=False, 
			sum_list=None, 
			build_endpoints_summary=False, 
			endpoints_sum_list=None):
		""" return a function that optimize @params.loss.
		and the returned function is callable like this:
		step, lr, loss, summary = function(sess, feed_dict)

		the optimizer type is written in the model config file, 
		@params.name + ' optimzer' and @params.name + ' optimizer params' must be set in config file

		Arguments: 
			name : the optimizer name in config file, 
					e.g.  if name is "classifier", the config json file is like this:
							"classifier optimizer" : "adam",
							"classifier optimizer params" : {
								"lr" : 0.0001,
								"lr scheme" : "exponential",
								"lr params" : {
									"decay_steps" : 45000,
									"decay_rate" : 0.1
								}
							},
			loss : the target loss variable,
			vars : the optimize model variable,

			step and step_update : 	1. if both step and step_update is None, then it will create an auto increment step var by calling train_function
									2. if step is not None and step_update is None, then calling the train_function will not increase the given step variable
									3. if both step and step_update is not None, 
			summary :
			build_summary and sum_list : if you want to see the loss and learning rate curve in the tensorboard summary, set @params.build_summary to True,
											and if you have some other variables to watch, just add the summary operation to the @params.sum_list.

			build_endpoints_summary and endpoints_sum_list : if you want to see the output of model interval layers and you dont want it be logged too frequently,
											set @params.build_endpoints_summary to True and add the endpoints summary to the @params.endpoints_sum_list.
											it will create another train function with endpoints logging.

		Return:
			if build_endpoints_summary is True:
				return train_function, train_function2, learning_rate_var,
					: where the train_function is without endpoints summary and train_function2 is with endpoints summary
			else:
				return train_function, learning_rate_var
		"""
		train_op, lr_var, step_var = self._build_optimizer(name, loss, vars, step, step_update)

		if build_summary:
			if sum_list is None:
				sum_list = []
			sum_list.append(tf.summary.scalar(name + '/loss', loss))
			sum_list.append(tf.summary.scalar(name + '/lr', lr_var))
			summary = tf.summary.merge(sum_list)

		if build_endpoints_summary:
			if sum_list is None:
				sum_list = []
			sum_list += endpoints_sum_list
			endpoint_summary = tf.summary.merge(sum_list)
		
		if build_endpoints_summary:
			return partial(self.train, update_op=train_op, step=step_var, learning_rate=lr_var, loss=loss, summary=summary),   \
					partial(self.train, update_op=train_op, step=step_var, learning_rate=lr_var, loss=loss, summary=endpoint_summary), lr_var
		else:
			return partial(self.train, update_op=train_op, step=step_var, learning_rate=lr_var, loss=loss, summary=summary), lr_var

	def draw_sample( self, mean, log_var):
		epsilon = tf.random_normal( ( tf.shape( mean ) ), 0, 1 )
		sample = mean + tf.exp( 0.5 * log_var ) * epsilon
		return sample

	#
	#
	#
	def train(self, sess, feed_dict,
				update_op=None, step=None, learning_rate=None, loss=None,
				summary=None):
		# if step is None and hasattr(self, 'global_step'):
		# 	step = self.global_step
		# if learning_rate is None:
		# 	learning_rate = self.learning_rate
		# if loss is None:
		# 	loss = self.loss

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

