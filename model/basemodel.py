import os
import sys
import time
import numpy as np
import tensorflow as tf
# from abc import ABCMeta, abstractmethod

class BaseModel(object):


	def __init__(self, config, **kwargs):
		'''
		Initialization
		'''
		self.name = config['name']
		self.is_summary = config.get('summary', False)
		self.is_training = tf.placeholder(tf.bool, name='is_training')


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



	def draw_sample( self, mean, log_var ):
		epsilon = tf.random_normal( ( tf.shape( mean ) ), 0, 1 )
		sample = mean + tf.exp( 0.5 * log_var ) * epsilon
		return sample


	'''
		train methods
	'''
	def train_on_batch_supervised(self, x_batch, y_batch):
		raise NotImplementedError
		'''
			Please override "train_on_batch_supervised" method in the derived model!
		'''

	def train_on_batch_unsupervised(self, x_batch):
		raise NotImplementedError
		'''
			Please override "train_on_batch_unsupervised" method in the derived model!
		'''

	'''
		test methods
	'''
	def predict(self, sess, x_batch):
		raise NotImplementedError

	def generate(self, sess, z_batch):
		raise NotImplementedError

	def hidden_distribution(self, sess, x_batch):
		raise NotImplementedError




	def train(self, sess, feed_dict,
				update_op=None, 
				step=None,
				learning_rate=None,
				loss=None,
				summary=None,
				):

		if step is None:
			step = self.global_step
		if learning_rate is None:
			learning_rate = self.learning_rate
		if loss is None:
			loss = self.loss

		if update_op is None:
			s, lr = sess.run([step, learning_rate])
			return s, lr, 0, None
		elif self.is_summary and summary is not None:
			_, s, lr, l, s_sum = sess.run([update_op, step, learning_rate, loss, summary],	
						feed_dict = feed_dict)
			return s, lr, l, s_sum
		else:
			_, s, lr, l = sess.run([update_op, step, learning_rate, loss],	feed_dict = feed_dict)
			return s, lr, l, None

