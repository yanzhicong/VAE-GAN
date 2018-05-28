import os
import sys
import time
import numpy as np

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# from keras.models import load_model
import tensorflow as tf
from abc import ABCMeta, abstractmethod

# from .utils import set_trainable, zero_loss, time_format

class BaseModel(object, metaclass=ABCMeta):

	# __metaclass__ = ABCMeta
	'''
	Base class for non-conditional generative networks
	'''

	def __init__(self, config, **kwargs):
		'''
		Initialization
		'''

		self.name = config['name']


	# def save_images(self, samples, filename):
	# 	'''
	# 	Save images generated from random sample numbers
	# 	'''
	# 	assert self.attr_names is not None
		
	# 	num_samples = len(samples)
	# 	attrs = np.identity(self.num_attrs)
	# 	attrs = np.tile(attrs, (num_samples, 1)) #TODO: Is there a better method on keras?

	# 	samples = np.tile(samples, (1, self.num_attrs))
	# 	samples = samples.reshape((num_samples * self.num_attrs, -1))

	# 	imgs = self.predict([samples, attrs]) * 0.5 + 0.5
	# 	imgs = np.clip(imgs, 0.0, 1.0)
		
	# 	if imgs.shape[3] == 1:
	# 		imgs = np.squeeze(imgs, axis=(3,))

	# 	fig = plt.figure(figsize=(self.num_attrs, 10))
	# 	grid = gridspec.GridSpec(num_samples, self.num_attrs, wspace=0.1, hspace=0.1)
	# 	for i in range(num_samples * self.num_attrs):
	# 		ax = plt.Subplot(fig, grid[i])
	# 		if imgs.ndim == 4:
	# 			ax.imshow(imgs[i, :, :, :], interpolation="none", vmin=0.0, vmax=1.0)
	# 		else:
	# 			ax.imshow(imgs[i, :, :], camp="gray", interpolation="none", vmin=0.0, vmax=1.0)
	# 		ax.axis("off")
	# 		fig.add_subplot(ax)

	# 	fig.savefig(filename, dpi=200)
	# 	plt.close(fig)


	def checkpoint_load(self, sess, log_dir):
		print(" [*] Reading checkpoint...")
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

	@abstractmethod
	def predict(self, z_sample):
		'''
		Plase override "predict" method in the derived model!
		'''
		pass


	@abstractmethod
	def train_on_batch_supervised(self, x_batch, y_batch):
		'''
		Plase override "train_on_batch" method in the derived model!
		'''
		pass


	@abstractmethod
	def train_on_batch_unsupervised(self, x_batch):
		'''
		Plase override "train_on_batch" method in the derived model!
		'''
		pass


