



import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class HiddenVariableValidator(object):
	def __init__(self, config):

		self.z_dim = config['z_dim']
		self.assets_dir = config['assets dir']
		self.log_dir = config.get('log dir', 'hidden')

		self.log_dir = os.path.join(self.assets_dir, self.log_dir)
		if not os.path.exists(self.log_dir):
			os.mkdir(self.log_dir)

		self.generate_method = config.get('generate_method', 'normppf')

		if self.generate_method == 'normppf':
			self.nb_samples = config.get('num_samples', 30)


	def validate(self, model, dataset, sess, step):
		if self.generate_method == 'normppf':
			n = self.nb_samples  # figure with 15x15 digits
			digit_size = 28
			figure = np.zeros((digit_size * n, digit_size * n))

			#用正态分布的分位数来构建隐变量对
			grid_x = norm.ppf(np.linspace(0.01, 0.99, n))
			grid_y = norm.ppf(np.linspace(0.01, 0.99, n))

			for i, yi in enumerate(grid_x):
				for j, xi in enumerate(grid_y):
					z_sample = np.array([[xi, yi]])
					x_decoded = model.predict(sess, z_sample)
					digit = x_decoded[0].reshape(digit_size, digit_size)
					figure[i * digit_size: (i + 1) * digit_size,
						j * digit_size: (j + 1) * digit_size] = digit

			plt.figure(figsize=(10, 10))
			plt.imshow(figure, cmap='Greys_r')
			plt.savefig(os.path.join(self.log_dir, '%07d.png'%step))

