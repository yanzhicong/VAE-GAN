import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.cifar10 import Cifar10 

if __name__ == '__main__':
	config = {
		'batch_size' : 16,
		'output shape' : [32, 32, 3]
	}

	dataset = Cifar10(config)

	for ind, x_batch, y_batch in dataset.iter_train_images_supervised():
		plt.figure(0)
		plt.imshow(x_batch[0, :, :, :])
		plt.pause(1)
