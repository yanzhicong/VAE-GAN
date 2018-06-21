
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from dataset.mnist import MNIST
from dataset.svhn import SVHN

if __name__ == '__main__':

	config = {
		'output shape' : [28, 28]
	}
	dataset = MNIST(config)

	# config = {
	# 	'output shape' : [32, 32, 3]
	# }
	# dataset = SVHN(config)
	
	indices = dataset.get_image_indices('train', 'supervised')
	for i, ind in enumerate(indices):
		img, label = dataset.read_image_by_index(ind, 'train', 'supervised')

		print(label, np.argmax(label))

		plt.figure(0)
		plt.imshow(img)
		plt.pause(1)

