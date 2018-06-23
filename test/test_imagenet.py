
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.imagenet import ImageNet 

if __name__ == '__main__':
	config = {
		'batch_size' : 16,
		'input_shape' : [224, 224, 3]
	}

	dataset = ImageNet(config)

	for ind, x_batch, y_batch in dataset.iter_train_images():

		print(ind, x_batch.shape, y_batch.shape)
		plt.figure(0)

		plt.imshow(x_batch[0, :, :, :])
		plt.pause(1)


