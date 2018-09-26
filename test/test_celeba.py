import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from dataset.celeba import CelebA

if __name__ == '__main__':

	config = {
		"output shape" : [256, 256, 3],
        "output scalar range" : [0, 1]
	}


	dataset = CelebA(config)
	indices = dataset.get_image_indices(phase='train', method='supervised')

	print(indices.shape)

	for ind in indices:

		img, attr = dataset.read_image_by_index(ind, phase='train', method='supervised')

		print(img.shape)
		print(attr.shape)
		print(attr.max(), attr.min())

		plt.figure(0)
		plt.imshow(img)
		plt.pause(4)


