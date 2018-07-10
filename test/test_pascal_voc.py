
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.pascal_voc import PASCAL_VOC

if __name__ == '__main__':

	config = {
		'output shape' : [224, 224, 3]
	}
	dataset = PASCAL_VOC(config)

	indices = dataset.get_image_indices(phase='train')

	print(indices.shape)

	for ind in indices:

		img, mask = dataset.read_image_by_index_supervised(ind)

		print(img.shape)
		print(mask.shape)

		plt.figure(0)
		plt.imshow(img)
		plt.pause(0.01)
		plt.figure(1)
		plt.imshow(mask)
		plt.pause(4)

