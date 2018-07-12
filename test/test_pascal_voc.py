
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.pascal_voc import PASCAL_VOC

if __name__ == '__main__':

	config = {
        "output shape" : [160, 160, 3],
        "scaling range" : [0.45, 1.0]
    }

	dataset = PASCAL_VOC(config)

	indices = dataset.get_image_indices(phase='train')

	print(indices.shape)

	for ind in indices:

		img, mask_onehot = dataset.read_image_by_index_supervised(ind)

		print(img.shape)
		print(mask_onehot.shape)

		print(mask_onehot.max())
		print(mask_onehot.min())

		plt.figure(0)
		plt.imshow(img)
		plt.pause(0.01)

		plt.figure(1)
		plt.imshow(mask_onehot[:, :, 0])
		plt.pause(4)

