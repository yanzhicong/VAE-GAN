
import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.pascal_voc import PASCAL_VOC

if __name__ == '__main__':

	config = {
        "output shape" : [64, 64, 3],
        "scaling range" : [0.15, 0.25],
        "crop range" : [0.3, 0.7],
		"task" :  "classification",
		"random mirroring" : False
    }

	dataset = PASCAL_VOC(config)

	indices = dataset.get_image_indices(phase='train')

	print(indices.shape)

	for ind in indices:

		img, label = dataset.read_image_by_index_supervised(ind)

		print(img.shape)
		print(label)

		plt.figure(0)
		plt.imshow(img)
		plt.pause(4)


