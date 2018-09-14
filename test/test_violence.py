import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.violence import Violence


if __name__ == '__main__':
	config = {
		"output shape" : [224, 224, 3]
	}

	dataset = Violence(config)
	indices = dataset.get_image_indices('train')

	for ind in indices:
		img, label = dataset.read_image_by_index(ind, phase='train', method='supervised')
		
		if img is not None:
			plt.figure(0)
			plt.clf()
			plt.imshow(img)
			plt.pause(3)

