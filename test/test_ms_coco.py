import os
import sys

sys.path.append('.')
sys.path.append('../')

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.coco import MSCOCO

if __name__ == '__main__':
	config = {
		"output shape" : [224, 224, 3],
		"show warning" : True,
	}

	dataset = MSCOCO(config)
	indices = dataset.get_image_indices('trainval')

	for ind in indices:
		img, anno = dataset.read_image_by_index(ind, phase='trainval', method='supervised')
		
		if img is not None:
			plt.figure(0)
			plt.clf()
			dataset.show_image_and_anno(plt, img, anno)
			plt.pause(3)

