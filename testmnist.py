
import os
import sys

sys.path.append('./')
sys.path.append('./lib')



import numpy as np
import tensorflow as tf

from cfgs.networkconfig import get_config
from dataset.dataset import get_dataset
from model.model import get_model
# from trainer.trainer import get_trainer



from sklearn import cluster
from collections import Counter

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '1'

	# load config file
	config = get_config("cla/mnist4")


	# prepare dataset
	config['dataset params']['semi-supervised'] = False
	dataset = get_dataset(config['dataset'], config['dataset params'])


	tfconfig = tf.ConfigProto()
	tfconfig.gpu_options.allow_growth = True
	with tf.Session(config=tfconfig) as sess:

		# build model
		config['model params']['assets dir'] = config['assets dir']
		model = get_model(config['model'], config['model params'])

		print("load checkpoint:")
		if model.checkpoint_load(sess, os.path.join(config['assets dir'], config['trainer params'].get('checkpoint dir', 'checkpoint'))):
			print('success')
		else:
			print('load checkpoint failed!')

		indices = dataset.get_image_indices(phase='train')
		data = [dataset.read_image_by_index_supervised(ind, phase='train') for ind in indices]
		data_x = [d[0] for d in data]
		data_y = [d[1] for d in data]
		data_p = []
		data_f = []

		batch_size = 100

		for i in range(len(data_x) // batch_size):
			batch_x = np.array(data_x[(i)*batch_size:(i+1)*batch_size])

			pred_p = model.predict(sess, batch_x)
			pred_f = model.reduce_features(sess, batch_x, 'fc_out')

			for i in range(batch_size):
				data_f.append(pred_f[i])
				data_p.append(pred_p[i])

		data_x = np.array(data_x)
		data_y = np.array(data_y)
		data_p = np.array(data_p)	# predict probs
		data_f = np.array(data_f)	# features

		accuracy = (np.argmax(data_y, axis=1) == np.argmax(data_p, axis=1)).sum() / float(data_y.shape[0])

		print('accuray : %f'%accuracy)

		def cluster_predict(nb_cluster):
			centroid, labels, inertia = cluster.k_means(data_f, n_clusters=nb_cluster)
			cluster_labels = np.zeros(nb_cluster, dtype='int32')
			for i in range(nb_cluster):
				cluster_element_indices = np.where(labels == i)[0]
				cluster_element_pred_result = data_p[cluster_element_indices]
				cluster_labels[i] = Counter(np.argmax(cluster_element_pred_result, axis=1)).most_common(1)[0][0]
			data_cl = np.array([ cluster_labels[labels[i]] for i in range(data_y.shape[0])])
			accuracy = (np.argmax(data_y, axis=1) == data_cl).sum() / float(data_y.shape[0])
			print('nb_cluster %d accuray : %f'%(nb_cluster, accuracy))
		
		cluster_predict(nb_cluster=10)
		cluster_predict(nb_cluster=15)
		cluster_predict(nb_cluster=17)
		cluster_predict(nb_cluster=19)
		cluster_predict(nb_cluster=20)
		cluster_predict(nb_cluster=23)
		cluster_predict(nb_cluster=25)
		cluster_predict(nb_cluster=27)
		cluster_predict(nb_cluster=30)	
		cluster_predict(nb_cluster=40)	
		cluster_predict(nb_cluster=50)
		cluster_predict(nb_cluster=75)
		cluster_predict(nb_cluster=100)
		cluster_predict(nb_cluster=150)
		cluster_predict(nb_cluster=200)
		cluster_predict(nb_cluster=300)
		cluster_predict(nb_cluster=400)
		cluster_predict(nb_cluster=800)
