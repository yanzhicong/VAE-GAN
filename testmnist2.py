
import os
import sys

sys.path.append('./')
sys.path.append('./lib')



import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


from cfgs.networkconfig import get_config
from dataset.dataset import get_dataset
from model.model import get_model



from sklearn import cluster
from collections import Counter

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '1'

	# load config file
	config = get_config("cla/mnist2")

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

		indices = dataset.get_image_indices(phase='train', method='supervised')
		data = [dataset.read_image_by_index(ind, phase='train', method='supervised') for ind in indices]
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


		crossentropy_matrix = np.zeros([10, 10], dtype=np.float32)

		for i in range(10):
			for j in range(10):
				if i > j:
					pred_class_i = data_p[:, i]
					pred_class_j = data_p[:, j]
					entropy = - np.mean(pred_class_i * np.log(pred_class_i+0.000000001) + pred_class_j * np.log(pred_class_j+0.000000001))
					crossentropy_matrix[i, j] = entropy
					crossentropy_matrix[j, i] = entropy
				elif i == j:
					pred_class_i = data_p[:, i]
					entropy = -np.mean( pred_class_i * np.log(pred_class_i+0.000000001) + (1-pred_class_i) * np.log(1-pred_class_i + 0.000000001) )
					crossentropy_matrix[i, i] = entropy
		print(crossentropy_matrix)

# cla/mnist4
# 0.0059269  0.00559208 0.00742071 0.00764763 0.00766863 0.0072737 0.0059385  0.00679747 0.00817884 0.00734019
# 0.00559208 0.00479852 0.00685371 0.00708064 0.00710164 0.00670671 0.00537151 0.00623048 0.00761185 0.0067732 
# 0.00742071 0.00685371 0.00844414 0.00890927 0.00893026 0.00853533 0.00720014 0.00805911 0.00944048 0.00860182
# 0.00764763 0.00708064 0.00890927 0.0089634  0.00915719 0.00876226 0.00742707 0.00828604 0.0096674  0.00882875
# 0.00766863 0.00710164 0.00893026 0.00915719 0.0089141  0.00878326 0.00744807 0.00830703 0.0096884  0.00884975
# 0.0072737  0.00670671 0.00853533 0.00876226 0.00878326 0.00806099 0.00705313 0.0079121  0.00929347 0.00845482
# 0.0059385  0.00537151 0.00720014 0.00742707 0.00744807 0.00705313 0.00552391 0.00657691 0.00795828 0.00711962
# 0.00679747 0.00623048 0.00805911 0.00828604 0.00830703 0.0079121 0.00657691 0.00716828 0.00881724 0.00797859
# 0.00817884 0.00761185 0.00944048 0.0096674  0.0096884  0.00929347 0.00795828 0.00881724 0.00985754 0.00935996
# 0.00734019 0.0067732  0.00860182 0.00882875 0.00884975 0.00845482 0.00711962 0.00797859 0.00935996 0.00828347


# cla/mnist2
# 9.2937538e-05,1.7710096e-04,1.4954635e-04,1.4131931e-04,2.9116604e-04,2.3577447e-04,1.9387594e-04,2.0710977e-04,1.7558823e-04,2.3517699e-04
# 1.7710096e-04,2.4385739e-04,2.2020475e-04,2.1197771e-04,3.6182441e-04,3.0643286e-04,2.6453432e-04,2.7776818e-04,2.4624661e-04,3.0583533e-04
# 1.4954635e-04,2.2020475e-04,1.8098285e-04,1.8442309e-04,3.3426983e-04,2.7887829e-04,2.3697970e-04,2.5021355e-04,2.1869202e-04,2.7828076e-04
# 1.4131931e-04,2.1197771e-04,1.8442309e-04,2.1235517e-04,3.2604279e-04,2.7065125e-04,2.2875270e-04,2.4198649e-04,2.1046498e-04,2.7005372e-04
# 2.9116604e-04,3.6182441e-04,3.3426983e-04,3.2604279e-04,4.2415247e-04,4.2049790e-04,3.7859939e-04,3.9183328e-04,3.6031168e-04,4.1990040e-04
# 2.3577447e-04,3.0643286e-04,2.7887829e-04,2.7065125e-04,4.2049790e-04,2.8689124e-04,3.2320785e-04,3.3644165e-04,3.0492016e-04,3.6450886e-04
# 1.9387594e-04,2.6453432e-04,2.3697970e-04,2.2875270e-04,3.7859939e-04,3.2320785e-04,2.8788712e-04,2.9454316e-04,2.6302159e-04,3.2261031e-04
# 2.0710977e-04,2.7776818e-04,2.5021355e-04,2.4198649e-04,3.9183328e-04,3.3644165e-04,2.9454316e-04,3.0473541e-04,2.7625542e-04,3.3584418e-04
# 1.7558823e-04,2.4624661e-04,2.1869202e-04,2.1046498e-04,3.6031168e-04,3.0492016e-04,2.6302159e-04,2.7625542e-04,2.4779662e-04,3.0432263e-04
# 2.3517699e-04,3.0583533e-04,2.7828076e-04,2.7005372e-04,4.1990040e-04,3.6450886e-04,3.2261031e-04,3.3584418e-04,3.0432263e-04,4.2435722e-04


		# def cluster_predict(nb_cluster):
		# 	centroid, labels, inertia = cluster.k_means(data_f, n_clusters=nb_cluster)
		# 	cluster_labels = np.zeros(nb_cluster, dtype='int32')
		# 	for i in range(nb_cluster):
		# 		cluster_element_indices = np.where(labels == i)[0]
		# 		cluster_element_pred_result = data_p[cluster_element_indices]
		# 		cluster_labels[i] = Counter(np.argmax(cluster_element_pred_result, axis=1)).most_common(1)[0][0]
		# 	data_cl = np.array([ cluster_labels[labels[i]] for i in range(data_y.shape[0])])
		# 	accuracy = (np.argmax(data_y, axis=1) == data_cl).sum() / float(data_y.shape[0])
		# 	print('nb_cluster %d accuray : %f'%(nb_cluster, accuracy))


		# assets_dir = config['assets dir']
		# log_dir = config.get('log dir', 'embedding_feature_10per_classes')
		# log_dir = os.path.join(assets_dir, log_dir)

		# if not os.path.exists(log_dir):
		# 	os.mkdir(log_dir)
		# with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
		# 	f.write("Index\tLabel\tPredict\tCorrect\n")
		# 	for i in range(data_x.shape[0]):
		# 		l = np.argmax(data_y[i])
		# 		p = np.argmax(data_p[i])
		# 		f.write("%d\t%d\t%d\n"%(i, l, p, 1 if l==p else 0))

		# summary_writer = tf.summary.FileWriter(log_dir)
		# config = projector.ProjectorConfig()
		# embedding = config.embeddings.add()
		# embedding.tensor_name = "test"
		# embedding.metadata_path = "metadata.tsv"
		# projector.visualize_embeddings(summary_writer, config)

		# plot_array_var = tf.get_variable('test', shape=data_f.shape)
		# saver = tf.train.Saver([plot_array_var])
		# sess.run(plot_array_var.assign(data_f))
		# saver.save(sess, os.path.join(log_dir, 'model.ckpt'), 
		# 					global_step=0, 
		# 					write_meta_graph=False,
		# 					strip_default_attrs=True)

		# cluster_predict(nb_cluster=10)
		# cluster_predict(nb_cluster=15)
		# cluster_predict(nb_cluster=17)
		# cluster_predict(nb_cluster=19)
		# cluster_predict(nb_cluster=20)
		# cluster_predict(nb_cluster=23)
		# cluster_predict(nb_cluster=25)
		# cluster_predict(nb_cluster=27)
		# cluster_predict(nb_cluster=30)
		# cluster_predict(nb_cluster=40)
		# cluster_predict(nb_cluster=50)
		# cluster_predict(nb_cluster=75)
		# cluster_predict(nb_cluster=100)
		# cluster_predict(nb_cluster=150)
		# cluster_predict(nb_cluster=200)
		# cluster_predict(nb_cluster=300)
		# cluster_predict(nb_cluster=400)
		# cluster_predict(nb_cluster=800)
