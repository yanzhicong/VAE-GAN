
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np





def get_sample(name, args):
    if name == 'normal' : 
        z_avg, z_log_var = args
        # batch_size = z_avg.get_shape()[0]
        # z_dioms = z_avg
        eps = tf.random_normal(shape=tf.shape(z_avg), mean=0.0, stddev=1.0)
        return z_avg + tf.exp(z_log_var / 2.0) * eps
    else:
        raise Exception("None sample function named " + name)


